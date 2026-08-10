"""WISEPACK dashboard backend.

Structure follows TEMPO's web/app.py — FastAPI, an in-memory cache, REST plus an
optional WebSocket, and KPIs read from the newest results/ artefact rather than
invented live. What is different is the source model and the fact that this
dashboard *commands* as well as observes: WISEPACK's whole point is a human in
the loop, so approval and rejection have to travel from here to the workflow.

THREE SOURCES, selected with --source. The badge in the header always names the
one in use, and no figure is ever shown without it.

  sim     Self-contained. Drives wisepack_core.WorkflowEngine in a background
          task. NO ROS, NO FIWARE, NO Docker. Every number is produced by the
          same domain logic the live stack runs — this is not a separate
          animation — but execution outcomes are simulated and labelled so.

  ros     Read-mostly observer of the canonical WISEPACK topics, plus a command
          publisher for the two operator topics. Packing figures here are the
          same measured values; robot outcomes are still simulated, because the
          robot is still a simulator.

  fiware  Authoritative state read back from Orion-LD over NGSI-LD. This is the
          mode that proves the audit path end to end: if a value renders here it
          survived ROS -> DDS -> Orion-LD. High-frequency animation data still
          come from ROS, and the badge says "fiware+ros" when that is the case.

Run:  python3 app.py --source sim --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# wisepack_core is imported from the workspace source tree, so the dashboard
# runs with nothing installed and nothing built. This is the same import the
# ROS nodes do after a colcon build; only the path discovery differs.
for _pkg in ("wisepack_core", "wisepack_fiware", "wisepack_bringup"):
    _path = os.path.join(REPO, "wisepack_ws", "src", _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from wisepack_core.artifacts import (                              # noqa: E402
    latest_artifact, latest_latency_p50_ms, resolve_results_dir,
    write_run_artifacts,
    write_validation_report,
)

#: RESOLVED, not assumed. This was hard-coded to `<repo>/results`, which
#: bypassed the WISEPACK_RESULTS_DIR override the rest of the project honours —
#: and on a shared checkout that directory can belong to another user, so every
#: run finished by reporting a permission error it could do nothing about.
#: Assigned AFTER the import it depends on; it sat above it and broke start-up.
RESULTS_DIR = resolve_results_dir(repo_root=REPO)
from wisepack_core.domain import Strategy                          # noqa: E402
from wisepack_core.execution import physical_presets              # noqa: E402
from wisepack_core.acquisition import (                            # noqa: E402
    ACQUISITION_ISAAC, ACQUISITION_PLANAR, ACQUISITION_REALSENSE,
    AcquisitionState)
from wisepack_core.perception import (                             # noqa: E402
    ObservationBatch, ObjectSourceState, PerceptionMethod, PerceptionMethodState,
    PerceptionSource, ProxyGeometry, WorkAreaFrame,
    DEFAULT_PERCEPTION_METHOD, resolve_object_source,
    resolve_perception_method, resolve_perception_method_selection,
    resolve_perception_source,
)
from wisepack_core.events import (                                 # noqa: E402
    DynamicEvent, DynamicEventType, Stage,
)
from wisepack_core.generator import CONTAINER_SPECS, PRESETS       # noqa: E402
from wisepack_bringup.topics import OPERATOR_COMMANDS              # noqa: E402
from wisepack_core.kpi import compare_strategies                   # noqa: E402
from wisepack_core.packing import OptimizerConfig                  # noqa: E402
from wisepack_core.workflow import (                               # noqa: E402
    AnomalyHold, ApprovalRequired, PerceptionUnavailable, RobotSimConfig,
    WorkflowConfig, WorkflowEngine, WorkflowError,
)
from wisepack_core.whole_process import WholeProcessError          # noqa: E402
from wisepack_core.inventory import InvalidTransition             # noqa: E402
from snapshot import (                                             # noqa: E402
    FiwareSnapshotProvider, RosSnapshotProvider, SimSnapshotProvider, parse_attr,
)

SOURCE = os.environ.get("WISEPACK_SOURCE", "sim")
ORION = os.environ.get("ORION", "http://localhost:1026").rstrip("/")

# THE PERCEPTION SOURCE. A THIRD, INDEPENDENT AXIS — not a data source (SOURCE
# above) and not an execution backend. `sim` | `camera`; unset == `sim`, so
# nothing existing changes. WHICH detector processes a camera frame is a
# separate setting again (WISEPACK_PERCEPTION_DETECTOR) and is the perception
# service's business, not the dashboard's.
#
# AN UNRECOGNISED VALUE RAISES HERE AND THE PROCESS DOES NOT START. This is
# deliberately NOT caught. It was, once: the exception was swallowed, the source
# was downgraded to `sim`, and the reason was put in a `config_error` field that
# the only renderer never reached — so a bad `WISEPACK_PERCEPTION_SOURCE`
# (an old or misspelled value) started a dashboard that quietly produced
# SIMULATED detections for an operator who had asked for a camera. A silent
# downgrade of a perception source is precisely the failure this repository
# refuses to allow.
#
# `wisepack_orchestration.hitl_orchestrator` resolves the same value the same
# way and lets it raise before any publisher exists. The two must agree: a
# configuration that kills the ROS stack must not quietly run in the dashboard.
PERCEPTION_SOURCE = resolve_perception_source()

#: How long a camera-capability answer is reused before the perception service
#: is asked again. See `camera_capability()`.
CAMERA_CAPABILITY_TTL_S = 2.0

#: Seconds between execution steps in sim mode. Slow enough to watch, fast
#: enough that a 40-item scenario finishes inside a demo slot.
STEP_PERIOD_S = float(os.environ.get("WISEPACK_STEP_PERIOD_S", "0.7"))


# --------------------------------------------------------------------------- #
# Shared state
# --------------------------------------------------------------------------- #


class DemoState:
    """The dashboard's view of one run, guarded by one lock.

    A single lock rather than per-field locking: the workflow engine mutates
    several related fields per step and a reader that saw half an update would
    render a container holding an item the plan no longer contains.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.engine: Optional[WorkflowEngine] = None
        self.source = SOURCE
        self.events: List[Dict[str, Any]] = []
        self.running = False
        self.auto_step = False
        #: Populated by ros_observer in live mode: the latest value seen on each
        #: canonical topic. None in sim mode, where there is no ROS.
        self.ros_mirror: Optional[Dict[str, Any]] = None
        #: The perception-service client, built on first use. Built whatever
        #: the selected source is: the camera is a capability, and the dashboard
        #: asks about it while running preset scenarios.
        self.perception_client = None
        #: Metadata about the simulated-RGB-D acquisition currently on screen,
        #: or None. Set only by `acquire_simulated_rgbd`.
        self.simulated_rgbd: Optional[Dict[str, Any]] = None
        #: The outcome of the last artefact write: where it went, or why it
        #: could not. Carried on the run rather than left in a transient notice.
        self.artifacts: Optional[Dict[str, Any]] = None
        #: (usable, reason) and when it was asked. See `camera_capability`.
        self.camera_capability = (False, "")
        self.camera_capability_at = None
        self.fiware_connected: Optional[bool] = None
        self.fiware_last_error = ""
        self.notice = ""
        #: True once the operator has edited the draft. Until then the draft
        #: tracks the active run; afterwards the operator owns it and no poll
        #: may overwrite their selection.
        self.settings_touched = False
        self.settings: Dict[str, Any] = {
            "preset": "mixed_pipes_dense",
            "seed": 42,
            "strategy": Strategy.MAX_DENSITY.value,
            "item_count": None,
            "length_range_mm": None,
            "diameter_range_mm": None,
            "container_spec": None,
            "dynamic_events_enabled": True,
            "pick_failure_probability": 0.08,
            # THE DRAFT ROBOT for the NEXT run. None means "not chosen yet", so
            # it is seeded from the active run or the configured default on the
            # first render and owned by the operator afterwards. Deliberately
            # NOT the active robot: changing this must never touch a running
            # scene, and only "Reset run & generate" carries it into a run.
            "robot_id": None,
            # THE DRAFT OBJECT SOURCE for the NEXT run: `sim` (generate from the
            # selected preset) or `camera` (detect from a real frame). Seeded
            # from WISEPACK_PERCEPTION_SOURCE, which is the INITIAL selection
            # and not a mode lock — the operator switches per run, in one
            # session, without restarting anything. Like the robot draft, it is
            # inert until an acquisition is actually started.
            "object_source": PERCEPTION_SOURCE.value,
            # THE METHOD IS A DRAFT LIKE THE SOURCE. Planar is the default and
            # stays the default: it is the validated path, it needs no depth
            # camera, no GPU and no CAD model, and every physical run WISEPACK
            # has done used it.
            "perception_method": resolve_perception_method(),
        }

    # -- event capture ---------------------------------------------------- #

    def sink(self, event) -> None:
        with self.lock:
            self.events.append(event.to_dict())
            if len(self.events) > 4000:
                del self.events[:len(self.events) - 4000]

    def recent(self, limit: int = 120) -> List[Dict[str, Any]]:
        with self.lock:
            return list(reversed(self.events[-limit:]))


STATE = DemoState()


# --------------------------------------------------------------------------- #
# Scenario construction
# --------------------------------------------------------------------------- #

#: The dynamic-event script used by the interactive demo. Deterministic and
#: declared here rather than randomly generated, so a presenter knows exactly
#: what will happen and when.
DEMO_EVENTS = [
    DynamicEvent(
        event_type=DynamicEventType.ITEM_INJECT,
        trigger="placement:4",
        label="High-priority ILW component arrives late",
        payload={"item": {"length_mm": 1200, "outer_diameter_mm": 220,
                          "inner_diameter_mm": 186, "material": "stainless_316L",
                          "priority": 9, "dose_class": "ILW"}}),
]


def _preset_compatibility(robot_id: Optional[str]) -> Dict[str, str]:
    """{preset: reason} for the physical backend, for THIS robot.

    Two different bounds, and the operator has to be able to tell them apart:
    the backend-level ones (too many items, items wider than any gripper) apply
    to every arm, and the robot's own ``supported_presets`` applies to one. A
    reason that says only "unavailable" sends someone looking at the asset when
    the answer is the selector two fields up.
    """
    profile = None
    if robot_id:
        try:
            from wisepack_core.robots import load_registry         # noqa: PLC0415
            profile = load_registry().profiles.get(str(robot_id).lower())
        except Exception:                                          # noqa: BLE001
            profile = None
    return physical_presets(profile)


def _generator_overrides(settings: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if settings.get("item_count"):
        overrides["item_count"] = int(settings["item_count"])
    if settings.get("length_range_mm"):
        overrides["length_range_mm"] = tuple(settings["length_range_mm"])
    if settings.get("diameter_range_mm"):
        overrides["diameter_range_mm"] = tuple(settings["diameter_range_mm"])
    if settings.get("container_spec"):
        overrides["container_spec"] = settings["container_spec"]
    return overrides


def build_engine(settings: Dict[str, Any],
                 allow_sources: Optional[List[str]] = None) -> WorkflowEngine:
    preset = settings.get("preset", "mixed_pipes_dense")
    seed = int(settings.get("seed", 42))
    # WHERE THIS RUN'S OBJECTS COME FROM. Validated against what is actually
    # available, so asking for a camera that is not there fails here with the
    # reason rather than quietly producing a generated scenario.
    # `allow_sources` lets a caller that KNOWS its backend say so — the
    # simulated RGB-D acquisition does not use the planar service and must not
    # be vetoed by its absence. Everything else keeps the live availability
    # check, so a source is never offered on the strength of a stale artefact.
    object_source = resolve_object_source(
        settings.get("object_source"),
        allow_sources or available_object_sources(),
        fallback=PERCEPTION_SOURCE.value)
    # The in-process demo engine is the LOGICAL backend and has no robot. The
    # draft robot is carried on the settings for the live modes, where the
    # orchestrator owns the run; recording it here would attach a robot name to
    # a run in which nothing robotic happens.
    # The curated dataset is hand-built; generator overrides do not apply to it
    # and silently accepting them would misrepresent what was run.
    overrides = ({} if preset == "curated_volume_reduction"
                 else _generator_overrides(settings))
    config = WorkflowConfig(
        preset=preset,
        seed=seed,
        # PERCEPTION SOURCE AND EXECUTION BACKEND ARE SET INDEPENDENTLY. This
        # engine is the LOGICAL execution backend either way; selecting a camera
        # changes where the OBJECTS come from and nothing else.
        #
        # PER RUN, from the operator's draft — not from a process-wide setting.
        # `PERCEPTION_SOURCE` is only where the draft STARTED.
        perception_source=object_source,
        proxy_geometry=ProxyGeometry.from_env(),
        work_area=WorkAreaFrame.from_env(),
        strategy=Strategy(settings.get("strategy", "max_density")),
        optimizer=OptimizerConfig(seed=seed, restarts=6, time_budget_ms=4000.0),
        robot=RobotSimConfig(
            pick_failure_probability=float(
                settings.get("pick_failure_probability", 0.08)),
            seed=seed),
        dynamic_events=([DynamicEvent.from_dict(e.to_dict()) for e in DEMO_EVENTS]
                        if settings.get("dynamic_events_enabled", True) else []),
        generator_overrides=overrides,
        auto_approve=False)
    engine = WorkflowEngine(config)
    if object_source.is_physical:
        # THE ONLY PLACE THE DASHBOARD LEARNS HOW OBSERVATIONS ARRIVE. The
        # engine stays transport-free; it calls a callable and gets a batch.
        from wisepack_core.perception_client import (              # noqa: PLC0415
            make_observation_provider)
        engine.observation_provider = make_observation_provider(
            perception_client())
    engine.log.add_sink(STATE.sink)
    return engine


def perception_client():
    """The one shared client for the perception service. Built lazily.

    Built REGARDLESS of which source is selected: the camera is a capability of
    the deployment, and the dashboard has to be able to ask whether one is there
    while running a preset scenario. Construction opens no socket.
    """
    from wisepack_core.perception_client import PerceptionClient  # noqa: PLC0415
    with STATE.lock:
        if STATE.perception_client is None:
            STATE.perception_client = PerceptionClient()
        return STATE.perception_client


def camera_capability(health: Optional[Dict[str, Any]] = None):
    """(camera usable, reason), asked live and cached for a second or two.

    A LIVE QUESTION, NOT A START-UP ONE. An operator can start the perception
    service at any moment — from the launcher, or by hand in another terminal —
    and the object-source selector must offer the camera without the dashboard
    being restarted. Equally, a service that dies must stop being offered. The
    short cache exists only so a 1 Hz page poll does not become an HTTP load
    generator against the detector.
    """
    now = time.monotonic()
    with STATE.lock:
        cached = STATE.camera_capability
        stamped = STATE.camera_capability_at
    if health is None and stamped is not None and now - stamped < CAMERA_CAPABILITY_TTL_S:
        return cached
    answer = perception_client().capability(health)
    with STATE.lock:
        STATE.camera_capability = answer
        STATE.camera_capability_at = now
    return answer


def simulated_frames_available() -> bool:
    """Can the simulated RGB-D backend acquire right now?

    A LIVE CHECK, not the mere existence of a file: the FoundationPose worker
    must be usable AND a simulated frame must be present. A leftover artefact
    with no worker is not an acquisition capability.
    """
    # AN EXPLICIT OFF SWITCH, for tests that need a deployment with NO camera
    # of any kind. Without it, "is a camera available" depended on whether a
    # previous acquisition had left a file on the machine — which made the
    # absent-camera test pass or fail on unrelated state.
    if os.environ.get("WISEPACK_DISABLE_SIMULATED_RGBD", "").strip():
        return False
    if _simulated_rgbd_observation() is None:
        return False
    provider = foundationpose_provider()
    if provider is None:
        return False
    return bool(provider.client.capability()[0])


def acquisition_state() -> "AcquisitionState":
    """Which cameras can acquire, which one produced the batch on screen.

    EACH SOURCE ANSWERS FOR ITSELF. A missing webcam must not veto a simulated
    run, and a missing RealSense must not veto either of the others — which is
    exactly what a single "camera available" flag did.
    """
    available: List[str] = []
    reasons: Dict[str, str] = {}

    planar_ok, planar_reason = camera_capability()
    if planar_ok:
        available.append(ACQUISITION_PLANAR)
    else:
        reasons[ACQUISITION_PLANAR] = planar_reason or "no perception service"

    fp = foundationpose_capability()
    if fp.get("rgbd_camera_available"):
        available.append(ACQUISITION_REALSENSE)
    else:
        reasons[ACQUISITION_REALSENSE] = "; ".join(
            str(b) for b in (fp.get("blocked_by") or [])) or "no D435 attached"

    if simulated_frames_available():
        available.append(ACQUISITION_ISAAC)
    else:
        reasons[ACQUISITION_ISAAC] = (
            "no simulated RGB-D frame is available; produce one with "
            "./scripts/stage_c.sh")

    with STATE.lock:
        engine = STATE.engine
    current = ""
    batch = getattr(engine, "observation_batch", None) if engine else None
    if batch is not None and getattr(batch, "acquisition", ""):
        current = (ACQUISITION_ISAAC
                   if batch.acquisition == "simulated_rgbd"
                   else ACQUISITION_PLANAR)
    return AcquisitionState(current=current, selected=current or "",
                            available=available, unavailable_reasons=reasons)


def available_object_sources(health: Optional[Dict[str, Any]] = None):
    """Which object sources this deployment can use right now.

    AVAILABILITY MEANS A SERVICE ANSWERED JUST NOW. A simulated RGB-D
    observation is a FILE, and a file left over from last week is no evidence
    that a camera is usable — so it does not make `camera` available here. The
    simulated acquisition instead names its own source when it runs, which is an
    explicit act rather than an inference from a stale artefact.
    """
    sources = [PerceptionSource.SIM.value]
    # ANY camera backend makes the CAMERA source usable. Keying this on the
    # planar service alone refused a simulated run because an unrelated webcam
    # was unplugged, and left `current: camera` sitting outside `available`.
    if camera_capability(health)[0] or simulated_frames_available():
        sources.append(PerceptionSource.CAMERA.value)
    return sources


#: The reference dataset and model the offline regression uses by default. The
#: tutorial bolt is the ONLY complete FoundationPose input set available —
#: intrinsics, RGB, depth, a registration mask and a textured mesh — which is
#: why it, and not a WISEPACK pipe section, is the regression object. WISEPACK
#: does not package bolts; see perception/foundationpose/REFERENCE_ASSETS.md.
DEFAULT_REFERENCE_DATASET = (
    "Robot-Mania-Bin-Picking-Tutorial/isaac_bin_picking/"
    "FoundationPose_related/bolt")
DEFAULT_REFERENCE_MODEL_ID = "tutorial_bolt"

#: The FoundationPose provider, built lazily and shared. Cheap to construct —
#: it holds a URL and a registry, opens no camera and imports no torch — but
#: rebuilding it per request would re-read the object registry on every poll.
_FOUNDATIONPOSE_PROVIDER = None
_FOUNDATIONPOSE_CAPABILITY: Dict[str, Any] = {}
_FOUNDATIONPOSE_CAPABILITY_AT: Optional[float] = None


def foundationpose_provider():
    """The shared provider, or None when this deployment cannot load it.

    NEVER RAISES INTO A HANDLER. FoundationPose is opt-in; a deployment that
    never built the worker must render the rest of the dashboard normally, so an
    import failure becomes "unavailable, here is why" rather than a 500.
    """
    global _FOUNDATIONPOSE_PROVIDER
    if _FOUNDATIONPOSE_PROVIDER is not None:
        return _FOUNDATIONPOSE_PROVIDER
    try:
        perception_dir = os.path.join(REPO, "perception")
        if perception_dir not in sys.path:
            sys.path.insert(0, perception_dir)
        from providers.foundationpose_rgbd import (              # noqa: PLC0415
            FoundationPoseProvider)
        _FOUNDATIONPOSE_PROVIDER = FoundationPoseProvider()
    except Exception:                                            # noqa: BLE001
        return None
    return _FOUNDATIONPOSE_PROVIDER


def foundationpose_capability(force: bool = False) -> Dict[str, Any]:
    """The whole FoundationPose inference chain. A LIVE question, cached briefly.

    Cached for the same reason the camera capability is: the panel polls, and
    asking a container over HTTP on every refresh would make the dashboard's
    responsiveness depend on a worker that is allowed to be absent. The TTL is
    short enough that starting the worker is noticed without a restart.
    """
    global _FOUNDATIONPOSE_CAPABILITY, _FOUNDATIONPOSE_CAPABILITY_AT
    now = time.time()
    if (not force and _FOUNDATIONPOSE_CAPABILITY_AT is not None
            and now - _FOUNDATIONPOSE_CAPABILITY_AT < CAMERA_CAPABILITY_TTL_S):
        return _FOUNDATIONPOSE_CAPABILITY
    provider = foundationpose_provider()
    if provider is None:
        answer = {
            "method": PerceptionMethod.FOUNDATIONPOSE_RGBD.value,
            "worker_reachable": False, "runtime_ready": False,
            "inference_ready": False, "offline_regression_available": False,
            "rgbd_camera_available": False, "models": [],
            "blocked_by": [
                "the FoundationPose provider could not be loaded in this "
                "deployment. It is OPT-IN: build the worker with "
                "./scripts/setup_foundationpose.sh."],
        }
    else:
        # THE WORKER ANSWERS THIS NOW. RGB-D acquisition lives in the
        # container, so the camera's presence is the worker's observation, not
        # a constant here — and a camera plugged in while WISEPACK runs is
        # noticed without a restart, which a hard-coded False could never do.
        # ASKED ABOUT THE ACQUISITION IN USE. The same worker gives different
        # answers for a physical D435 and a simulated camera, and a capability
        # with no stated source cannot be read correctly by either.
        with STATE.lock:
            engine = STATE.engine
        batch = getattr(engine, "observation_batch", None) if engine else None
        acquisition = (ACQUISITION_ISAAC
                       if getattr(batch, "acquisition", "") == "simulated_rgbd"
                       else ACQUISITION_REALSENSE)
        answer = provider.capability(
            acquisition=acquisition,
            simulated_frames_available=_simulated_rgbd_observation() is not None)
    # WHO HOLDS THE CAMERA. Carried with the capability so the panel can say
    # what a method switch would require, rather than discovering it when two
    # providers both open one device.
    #
    # THE DEPTH FLAG IS THE WORKER'S ANSWER, NOT A CONSTANT. It was hard-coded
    # False, which was true only while no D435 existed; with one attached the
    # same response said `rgbd_camera_available: true` and, four lines below,
    # that no RGB-D camera was attached to this host. It is read from the
    # capability just computed — the worker's own observation, through the
    # provider — so the dashboard still never opens the device itself.
    #
    # `depth_holder` STAYS EMPTY, so an attached camera reads as FREE rather
    # than HELD. The worker owns the device but opens it only to acquire, and
    # no acquisition has yet put a physical RGB-D frame into a run. Naming a
    # holder here would claim a stream nobody has started.
    from wisepack_core.camera_ownership import current_ownership  # noqa: PLC0415
    answer["camera_ownership"] = current_ownership(
        colour_available=camera_capability()[0],
        depth_available=bool(answer.get("rgbd_camera_available")),
        colour_holder=PerceptionMethod.PLANAR_FASTERRCNN.value
        if camera_capability()[0] else "").to_dict()
    _FOUNDATIONPOSE_CAPABILITY = answer
    _FOUNDATIONPOSE_CAPABILITY_AT = now
    return answer


def available_perception_methods(capability: Optional[Dict[str, Any]] = None
                                 ) -> List[str]:
    """Methods this deployment can run RIGHT NOW.

    The planar method is always listed: it is the default and its availability
    is the camera's, which the object-source selector already reports. Listing
    it as unavailable when the camera is unplugged would make the METHOD
    selector a second, competing camera indicator.
    """
    methods = [PerceptionMethod.PLANAR_FASTERRCNN.value]
    document = foundationpose_capability() if capability is None else capability
    # AVAILABLE MEANS THE METHOD CAN RUN, which for FoundationPose means the
    # whole chain: worker, GPU, weights and a CAD model. Not "Docker exists".
    if document.get("runtime_ready") and document.get("models"):
        methods.append(PerceptionMethod.FOUNDATIONPOSE_RGBD.value)
    return methods


def perception_method_state(capability: Optional[Dict[str, Any]] = None
                            ) -> PerceptionMethodState:
    """available + draft + what the RUNNING batch was actually measured with."""
    document = foundationpose_capability() if capability is None else capability
    available = available_perception_methods(document)
    with STATE.lock:
        selected = str(STATE.settings.get("perception_method")
                       or DEFAULT_PERCEPTION_METHOD)
        engine = STATE.engine
    selected = resolve_perception_method_selection(selected, available)

    # CURRENT IS THE BATCH'S OWN PROVENANCE, read off the batch rather than
    # from any setting. A preset run has no physical batch and therefore no
    # method: empty, not "planar", because nothing measured anything.
    current = ""
    if engine is not None:
        try:
            state = engine.perception_state() or {}
            current = str((state.get("batch") or {}).get("perception_method") or "")
        except Exception:                                        # noqa: BLE001
            current = ""
    else:
        # LIVE MODES: this process does not run the engine. The orchestrator
        # publishes the measuring method beside the object source on the
        # perception status topic, and that document is authoritative.
        with STATE.lock:
            mirror = STATE.ros_mirror
        published = (mirror or {}).get("perception_status") or {}
        current = str(published.get("run_perception_method") or "")

    reasons: Dict[str, str] = {}
    if PerceptionMethod.FOUNDATIONPOSE_RGBD.value not in available:
        blocked = document.get("blocked_by") or []
        reasons[PerceptionMethod.FOUNDATIONPOSE_RGBD.value] = (
            "; ".join(str(b) for b in blocked)
            or "the FoundationPose worker is not available")
    return PerceptionMethodState(current=current, selected=selected,
                                 available=available,
                                 unavailable_reasons=reasons)


def object_source_state(health: Optional[Dict[str, Any]] = None
                        ) -> ObjectSourceState:
    """Capability + draft + what the RUNNING run actually used."""
    usable, reason = camera_capability(health)
    simulated = simulated_frames_available()
    available = [PerceptionSource.SIM.value]
    if usable or simulated:
        available.append(PerceptionSource.CAMERA.value)
    if simulated and not usable:
        # The camera source IS usable — through the simulated backend. The
        # planar service's absence is context, not a blocker, and saying
        # otherwise beside a working run describes the wrong device.
        reason = ("the planar webcam service is not answering; the simulated "
                  "RGB-D backend is available")
    with STATE.lock:
        selected = str(STATE.settings.get("object_source")
                       or PERCEPTION_SOURCE.value)
        engine = STATE.engine
        mirror = STATE.ros_mirror
    if SOURCE == "sim":
        current = (engine.config.perception_source.value if engine
                   else PERCEPTION_SOURCE.value)
    else:
        # LIVE MODES: the orchestrator owns the run, and it latches its own
        # object-source document on the perception status topic. Its `current`
        # is authoritative — this process does not run the engine and must not
        # guess what the container is doing.
        published = ((mirror or {}).get("perception_status") or {}
                     ).get("object_source") or {}
        current = str(published.get("current") or PERCEPTION_SOURCE.value)
        if published.get("selected"):
            # The orchestrator's draft is the one that will actually be used;
            # the browser's copy is only a rendering of it.
            selected = str(published["selected"])
    return ObjectSourceState(
        current=current, selected=selected, available=available,
        camera_unavailable_reason="" if usable else reason,
        service_url=perception_client().url)


def start_run(settings: Dict[str, Any],
              acquire: bool = True,
              allow_sources: Optional[List[str]] = None) -> WorkflowEngine:
    """Plan a fresh run and stop at the approval gate. Never auto-executes.

    TWO SOURCES, TWO WAYS TO ACQUIRE A BATCH. A preset run has its objects the
    moment the scenario is generated and plans straight through to the gate,
    exactly as it always has. A camera run has nothing until a frame has been
    analysed, so with `acquire=False` the run is created and left at the scan
    for `detect_physical_objects` to fill — which is what lets the operator
    switch source without the dashboard restarting.

    IN PHYSICAL PERCEPTION MODE THIS MAY LEGITIMATELY STOP EARLY. If the camera
    or the detector cannot deliver a batch there is nothing to plan, and §15
    forbids substituting simulated detections. The engine is returned in its
    failed-perception state — the run exists, the Physical Perception panel
    shows why it has no objects, and the operator retries after fixing the cause.
    """
    engine = build_engine(settings, allow_sources=allow_sources)
    engine.generate_or_load_scenario()
    if not acquire:
        return engine
    try:
        engine.scan_and_detect()
    except PerceptionUnavailable as exc:
        STATE.notice = f"physical perception unavailable: {exc}"
        return engine
    engine.generate_plans()
    engine.digital_twin_validate()
    engine.request_approval()
    return engine


# --------------------------------------------------------------------------- #
# Background driver (sim mode)
# --------------------------------------------------------------------------- #


async def sim_driver() -> None:
    """Advance execution one placement at a time once the operator approves.

    It never approves on its own. If the plan is unapproved this loop simply
    idles — which is the visible, demonstrable form of the safety invariant.
    """
    with STATE.lock:
        STATE.engine = start_run(STATE.settings)
        STATE.events.clear()
        for event in STATE.engine.log.events():
            STATE.events.append(event.to_dict())
    while True:
        await asyncio.sleep(STEP_PERIOD_S)
        with STATE.lock:
            engine = STATE.engine
            if engine is None or not STATE.auto_step or engine.finished:
                continue
            try:
                engine.step_execution()
            except (ApprovalRequired, AnomalyHold):
                # Expected while awaiting a decision or held by an anomaly.
                STATE.auto_step = False
            except Exception as exc:                    # noqa: BLE001
                STATE.notice = f"execution error: {exc}"
                STATE.auto_step = False
            if engine.finished:
                STATE.auto_step = False
                _write_artifacts_locked()


def _write_artifacts_locked() -> Dict[str, Any]:
    """Persist evidence at the end of a run. Caller holds the lock.

    THE OUTCOME IS RECORDED, not just announced. A run that completed and then
    failed to save its evidence has half-failed, and a notice that scrolls away
    beside a green COMPLETE badge is not a report of that.
    """
    engine = STATE.engine
    if engine is None or engine.selected is None:
        return {"ok": False, "reason": "no completed run to write"}
    try:
        kpis = engine.kpis(latest_latency_p50_ms(RESULTS_DIR))
        artifacts = write_run_artifacts(
            engine.scenario, engine.baseline, engine.optimized, engine.selected,
            kpis, engine.log, RESULTS_DIR)
        write_validation_report(
            engine.scenario, engine.baseline, engine.optimized, engine.selected,
            kpis, engine.log, artifacts, RESULTS_DIR)
        # THE ACTUAL PATH, not a guess at one. The old message said "results/…"
        # whatever the destination really was, so an operator following it
        # looked in the wrong place.
        written = os.path.join(RESULTS_DIR, f"wisepack-run-{artifacts.stamp}.json")
        STATE.artifacts = {"ok": True, "directory": RESULTS_DIR,
                           "path": written, "stamp": artifacts.stamp,
                           "error": ""}
        STATE.notice = f"artefacts written: {written}"
    except Exception as exc:                            # noqa: BLE001
        STATE.artifacts = {"ok": False, "directory": RESULTS_DIR, "path": "",
                           "stamp": "", "error": str(exc)}
        STATE.notice = f"artefact write failed: {exc}"
    return STATE.artifacts


# --------------------------------------------------------------------------- #
# FIWARE reader
# --------------------------------------------------------------------------- #


def _orion_get(path: str, timeout: float = 3.0):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(f"{ORION}{path}",
                                 headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception:                                   # noqa: BLE001
        return None, None


def fiware_health() -> Dict[str, Any]:
    """Is an Orion-LD (not a plain NGSI-v2 Orion) reachable?"""
    status, body = _orion_get("/version")
    if status is None:
        return {"connected": False, "reason": f"no response from {ORION}"}
    text = json.dumps(body or {}).lower()
    if "orionld" not in text and "orion" in text:
        return {"connected": False,
                "reason": "broker on :1026 is NGSI-v2 Orion, not Orion-LD"}
    return {"connected": True, "broker": ORION,
            "version": (body or {}).get("orionld version")
                       or (body or {}).get("version", "unknown")}


def fiware_entity(entity_id: str) -> Optional[Dict[str, Any]]:
    status, body = _orion_get(
        f"/ngsi-ld/v1/entities/{entity_id}?local=true")
    return body if status == 200 else None


def _attr(entity: Optional[Dict[str, Any]], name: str) -> Any:
    """Read ``<attr>.value.data`` — the shape the Orion-LD DDS bridge produces.

    Documented in HARMONY's dds_native_contract.md: the bridge wraps every
    std_msgs payload under ``value.data``, for String and Int32 alike. A plain
    ``value`` is also accepted so a hand-PATCHed entity still renders.
    """
    if not entity:
        return None
    attr = entity.get(name)
    if not isinstance(attr, dict):
        return None
    value = attr.get("value")
    if isinstance(value, dict) and "data" in value:
        return value["data"]
    if value == "uninitialized":
        return None
    return value


# --------------------------------------------------------------------------- #
# Topology
# --------------------------------------------------------------------------- #

# Position-free graph, exactly as TEMPO does it: a node declares what it IS and
# which layer it sits in, and the frontend derives coordinates. Adding a node
# needs no layout change here and no renderer change there.
TOPOLOGY = {
    "layers": ["perception", "planning", "supervision", "bus", "cloud"],
    "nodes": [
        {"id": "generator", "label": "Task generator", "kind": "Scenario",
         "role": "source", "layer": 0},
        {"id": "perception", "label": "Perception simulator", "kind": "Simulated",
         "role": "sim", "layer": 0},
        {"id": "optimizer", "label": "Packing optimizer", "kind": "Intelligence",
         "role": "brain", "layer": 1},
        {"id": "twin", "label": "Digital Twin validator", "kind": "Validation",
         "role": "check", "layer": 1},
        {"id": "orchestrator", "label": "HitL orchestrator", "kind": "py_trees",
         "role": "control", "layer": 2},
        {"id": "operator", "label": "Operator", "kind": "Human",
         "role": "human", "layer": 2},
        {"id": "robot", "label": "Robot simulator", "kind": "Simulated",
         "role": "sim", "layer": 2},
        {"id": "dds", "label": "ROS 2 / DDS", "kind": "Middleware",
         "role": "bus", "layer": 3},
        {"id": "orion", "label": "Orion-LD (NGSI-LD)", "kind": "Context broker",
         "role": "cloud", "layer": 4},
        {"id": "dashboard", "label": "Dashboard / analytics", "kind": "UI",
         "role": "cloud", "layer": 4},
    ],
    # c: "telemetry" renders solid, "control" renders dashed.
    "edges": [
        {"s": "generator", "t": "optimizer", "c": "telemetry"},
        {"s": "perception", "t": "optimizer", "c": "telemetry"},
        {"s": "optimizer", "t": "twin", "c": "telemetry"},
        {"s": "twin", "t": "orchestrator", "c": "telemetry"},
        {"s": "orchestrator", "t": "operator", "c": "telemetry"},
        {"s": "operator", "t": "orchestrator", "c": "control"},
        {"s": "orchestrator", "t": "robot", "c": "control"},
        {"s": "robot", "t": "orchestrator", "c": "telemetry"},
        {"s": "generator", "t": "dds", "c": "telemetry"},
        {"s": "orchestrator", "t": "dds", "c": "telemetry"},
        {"s": "robot", "t": "dds", "c": "telemetry"},
        {"s": "optimizer", "t": "dds", "c": "telemetry"},
        {"s": "dds", "t": "orion", "c": "telemetry"},
        {"s": "orion", "t": "dds", "c": "control"},
        {"s": "orion", "t": "dashboard", "c": "telemetry"},
        {"s": "dashboard", "t": "orion", "c": "control"},
    ],
}


def topology_status(snap=None) -> Dict[str, str]:
    """Colour each node from the snapshot's stage — same logic in every mode."""
    if snap is None:
        snap = _provider().snapshot()
    stage = snap.stage
    degraded = stage == "DEGRADED"
    approved = snap.approval_state == "approved"
    fiware_ok = snap.fiware_connected

    def s(active: bool, warn: bool = False) -> str:
        if degraded:
            return "fault"
        if warn:
            return "standby"
        return "active" if active else "idle"

    planning = stage in ("GENERATE_BASELINE_PLAN", "GENERATE_OPTIMIZED_PLAN",
                         "REPLAN")
    executing = stage in ("PICK_ITEM", "VERIFY_PICK", "PLACE_ITEM",
                          "VERIFY_PLACEMENT", "UPDATE_CONTAINER_STATE",
                          "NEXT_ITEM")
    has_plan = snap.plans_ready
    return {
        "generator": s(stage == "GENERATE_OR_LOAD_SCENARIO" or bool(snap.scenario)),
        "perception": s(stage in ("SCAN_SOURCE_BIN", "DETECT_ITEMS")
                        or snap.detected_count > 0),
        "optimizer": s(planning or has_plan),
        "twin": s(stage == "DIGITAL_TWIN_VALIDATE" or bool(snap.plan_status)),
        "orchestrator": s(True, warn=degraded),
        "operator": s(stage == "WAIT_FOR_OPERATOR_APPROVAL",
                      warn=stage == "WAIT_FOR_OPERATOR_APPROVAL"),
        "robot": s(executing and approved),
        "dds": s(SOURCE in ("ros", "fiware")),
        "orion": ("active" if fiware_ok else
                  ("idle" if fiware_ok is None else "fault")),
        "dashboard": "active",
    }


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = []
    if SOURCE == "sim":
        tasks.append(asyncio.create_task(sim_driver()))
    else:
        from ros_observer import start_ros_observer                # noqa: PLC0415
        tasks.append(asyncio.create_task(start_ros_observer(STATE)))
    if SOURCE in ("ros", "fiware"):
        tasks.append(asyncio.create_task(_fiware_poller()))
    yield
    # Clean shutdown: cancel the background tasks and await them so their
    # CancelledError is consumed here rather than surfacing as an uvicorn
    # lifespan traceback on Ctrl+C. Container lifecycle is unaffected.
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _fiware_poller() -> None:
    while True:
        health = await asyncio.get_event_loop().run_in_executor(None, fiware_health)
        with STATE.lock:
            STATE.fiware_connected = health["connected"]
            STATE.fiware_last_error = health.get("reason", "")
        await asyncio.sleep(5.0)


app = FastAPI(title="WISEPACK dashboard", lifespan=lifespan)


def _provider():
    """The one place a mode maps to a snapshot source.

    Every endpoint below goes through this, so no endpoint can accidentally
    depend on STATE.engine in a live mode — which is precisely how the ROS and
    FIWARE dashboards ended up rendering empty panels while reporting healthy.
    """
    if SOURCE == "sim":
        return SimSnapshotProvider(STATE, lambda: latest_latency_p50_ms(RESULTS_DIR))
    if SOURCE == "fiware":
        return FiwareSnapshotProvider(STATE, read_fiware_entities)
    return RosSnapshotProvider(STATE)


def read_fiware_entities() -> Dict[str, Any]:
    """Read every mapped entity from Orion-LD, unwrapped to plain values."""
    from wisepack_fiware.entities import ENTITY_IDS                # noqa: PLC0415
    out: Dict[str, Any] = {}
    for name, entity_id in ENTITY_IDS.items():
        entity = fiware_entity(entity_id)
        if not entity:
            continue
        out[name] = {k: parse_attr(v) for k, v in entity.items()
                     if k not in ("id", "type", "@context")}
    return out


@app.get("/api/state")
def api_state():
    """Complete initial state. The page renders fully from this one call."""
    snap = _provider().snapshot()
    with STATE.lock:
        settings = dict(STATE.settings)
    payload = snap.to_state()
    payload["fiware"]["broker"] = ORION

    # THE SCENARIO CONTROLS MUST DESCRIBE THE ACTIVE RUN.
    #
    # `STATE.settings` is this process's own last-submitted form state. In live
    # mode the run belongs to the orchestrator in another process, so the form
    # sat at its default (`mixed_pipes_dense`) while the header correctly showed
    # the running scenario (`isaac_cylinders_smoke-s42`) — the same screen
    # naming two different scenarios. The ACTIVE run wins wherever it is known.
    scenario = snap.scenario or {}
    active_preset = scenario.get("preset")
    active_seed = scenario.get("seed")

    # The draft is seeded from the active run ONLY while the operator has not
    # chosen anything yet. Overwriting it on every poll — which is what this
    # did — meant a selection was reverted within a second of being made, and
    # made the dropdown feel broken even when it was not locked.
    if not STATE.settings_touched:
        if active_preset:
            settings = {**settings, "preset": active_preset}
        if active_seed is not None:
            settings = {**settings, "seed": active_seed}

    # -- THE ROBOT -------------------------------------------------------- #
    #
    # Three separate things, kept separate on purpose:
    #
    #   `robots`        the catalogue, from the registry (see /api/config/robots)
    #   `active_robot`  what the RUNNING run is executing with
    #   `settings.robot_id`  the DRAFT for the next run
    #
    # The draft follows exactly the same rule as the preset: seeded from the
    # active run (or the configured default) while untouched, owned by the
    # operator afterwards, and never overwritten by a poll. Conflating draft
    # and active is what made the preset dropdown unusable before, and a robot
    # selector that silently reverts is worse — it would look as though the
    # operator had chosen an arm that is not the one that then moves.
    robots_catalogue: Dict[str, Any] = {"robots": [], "default_robot": None,
                                        "error": ""}
    try:
        from wisepack_core.robots import load_registry              # noqa: PLC0415
        robots_catalogue = load_registry().to_public_dict()
    except Exception as exc:                                        # noqa: BLE001
        # Named, not swallowed. A broken registry must show as a broken
        # registry in the panel, not as "no robots exist".
        robots_catalogue["error"] = str(exc)

    execution = payload.get("execution", {}) or {}
    active_robot = execution.get("robot")
    active_robot_id = execution.get("robot_id")

    # ROBOT SELECTION IS ONLY MEANINGFUL FOR A PHYSICAL BACKEND. In the logical
    # modes there is no robot at all, and offering a choice between two arms
    # neither of which will move is worse than offering none — the frontend
    # renders a fixed "Logical workflow simulator" line instead.
    #
    # `known` matters as much as `physical`: until the orchestrator has said
    # which backend is authoritative, claiming "no robot" would be asserting
    # something about a run this process has not heard from.
    physical = bool(execution.get("physical"))
    if physical:
        # Seeded only while untouched, exactly like the preset.
        if not STATE.settings_touched and settings.get("robot_id") is None:
            settings = {**settings,
                        "robot_id": active_robot_id
                        or robots_catalogue.get("default_robot")}
    else:
        # NOT "whatever the default happens to be". A logical run has no robot,
        # and reporting one here is what put a disabled "Franka Emika Panda"
        # beside help text saying no robot could be selected.
        settings = {**settings, "robot_id": None}
        active_robot, active_robot_id = None, None
    draft_robot_id = settings.get("robot_id") if physical else None
    draft_profile = next((r for r in robots_catalogue.get("robots", [])
                          if r.get("id") == draft_robot_id), None)
    draft_preset = settings.get("preset")
    robot_preset_conflict = ""
    if draft_profile and draft_preset:
        compatible = draft_profile.get("compatible_presets") or []
        if compatible and draft_preset not in compatible:
            robot_preset_conflict = (
                f"{draft_profile['display_name']} is configured for "
                f"{', '.join(compatible)}; {draft_preset} is not among them")

    # ACTIVE SCENARIO vs DRAFT — two different things, and conflating them made
    # the preset dropdown unusable in every live mode.
    #
    # Every launcher starts a run automatically, so the workflow is at
    # WAIT_FOR_OPERATOR_APPROVAL within seconds of the page loading. The controls
    # were locked in exactly that state, which meant the operator never got a
    # chance to choose a preset at all — the UI said "cannot be changed until it
    # finishes or is reset" from the moment it appeared.
    #
    # The controls are therefore a DRAFT for the NEXT run. They never mutate the
    # running scenario: only "Generate & plan" does, and only when the operator
    # asks for it. The header and the Digital Twin keep showing the ACTIVE
    # scenario, so the two are always distinguishable.
    # THE OBJECT SOURCE, resolved once per render: what is available, what the
    # operator drafted, and what the run on screen actually used.
    source_state = object_source_state()
    run_source = PerceptionSource(source_state.current)

    running_stages = ("WAIT_FOR_OPERATOR_APPROVAL", "PICK_ITEM", "VERIFY_PICK",
                      "PLACE_ITEM", "VERIFY_PLACEMENT", "UPDATE_CONTAINER_STATE",
                      "NEXT_ITEM", "REPLAN")
    run_active = snap.control_stage in running_stages
    payload.update({
        # The draft the controls edit. Seeded from the active run the first time
        # the page loads; after that the browser owns it (see the frontend), so
        # a poll can never overwrite a selection mid-edit.
        "settings": settings,
        # Presets this backend can actually execute. The Isaac backend is a
        # bench-scale Panda: offering it the 40-item industrial benchmark would
        # be offering a run that cannot physically happen.
        # Which presets the ACTIVE backend can physically instantiate and reach.
        # The dashboard marks the rest unavailable with the reason, rather than
        # offering a run whose first pick is impossible.
        "preset_compatibility": (_preset_compatibility(draft_robot_id)
                                 if snap.execution_backend == "isaac" else {}),
        "default_preset": ("isaac_cylinders_smoke"
                           if snap.execution_backend == "isaac"
                           else "mixed_pipes_dense"),
        # Explicitly NOT locked any more. Kept in the payload because the
        # frontend and the tests both assert on it.
        "settings_locked": False,
        "settings_locked_reason": "",
        # Is a run in progress or awaiting a decision? Not a lock — a reason to
        # ASK before discarding it.
        "run_active": run_active,
        "run_active_reason": (
            "a run is active or awaiting your decision — generating a new "
            "scenario will discard it" if run_active else ""),
        # What is ACTUALLY running, for the header and the Digital Twin.
        "active_scenario": {
            "preset": active_preset,
            "seed": active_seed,
            "scenario_id": scenario.get("scenario_id"),
        },
        "active_preset": active_preset,
        # The catalogue the selector is built from. Never duplicated in HTML.
        "robots": robots_catalogue.get("robots", []),
        "default_robot": robots_catalogue.get("default_robot"),
        "robots_error": robots_catalogue.get("error", ""),
        "registry_revision": robots_catalogue.get("registry_revision"),
        # Whether a robot CHOICE is meaningful at all here. False in the logical
        # modes, where the frontend shows a fixed execution-source line rather
        # than a dropdown of arms that will not move.
        "robot_selector": physical,
        # The EXECUTION SOURCE, named. In a logical mode this is what the panel
        # shows in place of a robot, so it has to be a real answer rather than
        # an absence.
        "execution_source_label": ("" if physical
                                   else "Logical workflow simulator"),
        "robot_selector_reason": (
            "" if physical else
            "the logical workflow simulator executes this run — there is no "
            "robot to select"),
        "active_robot": active_robot,
        "active_robot_id": active_robot_id,
        "draft_robot_id": draft_robot_id,
        # FOUR SEPARATE ANSWERS, never collapsed:
        #   active     what the run is executing with
        #   requested  what a switch asked for
        #   host       what the host supervisor says is actually running
        #   scene      what the acknowledged scene was built by
        "robot_switch": (execution.get("isaac") or {}).get("robot_switch"),
        "acknowledged_scene_robot_id": (
            ((execution.get("isaac") or {}).get("acknowledged_scene") or {})
            .get("robot_id")),
        "robot_preset_conflict": robot_preset_conflict,
        # Changing the robot is a NEW RUN, never a live switch. The frontend
        # uses this to require confirmation before resetting an Isaac run.
        "robot_change_requires_reset": bool(
            physical and active_robot_id and draft_robot_id
            and draft_robot_id != active_robot_id),
        "presets": sorted(PRESETS),
        "container_specs": {k: v["description"] for k, v in CONTAINER_SPECS.items()},
        "strategies": [s.value for s in Strategy],
        "topology_status": topology_status(snap),
        "commands": list(OPERATOR_COMMANDS),
        # THE THIRD AXIS, reported beside the other two and never folded into
        # either. The frontend shows the Physical Perception panel from this and
        # from nothing else.
        #
        # `source` here is THE RUNNING RUN'S — provenance, not configuration.
        # The operator's selection for the NEXT run is `object_source.selected`,
        # and the two are deliberately different fields: an operator who has
        # picked the camera for the next run is still watching a preset run, and
        # a panel that showed one number for both would be lying about one of
        # them.
        "perception": {
            "source": run_source.value,
            "label": run_source.label,
            "detail": run_source.detail,
            "physical": run_source.is_physical,
            "provenance": run_source.provenance,
        },
        # WHERE THE NEXT BATCH OF OBJECTS COMES FROM: capability, draft, and
        # what is running. Per run and switchable at runtime — no restart, and
        # no application-wide "camera mode".
        "object_source": source_state.to_dict(),
        # HOW the next physical batch is read. A THIRD axis: independent of the
        # source above, independent of the execution backend, and — like the
        # source — a per-run draft that never touches the run on screen.
        "perception_method": perception_method_state().to_dict(),
        # THE ACQUISITION AXIS travels with the state too, so the Scenario
        # controls can name the device the running run actually used instead of
        # the coarser object source.
        "acquisition": acquisition_state().to_dict(),
        # WHERE THE EVIDENCE WENT — or why it did not. A COMPLETE run whose
        # artefacts failed to save must say so where the completion is shown.
        "artifacts": (STATE.artifacts
                      or {"ok": None, "directory": RESULTS_DIR, "path": "",
                          "error": ""}),
        "ts": time.time(),
    })
    return payload


@app.get("/api/config/robots")
def api_config_robots():
    """The supported Isaac robots — READ-ONLY, and the ONLY robot list.

    The dashboard's Robot selector is built from this response. There is no
    hard-coded ``<option>`` list in index.html and no robot array in the
    JavaScript, and ``tests/test_isaac_robots.py`` fails the build if one
    appears: two lists that agree today are two lists that disagree after the
    next edit, and the one the operator sees would be the stale one.

    PUBLIC-SAFE SUBSET. Identity, capability and status only — never the asset
    URL, the prim paths or the joint names. Those are of no use to an operator
    choosing an arm, and an asset-server path in a web response is free
    reconnaissance.

    Serving a 503 rather than an empty list when the registry cannot be read: an
    empty list renders as "no robots configured", which is a different and
    wrong claim from "the robot registry is broken".
    """
    from wisepack_core.robots import RobotConfigError, load_registry  # noqa: PLC0415
    try:
        return load_registry().to_public_dict()
    except RobotConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/plans")
def api_plans():
    """Geometry for the Digital Twin view and the baseline/optimized comparison."""
    return _provider().snapshot().to_plans()


@app.get("/api/kpis")
def api_kpis():
    return _provider().snapshot().to_kpis()


@app.get("/api/strategies")
def api_strategies():
    """Return the latest strategy comparison from the unified snapshot.

    This is a READ. In every mode it reflects whatever comparison the
    authoritative source last produced — the local engine in sim mode, the
    orchestrator's `/wisepack/plan/strategy_comparison` topic in ROS/FIWARE. It
    never runs a comparison itself, so it cannot be a second source of truth.
    Triggering is POST /api/command {"command": "compare_strategies"}.
    """
    return _provider().snapshot().to_strategies()


@app.get("/api/events")
def api_events(limit: int = 150):
    return _provider().snapshot().to_events(limit)


@app.get("/api/analytics")
def api_analytics():
    """Aggregations for the analytics panel, in every mode."""
    payload = _provider().snapshot().to_analytics()
    payload["latency"] = latest_artifact("dds-fiware-latency", RESULTS_DIR)
    return payload


@app.get("/api/topology")
def api_topology():
    """The node graph, with the perception node naming the ACTIVE source.

    The graph is otherwise unchanged: a camera replaces the perception
    simulator in place rather than adding a branch, because that is exactly
    what it does — one node, two possible implementations, the same edges.
    """
    topology = {**TOPOLOGY, "nodes": [dict(n) for n in TOPOLOGY["nodes"]]}
    # THE RUNNING RUN'S source, so the graph describes what actually produced
    # the objects on screen — not what the operator has drafted for the next
    # one.
    if PerceptionSource(object_source_state().current).is_physical:
        for node in topology["nodes"]:
            if node["id"] == "perception":
                node["label"] = "Physical perception"
                node["kind"] = "Camera + detector"
                node["role"] = "sensor"
    return {**topology, "status": topology_status(_provider().snapshot())}


@app.get("/api/execution")
def api_execution():
    """Which backend executed, and — when it is Isaac — what physically happened.

    A COMPACT DIAGNOSTIC, not a redesign. The item-by-item progression already
    renders on the existing execution timeline, because Isaac's physical states
    are mapped onto the existing WISEPACK stages rather than shown as a parallel
    state machine. What this adds is the part the timeline has nowhere to put:
    the measured final pose of each item and its distance from the planned one.

    `target_pose` and `actual_pose` are always reported as a PAIR. Showing the
    measured pose without the target it was aiming at invites reading it as
    agreement, and a released cylinder does not land exactly where the optimizer
    planned — that is the physics working, not a defect.
    """
    snap = _provider().snapshot()
    payload = snap.backend_badge()
    payload["source"] = snap.mode
    payload["results"] = list(snap.isaac_results)
    errors = [r["position_error_mm"] for r in snap.isaac_results
              if r.get("position_error_mm") is not None]
    payload["summary"] = {
        "items_reported": len(snap.isaac_results),
        "items_completed": sum(1 for r in snap.isaac_results
                               if r.get("state") == "ITEM_COMPLETED"),
        "items_failed": sum(1 for r in snap.isaac_results
                            if r.get("state") == "ITEM_FAILED"),
        # Mean and max placement error, in millimetres, MEASURED. Absent rather
        # than zero when nothing has been placed yet: a measured zero and "no
        # measurement" must never render the same.
        "mean_position_error_mm": (round(sum(errors) / len(errors), 1)
                                   if errors else None),
        "max_position_error_mm": round(max(errors), 1) if errors else None,
    }
    return payload


# --------------------------------------------------------------------------- #
# Physical perception (§9)
# --------------------------------------------------------------------------- #


#: Where the Isaac -> FoundationPose -> workarea chain leaves its artifacts.
SIMULATED_RGBD_DIR = os.path.join(REPO, ".cache-perception")
SIMULATED_RGBD_RESULT = os.path.join(SIMULATED_RGBD_DIR, "stage-c", "stage_c.json")

#: Where ./scripts/physical_c5.sh leaves the PHYSICAL D435 result.
#:
#: READ, NEVER RECOMPUTED, and deliberately NOT routed through Stage C. The
#: simulated chain carries a camera-to-work-area transform that Isaac knows and
#: the physical camera does not have; feeding a physical pose through it would
#: place a real object in a frame nobody has measured. This stays in the camera
#: optical frame all the way to the screen.
PHYSICAL_C5_DIR = os.path.join(REPO, ".cache-perception", "physical-c5")
PHYSICAL_C5_RESULT = os.path.join(PHYSICAL_C5_DIR, "physical_c5.json")

#: The artifact files the panel shows, and the only ones it will serve.
PHYSICAL_C5_IMAGES = {
    "rgb": "rgb.jpg",
    "depth": "depth_aligned.jpg",
    "mask": "mask_overlay.jpg",
    "overlay": "pose_overlay.jpg",
}


def _physical_c5_document():
    """The last physical D435 result, or None. NEVER raises."""
    if not os.path.isfile(PHYSICAL_C5_RESULT):
        return None
    try:
        with open(PHYSICAL_C5_RESULT, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:                                            # noqa: BLE001
        return None


def _simulated_rgbd_observation():
    """The most recent simulated-RGB-D observation, or None.

    READ, NEVER RECOMPUTED. The estimate was produced by the real FoundationPose
    worker against a real Isaac frame; re-running it here under different inputs
    would put a different number on screen from the one the workflow used.
    """
    if not os.path.isfile(SIMULATED_RGBD_RESULT):
        return None
    try:
        with open(SIMULATED_RGBD_RESULT, encoding="utf-8") as handle:
            document = json.load(handle)
        from wisepack_core.domain import PhysicalObservation     # noqa: PLC0415
        observation = PhysicalObservation.from_dict(document["observation"])
    except Exception:                                            # noqa: BLE001
        return None
    return observation, {
        "acquisition": document.get("acquisition", {}),
        "camera_to_workarea": document.get("camera_to_workarea_transform", {}),
        "model_frame_pose": document.get("model_frame_pose", {}),
        "task_reference_point": document.get("task_reference_point", {}),
        "evaluation": document.get("evaluation", {}),
        "preset": "cad_cylinder5_single",
        "batch_id": "simulated-rgbd-1",
    }


def _perception_payload() -> Dict[str, Any]:
    """Everything the Physical Perception panel renders, in every mode.

    ALWAYS ANSWERS, including in `sim` perception mode, where it says plainly
    that perception is simulated rather than returning an empty physical panel
    that could be read as a broken camera.
    """
    # THE RUNNING RUN'S source, not a process-wide setting: the panel describes
    # the batch WISEPACK is planning from. The selector's draft travels
    # separately in `object_source`, because "running now" and "next run" are
    # different answers and the panel shows both.
    source_state = object_source_state()
    fp_capability = foundationpose_capability()
    method_state = perception_method_state(fp_capability)
    run_source = PerceptionSource(source_state.current)
    payload: Dict[str, Any] = {
        "perception_source": run_source.value,
        "perception_source_label": run_source.label,
        "perception_source_detail": run_source.detail,
        "physical": run_source.is_physical,
        "object_source": source_state.to_dict(),
        # THE METHOD IS A THIRD AXIS and travels with the panel in every mode,
        # including preset mode: an operator deciding whether to switch to the
        # camera needs to see which methods that camera could be read with.
        "perception_method": method_state.to_dict(),
        "foundationpose": fp_capability,
        # THE SIMULATED RGB-D RUN, when one is on screen. Everything here comes
        # from the SAME artifact the workflow consumed, so the images, the
        # estimate and the plan cannot disagree.
        "simulated_rgbd": _simulated_rgbd_panel(),
        # THE FOURTH AXIS, exposed so the panel can name the device that
        # actually produced the batch rather than describing a webcam.
        "acquisition": acquisition_state().to_dict(),
        # WHICH CAD MODEL THIS SCENARIO IS ABOUT, resolved from the scenario's
        # own items rather than guessed from its name.
        "object_model": _scenario_object_model(),
        # THE PROXY DISCLOSURE (§9). Unobtrusive, but always present in the
        # payload so the panel cannot render real detections without it.
        "proxy_note": (
            "Physical bottles are currently used as proxies for the cylindrical "
            "workpieces WISEPACK packages. Their detected position and "
            "orientation become domain-neutral object observations."),
        # THE §12 GUARD, carried with the data rather than left to the frontend.
        "confidence_note": (
            "Detector confidence is not a detection rate. Vision detection "
            "rate: not measured — real detector active; no ground-truth trial."),
        # EXECUTION IS A SEPARATE AXIS and the panel says so, so nobody reads
        # "camera perception" as "physical robot".
        "independent_of_execution_backend": True,
    }
    client = perception_client()
    if not run_source.is_physical:
        # THE PANEL DOES NOT DISAPPEAR because this run came from a preset.
        #
        # The camera is a capability of the deployment, and an operator about to
        # switch to it needs to see whether it is there — "Camera: connected,
        # Detector: ready, Current run source: Preset scenario" is exactly the
        # state that makes the switch an informed one. Hiding the whole panel
        # made the capability invisible until after a restart, which is the
        # behaviour this change removes.
        payload["health"] = (client.health() if source_state.camera_available
                             else {"source": run_source.value,
                                   "service_reachable": None})
        payload["camera_available"] = source_state.camera_available
        payload["camera_unavailable_reason"] = (
            source_state.camera_unavailable_reason)
        # NO BATCH, because this run has none: its objects were generated. The
        # panel says so rather than showing a camera batch from an earlier run
        # beside a plan that has nothing to do with it.
        payload["batch"] = None
        payload["scene_objects"] = []
        if source_state.camera_available:
            payload["live_url"] = client.live_url
            payload["annotated_url"] = "/api/perception/image/annotated"
            payload["raw_url"] = "/api/perception/image/raw"
        return payload

    payload["health"] = client.health()
    payload["camera_available"] = source_state.camera_available
    payload["camera_unavailable_reason"] = source_state.camera_unavailable_reason
    payload["live_url"] = client.live_url
    payload["annotated_url"] = "/api/perception/image/annotated"
    payload["raw_url"] = "/api/perception/image/raw"

    # THE ENGINE'S VIEW WINS when it has one: the panel must describe the batch
    # WISEPACK is actually planning from, not a later one the service happens to
    # hold. Those differ the moment a detection fails — and showing the
    # service's newest success beside a plan built from an older batch is
    # precisely the confusion §15 is about.
    state = None
    with STATE.lock:
        engine = STATE.engine
        if engine is not None:
            state = engine.perception_state()
    if state is not None:
        payload.update({k: v for k, v in state.items()
                        if k not in ("perception_source",
                                     "perception_source_label",
                                     "perception_source_detail", "physical")})
    else:
        batch = client.last_detection()
        payload["batch"] = batch.to_dict() if batch else None
        payload["scene_objects"] = (batch.scene_objects()
                                    if batch and batch.ok else [])
    return payload


def _scenario_object_model() -> Dict[str, Any]:
    """The CAD model the current scenario is about, from its ITEMS.

    Resolved from the scenario rather than parsed out of a preset name: the
    items already declare `model_id`, and a name is not a specification.
    """
    with STATE.lock:
        engine = STATE.engine
    scenario = getattr(engine, "scenario", None) if engine else None
    items = list(getattr(scenario, "items", []) or [])
    models = sorted({getattr(i, "model_id", "") for i in items
                     if getattr(i, "model_id", "")})
    return {
        "model_ids": models,
        "model_id": models[0] if len(models) == 1 else "",
        "geometry_sources": sorted({getattr(i, "geometry_source", "")
                                    for i in items}),
        "preset": getattr(scenario, "preset", "") if scenario else "",
    }


def _simulated_rgbd_panel() -> Optional[Dict[str, Any]]:
    """What the dashboard shows about a simulated-RGB-D run, or None.

    STALE ARTEFACTS ARE LABELLED, NOT SHOWN AS CURRENT (§6). The panel reports
    the revision the observation was applied at; if the engine has moved on, the
    images belong to an earlier batch and the panel says so rather than
    presenting them beside a newer plan.
    """
    with STATE.lock:
        meta = getattr(STATE, "simulated_rgbd", None)
        engine = STATE.engine
    if not meta:
        return None
    document = dict(meta)
    document["available"] = True
    document["images"] = {
        "rgb": "/api/perception/simulated/image/rgb",
        "depth": "/api/perception/simulated/image/depth",
        "mask": "/api/perception/simulated/image/mask",
        "overlay": "/api/perception/simulated/image/overlay",
    }
    batch = engine.observation_batch if engine else None
    document["batch_acquisition"] = getattr(batch, "acquisition", "")
    document["applied_at_revision"] = document.get("applied_at_revision")
    if engine is not None:
        document["run_id"] = engine.run_id
        document["scenario_revision"] = engine.scenario_revision
        document["stale"] = bool(
            batch is None or getattr(batch, "acquisition", "") != "simulated_rgbd")
        if document["stale"]:
            document["stale_reason"] = (
                "the run on screen was not built from this simulated RGB-D "
                "acquisition; the images below belong to an earlier batch")
    return document


#: The Stage A/B artefacts, by name. Served rather than regenerated: they are
#: the frames the estimate was actually computed from.
_SIMULATED_IMAGES = {
    "rgb": ("stage-a", "d435_rgb.png"),
    "depth": ("stage-a", "d435_depth.png"),
    "mask": ("stage-a", "cylinder5_mask.png"),
    "overlay": ("stage-b", "overlay_estimate.png"),
}


@app.get("/api/perception/simulated/image/{kind}")
def api_simulated_image(kind: str):
    """Serve one acquisition image. A PROXY OF THE ARTEFACT, not a redraw."""
    from fastapi.responses import FileResponse                    # noqa: PLC0415
    entry = _SIMULATED_IMAGES.get(kind)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown image {kind!r}")
    path = os.path.join(SIMULATED_RGBD_DIR, entry[0], entry[1])
    if not os.path.isfile(path):
        raise HTTPException(
            status_code=404,
            detail=f"{kind} image not available; run ./scripts/stage_c.sh")
    return FileResponse(path, media_type="image/png")


@app.get("/api/perception")
def api_perception():
    """Physical perception status, the current observation batch and its poses."""
    return _perception_payload()


@app.get("/api/perception/foundationpose")
def api_foundationpose():
    """FoundationPose capability, object models, and the last estimate."""
    capability = foundationpose_capability(force=True)
    provider = foundationpose_provider()
    document: Dict[str, Any] = {
        "capability": capability,
        "method": perception_method_state(capability).to_dict(),
        "last_result": None,
        "last_result_error": "",
    }
    if provider is not None:
        result, reason = provider.client.last_result()
        document["last_result"] = result
        document["last_result_error"] = reason
        document["datasets"] = provider.client.datasets()[0]
    return document


@app.post("/api/perception/foundationpose/reference-regression")
def api_foundationpose_reference_regression(payload: Optional[Dict[str, Any]] = None):
    """Run the OFFLINE reference regression. NOT a camera acquisition.

    This exists so the whole WISEPACK path — worker, provider,
    PhysicalObservation, serialisation, dashboard — can be exercised while no
    depth camera exists. It is a separate endpoint from `detect_physical_objects`
    on purpose: routing a saved dataset through the control labelled "detect"
    would put a reference pose into a run as though a camera had produced it,
    which is precisely the deception the physical-camera work exists to avoid.

    The batch it returns is stamped `acquisition="reference"` and is NOT
    installed into the running scenario. Its poses are VALID estimates in the
    camera frame — `pose_valid` is True — and carry
    `workarea_pose_available=False`, which is the flag that keeps them out of
    the work area. Those are two different facts and the payload states both.
    """
    body = dict(payload or {})
    provider = foundationpose_provider()
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail="the FoundationPose provider is not available in this build")
    capability = foundationpose_capability(force=True)
    if not capability.get("offline_regression_available"):
        raise HTTPException(
            status_code=409,
            detail={"error": "the offline reference regression cannot run",
                    "blocked_by": capability.get("blocked_by", [])})
    dataset = str(body.get("dataset") or DEFAULT_REFERENCE_DATASET)
    model_id = str(body.get("model_id") or DEFAULT_REFERENCE_MODEL_ID)
    if "depth_scale_mm" not in body:
        # NO DEFAULT, the same rule the worker enforces: a uint16 millimetre
        # image and a float32 metre image are indistinguishable from the pixels,
        # and guessing is a factor-of-1000 error that looks like a real pose.
        raise HTTPException(
            status_code=400,
            detail="depth_scale_mm is required — how many millimetres one raw "
                   "depth unit represents (1.0 for a uint16 millimetre image)")
    batch = provider.acquire_reference(
        dataset=dataset, model_id=model_id,
        depth_scale_mm=float(body["depth_scale_mm"]),
        frame=int(body.get("frame", 0)),
        refine_iterations=int(body.get("refine_iterations", 5)))
    return {
        "batch": batch.to_dict(),
        "acquisition": batch.acquisition,
        # SAID TWICE, AND DELIBERATELY: once in the data and once in a label the
        # dashboard renders. A reference result that reaches an operator without
        # this wording is indistinguishable from a live measurement.
        # The wording follows the control: this is a SELF-TEST, and naming the
        # dataset in the label is what stops a bolt pose being read as a
        # WISEPACK measurement. "offline" stays — it is the load-bearing word.
        "label": "FoundationPose self-test — offline tutorial bolt dataset",
        "live": False,
        "note": _REFERENCE_NOTE(),
    }


def _REFERENCE_NOTE() -> str:
    provider = foundationpose_provider()
    if provider is None:
        return ""
    from providers.foundationpose_rgbd import REFERENCE_NOTE     # noqa: PLC0415
    return REFERENCE_NOTE


@app.get("/api/perception/physical")
def api_perception_physical():
    """The PHYSICAL D435 result, exactly as ./scripts/physical_c5.sh left it.

    WHAT THIS ROUTE WILL NOT DO. It does not estimate, does not open a camera,
    does not re-run FoundationPose, and does not convert anything into the work
    area. It reads one file and states what is in it. The pose stays in
    `camera_color_optical_frame` because no validated camera-to-work-area
    extrinsic exists for the physical camera — so this run is EVIDENCE OF
    PERCEPTION, not an input to planning, and the panel says so rather than
    leaving an operator to notice.
    """
    document = _physical_c5_document()
    if document is None:
        return {
            "available": False,
            "reason": ("no physical D435 result yet. Run "
                       "./scripts/physical_c5.sh --model cylinder5 --frames 5 "
                       "--roi 255,70,445,719, or replay a recorded capture "
                       "with --dataset."),
        }
    observation = document.get("observation") or {}
    pose = observation.get("pose") or {}
    task = observation.get("task") or {}
    device = document.get("device") or {}
    return {
        "available": True,
        "reason": "",
        # PROVENANCE FIRST, because everything below is only meaningful with it.
        "acquisition": "Intel RealSense D435",
        "acquisition_backend": document.get("acquisition_backend", "realsense"),
        "provenance": document.get("provenance", "measured"),
        "run_mode": document.get("run_mode", ""),
        "run_label": document.get("run_label", ""),
        "run_note": document.get("run_note", ""),
        "perception_method": observation.get("perception_method", ""),
        "object_model_id": observation.get("object_model_id",
                                           document.get("model_id", "")),
        "device": {k: device.get(k) for k in
                   ("name", "serial_number", "firmware_version",
                    "usb_type_descriptor")},
        "selected_profile": document.get("selected_profile", {}),
        "operator_roi_px": document.get("operator_roi_px"),
        "roi_note": document.get("roi_note", ""),
        "frame_id": observation.get("frame_id", ""),
        "pose_valid": pose.get("valid"),
        "workarea_pose_available": pose.get("workarea_pose_available"),
        # THE PHYSICAL BODY CENTRE, in the CAMERA frame. Not a work-area
        # coordinate: there is none for this camera, and showing the simulated
        # run's work-area centre beside a physical pose would read as though
        # this object had been located in the cell.
        "object_centre_mm": task.get("object_center_mm"),
        "model_frame_origin_mm": [pose.get("x_mm"), pose.get("y_mm"),
                                  pose.get("z_mm")],
        "tube_axis_line": task.get("tube_axis_line"),
        "measured_dof": pose.get("measured_dof", []),
        "segmentation": {k: (document.get("segmentation") or {}).get(k)
                         for k in ("mask_source", "mask_valid", "mask_pixels",
                                   "components", "roi_px",
                                   "mask_extent_long_mm",
                                   "mask_extent_across_mm",
                                   "mask_median_range_mm")},
        "repeatability": document.get("repeatability", {}),
        "plausibility": document.get("plausibility", {}),
        "images": sorted(k for k, name in PHYSICAL_C5_IMAGES.items()
                         if os.path.isfile(os.path.join(PHYSICAL_C5_DIR, name))),
        "completed_at": document.get("completed_at", ""),
        # SAID IN THE DATA, not only in the markup, so no consumer can render
        # this result as a planning input by omitting a label.
        "planning_available": False,
        "planning_blocked_reason": (
            "Physical 6-DoF perception validated in camera coordinates. "
            "Work-area calibration is required before planning or execution."),
        "accuracy_note": document.get("accuracy_note", ""),
    }


@app.get("/api/perception/physical/image/{kind}")
def api_perception_physical_image(kind: str):
    """The REAL frames the physical estimate was computed from.

    SERVED FROM THE ARTIFACT, not regenerated. These are the actual images the
    run produced; re-rendering them here would put a different picture on screen
    from the one the measurement came from.
    """
    from fastapi.responses import FileResponse                    # noqa: PLC0415
    name = PHYSICAL_C5_IMAGES.get(kind)
    if name is None:
        raise HTTPException(status_code=404, detail=f"unknown image {kind!r}")
    path = os.path.join(PHYSICAL_C5_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(
            status_code=404,
            detail=f"no {kind!r} image yet — run ./scripts/physical_c5.sh")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/perception/foundationpose/image/{kind}")
def api_foundationpose_image(kind: str):
    """Proxy the worker's diagnostic images. A PROXY, never a cache."""
    from fastapi.responses import Response                        # noqa: PLC0415
    if kind not in ("rgb", "depth", "mask", "overlay"):
        raise HTTPException(status_code=404, detail=f"unknown image {kind!r}")
    provider = foundationpose_provider()
    if provider is None:
        raise HTTPException(status_code=503,
                            detail="the FoundationPose provider is not available")
    image, error = provider.client.image(kind)
    if image is None:
        raise HTTPException(status_code=503, detail=error)
    return Response(image, media_type="image/jpeg")


@app.get("/api/perception/image/{kind}")
def api_perception_image(kind: str):
    """Proxy the detector's images so the browser needs no second origin.

    A PROXY, NOT A CACHE. The dashboard never stores or re-encodes a frame; it
    forwards the bytes the detector produced, so what the operator sees is what
    the detector saw.
    """
    from fastapi.responses import Response                        # noqa: PLC0415
    if kind not in ("annotated", "raw", "snapshot"):
        raise HTTPException(status_code=404, detail=f"unknown image {kind!r}")
    # KEYED ON THE CAPABILITY, not on the selected source. A live preview is
    # useful precisely while deciding whether to switch to the camera, and the
    # frames come from the service rather than from any run.
    usable, reason = camera_capability()
    if not usable:
        raise HTTPException(
            status_code=409,
            detail=("the physical camera is not available: "
                    + (reason or "no reason reported")))
    image, error = perception_client().image(kind)
    if image is None:
        raise HTTPException(status_code=503, detail=error)
    return Response(image, media_type="image/jpeg")


@app.post("/api/draft")
def api_draft(payload: Dict[str, Any]):
    """Record the operator's DRAFT scenario for the next run.

    Never touches the running scenario — that only changes on an explicit
    `reset` ("Generate & plan"). Recording it server-side is what lets the draft
    survive a page switch to Container Inventory or Diagnostics and back.
    """
    incoming = dict(payload or {})
    # THE OBJECT SOURCE IS NOT AN ORDINARY DRAFT FIELD. Every other setting is a
    # preference the next run will honour; this one names a capability that may
    # not exist, so it is validated here as well as on the command path. The
    # draft form posts the whole form on any change, and a stale `camera` in it
    # must not be able to re-select a camera that has since gone away.
    if "perception_method" in incoming:
        # SAME RULE AS THE OBJECT SOURCE. The draft form posts every field on
        # any change, so a stale `foundationpose_rgbd` in it must not re-select
        # a worker that has since gone away. Dropped, not raised: the operator
        # was editing something else.
        if str(incoming.get("perception_method") or "") not in (
                perception_method_state().available):
            incoming.pop("perception_method")
    if "object_source" in incoming:
        # DROPPED, NOT RAISED, and checked without an exception handler: the
        # operator was editing the preset, not the source, and failing their
        # edit over a field they did not touch would be baffling. The source
        # then keeps whatever is actually in force. Selecting a source
        # DELIBERATELY goes through `set_object_source`, which refuses loudly.
        if str(incoming.get("object_source") or "") not in (
                object_source_state().available):
            incoming.pop("object_source")
    with STATE.lock:
        for key, value in incoming.items():
            if key in STATE.settings:
                STATE.settings[key] = value
        STATE.settings_touched = True
        return {"ok": True, "settings": dict(STATE.settings)}


@app.get("/api/visualization")
def api_visualization():
    """How to WATCH the active execution backend — metadata only.

    Never carries a frame. Rendered video travels over its own transport
    (WebRTC for Isaac); this endpoint answers "is there a stream, where, and
    what state is it in" so the Simulator View can show an honest status
    instead of an empty player.
    """
    snap = _provider().snapshot()
    payload = snap.to_visualization()
    payload["execution_backend"] = snap.execution_backend
    payload["execution_backend_label"] = snap.execution_backend_label
    # WHICH ARM the stream is showing. Public-safe subset only — no asset URLs.
    payload["robot"] = snap.active_robot
    payload["robot_id"] = snap.active_robot_id
    # AFTER A ROBOT SWITCH THE STREAM IS A NEW ONE. Isaac restarted, so its
    # WebRTC listener restarted with it: the descriptor below is re-read from
    # the new simulator's READY, and a native client that was connected to the
    # previous process has to reconnect. Said explicitly, because a frozen last
    # frame looks exactly like a working stream.
    switch = (snap.isaac or {}).get("robot_switch") or {}
    payload["robot_switch"] = switch
    payload["simulator_generation"] = switch.get("host_generation") or 0
    payload["stream_reconnect_required"] = bool(
        switch.get("in_flight") or switch.get("failed"))
    payload["stream_note"] = (
        "The simulator restarted for a different robot — this is a NEW stream. "
        "Reconnect the native WebRTC client if it is still showing the previous "
        "session." if switch.get("host_generation", 0) > 1 else "")
    payload["current_item_id"] = snap.current_item_id
    payload["stage"] = snap.stage
    payload["isaac_state"] = (snap.isaac or {}).get("last_state")
    payload["simulator_version"] = (snap.isaac or {}).get("simulator_version")
    return payload


@app.get("/simulator", response_class=HTMLResponse)
def simulator_page():
    with open(os.path.join(HERE, "simulator.html"), encoding="utf-8") as fh:
        return fh.read()


@app.get("/api/whole_process")
def api_whole_process():
    """Cut-aware comparison + inventory + logistics, in every mode."""
    return _provider().snapshot().to_whole_process()


@app.get("/api/inventory")
def api_inventory():
    """FIWARE-backed container inventory view (brief §12)."""
    return _provider().snapshot().to_inventory()


@app.get("/api/logistics")
def api_logistics():
    """Simulated container-logistics facility map + tasks (brief §16)."""
    return _provider().snapshot().to_logistics()


@app.get("/api/diagnostics")
def api_diagnostics():
    """Read-only engineering diagnostics. Allowlisted, secret-free (see diagnostics.py)."""
    import diagnostics                                          # noqa: PLC0415
    snap = _provider().snapshot()
    with STATE.lock:
        mirror = STATE.ros_mirror
    return diagnostics.build(
        snap, SOURCE, mirror, latest_artifact("dds-fiware-latency", RESULTS_DIR),
        perception=_perception_payload())


@app.get("/api/inspector")
def api_inspector():
    """Latest of each message kind, for the diagnostics message inspector.

    Payloads are length-capped; no secrets pass through the workflow, and the
    fields shown are the same the audit trail already publishes.
    """
    snap = _provider().snapshot()
    events = snap.events
    def latest(pred):
        for e in events:
            if pred(e):
                return e
        return None
    cap = lambda o: (json.dumps(o)[:1200] if o is not None else None)
    return {
        "action_event": events[0] if events else None,
        "operator_command": latest(lambda e: e.get("actor") == "operator"),
        "approval": latest(lambda e: e.get("action") in ("approve_plan", "reject_plan")),
        "dynamic_event": latest(lambda e: str(e.get("action", "")).startswith("dynamic_event")),
        "anomaly_event": (snap.anomaly or {}).get("latest") if snap.anomaly else None,
        "strategy_comparison": snap.strategy_comparison,
        "baseline_plan": {"plan_id": (snap.baseline or {}).get("plan_id"),
                          "containers_required": (snap.baseline or {}).get("containers_required"),
                          "summary": cap((snap.baseline or {}).get("details"))}
                          if snap.baseline else None,
        "optimized_plan": {"plan_id": (snap.optimized or {}).get("plan_id"),
                           "containers_required": (snap.optimized or {}).get("containers_required")}
                           if snap.optimized else None,
    }


@app.get("/diagnostics", response_class=HTMLResponse)
def diagnostics_page():
    with open(os.path.join(HERE, "diagnostics.html"), encoding="utf-8") as fh:
        return fh.read()


@app.get("/inventory", response_class=HTMLResponse)
def inventory_page():
    with open(os.path.join(HERE, "inventory.html"), encoding="utf-8") as fh:
        return fh.read()


@app.get("/logistics", response_class=HTMLResponse)
def logistics_page():
    # The inventory page carries both the container inventory and the logistics
    # facility map; /logistics deep-links to the logistics section.
    with open(os.path.join(HERE, "inventory.html"), encoding="utf-8") as fh:
        return fh.read()


@app.get("/api/fiware")
def api_fiware():
    """Live NGSI-LD read-back — the proof that the audit path works."""
    if SOURCE == "sim":
        raise HTTPException(
            status_code=409,
            detail="sim mode has no FIWARE. Start with --source ros or fiware.")
    from wisepack_fiware.entities import ENTITY_IDS                # noqa: PLC0415
    out = {}
    for name, entity_id in ENTITY_IDS.items():
        entity = fiware_entity(entity_id)
        out[name] = {
            "entity_id": entity_id,
            "present": entity is not None,
            "attributes": {k: _attr(entity, k) for k in entity
                           if k not in ("id", "type", "@context")}
            if entity else {},
        }
    return {"broker": ORION, "health": fiware_health(), "entities": out}


# --------------------------------------------------------------------------- #
# Operator commands
# --------------------------------------------------------------------------- #


@app.post("/api/command")
async def api_command(payload: Dict[str, Any]):
    """Operator actions.

    In live (ros/fiware) mode these do NOT mutate Python state directly: they are
    published on the documented operator command path
    (/wisepack/operator/approval, /wisepack/operator/command), which reaches the
    orchestrator over DDS exactly as an NGSI-LD PATCH from an external HMI would.
    Mutating the engine in-process would demo a control path that does not exist.
    """
    command = str(payload.get("command", "")).strip()
    args = payload.get("args", {}) or {}

    # STALE-REVISION GUARD, enforced here and not only in the UI.
    #
    # The dashboard pins the plan_id a decision was taken against. If the plan
    # has been replaced since — a re-plan, an injected item, a strategy change —
    # the operator is answering a question that no longer exists, and approving
    # a superseded plan is exactly the authorisation mistake the gate exists to
    # prevent. A disabled button is a courtesy; this is the check.
    if command in ("approve", "reject") and args.get("plan_id"):
        snap = _provider().snapshot()
        current = snap.control_plan_id
        if current and args["plan_id"] != current:
            raise HTTPException(
                status_code=409,
                detail=(f"that decision was taken against plan "
                        f"{args['plan_id']}, but the current plan is {current}. "
                        "The plan changed while you were deciding — review the "
                        "new one and decide again."))
        # The orchestrator's command vocabulary does not take a plan_id; it is a
        # dashboard-side revision token only, so it is dropped before publishing.
        args = {k: v for k, v in args.items() if k != "plan_id"}

    if SOURCE != "sim":
        from ros_observer import publish_operator_command           # noqa: PLC0415
        ok, detail = publish_operator_command(command, args)
        return {"ok": ok, "command": command, "path": "ros2 -> dds -> orchestrator",
                "detail": detail}

    with STATE.lock:
        engine = STATE.engine
        if engine is None:
            raise HTTPException(status_code=409, detail="no run in progress")
        try:
            return _apply_command_locked(engine, command, args)
        except (ApprovalRequired, WorkflowError, WholeProcessError,
                InvalidTransition, ValueError) as exc:
            # 409, not 500. These are all "that command is not legal right now"
            # — an operator double-clicking Approve, pressing Resume on an
            # unapproved plan, approving a cut with none selected, or an illegal
            # container transition. A server error would tell them nothing and
            # would look like a crash in front of an audience.
            raise HTTPException(status_code=409, detail=str(exc)) from exc


def _apply_command_locked(engine: WorkflowEngine, command: str,
                          args: Dict[str, Any]) -> Dict[str, Any]:
    if command == "approve":
        engine.approve(operator=str(args.get("operator", "dashboard operator")))
        STATE.auto_step = True
        return {"ok": True, "stage": engine.stage.value}

    if command == "reject":
        engine.reject(reason=str(args.get("reason", "rejected from dashboard")))
        STATE.auto_step = False
        return {"ok": True, "stage": engine.stage.value,
                "replans": engine.stats.replans}

    if command == "alternative_strategy":
        strategy = Strategy(args.get("strategy", "retrievability"))
        engine.config.strategy = strategy
        engine.stats.operator_interventions += 1
        engine.generate_plans(strategy)
        engine.digital_twin_validate()
        engine.request_approval()
        STATE.auto_step = False
        return {"ok": True, "strategy": strategy.value,
                "containers": engine.optimized.containers_required}

    if command == "compare_strategies":
        # Sim mode may use the local engine directly (the brief permits this).
        # It mutates no plan and leaves approval untouched.
        req_rev = args.get("scenario_revision")
        if req_rev is not None and int(req_rev) != engine.scenario_revision:
            raise ValueError(f"scenario revision {req_rev} is stale "
                             f"(current {engine.scenario_revision})")
        comparison = engine.build_strategy_comparison(args.get("strategies"))
        return {"ok": True, "comparison_id": comparison["comparison_id"],
                "scenario_revision": comparison["scenario_revision"]}

    if command == "inject_item":
        engine.apply_dynamic_event(DynamicEvent(
            event_type=DynamicEventType.ITEM_INJECT,
            trigger=f"placement:{engine.cursor.index}",
            label=str(args.get("label", "Operator-injected component")),
            payload={"item": args.get("item", {
                "length_mm": 1100, "outer_diameter_mm": 200,
                "inner_diameter_mm": 170, "priority": 9, "dose_class": "ILW"})}))
        STATE.auto_step = False
        return {"ok": True, "replans": engine.stats.replans}

    if command == "container_unavailable":
        container_id = args.get("container_id") or (
            engine.selected.containers_used[0].container_id
            if engine.selected and engine.selected.containers_used else None)
        if not container_id:
            raise ValueError("no container to mark unavailable")
        engine.apply_dynamic_event(DynamicEvent(
            event_type=DynamicEventType.CONTAINER_UNAVAILABLE,
            trigger=f"placement:{engine.cursor.index}",
            label=f"{container_id} taken out of service",
            payload={"container_id": container_id}))
        STATE.auto_step = False
        return {"ok": True, "container_id": container_id}

    if command == "grasp_failure":
        engine.apply_dynamic_event(DynamicEvent(
            event_type=DynamicEventType.GRASP_FAILURE,
            trigger=f"placement:{engine.cursor.index}",
            label="Operator-injected grasp failure"))
        return {"ok": True}

    if command == "inject_anomaly":
        # SIMULATED Topic #2 anomaly, sim mode. Same deterministic reaction the
        # orchestrator applies in live modes.
        from wisepack_core.anomaly import AnomalyEvent                # noqa: PLC0415
        event = AnomalyEvent.simulate(
            args.get("anomaly_class", "camera_view_lost"),
            severity=args.get("severity"), confidence=args.get("confidence"))
        record = engine.apply_anomaly(event)
        if record.get("reaction") in ("pause", "hold"):
            STATE.auto_step = False
        return {"ok": True, "reaction": record.get("reaction"),
                "anomaly_class": record.get("anomaly_class")}

    if command == "acknowledge_anomaly":
        engine.acknowledge_anomaly(str(args.get("operator", "dashboard operator")))
        return {"ok": True}

    if command == "set_object_source":
        # THE DRAFT ONLY. The run on screen keeps its own source, its plan and
        # its approval until the operator actually starts the next acquisition.
        state = object_source_state()
        source = resolve_object_source(str(args.get("source", "")),
                                       state.available,
                                       fallback=state.selected)
        STATE.settings["object_source"] = source.value
        STATE.settings_touched = True
        return {"ok": True, "object_source": source.value,
                "label": source.selector_label,
                "action_label": source.action_label,
                "current": state.current,
                "changes_next_run": source.value != state.current}

    if command == "set_perception_method":
        # THE DRAFT ONLY, exactly like set_object_source. Changing the selector
        # must NOT mutate the run on screen: the batch it is planning from was
        # measured by whichever method actually produced it, and relabelling it
        # would rewrite that batch's provenance. A method becomes authoritative
        # when a new physical acquisition starts, and not before.
        state = perception_method_state()
        method = resolve_perception_method_selection(
            str(args.get("method", "")), state.available,
            fallback=state.selected)
        if method != str(args.get("method", "")).strip().lower() and args.get("method"):
            # ASKED FOR SOMETHING UNAVAILABLE. Refused loudly rather than
            # silently substituted — a selector that quietly keeps the old
            # method while showing the new one is how an operator ends up
            # believing a 6-DoF run is under way.
            raise ValueError(
                f"perception method {str(args.get('method'))!r} is not "
                "available: "
                + (state.unavailable_reasons.get(str(args.get("method")))
                   or "unknown method"))
        STATE.settings["perception_method"] = method
        STATE.settings_touched = True
        return {"ok": True, "perception_method": method,
                "label": PerceptionMethod(method).selector_label,
                "current": state.current,
                "changes_next_run": bool(state.current) and method != state.current}

    if command == "acquire_simulated_rgbd":
        # THE SAME PATH AS `detect_physical_objects`, with a different
        # ACQUISITION BACKEND. Not a second perception pipeline: the batch goes
        # through `apply_observation_batch`, `generate_plans`,
        # `digital_twin_validate` and `request_approval` exactly as a planar
        # camera batch does, so run id, revision, plan, twin and approval are
        # the live engine's and cannot drift from one another.
        #
        # The observation itself was produced offline by the Isaac -> FoundationPose
        # -> workarea chain (Stages A-C) and is read from its artifact. What the
        # dashboard must never do is RE-ESTIMATE it here under different inputs.
        document = _simulated_rgbd_observation()
        if document is None:
            raise ValueError(
                "no simulated RGB-D observation is available. Produce one "
                "first:  ./scripts/stage_c.sh")
        observation, meta = document
        STATE.settings.update({k: v for k, v in args.items()
                               if k in STATE.settings})
        STATE.settings["object_source"] = PerceptionSource.CAMERA.value
        STATE.settings["preset"] = meta.get("preset", "cad_cylinder5_single")
        # THE DRAFT IS SET TO WHAT THIS RUN ACTUALLY USED. Leaving the method
        # selector on the planar default made the Scenario panel contradict the
        # run beside it — the controls describe the NEXT run, and the sensible
        # next run is the one just performed.
        STATE.settings["perception_method"] = \
            PerceptionMethod.FOUNDATIONPOSE_RGBD.value
        STATE.settings_touched = True
        # A NEW RUN, for the same reason the camera path starts one: the objects
        # come from somewhere else entirely than the scenario on screen.
        # THE SOURCE IS NAMED, NOT INFERRED. `start_run` clamps the requested
        # source to what is available, and the planar service being absent must
        # not veto an acquisition that does not use it. This call states which
        # backend it is, which is the explicit act that a leftover artefact on
        # disk is not.
        STATE.engine = engine = start_run(
            STATE.settings, acquire=False,
            allow_sources=[PerceptionSource.SIM.value,
                           PerceptionSource.CAMERA.value])
        STATE.events.clear()
        for event in engine.log.events():
            STATE.events.append(event.to_dict())
        STATE.auto_step = False

        from wisepack_core.perception import BatchStatus              # noqa: PLC0415
        batch = ObservationBatch(
            batch_id=meta.get("batch_id", "simulated-rgbd-1"),
            source=PerceptionSource.CAMERA.value, status=BatchStatus.OK,
            observations=[observation], frame_id=observation.frame_id,
            captured_at=observation.captured_at,
            detector=observation.detector,
            perception_method=observation.perception_method,
            acquisition="simulated_rgbd",
            model_id=observation.object_model_id,
            calibration_status="not_applicable")
        engine.apply_observation_batch(batch)
        engine.generate_plans()
        engine.digital_twin_validate()
        engine.request_approval()
        with STATE.lock:
            STATE.simulated_rgbd = meta
        return {"ok": True, "batch_id": batch.batch_id,
                "model_id": observation.object_model_id,
                "scenario_revision": engine.scenario_revision,
                "run_id": engine.run_id,
                "containers": (engine.selected.containers_required
                               if engine.selected else None)}

    if command == "detect_physical_objects":
        # REFUSED, NOT SILENTLY SIMULATED. A control labelled "detect" that
        # returns generated objects is the exact deception §15 forbids — so an
        # unavailable camera raises here, with the capability's own reason, and
        # no preset scenario is produced in its place.
        usable, reason = camera_capability()
        if not usable:
            raise ValueError(
                "the physical camera is not available as an object source: "
                + (reason or "no reason reported"))
        # The acquisition button carries the whole scenario form, exactly as
        # "Generate & plan" does — the container spec and the packaging target
        # still come from it even when the objects come from a camera.
        STATE.settings.update({k: v for k, v in args.items()
                               if k in STATE.settings})
        awaiting = engine.config.perception_source.is_physical and not engine.detected
        if not engine.config.perception_source.is_physical or awaiting:
            # THE RUN ON SCREEN CAME FROM A PRESET (or is a camera run that has
            # not acquired yet), so the objects are about to come from somewhere
            # else entirely: start a NEW run rather than grafting a camera batch
            # onto a generated scenario. The old approval goes with the old
            # engine, which is the point.
            STATE.settings["object_source"] = PerceptionSource.CAMERA.value
            STATE.settings_touched = True
            STATE.engine = engine = start_run(STATE.settings, acquire=False)
            STATE.events.clear()
            for event in engine.log.events():
                STATE.events.append(event.to_dict())
        STATE.auto_step = False
        batch = perception_client().detect()
        try:
            engine.apply_observation_batch(batch)
        except PerceptionUnavailable as exc:
            # 409 with the detector's own reason. The observation batch is
            # already recorded on the engine, so the panel renders the failure.
            raise ValueError(str(exc)) from exc
        # A NEW BATCH IS A NEW BATCH REVISION, so the plan is rebuilt and the
        # approval gate is re-entered. `apply_observation_batch` already revoked
        # any outstanding approval; re-planning here is what makes the workflow
        # actually use the objects now on the table (§10).
        engine.generate_plans()
        engine.digital_twin_validate()
        engine.request_approval()
        return {"ok": True, "batch_id": batch.batch_id, "detected": batch.count,
                "calibration_status": batch.calibration_status,
                "scenario_revision": engine.scenario_revision,
                "containers": (engine.selected.containers_required
                               if engine.selected else None)}

    if command == "pause":
        STATE.auto_step = False
        return {"ok": True, "auto_step": False}

    if command == "resume":
        if engine.selected and engine.selected.approval_state.value == "approved":
            STATE.auto_step = True
            return {"ok": True, "auto_step": True}
        raise ApprovalRequired("cannot resume: the current plan is not approved")

    if command == "step":
        engine.step_execution()
        return {"ok": True, "progress_pct": round(engine.progress_pct, 1)}

    if command == "reset":
        STATE.settings.update({k: v for k, v in args.items()
                               if k in STATE.settings})
        STATE.events.clear()
        STATE.notice = ""
        # A RESET IS WHERE THE DRAFT SOURCE BECOMES REAL — and a camera run
        # starts empty, waiting for a detection, rather than pretending to
        # acquire a batch this call cannot wait for.
        state = object_source_state()
        source = resolve_object_source(
            args.get("object_source"), state.available,
            fallback=state.selected)
        STATE.settings["object_source"] = source.value
        STATE.engine = start_run(STATE.settings,
                                 acquire=not source.is_physical)
        STATE.auto_step = False
        for event in STATE.engine.log.events():
            STATE.events.append(event.to_dict())
        return {"ok": True, "scenario_id": STATE.engine.scenario.scenario_id,
                "object_source": source.value,
                "awaiting_detection": source.is_physical}

    if command == "write_artifacts":
        # THE REAL OUTCOME. This returned ok:True whatever happened, so a
        # permission failure was reported to the operator as a success with a
        # contradictory notice beside it.
        outcome = _write_artifacts_locked()
        if not outcome.get("ok"):
            raise HTTPException(
                status_code=500,
                detail=(f"could not write artefacts to {outcome.get('directory')}: "
                        f"{outcome.get('error') or outcome.get('reason')}. "
                        "Set WISEPACK_RESULTS_DIR to a writable directory."))
        return {"ok": True, "path": outcome["path"],
                "directory": outcome["directory"], "notice": STATE.notice}

    # -- cut-aware HITL controls (brief §6) --
    if command == "compare_cut_aware":
        cmp = engine.wp.generate_cut_alternatives()
        STATE.auto_step = False
        return {"ok": True, "recommend_cut": cmp.recommend_cut,
                "recommended": cmp.recommended_label,
                "containers_saved": cmp.no_cut.containers - cmp.recommended.containers}

    if command == "select_cut_alternative":
        engine.wp.select_alternative(str(args.get("label", "no_cut")))
        return {"ok": True, "selected": engine.wp.selected_cut_label}

    if command == "limit_cuts":
        engine.wp.limit_cuts(int(args.get("max_cuts", 1)))
        return {"ok": True}

    if command == "set_min_segment":
        engine.wp.set_minimum_segment_mm(int(args.get("mm", 400)))
        return {"ok": True}

    if command == "prefer_no_cut":
        engine.wp.set_prefer_no_cut(bool(args.get("prefer", True)))
        return {"ok": True, "selected": engine.wp.selected_cut_label}

    if command == "approve_cut":
        engine.wp.approve_cut(str(args.get("operator", "dashboard operator")))
        return {"ok": True, "cut_approval_state": engine.wp.cut_approval_state.value,
                "stage": engine.stage.value}

    if command == "reject_cut":
        engine.wp.reject_cut(str(args.get("reason", "operator preferred no cutting")))
        return {"ok": True, "selected": engine.wp.selected_cut_label}

    if command == "simulate_cut":
        engine.wp.simulate_cut(deviation_mm=int(args.get("deviation_mm", 0)))
        STATE.auto_step = False
        return {"ok": True, "stage": engine.stage.value,
                "containers": engine.selected.containers_required
                if engine.selected else None}

    if command == "simulate_cut_failure":
        engine.wp.simulate_cut_failure(str(args.get("reason", "blade jam (simulated)")))
        return {"ok": True, "stage": engine.stage.value}

    # -- inventory + logistics controls (brief §13) --
    if command == "init_inventory":
        engine.wp.initialise_simulated_inventory(int(args.get("count", 4)))
        return {"ok": True, "containers": len(engine.wp.inventory)}

    if command == "check_containers":
        pr = engine.wp.check_container_availability()
        engine.wp.run_logistics_to_quiescence()
        return {"ok": True, **{k: pr[k] for k in
                ("reservations_created", "additional_containers_required",
                 "inventory_shortage", "plan_status")}}

    if command in ("reserve_container", "release_container", "request_delivery",
                   "mark_container_unavailable", "restore_container",
                   "mark_container_full"):
        op = {"reserve_container": "reserve",
              "release_container": "release_reservation",
              "request_delivery": "request_delivery",
              "mark_container_unavailable": "mark_unavailable",
              "restore_container": "restore",
              "mark_container_full": "mark_full"}[command]
        cid = args.get("container_id")
        if not cid:
            raise ValueError(f"{command} requires container_id")
        engine.wp.inventory_operation(
            op, str(cid), actor=str(args.get("operator", "dashboard")),
            reason=str(args.get("reason", "")),
            holder=args.get("holder", engine.selected.plan_id
                            if engine.selected else "operator"))
        engine.wp.run_logistics_to_quiescence()
        return {"ok": True, "state":
                engine.wp.inventory.get(str(cid)).state.value}

    if command == "collect_full_containers":
        collected = engine.wp.collect_full_containers()
        return {"ok": True, "collected": collected}

    raise ValueError(f"unknown command {command!r}")


# --------------------------------------------------------------------------- #
# WebSocket (enhancement only — the page works on polling alone)
# --------------------------------------------------------------------------- #


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    try:
        while True:
            snap = _provider().snapshot()
            await sock.send_json({
                "type": "tick",
                "state": api_state(),
                "events": snap.to_events(12)["events"],
            })
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:                                   # noqa: BLE001
        # A failed socket must degrade to polling, never take the server down.
        pass


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as fh:
        return fh.read()


@app.get("/healthz")
def healthz():
    with STATE.lock:
        return {"ok": True, "source": SOURCE,
                "stage": STATE.engine.stage.value if STATE.engine else "idle"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["sim", "ros", "fiware"],
                        default=os.environ.get("WISEPACK_SOURCE", "sim"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("WISEPACK_DASH_PORT", "8080")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--preset", default=None)
    parser.add_argument("--seed", type=int, default=None)
    # THE PERCEPTION SOURCE IS ITS OWN FLAG. Not a `--source` value: `--source`
    # selects where the dashboard READS state from, and a camera is not a way of
    # reading state. Conflating them is what §-architecture forbids.
    parser.add_argument("--perception-source",
                        choices=[s.value for s in PerceptionSource],
                        default=None,
                        help=("where object observations come from. `sim` "
                              "(default, unchanged) or `camera` (measured from "
                              "a real camera frame by the configured perception "
                              "provider). INDEPENDENT of --source and of the "
                              "execution backend."))
    args = parser.parse_args()

    os.environ["WISEPACK_SOURCE"] = args.source
    SOURCE = args.source
    STATE.source = args.source
    if args.perception_source:
        # THE INITIAL SELECTION. It seeds the draft the dashboard opens on; the
        # operator changes the object source per run afterwards, in the same
        # session, without restarting anything.
        os.environ["WISEPACK_PERCEPTION_SOURCE"] = args.perception_source
        PERCEPTION_SOURCE = resolve_perception_source(args.perception_source)
        STATE.settings["object_source"] = PERCEPTION_SOURCE.value
    if args.preset:
        STATE.settings["preset"] = args.preset
    if args.seed is not None:
        STATE.settings["seed"] = args.seed

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
