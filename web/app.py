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
RESULTS_DIR = os.path.join(REPO, "results")

# wisepack_core is imported from the workspace source tree, so the dashboard
# runs with nothing installed and nothing built. This is the same import the
# ROS nodes do after a colcon build; only the path discovery differs.
for _pkg in ("wisepack_core", "wisepack_fiware", "wisepack_bringup"):
    _path = os.path.join(REPO, "wisepack_ws", "src", _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from wisepack_core.artifacts import (                              # noqa: E402
    latest_artifact, latest_latency_p50_ms, write_run_artifacts,
    write_validation_report,
)
from wisepack_core.domain import Strategy                          # noqa: E402
from wisepack_core.execution import physical_presets              # noqa: E402
from wisepack_core.perception import (                             # noqa: E402
    PerceptionConfigError, PerceptionSource, ProxyGeometry, WorkAreaFrame,
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
# above) and not an execution backend. Resolved once at import so an unknown
# value fails loudly at start-up instead of silently running the simulator while
# the header claims a camera. Unset == `sim`, so nothing existing changes.
try:
    PERCEPTION_SOURCE = resolve_perception_source()
    PERCEPTION_CONFIG_ERROR = ""
except PerceptionConfigError as _exc:
    PERCEPTION_SOURCE = PerceptionSource.SIM
    PERCEPTION_CONFIG_ERROR = str(_exc)

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
        #: The perception-service client, built on first use. None in `sim`
        #: perception mode, where there is no service to talk to.
        self.perception_client = None
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


def build_engine(settings: Dict[str, Any]) -> WorkflowEngine:
    preset = settings.get("preset", "mixed_pipes_dense")
    seed = int(settings.get("seed", 42))
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
        perception_source=PERCEPTION_SOURCE,
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
    if PERCEPTION_SOURCE.is_physical:
        # THE ONLY PLACE THE DASHBOARD LEARNS HOW OBSERVATIONS ARRIVE. The
        # engine stays transport-free; it calls a callable and gets a batch.
        from perception_client import make_observation_provider   # noqa: PLC0415
        engine.observation_provider = make_observation_provider(
            perception_client())
    engine.log.add_sink(STATE.sink)
    return engine


def perception_client():
    """The one shared client for the perception service. Built lazily."""
    from perception_client import PerceptionClient                # noqa: PLC0415
    with STATE.lock:
        if STATE.perception_client is None:
            STATE.perception_client = PerceptionClient()
        return STATE.perception_client


def start_run(settings: Dict[str, Any]) -> WorkflowEngine:
    """Plan a fresh run and stop at the approval gate. Never auto-executes.

    IN PHYSICAL PERCEPTION MODE THIS MAY LEGITIMATELY STOP EARLY. If the camera
    or the detector cannot deliver a batch there is nothing to plan, and §15
    forbids substituting simulated detections. The engine is returned in its
    failed-perception state — the run exists, the Physical Perception panel
    shows why it has no objects, and the operator retries after fixing the cause.
    """
    engine = build_engine(settings)
    engine.generate_or_load_scenario()
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


def _write_artifacts_locked() -> None:
    """Persist evidence at the end of a run. Caller holds the lock."""
    engine = STATE.engine
    if engine is None or engine.selected is None:
        return
    try:
        kpis = engine.kpis(latest_latency_p50_ms(RESULTS_DIR))
        artifacts = write_run_artifacts(
            engine.scenario, engine.baseline, engine.optimized, engine.selected,
            kpis, engine.log, RESULTS_DIR)
        write_validation_report(
            engine.scenario, engine.baseline, engine.optimized, engine.selected,
            kpis, engine.log, artifacts, RESULTS_DIR)
        STATE.notice = f"artefacts written: results/wisepack-run-{artifacts.stamp}.json"
    except Exception as exc:                            # noqa: BLE001
        STATE.notice = f"artefact write failed: {exc}"


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
        "perception": {
            "source": PERCEPTION_SOURCE.value,
            "label": PERCEPTION_SOURCE.label,
            "detail": PERCEPTION_SOURCE.detail,
            "physical": PERCEPTION_SOURCE.is_physical,
            "config_error": PERCEPTION_CONFIG_ERROR,
        },
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
    if PERCEPTION_SOURCE.is_physical:
        for node in topology["nodes"]:
            if node["id"] == "perception":
                node["label"] = "HARMONY camera detector"
                node["kind"] = "Faster R-CNN"
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


def _perception_payload() -> Dict[str, Any]:
    """Everything the Physical Perception panel renders, in every mode.

    ALWAYS ANSWERS, including in `sim` perception mode, where it says plainly
    that perception is simulated rather than returning an empty physical panel
    that could be read as a broken camera.
    """
    payload: Dict[str, Any] = {
        "perception_source": PERCEPTION_SOURCE.value,
        "perception_source_label": PERCEPTION_SOURCE.label,
        "perception_source_detail": PERCEPTION_SOURCE.detail,
        "physical": PERCEPTION_SOURCE.is_physical,
        "config_error": PERCEPTION_CONFIG_ERROR,
        # THE PROXY DISCLOSURE (§9). Unobtrusive, but always present in the
        # payload so the panel cannot render real detections without it.
        "proxy_note": (
            "Physical bottles are used as proxies for the cylindrical "
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
    if not PERCEPTION_SOURCE.is_physical:
        payload["health"] = {"source": PERCEPTION_SOURCE.value,
                             "service_reachable": None}
        payload["batch"] = None
        payload["scene_objects"] = []
        return payload

    client = perception_client()
    payload["health"] = client.health()
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


@app.get("/api/perception")
def api_perception():
    """Physical perception status, the current observation batch and its poses."""
    return _perception_payload()


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
    if not PERCEPTION_SOURCE.is_physical:
        raise HTTPException(
            status_code=409,
            detail=("perception source is `sim` — there is no camera. Start "
                    "with WISEPACK_PERCEPTION_SOURCE=harmony_camera."))
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
    with STATE.lock:
        for key, value in (payload or {}).items():
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

    if command == "detect_physical_objects":
        # REFUSED, NOT SILENTLY SIMULATED, when there is no physical source.
        # Running the simulator behind a button labelled "Detect physical
        # objects" is the exact deception §15 forbids.
        if not PERCEPTION_SOURCE.is_physical:
            raise ValueError(
                "perception source is `sim` — there is no camera to detect "
                "with. Restart with WISEPACK_PERCEPTION_SOURCE=harmony_camera.")
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
        STATE.engine = start_run(STATE.settings)
        STATE.auto_step = False
        for event in STATE.engine.log.events():
            STATE.events.append(event.to_dict())
        return {"ok": True, "scenario_id": STATE.engine.scenario.scenario_id}

    if command == "write_artifacts":
        _write_artifacts_locked()
        return {"ok": True, "notice": STATE.notice}

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
                              "(default, unchanged) or `harmony_camera` (a real "
                              "camera through the HARMONY detector). "
                              "INDEPENDENT of --source and of the execution "
                              "backend."))
    args = parser.parse_args()

    os.environ["WISEPACK_SOURCE"] = args.source
    SOURCE = args.source
    STATE.source = args.source
    if args.perception_source:
        os.environ["WISEPACK_PERCEPTION_SOURCE"] = args.perception_source
        PERCEPTION_SOURCE = resolve_perception_source(args.perception_source)
    if args.preset:
        STATE.settings["preset"] = args.preset
    if args.seed is not None:
        STATE.settings["seed"] = args.seed

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
