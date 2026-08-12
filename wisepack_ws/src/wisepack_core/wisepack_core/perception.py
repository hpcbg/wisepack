"""Perception sources: where WISEPACK's object observations come from.

WISEPACK now has THREE orthogonal axes, and collapsing any two of them is the
mistake this module exists to prevent:

    DATA SOURCE        where the DASHBOARD reads state from.
                       sim | ros | fiware.            (see web/snapshot.py)

    PERCEPTION SOURCE  where the OBJECT OBSERVATIONS come from.
                       sim | camera.                  (this module)

    EXECUTION BACKEND  what performs the pick-and-place the operator approved.
                       simulated | isaac.             (see execution.py)

A camera is NOT an execution backend. Every combination is legal and none is
implied by another: camera perception with simulated execution, camera
perception with Isaac execution, simulated perception with Isaac execution.
Selecting a real camera must therefore never touch the execution backend, and
`WISEPACK_EXECUTION_BACKEND` keeps working exactly as before.

WHAT FLOWS BETWEEN THEM
-----------------------
    WISEPACK PERCEPTION
          |
          +-- simulated provider
          |
          +-- physical camera
                  |
                  +-- perception provider   (perception/providers/*.py)
                  |     the ONLY detector-aware code in the system
                  v
    ObservationBatch of PhysicalObservation   <- DOMAIN-NEUTRAL, this module
                  |
                  +--> WasteItem batch --> packing / workflow / validation
                  +--> scene_objects() --> (future) Isaac scene synchronizer

WISEPACK OWNS PERCEPTION. A provider is an implementation detail behind this
boundary, and swapping one for another — a different RGB detector, YOLO/OBB,
RGB-D, 6-DoF pose estimation, segmentation — changes nothing above it. Nothing
downstream of `ObservationBatch` knows which provider ran, what it detects, or
who wrote it; a future Isaac synchronizer reads `scene_objects()` rather than
parsing any detector's JSON.

GEOMETRY IS CONFIGURED, NOT INFERRED
------------------------------------
A calibrated 2-D detector measures WHERE an object is, not how big it is. The
packing arithmetic needs a diameter and a length, and inventing them from a
bounding box would feed a made-up number into the one part of this repository
that is genuinely measured. So the proxy cylinder's dimensions are declared
(`WISEPACK_PHYSICAL_PROXY_DIAMETER_MM` / `_LENGTH_MM`) and every item built from
an observation is stamped `geometry_source="configured_proxy"`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .domain import (
    GEOMETRY_SOURCE_CAD_MESH,
    Axis, DomainError, GeometryType, PhysicalObservation, Vec3, WasteItem,
)

# --------------------------------------------------------------------------- #
# Perception source
# --------------------------------------------------------------------------- #


class PerceptionSource(str, Enum):
    """Where the object observations the planner packs actually came from."""

    #: The existing seeded perception simulator. There is no camera and no
    #: detector; ground truth is republished with a simulated confidence. THE
    #: DEFAULT, and its behaviour is unchanged by this module's existence.
    SIM = "sim"
    #: A REAL CAMERA, through whichever detector provider is configured. The
    #: source answers "where do observations come from", never "which neural
    #: network processed the image" — that is `PERCEPTION_DETECTOR_ENV`, so a
    #: second provider (YOLO/OBB, RGB-D pose, segmentation) can be added without
    #: inventing a second perception source or renaming this one.
    CAMERA = "camera"

    @property
    def is_physical(self) -> bool:
        return self is not PerceptionSource.SIM

    @property
    def label(self) -> str:
        """Dashboard badge text. Never claims more than the source is."""
        return {"sim": "SIMULATED PERCEPTION",
                "camera": "PHYSICAL CAMERA"}[self.value]

    @property
    def selector_label(self) -> str:
        """What the OBJECT SOURCE selector calls this, in the operator's terms.

        Deliberately different from `label`. `label` is the provenance badge on
        a run that already exists — "this batch was measured by a camera". This
        is the CHOICE offered before a run exists, and at that moment `sim` is
        not "simulated perception" in the abstract: it is "generate a scenario
        from the selected preset". Naming the choice after the internal enum
        made the selector read like an architecture switch rather than an
        operator's decision about where the next batch of objects comes from.
        """
        return {"sim": "Preset scenario",
                "camera": "Physical camera"}[self.value]

    @property
    def selector_detail(self) -> str:
        """One line of help under the selector."""
        return {
            "sim": ("objects come from the selected preset's deterministic "
                    "generator — no camera and no detector are involved"),
            "camera": ("objects are detected from a real camera frame and "
                       "measured on the calibrated plane"),
        }[self.value]

    @property
    def action_label(self) -> str:
        """What the button that ACQUIRES a batch from this source is called.

        One button, one meaning. Showing "Generate & plan" while the camera is
        selected would either be a dead control or a silent fall back to the
        simulator; both are worse than relabelling the control the operator is
        about to press.
        """
        return {"sim": "Generate & plan", "camera": "Detect & plan"}[self.value]

    @property
    def provenance(self) -> str:
        """How a run acquired its objects, for the audit trail and FIWARE.

        Stamped from the run's OWN source — never inferred from the execution
        backend or from where the dashboard reads its state.
        """
        return {"sim": "preset/generated",
                "camera": "camera/measured"}[self.value]

    @property
    def detail(self) -> str:
        return {
            "sim": ("object observations are the generated ground truth with a "
                    "seeded confidence — no camera, no image, no detector"),
            "camera": (
                "object observations are measured from a real camera frame by "
                "the configured detector provider and positioned on a "
                "calibrated plane"),
        }[self.value]


#: The INITIAL object source — the dashboard's selection when the page first
#: loads, and the source of the run every launcher starts automatically.
#:
#: IT IS NOT A MODE LOCK, and it used to be read as one. The operator changes
#: the object source at runtime, per run, from the dashboard; this only decides
#: where a session STARTS. Unset == `sim`, so every existing invocation keeps
#: its exact behaviour: `./run_wisepack_dashboard.sh` opens on the preset
#: workflow, unchanged.
PERCEPTION_SOURCE_ENV = "WISEPACK_PERCEPTION_SOURCE"

#: Ask the launcher to start the perception service WITHOUT making the camera
#: the initial selection. The camera then appears as an available object source
#: the operator can switch to, and preset runs are unaffected until they do.
#:
#: The distinction that matters: `WISEPACK_PERCEPTION_SOURCE=camera` says "start
#: this session on the camera", this says "have the camera ready". Neither
#: removes the other source.
PERCEPTION_ENABLE_ENV = "WISEPACK_PERCEPTION_ENABLE"


def camera_capability_requested(env: Optional[Dict[str, str]] = None) -> bool:
    """Should this host have a perception service running at all?

    Answered from configuration, not from whether one happens to be up: the
    launcher uses it to decide whether to START the service. Whether the camera
    is actually USABLE is a separate, live question — see
    `PerceptionClient.capability()` — because an operator may start the service
    by hand at any time, and the dashboard must notice without a restart.
    """
    env = os.environ if env is None else env
    enabled = str(env.get(PERCEPTION_ENABLE_ENV, "") or "").strip().lower()
    if enabled in ("1", "true", "yes", "on"):
        return True
    return resolve_perception_source(
        env.get(PERCEPTION_SOURCE_ENV, "")).is_physical

#: WHICH PROVIDER processes the camera image. A SEPARATE axis from the source:
#: `camera` says observations are measured from a real frame, this says how. One
#: provider exists today; the variable exists so the second one is a
#: configuration change rather than a rename of everything above it.
PERCEPTION_DETECTOR_ENV = "WISEPACK_PERCEPTION_DETECTOR"

#: The provider used when none is named. A Faster R-CNN detector that locates
#: bottles on an ArUco-calibrated plane; its implementation is adapted from the
#: HARMONY project and lives in `perception/providers/`, inside this repository.
#: Bottles are PHYSICAL PROXIES for cylindrical workpieces — that is a property
#: of the objects on the table, not of this architecture.
DEFAULT_DETECTOR = "fasterrcnn_bottle"

#: Providers this build can run. Names describe the METHOD, never its origin, so
#: nothing downstream has to be renamed when a provider is added.
#:
#: LEGACY. `fasterrcnn_bottle` is a PROVIDER MODULE name — it names an
#: implementation (`perception/providers/fasterrcnn_bottle.py`), not a
#: capability. See `PerceptionMethod` below, which is the abstraction that
#: replaced it, and `resolve_perception_method()`, which accepts this value as a
#: compatibility alias.
KNOWN_DETECTORS = (DEFAULT_DETECTOR,)


# --------------------------------------------------------------------------- #
# Perception METHOD — how a physical frame becomes observations
# --------------------------------------------------------------------------- #
#
# WHY THIS REPLACED "DETECTOR". `WISEPACK_PERCEPTION_DETECTOR` was named when
# there was one provider and it was a detector: a network that finds bounding
# boxes, with the geometry supplied by a separate ArUco homography.
#
# FoundationPose is not that. It is a complete pose-estimation pipeline — mesh
# in, RGB-D in, 6-DoF pose out — and calling it a "detector" would describe the
# smallest part of what it does while implying it produces the same kind of
# answer as the planar provider. It does not: one measures x/y/yaw on a
# calibrated plane, the other measures a full rigid transform in the camera
# frame.
#
# The METHOD is therefore the axis, and it is a THIRD axis, independent of the
# two that already exist:
#
#     dashboard data source      sim | ros | fiware      (where the UI reads)
#     object source              preset | camera         (where objects come from)
#     perception method          planar | rgbd 6-DoF     (how a frame is read)
#
# Only the last is new. Selecting a method never implies a camera is present,
# and never implies anything about the execution backend.


class PerceptionMethod(str, Enum):
    """How a physical camera frame becomes object observations."""

    #: Faster R-CNN detection plus an ArUco homography onto the calibrated
    #: plane. THE DEFAULT, and deliberately so: it is the validated path, it
    #: needs no CAD model, no depth sensor and no GPU, and every physical run
    #: WISEPACK has ever done used it.
    PLANAR_FASTERRCNN = "planar_fasterrcnn"
    #: FoundationPose: model-based 6-DoF pose from RGB-D against a known mesh.
    #: Never the default — it needs a depth camera, a GPU, licensed weights and
    #: a CAD model for the object in view, and any of those being absent must be
    #: visible rather than silently worked around.
    FOUNDATIONPOSE_RGBD = "foundationpose_rgbd"
    #: FoundationPose again, and deliberately the SAME estimator — what differs
    #: is the geometry it is given. Instead of the object's CAD mesh it receives
    #: a representation LEARNED from reference views of the object, so no CAD
    #: model reaches the estimator at all.
    #:
    #: A METHOD, NOT A SOURCE OR A BACKEND. It reads the same RGB-D frames from
    #: the same two devices and produces the same ObservationBatch; nothing
    #: downstream branches on it. Making it an object source would have implied
    #: the objects come from somewhere else, which they do not.
    #:
    #: NOT CAD-FREE PACKING. It removes CAD from POSE ESTIMATION only. WISEPACK
    #: continues to pack against exact engineering geometry where it exists —
    #: see `wisepack_core.representation`, where the reconstruction records that
    #: it is not authoritative for packing.
    FOUNDATIONPOSE_RGBD_MODEL_FREE = "foundationpose_rgbd_model_free"

    @property
    def provider_module(self) -> str:
        """The module under `perception/providers/` that implements this.

        BOTH FOUNDATIONPOSE METHODS SHARE ONE PROVIDER. They differ in which
        geometry the estimator is handed, not in how a frame is read, and a
        second module would be the same code twice with one substitution.
        """
        return {"planar_fasterrcnn": "fasterrcnn_bottle",
                "foundationpose_rgbd": "foundationpose_rgbd",
                "foundationpose_rgbd_model_free": "foundationpose_rgbd",
                }[self.value]

    @property
    def selector_label(self) -> str:
        return {"planar_fasterrcnn": "Planar RGB — Faster R-CNN",
                "foundationpose_rgbd": "RGB-D 6-DoF — FoundationPose (CAD)",
                "foundationpose_rgbd_model_free":
                    "RGB-D 6-DoF — FoundationPose (model-free)",
                }[self.value]

    @property
    def selector_detail(self) -> str:
        return {
            "planar_fasterrcnn": (
                # NO CALIBRATION TECHNOLOGY NAMED. Which marker system defines
                # the plane is the provider's business; the domain describes
                # what the method MEASURES, so replacing the markers does not
                # mean editing operator-facing text in the core.
                "objects are detected in a colour frame and measured on the "
                "calibrated plane: x, y and yaw"),
            "foundationpose_rgbd": (
                "a full 6-DoF pose is estimated from RGB-D against the "
                "object's CAD model, in the camera frame"),
            "foundationpose_rgbd_model_free": (
                # SAYS WHAT IS AND IS NOT SUPPLIED, because that is the whole
                # difference between the two, and "model-free" alone is easy to
                # read as "needs nothing" or as "CAD-free packing".
                "a full 6-DoF pose is estimated from RGB-D against a learned "
                "representation built from reference views; no CAD mesh is "
                "supplied to the estimator"),
        }[self.value]

    @property
    def measures(self) -> Tuple[str, ...]:
        """The degrees of freedom this method can actually produce.

        Carried so the dashboard never has to infer it from the method name,
        and so a planar observation is never rendered as though it had
        measured a full orientation.
        """
        return {"planar_fasterrcnn": ("x", "y", "yaw"),
                "foundationpose_rgbd": ("x", "y", "z", "orientation"),
                "foundationpose_rgbd_model_free":
                    ("x", "y", "z", "orientation"),
                }[self.value]

    @property
    def is_foundationpose(self) -> bool:
        """Both FoundationPose methods, whichever geometry they are given."""
        return self in (PerceptionMethod.FOUNDATIONPOSE_RGBD,
                        PerceptionMethod.FOUNDATIONPOSE_RGBD_MODEL_FREE)

    @property
    def requires_depth(self) -> bool:
        return self.is_foundationpose

    @property
    def requires_object_model(self) -> bool:
        """Model-BASED means exactly that: no mesh, no estimate.

        FALSE FOR MODEL-FREE, and this is the property the whole separation
        turns on: it is what tells every caller not to look up a CAD mesh. The
        estimator is given a learned representation instead, and handing it CAD
        would make the selected method a lie.
        """
        return self is PerceptionMethod.FOUNDATIONPOSE_RGBD

    @property
    def requires_representation(self) -> bool:
        """Needs a representation LEARNED from reference views, prepared offline.

        Its absence is a refusal, never a fall back to CAD — see
        `wisepack_core.representation.RepresentationRegistry.require`.
        """
        return self is PerceptionMethod.FOUNDATIONPOSE_RGBD_MODEL_FREE

    @property
    def estimator_geometry(self) -> str:
        """WHAT THE ESTIMATOR IS GIVEN, in one word, for provenance and the UI.

        Carried here so no panel has to infer "is this the CAD one?" from a
        method name, and so a run's record states which geometry produced it.
        """
        return {"planar_fasterrcnn": "",
                "foundationpose_rgbd": "cad",
                "foundationpose_rgbd_model_free": "learned_representation",
                }[self.value]


#: The public setting. Supersedes `WISEPACK_PERCEPTION_DETECTOR`, which is still
#: read when this is unset so an existing deployment keeps working.
PERCEPTION_METHOD_ENV = "WISEPACK_PERCEPTION_METHOD"

DEFAULT_PERCEPTION_METHOD = PerceptionMethod.PLANAR_FASTERRCNN.value

KNOWN_PERCEPTION_METHODS = tuple(m.value for m in PerceptionMethod)

#: Values accepted for compatibility, mapped to the method they mean. The old
#: variable held a PROVIDER MODULE name; both spellings resolve to one method so
#: an existing `WISEPACK_PERCEPTION_DETECTOR=fasterrcnn_bottle` is not a
#: configuration error after this change.
PERCEPTION_METHOD_ALIASES = {
    "fasterrcnn_bottle": PerceptionMethod.PLANAR_FASTERRCNN.value,
    "planar": PerceptionMethod.PLANAR_FASTERRCNN.value,
    # BARE `foundationpose` STAYS THE CAD METHOD. It named the only
    # FoundationPose method that existed when the alias was written, and
    # repointing it at model-free would silently change what an existing
    # deployment's configuration means.
    "foundationpose": PerceptionMethod.FOUNDATIONPOSE_RGBD.value,
    "foundationpose_cad": PerceptionMethod.FOUNDATIONPOSE_RGBD.value,
    "foundationpose_model_free":
        PerceptionMethod.FOUNDATIONPOSE_RGBD_MODEL_FREE.value,
    "model_free": PerceptionMethod.FOUNDATIONPOSE_RGBD_MODEL_FREE.value,
}


def resolve_perception_method(value: Optional[str] = None,
                              env: Optional[Dict[str, str]] = None) -> str:
    """Resolve the perception method: argument, then env, then the planar default.

    AN UNKNOWN VALUE IS AN ERROR, never a silent fall back — the same rule the
    perception source follows, and for the same reason: quietly running the
    planar detector when someone asked for 6-DoF produces measurements that are
    correct-looking and mean something else entirely.
    """
    environment = os.environ if env is None else env
    raw = value
    if raw is None:
        raw = environment.get(PERCEPTION_METHOD_ENV, "")
        if not str(raw or "").strip():
            # LEGACY FALLBACK, read only when the current variable is unset.
            raw = environment.get(PERCEPTION_DETECTOR_ENV, "")
    raw = str(raw or "").strip().lower()
    if not raw:
        return DEFAULT_PERCEPTION_METHOD
    raw = PERCEPTION_METHOD_ALIASES.get(raw, raw)
    if raw not in KNOWN_PERCEPTION_METHODS:
        raise PerceptionConfigError(
            f"unknown perception method {raw!r}; known: "
            + ", ".join(KNOWN_PERCEPTION_METHODS))
    return raw


def resolve_detector(value: Optional[str] = None) -> str:
    """Resolve the camera provider: explicit argument, then env, then default.

    An unknown provider is an error for the same reason an unknown perception
    source is: quietly running a different detector than the one that was asked
    for produces measurements nobody can interpret.
    """
    raw = value if value is not None else os.environ.get(PERCEPTION_DETECTOR_ENV, "")
    raw = str(raw or "").strip().lower()
    if not raw:
        return DEFAULT_DETECTOR
    if raw not in KNOWN_DETECTORS:
        raise PerceptionConfigError(
            f"unknown perception detector {raw!r}; known: "
            + ", ".join(KNOWN_DETECTORS))
    return raw


class PerceptionConfigError(ValueError):
    """Raised when the perception configuration cannot be honoured."""


def resolve_perception_source(value: Optional[str] = None) -> PerceptionSource:
    """Resolve the perception source: explicit argument, then env, then `sim`.

    An UNRECOGNISED value is an error, never a silent fall back to `sim`. A
    typo in `WISEPACK_PERCEPTION_SOURCE=kamera` that quietly ran the
    simulator while the operator believed a camera was live is exactly the
    failure §15 forbids.
    """
    raw = value if value is not None else os.environ.get(PERCEPTION_SOURCE_ENV, "")
    raw = str(raw or "").strip().lower()
    if not raw:
        return PerceptionSource.SIM
    try:
        return PerceptionSource(raw)
    except ValueError as exc:
        known = ", ".join(s.value for s in PerceptionSource)
        raise PerceptionConfigError(
            f"unknown perception source {raw!r}; known: {known}") from exc


# --------------------------------------------------------------------------- #
# Proxy geometry
# --------------------------------------------------------------------------- #

#: Defaults describing an ordinary 0.5 l PET bottle standing in for a cylindrical
#: workpiece: ~65 mm across, ~215 mm tall. They are a DOCUMENTED DEFAULT, not a
#: measurement of the bottles on any particular table — override them for the
#: objects actually in front of the camera.
DEFAULT_PROXY_DIAMETER_MM = 65
DEFAULT_PROXY_LENGTH_MM = 215


@dataclass(frozen=True)
class ProxyGeometry:
    """The KNOWN dimensions of the physical proxy cylinders, in millimetres.

    Declared rather than detected, and stamped onto every item built from an
    observation. ``wall_thickness_mm`` exists only so the item has a plausible
    material volume for the mass-balance reporting; it never changes the
    occupied bounding box that decides container demand.
    """

    diameter_mm: int = DEFAULT_PROXY_DIAMETER_MM
    length_mm: int = DEFAULT_PROXY_LENGTH_MM
    wall_thickness_mm: int = 2
    material: str = "carbon_steel"
    segregation_group: str = "A"

    def __post_init__(self) -> None:
        if self.diameter_mm <= 0 or self.length_mm <= 0:
            raise PerceptionConfigError(
                f"proxy geometry must be positive, got diameter "
                f"{self.diameter_mm} mm, length {self.length_mm} mm")
        if not 0 < self.wall_thickness_mm * 2 < self.diameter_mm:
            raise PerceptionConfigError(
                f"proxy wall thickness {self.wall_thickness_mm} mm does not fit "
                f"inside diameter {self.diameter_mm} mm")

    @property
    def inner_diameter_mm(self) -> int:
        return self.diameter_mm - 2 * self.wall_thickness_mm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diameter_mm": self.diameter_mm,
            "length_mm": self.length_mm,
            "inner_diameter_mm": self.inner_diameter_mm,
            "wall_thickness_mm": self.wall_thickness_mm,
            "material": self.material,
            "segregation_group": self.segregation_group,
            "source": "configured_proxy",
        }

    @staticmethod
    def from_env(env: Optional[Dict[str, str]] = None) -> "ProxyGeometry":
        env = os.environ if env is None else env

        def number(name: str, default: int) -> int:
            raw = str(env.get(name, "") or "").strip()
            if not raw:
                return default
            try:
                return int(round(float(raw)))
            except ValueError as exc:
                raise PerceptionConfigError(
                    f"{name} must be a number of millimetres, got {raw!r}") from exc

        return ProxyGeometry(
            diameter_mm=number("WISEPACK_PHYSICAL_PROXY_DIAMETER_MM",
                               DEFAULT_PROXY_DIAMETER_MM),
            length_mm=number("WISEPACK_PHYSICAL_PROXY_LENGTH_MM",
                             DEFAULT_PROXY_LENGTH_MM),
            wall_thickness_mm=number("WISEPACK_PHYSICAL_PROXY_WALL_MM", 2),
            material=str(env.get("WISEPACK_PHYSICAL_PROXY_MATERIAL",
                                 "carbon_steel") or "carbon_steel"),
            segregation_group=str(env.get("WISEPACK_PHYSICAL_PROXY_GROUP", "A")
                                  or "A"),
        )


# --------------------------------------------------------------------------- #
# Work-area frame
# --------------------------------------------------------------------------- #

#: The name of the physical coordinate frame a detection is expressed in. It is
#: carried on every observation and every item, so a consumer never has to guess
#: which origin three numbers belong to.
WORKAREA_FRAME_ID = "wisepack_workarea"


@dataclass(frozen=True)
class WorkAreaFrame:
    """How the detector's calibrated plane maps onto the WISEPACK work area.

    The ArUco calibration in `perception/calibration.py` defines a plane whose
    corner markers sit at configured millimetre coordinates — with the default
    A4 sheet, a 130 x 130 mm square with the origin at marker 11. That square is
    small for a realistic work area, so the mapping is IDENTITY by default (the
    measured plane IS the work-area frame, with no transform of WISEPACK's own)
    and the extent is CONFIGURABLE, so a larger printed board needs a
    configuration change rather than a code change.

    ``width_mm``/``depth_mm`` are the declared extent of the calibrated plane.
    They are used only to report whether an observation landed inside it —
    never to clamp or rescale a measured coordinate, because silently moving a
    measurement is worse than reporting that it fell outside.
    """

    frame_id: str = WORKAREA_FRAME_ID
    width_mm: int = 130
    depth_mm: int = 130
    origin_x_mm: float = 0.0
    origin_y_mm: float = 0.0

    def contains(self, x_mm: float, y_mm: float, tolerance_mm: float = 0.0) -> bool:
        return (self.origin_x_mm - tolerance_mm <= x_mm
                <= self.origin_x_mm + self.width_mm + tolerance_mm
                and self.origin_y_mm - tolerance_mm <= y_mm
                <= self.origin_y_mm + self.depth_mm + tolerance_mm)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "width_mm": self.width_mm,
            "depth_mm": self.depth_mm,
            "origin_x_mm": self.origin_x_mm,
            "origin_y_mm": self.origin_y_mm,
        }

    @staticmethod
    def from_env(env: Optional[Dict[str, str]] = None) -> "WorkAreaFrame":
        env = os.environ if env is None else env

        def number(name: str, default: float) -> float:
            raw = str(env.get(name, "") or "").strip()
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError as exc:
                raise PerceptionConfigError(
                    f"{name} must be a number of millimetres, got {raw!r}") from exc

        return WorkAreaFrame(
            frame_id=str(env.get("WISEPACK_PHYSICAL_FRAME_ID", WORKAREA_FRAME_ID)
                         or WORKAREA_FRAME_ID),
            width_mm=int(round(number("WISEPACK_PHYSICAL_WORKAREA_WIDTH_MM", 130))),
            depth_mm=int(round(number("WISEPACK_PHYSICAL_WORKAREA_DEPTH_MM", 130))),
            origin_x_mm=number("WISEPACK_PHYSICAL_WORKAREA_ORIGIN_X_MM", 0.0),
            origin_y_mm=number("WISEPACK_PHYSICAL_WORKAREA_ORIGIN_Y_MM", 0.0),
        )


# --------------------------------------------------------------------------- #
# Observation batch
# --------------------------------------------------------------------------- #


class BatchStatus(str, Enum):
    """The outcome of one detection attempt. `EMPTY` is not `ERROR`.

    A scan that ran correctly and found nothing is a valid, useful result: the
    table is empty. A scan that failed is a different fact and must never render
    the same way, because "0 objects" would read as a successful measurement.
    """

    OK = "ok"
    EMPTY = "empty"
    ERROR = "error"


@dataclass
class ObservationBatch:
    """The complete result of ONE detection request. Replaces its predecessor.

    A batch is atomic and total: it is the current physical observation, not an
    increment on the previous one. §6 requires repeated detection to REPLACE the
    observation rather than accumulate stale duplicates, and modelling the result
    as a whole batch is what makes that structural instead of a rule someone has
    to remember. `batch_id` increases monotonically per process so a consumer can
    tell a re-detection from a re-publication of the same one.
    """

    batch_id: str
    source: str = PerceptionSource.SIM.value
    status: BatchStatus = BatchStatus.OK
    observations: List[PhysicalObservation] = field(default_factory=list)
    frame_id: str = WORKAREA_FRAME_ID
    #: WHEN THE CAMERA FRAME THIS BATCH DESCRIBES WAS ACQUIRED — not when the
    #: detection was asked for. The two differ by the model load (~30 s on a
    #: cold start) plus the frame wait, and every consumer that matters treats
    #: this as a measurement time: staleness is computed from it, and the future
    #: Isaac synchronizer will place objects as they were at this instant.
    #:
    #: EMPTY WHEN NO FRAME WAS ACQUIRED. A batch that failed before the grab has
    #: no capture time, and inventing one would be asserting a measurement that
    #: never happened. `observation_age_s` returns None for an unstamped batch,
    #: so such a batch is never reported stale either — it is reported FAILED,
    #: which is the accurate thing to say about it.
    captured_at: str = ""
    #: When the detection was REQUESTED. Diagnostics only: it is what makes a
    #: slow cold start legible ("requested at T, frame at T+31 s") and it is the
    #: only timestamp a batch that never reached the camera can carry.
    requested_at: str = ""
    detector: str = ""
    #: WHICH METHOD MEASURED THIS BATCH. Additive: an older batch document has
    #: no such key and `from_dict` leaves it empty rather than assuming planar,
    #: because assuming is how a 6-DoF batch would come back labelled as a
    #: planar one after a round trip.
    perception_method: str = ""
    model_id: str = ""
    #: The 6-DoF fields below are EMPTY for a planar batch and must stay that
    #: way. A planar observation has no depth camera behind it and no CAD model;
    #: filling these in with defaults would make the two methods indistinguishable
    #: downstream, which is the one thing this whole boundary exists to prevent.
    #:
    #: Whether the batch came from a live acquisition or from the saved
    #: reference dataset. "reference" is never presented as a measurement of the
    #: physical work area — see the provider.
    acquisition: str = ""
    calibration_status: str = "unknown"
    calibration_revision: str = ""
    error: str = ""
    #: The detector's own status document, verbatim, for debugging. Never parsed
    #: by anything downstream — it is evidence, not an interface.
    detector_status: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = BatchStatus(self.status)
        if self.status is BatchStatus.ERROR and not self.error:
            raise DomainError("an error batch must carry a reason")

    @property
    def count(self) -> int:
        return len(self.observations)

    @property
    def ok(self) -> bool:
        return self.status is not BatchStatus.ERROR

    @property
    def calibration_valid(self) -> bool:
        return self.calibration_status == "valid"

    @property
    def mean_confidence(self) -> Optional[float]:
        """Mean detector confidence, or None. NOT a detection rate — see kpi.py."""
        values = [o.confidence for o in self.observations if o.confidence is not None]
        return round(sum(values) / len(values), 4) if values else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "source": self.source,
            "status": self.status.value,
            "count": self.count,
            "frame_id": self.frame_id,
            "captured_at": self.captured_at,
            "requested_at": self.requested_at,
            "detector": self.detector,
            "perception_method": self.perception_method,
            "acquisition": self.acquisition,
            "model_id": self.model_id,
            "calibration_status": self.calibration_status,
            "calibration_revision": self.calibration_revision,
            "error": self.error,
            # Mean confidence is published as `mean_confidence`, deliberately NOT
            # as anything containing the words "rate" or "accuracy". See §12.
            "mean_confidence": self.mean_confidence,
            "observations": [o.to_dict() for o in self.observations],
            "detector_status": dict(self.detector_status),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ObservationBatch":
        return ObservationBatch(
            batch_id=str(d.get("batch_id", "batch-0")),
            source=str(d.get("source", PerceptionSource.SIM.value)),
            status=BatchStatus(d.get("status", "ok")),
            observations=[PhysicalObservation.from_dict(o)
                          for o in d.get("observations", [])],
            frame_id=str(d.get("frame_id", WORKAREA_FRAME_ID)),
            captured_at=str(d.get("captured_at", "")),
            requested_at=str(d.get("requested_at", "")),
            detector=str(d.get("detector", "")),
            perception_method=str(d.get("perception_method", "")),
            acquisition=str(d.get("acquisition", "")),
            model_id=str(d.get("model_id", "")),
            calibration_status=str(d.get("calibration_status", "unknown")),
            calibration_revision=str(d.get("calibration_revision", "")),
            error=str(d.get("error", "")),
            detector_status=dict(d.get("detector_status", {})),
        )

    @staticmethod
    def failed(batch_id: str, source: str, error: str,
               **kw: Any) -> "ObservationBatch":
        """A batch that records a FAILURE. Never an empty successful batch.

        This is the constructor §15 exists for: camera absent, model missing,
        inference timeout, perception service unreachable, malformed detector
        output. The
        dashboard renders it as a failure and the workflow refuses to plan from
        it, rather than treating "no objects" as a measurement.
        """
        return ObservationBatch(batch_id=batch_id, source=source,
                                status=BatchStatus.ERROR,
                                error=error or "unspecified perception failure",
                                **kw)

    # -- conversion to the packing domain ---------------------------------- #

    def to_waste_items(self, geometry: Optional[ProxyGeometry] = None,
                       id_prefix: str = "item",
                       permitted_axes: Sequence[str] = ("x", "y")) -> List[WasteItem]:
        """Convert this batch into the generic items the planner packs.

        THE PACKING LAYER NEVER SEES A DETECTOR. It receives ordinary
        ``WasteItem``s whose dimensions are the configured proxy geometry and
        whose measured pose, confidence and detector provenance ride along in
        ``WasteItem.observation``.

        ``permitted_axes`` defaults to horizontal only: the observed objects lie
        on a table and are grasped from above, and standing a cylinder on its end
        needs a regrasp no configured backend performs. This mirrors the reason
        the Isaac smoke preset drops "z" (see generator.py).

        Item ids are assigned from a counter over the batch order, so the same
        batch always produces the same ids and a re-detection produces a fresh,
        complete set rather than merging into the previous one.
        """
        geometry = geometry or ProxyGeometry()
        density = 7850.0
        items: List[WasteItem] = []
        for index, obs in enumerate(self.observations, start=1):
            # A MODEL-BASED OBSERVATION ALREADY KNOWS ITS OBJECT, and knows it
            # better than the configured proxy does: it was matched against a
            # named CAD model whose nominal dimensions came from the engineering
            # table. Overwriting those with the proxy's would replace a real
            # part with an anonymous cylinder — and would then plan the wrong
            # geometry into a container.
            cad_backed = bool(obs.object_model_id)
            if cad_backed:
                diameter = obs.diameter_mm or geometry.diameter_mm
                length = obs.length_mm or geometry.length_mm
                inner = obs.inner_diameter_mm if hasattr(
                    obs, "inner_diameter_mm") else geometry.inner_diameter_mm
            else:
                # Stamp the CONFIGURED geometry onto the observation too, so an
                # observation that travels alone (to the Isaac synchronizer,
                # say) still carries everything needed to instantiate the
                # object.
                obs.diameter_mm = geometry.diameter_mm
                obs.length_mm = geometry.length_mm
                obs.geometry_source = "configured_proxy"
                diameter, length = geometry.diameter_mm, geometry.length_mm
                inner = geometry.inner_diameter_mm

            # THE PHYSICAL CENTRE, NOT THE MODEL ORIGIN.
            #
            # `obs.position` is the integer projection of x_mm/y_mm/z_mm, which
            # for a model-based observation locates the CAD MODEL FRAME. For a
            # part drawn obliquely that origin lies OUTSIDE the body —
            # Cylinder5's by 141 mm — so planning or grasping from it would
            # target empty space. `object_center` is the body, and for a planar
            # observation (no model centre declared) it IS the reported
            # position, so this path is unchanged for the working detector.
            centre = obs.object_center
            source_position = Vec3(int(round(centre[0])), int(round(centre[1])),
                                   int(round(centre[2])))

            item = WasteItem(
                item_id=f"{id_prefix}-{index:03d}",
                length_mm=length,
                outer_diameter_mm=diameter,
                geometry_type=GeometryType.TUBE,
                inner_diameter_mm=inner,
                material=geometry.material,
                segregation_group=geometry.segregation_group,
                source_position=source_position,
                permitted_axes=tuple(Axis(a) for a in permitted_axes),
                observation=obs,
            )
            if cad_backed:
                # CAD IDENTITY SURVIVES INTO PLANNING. The item is not collapsed
                # into an anonymous generated cylinder: the twin, the audit
                # trail and any later grasp generation can all still name the
                # part they are handling.
                item.geometry_source = GEOMETRY_SOURCE_CAD_MESH
                item.model_id = obs.object_model_id
            item.weight_kg = round(item.material_volume_mm3 * 1e-9 * density, 3)
            items.append(item)
        return items

    # -- the boundary a future Isaac scene synchronizer reads --------------- #

    def scene_objects(self) -> List[Dict[str, Any]]:
        """``PhysicalObservation`` -> a backend-neutral scene object.

        THE MARKED BOUNDARY FOR §14. A scene synchronizer needs exactly four
        things to instantiate a cylinder: an identity, a planar pose, a geometry
        and the frame the pose is expressed in. That is what this returns, and it
        is why no consumer will ever need to parse a detector's own JSON.

        This is not speculative dead code: it is what `/api/perception` publishes
        as `scene_objects`, so the contract is exercised and visible today and
        the synchronizer that arrives later has something already proven to read.
        """
        objects = []
        for obs in self.observations:
            # THE PHYSICAL BODY, not the model frame. A scene synchronizer or a
            # twin renderer placing the CAD ORIGIN would draw Cylinder5 141 mm
            # from where it is, in empty space. For a planar observation the two
            # coincide, so this is unchanged for the working detector.
            centre = obs.object_center
            axis = obs.tube_axis
            objects.append({
                "object_id": obs.observation_id,
                "object_type": obs.object_type,
                "frame_id": obs.frame_id,
                "pose": {"x_mm": round(centre[0], 3),
                         "y_mm": round(centre[1], 3),
                         "z_mm": round(centre[2], 3),
                         "yaw_deg": round(obs.yaw_deg, 3),
                         "reference_point": "object_body"},
                # The long axis, when the method measured one. A LINE: for a
                # straight tube either direction describes the same object.
                "tube_axis_line": ([round(v, 6) for v in axis] if axis else None),
                "geometry": {"shape": "cylinder",
                             "diameter_mm": obs.diameter_mm,
                             "length_mm": obs.length_mm,
                             "inner_diameter_mm": obs.inner_diameter_mm,
                             "source": obs.geometry_source},
                # CAD IDENTITY, so the twin can name the part rather than
                # drawing an anonymous cylinder.
                "model_id": obs.object_model_id,
                "perception_method": obs.perception_method,
                "source": obs.source,
            })
        return objects


# --------------------------------------------------------------------------- #
# Model resolution
# --------------------------------------------------------------------------- #

#: The shared machine-local copy of the trained detector on the ARISE host.
#: Second in the resolution order below: present on the demonstrator machine,
#: absent everywhere else, and never committed to this repository.
ARISE_MODEL_PATH = "/data/arise/models/best_model.pth"

#: The published weights: the bottle detector trained in the HARMONY project and
#: released on Hugging Face. WISEPACK downloads that public artefact into its OWN
#: cache — no HARMONY checkout and no HARMONY installer are involved. Attribution
#: for the model lives in NOTICE and in the provider's `MODEL_ORIGIN`.
HUGGINGFACE_REPO = "hpcbg/harmony-bottle-detector"
HUGGINGFACE_MODEL_URL = (
    f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/main/best_model.pth")


@dataclass
class ModelResolution:
    """Where the detector weights came from, and what to say if they are absent.

    ``origin`` is one of ``configured`` | ``arise_shared`` | ``wisepack_cache``
    | ``downloaded`` | ``absent``.
    """

    path: Optional[str]
    origin: str
    available: bool
    searched: List[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "origin": self.origin,
            "available": self.available,
            "searched": list(self.searched),
            "message": self.message,
            "download_url": HUGGINGFACE_MODEL_URL,
            "repo": HUGGINGFACE_REPO,
        }


def resolve_model_path(configured: Optional[str] = None,
                       cache_dir: Optional[str] = None,
                       env: Optional[Dict[str, str]] = None,
                       exists=os.path.exists) -> ModelResolution:
    """Find the detector weights, in the documented order.

        1. an explicitly configured path (`WISEPACK_PERCEPTION_MODEL_PATH`)
        2. /data/arise/models/best_model.pth, if present
        3. the WISEPACK-owned cache (`<cache_dir>/best_model.pth`)
        4. otherwise: absent, with the Hugging Face URL and the command to fetch
           it — `perception/model_store.py` does that automatically

    NO FOREIGN CHECKOUT IS SEARCHED. An earlier revision looked inside a HARMONY
    clone, which made another repository's directory layout part of WISEPACK's
    runtime contract; the cache is WISEPACK's own and lives in the working
    directory.

    ABSENCE IS A DIAGNOSTIC, NOT A CRASH: a clear message rather than a cryptic
    ``FileNotFoundError`` from deep inside ``torch.load``, so resolution is a
    plain filesystem question answered before torch is imported at all.
    ``exists`` is injectable so the order can be tested without a model.
    """
    env = os.environ if env is None else env
    configured = (configured
                  if configured is not None
                  else env.get("WISEPACK_PERCEPTION_MODEL_PATH", ""))
    configured = str(configured or "").strip()

    searched: List[str] = []

    if configured:
        searched.append(configured)
        if exists(configured):
            return ModelResolution(configured, "configured", True, searched)
        # An EXPLICIT path that does not exist is an error, not a reason to go
        # looking elsewhere: silently loading different weights than the ones
        # that were asked for is worse than reporting the miss.
        return ModelResolution(
            None, "absent", False, searched,
            message=(f"WISEPACK_PERCEPTION_MODEL_PATH={configured!r} does not "
                     "exist. Correct it or unset it to use the default search "
                     "order."))

    searched.append(ARISE_MODEL_PATH)
    if exists(ARISE_MODEL_PATH):
        return ModelResolution(ARISE_MODEL_PATH, "arise_shared", True, searched)

    cache_dir = cache_dir or str(
        env.get("WISEPACK_PERCEPTION_MODEL_CACHE", "") or "").strip()
    if cache_dir:
        cached = os.path.join(cache_dir, "best_model.pth")
        searched.append(cached)
        if exists(cached):
            return ModelResolution(cached, "wisepack_cache", True, searched)

    hint = os.path.join(cache_dir or ".cache-perception/models",
                        "best_model.pth")
    return ModelResolution(
        None, "absent", False, searched,
        message=(
            "No detector weights found. Searched: "
            + ", ".join(searched)
            + ". They are fetched automatically on the next camera start "
              f"(from {HUGGINGFACE_REPO}); to do it by hand:\n"
              f"    ./scripts/setup_perception.sh --model\n"
              f"    curl -L --fail --create-dirs -o {hint} "
              f"{HUGGINGFACE_MODEL_URL}\n"
              "or point WISEPACK_PERCEPTION_MODEL_PATH at an existing copy. "
              "The weights are ~159 MB and are never committed to this "
              "repository."))


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


@dataclass
class PerceptionHealth:
    """The answer to §5's health questions, in one shape, for every surface.

    Every field is tri-state where that is the honest answer: ``None`` means
    "not known from here", which is different from ``False`` ("checked, and
    it is not working"). A dashboard that renders those two the same way is how
    an unreachable detector comes to look like a broken camera.
    """

    source: str = PerceptionSource.SIM.value
    service_url: str = ""
    service_reachable: Optional[bool] = None
    camera_configured: Optional[bool] = None
    camera_available: Optional[bool] = None
    model_available: Optional[bool] = None
    model_loaded: Optional[bool] = None
    model_path: str = ""
    model_origin: str = ""
    calibration_status: str = "unknown"
    last_inference_at: str = ""
    last_error: str = ""
    detected_objects: Optional[int] = None
    detector: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "service_url": self.service_url,
            "service_reachable": self.service_reachable,
            "camera_configured": self.camera_configured,
            "camera_available": self.camera_available,
            "model_available": self.model_available,
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "model_origin": self.model_origin,
            "calibration_status": self.calibration_status,
            "calibration_valid": self.calibration_status == "valid",
            "last_inference_at": self.last_inference_at,
            "last_error": self.last_error,
            "detected_objects": self.detected_objects,
            "detector": self.detector,
        }


# --------------------------------------------------------------------------- #
# Object source: capability, selection, and what the current run actually used
# --------------------------------------------------------------------------- #


@dataclass
class ObjectSourceState:
    """THREE DIFFERENT THINGS, and conflating any two of them is the bug.

        available   which sources this deployment CAN use right now. `camera`
                    is available when a perception service answers, which an
                    operator can make true at any moment by starting one — so it
                    is re-evaluated, never latched at start-up.
        selected    the source the NEXT run will acquire objects from. A DRAFT,
                    owned by the operator, and inert until they start that run.
        current     the source the RUNNING run actually used. Provenance: it is
                    stamped on the scenario and travels to FIWARE, and it does
                    not change when the operator edits the draft.

    The failure this prevents is the one that made the feature unusable: a
    single global "perception mode" meant selecting a camera reached back into
    the run already on screen, and switching back required restarting WISEPACK.
    Following the robot selector's pattern — draft versus active — makes the two
    independent and makes "Running now: Physical camera / Next run: Preset
    scenario" a state the dashboard can simply display.
    """

    current: str = PerceptionSource.SIM.value
    selected: str = PerceptionSource.SIM.value
    available: List[str] = field(
        default_factory=lambda: [PerceptionSource.SIM.value])
    #: Why `camera` is not available, when it is not. Empty when it is.
    camera_unavailable_reason: str = ""
    #: The perception service this deployment would use, for the diagnostic.
    service_url: str = ""

    @property
    def camera_available(self) -> bool:
        return PerceptionSource.CAMERA.value in self.available

    @property
    def changes_next_run(self) -> bool:
        """True when the draft differs from what is running — worth saying so."""
        return self.selected != self.current

    def to_dict(self) -> Dict[str, Any]:
        current = PerceptionSource(self.current)
        selected = PerceptionSource(self.selected)
        return {
            "current": current.value,
            "current_label": current.selector_label,
            "current_provenance": current.provenance,
            "selected": selected.value,
            "selected_label": selected.selector_label,
            "selected_detail": selected.selector_detail,
            "action_label": selected.action_label,
            "changes_next_run": self.changes_next_run,
            "available": list(self.available),
            "camera_available": self.camera_available,
            "camera_unavailable_reason": self.camera_unavailable_reason,
            "service_url": self.service_url,
            "options": [
                {"value": source.value,
                 "label": source.selector_label,
                 "detail": source.selector_detail,
                 "action_label": source.action_label,
                 "available": source.value in self.available,
                 "reason": (self.camera_unavailable_reason
                            if source is PerceptionSource.CAMERA
                            and not self.camera_available else "")}
                for source in PerceptionSource
            ],
        }


@dataclass
class PerceptionMethodState:
    """available / selected / current — the SAME three states as the object source.

    Deliberately the same shape, because it is the same problem. A single global
    "perception method" would mean choosing FoundationPose reached back into the
    run already on screen and relabelled a batch that a planar detector actually
    produced. The batch on screen was measured one way; the operator's draft is
    a different question; conflating them rewrites history.

        available   which methods this deployment can run RIGHT NOW. Re-evaluated
                    on every ask, never latched: an operator can start the
                    FoundationPose worker at any moment and the dashboard has to
                    notice without a restart.
        selected    the method the NEXT physical acquisition will use. A draft.
        current     the method the RUNNING batch was actually measured with.
                    Provenance — it is stamped on the batch and travels to FIWARE.

    `current` is empty when no physical batch exists, which is not the same as
    "planar". A preset run was measured by nothing at all, and saying it used the
    planar detector would be an invented provenance.
    """

    current: str = ""
    selected: str = DEFAULT_PERCEPTION_METHOD
    available: List[str] = field(
        default_factory=lambda: [DEFAULT_PERCEPTION_METHOD])
    #: Why each unavailable method is unavailable, keyed by method. A method
    #: offered as "unavailable" with no reason is a dead end for the operator.
    unavailable_reasons: Dict[str, str] = field(default_factory=dict)

    @property
    def changes_next_run(self) -> bool:
        """True when the draft differs from the running batch's provenance."""
        return bool(self.current) and self.selected != self.current

    def is_available(self, method: str) -> bool:
        return method in self.available

    def to_dict(self) -> Dict[str, Any]:
        selected = PerceptionMethod(self.selected)
        document: Dict[str, Any] = {
            "selected": selected.value,
            "selected_label": selected.selector_label,
            "selected_detail": selected.selector_detail,
            "selected_measures": list(selected.measures),
            # EMPTY, NOT DEFAULTED. No physical batch means no method measured
            # anything, and naming one would fabricate provenance.
            "current": self.current,
            "current_label": (PerceptionMethod(self.current).selector_label
                              if self.current else ""),
            "changes_next_run": self.changes_next_run,
            "available": list(self.available),
            "unavailable_reasons": dict(self.unavailable_reasons),
            "options": [],
        }
        for method in PerceptionMethod:
            document["options"].append({
                "value": method.value,
                "label": method.selector_label,
                "detail": method.selector_detail,
                "measures": list(method.measures),
                "requires_depth": method.requires_depth,
                "requires_object_model": method.requires_object_model,
                # WHICH GEOMETRY THE ESTIMATOR GETS, so the panel can show a
                # CAD model for one and a representation for the other without
                # matching on method names in JavaScript.
                "requires_representation": method.requires_representation,
                "estimator_geometry": method.estimator_geometry,
                "available": method.value in self.available,
                "reason": self.unavailable_reasons.get(method.value, ""),
            })
        selected_method = PerceptionMethod(self.selected)
        document["selected_requires_object_model"] = \
            selected_method.requires_object_model
        document["selected_requires_representation"] = \
            selected_method.requires_representation
        document["selected_estimator_geometry"] = \
            selected_method.estimator_geometry
        if self.current:
            current_method = PerceptionMethod(self.current)
            document["current_estimator_geometry"] = \
                current_method.estimator_geometry
            document["current_requires_representation"] = \
                current_method.requires_representation
        else:
            # EMPTY, NOT DEFAULTED — the same rule as `current` itself. A run
            # that measured nothing was not a CAD run.
            document["current_estimator_geometry"] = ""
            document["current_requires_representation"] = False
        return document


def resolve_perception_method_selection(
        value: Optional[str], available: Sequence[str],
        fallback: str = DEFAULT_PERCEPTION_METHOD) -> str:
    """The operator's draft method, clamped to what is actually runnable.

    A METHOD THAT IS NO LONGER AVAILABLE FALLS BACK, and this is the one place a
    fall back is right: the selection is a DRAFT for a run that has not started.
    A draft naming a worker that has since died must not become a run that
    silently fails, and it must not become a run that quietly uses the other
    method either — so the fallback is to the planar default, which the caller
    then displays. Nothing about an EXISTING batch is changed by this.
    """
    candidates = list(available) or [fallback]
    raw = str(value or "").strip().lower()
    raw = PERCEPTION_METHOD_ALIASES.get(raw, raw)
    if raw in candidates:
        return raw
    return fallback if fallback in candidates else candidates[0]


def resolve_object_source(value: Optional[str], available: Sequence[str],
                          fallback: str = PerceptionSource.SIM.value
                          ) -> PerceptionSource:
    """Resolve a REQUESTED next-run source against what is actually available.

    NEVER SILENTLY SUBSTITUTES. An unknown value raises, and a value that is
    known but unavailable raises with the reason — because the one behaviour
    this must not have is "you asked for the camera, so here is a generated
    scenario". `fallback` is the operator's standing selection, used when this
    particular request named no source; it is checked for availability too,
    because a camera that has gone away since it was selected must produce a
    refusal rather than a preset run wearing a camera label.
    """
    requested = (str(value).strip() if value is not None and str(value).strip()
                 else str(fallback))
    source = resolve_perception_source(requested)
    if source.value not in available:
        raise PerceptionConfigError(
            f"object source {source.selector_label!r} is not available on this "
            "deployment (available: "
            + ", ".join(PerceptionSource(a).selector_label for a in available)
            + ")")
    return source


# --------------------------------------------------------------------------- #
# Staleness
# --------------------------------------------------------------------------- #

#: Seconds after which a physical observation is reported STALE. The objects on
#: a real table can be moved at any moment and the detector has no way to know;
#: an observation is a measurement of a past instant, and a plan built from a
#: ten-minute-old scan should say so rather than look current.
DEFAULT_OBSERVATION_TTL_S = 300.0


def observation_age_s(batch: Optional[ObservationBatch],
                      now_epoch: Optional[float] = None) -> Optional[float]:
    """Age of a batch in seconds, or None when it cannot be determined."""
    if batch is None or not batch.captured_at:
        return None
    stamp = _parse_iso_epoch(batch.captured_at)
    if stamp is None:
        return None
    if now_epoch is None:
        import time                                        # noqa: PLC0415
        now_epoch = time.time()
    return max(0.0, now_epoch - stamp)


def is_stale(batch: Optional[ObservationBatch],
             ttl_s: float = DEFAULT_OBSERVATION_TTL_S,
             now_epoch: Optional[float] = None) -> bool:
    age = observation_age_s(batch, now_epoch)
    return age is not None and age > ttl_s


_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})")


def _parse_iso_epoch(text: str) -> Optional[float]:
    """Parse the UTC ISO-8601 stamps this repository emits. None if unparseable."""
    import calendar                                          # noqa: PLC0415
    match = _ISO_RE.match(str(text or ""))
    if not match:
        return None
    parts = [int(p) for p in match.groups()]
    try:
        return float(calendar.timegm(tuple(parts) + (0, 0, 0)))
    except (ValueError, OverflowError):
        return None


__all__ = [
    "PerceptionSource", "PERCEPTION_SOURCE_ENV", "PerceptionConfigError",
    "resolve_perception_source", "PERCEPTION_DETECTOR_ENV", "DEFAULT_DETECTOR",
    "KNOWN_DETECTORS", "resolve_detector", "ProxyGeometry", "DEFAULT_PROXY_DIAMETER_MM",
    "DEFAULT_PROXY_LENGTH_MM", "WorkAreaFrame", "WORKAREA_FRAME_ID",
    "BatchStatus", "ObservationBatch", "ModelResolution", "resolve_model_path",
    "ARISE_MODEL_PATH", "HUGGINGFACE_REPO", "HUGGINGFACE_MODEL_URL",
    "PerceptionHealth", "DEFAULT_OBSERVATION_TTL_S", "observation_age_s",
    "is_stale", "ObjectSourceState", "resolve_object_source",
    "PERCEPTION_ENABLE_ENV", "camera_capability_requested",
    "PerceptionMethod", "PERCEPTION_METHOD_ENV", "DEFAULT_PERCEPTION_METHOD",
    "KNOWN_PERCEPTION_METHODS", "PERCEPTION_METHOD_ALIASES",
    "resolve_perception_method", "PerceptionMethodState",
    "resolve_perception_method_selection",
]
