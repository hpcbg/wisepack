"""Perception sources: where WISEPACK's object observations come from.

WISEPACK now has THREE orthogonal axes, and collapsing any two of them is the
mistake this module exists to prevent:

    DATA SOURCE        where the DASHBOARD reads state from.
                       sim | ros | fiware.            (see web/snapshot.py)

    PERCEPTION SOURCE  where the OBJECT OBSERVATIONS come from.
                       sim | harmony_camera.          (this module)

    EXECUTION BACKEND  what performs the pick-and-place the operator approved.
                       simulated | isaac.             (see execution.py)

A camera is NOT an execution backend. Every combination is legal and none is
implied by another: camera perception with simulated execution, camera
perception with Isaac execution, simulated perception with Isaac execution.
Selecting a real camera must therefore never touch the execution backend, and
`WISEPACK_EXECUTION_BACKEND` keeps working exactly as before.

WHAT FLOWS BETWEEN THEM
-----------------------
    <any physical detector>
            |
            v
    perception adapter          (harmony_adapter.py — the only detector-aware code)
            |
            v
    ObservationBatch of PhysicalObservation   <- DOMAIN-NEUTRAL, this module
            |
            +--> WasteItem batch --> packing / workflow / validation
            +--> scene_objects() --> (future) Isaac scene synchronizer

Nothing downstream of `ObservationBatch` knows what a bottle is, and nothing
downstream of it knows which detector ran. That is the whole point: the planning
and workflow layers consume real perception data without acquiring a dependency
on the perception vendor, and a future Isaac synchronizer reads `scene_objects()`
rather than parsing any detector's JSON.

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
    Axis, DomainError, GeometryType, PhysicalObservation, WasteItem,
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
    #: The HARMONY Faster R-CNN detector looking at a real camera. Physical
    #: bottles stand in for the cylindrical workpieces WISEPACK handles; their
    #: measured position and orientation become domain-neutral observations.
    HARMONY_CAMERA = "harmony_camera"

    @property
    def is_physical(self) -> bool:
        return self is not PerceptionSource.SIM

    @property
    def label(self) -> str:
        """Dashboard badge text. Never claims more than the source is."""
        return {"sim": "SIMULATED PERCEPTION",
                "harmony_camera": "HARMONY CAMERA"}[self.value]

    @property
    def detail(self) -> str:
        return {
            "sim": ("object observations are the generated ground truth with a "
                    "seeded confidence — no camera, no image, no detector"),
            "harmony_camera": (
                "object observations are produced by the HARMONY Faster R-CNN "
                "detector from a real camera frame, positioned by ArUco "
                "homography; physical bottles are proxies for cylindrical "
                "workpieces"),
        }[self.value]


#: The environment variable that selects the perception source, following the
#: same convention as `WISEPACK_EXECUTION_BACKEND`. Unset == `sim`, so every
#: existing invocation keeps its exact behaviour.
PERCEPTION_SOURCE_ENV = "WISEPACK_PERCEPTION_SOURCE"


class PerceptionConfigError(ValueError):
    """Raised when the perception configuration cannot be honoured."""


def resolve_perception_source(value: Optional[str] = None) -> PerceptionSource:
    """Resolve the perception source: explicit argument, then env, then `sim`.

    An UNRECOGNISED value is an error, never a silent fall back to `sim`. A
    typo in `WISEPACK_PERCEPTION_SOURCE=harmony-camera` that quietly ran the
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

    HARMONY's ArUco calibration defines a plane whose corner markers sit at
    configured millimetre coordinates — with the shipped A4 sheet, a 130 x 130 mm
    square with the origin at marker 11. That square is small for WISEPACK's
    work area, and §13 is explicit that this task must NOT redesign the
    calibration. So the mapping is IDENTITY by default (WISEPACK simply adopts
    HARMONY's calibrated frame verbatim) and the extent is CONFIGURABLE, so a
    larger printed board needs a configuration change and a documented note
    rather than a code change.

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
    captured_at: str = ""
    detector: str = ""
    model_id: str = ""
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
            "detector": self.detector,
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
            detector=str(d.get("detector", "")),
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
        inference timeout, HARMONY unreachable, malformed detector output. The
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
            # Stamp the CONFIGURED geometry onto the observation too, so an
            # observation that travels alone (to the Isaac synchronizer, say)
            # still carries everything needed to instantiate the object.
            obs.diameter_mm = geometry.diameter_mm
            obs.length_mm = geometry.length_mm
            obs.geometry_source = "configured_proxy"
            item = WasteItem(
                item_id=f"{id_prefix}-{index:03d}",
                length_mm=geometry.length_mm,
                outer_diameter_mm=geometry.diameter_mm,
                geometry_type=GeometryType.TUBE,
                inner_diameter_mm=geometry.inner_diameter_mm,
                material=geometry.material,
                segregation_group=geometry.segregation_group,
                source_position=obs.position,
                permitted_axes=tuple(Axis(a) for a in permitted_axes),
                observation=obs,
            )
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
        return [
            {
                "object_id": obs.observation_id,
                "object_type": obs.object_type,
                "frame_id": obs.frame_id,
                "pose": {"x_mm": round(obs.x_mm, 3), "y_mm": round(obs.y_mm, 3),
                         "z_mm": round(obs.z_mm, 3),
                         "yaw_deg": round(obs.yaw_deg, 3)},
                "geometry": {"shape": "cylinder",
                             "diameter_mm": obs.diameter_mm,
                             "length_mm": obs.length_mm,
                             "source": obs.geometry_source},
                "source": obs.source,
            }
            for obs in self.observations
        ]


# --------------------------------------------------------------------------- #
# Model resolution
# --------------------------------------------------------------------------- #

#: The shared machine-local copy of the trained HARMONY detector on the ARISE
#: host. Second in the resolution order below: present on the demonstrator
#: machine, absent everywhere else, and never committed to this repository.
ARISE_MODEL_PATH = "/data/arise/models/best_model.pth"

#: The published weights. HARMONY's own `setup.py` downloads exactly this URL
#: into `ai-bottle-detector-fiware/models/best_model.pth`; WISEPACK reuses that
#: destination and that URL rather than inventing a second download scheme.
HUGGINGFACE_REPO = "hpcbg/harmony-bottle-detector"
HUGGINGFACE_MODEL_URL = (
    f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/main/best_model.pth")


@dataclass
class ModelResolution:
    """Where the detector weights came from, and what to say if they are absent."""

    path: Optional[str]
    origin: str                     # configured | arise_shared | harmony_cache | absent
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
                       harmony_dir: Optional[str] = None,
                       env: Optional[Dict[str, str]] = None,
                       exists=os.path.exists) -> ModelResolution:
    """Find the detector weights, in the documented order (§4).

        1. an explicitly configured path (`WISEPACK_HARMONY_MODEL_PATH`)
        2. /data/arise/models/best_model.pth, if present
        3. the local HARMONY cache (`<harmony>/models/best_model.pth`) — the
           exact destination HARMONY's own installer downloads into
        4. otherwise: absent, with the Hugging Face URL and the command to fetch
           it, so the download step is HARMONY's rather than a second one here

    ABSENCE IS A DIAGNOSTIC, NOT A CRASH. §4 requires a clear message rather
    than a cryptic ``FileNotFoundError`` from deep inside ``torch.load``, so
    resolution is a plain filesystem question answered before torch is imported
    at all. ``exists`` is injectable so the order can be tested without a model.
    """
    env = os.environ if env is None else env
    configured = (configured
                  if configured is not None
                  else env.get("WISEPACK_HARMONY_MODEL_PATH", ""))
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
            message=(f"WISEPACK_HARMONY_MODEL_PATH={configured!r} does not "
                     "exist. Correct it or unset it to use the default search "
                     "order."))

    searched.append(ARISE_MODEL_PATH)
    if exists(ARISE_MODEL_PATH):
        return ModelResolution(ARISE_MODEL_PATH, "arise_shared", True, searched)

    if harmony_dir:
        cached = os.path.join(harmony_dir, "models", "best_model.pth")
        searched.append(cached)
        if exists(cached):
            return ModelResolution(cached, "harmony_cache", True, searched)

    hint = os.path.join(harmony_dir or "<harmony>/ai-bottle-detector-fiware",
                        "models", "best_model.pth")
    return ModelResolution(
        None, "absent", False, searched,
        message=(
            "No HARMONY detector weights found. Searched: "
            + ", ".join(searched)
            + ". Fetch them with HARMONY's own installer "
              "(`python3 setup.py` in the HARMONY repository, which downloads "
              f"{HUGGINGFACE_REPO}), or directly:\n"
              f"    curl -L --fail -o {hint} {HUGGINGFACE_MODEL_URL}\n"
              "or point WISEPACK_HARMONY_MODEL_PATH at an existing copy. "
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
    "resolve_perception_source", "ProxyGeometry", "DEFAULT_PROXY_DIAMETER_MM",
    "DEFAULT_PROXY_LENGTH_MM", "WorkAreaFrame", "WORKAREA_FRAME_ID",
    "BatchStatus", "ObservationBatch", "ModelResolution", "resolve_model_path",
    "ARISE_MODEL_PATH", "HUGGINGFACE_REPO", "HUGGINGFACE_MODEL_URL",
    "PerceptionHealth", "DEFAULT_OBSERVATION_TTL_S", "observation_age_s",
    "is_stale",
]
