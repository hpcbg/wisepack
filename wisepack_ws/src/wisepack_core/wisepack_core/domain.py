"""WISEPACK domain model — typed, validated, and free of ROS.

Nothing in this module imports rclpy, FastAPI or yaml. That is deliberate and it
is the most important structural rule in the repository: the *same* objects and
the *same* packing arithmetic back the ROS 2 nodes, the dashboard's live mode and
its no-ROS simulation mode. A demo whose "sim mode" re-implements the numbers
proves nothing, so sim mode here is this code with a different transport.

Units are millimetres and kilograms. Every length is an integer number of
millimetres: the packing search compares and adds coordinates constantly, and
float drift there turns an exact "fits" into a random near-miss. Volumes are
computed in mm^3 and converted only at the reporting boundary.

Two volumes exist for every waste item and they are NOT interchangeable:

  * material volume  — the metal actually present (pi/4 * (OD^2 - ID^2) * L for a
    tube). A hollow pipe is mostly air; this is what a mass balance uses.
  * occupied volume  — the axis-aligned bounding box the item consumes inside a
    container (L * OD * OD for a tube lying along one axis).

Container demand is driven by *occupied* volume. Conflating the two is the
classic way to report a packing improvement that does not exist: material volume
is identical whichever algorithm placed it, so using it as a denominator makes
every algorithm look equally good (or equally bad). See kpi.py.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 3-D pose, symmetry and frames. Standard library only — importing it here does
# not change what `domain` costs to load, and it keeps the quaternion rules in
# one place instead of spread across the producers.
from .pose import (WORKAREA_FRAME, Orientation, Symmetry,
                   canonicalize)

#: How far the planar `yaw_deg` and the quaternion's projection may differ and
#: still be considered the same angle. Comfortably above quaternion round-trip
#: noise (~1e-9 deg) and far below any disagreement that could matter.
_YAW_AGREEMENT_DEG = 1e-6


def _wrapped_degrees(delta: float) -> float:
    """`delta` folded into (-180, 180]. -179 and +181 are the same angle."""
    return (float(delta) + 180.0) % 360.0 - 180.0

SCHEMA_VERSION = "wisepack/1"

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


#: How an item's shape is obtained. See `WasteItem.geometry_source`.
GEOMETRY_SOURCE_GENERATED = "generated"
GEOMETRY_SOURCE_CAD_MESH = "cad_mesh"
GEOMETRY_SOURCES = (GEOMETRY_SOURCE_GENERATED, GEOMETRY_SOURCE_CAD_MESH)


class GeometryType(str, Enum):
    """The six EDF/CEA waste geometry classes named in the WISEPACK proposal.

    Only ``TUBE`` (a straight cylindrical pipe) has an exact analytic model in
    this demonstrator. The other five are the documented stretch geometries and
    are represented by a *conservative bounding box* — an over-estimate of the
    space consumed, never an under-estimate, so a plan built from them is safe
    but pessimistic. ``WasteItem.is_approximated`` reports this per item and
    every surface that shows such an item labels it.
    """

    TUBE = "tube"                    # straight pipe — exact model
    BENT_TUBE = "bent_tube"          # approximated by bounding box
    FLAT_SHEET = "flat_sheet"        # approximated by bounding box
    CURVED_SHEET = "curved_sheet"    # approximated by bounding box
    CURVED_PANEL = "curved_panel"    # approximated by bounding box
    I_BEAM = "i_beam"                # approximated by bounding box


#: Geometry classes whose occupied volume is an over-estimate, not an exact box.
APPROXIMATED_GEOMETRIES = frozenset(
    g for g in GeometryType if g is not GeometryType.TUBE)


class ItemStatus(str, Enum):
    PENDING = "pending"            # generated, not yet planned
    PLANNED = "planned"            # has a validated placement in the selected plan
    PICKED = "picked"              # robot holds it
    PLACED = "placed"              # executed into a container
    UNPLACED = "unplaced"          # no feasible placement found
    REMOVED = "removed"            # withdrawn by a dynamic event


class ContainerStatus(str, Enum):
    AVAILABLE = "available"
    FILLING = "filling"
    COMPLETE = "complete"
    UNAVAILABLE = "unavailable"


class ValidationStatus(str, Enum):
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"


class ApprovalState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"      # replaced by a re-plan before a decision


class Axis(str, Enum):
    """Axis along which the item's longest dimension points, in container frame.

    Only axis-aligned orientations are modelled. A straight tube is rotationally
    symmetric about its own axis, so these three choices are the complete set of
    distinct axis-aligned orientations for it — there is no 6-permutation
    explosion. Non-symmetric geometries reuse the same three via their bounding
    box, which is why that box must be conservative.
    """

    X = "x"
    Y = "y"
    Z = "z"


class Source(str, Enum):
    """Provenance of a reported figure. Never guess this value.

    MEASURED  — produced by running code on this machine (optimizer timings,
                container counts, utilization, DDS->FIWARE latency) or read from
                a real sensor (a physical perception observation).
    SIMULATED — produced by the simulator (pick outcomes, perception confidence,
                dose class). Real inside the demo, not real in the world.
    OPERATOR  — supplied by a human through the dashboard.
    TARGET    — a WISEPACK proposal target (KPI1-KPI4). NOT a result.

    MEASURED on a perception observation says the POSE was measured by a real
    detector. It says nothing about detection *accuracy* — that needs a
    ground-truth trial, and kpi.py refuses to invent one from confidence.
    """

    MEASURED = "measured"
    SIMULATED = "simulated"
    OPERATOR = "operator"
    TARGET = "target"


class Strategy(str, Enum):
    """Operator-selectable packing strategies.

    The proposal promises that "packing strategies focused on maximum density,
    improved retrievability or waste segregation can be selected by the operator
    and compared before execution". These are those three. They differ only in
    the objective weights (see optimizer.py) — the hard constraints are identical
    in all three, because a strategy must never be able to buy density with a
    boundary or segregation violation.
    """

    MAX_DENSITY = "max_density"
    RETRIEVABILITY = "retrievability"
    SEGREGATION = "segregation"


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


class DomainError(ValueError):
    """Raised when a domain object is constructed with impossible values."""


# Identifiers end up inside NGSI-LD entity ids (urn:ngsi-ld:Type:id), where
# spaces and '/' are not safe, so they are constrained at construction.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")


def _require_id(name: str, value: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise DomainError(
            f"{name} must match {_ID_RE.pattern!r}, got {value!r}. "
            "Identifiers travel into NGSI-LD entity ids.")
    return value


def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainError(f"{name} must be a number, got {value!r}")
    if isinstance(value, float) and not float(value).is_integer():
        # Refuse silently-rounded millimetres rather than absorbing the error.
        raise DomainError(f"{name} must be a whole number of mm, got {value!r}")
    ivalue = int(value)
    if ivalue <= 0:
        raise DomainError(f"{name} must be > 0 mm, got {ivalue}")
    return ivalue


def _require_non_negative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainError(f"{name} must be a number, got {value!r}")
    if value < 0:
        raise DomainError(f"{name} must be >= 0, got {value}")
    return float(value)


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Vec3:
    """An integer millimetre position or size."""

    x: int = 0
    y: int = 0
    z: int = 0

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.x, self.y, self.z)

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "z": self.z}

    @staticmethod
    def from_dict(d: Any) -> "Vec3":
        if d is None:
            return Vec3()
        if isinstance(d, (list, tuple)):
            return Vec3(int(d[0]), int(d[1]), int(d[2]))
        return Vec3(int(d.get("x", 0)), int(d.get("y", 0)), int(d.get("z", 0)))


@dataclass(frozen=True)
class Box:
    """An axis-aligned box: origin (min corner) plus size, both in mm.

    Intervals are half-open, ``[min, min+size)`` — two boxes sharing a face do
    NOT overlap. Getting this wrong makes every flush placement look like a
    collision, which silently halves the achievable density.
    """

    origin: Vec3
    size: Vec3

    @property
    def max_corner(self) -> Vec3:
        return Vec3(self.origin.x + self.size.x,
                    self.origin.y + self.size.y,
                    self.origin.z + self.size.z)

    @property
    def volume_mm3(self) -> int:
        return self.size.x * self.size.y * self.size.z

    def overlaps(self, other: "Box") -> bool:
        a_max, b_max = self.max_corner, other.max_corner
        return (self.origin.x < b_max.x and other.origin.x < a_max.x
                and self.origin.y < b_max.y and other.origin.y < a_max.y
                and self.origin.z < b_max.z and other.origin.z < a_max.z)

    def within(self, outer_size: Vec3) -> bool:
        """True when this box lies fully inside a container of ``outer_size``."""
        m = self.max_corner
        return (self.origin.x >= 0 and self.origin.y >= 0 and self.origin.z >= 0
                and m.x <= outer_size.x and m.y <= outer_size.y
                and m.z <= outer_size.z)

    def footprint_overlap_mm2(self, other: "Box") -> int:
        """Overlap area of the two boxes projected onto the XY plane."""
        a_max, b_max = self.max_corner, other.max_corner
        dx = min(a_max.x, b_max.x) - max(self.origin.x, other.origin.x)
        dy = min(a_max.y, b_max.y) - max(self.origin.y, other.origin.y)
        return max(0, dx) * max(0, dy)

    def gap_to(self, other: "Box") -> int:
        """Chebyshev-style separation in mm: 0 when touching or overlapping."""
        a_max, b_max = self.max_corner, other.max_corner
        gx = max(other.origin.x - a_max.x, self.origin.x - b_max.x)
        gy = max(other.origin.y - a_max.y, self.origin.y - b_max.y)
        gz = max(other.origin.z - a_max.z, self.origin.z - b_max.z)
        return max(0, max(gx, gy, gz))

    def to_dict(self) -> Dict[str, Any]:
        return {"origin": self.origin.to_dict(), "size": self.size.to_dict()}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Box":
        return Box(Vec3.from_dict(d["origin"]), Vec3.from_dict(d["size"]))


# --------------------------------------------------------------------------- #
# PhysicalObservation
# --------------------------------------------------------------------------- #


@dataclass
class PhysicalObservation:
    """One object OBSERVED by a real perception system, in physical units.

    DOMAIN-NEUTRAL BY CONSTRUCTION. Nothing here names a bottle, a detector
    architecture or a camera vendor. A physical detector reports "there is a
    cylindrical object at (x, y) rotated by yaw, and I am this confident"; that
    is the whole content of this type. Whichever detector produced it goes in
    ``detector``/``model_id`` as provenance, never in the shape of the data.

    WHY THE UNITS ARE FLOATS HERE AND INTEGERS EVERYWHERE ELSE. The packing
    arithmetic uses integer millimetres on purpose (see the module docstring):
    float drift there turns an exact "fits" into a random near-miss. A sensor
    reading is a different kind of number — it has sub-millimetre precision that
    is genuinely part of the measurement, and rounding it at the sensor boundary
    would throw away the only copy. So the observation keeps the measured floats
    and ``WasteItem.source_position`` carries the rounded integer projection the
    planner uses. Both are published; neither is derived from the other twice.

    ``x_mm``/``y_mm``/``yaw_deg`` are expressed in ``frame_id``, and ``frame_id``
    is mandatory. A pose without a frame is not a pose — it is three numbers, and
    the later Isaac scene synchronizer would have no way to place them.

    GEOMETRY IS NOT MEASURED. ``diameter_mm``/``length_mm`` are the configured
    known dimensions of the physical proxy object, carried here so a consumer
    has one complete record. ``geometry_source`` says so explicitly. A 2-D
    detector cannot measure the diameter of a cylinder, and fabricating one from
    a bounding box would put an invented number into the packing arithmetic.
    """

    observation_id: str
    x_mm: float
    y_mm: float
    yaw_deg: float = 0.0
    z_mm: float = 0.0
    confidence: Optional[float] = None
    object_type: str = "cylindrical_proxy"
    source: str = "unknown"                 # perception source id, e.g. camera
    frame_id: str = WORKAREA_FRAME
    #: -- provenance (§11): enough to debug or re-analyse this detection later --
    detector: str = ""                      # detector/model family identification
    model_id: str = ""                      # weights identity (path, hash or repo id)
    detector_class: str = ""                # the DETECTOR's own class label
    detector_object_index: Optional[int] = None   # index/id within its own result
    captured_at: str = ""                   # ISO-8601 capture/detection timestamp
    calibration_status: str = "unknown"     # valid | invalid | unknown
    calibration_revision: str = ""
    #: -- known proxy geometry (configured, never inferred from the detector) --
    diameter_mm: Optional[int] = None
    length_mm: Optional[int] = None
    #: The bore, for a hollow part. Carried so the planner weighs a tube rather
    #: than the solid rod that bounds it.
    inner_diameter_mm: Optional[int] = None
    geometry_source: str = "configured_proxy"
    # -- FULL 3-D ORIENTATION ------------------------------------------------ #
    #
    # ADDITIVE AND OPTIONAL, so every existing producer and consumer is
    # unaffected. A planar detector sets none of these and gets an orientation
    # derived from its own yaw, which is exactly what its yaw means; a
    # model-based RGB-D estimator sets them explicitly.
    #
    # THE QUATERNION IS AUTHORITATIVE — `yaw_deg` above remains the planar
    # projection every existing consumer already reads, and the two are kept
    # consistent rather than allowed to disagree. See `wisepack_core.pose`.
    orientation: Optional["Orientation"] = None
    #: The estimator's UNMODIFIED output, kept when symmetry canonicalisation
    #: changed the reported orientation. Diagnostics only: it is evidence about
    #: the estimator, not a second opinion about the object.
    orientation_raw: Optional["Orientation"] = None
    #: What the object's SHAPE makes unobservable. Carried on the observation so
    #: a consumer never has to know what an `object_type` implies.
    symmetry: Optional["Symmetry"] = None
    #: WHICH METHOD produced this — `planar_fasterrcnn`, `foundationpose_rgbd`,
    #: `sim`. Provenance beside `detector`, never a type distinction: nothing
    #: downstream branches on it.
    perception_method: str = ""
    #: The object model this pose was estimated against, when the method is
    #: model-based. Empty for methods that use no CAD.
    object_model_id: str = ""
    #: Whether THE ESTIMATE ITSELF is structurally and numerically valid IN
    #: `frame_id`. That is the only question this field answers.
    #:
    #: TWO DIFFERENT VALIDITIES, AND CONFLATING THEM WAS A BUG. A model-based
    #: estimator can produce a perfectly good 6-DoF pose in the camera optical
    #: frame while no camera-to-work-area extrinsic exists. The pose is real,
    #: reproducible and correct where it lives; what is missing is a way to move
    #: it somewhere else. Reporting that as `pose_valid=False` said the
    #: measurement was bad, which was not true and hid the actual gap.
    #:
    #: So: `pose_valid` is about the ESTIMATE, `frame_id` says WHERE it lives,
    #: and `workarea_transform_valid` below says whether it can be placed in the
    #: work area. False here means the estimate genuinely failed or is kept only
    #: as diagnostic evidence.
    pose_valid: bool = True
    #: Whether a VALIDATED SE(3) transform exists from `frame_id` into the work
    #: area. Only meaningful when the pose is not already expressed there.
    #:
    #: Never set true by an identity transform standing in for a measurement:
    #: an unmeasured extrinsic is missing, not identity, and assuming identity
    #: puts objects wherever the camera happens to be. The transform itself,
    #: when one exists, belongs in a `wisepack_core.pose.RigidTransform`, whose
    #: `valid` already requires a named `method` for exactly this reason.
    workarea_transform_valid: bool = False
    #: Which degrees of freedom this method actually MEASURED. A planar detector
    #: measures three of six; saying so stops a consumer reading an assumed zero
    #: as a measured height.
    measured_dof: Tuple[str, ...] = ()
    # -- TASK-LEVEL GEOMETRY ------------------------------------------------ #
    #
    # WHAT `x_mm`/`y_mm`/`z_mm` ACTUALLY LOCATE, and why that is not one answer.
    #
    # A model-based estimator reports the pose of the CAD MODEL FRAME, and for a
    # part drawn obliquely that frame's origin can sit far outside the body —
    # Cylinder5's is 141 mm away, in empty space. A planar detector reports the
    # object itself. Both are correct; they are different points.
    #
    # So the model frame stays in `x_mm`/`y_mm`/`z_mm` (unchanged, and correct
    # for FoundationPose), and the PHYSICAL point a gripper must go to is
    # derived and named separately. A grasp planner that consumed the model
    # origin would send the arm 141 mm into thin air.
    #
    # Carried ON the observation rather than looked up, so a consumer receiving
    # it over DDS or FIWARE can grasp without the object registry.
    model_center_mm: Tuple[float, ...] = ()
    #: The object's long axis in MODEL coordinates. Measured, and a vector
    #: because these parts are drawn obliquely.
    task_axis_vector: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _require_id("observation_id", self.observation_id)
        for name in ("x_mm", "y_mm", "yaw_deg", "z_mm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DomainError(f"{self.observation_id}: {name} must be a "
                                  f"number, got {value!r}")
            setattr(self, name, float(value))
        if self.confidence is not None:
            self.confidence = float(self.confidence)
            if not 0.0 <= self.confidence <= 1.0:
                raise DomainError(
                    f"{self.observation_id}: confidence must be in [0, 1], "
                    f"got {self.confidence}")
        if not self.frame_id:
            raise DomainError(
                f"{self.observation_id}: frame_id is mandatory — a pose without "
                "a coordinate frame cannot be placed by any consumer")
        if self.diameter_mm is not None:
            self.diameter_mm = _require_positive_int("diameter_mm", self.diameter_mm)
        if self.length_mm is not None:
            self.length_mm = _require_positive_int("length_mm", self.length_mm)

        # -- 3-D orientation: derived when absent, reconciled when present --- #
        #
        # ONE POSE, TWO REPRESENTATIONS, NEVER ALLOWED TO DISAGREE.
        #
        #   no quaternion  -> build it from the planar yaw. That is what the
        #                     yaw means, so nothing is invented.
        #   a quaternion   -> `yaw_deg` becomes its planar projection, so the
        #                     existing planar consumers keep working unchanged
        #                     on a 6-DoF observation instead of reading a stale
        #                     number that was never updated.
        if isinstance(self.orientation, dict):
            self.orientation = Orientation.from_dict(self.orientation)
        if isinstance(self.orientation_raw, dict):
            self.orientation_raw = Orientation.from_dict(self.orientation_raw)
        if isinstance(self.symmetry, dict):
            self.symmetry = Symmetry.from_dict(self.symmetry)
        if self.orientation is None:
            self.orientation = Orientation.from_yaw_deg(self.yaw_deg)
        else:
            projected = float(self.orientation.yaw_deg)
            # THE SUPPLIED YAW WINS WHEN IT ALREADY AGREES. Converting a yaw to
            # a quaternion and back is lossy at the 1e-9 level, and overwriting
            # a caller's exact -31.0 with -30.999999997 would put float noise
            # into an audit trail and into every test that compares poses. A
            # DISAGREEMENT, though, is resolved in favour of the quaternion:
            # it is the authoritative representation.
            if abs(_wrapped_degrees(projected - self.yaw_deg)) > _YAW_AGREEMENT_DEG:
                self.yaw_deg = projected
        self.measured_dof = tuple(str(d) for d in (self.measured_dof or ()))

    @property
    def object_center(self) -> Tuple[float, float, float]:
        """The PHYSICAL centre of the object, in `frame_id`. Millimetres.

        THIS is what a grasp targets. When a model centre is declared it is
        transformed by the estimated orientation and added to the model-frame
        position; when none is declared — the planar case, where the detector
        already reports the object itself — the reported position IS the body
        and is returned unchanged.
        """
        if not self.model_center_mm or self.orientation is None:
            return (self.x_mm, self.y_mm, self.z_mm)
        rotated = self.orientation.rotate(list(self.model_center_mm))
        return (self.x_mm + rotated[0], self.y_mm + rotated[1],
                self.z_mm + rotated[2])

    @property
    def tube_axis(self) -> Optional[Tuple[float, float, float]]:
        """The object's long axis as a unit vector in `frame_id`, or None.

        A LINE, not an arrow: for a straight tube either direction describes the
        same object, and no consumer may read the sign as meaningful.
        """
        if not self.task_axis_vector or self.orientation is None:
            return None
        rotated = self.orientation.rotate(list(self.task_axis_vector))
        length = math.sqrt(sum(v * v for v in rotated))
        if length < 1e-12:
            return None
        return tuple(v / length for v in rotated)

    def task_geometry(self) -> Dict[str, Any]:
        """Everything a pick needs, and nothing that would mislead it."""
        axis = self.tube_axis
        return {
            "object_center_mm": [round(v, 3) for v in self.object_center],
            "tube_axis_line": ([round(v, 6) for v in axis] if axis else None),
            "tube_axis_is_a_line_not_a_direction": True,
            "diameter_mm": self.diameter_mm,
            "length_mm": self.length_mm,
            "frame_id": self.frame_id,
            "note": ("object_center_mm is the physical body centre and is what "
                     "a grasp targets. It is NOT pose.model_frame_origin_mm, "
                     "which is where the CAD model's own origin lands and can "
                     "be outside the object."),
        }

    @property
    def workarea_pose_available(self) -> bool:
        """Can this observation be placed in the work area?

        DERIVED, so the two ways of being placeable cannot drift apart: a pose
        already expressed in the work-area frame needs no transform, and one in
        another frame needs a validated transform into it. This is the question
        the Isaac scene synchronizer and any planner must ask — never
        `pose_valid`, which is about the measurement rather than about where it
        can be put.
        """
        if not self.pose_valid:
            return False
        if self.frame_id == WORKAREA_FRAME:
            return True
        return bool(self.workarea_transform_valid)

    @property
    def position(self) -> Vec3:
        """The integer-millimetre projection the packing layer consumes."""
        return Vec3(int(round(self.x_mm)), int(round(self.y_mm)),
                    int(round(self.z_mm)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "object_type": self.object_type,
            "source": self.source,
            "frame_id": self.frame_id,
            # THE PLANAR KEYS ARE UNCHANGED AND STILL FIRST. Every existing
            # consumer — the dashboard, the twin validator, the FIWARE bridge —
            # reads exactly what it read before; the 3-D content is additive.
            "pose": {
                # UNCHANGED KEYS, so every existing consumer keeps working —
                # but NAMED below, because for a model-based method these
                # locate the CAD model frame and not the object.
                "x_mm": round(self.x_mm, 3),
                "y_mm": round(self.y_mm, 3),
                "z_mm": round(self.z_mm, 3),
                "reference_point": ("model_frame_origin" if self.model_center_mm
                                    else "object_body"),
                "yaw_deg": round(self.yaw_deg, 3),
                # The authoritative orientation. `yaw_deg` above is its planar
                # projection, kept for the consumers that only need a plane.
                "orientation": self.orientation.to_dict() if self.orientation else None,
                # Derived, for humans reading a dashboard. Never consumed.
                "rpy_deg": ([round(v, 3) for v in self.orientation.rpy_deg()]
                            if self.orientation else None),
                # THE ESTIMATE'S OWN VALIDITY, in `frame_id` above. Not a
                # statement about whether it can be placed in the work area —
                # that is the next two keys.
                "valid": bool(self.pose_valid),
                "workarea_transform_valid": bool(self.workarea_transform_valid),
                "workarea_pose_available": self.workarea_pose_available,
                "measured_dof": list(self.measured_dof),
            },
            # THE TASK-LEVEL VIEW, derived and named. A planner reads this and
            # never has to know which reference point `pose` used.
            "task": self.task_geometry(),
            "model_center_mm": list(self.model_center_mm),
            "task_axis_vector": list(self.task_axis_vector),
            "confidence": (round(self.confidence, 4)
                           if self.confidence is not None else None),
            "perception_method": self.perception_method,
            "object_model_id": self.object_model_id,
            "symmetry": self.symmetry.to_dict() if self.symmetry else None,
            # The estimator's own output, when canonicalisation changed what is
            # reported above. Present only when the two genuinely differ, so its
            # presence is itself the signal that a symmetry was collapsed.
            "orientation_raw": (self.orientation_raw.to_dict()
                                if self.orientation_raw else None),
            "detector": self.detector,
            "model_id": self.model_id,
            "detector_class": self.detector_class,
            "detector_object_index": self.detector_object_index,
            "captured_at": self.captured_at,
            "calibration_status": self.calibration_status,
            "calibration_revision": self.calibration_revision,
            "geometry": {
                "diameter_mm": self.diameter_mm,
                "length_mm": self.length_mm,
                "inner_diameter_mm": self.inner_diameter_mm,
                "source": self.geometry_source,
            },
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PhysicalObservation":
        pose = d.get("pose") or {}
        geometry = d.get("geometry") or {}
        return PhysicalObservation(
            observation_id=d["observation_id"],
            x_mm=float(pose.get("x_mm", d.get("x_mm", 0.0))),
            y_mm=float(pose.get("y_mm", d.get("y_mm", 0.0))),
            yaw_deg=float(pose.get("yaw_deg", d.get("yaw_deg", 0.0))),
            z_mm=float(pose.get("z_mm", d.get("z_mm", 0.0))),
            confidence=(None if d.get("confidence") is None
                        else float(d["confidence"])),
            object_type=d.get("object_type", "cylindrical_proxy"),
            source=d.get("source", "unknown"),
            frame_id=d.get("frame_id", "wisepack_workarea"),
            detector=d.get("detector", ""),
            model_id=d.get("model_id", ""),
            detector_class=d.get("detector_class", ""),
            detector_object_index=d.get("detector_object_index"),
            captured_at=d.get("captured_at", ""),
            calibration_status=d.get("calibration_status", "unknown"),
            calibration_revision=d.get("calibration_revision", ""),
            diameter_mm=geometry.get("diameter_mm"),
            inner_diameter_mm=geometry.get("inner_diameter_mm"),
            length_mm=geometry.get("length_mm"),
            geometry_source=geometry.get("source", "configured_proxy"),
            # ABSENT MEANS PLANAR, not malformed: a document written before
            # 6-DoF existed still parses, and its orientation is derived from
            # its yaw exactly as it always implied.
            orientation=(Orientation.from_dict(pose["orientation"])
                         if pose.get("orientation") else None),
            orientation_raw=(Orientation.from_dict(d["orientation_raw"])
                             if d.get("orientation_raw") else None),
            symmetry=(Symmetry.from_dict(d["symmetry"])
                      if d.get("symmetry") else None),
            perception_method=d.get("perception_method", ""),
            object_model_id=d.get("object_model_id", ""),
            pose_valid=bool(pose.get("valid", True)),
            model_center_mm=tuple(d.get("model_center_mm") or ()),
            task_axis_vector=tuple(d.get("task_axis_vector") or ()),
            # ABSENT MEANS "not stated", and for a document written before this
            # field existed the honest reading is that nothing claimed a
            # transform. A planar observation is unaffected: it is already in
            # the work-area frame, so `workarea_pose_available` derives True
            # without one.
            workarea_transform_valid=bool(pose.get("workarea_transform_valid",
                                                   False)),
            measured_dof=tuple(pose.get("measured_dof") or ()),
        )


# --------------------------------------------------------------------------- #
# WasteItem
# --------------------------------------------------------------------------- #


@dataclass
class WasteItem:
    """One piece of metallic waste to be packaged.

    ``dose_class`` is SIMULATED METADATA. There is no radiation model anywhere in
    this repository; the field exists so the segregation and priority machinery
    has the shape the real system needs, and every surface showing it says
    "simulated".

    For the five approximated geometry classes, ``length_mm`` and
    ``outer_diameter_mm`` describe the *enclosing box* (length x d x d), which
    over-states the item. ``profile_fill_ratio`` records how much of that box the
    real part occupies, purely so the analytics layer can report how pessimistic
    the approximation is. It is NEVER used to shrink the box used for packing.
    """

    item_id: str
    length_mm: int
    outer_diameter_mm: int
    geometry_type: GeometryType = GeometryType.TUBE
    inner_diameter_mm: Optional[int] = None
    material: str = "carbon_steel"
    segregation_group: str = "A"
    weight_kg: float = 0.0
    source_position: Vec3 = field(default_factory=Vec3)
    priority: int = 0                       # higher is handled earlier
    status: ItemStatus = ItemStatus.PENDING
    dose_class: Optional[str] = None        # SIMULATED metadata only
    permitted_axes: Tuple[Axis, ...] = (Axis.X, Axis.Y, Axis.Z)
    injected: bool = False                  # arrived via a dynamic event
    profile_fill_ratio: float = 1.0         # reporting only, never packs smaller

    # -- cut-aware planning metadata -------------------------------------- #
    # These describe whether and how a straight pipe MAY be segmented so it fits
    # residual container cavities. They are pure metadata on the un-cut item: no
    # field here shrinks the packing box. An actual cut produces NEW WasteItems
    # (the derived segments) via cutting.py; this item then leaves the packable
    # set. All default to "no cutting", so every pre-existing scenario JSON keeps
    # its exact behaviour (backward compatible — see from_dict).
    # -- geometry provenance ------------------------------------------------ #
    #
    # WHERE THIS ITEM'S SHAPE COMES FROM. Two paths coexist and neither replaces
    # the other:
    #
    #   GENERATED  a parametric tube described by the fields above. The existing
    #              preset scenarios, the optimizer regressions and every test
    #              that needs no CAD use this, and it stays the DEFAULT so those
    #              are byte-for-byte unaffected.
    #   CAD_MESH   a real reference part, identified by `model_id` and resolved
    #              through the object-model registry. Used by the perception,
    #              FoundationPose and sim-to-real scenarios, where the exact
    #              geometry — hollow bore, saddle ends — is the whole point.
    #
    # THE PATH IS DECLARED, NEVER INFERRED, and this layer never resolves it: a
    # mesh path is looked up by whoever needs the geometry (the Isaac adapter,
    # the perception provider), so planning code neither imports a simulator nor
    # parses an STL.
    geometry_source: str = GEOMETRY_SOURCE_GENERATED
    #: The object-model registry key, for a CAD-backed item. Empty otherwise.
    model_id: str = ""
    cut_allowed: bool = False
    minimum_segment_length_mm: Optional[int] = None   # None == no explicit floor
    maximum_number_of_cuts: int = 0                   # 0 == uncut only
    protected_end_length_mm: int = 0                  # keep-out zone at each end
    parent_item_id: Optional[str] = None              # None == an original item
    generation: int = 0                               # 0 == original, 1.. derived
    cut_history: List[Dict[str, Any]] = field(default_factory=list)
    derived_item_ids: List[str] = field(default_factory=list)

    # -- physical perception provenance ----------------------------------- #
    # Set ONLY when this item came from a real perception source. None for every
    # generated item, which is what keeps the default `sim` behaviour and every
    # pre-existing scenario JSON byte-identical (see from_dict).
    #
    # The packing algorithms never read this. It exists so the measured pose,
    # its confidence and its detector provenance survive into the item state and
    # out through the API — §3 requires x/y/yaw/confidence to be preserved even
    # though the packer does not need them yet, and the Isaac scene synchronizer
    # will read exactly this field rather than any detector-specific JSON.
    observation: Optional["PhysicalObservation"] = None

    def __post_init__(self) -> None:
        _require_id("item_id", self.item_id)
        self.length_mm = _require_positive_int("length_mm", self.length_mm)
        self.outer_diameter_mm = _require_positive_int(
            "outer_diameter_mm", self.outer_diameter_mm)
        if self.inner_diameter_mm is not None:
            self.inner_diameter_mm = _require_positive_int(
                "inner_diameter_mm", self.inner_diameter_mm)
            if self.inner_diameter_mm >= self.outer_diameter_mm:
                raise DomainError(
                    f"{self.item_id}: inner_diameter_mm ({self.inner_diameter_mm}) "
                    f"must be < outer_diameter_mm ({self.outer_diameter_mm})")
        self.weight_kg = _require_non_negative("weight_kg", self.weight_kg)
        self.geometry_type = GeometryType(self.geometry_type)
        self.status = ItemStatus(self.status)
        if not self.permitted_axes:
            raise DomainError(f"{self.item_id}: permitted_axes must not be empty")
        self.permitted_axes = tuple(dict.fromkeys(Axis(a) for a in self.permitted_axes))
        if not isinstance(self.segregation_group, str) or not self.segregation_group:
            raise DomainError(
                f"{self.item_id}: segregation_group must be a non-empty string")
        if not 0.0 < self.profile_fill_ratio <= 1.0:
            raise DomainError(
                f"{self.item_id}: profile_fill_ratio must be in (0, 1], "
                f"got {self.profile_fill_ratio}")
        # -- cut metadata coercion / validation ---------------------------- #
        if self.parent_item_id is not None:
            _require_id("parent_item_id", self.parent_item_id)
        if self.generation < 0 or not isinstance(self.generation, int) \
                or isinstance(self.generation, bool):
            raise DomainError(
                f"{self.item_id}: generation must be a non-negative int, "
                f"got {self.generation!r}")
        if self.maximum_number_of_cuts < 0:
            raise DomainError(
                f"{self.item_id}: maximum_number_of_cuts must be >= 0")
        if self.protected_end_length_mm < 0:
            raise DomainError(
                f"{self.item_id}: protected_end_length_mm must be >= 0")
        if self.minimum_segment_length_mm is not None:
            self.minimum_segment_length_mm = _require_positive_int(
                "minimum_segment_length_mm", self.minimum_segment_length_mm)
        # Cutting is only modelled for straight tubes (the sole exact geometry).
        if self.cut_allowed and self.geometry_type is not GeometryType.TUBE:
            raise DomainError(
                f"{self.item_id}: cut_allowed is only supported for tubes, "
                f"not {self.geometry_type.value}")
        self.derived_item_ids = [
            _require_id("derived_item_id", d) for d in self.derived_item_ids]
        self.cut_history = list(self.cut_history)

    # -- cut metadata helpers --------------------------------------------- #

    @property
    def is_cuttable(self) -> bool:
        """True when this specific item may be segmented by the cut planner."""
        return (self.cut_allowed
                and self.geometry_type is GeometryType.TUBE
                and self.maximum_number_of_cuts > 0)

    @property
    def is_derived(self) -> bool:
        """True when this item is a segment produced by an earlier cut."""
        return self.parent_item_id is not None or self.generation > 0

    @property
    def effective_minimum_segment_mm(self) -> int:
        """Smallest segment length the planner may create for this pipe.

        Falls back to the diameter when no explicit floor is given: a segment
        shorter than its own diameter is not a pipe any packer should reason
        about as a tube, and it protects the arithmetic from degenerate cuts.
        """
        floor = self.minimum_segment_length_mm or self.outer_diameter_mm
        return max(1, int(floor))

    # -- geometry ---------------------------------------------------------- #

    @property
    def is_approximated(self) -> bool:
        """True when the occupied volume is a conservative over-estimate."""
        return self.geometry_type in APPROXIMATED_GEOMETRIES

    def size_for_axis(self, axis: Axis) -> Vec3:
        """Bounding-box size with the item's length along ``axis``."""
        length, diameter = self.length_mm, self.outer_diameter_mm
        if axis is Axis.X:
            return Vec3(length, diameter, diameter)
        if axis is Axis.Y:
            return Vec3(diameter, length, diameter)
        return Vec3(diameter, diameter, length)

    @property
    def occupied_volume_mm3(self) -> int:
        """Bounding-box volume — what a container actually loses to this item."""
        return self.length_mm * self.outer_diameter_mm * self.outer_diameter_mm

    @property
    def material_volume_mm3(self) -> float:
        """Volume of metal present. NOT a substitute for occupied volume."""
        if self.geometry_type is not GeometryType.TUBE:
            # No exact model for the approximated classes. profile_fill_ratio is
            # the generator's declared solid fraction of the enclosing box.
            return float(self.occupied_volume_mm3) * self.profile_fill_ratio
        inner = self.inner_diameter_mm or 0
        return ((math.pi / 4.0)
                * (self.outer_diameter_mm ** 2 - inner ** 2)
                * self.length_mm)

    @property
    def void_fraction_pct(self) -> float:
        """How much of the bounding box is air. This is why packing matters."""
        if self.occupied_volume_mm3 == 0:
            return 0.0
        return 100.0 * (1.0 - self.material_volume_mm3 / self.occupied_volume_mm3)

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "geometry_type": self.geometry_type.value,
            "length_mm": self.length_mm,
            "outer_diameter_mm": self.outer_diameter_mm,
            "inner_diameter_mm": self.inner_diameter_mm,
            "material": self.material,
            "segregation_group": self.segregation_group,
            "weight_kg": round(self.weight_kg, 3),
            "source_position": self.source_position.to_dict(),
            "priority": self.priority,
            "status": self.status.value,
            "dose_class": self.dose_class,
            "dose_class_source": Source.SIMULATED.value,
            "permitted_axes": [a.value for a in self.permitted_axes],
            "injected": self.injected,
            "material_volume_mm3": round(self.material_volume_mm3, 1),
            "occupied_volume_mm3": self.occupied_volume_mm3,
            "is_approximated": self.is_approximated,
            "profile_fill_ratio": self.profile_fill_ratio,
            # ADDITIVE. A document written before CAD-backed items existed has
            # neither key, and `from_dict` defaults to `generated` — so an old
            # scenario keeps its exact behaviour.
            "geometry_source": self.geometry_source,
            "model_id": self.model_id,
            "cut_allowed": self.cut_allowed,
            "minimum_segment_length_mm": self.minimum_segment_length_mm,
            "maximum_number_of_cuts": self.maximum_number_of_cuts,
            "protected_end_length_mm": self.protected_end_length_mm,
            "parent_item_id": self.parent_item_id,
            "generation": self.generation,
            "cut_history": list(self.cut_history),
            "derived_item_ids": list(self.derived_item_ids),
            "is_cuttable": self.is_cuttable,
            "is_derived": self.is_derived,
            # Absent-as-None rather than omitted: a consumer can then tell
            # "generated item" from "observed item whose provenance was lost".
            "observation": (self.observation.to_dict()
                            if self.observation else None),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WasteItem":
        return WasteItem(
            item_id=d["item_id"],
            length_mm=d["length_mm"],
            outer_diameter_mm=d["outer_diameter_mm"],
            geometry_type=GeometryType(d.get("geometry_type", "tube")),
            inner_diameter_mm=d.get("inner_diameter_mm"),
            material=d.get("material", "carbon_steel"),
            segregation_group=d.get("segregation_group", "A"),
            weight_kg=d.get("weight_kg", 0.0),
            source_position=Vec3.from_dict(d.get("source_position")),
            priority=int(d.get("priority", 0)),
            status=ItemStatus(d.get("status", "pending")),
            dose_class=d.get("dose_class"),
            permitted_axes=tuple(Axis(a) for a in d.get("permitted_axes", ["x", "y", "z"])),
            injected=bool(d.get("injected", False)),
            profile_fill_ratio=float(d.get("profile_fill_ratio", 1.0)),
            geometry_source=str(d.get("geometry_source",
                                      GEOMETRY_SOURCE_GENERATED)),
            model_id=str(d.get("model_id", "")),
            cut_allowed=bool(d.get("cut_allowed", False)),
            minimum_segment_length_mm=d.get("minimum_segment_length_mm"),
            maximum_number_of_cuts=int(d.get("maximum_number_of_cuts", 0)),
            protected_end_length_mm=int(d.get("protected_end_length_mm", 0)),
            parent_item_id=d.get("parent_item_id"),
            generation=int(d.get("generation", 0)),
            cut_history=list(d.get("cut_history", [])),
            derived_item_ids=list(d.get("derived_item_ids", [])),
            observation=(PhysicalObservation.from_dict(d["observation"])
                         if d.get("observation") else None),
        )


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #


@dataclass
class Container:
    """A standard rectangular waste container, given by its INNER dimensions."""

    container_id: str
    inner_width_mm: int
    inner_depth_mm: int
    inner_height_mm: int
    max_payload_kg: float = 1000.0
    allowed_segregation_groups: Tuple[str, ...] = ()   # empty == accepts all
    status: ContainerStatus = ContainerStatus.AVAILABLE
    placements: List["Placement"] = field(default_factory=list)
    #: Z heights (mm) at which a rigid shelf plate / dunnage sheet is installed.
    #: Anything resting on a plate is fully supported regardless of what is below
    #: it. This is how level-based industrial packing actually stands up, and it
    #: is what the arrival-order shelf baseline uses. The plates are modelled as
    #: zero-thickness, which FAVOURS the baseline: a real plate would consume
    #: height the baseline is not charged for here, so any margin the optimizer
    #: shows over it is understated rather than inflated.
    shelf_levels_mm: Tuple[int, ...] = ()
    #: When true, an unrestricted container LOCKS to the segregation group of the
    #: first item placed in it. This is how segregated waste packaging actually
    #: works — a box becomes "the stainless box" the moment stainless goes in —
    #: and it turns segregation from a decoration into a binding constraint that
    #: the validator can check independently of the packer that produced it.
    segregation_locking: bool = False

    def __post_init__(self) -> None:
        _require_id("container_id", self.container_id)
        self.inner_width_mm = _require_positive_int("inner_width_mm", self.inner_width_mm)
        self.inner_depth_mm = _require_positive_int("inner_depth_mm", self.inner_depth_mm)
        self.inner_height_mm = _require_positive_int("inner_height_mm", self.inner_height_mm)
        self.max_payload_kg = _require_non_negative("max_payload_kg", self.max_payload_kg)
        self.status = ContainerStatus(self.status)
        self.allowed_segregation_groups = tuple(self.allowed_segregation_groups)
        self.shelf_levels_mm = tuple(sorted({int(z) for z in self.shelf_levels_mm}))

    @property
    def inner_size(self) -> Vec3:
        return Vec3(self.inner_width_mm, self.inner_depth_mm, self.inner_height_mm)

    @property
    def capacity_mm3(self) -> int:
        return self.inner_width_mm * self.inner_depth_mm * self.inner_height_mm

    def accepts_group(self, group: str) -> bool:
        """An empty allow-list means the container is unrestricted."""
        return (not self.allowed_segregation_groups
                or group in self.allowed_segregation_groups)

    def lock_to_group(self, group: str) -> None:
        """Fix an unrestricted locking container to ``group``. Idempotent."""
        if self.segregation_locking and not self.allowed_segregation_groups:
            self.allowed_segregation_groups = (group,)

    @property
    def is_usable(self) -> bool:
        return self.status is not ContainerStatus.UNAVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "container_id": self.container_id,
            "inner_width_mm": self.inner_width_mm,
            "inner_depth_mm": self.inner_depth_mm,
            "inner_height_mm": self.inner_height_mm,
            "capacity_mm3": self.capacity_mm3,
            "max_payload_kg": self.max_payload_kg,
            "allowed_segregation_groups": list(self.allowed_segregation_groups),
            "shelf_levels_mm": list(self.shelf_levels_mm),
            "segregation_locking": self.segregation_locking,
            "status": self.status.value,
            "placements": [p.to_dict() for p in self.placements],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Container":
        c = Container(
            container_id=d["container_id"],
            inner_width_mm=d["inner_width_mm"],
            inner_depth_mm=d["inner_depth_mm"],
            inner_height_mm=d["inner_height_mm"],
            max_payload_kg=d.get("max_payload_kg", 1000.0),
            allowed_segregation_groups=tuple(d.get("allowed_segregation_groups", ())),
            shelf_levels_mm=tuple(d.get("shelf_levels_mm", ())),
            segregation_locking=bool(d.get("segregation_locking", False)),
            status=ContainerStatus(d.get("status", "available")),
        )
        c.placements = [Placement.from_dict(p) for p in d.get("placements", [])]
        return c

    def respec(self, container_id: str) -> "Container":
        """An empty container with the same specification but a new id."""
        return Container(
            container_id=container_id,
            inner_width_mm=self.inner_width_mm,
            inner_depth_mm=self.inner_depth_mm,
            inner_height_mm=self.inner_height_mm,
            max_payload_kg=self.max_payload_kg,
            allowed_segregation_groups=self.allowed_segregation_groups,
            shelf_levels_mm=self.shelf_levels_mm,
            segregation_locking=self.segregation_locking,
            status=ContainerStatus.AVAILABLE,
        )


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #


@dataclass
class Placement:
    """One item positioned inside one container."""

    item_id: str
    container_id: str
    position: Vec3                      # min corner of the bounding box, mm
    axis: Axis                          # orientation: which way the length points
    size: Vec3                          # bounding-box size at that orientation
    placement_order: int = 0            # execution sequence within the plan
    validation_status: ValidationStatus = ValidationStatus.PENDING
    clearance_mm: int = 0               # min gap to nearest neighbour or wall
    score_contribution: float = 0.0     # this placement's share of the objective
    executed: bool = False

    def __post_init__(self) -> None:
        _require_id("item_id", self.item_id)
        _require_id("container_id", self.container_id)
        self.axis = Axis(self.axis)
        self.validation_status = ValidationStatus(self.validation_status)

    @property
    def box(self) -> Box:
        return Box(self.position, self.size)

    @property
    def occupied_volume_mm3(self) -> int:
        return self.box.volume_mm3

    @property
    def top_z_mm(self) -> int:
        return self.position.z + self.size.z

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "container_id": self.container_id,
            "position": self.position.to_dict(),
            "axis": self.axis.value,
            "size": self.size.to_dict(),
            "occupied_bounds": self.box.to_dict(),
            "placement_order": self.placement_order,
            "validation_status": self.validation_status.value,
            "clearance_mm": self.clearance_mm,
            "score_contribution": round(self.score_contribution, 6),
            "executed": self.executed,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Placement":
        return Placement(
            item_id=d["item_id"],
            container_id=d["container_id"],
            position=Vec3.from_dict(d["position"]),
            axis=Axis(d["axis"]),
            size=Vec3.from_dict(d["size"]),
            placement_order=int(d.get("placement_order", 0)),
            validation_status=ValidationStatus(d.get("validation_status", "pending")),
            clearance_mm=int(d.get("clearance_mm", 0)),
            score_contribution=float(d.get("score_contribution", 0.0)),
            executed=bool(d.get("executed", False)),
        )


# --------------------------------------------------------------------------- #
# PackingPlan
# --------------------------------------------------------------------------- #


@dataclass
class PackingPlan:
    """The output of one packing algorithm over one scenario."""

    plan_id: str
    scenario_id: str
    algorithm: str
    strategy: Strategy = Strategy.MAX_DENSITY
    containers: List[Container] = field(default_factory=list)
    placements: List[Placement] = field(default_factory=list)
    unplaced_item_ids: List[str] = field(default_factory=list)
    computation_time_ms: float = 0.0
    constraint_violations: List[str] = field(default_factory=list)
    approval_state: ApprovalState = ApprovalState.PENDING
    objective_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id("plan_id", self.plan_id)
        self.approval_state = ApprovalState(self.approval_state)
        self.strategy = Strategy(self.strategy)

    # -- derived quantities ------------------------------------------------- #

    @property
    def containers_used(self) -> List[Container]:
        """Containers holding at least one placement, in plan order.

        Container-count KPIs use this, not ``len(self.containers)``: an algorithm
        that opens a container and packs nothing into it must be neither charged
        nor credited for it.
        """
        used_ids = {p.container_id for p in self.placements}
        return [c for c in self.containers if c.container_id in used_ids]

    @property
    def containers_required(self) -> int:
        return len(self.containers_used)

    @property
    def occupied_volume_mm3(self) -> int:
        return sum(p.occupied_volume_mm3 for p in self.placements)

    @property
    def required_capacity_mm3(self) -> int:
        """Container capacity this plan consumes: sum over containers actually used.

        This is the denominator that *changes* between algorithms, and therefore
        the only honest basis for a volume-requirement-reduction claim.
        """
        return sum(c.capacity_mm3 for c in self.containers_used)

    @property
    def utilization_pct(self) -> float:
        cap = self.required_capacity_mm3
        return 0.0 if cap == 0 else 100.0 * self.occupied_volume_mm3 / cap

    @property
    def unused_capacity_mm3(self) -> int:
        return max(0, self.required_capacity_mm3 - self.occupied_volume_mm3)

    def placements_for(self, container_id: str) -> List[Placement]:
        return [p for p in self.placements if p.container_id == container_id]

    def placement_for_item(self, item_id: str) -> Optional[Placement]:
        for p in self.placements:
            if p.item_id == item_id:
                return p
        return None

    def container(self, container_id: str) -> Optional[Container]:
        for c in self.containers:
            if c.container_id == container_id:
                return c
        return None

    @property
    def is_valid(self) -> bool:
        """Every placement validated AND no recorded violation.

        An empty plan is not "valid" — it is vacuous, and treating it as valid
        would let a total failure pass an acceptance check.
        """
        return (not self.constraint_violations
                and bool(self.placements)
                and all(p.validation_status is ValidationStatus.VALID
                        for p in self.placements))

    @property
    def ordered_placements(self) -> List[Placement]:
        return sorted(self.placements, key=lambda p: p.placement_order)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "scenario_id": self.scenario_id,
            "algorithm": self.algorithm,
            "strategy": self.strategy.value,
            "containers": [c.to_dict() for c in self.containers],
            "containers_required": self.containers_required,
            "placements": [p.to_dict() for p in self.placements],
            "unplaced_item_ids": list(self.unplaced_item_ids),
            "occupied_volume_mm3": self.occupied_volume_mm3,
            "required_capacity_mm3": self.required_capacity_mm3,
            "utilization_pct": round(self.utilization_pct, 3),
            "unused_capacity_mm3": self.unused_capacity_mm3,
            "computation_time_ms": round(self.computation_time_ms, 3),
            "constraint_violations": list(self.constraint_violations),
            "approval_state": self.approval_state.value,
            "objective_score": round(self.objective_score, 6),
            "is_valid": self.is_valid,
            "details": dict(self.details),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PackingPlan":
        return PackingPlan(
            plan_id=d["plan_id"],
            scenario_id=d.get("scenario_id", "unknown"),
            algorithm=d["algorithm"],
            strategy=Strategy(d.get("strategy", "max_density")),
            containers=[Container.from_dict(c) for c in d.get("containers", [])],
            placements=[Placement.from_dict(p) for p in d.get("placements", [])],
            unplaced_item_ids=list(d.get("unplaced_item_ids", [])),
            computation_time_ms=float(d.get("computation_time_ms", 0.0)),
            constraint_violations=list(d.get("constraint_violations", [])),
            approval_state=ApprovalState(d.get("approval_state", "pending")),
            objective_score=float(d.get("objective_score", 0.0)),
            details=dict(d.get("details", {})),
        )

    def summary(self) -> Dict[str, Any]:
        """Compact form for topics and dashboard tiles (no per-placement data)."""
        return {
            "plan_id": self.plan_id,
            "algorithm": self.algorithm,
            "strategy": self.strategy.value,
            "containers_required": self.containers_required,
            "placed": len(self.placements),
            "unplaced": len(self.unplaced_item_ids),
            "utilization_pct": round(self.utilization_pct, 2),
            "required_capacity_mm3": self.required_capacity_mm3,
            "computation_time_ms": round(self.computation_time_ms, 3),
            "objective_score": round(self.objective_score, 4),
            "approval_state": self.approval_state.value,
            "is_valid": self.is_valid,
        }


# --------------------------------------------------------------------------- #
# Scenario
# --------------------------------------------------------------------------- #


@dataclass
class Scenario:
    """A reproducible packaging problem: items + container spec + seed."""

    scenario_id: str
    preset: str
    seed: int
    items: List[WasteItem] = field(default_factory=list)
    container_template: Optional[Container] = None
    max_containers: int = 8
    description: str = ""
    dynamic_events: List[Dict[str, Any]] = field(default_factory=list)
    curated: bool = False           # True == hand-built demonstration dataset

    def __post_init__(self) -> None:
        _require_id("scenario_id", self.scenario_id)

    @property
    def total_occupied_volume_mm3(self) -> int:
        return sum(i.occupied_volume_mm3 for i in self.items)

    @property
    def total_material_volume_mm3(self) -> float:
        return sum(i.material_volume_mm3 for i in self.items)

    @property
    def total_weight_kg(self) -> float:
        return sum(i.weight_kg for i in self.items)

    @property
    def segregation_groups(self) -> List[str]:
        return sorted({i.segregation_group for i in self.items})

    @property
    def has_approximated_items(self) -> bool:
        return any(i.is_approximated for i in self.items)

    def item(self, item_id: str) -> Optional[WasteItem]:
        for i in self.items:
            if i.item_id == item_id:
                return i
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "preset": self.preset,
            "seed": self.seed,
            "description": self.description,
            "curated": self.curated,
            "max_containers": self.max_containers,
            "container_template": (self.container_template.to_dict()
                                   if self.container_template else None),
            "items": [i.to_dict() for i in self.items],
            "dynamic_events": list(self.dynamic_events),
            "totals": {
                "items": len(self.items),
                "occupied_volume_mm3": self.total_occupied_volume_mm3,
                "material_volume_mm3": round(self.total_material_volume_mm3, 1),
                "weight_kg": round(self.total_weight_kg, 3),
                "segregation_groups": self.segregation_groups,
                "has_approximated_items": self.has_approximated_items,
            },
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Scenario":
        tpl = d.get("container_template")
        return Scenario(
            scenario_id=d["scenario_id"],
            preset=d.get("preset", "custom"),
            seed=int(d.get("seed", 0)),
            items=[WasteItem.from_dict(i) for i in d.get("items", [])],
            container_template=Container.from_dict(tpl) if tpl else None,
            max_containers=int(d.get("max_containers", 8)),
            description=d.get("description", ""),
            dynamic_events=list(d.get("dynamic_events", [])),
            curated=bool(d.get("curated", False)),
        )

    def csv_rows(self) -> Tuple[Sequence[str], List[Sequence[Any]]]:
        """(header, rows) for the flat CSV export of the generated items."""
        header = ["item_id", "geometry_type", "length_mm", "outer_diameter_mm",
                  "inner_diameter_mm", "material", "segregation_group",
                  "weight_kg", "priority", "dose_class_simulated",
                  "material_volume_mm3", "occupied_volume_mm3", "is_approximated"]
        rows: List[Sequence[Any]] = []
        for i in self.items:
            rows.append([i.item_id, i.geometry_type.value, i.length_mm,
                         i.outer_diameter_mm, i.inner_diameter_mm or "",
                         i.material, i.segregation_group, round(i.weight_kg, 3),
                         i.priority, i.dose_class or "",
                         round(i.material_volume_mm3, 1), i.occupied_volume_mm3,
                         int(i.is_approximated)])
        return header, rows


__all__ = [
    "SCHEMA_VERSION", "GeometryType", "APPROXIMATED_GEOMETRIES", "ItemStatus",
    "ContainerStatus", "ValidationStatus", "ApprovalState", "Axis", "Source",
    "Strategy", "DomainError", "Vec3", "Box", "PhysicalObservation", "WasteItem",
    "Container", "Placement", "PackingPlan", "Scenario",
]
