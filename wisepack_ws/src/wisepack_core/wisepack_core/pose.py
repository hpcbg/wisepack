"""6-DoF pose, symmetry, and the frames a pose is allowed to live in.

WHY THIS MODULE EXISTS SEPARATELY FROM `domain.py`
--------------------------------------------------
The planar detector measures three numbers on a calibrated plane: x, y and a
yaw. A model-based RGB-D estimator measures a full rigid transform. Those are
not the same measurement, and the difference is not cosmetic — a full
orientation has failure modes (unnormalised quaternions, unobservable rotations
about a symmetry axis, camera-frame results mislabelled as world-frame) that
three numbers simply do not have.

So the 3-D representation and its rules live here, with no dependency on the
packing domain, and `PhysicalObservation` composes them. Nothing in this file
knows what a detector, a camera or a container is.

THE QUATERNION IS AUTHORITATIVE
-------------------------------
Euler angles are derived, published for humans, and never used as the source of
truth. Two reasons, both encountered rather than theoretical:

  * roll/pitch/yaw needs a convention (which axes, which order, intrinsic or
    extrinsic) and the convention is exactly what gets lost when a number is
    copied between two systems;
  * near a singularity — a cylinder standing on end is one — the Euler triple
    changes discontinuously while the physical pose does not, so a consumer that
    differences Euler angles measures noise that is not in the sensor.

A PLANAR OBSERVATION IS A SPECIAL CASE, NOT A DIFFERENT KIND
------------------------------------------------------------
`Orientation.from_yaw_deg()` builds the quaternion a planar yaw means, so every
observation carries a full orientation and consumers need only one code path.
The planar path keeps reporting yaw exactly as before; it simply also has a
quaternion now, and `pose_dof` says which of the six degrees of freedom were
actually measured rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #

#: The WISEPACK work-area frame. Planar observations are already expressed here,
#: because the ArUco homography maps directly onto it.
WORKAREA_FRAME = "wisepack_workarea"

#: The optical frame of the colour camera: +Z forward along the optical axis,
#: +X right, +Y down — the OpenCV/ROS optical convention every RGB-D estimator
#: reports in.
#:
#: A MODEL-BASED ESTIMATOR NATURALLY PRODUCES A POSE IN THIS FRAME, and calling
#: that `wisepack_workarea` would be a lie with real consequences: the numbers
#: would be metres from a lens rather than millimetres from the corner of a
#: table, and a scene synchronizer would place objects inside the camera.
CAMERA_OPTICAL_FRAME = "camera_color_optical_frame"

#: Frames a pose may claim without an explicit registration. Anything else is a
#: caller's own frame and is carried verbatim — this list exists so the two that
#: WISEPACK reasons about are named in one place, not to restrict anyone.
KNOWN_FRAMES = (WORKAREA_FRAME, CAMERA_OPTICAL_FRAME)


class PoseError(ValueError):
    """Raised when a pose, orientation or transform cannot be trusted."""


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #

#: How far a quaternion's norm may drift before it is rejected outright rather
#: than normalised. A float32 round trip loses ~1e-7; anything near 1e-3 is a
#: different bug — an uninitialised value, a scale factor, or three components
#: of a four-component quantity.
QUATERNION_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class Orientation:
    """A rotation, as a unit quaternion in (x, y, z, w) order.

    (x, y, z, w) rather than (w, x, y, z): it is what ROS `geometry_msgs/Quaternion`,
    `scipy.spatial.transform.Rotation.as_quat()` and every estimator in this
    pipeline use. Picking the other order would put a silent transposition
    between WISEPACK and everything it talks to.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def __post_init__(self) -> None:
        for name in ("x", "y", "z", "w"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PoseError(f"quaternion component {name} must be a number, "
                                f"got {value!r}")
            if value != value or abs(value) == float("inf"):
                raise PoseError(f"quaternion component {name} is not finite")
            object.__setattr__(self, name, float(value))

        norm = self.norm
        if norm < QUATERNION_NORM_TOLERANCE:
            # A zero quaternion is the classic "nobody filled this in" value. It
            # has no rotation to normalise towards, so it cannot be repaired.
            raise PoseError(
                "quaternion has (near) zero norm — this is an unset value, not "
                "a rotation")
        if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
            raise PoseError(
                f"quaternion norm is {norm:.6f}, not 1. A rotation must be a "
                "unit quaternion; a value this far off is a different quantity, "
                "not float drift. Use Orientation.normalized() if the producer "
                "is known to emit unnormalised output.")
        # Float32 drift IS repaired, silently and deliberately: every producer
        # in this pipeline computes in float32 somewhere.
        if norm != 1.0:
            for name, value in zip(("x", "y", "z", "w"), self.as_tuple()):
                object.__setattr__(self, name, value / norm)

    # -- construction ------------------------------------------------------ #

    @staticmethod
    def normalized(x: float, y: float, z: float, w: float) -> "Orientation":
        """Build from a possibly-unnormalised quaternion, rejecting only zero."""
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if not norm or norm != norm or norm == float("inf"):
            raise PoseError("quaternion has zero or non-finite norm")
        return Orientation(x / norm, y / norm, z / norm, w / norm)

    @staticmethod
    def identity() -> "Orientation":
        return Orientation(0.0, 0.0, 0.0, 1.0)

    @staticmethod
    def from_yaw_deg(yaw_deg: float) -> "Orientation":
        """The rotation a PLANAR yaw means: about +Z of the work-area frame.

        This is what makes a planar observation and a 6-DoF observation the same
        type rather than two. The yaw is not thrown away — it is expressed in the
        representation every consumer can use.
        """
        half = math.radians(float(yaw_deg)) / 2.0
        return Orientation(0.0, 0.0, math.sin(half), math.cos(half))

    @staticmethod
    def from_matrix(rows: Sequence[Sequence[float]]) -> "Orientation":
        """From a 3x3 rotation matrix (or the top-left of a 4x4).

        Shepperd's method: pick the largest diagonal term to divide by, so the
        result stays conditioned for every rotation including the 180-degree
        cases where the naive trace formula divides by ~0.
        """
        m = [[float(rows[i][j]) for j in range(3)] for i in range(3)]
        trace = m[0][0] + m[1][1] + m[2][2]
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2][1] - m[1][2]) / s
            y = (m[0][2] - m[2][0]) / s
            z = (m[1][0] - m[0][1]) / s
        elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
            w = (m[2][1] - m[1][2]) / s
            x = 0.25 * s
            y = (m[0][1] + m[1][0]) / s
            z = (m[0][2] + m[2][0]) / s
        elif m[1][1] > m[2][2]:
            s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
            w = (m[0][2] - m[2][0]) / s
            x = (m[0][1] + m[1][0]) / s
            y = 0.25 * s
            z = (m[1][2] + m[2][1]) / s
        else:
            s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
            w = (m[1][0] - m[0][1]) / s
            x = (m[0][2] + m[2][0]) / s
            y = (m[1][2] + m[2][1]) / s
            z = 0.25 * s
        return Orientation.normalized(x, y, z, w)

    # -- use --------------------------------------------------------------- #

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)

    @property
    def norm(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2 + self.w ** 2)

    def to_matrix(self) -> List[List[float]]:
        x, y, z, w = self.as_tuple()
        return [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]

    def multiply(self, other: "Orientation") -> "Orientation":
        """`self ∘ other` — apply `other` first, then `self`."""
        x1, y1, z1, w1 = self.as_tuple()
        x2, y2, z2, w2 = other.as_tuple()
        return Orientation.normalized(
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)

    def rotate(self, vector: Sequence[float]) -> Tuple[float, float, float]:
        m = self.to_matrix()
        v = [float(vector[0]), float(vector[1]), float(vector[2])]
        return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))

    def axis(self, which: str = "z") -> Tuple[float, float, float]:
        """The object's own axis, expressed in the parent frame."""
        index = {"x": 0, "y": 1, "z": 2}[which.lower()]
        basis = [0.0, 0.0, 0.0]
        basis[index] = 1.0
        return self.rotate(basis)

    # -- derived, for humans ONLY ------------------------------------------ #

    def rpy_deg(self) -> Tuple[float, float, float]:
        """(roll, pitch, yaw) in degrees, extrinsic XYZ. DIAGNOSTIC ONLY.

        Published so an operator can read a pose at a glance. Never consumed:
        see the module docstring for why differencing these is a mistake.
        """
        x, y, z, w = self.as_tuple()
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        # Clamped rather than allowed to raise: |sinp| slightly over 1 is float
        # noise at gimbal lock, and asin() would throw on a perfectly good pose.
        pitch = (math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1
                 else math.asin(sinp))
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

    @property
    def yaw_deg(self) -> float:
        return self.rpy_deg()[2]

    def angle_to_deg(self, other: "Orientation") -> float:
        """Smallest rotation angle between two orientations, in degrees.

        The metric a repeatability report should use — it is continuous and has
        no convention, unlike a difference of Euler triples.
        """
        dot = abs(sum(a * b for a, b in zip(self.as_tuple(), other.as_tuple())))
        return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))

    def to_dict(self) -> Dict[str, float]:
        return {"x": round(self.x, 9), "y": round(self.y, 9),
                "z": round(self.z, 9), "w": round(self.w, 9)}

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "Orientation":
        if not d:
            return Orientation.identity()
        return Orientation.normalized(float(d.get("x", 0.0)),
                                      float(d.get("y", 0.0)),
                                      float(d.get("z", 0.0)),
                                      float(d.get("w", 1.0)))


# --------------------------------------------------------------------------- #
# Symmetry
# --------------------------------------------------------------------------- #


class SymmetryType(str, Enum):
    """What an object's shape makes UNOBSERVABLE about its orientation."""

    #: No symmetry: every rotational DoF is observable in principle.
    NONE = "none"
    #: Continuous rotational symmetry about one axis — a cylinder, a pipe, a
    #: bottle. The rotation ABOUT that axis is not observable at all.
    AXIAL = "axial"
    #: Discrete N-fold symmetry about an axis — a hex nut, a square flange. The
    #: rotation about the axis is observable only modulo 360/N degrees.
    DISCRETE = "discrete"
    #: Symmetric under a 180-degree flip about an axis — a plain cylinder with
    #: two identical ends. Reported separately because it is an ambiguity of
    #: DIRECTION, not of angle.
    FLIP = "flip"


@dataclass(frozen=True)
class Symmetry:
    """Which rotational degrees of freedom this object's shape hides.

    THE POINT IS NOT TIDINESS. A model-based estimator returns a full
    orientation for a cylinder even though the rotation about its axis fits the
    data equally well at every value. Publishing that number as a measurement
    puts a fabricated quantity into an audit trail, and a planner that grasps
    according to it is acting on noise. So the shape's symmetry is DECLARED, the
    ambiguous component is named, and the honest pose is the canonical one.
    """

    type: SymmetryType = SymmetryType.NONE
    #: The symmetry axis in the OBJECT's own frame. Only meaningful for
    #: AXIAL/DISCRETE/FLIP.
    axis: str = "z"
    #: For DISCRETE: how many equivalent orientations there are per turn.
    fold: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", SymmetryType(self.type))
        axis = str(self.axis or "z").lower()
        if axis not in ("x", "y", "z"):
            raise PoseError(f"symmetry axis must be x, y or z, got {self.axis!r}")
        object.__setattr__(self, "axis", axis)
        if self.type is SymmetryType.DISCRETE:
            if not self.fold or int(self.fold) < 2:
                raise PoseError(
                    "a discrete symmetry needs fold >= 2 (a hex nut is fold 6)")
            object.__setattr__(self, "fold", int(self.fold))
        elif self.fold is not None:
            object.__setattr__(self, "fold", int(self.fold))

    @property
    def rotation_observable(self) -> bool:
        """False when rotation about the axis carries no information at all."""
        return self.type is not SymmetryType.AXIAL

    @property
    def ambiguous_dof(self) -> List[str]:
        """Which rotational DoFs this shape makes unobservable, named.

        Carried on every observation so a consumer never has to infer it from
        an object type it does not understand.
        """
        if self.type is SymmetryType.NONE:
            return []
        if self.type is SymmetryType.AXIAL:
            return [f"rotation_about_{self.axis}"]
        if self.type is SymmetryType.FLIP:
            return [f"rotation_about_{self.axis}_modulo_180deg"]
        return [f"rotation_about_{self.axis}_modulo_{360.0 / self.fold:.4g}deg"]

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "axis": self.axis, "fold": self.fold,
                "rotation_observable": self.rotation_observable,
                "ambiguous_dof": self.ambiguous_dof}

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "Symmetry":
        if not d:
            return Symmetry()
        return Symmetry(type=SymmetryType(d.get("type", "none")),
                        axis=str(d.get("axis", "z")),
                        fold=d.get("fold"))


def canonicalize(orientation: Orientation, symmetry: Symmetry) -> Orientation:
    """Collapse symmetry-equivalent orientations onto one representative.

    WHAT THIS IS FOR. Two estimates of the same physical cylinder can differ by
    an arbitrary rotation about its axis and by a 180-degree flip, and both are
    equally correct. Comparing them raw makes a perfectly repeatable sensor look
    like a wildly unstable one — the error is in the comparison, not the sensor.

    The representative is chosen deterministically:

      * AXIAL — remove the spin about the symmetry axis entirely, by rebuilding
        the rotation from the axis direction alone. What remains is exactly what
        was observed: where the axis points.
      * FLIP — pick the direction whose axis has a non-negative component along
        the parent frame's +Z, so the two ends of an end-for-end-identical
        object stop alternating.
      * DISCRETE — reduce the angle about the axis into [0, 360/fold).
      * NONE — unchanged.

    THE RAW ORIENTATION IS NOT DESTROYED. Callers keep it (see
    `PhysicalObservation.orientation_raw`); this is what planning consumes.
    """
    if symmetry.type is SymmetryType.NONE:
        return orientation

    axis_vector = orientation.axis(symmetry.axis)

    if symmetry.type is SymmetryType.AXIAL:
        return _orientation_from_axis(axis_vector, symmetry.axis)

    if symmetry.type is SymmetryType.FLIP:
        # The flip ambiguity is about WHICH WAY the axis points, so the
        # correcting rotation must be about an axis PERPENDICULAR to it — a
        # half turn about the symmetry axis itself leaves the axis direction
        # exactly where it was and corrects nothing.
        if axis_vector[2] < 0:
            perpendicular = {"x": "y", "y": "z", "z": "x"}[symmetry.axis]
            flip = _orientation_about(perpendicular, 180.0)
            return orientation.multiply(flip)
        return orientation

    # DISCRETE: wrap the spin into one sector.
    step = 360.0 / float(symmetry.fold)
    spin = _spin_about_axis(orientation, symmetry.axis)
    wrapped = spin - step * math.floor(spin / step)
    correction = _orientation_about(symmetry.axis, wrapped - spin)
    return orientation.multiply(correction)


def _orientation_about(axis: str, degrees: float) -> Orientation:
    half = math.radians(degrees) / 2.0
    s, c = math.sin(half), math.cos(half)
    return {"x": Orientation(s, 0.0, 0.0, c),
            "y": Orientation(0.0, s, 0.0, c),
            "z": Orientation(0.0, 0.0, s, c)}[axis]


def _spin_about_axis(orientation: Orientation, axis: str) -> float:
    """The rotation about the object's own `axis`, in degrees (swing/twist).

    Twist part of a swing-twist decomposition: project the quaternion's vector
    part onto the axis and rebuild. Standard, and the only decomposition that
    isolates the component a symmetry makes meaningless.
    """
    index = {"x": 0, "y": 1, "z": 2}[axis]
    vector = [orientation.x, orientation.y, orientation.z]
    twist_v = [0.0, 0.0, 0.0]
    twist_v[index] = vector[index]
    twist = Orientation.normalized(twist_v[0], twist_v[1], twist_v[2],
                                   orientation.w) if any(twist_v) or orientation.w \
        else Orientation.identity()
    angle = 2.0 * math.atan2(
        math.copysign(math.sqrt(sum(v * v for v in twist_v)), vector[index]),
        twist.w)
    return math.degrees(angle)


def _orientation_from_axis(direction: Sequence[float], axis: str) -> Orientation:
    """The rotation that takes the object's `axis` onto `direction`, no spin.

    The minimal rotation between two unit vectors, which is precisely "point the
    axis there and add nothing else" — the only orientation content an axially
    symmetric object actually carries.
    """
    index = {"x": 0, "y": 1, "z": 2}[axis]
    source = [0.0, 0.0, 0.0]
    source[index] = 1.0
    target = list(float(v) for v in direction)
    norm = math.sqrt(sum(v * v for v in target))
    if norm < 1e-9:
        return Orientation.identity()
    target = [v / norm for v in target]

    dot = sum(a * b for a, b in zip(source, target))
    if dot > 1.0 - 1e-9:
        return Orientation.identity()
    if dot < -1.0 + 1e-9:
        # Antiparallel: any perpendicular axis is a valid 180-degree rotation.
        perpendicular = [1.0, 0.0, 0.0] if index != 0 else [0.0, 1.0, 0.0]
        cross = _cross(source, perpendicular)
        return Orientation.normalized(cross[0], cross[1], cross[2], 0.0)

    cross = _cross(source, target)
    return Orientation.normalized(cross[0], cross[1], cross[2], 1.0 + dot)


def _cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


# --------------------------------------------------------------------------- #
# Rigid transform between frames
# --------------------------------------------------------------------------- #


@dataclass
class RigidTransform:
    """`T_parent_from_child` — a full SE(3) transform, with its provenance.

    NOT A HOMOGRAPHY, AND THE DISTINCTION IS THE WHOLE POINT. The planar
    detector's ArUco homography maps image pixels onto ONE plane. It is a
    perfectly good instrument for x/y/yaw on that plane and it is NOT a
    camera-to-world rigid transform: it cannot place a point off the plane, and
    it carries no depth. Reusing it to place a 6-DoF pose would produce numbers
    that look plausible and are wrong by however far the object sticks up.

    So a 3-D transform is its own object, obtained its own way, and carries the
    evidence for itself: which method produced it, when, from how many
    observations, and with what residual. A transform that cannot say where it
    came from is not usable for a physical action.
    """

    parent_frame: str
    child_frame: str
    translation_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Orientation = field(default_factory=Orientation.identity)
    #: How it was obtained: `charuco_solvepnp`, `fixed_mount`, `simulation_tf`, …
    method: str = ""
    #: Bumped whenever the transform is re-measured, so a consumer can tell two
    #: poses expressed under different calibrations apart.
    revision: str = ""
    measured_at: str = ""
    #: Mean reprojection error in pixels, when the method produces one. `None`
    #: means "this method does not measure its own error", which is different
    #: from zero and must not render as a perfect calibration.
    reprojection_error_px: Optional[float] = None
    sample_count: Optional[int] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.parent_frame or not self.child_frame:
            raise PoseError("a transform needs both a parent and a child frame")
        if self.parent_frame == self.child_frame:
            raise PoseError(
                f"a transform from {self.child_frame!r} to itself is not a "
                "transform")
        values = tuple(float(v) for v in self.translation_mm)
        if len(values) != 3:
            raise PoseError("translation_mm must have three components")
        for value in values:
            if value != value or abs(value) == float("inf"):
                raise PoseError("translation_mm must be finite")
        self.translation_mm = values
        if isinstance(self.rotation, dict):
            self.rotation = Orientation.from_dict(self.rotation)

    @property
    def valid(self) -> bool:
        """Usable for a physical claim.

        An identity transform with no method is the DEFAULT, not a measurement,
        and treating it as one is how a camera-frame pose ends up labelled as a
        world pose. So a transform counts as valid only when something actually
        produced it.
        """
        return bool(self.method)

    def apply_to_position(self, position_mm: Sequence[float]
                          ) -> Tuple[float, float, float]:
        rotated = self.rotation.rotate(position_mm)
        return tuple(rotated[i] + self.translation_mm[i] for i in range(3))

    def apply_to_orientation(self, orientation: Orientation) -> Orientation:
        return self.rotation.multiply(orientation)

    def inverse(self) -> "RigidTransform":
        inverse_rotation = Orientation(-self.rotation.x, -self.rotation.y,
                                       -self.rotation.z, self.rotation.w)
        moved = inverse_rotation.rotate(self.translation_mm)
        return RigidTransform(
            parent_frame=self.child_frame, child_frame=self.parent_frame,
            translation_mm=tuple(-v for v in moved),
            rotation=inverse_rotation, method=self.method,
            revision=self.revision, measured_at=self.measured_at,
            reprojection_error_px=self.reprojection_error_px,
            sample_count=self.sample_count, notes=self.notes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "translation_mm": [round(v, 4) for v in self.translation_mm],
            "rotation": self.rotation.to_dict(),
            "method": self.method,
            "revision": self.revision,
            "measured_at": self.measured_at,
            "reprojection_error_px": self.reprojection_error_px,
            "sample_count": self.sample_count,
            "valid": self.valid,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RigidTransform":
        return RigidTransform(
            parent_frame=str(d.get("parent_frame", "")),
            child_frame=str(d.get("child_frame", "")),
            translation_mm=tuple(float(v) for v in
                                 (d.get("translation_mm") or (0.0, 0.0, 0.0))),
            rotation=Orientation.from_dict(d.get("rotation")),
            method=str(d.get("method", "")),
            revision=str(d.get("revision", "")),
            measured_at=str(d.get("measured_at", "")),
            reprojection_error_px=d.get("reprojection_error_px"),
            sample_count=d.get("sample_count"),
            notes=str(d.get("notes", "")))


__all__ = [
    "WORKAREA_FRAME", "CAMERA_OPTICAL_FRAME", "KNOWN_FRAMES", "PoseError",
    "Orientation", "SymmetryType", "Symmetry", "canonicalize", "RigidTransform",
    "QUATERNION_NORM_TOLERANCE",
]
