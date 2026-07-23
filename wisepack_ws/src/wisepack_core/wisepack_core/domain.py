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

SCHEMA_VERSION = "wisepack/1"

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


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
                container counts, utilization, DDS->FIWARE latency).
    SIMULATED — produced by the simulator (pick outcomes, perception confidence,
                dose class). Real inside the demo, not real in the world.
    OPERATOR  — supplied by a human through the dashboard.
    TARGET    — a WISEPACK proposal target (KPI1-KPI4). NOT a result.
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
    "Strategy", "DomainError", "Vec3", "Box", "WasteItem", "Container",
    "Placement", "PackingPlan", "Scenario",
]
