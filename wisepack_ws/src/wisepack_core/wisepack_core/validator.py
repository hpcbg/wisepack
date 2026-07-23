"""Independent placement validator — the Digital Twin feasibility check.

This module is deliberately written as if the optimizer did not exist. It shares
no candidate-generation code with optimizer.py, keeps no state the optimizer
built, and re-derives every box from the item and the recorded axis. An optimizer
that validates itself with its own collision routine cannot detect a bug in that
routine; this one can, and the tests exercise exactly that by feeding it plans
containing deliberate violations.

The proposal's Digital Twin "continuously evaluates geometric feasibility,
collision avoidance, segregation constraints and packing density". Those first
three are the hard constraints below. Density is a score, not a constraint, and
it lives in optimizer.py — a plan is never rejected for being loose.

HARD CONSTRAINTS (a plan containing any of these is invalid, never merely
penalised):

  H1  every placement lies fully within its container's inner volume
  H2  no two placements in a container overlap
  H3  each container's payload limit is respected
  H4  each item's segregation group is accepted by its container
  H5  each placement's orientation is one the item permits
  H6  the recorded bounding-box size matches the item and axis
  H7  each item is placed at most once, and only items in the scenario
  H8  no NEW placement targets a container marked unavailable
  H9  each placement is supported from below (floor or other items)

H9 is the one that is a modelling choice rather than a physical law: a floating
pipe is geometrically feasible but not physically placeable, and allowing it
would let the optimizer "win" with plans a robot cannot execute. The support
threshold is configurable and reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .domain import (
    Axis, Box, Container, ContainerStatus, PackingPlan, Placement, Scenario,
    ValidationStatus, Vec3, WasteItem,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ValidationConfig:
    """Tunable parts of the feasibility check.

    ``min_support_fraction`` is the fraction of a placement's bottom face that
    must rest on the container floor or on the top faces of other placements.
    0.0 disables the support check entirely (and the report says so).

    ``min_clearance_mm`` is a required *gap* between neighbouring items (walls
    are excluded — hugging a wall is good packing, not a clearance defect). It
    defaults to 0: real handling needs finger clearance, but demanding it by
    default would make the demo's density figures incomparable with the textbook
    bin-packing results a reviewer might have in mind. Raising it is a one-line
    config change and the effect is visible in the KPIs.
    """

    min_support_fraction: float = 0.70
    min_clearance_mm: int = 0
    #: Placements are integer mm, so no geometric tolerance is needed; this only
    #: guards the payload sum, which is float.
    payload_tolerance_kg: float = 1e-6


DEFAULT_VALIDATION = ValidationConfig()


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class Violation:
    """One specific, addressable reason a plan is not executable."""

    code: str                       # H1..H9
    message: str
    item_id: Optional[str] = None
    container_id: Optional[str] = None
    other_item_id: Optional[str] = None

    def __str__(self) -> str:
        where = ""
        if self.container_id:
            where += f" [{self.container_id}]"
        if self.item_id:
            where += f" ({self.item_id})"
        return f"{self.code}{where}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "item_id": self.item_id,
            "container_id": self.container_id,
            "other_item_id": self.other_item_id,
        }


@dataclass
class ValidationReport:
    """Outcome of validating one plan."""

    plan_id: str
    valid: bool
    violations: List[Violation] = field(default_factory=list)
    placements_checked: int = 0
    placements_valid: int = 0
    config: ValidationConfig = DEFAULT_VALIDATION
    #: item_id -> minimum gap in mm to any neighbour or wall (informational)
    clearances_mm: Dict[str, int] = field(default_factory=dict)

    @property
    def violation_strings(self) -> List[str]:
        return [str(v) for v in self.violations]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "valid": self.valid,
            "placements_checked": self.placements_checked,
            "placements_valid": self.placements_valid,
            "violations": [v.to_dict() for v in self.violations],
            "config": {
                "min_support_fraction": self.config.min_support_fraction,
                "min_clearance_mm": self.config.min_clearance_mm,
                "support_check_enabled": self.config.min_support_fraction > 0.0,
            },
        }

    def summary(self) -> str:
        if self.valid:
            return (f"{self.plan_id}: VALID — {self.placements_valid}/"
                    f"{self.placements_checked} placements pass all hard constraints")
        return (f"{self.plan_id}: INVALID — {len(self.violations)} violation(s), "
                f"{self.placements_valid}/{self.placements_checked} placements pass")


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


class PlacementValidator:
    """Re-derives and checks a plan from first principles."""

    #: Sentinel for "no neighbour found yet" when scanning clearances.
    _MAX_GAP = 1 << 30

    def __init__(self, config: ValidationConfig = DEFAULT_VALIDATION) -> None:
        self.config = config

    # -- public API --------------------------------------------------------- #

    def validate_plan(self, plan: PackingPlan, scenario: Scenario,
                      *, mark: bool = True) -> ValidationReport:
        """Validate every placement in ``plan`` against ``scenario``.

        When ``mark`` is true (the default) each Placement's validation_status is
        set, so downstream code and the dashboard can colour individual boxes.
        The plan's own ``constraint_violations`` list is also refreshed — the
        orchestrator refuses to execute a plan with a non-empty list, so leaving
        it stale would be a safety-relevant bug, not a cosmetic one.
        """
        violations: List[Violation] = []
        items_by_id = {i.item_id: i for i in scenario.items}
        containers_by_id = {c.container_id: c for c in plan.containers}
        clearances: Dict[str, int] = {}
        bad_items: set[str] = set()

        # -- H7: each item placed at most once, and known to the scenario ---- #
        seen: Dict[str, str] = {}
        for p in plan.placements:
            if p.item_id not in items_by_id:
                violations.append(Violation(
                    "H7", f"placement references item not in scenario "
                          f"{scenario.scenario_id}",
                    item_id=p.item_id, container_id=p.container_id))
                bad_items.add(p.item_id)
            if p.item_id in seen:
                violations.append(Violation(
                    "H7", f"item placed more than once (also in {seen[p.item_id]})",
                    item_id=p.item_id, container_id=p.container_id))
                bad_items.add(p.item_id)
            else:
                seen[p.item_id] = p.container_id
            if p.container_id not in containers_by_id:
                violations.append(Violation(
                    "H7", "placement references a container not in the plan",
                    item_id=p.item_id, container_id=p.container_id))
                bad_items.add(p.item_id)

        # -- per-container checks -------------------------------------------- #
        for container in plan.containers:
            placements = [p for p in plan.placements
                          if p.container_id == container.container_id]
            if not placements:
                continue

            # H8: an unavailable container must not receive any FURTHER item.
            # Items already executed into it stay where they physically are —
            # a container going out of service does not levitate its contents
            # back out, and flagging them would make every post-event re-plan
            # permanently invalid.
            if container.status is ContainerStatus.UNAVAILABLE:
                for p in placements:
                    if p.executed:
                        continue
                    violations.append(Violation(
                        "H8", "container is marked unavailable",
                        item_id=p.item_id, container_id=container.container_id))
                    bad_items.add(p.item_id)

            boxes: List[Tuple[Placement, Box]] = []
            payload_kg = 0.0

            for p in placements:
                item = items_by_id.get(p.item_id)
                if item is None:
                    continue                     # already reported under H7

                # H5: orientation must be one the item permits.
                if p.axis not in item.permitted_axes:
                    violations.append(Violation(
                        "H5", f"axis {p.axis.value} not in permitted axes "
                              f"{[a.value for a in item.permitted_axes]}",
                        item_id=p.item_id, container_id=container.container_id))
                    bad_items.add(p.item_id)

                # H6: recorded size must match the item at that axis. This is the
                # check that catches an optimizer writing a shrunken box to make
                # a placement fit — the single most dangerous class of bug here.
                expected = item.size_for_axis(p.axis)
                if p.size.as_tuple() != expected.as_tuple():
                    violations.append(Violation(
                        "H6", f"recorded size {p.size.as_tuple()} != geometry "
                              f"{expected.as_tuple()} for axis {p.axis.value}",
                        item_id=p.item_id, container_id=container.container_id))
                    bad_items.add(p.item_id)

                # Re-derive the box from the ITEM, never trusting p.size.
                box = Box(p.position, expected)

                # H1: fully inside the container.
                if not box.within(container.inner_size):
                    violations.append(Violation(
                        "H1", f"box {box.origin.as_tuple()}+{box.size.as_tuple()} "
                              f"exceeds inner volume {container.inner_size.as_tuple()}",
                        item_id=p.item_id, container_id=container.container_id))
                    bad_items.add(p.item_id)

                # H4: segregation compatibility.
                if not container.accepts_group(item.segregation_group):
                    violations.append(Violation(
                        "H4", f"segregation group {item.segregation_group!r} not in "
                              f"container allow-list "
                              f"{list(container.allowed_segregation_groups)}",
                        item_id=p.item_id, container_id=container.container_id))
                    bad_items.add(p.item_id)

                payload_kg += item.weight_kg
                boxes.append((p, box))

            # H2: pairwise overlap, and the clearance report.
            #
            # Clearance is the gap to the nearest NEIGHBOURING ITEM, not to the
            # walls. Every item rests on the floor and most hug a wall, so
            # including walls would report 0 for almost everything and tell us
            # nothing. Touching a wall is desirable; touching another item is
            # what a gripper has to negotiate.
            for i in range(len(boxes)):
                p_i, box_i = boxes[i]
                min_gap = self._MAX_GAP
                for j in range(len(boxes)):
                    if i == j:
                        continue
                    p_j, box_j = boxes[j]
                    if j > i and box_i.overlaps(box_j):
                        violations.append(Violation(
                            "H2", f"overlaps {p_j.item_id}",
                            item_id=p_i.item_id,
                            container_id=container.container_id,
                            other_item_id=p_j.item_id))
                        bad_items.add(p_i.item_id)
                        bad_items.add(p_j.item_id)
                    min_gap = min(min_gap, box_i.gap_to(box_j))
                # A lone item in a container has no neighbour to clear.
                clearances[p_i.item_id] = 0 if min_gap == self._MAX_GAP else min_gap
                if (self.config.min_clearance_mm > 0
                        and min_gap < self.config.min_clearance_mm
                        and len(boxes) > 1):
                    violations.append(Violation(
                        "H2", f"clearance {min_gap} mm < required "
                              f"{self.config.min_clearance_mm} mm",
                        item_id=p_i.item_id, container_id=container.container_id))
                    bad_items.add(p_i.item_id)

            # H3: payload.
            if payload_kg > container.max_payload_kg + self.config.payload_tolerance_kg:
                violations.append(Violation(
                    "H3", f"payload {payload_kg:.2f} kg exceeds limit "
                          f"{container.max_payload_kg:.2f} kg",
                    container_id=container.container_id))
                for p in placements:
                    bad_items.add(p.item_id)

            # H9: support from below.
            if self.config.min_support_fraction > 0.0:
                for p, box in boxes:
                    frac = self._support_fraction(
                        box, [b for q, b in boxes if q is not p],
                        container.shelf_levels_mm)
                    if frac + 1e-9 < self.config.min_support_fraction:
                        violations.append(Violation(
                            "H9", f"support {frac * 100:.1f}% of bottom face < "
                                  f"required {self.config.min_support_fraction * 100:.0f}%",
                            item_id=p.item_id, container_id=container.container_id))
                        bad_items.add(p.item_id)

        # -- mark placements -------------------------------------------------- #
        n_valid = 0
        for p in plan.placements:
            ok = p.item_id not in bad_items
            n_valid += int(ok)
            if mark:
                p.validation_status = (ValidationStatus.VALID if ok
                                       else ValidationStatus.INVALID)
                p.clearance_mm = clearances.get(p.item_id, 0)

        report = ValidationReport(
            plan_id=plan.plan_id,
            valid=not violations,
            violations=violations,
            placements_checked=len(plan.placements),
            placements_valid=n_valid,
            config=self.config,
            clearances_mm=clearances,
        )
        if mark:
            plan.constraint_violations = report.violation_strings
        return report

    # -- single-placement feasibility (used by the optimizer, same rules) ---- #

    def placement_is_feasible(self, item: WasteItem, axis: Axis, position: Vec3,
                              container: Container,
                              placed: Sequence[Tuple[WasteItem, Box]],
                              payload_kg: float) -> bool:
        """Would adding this one placement keep the container feasible?

        The optimizer calls this to filter candidates. It is intentionally the
        SAME rule set as validate_plan, expressed once — but validate_plan does
        not call it, so a bug here produces plans that validate_plan then rejects
        loudly, rather than plans that quietly pass both.
        """
        if not container.is_usable:
            return False
        if axis not in item.permitted_axes:
            return False
        if not container.accepts_group(item.segregation_group):
            return False
        if payload_kg + item.weight_kg > container.max_payload_kg:
            return False

        box = Box(position, item.size_for_axis(axis))
        if not box.within(container.inner_size):
            return False

        other_boxes = [b for _, b in placed]
        for other in other_boxes:
            if box.overlaps(other):
                return False
            if (self.config.min_clearance_mm > 0
                    and box.gap_to(other) < self.config.min_clearance_mm):
                return False

        if self.config.min_support_fraction > 0.0:
            if self._support_fraction(box, other_boxes,
                                      container.shelf_levels_mm) + 1e-9 < \
                    self.config.min_support_fraction:
                return False
        return True

    # -- helpers ------------------------------------------------------------- #

    @staticmethod
    def _support_fraction(box: Box, others: Sequence[Box],
                          shelf_levels_mm: Sequence[int] = ()) -> float:
        """Fraction of ``box``'s bottom face resting on the floor, a shelf or others.

        Supporting boxes are those whose top face is exactly at this box's
        bottom z. Their footprints are unioned by rectangle decomposition rather
        than summed: two supports that overlap each other in XY would otherwise
        double-count and report >100% support for a box that is barely held.

        A box whose bottom sits on a declared shelf plate is fully supported —
        that is what the plate is for, and it is how level-based industrial
        packing physically stands up.
        """
        if box.origin.z == 0:
            return 1.0                              # resting on the container floor
        if box.origin.z in tuple(shelf_levels_mm):
            return 1.0                              # resting on a rigid shelf plate
        footprint = box.size.x * box.size.y
        if footprint == 0:
            return 0.0
        rects: List[Tuple[int, int, int, int]] = []
        for other in others:
            if other.max_corner.z != box.origin.z:
                continue
            x0 = max(box.origin.x, other.origin.x)
            x1 = min(box.max_corner.x, other.max_corner.x)
            y0 = max(box.origin.y, other.origin.y)
            y1 = min(box.max_corner.y, other.max_corner.y)
            if x1 > x0 and y1 > y0:
                rects.append((x0, y0, x1, y1))
        return _union_area(rects) / footprint


def _union_area(rects: Sequence[Tuple[int, int, int, int]]) -> int:
    """Exact union area of axis-aligned rectangles by coordinate compression.

    The rectangle count per support query is small (the items directly under one
    pipe), so the O(n^2) grid is cheaper than an interval tree and has no edge
    cases to get wrong.
    """
    if not rects:
        return 0
    xs = sorted({x for r in rects for x in (r[0], r[2])})
    ys = sorted({y for r in rects for y in (r[1], r[3])})
    area = 0
    for xi in range(len(xs) - 1):
        for yi in range(len(ys) - 1):
            cx0, cx1 = xs[xi], xs[xi + 1]
            cy0, cy1 = ys[yi], ys[yi + 1]
            for rx0, ry0, rx1, ry1 in rects:
                if rx0 <= cx0 and rx1 >= cx1 and ry0 <= cy0 and ry1 >= cy1:
                    area += (cx1 - cx0) * (cy1 - cy0)
                    break
    return area


__all__ = [
    "ValidationConfig", "DEFAULT_VALIDATION", "Violation", "ValidationReport",
    "PlacementValidator",
]
