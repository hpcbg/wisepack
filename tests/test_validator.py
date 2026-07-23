"""The independent validator must catch violations the optimizer cannot produce.

Every test here builds a plan BY HAND containing one specific defect and asserts
the validator names it. That is the point of the module: if the validator only
ever saw optimizer output it would never be exercised on the failure cases, and a
validator that has never rejected anything is not evidence of anything.
"""

from __future__ import annotations

import pytest

from wisepack_core.domain import (
    Axis, Box, Container, ContainerStatus, PackingPlan, Placement, Scenario,
    ValidationStatus, Vec3, WasteItem,
)
from wisepack_core.validator import (
    PlacementValidator, ValidationConfig, _union_area,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def make_item(item_id: str, length: int = 400, diameter: int = 100,
              group: str = "A", weight: float = 10.0) -> WasteItem:
    return WasteItem(item_id=item_id, length_mm=length,
                     outer_diameter_mm=diameter, inner_diameter_mm=diameter - 20,
                     segregation_group=group, weight_kg=weight)


def make_container(container_id: str = "CNT-01", w: int = 1000, d: int = 800,
                   h: int = 600, payload: float = 1000.0,
                   groups=()) -> Container:
    return Container(container_id=container_id, inner_width_mm=w,
                     inner_depth_mm=d, inner_height_mm=h,
                     max_payload_kg=payload, allowed_segregation_groups=groups)


def make_plan(items, placements, container) -> tuple:
    scenario = Scenario(scenario_id="test", preset="mixed_pipes_small", seed=1,
                        items=items, container_template=container)
    plan = PackingPlan(plan_id="plan-test", scenario_id="test",
                       algorithm="handmade", containers=[container],
                       placements=placements)
    return scenario, plan


def place(item: WasteItem, container: Container, x: int, y: int, z: int,
          axis: Axis = Axis.X) -> Placement:
    return Placement(item_id=item.item_id, container_id=container.container_id,
                     position=Vec3(x, y, z), axis=axis,
                     size=item.size_for_axis(axis))


NO_SUPPORT = ValidationConfig(min_support_fraction=0.0)


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #

def test_a_correct_plan_validates():
    container = make_container()
    a, b = make_item("a"), make_item("b")
    scenario, plan = make_plan(
        [a, b], [place(a, container, 0, 0, 0), place(b, container, 400, 0, 0)],
        container)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert report.valid, report.violation_strings
    assert report.placements_valid == 2
    assert all(p.validation_status is ValidationStatus.VALID for p in plan.placements)


def test_boxes_sharing_a_face_do_not_collide():
    """Half-open intervals: flush placement is legal, and must stay legal."""
    container = make_container()
    a, b = make_item("a", length=400), make_item("b", length=400)
    scenario, plan = make_plan(
        [a, b], [place(a, container, 0, 0, 0), place(b, container, 400, 0, 0)],
        container)
    assert PlacementValidator().validate_plan(plan, scenario).valid


# --------------------------------------------------------------------------- #
# H1 — container bounds
# --------------------------------------------------------------------------- #

def test_h1_item_past_the_container_wall_is_rejected():
    container = make_container(w=500)
    a = make_item("a", length=400)
    scenario, plan = make_plan([a], [place(a, container, 200, 0, 0)], container)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert not report.valid
    assert any(v.code == "H1" for v in report.violations)


def test_h1_negative_position_is_rejected():
    container = make_container()
    a = make_item("a")
    scenario, plan = make_plan([a], [place(a, container, -10, 0, 0)], container)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert any(v.code == "H1" for v in report.violations)


# --------------------------------------------------------------------------- #
# H2 — collisions
# --------------------------------------------------------------------------- #

def test_h2_overlapping_items_are_rejected():
    container = make_container()
    a, b = make_item("a"), make_item("b")
    scenario, plan = make_plan(
        [a, b], [place(a, container, 0, 0, 0), place(b, container, 200, 0, 0)],
        container)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert not report.valid
    overlap = [v for v in report.violations if v.code == "H2"]
    assert overlap
    assert {overlap[0].item_id, overlap[0].other_item_id} == {"a", "b"}
    assert all(p.validation_status is ValidationStatus.INVALID
               for p in plan.placements)


def test_h2_clearance_requirement_is_enforced_when_configured():
    container = make_container()
    a, b = make_item("a", length=400), make_item("b", length=400)
    scenario, plan = make_plan(
        [a, b], [place(a, container, 0, 0, 0), place(b, container, 400, 0, 0)],
        container)
    strict = PlacementValidator(
        ValidationConfig(min_support_fraction=0.0, min_clearance_mm=20))
    report = strict.validate_plan(plan, scenario)
    assert not report.valid
    assert any("clearance" in v.message for v in report.violations)


# --------------------------------------------------------------------------- #
# H3 — payload
# --------------------------------------------------------------------------- #

def test_h3_payload_limit_is_enforced():
    container = make_container(payload=15.0)
    a, b = make_item("a", weight=10.0), make_item("b", weight=10.0)
    scenario, plan = make_plan(
        [a, b], [place(a, container, 0, 0, 0), place(b, container, 400, 0, 0)],
        container)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert any(v.code == "H3" for v in report.violations)


# --------------------------------------------------------------------------- #
# H4 — segregation
# --------------------------------------------------------------------------- #

def test_h4_wrong_segregation_group_is_rejected():
    container = make_container(groups=("A",))
    a = make_item("a", group="B")
    scenario, plan = make_plan([a], [place(a, container, 0, 0, 0)], container)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert any(v.code == "H4" for v in report.violations)


def test_h4_empty_allow_list_accepts_any_group():
    container = make_container(groups=())
    a = make_item("a", group="Z")
    scenario, plan = make_plan([a], [place(a, container, 0, 0, 0)], container)
    assert PlacementValidator().validate_plan(plan, scenario).valid


def test_locking_container_rejects_a_second_group():
    """A locked container's allow-list makes mixing an H4 violation."""
    container = make_container()
    container.segregation_locking = True
    container.lock_to_group("A")
    a, b = make_item("a", group="A"), make_item("b", group="B")
    scenario, plan = make_plan(
        [a, b], [place(a, container, 0, 0, 0), place(b, container, 400, 0, 0)],
        container)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert any(v.code == "H4" and v.item_id == "b" for v in report.violations)


# --------------------------------------------------------------------------- #
# H5 / H6 — orientation and recorded size
# --------------------------------------------------------------------------- #

def test_h5_disallowed_orientation_is_rejected():
    container = make_container()
    a = make_item("a")
    a.permitted_axes = (Axis.X,)
    scenario, plan = make_plan([a], [place(a, container, 0, 0, 0, Axis.Y)],
                               container)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert any(v.code == "H5" for v in report.violations)


def test_h6_shrunken_bounding_box_is_caught():
    """The most dangerous bug class: a packer writing a smaller box to make it fit."""
    container = make_container(w=300)
    a = make_item("a", length=400, diameter=100)
    cheat = Placement(item_id="a", container_id=container.container_id,
                      position=Vec3(0, 0, 0), axis=Axis.X,
                      size=Vec3(200, 100, 100))       # lies about its length
    scenario, plan = make_plan([a], [cheat], container)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert not report.valid
    assert any(v.code == "H6" for v in report.violations)


def test_orientation_transform_is_correct():
    item = make_item("a", length=400, diameter=100)
    assert item.size_for_axis(Axis.X).as_tuple() == (400, 100, 100)
    assert item.size_for_axis(Axis.Y).as_tuple() == (100, 400, 100)
    assert item.size_for_axis(Axis.Z).as_tuple() == (100, 100, 400)
    for axis in Axis:
        assert item.size_for_axis(axis).x * item.size_for_axis(axis).y \
               * item.size_for_axis(axis).z == item.occupied_volume_mm3


# --------------------------------------------------------------------------- #
# H7 / H8 — bookkeeping and availability
# --------------------------------------------------------------------------- #

def test_h7_double_placement_of_one_item_is_rejected():
    container = make_container()
    a = make_item("a")
    scenario, plan = make_plan(
        [a], [place(a, container, 0, 0, 0), place(a, container, 500, 0, 0)],
        container)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert any(v.code == "H7" for v in report.violations)


def test_h7_unknown_item_is_rejected():
    container = make_container()
    a, ghost = make_item("a"), make_item("ghost")
    scenario, plan = make_plan([a], [place(ghost, container, 0, 0, 0)], container)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert any(v.code == "H7" for v in report.violations)


def test_h8_unavailable_container_may_not_be_used():
    container = make_container()
    container.status = ContainerStatus.UNAVAILABLE
    a = make_item("a")
    scenario, plan = make_plan([a], [place(a, container, 0, 0, 0)], container)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert any(v.code == "H8" for v in report.violations)


# --------------------------------------------------------------------------- #
# H9 — support
# --------------------------------------------------------------------------- #

def test_h9_floating_item_is_rejected():
    container = make_container()
    a = make_item("a")
    scenario, plan = make_plan([a], [place(a, container, 0, 0, 300)], container)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert any(v.code == "H9" for v in report.violations)


def test_h9_item_stacked_on_another_is_accepted():
    container = make_container()
    a = make_item("a", length=400, diameter=100)
    b = make_item("b", length=400, diameter=100)
    scenario, plan = make_plan(
        [a, b], [place(a, container, 0, 0, 0), place(b, container, 0, 0, 100)],
        container)
    assert PlacementValidator().validate_plan(plan, scenario).valid


def test_h9_shelf_plate_supports_a_level():
    """This is what lets the arrival-order shelf baseline stand up at all."""
    container = make_container()
    container.shelf_levels_mm = (300,)
    a = make_item("a")
    scenario, plan = make_plan([a], [place(a, container, 0, 0, 300)], container)
    assert PlacementValidator().validate_plan(plan, scenario).valid


def test_h9_can_be_disabled():
    container = make_container()
    a = make_item("a")
    scenario, plan = make_plan([a], [place(a, container, 0, 0, 300)], container)
    assert PlacementValidator(NO_SUPPORT).validate_plan(plan, scenario).valid


def test_support_union_does_not_double_count_overlapping_supports():
    """Two supports overlapping in XY must not sum to >100% support."""
    # Two identical 100x100 rectangles at the same place: union is 10 000, not 20 000.
    assert _union_area([(0, 0, 100, 100), (0, 0, 100, 100)]) == 10_000
    assert _union_area([(0, 0, 100, 100), (50, 0, 150, 100)]) == 15_000
    assert _union_area([]) == 0


# --------------------------------------------------------------------------- #
# Plan-level bookkeeping
# --------------------------------------------------------------------------- #

def test_validator_refreshes_the_plans_violation_list():
    """A stale violation list would let the orchestrator run an invalid plan."""
    container = make_container()
    a, b = make_item("a"), make_item("b")
    scenario, plan = make_plan(
        [a, b], [place(a, container, 0, 0, 0), place(b, container, 200, 0, 0)],
        container)
    plan.constraint_violations = []
    PlacementValidator().validate_plan(plan, scenario)
    assert plan.constraint_violations
    assert plan.is_valid is False


def test_empty_plan_is_not_valid():
    """Vacuous success would let total failure pass an acceptance check."""
    container = make_container()
    scenario, plan = make_plan([make_item("a")], [], container)
    PlacementValidator().validate_plan(plan, scenario)
    assert plan.is_valid is False


def test_clearance_is_reported_per_item():
    container = make_container()
    a, b = make_item("a", length=400), make_item("b", length=400)
    scenario, plan = make_plan(
        [a, b], [place(a, container, 0, 0, 0), place(b, container, 500, 0, 0)],
        container)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert report.valid
    assert report.clearances_mm["a"] == 100        # 100 mm gap to b
