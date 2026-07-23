"""Packing algorithms: validity, determinism, honesty and speed."""

from __future__ import annotations

import time

import pytest

from wisepack_core.domain import (
    Axis, ContainerStatus, Strategy, ValidationStatus,
)
from wisepack_core.generator import build_curated_scenario, build_scenario
from wisepack_core.packing import (
    BASELINE_ALGORITHM, OPTIMIZED_ALGORITHM, OptimizerConfig, STRATEGY_WEIGHTS,
    pack_baseline, pack_optimized, score_plan, select_plan,
)
from wisepack_core.validator import PlacementValidator

GENERATED_PRESETS = ["mixed_pipes_small", "mixed_pipes_dense",
                     "segregated_materials", "late_arrival_replan",
                     "mixed_geometries"]


def optimize(scenario, **kw):
    cfg = OptimizerConfig(seed=scenario.seed, restarts=kw.pop("restarts", 6), **kw)
    return pack_optimized(scenario, config=cfg)


# --------------------------------------------------------------------------- #
# Everything the optimizer returns must validate
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("preset", GENERATED_PRESETS + ["curated_volume_reduction"])
def test_all_optimized_placements_validate(preset):
    scenario = build_scenario(preset, seed=42)
    plan = optimize(scenario)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert report.valid, report.violation_strings
    assert plan.placements, "optimizer placed nothing"
    assert all(p.validation_status is ValidationStatus.VALID
               for p in plan.placements)


@pytest.mark.parametrize("preset", GENERATED_PRESETS + ["curated_volume_reduction"])
def test_all_baseline_placements_validate(preset):
    """The comparison is only meaningful if the baseline is executable too."""
    scenario = build_scenario(preset, seed=42)
    plan = pack_baseline(scenario)
    report = PlacementValidator().validate_plan(plan, scenario)
    assert report.valid, report.violation_strings


@pytest.mark.parametrize("preset", GENERATED_PRESETS)
def test_no_item_is_lost(preset):
    """Every item is either placed exactly once or reported unplaced."""
    scenario = build_scenario(preset, seed=42)
    for plan in (pack_baseline(scenario), optimize(scenario)):
        placed = [p.item_id for p in plan.placements]
        assert len(placed) == len(set(placed)), "an item was placed twice"
        accounted = set(placed) | set(plan.unplaced_item_ids)
        assert accounted == {i.item_id for i in scenario.items}


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("preset", GENERATED_PRESETS)
def test_optimizer_is_reproducible(preset):
    scenario_a = build_scenario(preset, seed=42)
    scenario_b = build_scenario(preset, seed=42)
    a, b = optimize(scenario_a), optimize(scenario_b)
    # Timings differ between runs; the geometry must not.
    assert [p.to_dict() for p in a.ordered_placements] == \
           [p.to_dict() for p in b.ordered_placements]
    assert a.containers_required == b.containers_required
    assert a.objective_score == pytest.approx(b.objective_score)


def test_baseline_is_reproducible():
    a = pack_baseline(build_scenario("mixed_pipes_dense", seed=7))
    b = pack_baseline(build_scenario("mixed_pipes_dense", seed=7))
    assert [p.to_dict() for p in a.ordered_placements] == \
           [p.to_dict() for p in b.ordered_placements]


# --------------------------------------------------------------------------- #
# Multi-container correctness
# --------------------------------------------------------------------------- #

def test_multi_container_results_are_valid_and_contiguous():
    scenario = build_scenario("mixed_pipes_dense", seed=42)
    plan = optimize(scenario)
    assert plan.containers_required >= 2, "preset should need several containers"
    used = plan.containers_used
    # Every used container carries placements and is marked as filling/complete.
    for container in used:
        assert plan.placements_for(container.container_id)
        assert container.status is not ContainerStatus.AVAILABLE
    # Required capacity is exactly n_used x capacity_each.
    assert plan.required_capacity_mm3 == sum(c.capacity_mm3 for c in used)


def test_containers_required_ignores_containers_that_were_opened_but_unused():
    scenario = build_scenario("mixed_pipes_small", seed=42)
    plan = optimize(scenario)
    ghost = scenario.container_template.respec("CNT-99")
    plan.containers.append(ghost)
    assert ghost.container_id not in {c.container_id for c in plan.containers_used}
    assert plan.containers_required == len({p.container_id
                                            for p in plan.placements})


# --------------------------------------------------------------------------- #
# Honesty of the selection
# --------------------------------------------------------------------------- #

def test_optimized_is_never_silently_selected_when_worse():
    """select_plan must keep the better plan and say what it did."""
    scenario = build_scenario("mixed_pipes_dense", seed=42)
    baseline = pack_baseline(scenario)
    # A deliberately crippled "optimized" plan: one restart, no improvement pass.
    weak = pack_optimized(scenario, config=OptimizerConfig(
        seed=42, restarts=1, improve=False, time_budget_ms=50.0))
    selected, reason = select_plan(baseline, weak, scenario)
    base_score = score_plan(baseline, scenario, STRATEGY_WEIGHTS[weak.strategy])
    weak_score = score_plan(weak, scenario, STRATEGY_WEIGHTS[weak.strategy])
    if weak_score < base_score:
        assert selected is baseline
        assert "WORSE" in reason or "baseline" in reason
    else:
        assert selected is weak
    assert reason


def test_selection_prefers_the_plan_that_leaves_fewer_items_unplaced():
    """Density must never be bought by abandoning waste."""
    scenario = build_scenario("mixed_pipes_small", seed=42)
    baseline = pack_baseline(scenario)
    optimized = optimize(scenario)
    optimized.unplaced_item_ids = [scenario.items[0].item_id]
    selected, reason = select_plan(baseline, optimized, scenario)
    assert selected is baseline
    assert "unplaced" in reason


def test_invalid_optimized_plan_is_rejected_in_favour_of_the_baseline():
    scenario = build_scenario("mixed_pipes_small", seed=42)
    baseline = pack_baseline(scenario)
    optimized = optimize(scenario)
    optimized.constraint_violations = ["H2 (item-001): overlaps item-002"]
    selected, reason = select_plan(baseline, optimized, scenario)
    assert selected is baseline
    assert "validation" in reason


def test_both_invalid_is_reported_rather_than_hidden():
    scenario = build_scenario("mixed_pipes_small", seed=42)
    baseline, optimized = pack_baseline(scenario), optimize(scenario)
    baseline.constraint_violations = ["H1: broken"]
    optimized.constraint_violations = ["H1: broken"]
    selected, reason = select_plan(baseline, optimized, scenario)
    assert "BOTH" in reason


# --------------------------------------------------------------------------- #
# The curated scenario
# --------------------------------------------------------------------------- #

def test_curated_scenario_result_is_computed_not_constant():
    """The curated reduction must come out of the algorithms, not a literal.

    The assertion deliberately does NOT pin a specific percentage. It checks the
    structural claim the dataset is built to demonstrate — the optimizer needs
    strictly fewer containers — and recomputes the reduction from the plans.
    """
    scenario = build_curated_scenario()
    baseline = pack_baseline(scenario)
    optimized = optimize(scenario, restarts=8)

    assert PlacementValidator().validate_plan(baseline, scenario).valid
    assert PlacementValidator().validate_plan(optimized, scenario).valid
    assert not baseline.unplaced_item_ids and not optimized.unplaced_item_ids
    assert optimized.containers_required < baseline.containers_required

    reduction = 100.0 * (baseline.required_capacity_mm3
                         - optimized.required_capacity_mm3) \
        / baseline.required_capacity_mm3
    assert reduction > 0
    # Consistency: the reduction must equal the container-count ratio, since
    # every container has the same capacity.
    expected = 100.0 * (baseline.containers_required
                        - optimized.containers_required) \
        / baseline.containers_required
    assert reduction == pytest.approx(expected)


def test_curated_scenario_is_seed_independent():
    a = pack_optimized(build_curated_scenario(seed=1),
                       config=OptimizerConfig(seed=1, restarts=8))
    b = pack_optimized(build_curated_scenario(seed=2),
                       config=OptimizerConfig(seed=2, restarts=8))
    assert a.containers_required == b.containers_required


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("strategy", list(Strategy))
def test_every_strategy_produces_a_valid_plan(strategy):
    """Strategies change preferences, never constraints."""
    scenario = build_scenario("segregated_materials", seed=42)
    plan = pack_optimized(scenario, config=OptimizerConfig(
        strategy=strategy, seed=42, restarts=4))
    assert PlacementValidator().validate_plan(plan, scenario).valid
    assert plan.strategy is strategy


def test_strategies_are_distinguishable_in_their_weights():
    weights = {s: STRATEGY_WEIGHTS[s] for s in Strategy}
    assert weights[Strategy.RETRIEVABILITY].retrievability > \
           weights[Strategy.MAX_DENSITY].retrievability
    assert weights[Strategy.SEGREGATION].segregation_mixing > \
           weights[Strategy.MAX_DENSITY].segregation_mixing


# --------------------------------------------------------------------------- #
# Constraints hold under the optimizer
# --------------------------------------------------------------------------- #

def test_segregation_is_never_violated_by_the_optimizer():
    scenario = build_scenario("segregated_materials", seed=42)
    plan = optimize(scenario)
    for container in plan.containers_used:
        groups = {scenario.item(p.item_id).segregation_group
                  for p in plan.placements_for(container.container_id)}
        assert len(groups) == 1, f"{container.container_id} mixes {groups}"


def test_unavailable_container_is_not_used():
    scenario = build_scenario("mixed_pipes_dense", seed=42)
    plan = optimize(scenario)
    victim = plan.containers_used[0]
    victim.status = ContainerStatus.UNAVAILABLE
    report = PlacementValidator().validate_plan(plan, scenario)
    assert not report.valid
    assert any(v.code == "H8" for v in report.violations)


def test_payload_limit_is_respected():
    scenario = build_scenario("mixed_pipes_dense", seed=42)
    plan = optimize(scenario)
    for container in plan.containers_used:
        total = sum(scenario.item(p.item_id).weight_kg
                    for p in plan.placements_for(container.container_id))
        assert total <= container.max_payload_kg + 1e-6


def test_baseline_uses_a_single_fixed_orientation():
    scenario = build_scenario("mixed_pipes_dense", seed=42)
    plan = pack_baseline(scenario, axis=Axis.X)
    assert {p.axis for p in plan.placements} == {Axis.X}
    assert plan.algorithm == BASELINE_ALGORITHM


def test_optimizer_actually_uses_several_orientations():
    """If it never rotated anything it would not be geometry-aware."""
    scenario = build_scenario("mixed_pipes_dense", seed=42)
    plan = optimize(scenario)
    assert plan.algorithm == OPTIMIZED_ALGORITHM
    assert len({p.axis for p in plan.placements}) > 1


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #

def test_small_input_completes_quickly():
    scenario = build_scenario("mixed_pipes_small", seed=42)
    started = time.perf_counter()
    plan = optimize(scenario)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert elapsed_ms < 5000.0, f"took {elapsed_ms:.0f} ms"
    assert plan.computation_time_ms > 0


def test_time_budget_is_respected():
    """A tight budget returns a complete plan, not a partial one."""
    scenario = build_scenario("mixed_pipes_dense", seed=42)
    plan = pack_optimized(scenario, config=OptimizerConfig(
        seed=42, restarts=8, time_budget_ms=1.0))
    assert PlacementValidator().validate_plan(plan, scenario).valid
    accounted = {p.item_id for p in plan.placements} | set(plan.unplaced_item_ids)
    assert accounted == {i.item_id for i in scenario.items}
    assert plan.details["restarts_run"] >= 1


def test_optimizer_records_its_search_trace():
    scenario = build_scenario("mixed_pipes_small", seed=42)
    plan = optimize(scenario)
    assert plan.details["attempts"], "search trace should be recorded"
    assert plan.details["seed"] == 42
    for attempt in plan.details["attempts"]:
        assert "ordering" in attempt and "containers" in attempt
