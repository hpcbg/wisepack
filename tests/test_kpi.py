"""KPI arithmetic, zero-protection, and the measured/simulated/target labelling."""

from __future__ import annotations

import pytest

from wisepack_core.domain import Container, PackingPlan, Placement, Scenario, Source, Vec3, WasteItem, Axis
from wisepack_core.generator import build_curated_scenario, build_scenario
from wisepack_core.kpi import (
    PROPOSAL_TARGETS, ExecutionStats, compare_strategies, compute_kpis,
    cut_recommendations, packing_density_gain_pct, unused_capacity_reduction_pct,
    volume_requirement_reduction_pct,
)
from wisepack_core.packing import OptimizerConfig, pack_baseline, pack_optimized, select_plan


# --------------------------------------------------------------------------- #
# Exact known cases
# --------------------------------------------------------------------------- #

def _plan_with(n_containers: int, occupied_mm3: int,
               capacity_each: int = 1_000_000_000) -> PackingPlan:
    """A synthetic plan with exactly ``n_containers`` used and a known fill."""
    containers, placements = [], []
    per_container = occupied_mm3 // max(1, n_containers)
    for n in range(1, n_containers + 1):
        cid = f"CNT-{n:02d}"
        # capacity_each is a cube root-free construction: width carries it all.
        containers.append(Container(container_id=cid, inner_width_mm=capacity_each,
                                    inner_depth_mm=1, inner_height_mm=1))
        placements.append(Placement(item_id=f"item-{n:03d}", container_id=cid,
                                    position=Vec3(0, 0, 0), axis=Axis.X,
                                    size=Vec3(per_container, 1, 1)))
    return PackingPlan(plan_id="p", scenario_id="s", algorithm="synthetic",
                       containers=containers, placements=placements)


def test_volume_requirement_reduction_exact_case():
    """4 containers -> 2 containers is exactly 50%."""
    baseline = _plan_with(4, 400)
    optimized = _plan_with(2, 400)
    assert volume_requirement_reduction_pct(baseline, optimized) == pytest.approx(50.0)


def test_volume_requirement_reduction_3_to_2_is_one_third():
    assert volume_requirement_reduction_pct(_plan_with(3, 300), _plan_with(2, 300)) \
        == pytest.approx(100.0 / 3.0)


def test_no_reduction_when_container_counts_are_equal():
    assert volume_requirement_reduction_pct(_plan_with(2, 200), _plan_with(2, 200)) \
        == pytest.approx(0.0)


def test_reduction_is_negative_when_the_optimizer_does_worse():
    """A regression must show as a negative number, never be clamped to zero."""
    assert volume_requirement_reduction_pct(_plan_with(2, 200), _plan_with(3, 200)) < 0


def test_zero_baseline_is_protected():
    """An empty baseline yields 'not measured', never a division by zero."""
    empty = PackingPlan(plan_id="e", scenario_id="s", algorithm="none")
    assert volume_requirement_reduction_pct(empty, _plan_with(1, 100)) is None
    assert unused_capacity_reduction_pct(empty, _plan_with(1, 100)) is None
    assert packing_density_gain_pct(empty, _plan_with(1, 100)) is None


def test_utilization_is_occupied_over_required_capacity():
    plan = _plan_with(2, 1_000_000_000, capacity_each=1_000_000_000)
    # 2 containers of 1e9 mm3 each, holding 5e8 each == 50%.
    assert plan.required_capacity_mm3 == 2_000_000_000
    assert plan.utilization_pct == pytest.approx(50.0)


def test_packing_density_gain_is_relative_not_a_difference():
    """30% -> 60% is a 100% relative gain, not '+30'."""
    baseline = _plan_with(4, 1_000_000_000, capacity_each=1_000_000_000)
    optimized = _plan_with(2, 1_000_000_000, capacity_each=1_000_000_000)
    assert baseline.utilization_pct == pytest.approx(25.0)
    assert optimized.utilization_pct == pytest.approx(50.0)
    assert packing_density_gain_pct(baseline, optimized) == pytest.approx(100.0)


def test_material_volume_is_not_the_reduction_denominator():
    """The core anti-fudge test.

    Material volume is identical for both algorithms, so if it were the
    denominator the 'reduction' would be 0 no matter how much better the
    optimizer is. Assert the real KPI is non-zero where material volume is
    unchanged.
    """
    scenario = build_curated_scenario()
    baseline = pack_baseline(scenario)
    optimized = pack_optimized(scenario, config=OptimizerConfig(seed=7, restarts=8))
    # Same steel in both plans.
    base_material = sum(scenario.item(p.item_id).material_volume_mm3
                        for p in baseline.placements)
    opt_material = sum(scenario.item(p.item_id).material_volume_mm3
                       for p in optimized.placements)
    assert base_material == pytest.approx(opt_material)
    # ...yet the container requirement genuinely fell.
    assert volume_requirement_reduction_pct(baseline, optimized) > 0


# --------------------------------------------------------------------------- #
# Report assembly and labelling
# --------------------------------------------------------------------------- #

def _run(preset="mixed_pipes_dense", seed=42):
    scenario = build_scenario(preset, seed=seed)
    baseline = pack_baseline(scenario)
    optimized = pack_optimized(scenario, config=OptimizerConfig(seed=seed))
    selected, _ = select_plan(baseline, optimized, scenario)
    return scenario, baseline, optimized, selected


def test_measured_and_simulated_metrics_are_labelled():
    scenario, baseline, optimized, selected = _run()
    stats = ExecutionStats(pick_attempts=10, pick_successes=9,
                           cycles_attempted=9, cycles_completed=9,
                           detected_items=40, detectable_items=40)
    report = compute_kpis(scenario, baseline, optimized, selected, stats)

    assert report.metrics["containers_optimized"].source is Source.MEASURED
    assert report.metrics["volume_requirement_reduction_pct"].source is Source.MEASURED
    assert report.metrics["optimization_time_ms"].source is Source.MEASURED

    assert report.metrics["simulated_pick_success_rate_pct"].source is Source.SIMULATED
    assert report.metrics["simulated_end_to_end_success_rate_pct"].source is Source.SIMULATED
    assert report.metrics["detection_rate_pct"].source is Source.SIMULATED


def test_unmeasured_latency_is_none_not_zero():
    scenario, baseline, optimized, selected = _run()
    report = compute_kpis(scenario, baseline, optimized, selected)
    metric = report.metrics["dds_to_fiware_latency_ms"]
    assert metric.value is None
    assert metric.measured is False


def test_latency_is_reported_when_supplied():
    scenario, baseline, optimized, selected = _run()
    report = compute_kpis(scenario, baseline, optimized, selected,
                          latency_p50_ms=12.5)
    assert report.metrics["dds_to_fiware_latency_ms"].value == pytest.approx(12.5)


def test_rates_with_no_attempts_are_not_measured_rather_than_zero():
    """'no attempts yet' and '0% success' are different statements."""
    scenario, baseline, optimized, selected = _run()
    report = compute_kpis(scenario, baseline, optimized, selected,
                          ExecutionStats())
    assert report.metrics["simulated_pick_success_rate_pct"].value is None
    assert report.metrics["detection_rate_pct"].value is None


# --------------------------------------------------------------------------- #
# Proposal targets
# --------------------------------------------------------------------------- #

def test_unmeasurable_targets_are_never_scored():
    """KPI1-KPI3 have no counterpart here and must not show a pass or a fail."""
    scenario, baseline, optimized, selected = _run()
    stats = ExecutionStats(pick_attempts=10, pick_successes=10,
                           cycles_attempted=10, cycles_completed=10,
                           detected_items=40, detectable_items=40)
    rows = {r["key"]: r for r in
            compute_kpis(scenario, baseline, optimized, selected, stats).assess_targets()}
    for key in ("KPI1", "KPI2", "KPI3"):
        assert rows[key]["status"] == "not_applicable", \
            f"{key} was scored despite 100% simulated success"
    assert rows["KPI4"]["status"] in {"met", "not_met"}


def test_kpi4_status_matches_the_measured_value():
    scenario, baseline, optimized, selected = _run()
    report = compute_kpis(scenario, baseline, optimized, selected)
    row = next(r for r in report.assess_targets() if r["key"] == "KPI4")
    measured = report.value("volume_requirement_reduction_pct")
    assert row["status"] == ("met" if measured > 50.0 else "not_met")


def test_proposal_targets_are_declared_as_targets_not_results():
    for target in PROPOSAL_TARGETS:
        assert target.to_dict()["source"] == Source.TARGET.value
    measurable = [t for t in PROPOSAL_TARGETS if t.measurable_here]
    assert [t.key for t in measurable] == ["KPI4"], \
        "only the volume KPI is genuinely measurable in this demonstrator"


# --------------------------------------------------------------------------- #
# Supporting views
# --------------------------------------------------------------------------- #

def test_strategy_comparison_rows():
    scenario = build_scenario("segregated_materials", seed=42)
    plans = {s: pack_optimized(scenario, config=OptimizerConfig(strategy=s, seed=42,
                                                                restarts=3))
             for s in ("max_density", "retrievability", "segregation")}
    rows = compare_strategies(plans)
    assert len(rows) == 3
    assert {r["strategy"] for r in rows} == set(plans)
    for row in rows:
        assert row["valid"] is True
        assert row["containers"] >= 1


def test_cut_recommendations_are_advisory_only():
    scenario, baseline, optimized, selected = _run("mixed_pipes_dense")
    optimized.unplaced_item_ids = [scenario.items[0].item_id]
    recs = cut_recommendations(scenario, optimized)
    for rec in recs:
        assert rec["advisory"] is True
        assert rec["segments"] >= 2
    # It must not have mutated the plan.
    assert optimized.unplaced_item_ids == [scenario.items[0].item_id]


def test_kpi_csv_export_leaves_unmeasured_cells_empty():
    scenario, baseline, optimized, selected = _run()
    report = compute_kpis(scenario, baseline, optimized, selected)
    header, rows = report.csv_rows()
    assert "source" in header and "measured" in header
    latency = next(r for r in rows if r[0] == "dds_to_fiware_latency_ms")
    assert latency[1] == "", "unmeasured value must be blank, not 0"
