"""KPI computation, with provenance attached to every single number.

The WISEPACK proposal states four acceptance KPIs, and this module keeps them
rigorously apart from what the demonstrator actually measured:

    KPI1  Vision Detection Rate     > 85%   (proposal target)
    KPI2  Pick Success Rate         > 80%   (proposal target)
    KPI3  End-to-End Success Rate   > 80%   (proposal target)
    KPI4  Volume Reduction          > 50%   (proposal target)

A target is not a result. In this demonstrator:

  * KPI1 and KPI2 have NO real counterpart — there is no perception model and no
    robot. The simulator produces numbers for them, and those numbers are
    labelled ``simulated``: they describe the simulator's configured failure
    rate, nothing else. Quoting them as evidence of detection or grasp
    performance would be fabrication.
  * KPI3 is likewise simulated end to end.
  * KPI4 IS genuinely measured here, because container count and utilization are
    computed by real algorithms on real geometry. It is the one proposal KPI
    this demonstrator can speak to.

THE DENOMINATOR RULE. ``volume_requirement_reduction_pct`` compares REQUIRED
CONTAINER CAPACITY, defined as (containers used x capacity per container):

    reduction = 100 * (baseline_capacity - optimized_capacity) / baseline_capacity

Total pipe *material* volume must never be the denominator. It is identical for
both algorithms — nothing an optimizer does changes how much steel exists — so
using it produces a number that cannot distinguish a good plan from a bad one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .domain import PackingPlan, Scenario, Source
from .events import ActionLog

# --------------------------------------------------------------------------- #
# Proposal targets — constants, never results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProposalTarget:
    key: str
    label: str
    threshold_pct: float
    comparison: str = ">"
    measurable_here: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "threshold_pct": self.threshold_pct,
            "comparison": self.comparison,
            "source": Source.TARGET.value,
            "measurable_in_this_demo": self.measurable_here,
            "note": self.note,
        }


PROPOSAL_TARGETS: Sequence[ProposalTarget] = (
    ProposalTarget(
        "KPI1", "Vision detection rate", 85.0, measurable_here=False,
        note=("Not measured in either perception mode, for two different "
              "reasons. With simulated perception there is no vision model at "
              "all and the figure reflects the configured simulator "
              "confidence. With the real HARMONY camera detector there IS a "
              "vision model, but a detection RATE needs a ground-truth trial — "
              "how many objects were actually on the table — and detector "
              "confidence is a different quantity that must never be "
              "substituted for it.")),
    ProposalTarget(
        "KPI2", "Pick success rate", 80.0, measurable_here=False,
        note=("No robot and no grasp planner. The simulated figure reflects the "
              "configured, seeded failure-injection rate.")),
    ProposalTarget(
        "KPI3", "End-to-end success rate", 80.0, measurable_here=False,
        note="Simulated: derived from the simulated pick outcomes."),
    ProposalTarget(
        "KPI4", "Volume reduction vs baseline packing", 50.0, measurable_here=True,
        note=("Genuinely measured here: container counts and utilization come "
              "from two real algorithms run on real geometry and checked by an "
              "independent validator.")),
)


# --------------------------------------------------------------------------- #
# Metric with provenance
# --------------------------------------------------------------------------- #

#: Sentinel for a KPI that has not been measured yet. The dashboard renders this
#: as "not measured" — never as 0, which would read as a measured failure.
NOT_MEASURED = None


@dataclass
class Metric:
    """One reported number and where it came from."""

    key: str
    value: Optional[float]
    unit: str = ""
    source: Source = Source.MEASURED
    note: str = ""

    @property
    def measured(self) -> bool:
        return self.value is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": (round(self.value, 4) if isinstance(self.value, float)
                      else self.value),
            "unit": self.unit,
            "source": self.source.value,
            "measured": self.measured,
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# Execution statistics (simulated)
# --------------------------------------------------------------------------- #


@dataclass
class ExecutionStats:
    """Counters the robot simulator and orchestrator accumulate.

    Everything here is SIMULATED except ``replans`` and ``operator_interventions``,
    which are real counts of real control-flow events inside the demo.
    """

    pick_attempts: int = 0
    pick_successes: int = 0
    cycles_attempted: int = 0
    cycles_completed: int = 0
    placements_validated: int = 0
    replans: int = 0
    replan_causes: List[str] = field(default_factory=list)
    operator_interventions: int = 0
    detected_items: int = 0
    detectable_items: int = 0
    fiware_events_logged: int = 0
    fiware_events_confirmed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pick_attempts": self.pick_attempts,
            "pick_successes": self.pick_successes,
            "cycles_attempted": self.cycles_attempted,
            "cycles_completed": self.cycles_completed,
            "placements_validated": self.placements_validated,
            "replans": self.replans,
            "replan_causes": list(self.replan_causes),
            "operator_interventions": self.operator_interventions,
            "detected_items": self.detected_items,
            "detectable_items": self.detectable_items,
            "fiware_events_logged": self.fiware_events_logged,
            "fiware_events_confirmed": self.fiware_events_confirmed,
        }


# --------------------------------------------------------------------------- #
# KPI computation
# --------------------------------------------------------------------------- #


def _pct(numerator: float, denominator: float) -> Optional[float]:
    """Percentage, or None when the denominator makes it meaningless.

    Zero-denominator protection returns NOT_MEASURED rather than 0.0. A 0%
    success rate and "no attempts yet" are completely different statements and
    collapsing them is how a dashboard ends up claiming a failure that never
    happened.
    """
    if denominator <= 0:
        return None
    return 100.0 * numerator / denominator


def volume_requirement_reduction_pct(baseline: PackingPlan,
                                     optimized: PackingPlan) -> Optional[float]:
    """The KPI4 formula. See the module docstring for the denominator rule."""
    base_capacity = baseline.required_capacity_mm3
    if base_capacity <= 0:
        return None
    return 100.0 * (base_capacity - optimized.required_capacity_mm3) / base_capacity


def unused_capacity_reduction_pct(baseline: PackingPlan,
                                  optimized: PackingPlan) -> Optional[float]:
    """How much of the baseline's wasted space the optimizer recovered."""
    base_unused = baseline.unused_capacity_mm3
    if base_unused <= 0:
        return None
    return 100.0 * (base_unused - optimized.unused_capacity_mm3) / base_unused


def packing_density_gain_pct(baseline: PackingPlan,
                             optimized: PackingPlan) -> Optional[float]:
    """Relative improvement in utilization (not a difference of percentages).

    Reported as a *relative* gain because 30% -> 60% is a doubling, and calling
    that "+30%" understates it while calling it "+100 percentage points" would
    be wrong. The absolute percentage points are reported separately.
    """
    if baseline.utilization_pct <= 0:
        return None
    return 100.0 * (optimized.utilization_pct - baseline.utilization_pct) \
        / baseline.utilization_pct


@dataclass
class KPIReport:
    """Every KPI for one run, each carrying its provenance."""

    scenario_id: str
    run_id: str
    metrics: Dict[str, Metric] = field(default_factory=dict)
    targets: Sequence[ProposalTarget] = PROPOSAL_TARGETS
    curated_scenario: bool = False
    approximated_items: int = 0

    def add(self, key: str, value: Optional[float], unit: str = "",
            source: Source = Source.MEASURED, note: str = "") -> None:
        self.metrics[key] = Metric(key, value, unit, source, note)

    def value(self, key: str) -> Optional[float]:
        m = self.metrics.get(key)
        return m.value if m else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "curated_scenario": self.curated_scenario,
            "approximated_items": self.approximated_items,
            "metrics": {k: m.to_dict() for k, m in sorted(self.metrics.items())},
            "proposal_targets": [t.to_dict() for t in self.targets],
            "target_assessment": self.assess_targets(),
        }

    def assess_targets(self) -> List[Dict[str, Any]]:
        """Compare measured values to proposal targets, only where legitimate.

        A target whose ``measurable_in_this_demo`` is false is never marked
        met or missed — it is marked ``not_applicable``, because the demo has
        nothing to say about it and a green tick would be a lie.
        """
        mapping = {
            "KPI1": "detection_rate_pct",
            "KPI2": "simulated_pick_success_rate_pct",
            "KPI3": "simulated_end_to_end_success_rate_pct",
            "KPI4": "volume_requirement_reduction_pct",
        }
        out: List[Dict[str, Any]] = []
        for target in self.targets:
            metric = self.metrics.get(mapping[target.key])
            row: Dict[str, Any] = {
                "key": target.key,
                "label": target.label,
                "target_pct": target.threshold_pct,
                "measured_value": metric.value if metric else None,
                "measured_source": metric.source.value if metric else None,
            }
            if not target.measurable_here:
                row["status"] = "not_applicable"
                row["explanation"] = target.note
            elif metric is None or metric.value is None:
                row["status"] = "not_measured"
                row["explanation"] = "no measurement recorded in this run"
            elif metric.value > target.threshold_pct:
                row["status"] = "met"
                row["explanation"] = (
                    f"measured {metric.value:.1f}% > target "
                    f"{target.threshold_pct:.0f}%")
            else:
                row["status"] = "not_met"
                row["explanation"] = (
                    f"measured {metric.value:.1f}%, target is "
                    f"{target.comparison}{target.threshold_pct:.0f}%")
            out.append(row)
        return out

    def csv_rows(self):
        header = ("key", "value", "unit", "source", "measured", "note")
        rows = [(m.key,
                 "" if m.value is None else round(m.value, 4),
                 m.unit, m.source.value, int(m.measured), m.note)
                for _, m in sorted(self.metrics.items())]
        return header, rows


def compute_kpis(scenario: Scenario, baseline: PackingPlan,
                 optimized: PackingPlan, selected: PackingPlan,
                 stats: Optional[ExecutionStats] = None,
                 action_log: Optional[ActionLog] = None,
                 latency_p50_ms: Optional[float] = None,
                 run_id: str = "run",
                 perception_source: str = "sim",
                 perception_mean_confidence: Optional[float] = None) -> KPIReport:
    """Build the full KPI report for one run.

    ``latency_p50_ms`` comes from the DDS->FIWARE latency artefact when one
    exists. It is not synthesised: with no artefact the metric stays
    NOT_MEASURED and the dashboard shows "not measured".

    ``perception_source`` changes only the PROVENANCE AND WORDING of the
    detection metrics, never their arithmetic. A real detector does not turn
    ``detection_rate_pct`` into a measurement — see the KPI1 note above and §12.
    """
    stats = stats or ExecutionStats()
    report = KPIReport(
        scenario_id=scenario.scenario_id,
        run_id=run_id,
        curated_scenario=scenario.curated,
        approximated_items=sum(1 for i in scenario.items if i.is_approximated),
    )
    M, S = Source.MEASURED, Source.SIMULATED

    # -- inventory ---------------------------------------------------------- #
    report.add("items_generated", float(len(scenario.items)), "items", M)
    report.add("items_packed", float(len(selected.placements)), "items", M)
    report.add("unplaced_items", float(len(selected.unplaced_item_ids)), "items", M)

    # -- container demand (the core measured result) ------------------------ #
    report.add("containers_baseline", float(baseline.containers_required),
               "containers", M)
    report.add("containers_optimized", float(optimized.containers_required),
               "containers", M)
    report.add("container_utilization_baseline_pct", baseline.utilization_pct,
               "%", M)
    report.add("container_utilization_optimized_pct", optimized.utilization_pct,
               "%", M)
    report.add("packing_density_gain_pct",
               packing_density_gain_pct(baseline, optimized), "%", M,
               "relative gain in utilization, not a difference of percentages")
    report.add("utilization_gain_points",
               optimized.utilization_pct - baseline.utilization_pct,
               "percentage points", M)
    report.add("unused_capacity_reduction_pct",
               unused_capacity_reduction_pct(baseline, optimized), "%", M)
    report.add("volume_requirement_reduction_pct",
               volume_requirement_reduction_pct(baseline, optimized), "%", M,
               "denominator is required container capacity, not material volume")
    report.add("baseline_required_capacity_m3",
               baseline.required_capacity_mm3 / 1e9, "m3", M)
    report.add("optimized_required_capacity_m3",
               optimized.required_capacity_mm3 / 1e9, "m3", M)

    # Material volume is reported for context and explicitly flagged as the
    # quantity that does NOT change between algorithms.
    report.add("total_material_volume_m3",
               scenario.total_material_volume_mm3 / 1e9, "m3", M,
               "identical for both algorithms — never a reduction denominator")
    report.add("total_occupied_volume_m3",
               scenario.total_occupied_volume_mm3 / 1e9, "m3", M,
               "sum of item bounding boxes")

    # -- optimizer performance ---------------------------------------------- #
    report.add("optimization_time_ms", optimized.computation_time_ms, "ms", M)
    report.add("baseline_time_ms", baseline.computation_time_ms, "ms", M)
    report.add("placements_validated", float(stats.placements_validated
                                             or len(selected.placements)),
               "placements", M)

    # -- workflow ------------------------------------------------------------ #
    report.add("replans", float(stats.replans), "events", M)
    report.add("operator_interventions", float(stats.operator_interventions),
               "events", M)

    # -- simulated execution ------------------------------------------------- #
    report.add("simulated_pick_attempts", float(stats.pick_attempts), "attempts", S)
    report.add("simulated_pick_success_rate_pct",
               _pct(stats.pick_successes, stats.pick_attempts), "%", S,
               "simulator failure injection, NOT a robot measurement")
    report.add("simulated_end_to_end_success_rate_pct",
               _pct(stats.cycles_completed, stats.cycles_attempted), "%", S,
               "simulated packaging cycles, NOT a robot measurement")
    # -- perception ---------------------------------------------------------- #
    #
    # THE ONE PLACE §12 IS ENFORCED. `detection_rate_pct` is
    # detected / ACTUALLY-PRESENT, and only the simulator knows the denominator
    # because only the simulator generated the objects. A real camera does not:
    # `WorkflowEngine._scan_physical` leaves `detectable_items` at 0 on purpose,
    # `_pct` returns NOT_MEASURED for a zero denominator, and the dashboard
    # renders "not measured" rather than a number.
    #
    # A confidence of 0.94 must NEVER become "vision detection rate = 94%".
    # Those are different quantities: one is the detector's own certainty about
    # objects it did find, the other needs ground truth about objects it might
    # have missed. The mean confidence is still published — under a name that
    # cannot be mistaken for a rate, with SIMULATED/MEASURED provenance that
    # matches where it came from.
    physical_perception = str(perception_source or "sim") != "sim"
    report.add("detection_rate_pct",
               _pct(stats.detected_items, stats.detectable_items), "%",
               M if physical_perception else S,
               ("not measured — a real detector is active but no ground-truth "
                "trial has been run, so the number of objects actually present "
                "is unknown. Detector confidence is NOT a detection rate."
                if physical_perception else
                "simulated perception confidence, NOT a vision measurement"))
    report.add("detected_objects", float(stats.detected_items), "objects",
               M if physical_perception else S,
               ("objects reported by the real detector in the current "
                "observation batch" if physical_perception else
                "objects reported by the perception simulator"))
    if physical_perception:
        report.add("physical_detector_mean_confidence",
                   perception_mean_confidence, "confidence", M,
                   "mean of the detector's own per-object confidences. NOT a "
                   "detection rate, NOT an accuracy, and never a success metric")

    # -- integration --------------------------------------------------------- #
    logged = (action_log.count if action_log else stats.fiware_events_logged)
    report.add("action_events_published", float(logged), "events", M)
    report.add("fiware_events_logged", float(stats.fiware_events_logged), "events", M)
    report.add("fiware_events_confirmed", float(stats.fiware_events_confirmed),
               "events", M,
               "confirmed readable back from Orion-LD")
    report.add("dds_to_fiware_latency_ms", latency_p50_ms, "ms", M,
               "p50 from the latency artefact; absent until measure_dds_fiware_latency runs")

    return report


def compare_strategies(results: Dict[str, PackingPlan]) -> List[Dict[str, Any]]:
    """Side-by-side rows for the strategy-comparison view.

    Adapted from HARVEST's scenario-comparison idea: run the same input through
    several configurations and show the trade-off rather than declaring a single
    winner. Here the "scenarios" are the three operator-selectable strategies.
    """
    rows: List[Dict[str, Any]] = []
    for name, plan in sorted(results.items()):
        rows.append({
            "strategy": name,
            "algorithm": plan.algorithm,
            "containers": plan.containers_required,
            "utilization_pct": round(plan.utilization_pct, 2),
            "unplaced": len(plan.unplaced_item_ids),
            "required_capacity_m3": round(plan.required_capacity_mm3 / 1e9, 4),
            "computation_time_ms": round(plan.computation_time_ms, 1),
            "objective_score": round(plan.objective_score, 4),
            "valid": plan.is_valid,
        })
    return rows


def cut_recommendations(scenario: Scenario, plan: PackingPlan,
                        top_n: int = 3) -> List[Dict[str, Any]]:
    """Advisory: which item, if cut shorter, would best fill the leftover space.

    The proposal describes an analytics layer that produces "predictive
    recommendations, for example, whether a component should be cut into shorter
    segments to fill an otherwise unused container cavity". This is a first,
    deliberately simple version of that: for each unplaced or last-container
    item, report the shortest length that would have fitted the largest free
    axis-aligned gap.

    ADVISORY ONLY. Nothing consumes this, no KPI depends on it, and it does not
    modify any plan — cutting radioactive components is a decision with
    consequences far outside a packing heuristic.
    """
    recs: List[Dict[str, Any]] = []
    used = plan.containers_used
    if not used:
        return recs

    for item_id in plan.unplaced_item_ids[:top_n * 2]:
        item = scenario.item(item_id)
        if item is None:
            continue
        best: Optional[Dict[str, Any]] = None
        for container in used:
            free_z = container.inner_height_mm - max(
                (p.top_z_mm for p in plan.placements_for(container.container_id)),
                default=0)
            if free_z < item.outer_diameter_mm:
                continue
            # Longest free run along X at that height, approximated by the
            # container width. Deliberately coarse: it is a hint, not a plan.
            candidate = {
                "item_id": item_id,
                "current_length_mm": item.length_mm,
                "suggested_max_length_mm": container.inner_width_mm,
                "container_id": container.container_id,
                "free_height_mm": free_z,
                "segments": max(2, math.ceil(item.length_mm / container.inner_width_mm)),
                "advisory": True,
                "source": Source.MEASURED.value,
            }
            if best is None or candidate["free_height_mm"] > best["free_height_mm"]:
                best = candidate
        if best:
            recs.append(best)
    return recs[:top_n]


__all__ = [
    "PROPOSAL_TARGETS", "ProposalTarget", "NOT_MEASURED", "Metric",
    "ExecutionStats", "KPIReport", "compute_kpis",
    "volume_requirement_reduction_pct", "unused_capacity_reduction_pct",
    "packing_density_gain_pct", "compare_strategies", "cut_recommendations",
]
