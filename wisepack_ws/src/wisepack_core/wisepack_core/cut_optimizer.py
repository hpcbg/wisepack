"""Cut-aware whole-process planner.

The ordinary no-cut packer in packing.py is left completely untouched (brief §2):
this is a *separate layer* that calls it. The method is the deterministic pipeline
of brief §4:

  1. pack the scenario with the ordinary optimizer (the no-cut reference);
  2. find cuttable pipes that are unplaced or that push into the last container;
  3. read residual cavity lengths out of that validated plan;
  4. generate a *bounded* set of cut candidates from those cavities, the
     container inner dimensions and equal division;
  5. build a derived-item scenario per candidate (parent replaced by segments);
  6. re-pack it with the SAME geometry-aware packer, under each strategy;
  7. validate the cut (cut_validator) AND the packing (the packer's own
     validator) independently;
  8. score whole-process alternatives and pick — possibly "no cut".

Whole-process objective (brief §5)
----------------------------------
An extra waste container is an expensive, durable commitment (procurement,
storage footprint, transport, final-repository volume), so container count
dominates. Cutting is charged its real process cost — saw cuts, cutting and
handling time, kerf/material loss and an operational-complexity term. Hard
constraints (boundaries, segregation, minimum segment, maximum cuts) are NEVER
penalties: a candidate that breaks one fails validation and is discarded. The
"no cut" option always sits in the comparison at net benefit 0, so cutting is
recommended only when it strictly out-earns leaving the pipe whole.

Nothing here tunes the weights to flatter the demo — the weights are documented
process economics, and the two shipped scenarios (``cut_avoids_extra_container``,
``cut_not_worthwhile``) fall out of the same untouched formula.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from .cutting import CutConfig, CutProposal, derive_segments
from .cut_validator import validate_no_coexistence, validate_proposal
from .domain import (
    Axis, Container, GeometryType, PackingPlan, Scenario, Source, Strategy,
    WasteItem,
)
from .packing import OptimizerConfig, pack_optimized

#: The three strategies a cut-aware plan is explored under (brief §5).
CUT_STRATEGIES: Tuple[Strategy, ...] = (
    Strategy.MAX_DENSITY, Strategy.RETRIEVABILITY, Strategy.SEGREGATION)


@dataclass(frozen=True)
class CutPlannerConfig:
    """Bounds and the documented cost model. All deterministic given a seed."""

    # -- search bounds (brief §4: never an unbounded continuous space) ------ #
    max_pipes_considered: int = 3
    max_candidates_per_pipe: int = 4
    max_cuts_per_plan: int = 2
    max_cut_aware_plans: int = 12
    cut_increment_mm: int = 50
    time_budget_ms: float = 4000.0
    #: How many pipes a single candidate plan may cut. 1 keeps the demonstrator
    #: bounded and legible; the machinery supports more.
    max_pipes_per_plan: int = 1

    # -- whole-process cost model (documented economics) -------------------- #
    #: Value of avoiding one container. Deliberately large: an extra container is
    #: a durable, costly commitment against which a few saw cuts are cheap.
    container_cost_proxy: float = 1000.0
    #: Reward per percentage-point of utilization gained at equal container
    #: count — only ever a tie-break, never enough to justify a container.
    utilization_weight: float = 2.0
    per_cut_cost: float = 8.0
    cutting_time_cost_per_s: float = 0.10
    handling_time_cost_per_s: float = 0.10
    #: Cost per cm^3 of metal turned to swarf by the kerf.
    kerf_loss_cost_per_cm3: float = 0.5
    #: Fixed operational-complexity charge for any plan that cuts at all.
    complexity_cost_per_cut_plan: float = 15.0
    #: A cut alternative must beat no-cut by at least this net margin to be
    #: recommended, so a rounding-scale gain never flips the recommendation.
    recommendation_margin: float = 1.0

    cut: CutConfig = field(default_factory=CutConfig)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass
class CutAlternative:
    """One packed alternative in the whole-process comparison."""

    label: str
    strategy: Strategy
    is_cut: bool
    plan: PackingPlan
    proposals: List[CutProposal]
    containers: int
    utilization_pct: float
    n_cuts: int
    cutting_time_s: float
    handling_time_s: float
    kerf_loss_cm3: float
    process_cost: float
    value: float
    whole_process_score: float          # value - process_cost, relative to no-cut
    valid: bool
    reason: str = ""

    def summary(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "strategy": self.strategy.value,
            "is_cut": self.is_cut,
            "containers": self.containers,
            "utilization_pct": round(self.utilization_pct, 2),
            "n_cuts": self.n_cuts,
            "cutting_time_s": round(self.cutting_time_s, 1),
            "handling_time_s": round(self.handling_time_s, 1),
            "kerf_loss_cm3": round(self.kerf_loss_cm3, 2),
            "process_cost": round(self.process_cost, 2),
            "value": round(self.value, 2),
            "whole_process_score": round(self.whole_process_score, 2),
            "valid": self.valid,
            "objective_score": round(self.plan.objective_score, 4),
            "proposals": [p.to_dict() for p in self.proposals],
            "reason": self.reason,
        }


@dataclass
class WholeProcessComparison:
    """The full cut-aware verdict for a scenario."""

    scenario_id: str
    no_cut: CutAlternative
    alternatives: List[CutAlternative]
    recommended_label: str
    recommend_cut: bool
    reason: str
    candidates_evaluated: int
    pipes_considered: List[str]
    elapsed_ms: float
    timed_out: bool
    config: CutPlannerConfig

    @property
    def recommended(self) -> CutAlternative:
        for a in [self.no_cut] + self.alternatives:
            if a.label == self.recommended_label:
                return a
        return self.no_cut

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "wisepack/cut-comparison/1",
            "scenario_id": self.scenario_id,
            "no_cut": self.no_cut.summary(),
            "alternatives": [a.summary() for a in self.alternatives],
            "recommended_label": self.recommended_label,
            "recommend_cut": self.recommend_cut,
            "reason": self.reason,
            "candidates_evaluated": self.candidates_evaluated,
            "pipes_considered": list(self.pipes_considered),
            "containers_no_cut": self.no_cut.containers,
            "containers_recommended": self.recommended.containers,
            "containers_saved": self.no_cut.containers - self.recommended.containers,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "timed_out": self.timed_out,
            "source": Source.MEASURED.value,
            "label": "SIMULATED CUTTING, MEASURED PACKING ARITHMETIC",
        }


# --------------------------------------------------------------------------- #
# Candidate generation (bounded, deterministic)
# --------------------------------------------------------------------------- #


def _snap(value: int, increment: int) -> int:
    if increment <= 1:
        return int(value)
    return int(round(value / increment) * increment)


def _residual_axis_lengths(plan: PackingPlan) -> List[int]:
    """Leftover free lengths along each axis of each used container.

    A cheap, deterministic read of "how long a piece could still slide into a
    container that is already partly full". These only *seed* candidate lengths;
    the re-pack and the validator are what actually prove a candidate.
    """
    lengths: List[int] = []
    for c in plan.containers_used:
        placed = plan.placements_for(c.container_id)
        inner = (c.inner_width_mm, c.inner_depth_mm, c.inner_height_mm)
        for axis_idx, axis in enumerate((Axis.X, Axis.Y, Axis.Z)):
            top = 0
            for p in placed:
                extent = p.position.as_tuple()[axis_idx] + p.size.as_tuple()[axis_idx]
                top = max(top, extent)
            residual = inner[axis_idx] - top
            if residual > 0:
                lengths.append(residual)
    return lengths


def _candidate_segments(pipe: WasteItem, template: Container,
                        residuals: List[int], cfg: CutPlannerConfig) -> List[List[int]]:
    """Bounded set of segment-length lists for one pipe.

    Sources (brief §4): residual cavity lengths, container inner dimensions,
    standard equal division. Each candidate respects the minimum segment length,
    protected ends, the pipe's own maximum cut count and the planner cap.
    """
    L = pipe.length_mm
    min_seg = pipe.effective_minimum_segment_mm
    max_cuts = min(cfg.max_cuts_per_plan, pipe.maximum_number_of_cuts)
    if max_cuts < 1 or L < 2 * min_seg:
        return []
    kerf = cfg.cut.kerf_mm
    prot = pipe.protected_end_length_mm

    # First-segment length seeds, from residual cavities + container inner dims +
    # the halfway point, all snapped to the increment and kept feasible.
    seeds = set(residuals)
    seeds.update((template.inner_width_mm, template.inner_depth_mm,
                  template.inner_height_mm))
    seeds.add(L // 2)
    first_lengths: List[int] = []
    for s in sorted(seeds):
        f = _snap(int(s), cfg.cut_increment_mm)
        remainder = L - f - kerf
        if f >= max(min_seg, prot) and remainder >= max(min_seg, prot):
            first_lengths.append(f)
    # De-duplicate while preserving order, then cap per pipe.
    seen: set = set()
    first_lengths = [f for f in first_lengths
                     if not (f in seen or seen.add(f))][:cfg.max_candidates_per_pipe]

    candidates: List[List[int]] = []
    for f in first_lengths:
        remainder = L - f - kerf
        # 1 cut: [f, remainder]
        candidates.append([f, remainder])
        # 2 cuts: split the remainder in half, if allowed and feasible.
        if max_cuts >= 2:
            half = _snap(remainder // 2, cfg.cut_increment_mm)
            r2 = remainder - half - kerf
            if half >= min_seg and r2 >= min_seg:
                candidates.append([f, half, r2])
    # Final feasibility filter and cap.
    ok = [c for c in candidates
          if len(c) - 1 <= max_cuts and all(s >= min_seg for s in c)
          and sum(c) + (len(c) - 1) * kerf == L]
    return ok[:cfg.max_candidates_per_pipe]


def _marginal_pipes(scenario: Scenario, plan: PackingPlan,
                    cfg: CutPlannerConfig) -> List[WasteItem]:
    """Cuttable pipes worth cutting, in priority order.

    Cutting can help even when a pipe is neither unplaced nor in the last
    container — a shorter pipe may then fit a gap that frees a whole container
    elsewhere — so every cuttable pipe is a candidate. The heuristics only set
    the *order* in which the bounded budget is spent: unplaced pipes first (they
    cost feasibility outright), then pipes in the last-opened container (most
    likely to empty it), then the rest, longest first (most cut flexibility).
    """
    by_id = {i.item_id: i for i in scenario.items}
    priority: Dict[str, int] = {}
    for iid in plan.unplaced_item_ids:
        priority[iid] = 0
    used = plan.containers_used
    if used:
        last_id = used[-1].container_id
        for p in plan.placements_for(last_id):
            priority.setdefault(p.item_id, 1)

    cuttable = [i for i in scenario.items if i.is_cuttable]
    ordered = sorted(cuttable,
                     key=lambda it: (priority.get(it.item_id, 2), -it.length_mm))
    return ordered[:cfg.max_pipes_considered]


# --------------------------------------------------------------------------- #
# Scenario derivation and scoring
# --------------------------------------------------------------------------- #


def _derived_scenario(scenario: Scenario, cuts: List[Tuple[WasteItem, List[int]]],
                      cfg: CutPlannerConfig) -> Tuple[Scenario, List[CutProposal],
                                                      List[WasteItem]]:
    """Replace each cut pipe with its segments; return scenario + proposals."""
    remove_ids = {p.item_id for p, _ in cuts}
    items = [i for i in scenario.items if i.item_id not in remove_ids]
    proposals: List[CutProposal] = []
    all_children: List[WasteItem] = []
    for idx, (pipe, segs) in enumerate(cuts):
        children = derive_segments(pipe, segs, kerf_mm=cfg.cut.kerf_mm)
        items.extend(children)
        all_children.extend(children)
        prop = CutProposal.for_segments(
            f"cut-{scenario.scenario_id}-{pipe.item_id}-{idx}", pipe, segs,
            config=cfg.cut, reason=f"segment {pipe.item_id} to fit residual cavities")
        prop.validator_result = validate_proposal(prop, pipe, children=children)
        proposals.append(prop)
    derived = replace(scenario,
                      scenario_id=f"{scenario.scenario_id}-cut",
                      items=items)
    return derived, proposals, all_children


def _score_alternative(label: str, strategy: Strategy, is_cut: bool,
                       plan: PackingPlan, proposals: List[CutProposal],
                       no_cut_containers: int, no_cut_util: float,
                       cfg: CutPlannerConfig, valid: bool) -> CutAlternative:
    n_cuts = sum(p.n_cuts for p in proposals)
    cutting_time = sum(p.estimated_cutting_time_s for p in proposals)
    handling_time = sum(p.estimated_handling_time_s for p in proposals)
    kerf_cm3 = sum(p.kerf_material_loss_mm3 for p in proposals) / 1000.0

    containers = plan.containers_required
    util = plan.utilization_pct

    container_savings = no_cut_containers - containers
    util_gain = util - no_cut_util
    value = (container_savings * cfg.container_cost_proxy
             + max(0.0, util_gain) * cfg.utilization_weight)
    process_cost = 0.0
    if is_cut:
        process_cost = (n_cuts * cfg.per_cut_cost
                        + cutting_time * cfg.cutting_time_cost_per_s
                        + handling_time * cfg.handling_time_cost_per_s
                        + kerf_cm3 * cfg.kerf_loss_cost_per_cm3
                        + cfg.complexity_cost_per_cut_plan)
    score = value - process_cost
    return CutAlternative(
        label=label, strategy=strategy, is_cut=is_cut, plan=plan,
        proposals=proposals, containers=containers, utilization_pct=util,
        n_cuts=n_cuts, cutting_time_s=cutting_time, handling_time_s=handling_time,
        kerf_loss_cm3=kerf_cm3, process_cost=process_cost, value=value,
        whole_process_score=score, valid=valid)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def plan_cut_aware(scenario: Scenario, *,
                   optimizer: Optional[OptimizerConfig] = None,
                   config: Optional[CutPlannerConfig] = None
                   ) -> WholeProcessComparison:
    """Full deterministic cut-aware comparison for ``scenario``."""
    cfg = config or CutPlannerConfig()
    opt = optimizer or OptimizerConfig(seed=scenario.seed)
    started = time.perf_counter()
    deadline = started + cfg.time_budget_ms / 1000.0

    # (1) no-cut reference under MAX_DENSITY.
    base_opt = replace(opt, strategy=Strategy.MAX_DENSITY)
    no_cut_plan = pack_optimized(scenario, config=base_opt,
                                 plan_id=f"nocut-{scenario.scenario_id}")
    no_cut = _score_alternative(
        "no_cut", Strategy.MAX_DENSITY, False, no_cut_plan, [],
        no_cut_plan.containers_required, no_cut_plan.utilization_pct, cfg,
        valid=no_cut_plan.is_valid)

    # (2,3) marginal pipes + residual cavities.
    residuals = _residual_axis_lengths(no_cut_plan)
    pipes = _marginal_pipes(scenario, no_cut_plan, cfg)

    template = scenario.container_template
    alternatives: List[CutAlternative] = []
    candidates_evaluated = 0
    timed_out = False

    # (4,5,6,7,8) generate candidates, derive, re-pack per strategy, validate.
    for pipe in pipes:
        if template is None:
            break
        for segs in _candidate_segments(pipe, template, residuals, cfg):
            if candidates_evaluated >= cfg.max_cut_aware_plans:
                break
            if time.perf_counter() > deadline:
                timed_out = True
                break
            candidates_evaluated += 1
            derived, proposals, children = _derived_scenario(
                scenario, [(pipe, segs)], cfg)

            # Independent gates BEFORE packing: cut validity + no coexistence.
            cut_valid = all(p.is_validated for p in proposals)
            coexist_ok = validate_no_coexistence(derived.items)["valid"]

            for strategy in CUT_STRATEGIES:
                cut_opt = replace(opt, strategy=strategy)
                plan = pack_optimized(
                    derived, config=cut_opt,
                    plan_id=f"cut-{scenario.scenario_id}-{pipe.item_id}-{strategy.value}")
                valid = cut_valid and coexist_ok and plan.is_valid
                alt = _score_alternative(
                    f"cut:{pipe.item_id}:{'-'.join(map(str, segs))}:{strategy.value}",
                    strategy, True, plan, proposals,
                    no_cut.containers, no_cut.utilization_pct, cfg, valid)
                alternatives.append(alt)
        if candidates_evaluated >= cfg.max_cut_aware_plans or timed_out:
            break

    # (8) recommendation: best VALID alternative that beats no-cut by the margin.
    valid_alts = [a for a in alternatives if a.valid]
    best = max(valid_alts, key=lambda a: a.whole_process_score, default=None)
    recommend_cut = bool(best and best.whole_process_score
                         > no_cut.whole_process_score + cfg.recommendation_margin)
    if recommend_cut:
        recommended_label = best.label
        saved = no_cut.containers - best.containers
        reason = (f"Cutting saves {saved} container(s) for "
                  f"{best.n_cuts} cut(s); net whole-process benefit "
                  f"{best.whole_process_score:.0f} > 0.")
    else:
        recommended_label = no_cut.label
        reason = ("No cut recommended — no validated cut alternative out-earns "
                  "leaving the pipe whole once cutting time, kerf and "
                  "complexity are charged.")

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return WholeProcessComparison(
        scenario_id=scenario.scenario_id, no_cut=no_cut, alternatives=alternatives,
        recommended_label=recommended_label, recommend_cut=recommend_cut,
        reason=reason, candidates_evaluated=candidates_evaluated,
        pipes_considered=[p.item_id for p in pipes], elapsed_ms=elapsed_ms,
        timed_out=timed_out, config=cfg)


__all__ = [
    "CUT_STRATEGIES", "CutPlannerConfig", "CutAlternative",
    "WholeProcessComparison", "plan_cut_aware",
]
