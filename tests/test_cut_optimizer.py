"""Cut-aware whole-process planner.

Pins the two behaviours the brief singles out (§5, §8): a genuine container
saving is found and recommended, and a case where cutting does not pay returns
"No cut recommended" from the SAME untouched scoring. Also checks the search is
bounded and that every alternative is independently validated.
"""

from __future__ import annotations

from wisepack_core.cut_optimizer import (
    CUT_STRATEGIES, CutPlannerConfig, plan_cut_aware,
)
from wisepack_core.domain import Scenario, Strategy, WasteItem
from wisepack_core.generator import make_container
from wisepack_core.packing import OptimizerConfig


def _pipe(item_id, length_mm, od, *, cut=False, mc=0, minseg=None):
    return WasteItem(item_id, length_mm=length_mm, outer_diameter_mm=od,
                     inner_diameter_mm=od - 40, weight_kg=1.0, cut_allowed=cut,
                     maximum_number_of_cuts=mc, minimum_segment_length_mm=minseg,
                     protected_end_length_mm=20)


def _opt(seed=7):
    return OptimizerConfig(seed=seed, restarts=4, time_budget_ms=2500)


def container_saving_scenario():
    """Two 800 mm pipes leave two 700 mm floor gaps; a 1300 mm cuttable pipe fits
    whole only in a fresh container, but its halves drop into the gaps."""
    box = make_container("standard_box", "CNT")     # 1500 x 800 x 600
    items = [_pipe("A", 800, 360), _pipe("B", 800, 360),
             _pipe("C", 1300, 340, cut=True, mc=2, minseg=400)]
    return Scenario("cut_saving", preset="custom", seed=7, items=items,
                    container_template=box, max_containers=8)


def no_benefit_scenario():
    """A single cuttable pipe that already packs into one container: cutting can
    only add process cost, so the planner must decline."""
    box = make_container("standard_box", "CNT")
    items = [_pipe("A", 1400, 300, cut=True, mc=2, minseg=400)]
    return Scenario("cut_nobenefit", preset="custom", seed=7, items=items,
                    container_template=box, max_containers=8)


# --------------------------------------------------------------------------- #
# Positive path — cutting genuinely saves a container
# --------------------------------------------------------------------------- #

def test_cutting_saves_a_container_and_is_recommended():
    cmp = plan_cut_aware(container_saving_scenario(),
                         optimizer=_opt(), config=CutPlannerConfig())
    d = cmp.to_dict()
    assert d["containers_no_cut"] == 2
    assert cmp.recommend_cut is True
    assert d["containers_saved"] >= 1
    best = cmp.recommended
    assert best.is_cut and best.containers < cmp.no_cut.containers
    assert best.n_cuts >= 1
    # The saving is COMPUTED by the packer, not asserted into being.
    assert best.plan.containers_required == best.containers


def test_recommended_alternative_is_independently_validated():
    cmp = plan_cut_aware(container_saving_scenario(),
                         optimizer=_opt(), config=CutPlannerConfig())
    best = cmp.recommended
    assert best.valid
    for prop in best.proposals:
        assert prop.is_validated
        assert prop.validator_result["validator"] == "wisepack_core.cut_validator"


def test_whole_process_score_charges_the_cut():
    """The recommended cut's process cost must be real, not zero."""
    cmp = plan_cut_aware(container_saving_scenario(),
                         optimizer=_opt(), config=CutPlannerConfig())
    best = cmp.recommended
    assert best.process_cost > 0
    assert best.cutting_time_s > 0
    assert best.kerf_loss_cm3 > 0
    # Net benefit still positive because a saved container outweighs the cost.
    assert best.whole_process_score > cmp.no_cut.whole_process_score


def test_all_three_strategies_are_explored_for_cutting():
    cmp = plan_cut_aware(container_saving_scenario(),
                         optimizer=_opt(), config=CutPlannerConfig())
    strategies = {a.strategy for a in cmp.alternatives}
    assert set(CUT_STRATEGIES) <= strategies


# --------------------------------------------------------------------------- #
# Negative path — cutting does not pay
# --------------------------------------------------------------------------- #

def test_no_cut_recommended_when_it_does_not_pay():
    cmp = plan_cut_aware(no_benefit_scenario(),
                         optimizer=_opt(), config=CutPlannerConfig())
    assert cmp.recommend_cut is False
    assert cmp.recommended.label == "no_cut"
    assert "No cut recommended" in cmp.reason


def test_no_cut_recommended_uses_the_same_scoring_not_a_special_case():
    """The decline is a score comparison, so every cut alternative it saw scored
    at or below the no-cut baseline — not filtered out by a hand rule."""
    cmp = plan_cut_aware(no_benefit_scenario(),
                         optimizer=_opt(), config=CutPlannerConfig())
    for a in cmp.alternatives:
        assert a.whole_process_score <= cmp.no_cut.whole_process_score \
            + cmp.config.recommendation_margin


# --------------------------------------------------------------------------- #
# Bounded search
# --------------------------------------------------------------------------- #

def test_search_is_bounded_by_configuration():
    cfg = CutPlannerConfig(max_cut_aware_plans=3, max_candidates_per_pipe=2,
                           max_pipes_considered=1)
    cmp = plan_cut_aware(container_saving_scenario(), optimizer=_opt(), config=cfg)
    assert cmp.candidates_evaluated <= 3
    assert len(cmp.pipes_considered) <= 1


def test_non_cuttable_scenario_yields_no_alternatives():
    box = make_container("standard_box", "CNT")
    sc = Scenario("plain", preset="custom", seed=7,
                  items=[_pipe("A", 800, 360), _pipe("B", 800, 360)],
                  container_template=box, max_containers=8)
    cmp = plan_cut_aware(sc, optimizer=_opt(), config=CutPlannerConfig())
    assert cmp.alternatives == []
    assert cmp.recommend_cut is False
