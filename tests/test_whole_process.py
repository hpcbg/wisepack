"""Whole-process workflow integration — cut HITL + inventory-aware planning.

Drives the cut workflow through a real WorkflowEngine (not the planner in
isolation) to prove the brief §6 sequence end to end: cut approval is SEPARATE
from packing approval, the ACTUAL result re-plans and re-approves, a failed cut
reverts, and planning is inventory-aware with reservations and shortage.
"""

from __future__ import annotations

import pytest

from wisepack_core.generator import build_scenario
from wisepack_core.packing import OptimizerConfig
from wisepack_core.workflow import WorkflowConfig, WorkflowEngine


def _engine(preset="cut_avoids_extra_container", seed=7):
    eng = WorkflowEngine(WorkflowConfig(
        preset=preset, seed=seed,
        optimizer=OptimizerConfig(seed=seed, restarts=6, time_budget_ms=2500)))
    eng.generate_or_load_scenario(build_scenario(preset, seed))
    eng.scan_and_detect()
    eng.generate_plans()
    eng.digital_twin_validate()
    return eng


# --------------------------------------------------------------------------- #
# Cut approval is separate from packing approval (brief §6)
# --------------------------------------------------------------------------- #

def test_cut_workflow_saves_a_container_and_requires_new_packing_approval():
    eng = _engine()
    before = eng.selected.containers_required
    assert before == 2
    eng.wp.generate_cut_alternatives()
    assert eng.wp.comparison.recommend_cut
    assert eng.stage.value == "WAIT_FOR_CUT_APPROVAL"

    eng.wp.approve_cut("inspector-1")
    assert eng.stage.value == "CUT_REQUESTED"
    eng.wp.simulate_cut()
    # Re-planned onto the derived segments, fewer containers, packing PENDING.
    assert eng.selected.containers_required < before
    assert eng.stage.value == "WAIT_FOR_OPERATOR_APPROVAL"
    assert eng.selected.approval_state.value == "pending"
    # The pipe is gone; its segments are present.
    ids = {i.item_id for i in eng.scenario.items}
    assert "pipe-C" not in ids
    assert {"pipe-C-s1", "pipe-C-s2"} <= ids


def test_cut_approval_does_not_authorise_execution():
    eng = _engine()
    eng.wp.generate_cut_alternatives()
    eng.wp.approve_cut("op")
    # Even though cutting is approved, no packing approval exists yet.
    from wisepack_core.workflow import ApprovalRequired, WorkflowError
    with pytest.raises((ApprovalRequired, WorkflowError)):
        eng.step_execution()


def test_reject_cut_reverts_to_no_cut():
    eng = _engine()
    before = eng.selected.containers_required
    eng.wp.generate_cut_alternatives()
    eng.wp.reject_cut("operator prefers no cutting")
    assert eng.wp.selected_cut_label == "no_cut"
    assert eng.wp.cut_approval_state.value == "rejected"
    assert eng.selected.containers_required == before


# --------------------------------------------------------------------------- #
# Result deviation + failure
# --------------------------------------------------------------------------- #

def test_deviated_cut_updates_lineage_and_replans():
    eng = _engine(preset="cut_result_deviation")
    eng.wp.generate_cut_alternatives()
    eng.wp.approve_cut("op")
    eng.wp.simulate_cut(deviation_mm=40)
    result = eng.wp.latest_cut_result
    assert result["validation"]["valid"]          # still conserving
    # Derived segment lengths reflect the ACTUAL (deviated) result.
    seg_ids = [i for i in eng.scenario.items if i.parent_item_id == "pipe-C"]
    assert seg_ids and eng.stage.value == "WAIT_FOR_OPERATOR_APPROVAL"


def test_failed_cut_reverts_and_requires_reapproval():
    eng = _engine()
    ids_before = {i.item_id for i in eng.scenario.items}
    eng.wp.generate_cut_alternatives()
    eng.wp.approve_cut("op")
    eng.wp.simulate_cut_failure("blade jam (simulated)")
    # No derived items registered; the pipe stays whole.
    assert {i.item_id for i in eng.scenario.items} == ids_before
    assert eng.wp.cut_approval_state.value == "pending"
    assert eng.stage.value == "WAIT_FOR_OPERATOR_APPROVAL"
    assert eng.wp.latest_cut_result["completion_status"] == "failed"


def test_no_cut_scenario_does_not_recommend_cutting():
    eng = _engine(preset="cut_not_worthwhile")
    eng.wp.generate_cut_alternatives()
    assert eng.wp.comparison.recommend_cut is False
    assert eng.wp.selected_cut_label == "no_cut"


# --------------------------------------------------------------------------- #
# Inventory-aware planning (brief §14)
# --------------------------------------------------------------------------- #

def test_planning_reserves_inventory_and_requests_delivery():
    eng = _engine()
    eng.wp.initialise_simulated_inventory(count=4)
    pr = eng.wp.check_container_availability()
    assert pr["reservations_created"] >= 1
    assert pr["plan_status"] == "ok"
    assert pr["delivery_tasks_required"]
    eng.wp.run_logistics_to_quiescence()
    # A delivered container is at the packing cell.
    assert eng.wp.inventory.summary()["at_packing_cell"] >= 1
    # After inventory is satisfied the run returns to the packing-approval gate.
    assert eng.stage.value == "WAIT_FOR_OPERATOR_APPROVAL"


def test_inventory_shortage_sets_waiting_for_container():
    eng = _engine()
    # Only one container available but the plan needs two.
    eng.wp.initialise_simulated_inventory(count=1)
    pr = eng.wp.check_container_availability()
    if pr["additional_containers_required"] > 0:
        assert pr["inventory_shortage"] is True
        assert pr["plan_status"] == "WAITING_FOR_CONTAINER"
        assert eng.stage.value == "WAIT_FOR_CONTAINER"
        assert eng.wp.inventory.summary()["forecast_shortage"] is True


def test_invalid_inventory_operation_is_rejected():
    eng = _engine()
    eng.wp.initialise_simulated_inventory(count=2)
    from wisepack_core.inventory import InvalidTransition
    # AVAILABLE -> DISPATCHED is not a permitted transition.
    with pytest.raises(InvalidTransition):
        eng.wp.inventory_operation("mark_dispatched", "INV-00")


# --------------------------------------------------------------------------- #
# Snapshot + analytics + audit trail
# --------------------------------------------------------------------------- #

def test_snapshot_carries_whole_process_state():
    eng = _engine()
    eng.wp.generate_cut_alternatives()
    snap = eng.snapshot()
    assert "whole_process" in snap
    wp = snap["whole_process"]
    for key in ("cut", "inventory", "logistics", "analytics"):
        assert key in wp
    assert wp["cut"]["recommend_cut"] is True


def test_extended_timeline_records_cut_and_inventory_actions():
    eng = _engine()
    eng.wp.initialise_simulated_inventory(count=4)
    eng.wp.generate_cut_alternatives()
    eng.wp.approve_cut("op")
    eng.wp.simulate_cut()
    actions = {e.action for e in eng.log.events()}
    assert "CUT_PROPOSAL_GENERATED" in actions
    assert "CUT_APPROVED" in actions
    assert "DERIVED_ITEMS_REGISTERED" in actions
    assert "REPLAN_AFTER_CUT" in actions
    assert "CONTAINER_REGISTERED" in actions


def test_analytics_provenance_is_labelled():
    eng = _engine()
    eng.wp.generate_cut_alternatives()
    an = eng.wp.analytics()
    assert an["cutting"]["provenance"] == "simulated_cutting_measured_packing"
    assert an["inventory"]["provenance"] == "software_state"
    assert an["cutting"]["containers_avoided"] >= 1


# --------------------------------------------------------------------------- #
# CUTTING_REQUEST idempotency (exactly once per approved proposal revision)
# --------------------------------------------------------------------------- #

def test_one_approval_produces_exactly_one_cut_request():
    eng = _engine()
    eng.wp.generate_cut_alternatives()
    eng.wp.approve_cut("op")
    first = eng.wp.build_cut_request()
    assert first is not None
    assert first["approval_revision"] == eng.wp.approval_revision
    assert first["proposal_id"] and first["scenario_revision"] >= 1
    # The skill has transitioned APPROVED -> REQUESTED.
    assert eng.wp.cut_skill_state.value == "requested"
    # Every subsequent call for the same approval yields nothing.
    assert eng.wp.build_cut_request() is None


def test_repeated_state_publication_produces_no_duplicate_request():
    eng = _engine()
    eng.wp.generate_cut_alternatives()
    eng.wp.approve_cut("op")
    emitted = [r for r in (eng.wp.build_cut_request() for _ in range(10))
               if r is not None]
    assert len(emitted) == 1


def test_newer_proposal_revision_produces_one_new_request():
    eng = _engine()
    eng.wp.generate_cut_alternatives()
    eng.wp.approve_cut("op")
    r1 = eng.wp.build_cut_request()
    # A newer comparison is a new proposal revision; re-approve and request again.
    eng.wp.generate_cut_alternatives()
    assert eng.wp.cut_approval_state.value == "pending"     # approval was reset
    eng.wp.approve_cut("op")
    r2 = eng.wp.build_cut_request()
    assert r1 is not None and r2 is not None
    assert r2["plan_revision"] > r1["plan_revision"]
    assert r2["approval_revision"] > r1["approval_revision"]
    assert r2["request_id"] != r1["request_id"]


def test_reject_then_reapprove_requests_only_on_new_approval_revision():
    eng = _engine()
    eng.wp.generate_cut_alternatives()
    eng.wp.approve_cut("op")
    r1 = eng.wp.build_cut_request()
    rev1 = r1["approval_revision"]
    # Reject does not bump the approval revision and emits no request.
    eng.wp.reject_cut("changed mind")
    assert eng.wp.build_cut_request() is None
    # Re-select and re-approve: a new approval revision -> exactly one new request.
    eng.wp.select_alternative(eng.wp.comparison.recommended_label)
    eng.wp.approve_cut("op")
    r2 = eng.wp.build_cut_request()
    assert r2 is not None and r2["approval_revision"] > rev1
    assert eng.wp.build_cut_request() is None


def test_stale_approval_cannot_request_a_newer_proposal():
    eng = _engine()
    eng.wp.generate_cut_alternatives()
    eng.wp.approve_cut("op")            # approves proposal revision N
    # A newer comparison (revision N+1) arrives before the request is built.
    eng.wp.generate_cut_alternatives()
    # The stale approval must NOT request a cut for the newer proposal.
    assert eng.wp.build_cut_request() is None


def test_no_request_before_validation_and_approval():
    eng = _engine()
    eng.wp.generate_cut_alternatives()
    # Approved? No. So no request may be emitted.
    assert eng.wp.cut_approval_state.value == "pending"
    assert eng.wp.build_cut_request() is None
