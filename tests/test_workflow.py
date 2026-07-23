"""Workflow engine: the safety gate, re-planning, dynamic events, audit trail."""

from __future__ import annotations

import pytest

from wisepack_core.domain import (
    ApprovalState, ContainerStatus, ItemStatus, Strategy,
)
from wisepack_core.events import (
    Actor, DynamicEvent, DynamicEventType, PRE_APPROVAL_STAGES, Result, Stage,
)
from wisepack_core.packing import OptimizerConfig
from wisepack_core.workflow import (
    ApprovalRequired, RobotSimConfig, WorkflowConfig, WorkflowEngine,
    WorkflowError, run_headless,
)


def fast(**kw) -> WorkflowConfig:
    """A quick, deterministic configuration for tests."""
    kw.setdefault("preset", "mixed_pipes_small")
    kw.setdefault("seed", 42)
    kw.setdefault("optimizer", OptimizerConfig(seed=42, restarts=3,
                                               time_budget_ms=2000.0))
    return WorkflowConfig(**kw)


def planned(config: WorkflowConfig | None = None) -> WorkflowEngine:
    """Drive an engine up to (but not through) the approval gate."""
    engine = WorkflowEngine(config or fast())
    engine.generate_or_load_scenario()
    engine.scan_and_detect()
    engine.generate_plans()
    engine.digital_twin_validate()
    engine.request_approval()
    return engine


# --------------------------------------------------------------------------- #
# The safety gate
# --------------------------------------------------------------------------- #

def test_execution_requires_approval():
    engine = planned()
    assert engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    with pytest.raises(ApprovalRequired):
        engine.step_execution()


def test_no_placement_is_executed_before_approval():
    engine = planned()
    with pytest.raises(ApprovalRequired):
        engine.step_execution()
    assert not any(p.executed for p in engine.selected.placements)
    assert engine.progress_pct == 0.0


def test_every_pre_approval_stage_refuses_execution():
    engine = planned()
    for stage in PRE_APPROVAL_STAGES:
        engine.stage = stage
        with pytest.raises(ApprovalRequired):
            engine.step_execution()


def test_rejected_plan_cannot_be_executed():
    engine = planned()
    engine.reject("not acceptable")
    engine.selected.approval_state = ApprovalState.REJECTED
    engine.stage = Stage.PICK_ITEM
    with pytest.raises(ApprovalRequired):
        engine.step_execution()


def test_approve_out_of_order_is_refused():
    engine = WorkflowEngine(fast())
    engine.generate_or_load_scenario()
    engine.scan_and_detect()
    engine.generate_plans()
    engine.digital_twin_validate()
    with pytest.raises(WorkflowError):
        engine.approve()            # request_approval() was never called


def test_approval_is_recorded_in_the_audit_trail():
    engine = planned()
    engine.approve(operator="inspector-1")
    events = [e for e in engine.log.events() if e.action == "approve_plan"]
    assert len(events) == 1
    assert events[0].actor is Actor.OPERATOR
    assert engine.stats.operator_interventions == 1


def test_auto_approval_is_attributed_to_the_system_not_an_operator():
    """A headless run must not fabricate a human decision."""
    engine = planned()
    engine.approve(auto=True)
    event = next(e for e in engine.log.events() if e.action == "approve_plan")
    assert event.actor is Actor.SYSTEM
    assert event.details["auto"] is True
    assert engine.stats.operator_interventions == 0


# --------------------------------------------------------------------------- #
# Rejection and re-planning
# --------------------------------------------------------------------------- #

def test_rejection_causes_a_replan():
    engine = planned()
    engine.reject("operator wants a different arrangement")
    assert engine.stats.replans == 1
    assert "operator rejection" in engine.stats.replan_causes[0]
    assert engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert engine.selected.approval_state is ApprovalState.PENDING


def test_replan_freezes_already_executed_placements():
    """Re-planning must not move items that are physically in a container."""
    engine = planned(fast(auto_approve=True))
    engine.approve(auto=True)
    for _ in range(3):
        engine.step_execution()
    executed = {p.item_id for p in engine.selected.placements if p.executed}
    assert executed, "nothing was executed before the re-plan"

    engine.replan("test-triggered re-plan")

    still_executed = {p.item_id for p in engine.selected.placements if p.executed}
    assert executed <= still_executed
    for item_id in executed:
        placement = engine.selected.placement_for_item(item_id)
        assert placement is not None and placement.executed


def test_replan_limit_enters_degraded_rather_than_looping():
    engine = planned(fast(max_replans=2, auto_approve=True))
    engine.approve(auto=True)
    for n in range(4):
        engine.replan(f"forced {n}")
    assert engine.stage is Stage.DEGRADED
    assert "limit" in engine.degraded_reason
    assert engine.finished is True


def test_degraded_state_holds_rather_than_continuing():
    engine = planned()
    engine.enter_degraded("optimizer stopped publishing")
    assert engine.stage is Stage.DEGRADED
    assert engine.robot_state == "held"
    event = next(e for e in engine.log.events() if e.action == "enter_degraded")
    assert event.result is Result.FAILED


# --------------------------------------------------------------------------- #
# Dynamic events
# --------------------------------------------------------------------------- #

def test_injected_item_triggers_a_replan_and_is_packed():
    event = DynamicEvent(
        event_type=DynamicEventType.ITEM_INJECT, trigger="placement:2",
        label="Urgent ILW component arrives",
        payload={"item": {"length_mm": 900, "outer_diameter_mm": 180,
                          "inner_diameter_mm": 150, "priority": 9,
                          "dose_class": "ILW"}})
    engine = run_headless(fast(preset="late_arrival_replan", auto_approve=True,
                               dynamic_events=[event]))
    assert engine.stats.replans >= 1
    injected = [i for i in engine.scenario.items if i.injected]
    assert len(injected) == 1
    assert engine.selected.placement_for_item(injected[0].item_id) is not None
    assert engine.stage is Stage.COMPLETE


def test_removed_item_is_not_packed_after_the_replan():
    engine = planned(fast(auto_approve=True))
    engine.approve(auto=True)
    victim = engine.scenario.items[-1].item_id
    engine.apply_dynamic_event(DynamicEvent(
        event_type=DynamicEventType.ITEM_REMOVED, trigger="stage:PICK_ITEM",
        payload={"item_id": victim}))
    assert engine.scenario.item(victim).status is ItemStatus.REMOVED
    assert engine.selected.placement_for_item(victim) is None


def test_unavailable_container_is_not_used_after_the_replan():
    engine = planned(fast(preset="mixed_pipes_dense", auto_approve=True))
    engine.approve(auto=True)
    victim = engine.selected.containers_used[0].container_id
    engine.apply_dynamic_event(DynamicEvent(
        event_type=DynamicEventType.CONTAINER_UNAVAILABLE,
        trigger="stage:PICK_ITEM", payload={"container_id": victim}))
    for placement in engine.selected.placements:
        if placement.executed:
            continue                       # already physically placed, frozen
        assert placement.container_id != victim


def test_grasp_failure_event_is_logged_and_retried():
    engine = planned(fast(auto_approve=True))
    engine.approve(auto=True)
    engine.apply_dynamic_event(DynamicEvent(
        event_type=DynamicEventType.GRASP_FAILURE, trigger="stage:PICK_ITEM"))
    engine.step_execution()
    actions = [e.action for e in engine.log.events()]
    assert "dynamic_event:grasp_failure" in actions
    assert "retry_pick" in actions
    # A grasp failure is an execution-layer retry, NOT a re-plan trigger.
    assert engine.stats.replans == 0


def test_grasp_failure_is_retried_then_abandoned():
    engine = planned(fast(auto_approve=True,
                          robot=RobotSimConfig(pick_failure_probability=1.0,
                                               max_pick_retries=2, seed=1)))
    engine.approve(auto=True)
    for _ in range(6):
        engine.step_execution()
    actions = [e.action for e in engine.log.events()]
    assert actions.count("retry_pick") >= 2
    assert "abandon_item" in actions
    assert engine.stats.cycles_attempted > engine.stats.cycles_completed


def test_segregation_rule_change_forces_a_replan():
    engine = planned(fast(preset="segregated_materials", auto_approve=True))
    engine.approve(auto=True)
    before = engine.stats.replans
    engine.apply_dynamic_event(DynamicEvent(
        event_type=DynamicEventType.SEGREGATION_RULE_CHANGE,
        trigger="stage:PICK_ITEM",
        payload={"allowed_segregation_groups": ["A"]}))
    assert engine.stats.replans == before + 1


def test_dynamic_event_triggers_are_validated():
    with pytest.raises(ValueError, match="trigger"):
        DynamicEvent(event_type=DynamicEventType.ITEM_INJECT,
                     trigger="whenever I feel like it")


def test_dynamic_event_fires_once_only():
    event = DynamicEvent(event_type=DynamicEventType.GRASP_FAILURE,
                         trigger="stage:PICK_ITEM")
    assert event.matches(stage=Stage.PICK_ITEM)
    event.fired = True
    assert not event.matches(stage=Stage.PICK_ITEM)


def test_replan_triggers_are_declared_explicitly():
    from wisepack_core.events import REPLAN_TRIGGERS
    assert DynamicEventType.ITEM_INJECT in REPLAN_TRIGGERS
    assert DynamicEventType.CONTAINER_UNAVAILABLE in REPLAN_TRIGGERS
    # A slipped grasp is a retry, not a reason to re-plan a whole container.
    assert DynamicEventType.GRASP_FAILURE not in REPLAN_TRIGGERS


# --------------------------------------------------------------------------- #
# Completion
# --------------------------------------------------------------------------- #

def test_completion_only_after_every_placement_is_resolved():
    engine = run_headless(fast(auto_approve=True))
    assert engine.stage is Stage.COMPLETE
    assert all(p.executed for p in engine.selected.placements)
    assert engine.progress_pct == pytest.approx(100.0)


def test_completion_emits_a_terminal_event():
    engine = run_headless(fast(auto_approve=True))
    last = engine.log.events()[-1]
    assert last.stage is Stage.COMPLETE
    assert last.action == "cycle_complete"


def test_container_state_reaches_complete():
    engine = run_headless(fast(auto_approve=True))
    for container in engine.selected.containers_used:
        assert container.status is ContainerStatus.COMPLETE


# --------------------------------------------------------------------------- #
# The audit trail
# --------------------------------------------------------------------------- #

def test_action_sequence_is_monotonic_and_gap_free():
    engine = run_headless(fast(preset="mixed_pipes_dense", auto_approve=True))
    ok, note = engine.log.sequence_is_monotonic()
    assert ok, note
    assert engine.log.count > 20


def test_every_workflow_stage_emits_at_least_one_event():
    engine = run_headless(fast(auto_approve=True))
    stages = set(engine.log.by_stage())
    for required in (Stage.GENERATE_OR_LOAD_SCENARIO, Stage.SCAN_SOURCE_BIN,
                     Stage.DETECT_ITEMS, Stage.GENERATE_BASELINE_PLAN,
                     Stage.GENERATE_OPTIMIZED_PLAN, Stage.DIGITAL_TWIN_VALIDATE,
                     Stage.WAIT_FOR_OPERATOR_APPROVAL, Stage.PICK_ITEM,
                     Stage.VERIFY_PICK, Stage.PLACE_ITEM,
                     Stage.VERIFY_PLACEMENT, Stage.UPDATE_CONTAINER_STATE,
                     Stage.COMPLETE):
        assert required.value in stages, f"no event for stage {required.value}"


def test_simulated_events_are_labelled_simulated():
    engine = run_headless(fast(auto_approve=True))
    for event in engine.log.events():
        if event.actor in (Actor.ROBOT_SIM, Actor.PERCEPTION_SIM):
            assert event.source.value == "simulated", \
                f"{event.action} from {event.actor.value} is not labelled simulated"


def test_events_serialise_to_json_and_back():
    from wisepack_core.events import ActionEvent
    engine = run_headless(fast(auto_approve=True))
    for event in engine.log.events()[:20]:
        restored = ActionEvent.from_dict(event.to_dict())
        assert restored.sequence == event.sequence
        assert restored.stage is event.stage
        assert restored.source is event.source


def test_oversized_event_details_are_truncated_visibly():
    """A silently-truncated audit record is worse than a marked one."""
    from wisepack_core.events import ActionEvent
    event = ActionEvent(stage=Stage.COMPLETE, action="huge", actor=Actor.SYSTEM,
                        details={"blob": "x" * 20000})
    payload = event.to_json()
    assert len(payload.encode()) <= ActionEvent.MAX_SERIALISED_BYTES + 512
    assert "_truncated" in payload
    assert "_dropped_keys" in payload


def test_action_log_sinks_receive_every_event():
    seen = []
    engine = run_headless(fast(auto_approve=True), on_event=seen.append)
    assert len(seen) == engine.log.count
    assert [e.sequence for e in seen] == list(range(1, len(seen) + 1))


def test_a_broken_sink_does_not_stop_the_audit_trail():
    def explode(_event):
        raise RuntimeError("sink failure")
    engine = WorkflowEngine(fast())
    engine.log.add_sink(explode)
    engine.generate_or_load_scenario()
    assert engine.log.count == 1        # recorded despite the failing sink


# --------------------------------------------------------------------------- #
# Strategies and reporting
# --------------------------------------------------------------------------- #

def test_strategy_comparison_produces_a_plan_per_strategy():
    engine = planned()
    plans = engine.compare_strategies()
    assert set(plans) == {s.value for s in Strategy}
    for plan in plans.values():
        assert plan.is_valid


def test_snapshot_is_complete_and_json_safe():
    import json
    engine = run_headless(fast(auto_approve=True))
    snapshot = engine.snapshot()
    json.dumps(snapshot)                # must not raise
    for key in ("run_id", "stage", "scenario", "baseline", "optimized",
                "selected_plan_id", "progress_pct", "stats"):
        assert key in snapshot


def test_kpis_require_planning_first():
    engine = WorkflowEngine(fast())
    with pytest.raises(WorkflowError):
        engine.kpis()


def test_detection_is_labelled_simulated_in_the_kpis():
    engine = run_headless(fast(auto_approve=True))
    report = engine.kpis()
    assert report.metrics["detection_rate_pct"].source.value == "simulated"
    assert "NOT a vision measurement" in report.metrics["detection_rate_pct"].note


# --------------------------------------------------------------------------- #
# Re-plan must revoke authorisation immediately
# --------------------------------------------------------------------------- #

def test_a_replan_stops_execution_in_the_same_step():
    """Regression: one more item was placed on the SUPERSEDED plan.

    `step_execution` guarded only on `Stage.REPLAN`, but `replan()` finishes by
    calling `request_approval()`, which leaves the stage at
    WAIT_FOR_OPERATOR_APPROVAL. The guard never fired, so after a dynamic event
    revoked approval the method carried on and executed one more placement.
    Observed live as stage NEXT_ITEM immediately after a re-plan.
    """
    engine = planned(fast(auto_approve=False))
    engine.approve()
    for _ in range(2):
        engine.step_execution()
    executed_before = sum(1 for p in engine.selected.placements if p.executed)

    engine.dynamic_events = [DynamicEvent(
        event_type=DynamicEventType.ITEM_INJECT,
        trigger=f"placement:{engine.cursor.index}",
        label="late arrival",
        payload={"item": {"length_mm": 900, "outer_diameter_mm": 180,
                          "inner_diameter_mm": 150}})]

    engine.step_execution()          # fires the event -> re-plan -> gate

    assert engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert engine.selected.approval_state is not ApprovalState.APPROVED
    executed_after = sum(1 for p in engine.selected.placements if p.executed)
    assert executed_after == executed_before, (
        "a placement was executed after the re-plan revoked approval")


def test_step_execution_refuses_once_approval_is_revoked():
    engine = planned(fast(auto_approve=False))
    engine.approve()
    engine.step_execution()
    engine.selected.approval_state = ApprovalState.PENDING
    with pytest.raises(ApprovalRequired):
        engine.step_execution()
