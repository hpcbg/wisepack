"""The contradictory approval state, pinned — and the cross-run merge behind it.

THE DEFECT, as observed during manual validation:

    stage          = WAIT_FOR_OPERATOR_APPROVAL
    approval_state = approved
    anomaly_hold   = false

The dashboard said "Decision required" while correctly disabling Approve,
because the plan was already approved. There is no decision an operator can make
in that state, and no wording makes it honest — the state itself is wrong.

Alongside it, a second symptom of the same root cause:

    scenario_id       = mixed_pipes_dense-s42
    selected_plan_id  = plan-optimized-isaac_cylinders_smoke-s42

a scenario from one run rendered beside a plan from another. The orchestrator's
topics are latched and independent, so the dashboard mirror is always a mixture
of whatever each publisher last said; without an identity stamp it cannot tell a
coherent pair from two different runs.

Nothing here needs ROS, FIWARE, Isaac or a browser.
"""

from __future__ import annotations

import os

import pytest

from wisepack_core.domain import ApprovalState
from wisepack_core.events import Stage
from wisepack_core.workflow import (
    WorkflowConfig, WorkflowEngine, WorkflowError,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _engine_at_gate(**overrides) -> WorkflowEngine:
    """A engine driven to the approval gate, the way the orchestrator drives it."""
    engine = WorkflowEngine(WorkflowConfig(
        preset=overrides.pop("preset", "isaac_cylinders_smoke"),
        seed=overrides.pop("seed", 42), **overrides))
    engine.generate_or_load_scenario()
    engine.scan_and_detect()
    engine.generate_plans()
    engine.digital_twin_validate()
    engine.request_approval()
    return engine


# --------------------------------------------------------------------------- #
# 1. approved + no hold never remains at WAIT_FOR_OPERATOR_APPROVAL
# --------------------------------------------------------------------------- #

def test_approving_leaves_the_gate_immediately():
    engine = _engine_at_gate()
    engine.approve("operator")
    assert engine.selected.approval_state is ApprovalState.APPROVED
    assert engine.stage is not Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert engine.approval_inconsistency() == ""


def test_an_approved_plan_at_the_gate_is_reported_as_inconsistent():
    """The exact triple from the manual validation, detected rather than shown."""
    engine = _engine_at_gate()
    # Force the contradiction the way a stage-only rollback used to produce it.
    engine.selected.approval_state = ApprovalState.APPROVED
    engine._set_stage(Stage.WAIT_FOR_OPERATOR_APPROVAL)
    assert not engine.anomaly_hold
    reason = engine.approval_inconsistency()
    assert reason, "approved at the approval gate must be reported"
    assert "no decision to make" in reason


def test_resume_execution_stage_clears_the_contradiction():
    engine = _engine_at_gate()
    engine.selected.approval_state = ApprovalState.APPROVED
    engine._set_stage(Stage.WAIT_FOR_OPERATOR_APPROVAL)
    engine.resume_execution_stage()
    assert engine.stage is Stage.PICK_ITEM
    assert engine.approval_inconsistency() == ""


def test_resume_execution_stage_refuses_to_authorise_anything():
    """It moves an APPROVED plan out of the gate. It never approves one."""
    engine = _engine_at_gate()
    assert engine.selected.approval_state is ApprovalState.PENDING
    engine.resume_execution_stage()
    assert engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert engine.selected.approval_state is ApprovalState.PENDING


def test_resume_after_partial_execution_does_not_rewind_the_cursor():
    engine = _engine_at_gate()
    engine.approve("operator")
    engine.cursor.index = 2
    engine.revoke_approval("test")
    engine.approve("operator")
    assert engine.cursor.index == 0, "a fresh approval restarts the plan"

    engine.cursor.index = 2
    engine.selected.approval_state = ApprovalState.APPROVED
    engine._set_stage(Stage.WAIT_FOR_OPERATOR_APPROVAL)
    engine.resume_execution_stage()
    assert engine.cursor.index == 2, "resuming must not re-place completed items"
    assert engine.stage is Stage.NEXT_ITEM


# --------------------------------------------------------------------------- #
# 2. renewed approval sets pending BEFORE entering the gate
# --------------------------------------------------------------------------- #

def test_revoking_sets_pending_and_only_then_enters_the_gate():
    engine = _engine_at_gate()
    engine.approve("operator")
    engine.revoke_approval("a critical anomaly")
    assert engine.selected.approval_state is ApprovalState.PENDING
    assert engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert engine.approval_revision == engine.scenario_revision
    assert engine.approval_plan_id == engine.selected.plan_id
    assert engine.approval_inconsistency() == ""


def test_revocation_is_unconditional():
    """The old guard revoked ONLY when the plan was APPROVED.

    Any other state fell through with the stamp untouched, which is how the gate
    could be entered without a decision actually being pending for it.
    """
    engine = _engine_at_gate()
    engine.selected.approval_state = ApprovalState.SUPERSEDED
    engine.revoke_approval("scene reset")
    assert engine.selected.approval_state is ApprovalState.PENDING
    assert engine.approval_plan_id == engine.selected.plan_id


def test_revocation_is_recorded_in_the_audit_trail():
    engine = _engine_at_gate()
    engine.approve("operator")
    before = engine.log.count
    engine.revoke_approval("critical anomaly radiation_spike")
    assert engine.log.count > before, "a withdrawn authorisation must be auditable"


def test_a_critical_anomaly_revokes_then_gates():
    from wisepack_core.anomaly import AnomalyClass, AnomalyEvent, Severity

    engine = _engine_at_gate()
    engine.approve("operator")
    engine.apply_anomaly(AnomalyEvent(anomaly_class=AnomalyClass.SHEAR_POSITION_TOO_HIGH,
                                      severity=Severity.CRITICAL))
    assert engine.anomaly_hold
    assert engine.selected.approval_state is ApprovalState.PENDING
    assert engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert engine.approval_inconsistency() == ""


def test_a_replan_supersedes_an_approved_plan():
    """An approval is for one plan. It must not survive into its replacement."""
    engine = _engine_at_gate()
    engine.approve("operator")
    approved_plan = engine.selected.plan_id
    engine.replan("test cause")
    assert engine.selected.plan_id != approved_plan or \
        engine.selected.approval_state is not ApprovalState.APPROVED


# --------------------------------------------------------------------------- #
# 3. stale plan / scenario revisions are rejected
# --------------------------------------------------------------------------- #

def test_approve_refuses_a_decision_aimed_at_a_superseded_revision():
    engine = _engine_at_gate()
    engine._bump_scenario_revision()          # the batch changed underneath
    with pytest.raises(WorkflowError) as excinfo:
        engine.approve("operator")
    assert "revision" in str(excinfo.value)
    assert engine.selected.approval_state is ApprovalState.PENDING


def test_approve_refuses_a_decision_aimed_at_a_different_plan():
    engine = _engine_at_gate()
    engine.approval_plan_id = "plan-from-a-previous-run"
    with pytest.raises(WorkflowError) as excinfo:
        engine.approve("operator")
    assert "fresh decision" in str(excinfo.value)


def test_approve_refuses_to_approve_twice():
    engine = _engine_at_gate()
    engine.approve("operator")
    engine._set_stage(Stage.WAIT_FOR_OPERATOR_APPROVAL)
    with pytest.raises(WorkflowError):
        engine.approve("operator")


def test_an_approval_cannot_survive_the_batch_changing():
    """The stronger property: it is revoked, not merely reported afterwards.

    Every path that changes the batch — new scenario, injected or removed item,
    retired container, segregation-rule change — funnels through
    ``_bump_scenario_revision``, so revoking there is what makes the rule
    structural instead of something each caller has to remember.
    """
    engine = _engine_at_gate()
    engine.approve("operator")
    engine._bump_scenario_revision()
    assert engine.selected.approval_state is ApprovalState.PENDING
    assert engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert engine.approval_revision == engine.scenario_revision
    assert engine.approval_inconsistency() == ""
    # And the renewed decision is grantable, not a dead gate.
    engine.approve("operator")
    assert engine.selected.approval_state is ApprovalState.APPROVED


def test_an_approval_stamped_to_an_old_revision_is_still_reported():
    """The detector itself, independent of who calls it.

    Kept because ``approval_inconsistency`` is also the guard for states this
    process did not produce — a mirror assembled from ROS topics, for instance.
    """
    engine = _engine_at_gate()
    engine.approve("operator")
    engine.approval_revision = engine.scenario_revision - 1
    assert "revision" in engine.approval_inconsistency()


# --------------------------------------------------------------------------- #
# 4. cross-component identity — the scenario/plan mismatch
# --------------------------------------------------------------------------- #

def _mirror(**parts):
    base = {
        "stage": "WAIT_FOR_OPERATOR_APPROVAL",
        "stamps": {
            "scenario": {"run_id": "run-aaaa", "scenario_revision": 0,
                         "scenario_id": "isaac_cylinders_smoke-s42"},
            "selected": {"run_id": "run-aaaa", "scenario_revision": 0,
                         "scenario_id": "isaac_cylinders_smoke-s42"},
            "plan_summary": {"run_id": "run-aaaa", "scenario_revision": 0,
                             "scenario_id": "isaac_cylinders_smoke-s42"},
        },
        "selected": {"plan_id": "plan-optimized-isaac_cylinders_smoke-s42",
                     "approval_state": "pending", "is_valid": True},
        "plan_summary": {
            "selected_plan_id": "plan-optimized-isaac_cylinders_smoke-s42",
            "approval_revision": 0,
            "approval_plan_id": "plan-optimized-isaac_cylinders_smoke-s42"},
    }
    base.update(parts)
    return base


def test_a_coherent_mirror_reports_no_conflict():
    from web.snapshot import _identity_conflict
    reason, identity = _identity_conflict(_mirror())
    assert reason == ""
    assert identity["run_id"] == "run-aaaa"


def test_the_observed_scenario_plan_mismatch_is_detected():
    """scenario mixed_pipes_dense-s42 beside plan-...-isaac_cylinders_smoke-s42."""
    from web.snapshot import _identity_conflict
    mirror = _mirror()
    mirror["stamps"]["scenario"] = {"run_id": "run-bbbb", "scenario_revision": 1,
                                    "scenario_id": "mixed_pipes_dense-s42"}
    reason, _ = _identity_conflict(mirror)
    assert reason, "a scenario and plan from different runs must be rejected"
    assert "different runs" in reason


def test_a_revision_disagreement_is_detected():
    from web.snapshot import _identity_conflict
    mirror = _mirror()
    mirror["stamps"]["selected"]["scenario_revision"] = 3
    reason, _ = _identity_conflict(mirror)
    assert "scenario_revision" in reason


def test_a_summary_naming_a_different_plan_is_detected():
    from web.snapshot import _identity_conflict
    mirror = _mirror()
    mirror["plan_summary"]["selected_plan_id"] = "plan-baseline-other"
    reason, _ = _identity_conflict(mirror)
    assert "plan summary names" in reason


def test_an_approval_for_another_plan_is_detected():
    from web.snapshot import _identity_conflict
    mirror = _mirror()
    mirror["plan_summary"]["approval_plan_id"] = "plan-from-the-previous-run"
    reason, _ = _identity_conflict(mirror)
    assert "fresh decision is required" in reason


def test_an_approval_for_another_revision_is_detected():
    from web.snapshot import _identity_conflict
    mirror = _mirror()
    mirror["plan_summary"]["approval_revision"] = 7
    reason, _ = _identity_conflict(mirror)
    assert "fresh decision is required" in reason


def test_unstamped_components_are_not_treated_as_conflicting():
    """An older publisher is not a conflicting one — controls must not vanish."""
    from web.snapshot import _identity_conflict
    mirror = _mirror()
    mirror["stamps"] = {}
    mirror["plan_summary"].pop("approval_revision")
    mirror["plan_summary"].pop("approval_plan_id")
    reason, _ = _identity_conflict(mirror)
    assert reason == ""


def test_an_inconsistent_snapshot_withholds_every_operator_control():
    from web.snapshot import DashboardSnapshot
    snap = DashboardSnapshot(
        mode="ros",
        control_stage="WAIT_FOR_OPERATOR_APPROVAL",
        control_approval_state="pending",
        control_plan_id="plan-x",
        control_plan_valid=True,
        control_inconsistency="the scenario and the selected plan describe "
                              "different runs")
    assert snap.can_approve is False
    assert "different runs" in snap.approval_block_reason
    control = snap.control_state()
    assert control["consistent"] is False
    assert control["can_approve"] is False
    assert control["can_reject"] is False
    assert control["can_alternative"] is False


def test_the_guard_runs_before_every_other_block_reason():
    """Otherwise the operator is told about a hold that belongs to another run."""
    from web.snapshot import DashboardSnapshot
    snap = DashboardSnapshot(
        mode="ros",
        control_stage="WAIT_FOR_OPERATOR_APPROVAL",
        control_anomaly_hold=True,
        control_inconsistency="components describe different runs")
    assert snap.approval_block_reason == "components describe different runs"


# --------------------------------------------------------------------------- #
# 5. the UI never says "Decision required" for an approved plan
# --------------------------------------------------------------------------- #

def test_the_dashboard_never_asks_for_a_decision_on_an_approved_plan():
    src = _read(os.path.join(REPO, "web", "index.html"))
    start = src.index("const box = $(\"#approval-warn\");")
    banner = src[start:src.index("const ctl = s.control || {};", start)]
    approved_at = banner.index('ctlApproved')
    decision_at = banner.index('"Decision required')
    assert approved_at < decision_at, (
        "the approved branch must be evaluated BEFORE the 'Decision required' "
        "branch, or an approved plan still renders the contradiction")
    assert "No further decision is required" in banner


def test_the_dashboard_states_why_an_approved_plan_is_not_moving():
    src = _read(os.path.join(REPO, "web", "index.html"))
    assert "Execution is HELD by an anomaly" in src
    assert "Waiting for the physical scene" in src


def test_the_dashboard_withholds_controls_on_an_inconsistent_snapshot():
    src = _read(os.path.join(REPO, "web", "index.html"))
    assert "Inconsistent state — controls withheld" in src


# --------------------------------------------------------------------------- #
# 6. anomaly acknowledgement: execution or a genuine pending decision, never both
# --------------------------------------------------------------------------- #

_CLASS_FOR = {"warning": "CAMERA_VIEW_LOST",
              "critical": "SHEAR_POSITION_TOO_HIGH"}


@pytest.mark.parametrize("severity", ["warning", "critical"])
def test_acknowledgement_never_leaves_the_contradictory_state(severity):
    from wisepack_core.anomaly import AnomalyClass, AnomalyEvent, Severity

    engine = _engine_at_gate()
    engine.approve("operator")
    engine.apply_anomaly(AnomalyEvent(anomaly_class=getattr(AnomalyClass, _CLASS_FOR[severity]),
                                      severity=Severity(severity)))
    engine.acknowledge_anomaly("operator")

    assert not engine.anomaly_hold
    assert engine.approval_inconsistency() == "", (
        "acknowledgement must lead to execution or to a genuine pending "
        "decision, never to 'at the gate and already approved'")
    if engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL:
        assert engine.selected.approval_state is ApprovalState.PENDING
    else:
        assert engine.selected.approval_state is ApprovalState.APPROVED


def test_a_warning_acknowledgement_resumes_without_re_approval():
    from wisepack_core.anomaly import AnomalyClass, AnomalyEvent, Severity

    engine = _engine_at_gate()
    engine.approve("operator")
    engine.apply_anomaly(AnomalyEvent(anomaly_class=AnomalyClass.CAMERA_VIEW_LOST,
                                      severity=Severity.WARNING))
    engine.acknowledge_anomaly("operator")
    assert engine.selected.approval_state is ApprovalState.APPROVED
    assert engine.stage is not Stage.WAIT_FOR_OPERATOR_APPROVAL


def test_a_critical_acknowledgement_requires_a_genuine_new_decision():
    from wisepack_core.anomaly import AnomalyClass, AnomalyEvent, Severity

    engine = _engine_at_gate()
    engine.approve("operator")
    engine.apply_anomaly(AnomalyEvent(anomaly_class=AnomalyClass.SHEAR_POSITION_TOO_HIGH,
                                      severity=Severity.CRITICAL))
    engine.acknowledge_anomaly("operator")
    assert engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert engine.selected.approval_state is ApprovalState.PENDING
    # And the decision must actually be grantable — not a dead gate.
    engine.approve("operator")
    assert engine.selected.approval_state is ApprovalState.APPROVED
    assert engine.stage is not Stage.WAIT_FOR_OPERATOR_APPROVAL


# --------------------------------------------------------------------------- #
# Structure — the invariant has ONE enforcement point
# --------------------------------------------------------------------------- #

def test_the_gate_is_entered_only_through_request_approval_or_revoke():
    """Nothing may set the stage to the gate while leaving approval untouched."""
    src = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_core",
                             "wisepack_core", "workflow.py"))
    hits = []
    for line_no, line in enumerate(src.splitlines(), 1):
        if "_set_stage(Stage.WAIT_FOR_OPERATOR_APPROVAL)" in line:
            hits.append(line_no)
    # request_approval() and revoke_approval() — and nowhere else.
    assert len(hits) == 2, (
        f"the approval gate is entered from {len(hits)} places; every one of "
        "them must also set the approval state, so they belong in "
        "request_approval() or revoke_approval()")


def test_the_behaviour_tree_gate_checks_approval_state_not_just_the_stage():
    src = _read(os.path.join(REPO, "wisepack_ws", "src",
                             "wisepack_orchestration", "wisepack_orchestration",
                             "hitl_orchestrator.py"))
    start = src.index("class AwaitApproval")
    body = src[start:src.index("class ExecuteLoop")]
    assert "approval_state is ApprovalState.PENDING" in body
    assert "approval_plan_id" in body and "approval_revision" in body
    assert "resume_execution_stage()" in body
