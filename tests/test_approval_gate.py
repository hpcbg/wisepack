"""Approving the simulated-RGB-D plan must not cross an execution boundary.

WHAT "SAFE TO PRESS" HAS TO MEAN. Approval records a decision about one plan of
one batch revision. It must not, by itself, invoke an execution adapter, solve
IK, command an arm or a gripper, or dispatch an execution request — because the
operator pressing Approve in Stage E is authorising a plan, not starting a
machine.

The boundary is `WorkflowEngine.begin_physical_item`: it is what an execution
adapter calls to command a physical pick, and it is the only path that emits
`isaac_pick_commanded`. These tests spy on it and assert it never runs.

They use the EXISTING gate. Nothing here introduces a Stage-E-specific approval
path; the assertions are about the ordinary `approve()`.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))

from wisepack_core.domain import PhysicalObservation                 # noqa: E402
from wisepack_core.execution import ExecutionBackend                 # noqa: E402
from wisepack_core.generator import build_scenario                   # noqa: E402
from wisepack_core.perception import (BatchStatus, ObservationBatch,  # noqa: E402
                                      PerceptionSource)
from wisepack_core.workflow import (ApprovalState, Stage,            # noqa: E402
                                    WorkflowConfig, WorkflowEngine,
                                    WorkflowError)

STAGE_C = os.path.join(REPO, ".cache-perception", "stage-c", "stage_c.json")


def _observation():
    """The real Stage C observation when present, else an equivalent stand-in.

    The gate's behaviour does not depend on which, and the test must not require
    a GPU, Isaac or a prior acquisition to run in ordinary CI.
    """
    if os.path.isfile(STAGE_C):
        with open(STAGE_C, encoding="utf-8") as handle:
            return PhysicalObservation.from_dict(json.load(handle)["observation"])
    from wisepack_core.pose import Orientation, Symmetry, SymmetryType
    return PhysicalObservation(
        observation_id="sim-1", x_mm=383.63, y_mm=-240.84, z_mm=-45.12,
        object_type="pipe_section", source=PerceptionSource.CAMERA.value,
        frame_id="wisepack_workarea", orientation=Orientation.identity(),
        symmetry=Symmetry(type=SymmetryType.DISCRETE, axis="z", fold=2),
        perception_method="foundationpose_rgbd", object_model_id="cylinder5",
        diameter_mm=25, length_mm=342, inner_diameter_mm=19,
        model_center_mm=(-130.0, -54.44, 0.0),
        task_axis_vector=(0.9284, -0.3716, 0.0),
        pose_valid=True, workarea_transform_valid=True)


def _engine_at_the_gate():
    """A run built exactly as the dashboard builds one, stopped at approval."""
    engine = WorkflowEngine(WorkflowConfig(
        preset="cad_cylinder5_single",
        perception_source=PerceptionSource.CAMERA))
    engine.generate_or_load_scenario(build_scenario("cad_cylinder5_single"))
    batch = ObservationBatch(
        batch_id="simulated-rgbd-1", source=PerceptionSource.CAMERA.value,
        status=BatchStatus.OK, observations=[_observation()],
        frame_id="wisepack_workarea", acquisition="simulated_rgbd",
        perception_method="foundationpose_rgbd", model_id="cylinder5",
        calibration_status="not_applicable")
    engine.apply_observation_batch(batch)
    engine.generate_plans()
    engine.digital_twin_validate()
    engine.request_approval()
    return engine


@pytest.fixture
def execution_spy(monkeypatch):
    """Records any call to the physical-execution boundary. Never lets it run."""
    calls = []

    def refuse(self, placement):
        calls.append(placement.item_id)
        raise AssertionError(
            "begin_physical_item was called — approval crossed the execution "
            "boundary")

    monkeypatch.setattr(WorkflowEngine, "begin_physical_item", refuse)
    return calls


# --------------------------------------------------------------------------- #
# The gate itself
# --------------------------------------------------------------------------- #


def test_the_run_stops_at_the_gate_before_anything_is_approved(execution_spy):
    engine = _engine_at_the_gate()
    assert engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert engine.selected.approval_state is ApprovalState.PENDING
    assert not execution_spy


def test_approve_authorises_exactly_the_current_revision(execution_spy):
    engine = _engine_at_the_gate()
    revision = engine.scenario_revision
    engine.approve(operator="test operator")
    assert engine.approval_revision == revision
    assert engine.scenario_revision == revision
    assert engine.selected.approval_state is ApprovalState.APPROVED
    assert not execution_spy


def test_approve_moves_the_workflow_state_and_nothing_else(execution_spy):
    engine = _engine_at_the_gate()
    engine.approve(operator="test operator")
    # The stage advances to the first execution STEP — which is a workflow
    # position, not a command. Nothing has been dispatched.
    assert engine.stage is Stage.PICK_ITEM
    assert engine.progress_pct == 0.0
    assert all(not p.executed for p in engine.selected.placements)
    assert not execution_spy


def test_approve_invokes_no_execution_adapter(execution_spy):
    """§: no adapter, no IK, no arm, no gripper, no dispatch.

    The dashboard's engine holds no execution adapter at all — the backend is
    the logical simulator — and the one boundary an adapter would call is spied
    on above.
    """
    engine = _engine_at_the_gate()
    engine.approve(operator="test operator")
    assert engine.config.execution_backend is ExecutionBackend.SIMULATED
    assert getattr(engine, "isaac", None) is None
    assert not execution_spy


def test_approving_dispatches_no_execution_request(execution_spy):
    """The audit trail is the evidence: a commanded pick emits
    `isaac_pick_commanded`, and approving must produce no such event."""
    engine = _engine_at_the_gate()
    engine.approve(operator="test operator")
    actions = [event.to_dict().get("action", "") for event in engine.log.events()]
    assert "approve_plan" in actions
    for action in actions:
        assert "commanded" not in action, action
        assert "isaac_" not in action, action
    assert not execution_spy


def test_a_stale_approval_is_refused_by_the_existing_gate(execution_spy):
    """Unchanged behaviour, re-asserted here because Stage E must not have
    weakened it: a decision about a superseded revision authorises nothing."""
    engine = _engine_at_the_gate()
    # A new observation arrives while the operator is deciding.
    engine.apply_observation_batch(ObservationBatch(
        batch_id="simulated-rgbd-2", source=PerceptionSource.CAMERA.value,
        status=BatchStatus.OK, observations=[_observation()],
        frame_id="wisepack_workarea", acquisition="simulated_rgbd",
        perception_method="foundationpose_rgbd", model_id="cylinder5",
        calibration_status="not_applicable"))
    with pytest.raises(WorkflowError) as exc:
        engine.approve(operator="test operator")
    # THE REFUSAL IS STRONGER THAN THE REVISION CHECK. A new batch returns the
    # workflow to DETECT_ITEMS, so the decision is refused before the revision
    # comparison is even reached — there is no approval stage to be in. Either
    # message is a refusal; what matters is that nothing was authorised.
    message = str(exc.value)
    assert ("expected WAIT_FOR_OPERATOR_APPROVAL" in message
            or "revision" in message), message
    assert engine.stage is not Stage.PICK_ITEM
    assert not execution_spy


def test_the_revision_guard_itself_refuses_a_superseded_decision(execution_spy):
    """The other half: the scenario changes while the operator is AT the gate,
    so the stage is still right and only the revision has moved."""
    engine = _engine_at_the_gate()
    # An item arrives mid-decision. This bumps the batch revision and revokes
    # any outstanding authorisation, without leaving the approval stage.
    engine._bump_scenario_revision()
    with pytest.raises(WorkflowError) as exc:
        engine.approve(operator="test operator")
    assert "revision" in str(exc.value)
    assert engine.selected.approval_state is not ApprovalState.APPROVED
    assert not execution_spy


def test_execution_only_advances_when_something_steps_it(execution_spy):
    """Approval authorises; it does not drive. The simulated execution advances
    only when `step_execution` is called, which the dashboard does from its own
    loop and which a manual test can simply not trigger."""
    engine = _engine_at_the_gate()
    engine.approve(operator="test operator")
    before = engine.progress_pct
    assert before == 0.0
    # Nothing has moved because nothing stepped it.
    assert all(not p.executed for p in engine.selected.placements)
    assert not execution_spy
