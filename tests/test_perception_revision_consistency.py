"""A physical batch must leave every published document on ONE revision.

THE DEFECT, observed on the live camera path and reproduced here without a
camera, a GPU, a model or ROS:

    Inconsistent state — controls withheld:
    the scenario and the selected plan describe different runs
    (scenario_revision 1 vs 2)

The workflow had reached WAIT_FOR_OPERATOR_APPROVAL with two genuine camera
observations, the Digital Twin rendered the right two objects, and the operator
could not approve anything — permanently.

WHY IT HAPPENED. A physical scan is not a counting operation.
`WorkflowEngine.apply_observation_batch` REPLACES the scenario's item list with
the observed proxies and advances `scenario_revision`. The orchestrator's
`ScanAndDetect` behaviour published only the detected COUNT, so the scenario
topics kept describing the pre-scan generated items, stamped with the previous
revision, while the plans built from the camera batch went out stamped with the
new one. Nothing on that path republished the scenario, so the mixture was
permanent.

THE GATE WAS RIGHT AND IS NOT TOUCHED HERE. `web/snapshot.py::_identity_conflict`
is imported and run unchanged by these tests: what changed is the state the
orchestrator publishes, not the standard it is held to.

Everything below drives the REAL orchestrator through `orchestrator_harness`.
Its publishers record instead of transporting, so "what the dashboard would
read" is exactly "the newest message on each topic" — which is what a latched
DDS topic gives a late-joining subscriber.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, "tests") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "tests"))

from orchestrator_harness import build_orchestrator          # noqa: E402

from wisepack_bringup import topics as T                     # noqa: E402
from wisepack_core.domain import ApprovalState, PhysicalObservation  # noqa: E402
from wisepack_core.events import Stage                       # noqa: E402
from wisepack_core.perception import (                       # noqa: E402
    BatchStatus, ObservationBatch,
)

#: The two objects the live camera actually measured on the calibration board,
#: to 0.1 mm. Using the real numbers rather than round ones keeps the fixture
#: honest — including the first object at x = 184.5 mm, which is OUTSIDE the
#: declared 130 x 130 mm work area and must stay visible as a diagnostic rather
#: than be clamped.
MEASURED_A = [(184.5, 54.1, 51.7), (77.5, 53.9, -21.1)]

#: A second detection: the operator moved the proxies and scanned again.
MEASURED_B = [(64.2, 41.8, 12.0), (95.1, 88.7, -140.3), (30.0, 30.0, 5.0)]


def _batch(batch_id: str, poses, *, captured_at="2026-08-09T10:00:00.000Z"
           ) -> ObservationBatch:
    """One camera batch, shaped exactly as the provider's adapter emits it."""
    return ObservationBatch(
        batch_id=batch_id,
        source="camera",
        status=BatchStatus.OK,
        captured_at=captured_at,
        requested_at="2026-08-09T09:59:59.000Z",
        detector="fasterrcnn_resnet50_fpn/bottle",
        model_id="/data/arise/models/best_model.pth",
        calibration_status="valid",
        calibration_revision="8074644730b3",
        observations=[
            PhysicalObservation(
                observation_id=f"physical-cylinder-{index + 1:03d}",
                x_mm=x, y_mm=y, yaw_deg=yaw,
                confidence=0.99,
                object_type="cylindrical_proxy",
                source="camera",
                detector="fasterrcnn_resnet50_fpn/bottle",
                captured_at=captured_at,
                calibration_status="valid",
                calibration_revision="8074644730b3")
            for index, (x, y, yaw) in enumerate(poses)
        ])


@pytest.fixture
def camera_run():
    """An orchestrator in camera mode, held at the scan waiting for a batch.

    This is the real waiting state: with a physical source there are no objects
    until the operator triggers a detection, and `ScanAndDetect` returns RUNNING
    rather than inventing simulated ones.
    """
    harness = build_orchestrator(perception_source="camera")
    harness.tick(2)
    assert harness.node.engine.stage is Stage.DETECT_ITEMS
    assert not harness.node.engine.detected
    return harness


def _detect(harness, batch: ObservationBatch) -> None:
    """Deliver one batch the way the perception worker does, then let it settle.

    The worker parks the batch and the TICK adopts it — never the other way
    round, because adopting mutates the engine and re-plans.
    """
    harness.node._pending_observation = batch
    harness.tick_until_gate()


#: Every document an operator's approval decision depends on, and the topic it
#: travels on. If any of these disagrees about the revision, the decision on
#: screen is not a decision about what is on screen.
APPROVAL_RELEVANT_TOPICS = (
    ("scenario config", T.SCENARIO_CONFIG),
    ("scenario state", T.SCENARIO_STATE),
    ("selected plan", T.PLAN_SELECTED),
    ("plan summary", T.PLAN_SUMMARY),
)


def _revisions(harness):
    return {label: (harness.latest(topic) or {}).get("scenario_revision")
            for label, topic in APPROVAL_RELEVANT_TOPICS}


def _run_ids(harness):
    return {label: (harness.latest(topic) or {}).get("run_id")
            for label, topic in APPROVAL_RELEVANT_TOPICS}


# --------------------------------------------------------------------------- #
# 1. One batch -> one coherent published state
# --------------------------------------------------------------------------- #


def test_a_physical_batch_leaves_every_document_on_the_same_revision(camera_run):
    """THE REGRESSION. Before the fix: scenario 1, plans 2, controls withheld."""
    engine = camera_run.node.engine
    revision_before = engine.scenario_revision

    _detect(camera_run, _batch("batch-001", MEASURED_A))

    assert engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    # The batch advanced the revision — that part was always right.
    assert engine.scenario_revision == revision_before + 1

    revisions = _revisions(camera_run)
    assert set(revisions.values()) == {engine.scenario_revision}, (
        "published documents disagree about the revision: " + repr(revisions))
    run_ids = _run_ids(camera_run)
    assert set(run_ids.values()) == {engine.run_id}, repr(run_ids)


def test_the_dashboard_consistency_gate_is_satisfied(camera_run):
    """The production gate, imported and run unchanged, must find nothing."""
    _detect(camera_run, _batch("batch-001", MEASURED_A))
    assert camera_run.inconsistency() == "", (
        "the operator panel would still withhold controls")


def test_the_published_scenario_describes_the_physical_objects(camera_run):
    """The stale STAMP was telling the truth: the document was stale too.

    A revision number that agrees while the item list still describes the
    generated scenario would be a worse bug than the one being fixed — it would
    look coherent.
    """
    _detect(camera_run, _batch("batch-001", MEASURED_A))

    items = camera_run.latest(T.WASTE_ITEMS)
    assert isinstance(items, list) and len(items) == len(MEASURED_A)
    measured = [(round(i["observation"]["pose"]["x_mm"], 1),
                 round(i["observation"]["pose"]["y_mm"], 1),
                 round(i["observation"]["pose"]["yaw_deg"], 1)) for i in items]
    assert measured == MEASURED_A
    assert all(i["observation"]["source"] == "camera" for i in items)

    totals = camera_run.latest(T.SCENARIO_STATE) or {}
    assert totals.get("items") == len(MEASURED_A)


def test_the_out_of_work_area_observation_is_published_not_clamped(camera_run):
    """x = 184.5 mm on a 130 mm plane is a real measurement and a real warning."""
    _detect(camera_run, _batch("batch-001", MEASURED_A))
    items = camera_run.latest(T.WASTE_ITEMS)
    assert any(i["observation"]["pose"]["x_mm"] > 130.0 for i in items), (
        "the out-of-work-area measurement was moved or dropped")


def test_approval_is_genuinely_pending_for_the_new_revision(camera_run):
    """At the gate the decision must be about the plan and batch on screen."""
    _detect(camera_run, _batch("batch-001", MEASURED_A))
    engine = camera_run.node.engine

    assert engine.selected.approval_state is ApprovalState.PENDING
    assert engine.approval_revision == engine.scenario_revision
    assert engine.approval_plan_id == engine.selected.plan_id
    assert engine.approval_inconsistency() == ""

    summary = camera_run.latest(T.PLAN_SUMMARY) or {}
    selected = camera_run.latest(T.PLAN_SELECTED) or {}
    assert summary["approval_state"] == "pending"
    assert summary["approval_revision"] == engine.scenario_revision
    assert summary["approval_plan_id"] == selected["plan_id"]
    assert summary["selected_plan_id"] == selected["plan_id"]


def test_the_plans_were_built_from_the_camera_observations(camera_run):
    """Coherence is worthless if the plan is not about the detected objects."""
    _detect(camera_run, _batch("batch-001", MEASURED_A))
    plan = camera_run.latest(T.PLAN_SELECTED) or {}
    placements = plan.get("placements") or []
    assert len(placements) == len(MEASURED_A)


# --------------------------------------------------------------------------- #
# 2. A second detection revokes the first decision
# --------------------------------------------------------------------------- #


def test_a_second_batch_revokes_the_previous_approval_and_re_stamps(camera_run):
    """detection A -> approve -> detection B -> the old authorisation is void.

    The scenario the operator approved no longer exists. Carrying the approval
    across would authorise a pick against objects that have moved.
    """
    engine = camera_run.node.engine

    _detect(camera_run, _batch("batch-001", MEASURED_A))
    revision_a = engine.scenario_revision
    plan_a = engine.selected.plan_id
    engine.approve("operator")
    assert engine.selected.approval_state is ApprovalState.APPROVED

    # The operator moved the proxies and detected again. After the first scan
    # the engine has objects, so this batch travels the adoption path rather
    # than ScanAndDetect — both must produce the same coherent result.
    camera_run.node._pending_observation = _batch(
        "batch-002", MEASURED_B, captured_at="2026-08-09T10:05:00.000Z")
    camera_run.tick_until_gate()

    revision_b = engine.scenario_revision
    assert revision_b > revision_a

    # The old authorisation is gone, and a fresh decision is genuinely pending.
    assert engine.selected.approval_state is ApprovalState.PENDING
    assert engine.approval_revision == revision_b
    assert engine.approval_plan_id == engine.selected.plan_id

    # Everything published agrees on the NEW revision.
    revisions = _revisions(camera_run)
    assert set(revisions.values()) == {revision_b}, repr(revisions)
    assert camera_run.inconsistency() == ""

    # And the new plan is about the new objects.
    items = camera_run.latest(T.WASTE_ITEMS)
    assert len(items) == len(MEASURED_B)
    assert plan_a is not None


def test_re_detecting_at_the_gate_returns_to_the_gate_on_the_new_revision(
        camera_run):
    """The SECOND manifestation: detect again WITHOUT approving the first result.

    `AwaitApproval` asks for a decision from `initialise()`, and py_trees runs
    `initialise()` only when the behaviour was not already RUNNING — which is
    exactly what it is while parked at the gate. So the re-planned batch used to
    leave the stage at DIGITAL_TWIN_VALIDATE with the approval stamp still
    pointing at the previous revision, and no control was ever offered again.
    """
    engine = camera_run.node.engine

    _detect(camera_run, _batch("batch-001", MEASURED_A))
    revision_a = engine.scenario_revision
    assert engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL

    # No approval. The operator moved the proxies and pressed detect again.
    camera_run.node._pending_observation = _batch("batch-002", MEASURED_B)
    camera_run.tick(2)

    assert engine.scenario_revision == revision_a + 1
    assert engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL, (
        "the workflow did not return to the approval gate for the new batch")
    assert engine.approval_revision == engine.scenario_revision
    assert engine.approval_plan_id == engine.selected.plan_id
    assert engine.selected.approval_state is ApprovalState.PENDING

    assert set(_revisions(camera_run).values()) == {engine.scenario_revision}
    assert camera_run.inconsistency() == ""
    assert len(camera_run.latest(T.WASTE_ITEMS)) == len(MEASURED_B)


def test_before_any_detection_there_is_no_pending_decision_to_be_stale(
        camera_run):
    """`approval_revision` is `null` until a decision exists, never 0.

    Published as 0 it read as "a decision about revision 0", so the moment the
    first scan advanced the revision the gate reported a stale decision nobody
    had made — and withheld controls on a run that had not reached the gate.
    """
    summary = camera_run.latest(T.PLAN_SUMMARY) or {}
    assert summary.get("approval_plan_id") is None
    assert summary.get("approval_revision") is None
    assert camera_run.inconsistency() == ""


def test_an_approval_from_the_old_batch_cannot_authorise_the_new_plan(camera_run):
    """The safety property, asserted at the engine rather than at the UI."""
    from wisepack_core.workflow import WorkflowError          # noqa: PLC0415

    engine = camera_run.node.engine
    _detect(camera_run, _batch("batch-001", MEASURED_A))
    stale_revision = engine.approval_revision

    camera_run.node._pending_observation = _batch("batch-002", MEASURED_B)
    camera_run.tick_until_gate()

    # Re-stamping the approval to the OLD revision is the shape of the mistake
    # this guards against — carrying a decision across a batch change.
    engine.approval_revision = stale_revision
    with pytest.raises(WorkflowError) as exc:
        engine.approve("operator")
    assert "revision" in str(exc.value)
    assert engine.selected.approval_state is not ApprovalState.APPROVED


# --------------------------------------------------------------------------- #
# 3. The mechanism itself
# --------------------------------------------------------------------------- #


def test_the_scenario_is_republished_only_when_the_revision_moved(camera_run):
    """Cheap by construction: a no-op when nothing changed.

    A guard that republished unconditionally would put the whole item list on
    DDS on every tick of a 40-item run.
    """
    node = camera_run.node
    _detect(camera_run, _batch("batch-001", MEASURED_A))

    before = camera_run.topic(T.SCENARIO_STATE).count
    assert node.publish_scenario_if_stale() is False
    assert camera_run.topic(T.SCENARIO_STATE).count == before

    # ... and it fires exactly once when the revision does move.
    node.engine._bump_scenario_revision()
    assert node.publish_scenario_if_stale() is True
    assert camera_run.topic(T.SCENARIO_STATE).count == before + 1
    assert node.publish_scenario_if_stale() is False


def test_simulated_perception_publishes_one_revision_as_it_always_did():
    """The default path must be unchanged — and it never had this bug."""
    harness = build_orchestrator(perception_source="sim")
    harness.tick_until_gate()
    engine = harness.node.engine
    assert engine.scenario_revision == 1
    assert set(_revisions(harness).values()) == {1}
    assert harness.inconsistency() == ""
    assert engine.selected.approval_state is ApprovalState.PENDING


def test_a_failed_batch_neither_advances_the_revision_nor_plans(camera_run):
    """§15: a failed scan is not an empty successful one, and changes nothing."""
    engine = camera_run.node.engine
    revision_before = engine.scenario_revision

    camera_run.node._pending_observation = ObservationBatch.failed(
        batch_id="batch-bad", source="camera",
        error="the detector resolved no valid ArUco calibration for this frame")
    camera_run.tick(3)

    assert engine.scenario_revision == revision_before
    assert engine.stage is not Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert engine.selected is None
    # Nothing incoherent was published on the way to failing.
    assert camera_run.inconsistency() == ""
