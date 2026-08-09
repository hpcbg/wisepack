"""ONE application lifetime, both object sources, no restart anywhere.

THE DEFECT THIS PINS. `WISEPACK_PERCEPTION_SOURCE` used to be a launch-time lock
between two workflows: a session started on presets could not detect, and a
session started on the camera could not go back. Demonstrating both meant
stopping WISEPACK and starting it again — in front of an audience, between two
halves of the same story.

The object source is now a PER-RUN selection with three separate parts, and
every test here is about keeping them separate:

    available   what this deployment can do, asked live
    selected    what the NEXT run will use — a draft, inert until spent
    current     what the RUNNING run actually used — provenance

The whole sequence below runs against ONE `HitLOrchestrator` instance, which is
the assertion: `harness.node` is never rebuilt, so nothing in it restarted.

No camera, no torch, no ROS: the perception service is a stub that answers the
capability question and hands back a batch, which is exactly the surface the
orchestrator uses.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, "tests") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "tests"))

from orchestrator_harness import build_orchestrator            # noqa: E402

from wisepack_bringup import topics as T                       # noqa: E402
from wisepack_core.domain import ApprovalState, PhysicalObservation  # noqa: E402
from wisepack_core.events import Stage                         # noqa: E402
from wisepack_core.perception import (                         # noqa: E402
    BatchStatus, ObjectSourceState, ObservationBatch, PerceptionConfigError,
    PerceptionSource, resolve_object_source,
)

#: Two objects, as the camera actually measured them.
MEASURED = [(184.5, 54.1, 51.7), (77.5, 53.9, -21.1)]


def _batch(batch_id: str, poses) -> ObservationBatch:
    return ObservationBatch(
        batch_id=batch_id, source="camera", status=BatchStatus.OK,
        captured_at="2026-08-09T10:00:00.000Z",
        requested_at="2026-08-09T09:59:59.000Z",
        detector="fasterrcnn_resnet50_fpn/bottle",
        calibration_status="valid", calibration_revision="8074644730b3",
        observations=[
            PhysicalObservation(
                observation_id=f"physical-cylinder-{i + 1:03d}",
                x_mm=x, y_mm=y, yaw_deg=yaw, confidence=0.99,
                object_type="cylindrical_proxy", source="camera",
                captured_at="2026-08-09T10:00:00.000Z",
                calibration_status="valid")
            for i, (x, y, yaw) in enumerate(poses)])


class _StubService:
    """The perception service, reduced to what the orchestrator asks of it.

    `available` is a mutable flag so a test can take the camera away mid-session
    — which is the point of treating availability as a live question rather than
    a start-up decision.
    """

    url = "http://127.0.0.1:22101"

    def __init__(self, available: bool = True, poses=None) -> None:
        self.available = available
        self.poses = poses or MEASURED
        self.detections = 0

    def capability(self, health=None):
        if self.available:
            return True, ""
        return False, "no perception service is answering at " + self.url

    def detect(self) -> ObservationBatch:
        self.detections += 1
        return _batch(f"batch-{self.detections:03d}", self.poses)


@pytest.fixture
def session():
    """One orchestrator, started EXACTLY as `./run_wisepack_dashboard.sh` does.

    No `WISEPACK_PERCEPTION_SOURCE`, so the session opens on the preset
    workflow — the unchanged default — with a camera that happens to be
    available.
    """
    harness = build_orchestrator(perception_source="sim",
                                 preset="mixed_pipes_dense")
    harness.node.perception_client = _StubService()
    # The capability cache was primed during construction against the real
    # (absent) service; drop it so the stub is consulted.
    harness.node._camera_capability_at = None
    harness.tick_until_gate()
    return harness


def _items(harness) -> int:
    return len(harness.node.engine.scenario.items)


def _state(harness) -> ObjectSourceState:
    return harness.node.object_source_state()


# --------------------------------------------------------------------------- #
# 1. The default workflow is untouched
# --------------------------------------------------------------------------- #


def test_the_session_opens_on_the_preset_workflow(session):
    """Step 1-5: start normally, generate a preset scenario, approval available."""
    engine = session.node.engine
    assert engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert engine.config.perception_source is PerceptionSource.SIM
    assert _items(session) > 10, "the dense preset must produce a real batch"
    assert engine.selected.approval_state is ApprovalState.PENDING
    assert session.inconsistency() == ""

    state = _state(session)
    assert state.current == "sim" and state.selected == "sim"
    assert state.to_dict()["action_label"] == "Generate & plan"
    assert state.to_dict()["current_provenance"] == "preset/generated"


def test_the_camera_is_offered_without_being_selected(session):
    """A capability the session did not start on is still a capability."""
    state = _state(session)
    assert state.camera_available
    assert set(state.available) == {"sim", "camera"}
    assert state.current == "sim", "offering a camera must not select one"


# --------------------------------------------------------------------------- #
# 2. Switching to the camera, and back, in one process
# --------------------------------------------------------------------------- #


def test_selecting_the_camera_does_not_touch_the_running_preset_run(session):
    """Step 6: the draft moves; the run on screen does not.

    This is the robot selector's rule applied to the object source, and it is
    what makes the switch safe to offer while a plan is awaiting a decision.
    """
    engine = session.node.engine
    before = (engine.run_id, engine.scenario_revision, _items(session),
              engine.selected.plan_id, engine.selected.approval_state)

    session.node.set_object_source("camera")

    state = _state(session)
    assert state.selected == "camera"
    assert state.current == "sim"
    assert state.changes_next_run
    assert state.to_dict()["action_label"] == "Detect & plan"
    after = (engine.run_id, engine.scenario_revision, _items(session),
             engine.selected.plan_id, engine.selected.approval_state)
    assert before == after, "selecting a source mutated the running run"


def test_one_lifetime_preset_then_camera_then_preset(session):
    """Steps 1-14, in order, against ONE orchestrator instance."""
    node = session.node
    identity = id(node)
    engine = node.engine

    # 3-5. preset
    preset_items = _items(session)
    preset_run = engine.run_id
    assert preset_items > 10
    assert engine.selected.approval_state is ApprovalState.PENDING

    # 6-7. switch to the camera and detect
    node.set_object_source("camera")
    node._apply_command("detect_physical_objects", {})
    session.tick_until_gate()
    engine = node.engine                       # a NEW run, same process

    # 8. the Digital Twin's contents changed to exactly the detected objects
    assert _items(session) == len(MEASURED)
    assert engine.config.perception_source is PerceptionSource.CAMERA
    assert _state(session).current == "camera"
    assert engine.run_id != preset_run, "a source change must be a new run"

    # 9. fresh approval, on a coherent state set
    assert engine.selected.approval_state is ApprovalState.PENDING
    assert engine.approval_revision == engine.scenario_revision
    assert engine.approval_plan_id == engine.selected.plan_id
    assert session.inconsistency() == ""
    assert len(session.latest(T.WASTE_ITEMS)) == len(MEASURED)

    # 10-12. back to a preset, still without restarting
    node.set_object_source("sim")
    node._apply_command("reset", {"preset": "mixed_pipes_small", "seed": 7})
    engine = node.engine
    assert engine.config.perception_source is PerceptionSource.SIM
    assert _state(session).current == "sim"
    assert _items(session) > 2, "the preset generator must be back in charge"
    assert _items(session) != len(MEASURED)

    # 13. fresh approval again
    assert engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert engine.selected.approval_state is ApprovalState.PENDING
    assert engine.approval_revision == engine.scenario_revision
    assert session.inconsistency() == ""

    # 14. and nothing restarted: same node, same publishers, all the way through
    assert id(node) == identity
    assert node is session.node


def test_an_approval_never_crosses_a_source_change(session):
    """Approve a preset run, then detect: the authorisation must not survive.

    The objects the operator authorised do not exist any more — they were
    generated, and the new ones were measured from a table.
    """
    node = session.node
    node.engine.approve("operator")
    assert node.engine.selected.approval_state is ApprovalState.APPROVED
    approved_plan = node.engine.selected.plan_id
    approved_run = node.engine.run_id

    node.set_object_source("camera")
    node._apply_command("detect_physical_objects", {})
    session.tick_until_gate()

    engine = node.engine
    assert engine.run_id != approved_run
    assert engine.selected.approval_state is ApprovalState.PENDING
    assert engine.approval_revision == engine.scenario_revision
    # Even if the plan id repeats (they are derived from preset and seed), the
    # decision is stamped to THIS run's revision and nothing older can pass.
    assert engine.approval_plan_id == engine.selected.plan_id
    assert approved_plan  # the old one existed; it simply no longer authorises


def test_every_switch_publishes_a_coherent_identity(session):
    """run_id and scenario_revision stay coherent across both switches."""
    node = session.node
    for step in ("camera", "sim"):
        if step == "camera":
            node.set_object_source("camera")
            node._apply_command("detect_physical_objects", {})
            session.tick_until_gate()
        else:
            node.set_object_source("sim")
            node._apply_command("reset", {"preset": "mixed_pipes_small"})
        engine = node.engine
        for topic in (T.SCENARIO_CONFIG, T.SCENARIO_STATE, T.PLAN_SELECTED,
                      T.PLAN_SUMMARY):
            doc = session.latest(topic) or {}
            assert doc.get("run_id") == engine.run_id, (step, topic)
            assert doc.get("scenario_revision") == engine.scenario_revision, (
                step, topic)
        assert session.inconsistency() == "", step


# --------------------------------------------------------------------------- #
# 3. An unavailable camera never becomes a preset scenario
# --------------------------------------------------------------------------- #


def test_preset_operation_is_unaffected_when_the_camera_is_absent():
    """A deployment with no perception service is an ordinary WISEPACK."""
    harness = build_orchestrator(perception_source="sim",
                                 preset="mixed_pipes_dense")
    harness.node.perception_client = _StubService(available=False)
    harness.node._camera_capability_at = None
    harness.tick_until_gate()

    state = _state(harness)
    assert state.available == ["sim"]
    assert not state.camera_available
    assert state.camera_unavailable_reason
    # ... and the ordinary workflow is exactly the ordinary workflow.
    assert harness.node.engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL
    assert len(harness.node.engine.scenario.items) > 10
    harness.node._apply_command("reset", {"preset": "mixed_pipes_small"})
    assert harness.node.engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL


def test_selecting_an_unavailable_camera_is_refused_with_a_reason():
    harness = build_orchestrator(perception_source="sim")
    harness.node.perception_client = _StubService(available=False)
    harness.node._camera_capability_at = None
    harness.tick_until_gate()

    with pytest.raises(PerceptionConfigError) as exc:
        harness.node.set_object_source("camera")
    assert "Physical camera" in str(exc.value)
    assert "not available" in str(exc.value)
    # The draft did NOT move, so nothing downstream can act on a camera that is
    # not there.
    assert _state(harness).selected == "sim"


def test_detecting_without_a_camera_is_refused_and_changes_nothing():
    """NO SILENT FALLBACK. The one behaviour this must never have."""
    harness = build_orchestrator(perception_source="sim")
    harness.node.perception_client = _StubService(available=False)
    harness.node._camera_capability_at = None
    harness.tick_until_gate()
    engine = harness.node.engine
    before = (engine.run_id, engine.scenario_revision,
              len(engine.scenario.items))

    with pytest.raises(ValueError) as exc:
        harness.node._apply_command("detect_physical_objects", {})
    assert "not available" in str(exc.value)
    assert "no perception service is answering" in str(exc.value)

    engine = harness.node.engine
    assert (engine.run_id, engine.scenario_revision,
            len(engine.scenario.items)) == before
    assert engine.config.perception_source is PerceptionSource.SIM


def test_a_camera_that_goes_away_stops_being_offered(session):
    """Availability is a LIVE question, re-asked — not latched at start-up."""
    node = session.node
    assert _state(session).camera_available

    node.perception_client.available = False
    node._camera_capability_at = None           # the cache's TTL, forced
    state = _state(session)
    assert not state.camera_available
    assert state.available == ["sim"]

    # ... and comes back the same way, without anything being restarted.
    node.perception_client.available = True
    node._camera_capability_at = None
    assert _state(session).camera_available


# --------------------------------------------------------------------------- #
# 4. The domain rule underneath all of it
# --------------------------------------------------------------------------- #


def test_resolve_object_source_never_substitutes_silently():
    assert resolve_object_source("camera", ["sim", "camera"]) is (
        PerceptionSource.CAMERA)
    # Nothing requested -> the standing selection, still checked.
    assert resolve_object_source(None, ["sim"], fallback="sim") is (
        PerceptionSource.SIM)
    with pytest.raises(PerceptionConfigError):
        resolve_object_source("camera", ["sim"])
    with pytest.raises(PerceptionConfigError):
        resolve_object_source(None, ["sim"], fallback="camera")
    with pytest.raises(PerceptionConfigError):
        resolve_object_source("nonsense", ["sim", "camera"])


def test_the_selector_speaks_the_operators_language_not_the_enum():
    """`sim` in a selector is not "simulated perception" — it is "preset"."""
    assert PerceptionSource.SIM.selector_label == "Preset scenario"
    assert PerceptionSource.CAMERA.selector_label == "Physical camera"
    assert PerceptionSource.SIM.action_label == "Generate & plan"
    assert PerceptionSource.CAMERA.action_label == "Detect & plan"
    # PROVENANCE is stamped from the source itself, never inferred from the
    # execution backend or from where the dashboard reads its state.
    assert PerceptionSource.SIM.provenance == "preset/generated"
    assert PerceptionSource.CAMERA.provenance == "camera/measured"


def test_the_published_status_names_the_running_runs_source(session):
    """FIWARE/ROS provenance reports the SELECTED-AND-SPENT source, per run."""
    node = session.node
    node.publish_object_source()
    document = session.latest(T.PERCEPTION_STATUS) or {}
    assert document["run_object_source"] == "sim"
    assert document["run_object_provenance"] == "preset/generated"
    assert document["object_source"]["current"] == "sim"

    node.set_object_source("camera")
    node._apply_command("detect_physical_objects", {})
    session.tick_until_gate()
    node.publish_object_source()
    document = session.latest(T.PERCEPTION_STATUS) or {}
    assert document["run_object_source"] == "camera"
    assert document["run_object_provenance"] == "camera/measured"
    assert document["object_source"]["current"] == "camera"
    # The DRAFT is published beside it, and they are different fields.
    assert "selected" in document["object_source"]


def test_the_command_vocabulary_advertises_the_selector():
    """A control the dashboard offers must exist in the contract."""
    assert "set_object_source" in T.OPERATOR_COMMANDS
    assert "detect_physical_objects" in T.OPERATOR_COMMANDS
