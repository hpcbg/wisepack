"""A physically measured batch replaces the scenario and is re-planned.

THE TWO FACTS THIS PINS, which the physical RGB-D path depends on and which are
easy to lose independently:

    REPLACEMENT — a batch of one Cylinder5 arriving after a planar batch of two
    bottles leaves ONE item, not three. The property already holds for a pulled
    batch; these tests hold it across the DDS boundary a pushed one crosses.

    GEOMETRY WITHOUT A SOURCE POSE — the container Digital Twin asks how an
    object's GEOMETRY fits in a container. A model-backed observation carries
    that from the registry, so packing runs with `workarea_pose_available` False
    and a pose in `camera_color_optical_frame`. Where the object is in the cell
    is a different question, still unanswered, and still gating execution.

THE REAL ORCHESTRATOR NODE, through the real command path. No camera, no GPU, no
FoundationPose: the batch is constructed as the provider would have built it,
which is exactly what crosses the wire.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _pkg in ("wisepack_core", "wisepack_bringup", "wisepack_orchestration"):
    _path = os.path.join(REPO, "wisepack_ws", "src", _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator_harness import build_orchestrator                # noqa: E402
from wisepack_bringup import topics as T                           # noqa: E402
from wisepack_core.domain import PhysicalObservation               # noqa: E402
from wisepack_core.perception import (                             # noqa: E402
    BatchStatus, ObservationBatch, PerceptionSource)

#: The physical camera frame. NOT a work-area frame, and never relabelled.
CAMERA_FRAME = "camera_color_optical_frame"

#: Cylinder5's declared geometry, from config/perception_objects.yaml.
C5 = {"diameter_mm": 25, "length_mm": 342, "inner_diameter_mm": 19}

#: A camera-frame position that would be ABSURD as a container coordinate:
#: negative x, and a z of half a metre out in front of the lens. If anything
#: ever treats the source pose as a work-area pose, a placement built from these
#: numbers is unmistakable.
CAMERA_POSE = (-157.5, 11.3, 530.7)


def _planar_batch(count: int = 2) -> ObservationBatch:
    """What the A4Tech + Faster R-CNN path produces: proxies, no CAD model."""
    return ObservationBatch(
        batch_id="planar-001", source="camera", status=BatchStatus.OK,
        captured_at="2026-08-10T10:00:00Z", requested_at="2026-08-10T09:59:59Z",
        detector="fasterrcnn_resnet50_fpn/bottle", perception_method="planar_fasterrcnn",
        calibration_status="valid", calibration_revision="8074644730b3",
        observations=[
            PhysicalObservation(
                observation_id=f"physical-cylinder-{i + 1:03d}",
                x_mm=100.0 + i * 50, y_mm=60.0, yaw_deg=12.0, confidence=0.99,
                object_type="cylindrical_proxy", source="camera",
                captured_at="2026-08-10T10:00:00Z", calibration_status="valid")
            for i in range(count)])


def _physical_c5_batch() -> ObservationBatch:
    """What the D435 + FoundationPose path produces, as the provider builds it.

    CAD-BACKED AND NOT PLACEABLE, both at once: the geometry comes from the
    named model, and the pose stays in the camera frame with no work-area
    transform — which is the combination the packing layer has to accept.
    """
    observation = PhysicalObservation(
        observation_id="physical-c5-obj-1",
        x_mm=CAMERA_POSE[0], y_mm=CAMERA_POSE[1], z_mm=CAMERA_POSE[2],
        frame_id=CAMERA_FRAME, source="camera",
        object_type="pipe_section", object_model_id="cylinder5",
        geometry_source="cad_model", perception_method="foundationpose_rgbd",
        detector="foundationpose/rgbd-6dof",
        captured_at="2026-08-10T18:13:44Z", calibration_status="not_applicable",
        **C5)
    return ObservationBatch(
        batch_id="physical-c5-1", source="camera", status=BatchStatus.OK,
        captured_at="2026-08-10T18:13:44Z", requested_at="2026-08-10T18:13:20Z",
        detector="foundationpose/rgbd-6dof",
        perception_method="foundationpose_rgbd", acquisition="realsense_d435",
        model_id="cylinder5", frame_id=CAMERA_FRAME,
        calibration_status="not_applicable", observations=[observation])


class _StubService:
    """The planar perception service, reduced to what the node asks of it."""

    url = "http://127.0.0.1:22101"

    def capability(self, health=None):
        return True, ""

    def detect(self) -> ObservationBatch:
        return _planar_batch()


@pytest.fixture
def session():
    """One orchestrator, started exactly as the launcher starts it."""
    harness = build_orchestrator(perception_source="sim",
                                 preset="mixed_pipes_dense")
    harness.node.perception_client = _StubService()
    harness.node._camera_capability_at = None
    harness.tick_until_gate()
    return harness


def _items(harness):
    return list(harness.node.engine.scenario.items)


def _submit(harness, batch: ObservationBatch):
    """Publish through the REAL command path, as the dashboard does."""
    harness.node._apply_command("submit_observation_batch",
                                {"batch": batch.to_dict()})
    harness.tick_until_gate()


def _detect(harness):
    """The PULLED path: select the camera, then ask the node to acquire.

    The source is selected first, exactly as the dashboard does it — the
    detection runs on a worker thread and is adopted on a later tick, and a
    detect issued while the draft still says `sim` starts a new run in the same
    breath, which is a second thing to wait for.
    """
    harness.node.set_object_source("camera")
    harness.node._apply_command("detect_physical_objects", {})
    for _ in range(40):
        if harness.node._pending_observation is not None:
            break
        time.sleep(0.02)
    harness.tick_until_gate()


# --------------------------------------------------------------------------- #
# A. Replacement across the command boundary
# --------------------------------------------------------------------------- #


def test_a_physical_batch_replaces_the_planar_items(session):
    """THE HEADLINE. Two bottles, then one tube, leaves one tube."""
    _detect(session)
    assert len(_items(session)) == 2, "the planar batch must land first"

    _submit(session, _physical_c5_batch())

    items = _items(session)
    assert len(items) == 1, (
        "REPLACEMENT, NOT ACCUMULATION: the planar objects survived a physical "
        f"batch — {[i.item_id for i in items]}")
    assert items[0].observation.object_model_id == "cylinder5"


def test_the_reverse_replacement_also_holds(session):
    """One Cylinder5, then a planar batch, leaves exactly the planar items."""
    _submit(session, _physical_c5_batch())
    assert len(_items(session)) == 1

    _detect(session)

    items = _items(session)
    assert len(items) == 2
    assert all(i.observation.object_model_id in ("", None) for i in items), (
        "a planar observation has no CAD model and must not inherit one")


def test_each_submission_is_a_new_scenario_revision(session):
    """A new batch is a new revision, so a decision cannot be inherited from
    the batch it replaced."""
    _detect(session)
    before = session.node.engine.scenario_revision
    _submit(session, _physical_c5_batch())
    assert session.node.engine.scenario_revision > before


# --------------------------------------------------------------------------- #
# B/C/D. CAD geometry, without a work-area pose
# --------------------------------------------------------------------------- #


def test_the_cad_geometry_reaches_the_packing_item(session):
    """D25 x L342, bore 19 — the registry's declared numbers, not a proxy's.

    Overwriting them with the configured proxy geometry would plan an anonymous
    cylinder into the container in place of a real part.
    """
    _submit(session, _physical_c5_batch())
    item = _items(session)[0]
    assert item.length_mm == 342
    assert item.outer_diameter_mm == 25
    assert item.inner_diameter_mm == 19


def test_packing_runs_without_a_work_area_pose(session):
    """THE SEPARATION THIS WHOLE PATH RESTS ON. The Digital Twin is a PACKING
    twin: it asks how the geometry fits a container, which needs no source
    position. `workarea_pose_available` gates EXECUTION, not packing."""
    _submit(session, _physical_c5_batch())
    engine = session.node.engine
    observation = _items(session)[0].observation

    assert observation.workarea_pose_available is False
    assert observation.frame_id == CAMERA_FRAME
    # ... and yet a plan exists, was validated, and is at the gate.
    assert engine.selected is not None, "packing did not run"
    assert engine.selected.containers_required >= 1
    assert engine.baseline is not None


def test_the_camera_position_never_becomes_a_container_position(session):
    """A pose in the camera optical frame is not a work-area coordinate.

    The placement is computed by the optimizer from geometry alone, so no
    coordinate of the source pose may appear in it — and the frame must not be
    quietly relabelled to the work area either.
    """
    _submit(session, _physical_c5_batch())
    engine = session.node.engine
    placements = json.dumps(engine.selected.to_dict())

    for value in CAMERA_POSE:
        assert str(round(value)) not in placements, (
            f"the camera-frame coordinate {value} appears in the plan")
    assert "camera_color_optical_frame" not in placements, (
        "the plan must be expressed in container coordinates, not the camera's")
    assert _items(session)[0].observation.frame_id == CAMERA_FRAME, (
        "the OBSERVATION's frame was relabelled")


# --------------------------------------------------------------------------- #
# The contract itself
# --------------------------------------------------------------------------- #


def test_the_command_is_part_of_the_documented_vocabulary(session):
    assert "submit_observation_batch" in T.OPERATOR_COMMANDS


def test_the_command_carries_no_knowledge_of_the_detector():
    """GENERIC. The orchestrator adopts an ObservationBatch; which method
    produced it is provenance inside the batch. A command that named a model or
    a method would need a new command for every future detector."""
    source = open(os.path.join(
        REPO, "wisepack_ws", "src", "wisepack_orchestration",
        "wisepack_orchestration", "hitl_orchestrator.py"), encoding="utf-8").read()
    handler = source[source.index('elif command == "submit_observation_batch"'):
                     source.index('elif command == "acknowledge_anomaly"')]
    code = "\n".join(line.split("#")[0] for line in handler.splitlines())
    for forbidden in ("cylinder5", "foundationpose", "realsense", "fasterrcnn"):
        assert forbidden not in code.lower(), (
            f"the handler knows about {forbidden}")


def test_an_empty_batch_is_refused_rather_than_clearing_the_scenario(session):
    """An empty batch would silently empty the container plan while measuring
    nothing. That is a refusal, not a detection of zero objects."""
    _detect(session)
    before = len(_items(session))
    empty = _physical_c5_batch()
    empty.observations = []
    with pytest.raises(ValueError, match="no observations"):
        session.node._apply_command("submit_observation_batch",
                                    {"batch": empty.to_dict()})
    assert len(_items(session)) == before, "the scenario was cleared anyway"


def test_a_failed_batch_is_recorded_and_refused(session):
    """A failed scan must reach the operator as a failure, not be dropped."""
    failed = ObservationBatch.failed(
        batch_id="physical-fail", source="camera",
        error="segmentation produced no usable mask")
    with pytest.raises(ValueError, match="segmentation produced no usable mask"):
        session.node._apply_command("submit_observation_batch",
                                    {"batch": failed.to_dict()})


def test_a_batch_is_required(session):
    with pytest.raises(ValueError, match="`batch` is required"):
        session.node._apply_command("submit_observation_batch", {})
