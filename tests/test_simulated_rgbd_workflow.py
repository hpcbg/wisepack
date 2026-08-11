"""A simulated RGB-D batch through the ORDINARY workflow, and the stale guard.

BEHAVIOURAL, not source-level: these drive the real `WorkflowEngine` with a
batch shaped exactly as `simulated_rgbd_pipeline.workarea_batch` builds one. No
worker, no GPU, no Isaac and no dashboard — what is under test is the contract
the acquisition enters, which must be the same one a planar or a physical batch
enters.

    REPLACEMENT, NOT ACCUMULATION. Forty generated items plus one perceived
    Cylinder5 leaves ONE item. This already holds for the planar and physical
    paths; it has to hold for this one, through the same call.

    CAD GEOMETRY SURVIVES. The planner packs D25 x L342 with a 19 mm bore
    because the registry declares it — not because a camera measured it, and not
    from anything a simulator knew.

    NO GROUND TRUTH REACHES THE PLAN. The scenario the optimizer sees must
    contain no settled pose, no error and no simulator reference of any kind.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _pkg in ("wisepack_core", "wisepack_bringup", "wisepack_orchestration"):
    _path = os.path.join(REPO, "wisepack_ws", "src", _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)
sys.path.insert(0, os.path.join(REPO, "perception"))

from wisepack_core.acquisition import ACQUISITION_ISAAC            # noqa: E402
from wisepack_core.domain import PhysicalObservation               # noqa: E402
from wisepack_core.generator import build_scenario                 # noqa: E402
from wisepack_core.perception import (                             # noqa: E402
    BatchStatus, ObservationBatch, PerceptionSource)
from wisepack_core.pose import Orientation, WORKAREA_FRAME         # noqa: E402
from wisepack_core.workflow import WorkflowEngine, WorkflowConfig  # noqa: E402

#: Cylinder5's declared geometry, from config/perception_objects.yaml.
C5 = {"diameter_mm": 25, "length_mm": 342, "inner_diameter_mm": 19}


def _simulated_batch(model_id: str = "cylinder5") -> ObservationBatch:
    """What the simulated pipeline hands the workflow, as it builds it.

    A WORKAREA POSE, legitimately: the camera is part of the simulated scene,
    which exported an exact transform. That is the one thing that differs from
    the physical batch, and it differs for a stated reason rather than by
    borrowing the physical path's workaround.
    """
    observation = PhysicalObservation(
        observation_id="simulated-rgbd-1-obj-1",
        x_mm=480.0, y_mm=-320.0, z_mm=18.4,
        frame_id=WORKAREA_FRAME, source="camera",
        object_type="pipe_section", object_model_id=model_id,
        geometry_source="cad_model", perception_method="foundationpose_rgbd",
        detector="foundationpose/rgbd-6dof",
        orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
        captured_at="2026-08-11T09:00:00Z", calibration_status="not_applicable",
        pose_valid=True, workarea_transform_valid=True, **C5)
    return ObservationBatch(
        batch_id="simulated-rgbd-1", source=PerceptionSource.CAMERA.value,
        status=BatchStatus.OK, observations=[observation],
        frame_id=WORKAREA_FRAME, captured_at="2026-08-11T09:00:00Z",
        detector="foundationpose/rgbd-6dof",
        perception_method="foundationpose_rgbd",
        acquisition=ACQUISITION_ISAAC, model_id=model_id,
        calibration_status="not_applicable")


@pytest.fixture
def engine() -> WorkflowEngine:
    """An ordinary preset run, at the approval gate, exactly as a session opens."""
    node = WorkflowEngine(WorkflowConfig(
        preset="mixed_pipes_dense",
        perception_source=PerceptionSource.CAMERA))
    node.generate_or_load_scenario(build_scenario("mixed_pipes_dense", 42))
    node.generate_plans()
    node.digital_twin_validate()
    node.request_approval()
    return node


# --------------------------------------------------------------------------- #
# Replacement, planning, approval
# --------------------------------------------------------------------------- #


def test_one_simulated_observation_replaces_the_previous_objects(engine):
    """THE HEADLINE. Forty generated items plus one perceived tube is ONE item."""
    assert len(engine.scenario.items) == 40
    before = engine.scenario_revision

    engine.apply_observation_batch(_simulated_batch())

    assert len(engine.scenario.items) == 1, (
        "the batch accumulated onto the generated scenario instead of "
        "replacing it")
    assert engine.scenario_revision != before, "a new batch is a new revision"


def test_the_cad_geometry_reaches_the_planner(engine):
    engine.apply_observation_batch(_simulated_batch())
    item = engine.scenario.items[0]
    assert item.model_id == "cylinder5"
    # THE DOMAIN'S OWN VALUE, not the provider's input string: `to_waste_items`
    # normalises a CAD-backed observation to `cad_mesh`.
    assert item.geometry_source == "cad_mesh"
    assert (item.outer_diameter_mm, item.length_mm, item.inner_diameter_mm) == (
        C5["diameter_mm"], C5["length_mm"], C5["inner_diameter_mm"])


def test_the_ordinary_optimizer_plans_it_and_the_twin_validates(engine):
    """NO PERCEPTION-AWARE PLANNER. If this needed one, the provider boundary
    would have failed."""
    engine.apply_observation_batch(_simulated_batch())
    engine.generate_plans()
    assert engine.digital_twin_validate() is True
    plan = engine.selected
    assert len(plan.placements) == 1
    assert plan.containers_required == 1


def test_the_approval_gate_is_re_entered(engine):
    engine.approve(operator="test")
    engine.apply_observation_batch(_simulated_batch())
    engine.generate_plans()
    engine.digital_twin_validate()
    engine.request_approval()
    assert engine.stage.value == "WAIT_FOR_OPERATOR_APPROVAL", (
        "an authorisation for objects that are no longer there must never be "
        "carried forward")


def test_the_placement_is_a_new_pose_inside_the_container(engine):
    """The source pose is provenance, never a container coordinate."""
    engine.apply_observation_batch(_simulated_batch())
    engine.generate_plans()
    engine.digital_twin_validate()
    placement = engine.selected.placements[0]
    source = engine.scenario.items[0].source_position
    assert (placement.position.x, placement.position.y) != (source.x, source.y), (
        "the placement reused the measured pose; container packing must "
        "compute a NEW pose inside the container")


def test_no_ground_truth_appears_anywhere_in_the_planned_scenario(engine):
    engine.apply_observation_batch(_simulated_batch())
    engine.generate_plans()
    engine.digital_twin_validate()
    blob = json.dumps({"scenario": engine.scenario.to_dict(),
                       "plan": engine.selected.to_dict()}, default=str).lower()
    for forbidden in ("settled", "ground_truth", "t_camera_object",
                      "position_error", "tube_axis_line_error", "evaluation"):
        assert forbidden not in blob, (
            f"{forbidden!r} reached the plan; simulator ground truth is for "
            "evaluation only and must never enter planning")


def test_the_batch_provenance_survives_onto_the_item(engine):
    engine.apply_observation_batch(_simulated_batch())
    item = engine.scenario.items[0]
    assert item.observation is not None
    assert item.observation.perception_method == "foundationpose_rgbd"
    assert engine.observation_batch.acquisition == ACQUISITION_ISAAC, (
        "the acquisition must be readable off the batch, or the dashboard has "
        "to guess which device produced the run on screen")


# --------------------------------------------------------------------------- #
# Both directions, no restart
# --------------------------------------------------------------------------- #


def test_switching_back_to_a_preset_replaces_the_perceived_object(engine):
    engine.apply_observation_batch(_simulated_batch())
    assert len(engine.scenario.items) == 1
    engine.generate_or_load_scenario(build_scenario("mixed_pipes_dense", 42))
    assert len(engine.scenario.items) == 40, (
        "replacement has to work in BOTH directions: a preset run after a "
        "perceived one must not inherit the tube")


def test_a_planar_batch_after_a_simulated_one_replaces_it(engine):
    """Two planar proxies after one simulated tube leaves two items, not three."""
    engine.apply_observation_batch(_simulated_batch())
    planar = ObservationBatch(
        batch_id="planar-1", source=PerceptionSource.CAMERA.value,
        status=BatchStatus.OK, detector="fasterrcnn_resnet50_fpn/bottle",
        perception_method="planar_fasterrcnn", acquisition="planar_webcam",
        calibration_status="valid", calibration_revision="abc123",
        captured_at="2026-08-11T09:05:00Z",
        observations=[
            PhysicalObservation(
                observation_id=f"physical-cylinder-{i + 1:03d}",
                x_mm=100.0 + i * 50, y_mm=60.0, yaw_deg=12.0, confidence=0.99,
                object_type="cylindrical_proxy", source="camera",
                captured_at="2026-08-11T09:05:00Z", calibration_status="valid")
            for i in range(2)])
    engine.apply_observation_batch(planar)
    assert len(engine.scenario.items) == 2
    assert engine.observation_batch.acquisition == "planar_webcam"
