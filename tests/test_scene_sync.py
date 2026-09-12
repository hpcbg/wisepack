"""Physical ObservationBatch -> synchronized Isaac scene -> robot pick target.

THE MISSING LINK THIS PINS. A real D435 observation used to stop at the packing
twin: its pose stayed in `camera_color_optical_frame`, the Isaac scene was built
from the generated (preset, seed) row, and the robot was told to pick from
`table_pose_for_index`. These tests hold the path that replaces that:

    ObservationBatch
      -> configured camera->workarea->table transform   (config/isaac_workcell.yaml)
      -> SceneSpec of engineering-CAD source objects     (wisepack_core.scene_sync)
      -> RESET_SCENE / SYNC_SCENE carrying the spec      (isaac_contract 1.1)
      -> Isaac spawns exactly those objects at those poses
      -> EXECUTE_ITEM.source_pose IS the transformed observation

and the things it must never do: assume an identity transform, fall back to the
generated row, apply a stale batch over a newer scene, append an observation
beside the previous ones, hand CAD to the model-free estimator, or spawn the
estimator's reconstruction as the physical body.

Nothing here needs Isaac, a GPU, a camera or ROS: the transform chain, the
contract and the bridge are plain Python, and the simulator's behaviour is
pinned on its source where it cannot be executed.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wisepack_core.domain import (                                  # noqa: E402
    GEOMETRY_SOURCE_CAD_MESH, PhysicalObservation, Vec3)
from wisepack_core.generator import build_scenario                  # noqa: E402
from wisepack_core.isaac_contract import (                          # noqa: E402
    SCENE_SOURCE_GENERATED, SCENE_SOURCE_PHYSICAL, ContractError,
    Dimensions, IsaacCommand, IsaacCommandType, IsaacFeedback, IsaacState,
    Pose, SceneAcknowledgement, SceneObject, SceneSpec)
from wisepack_core.isaac_transform import (                         # noqa: E402
    DEFAULT_LAYOUT, SourcePoseUnavailable, alignment_to_local_z,
    pick_yaw_deg, pose_to_world, quaternion_for_axis, scene_fingerprint,
    source_pose_for, table_pose_for_index, world_to_pose)
from wisepack_core.perception import (                              # noqa: E402
    BatchStatus, ObservationBatch, PerceptionSource)
from wisepack_core.pose import (                                    # noqa: E402
    CAMERA_OPTICAL_FRAME, WORKAREA_FRAME, Orientation, RigidTransform,
    Symmetry)
from wisepack_core.scene_sync import (                              # noqa: E402
    SceneSyncRefused, body_orientation, scene_items, synchronize_scene,
    transform_observation)
from wisepack_core.workcell import (                                # noqa: E402
    PROVENANCE_CONFIGURED_DEMO, PROVENANCE_MEASURED_CALIBRATION, TABLE_FRAME,
    WorkareaBounds, WorkcellConfigError, WorkcellFrames, load_workcell,
    provenance_label, workcell_from_dict)

CORE = os.path.join(REPO, "wisepack_ws", "src", "wisepack_core", "wisepack_core")
SIM = os.path.join(REPO, "simulators", "isaac")
ORCH = os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                    "wisepack_orchestration")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# Fixtures: the bench observation, as the provider built it
# --------------------------------------------------------------------------- #

#: Cylinder5, from config/perception_objects.yaml.
C5_CENTRE = (-130.0, -54.44, 0.0)
C5_AXIS = (0.9284, -0.3716, 0.0)

#: The demo chain: camera looking straight down from 542 mm, work area 500 mm
#: in front of the robot base. Written out here INDEPENDENTLY of the YAML so
#: the tracked file is checked against these expectations, not assumed.
CAMERA_HEIGHT_MM = 542.0
WORKAREA_X_MM = 500.0


def _frames(provenance: str = PROVENANCE_CONFIGURED_DEMO,
            height_mm: float = CAMERA_HEIGHT_MM,
            bounds: WorkareaBounds = WorkareaBounds()) -> WorkcellFrames:
    return WorkcellFrames(
        camera_to_workarea=RigidTransform(
            parent_frame=WORKAREA_FRAME, child_frame=CAMERA_OPTICAL_FRAME,
            translation_mm=(0.0, 0.0, height_mm),
            rotation=Orientation(1.0, 0.0, 0.0, 0.0),     # 180 deg about X
            method=provenance, revision="test"),
        workarea_to_table=RigidTransform(
            parent_frame=TABLE_FRAME, child_frame=WORKAREA_FRAME,
            translation_mm=(WORKAREA_X_MM, 0.0, 0.0),
            method=provenance, revision="test"),
        provenance=provenance, bounds=bounds, path="<test>")


def _observation(observation_id: str = "physical-c5-obj-1",
                 model_origin=(-222.164, 97.268, 594.551),
                 orientation=(0.358507419, 0.221662745, 0.723006091, 0.547357516),
                 method: str = "foundationpose_rgbd_model_free",
                 frame_id: str = CAMERA_OPTICAL_FRAME,
                 pose_valid: bool = True) -> PhysicalObservation:
    """The saved bench result (cylinder5-20260813-153948), model-free method."""
    return PhysicalObservation(
        observation_id=observation_id,
        x_mm=model_origin[0], y_mm=model_origin[1], z_mm=model_origin[2],
        object_type="pipe_section", source="camera", frame_id=frame_id,
        detector="foundationpose/rgbd-6dof", model_id="a1b694b8",
        captured_at="2026-08-13T15:39:57Z", calibration_status="not_applicable",
        diameter_mm=25, length_mm=342, inner_diameter_mm=19,
        geometry_source="cad_model",
        orientation=Orientation(*orientation),
        symmetry=Symmetry(type="discrete", axis="z", fold=2),
        perception_method=method, object_model_id="cylinder5",
        pose_valid=pose_valid, workarea_transform_valid=False,
        measured_dof=("x", "y", "z", "orientation_partial"),
        model_center_mm=C5_CENTRE, task_axis_vector=C5_AXIS, confidence=None)


def _batch(*observations: PhysicalObservation, batch_id: str = "fp-physical-1",
           method: str = "foundationpose_rgbd_model_free") -> ObservationBatch:
    return ObservationBatch(
        batch_id=batch_id, source=PerceptionSource.CAMERA.value,
        status=BatchStatus.OK, observations=list(observations),
        frame_id=CAMERA_OPTICAL_FRAME, captured_at="2026-08-13T15:39:57Z",
        detector="foundationpose/rgbd-6dof", perception_method=method,
        model_id="cylinder5", acquisition="realsense_d435",
        calibration_status="not_applicable")


def _items(batch: ObservationBatch):
    return batch.to_waste_items()


def _spec(batch: ObservationBatch, frames=None, revision: int = 2,
          run_id: str = "run-1") -> SceneSpec:
    return synchronize_scene(_items(batch), batch, frames or _frames(),
                             run_id=run_id, scenario_revision=revision)


def _matrix_apply(rotation, vector):
    return tuple(sum(rotation[i][j] * vector[j] for j in range(3)) for i in range(3))


# =========================================================================== #
# 1. The transform chain: camera -> work area -> table -> Isaac world
# =========================================================================== #

def test_the_tracked_demo_config_loads_and_is_the_expected_chain():
    """config/isaac_workcell.yaml IS the assumption; check it, do not trust it."""
    frames = load_workcell()
    assert frames.available, frames.unavailable_reason
    assert frames.provenance == PROVENANCE_CONFIGURED_DEMO
    assert frames.frame_chain == [CAMERA_OPTICAL_FRAME, WORKAREA_FRAME,
                                  TABLE_FRAME, "world"]
    cam = frames.camera_to_workarea
    assert (cam.parent_frame, cam.child_frame) == (WORKAREA_FRAME, CAMERA_OPTICAL_FRAME)
    # Optical +Z (into the scene) must map onto work-area -Z (down).
    assert cam.rotation.rotate((0.0, 0.0, 1.0)) == pytest.approx((0.0, 0.0, -1.0))
    assert cam.translation_mm[2] == pytest.approx(CAMERA_HEIGHT_MM)
    tab = frames.workarea_to_table
    assert (tab.parent_frame, tab.child_frame) == (TABLE_FRAME, WORKAREA_FRAME)
    assert tab.translation_mm == pytest.approx((WORKAREA_X_MM, 0.0, 0.0))


def test_the_object_centre_is_carried_through_every_frame_of_the_chain():
    """Hand-computed: R = diag(1,-1,-1), t = (0,0,H); then +500 along table X."""
    obs = _observation()
    result = transform_observation(obs, _frames(), radius_mm=12.5)
    cx, cy, cz = obs.object_center            # the BODY centre, not the model origin
    assert result.camera.centre_mm == pytest.approx((cx, cy, cz))
    assert result.workarea.centre_mm == pytest.approx((cx, -cy, CAMERA_HEIGHT_MM - cz))
    assert result.table.centre_mm == pytest.approx(
        (cx + WORKAREA_X_MM, -cy, CAMERA_HEIGHT_MM - cz))
    origin = DEFAULT_LAYOUT.table_frame_origin_m
    assert result.world_position_m == pytest.approx(
        ((cx + WORKAREA_X_MM) / 1000.0 + origin[0], -cy / 1000.0 + origin[1],
         (CAMERA_HEIGHT_MM - cz) / 1000.0 + origin[2]))
    assert result.pose.frame == TABLE_FRAME
    assert result.pose.orientation is not None


def test_the_tube_axis_is_transformed_with_the_position():
    """The whole pose, not just the centre: the body's +Z is the observed axis."""
    obs = _observation()
    frames = _frames()
    result = transform_observation(obs, frames, radius_mm=12.5)
    expected = frames.workarea_to_table.rotation.rotate(
        frames.camera_to_workarea.rotation.rotate(obs.tube_axis))
    _, quaternion = pose_to_world(result.pose)
    w, x, y, z = quaternion
    body_z = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    # A line: either sign describes the same tube.
    dot = abs(sum(a * b for a, b in zip(body_z, expected)))
    assert dot == pytest.approx(1.0, abs=1e-6)
    assert result.table.axis_line == pytest.approx(tuple(body_z), abs=1e-6)


def test_the_model_origin_is_not_what_is_placed():
    """Cylinder5's CAD origin sits 141 mm outside the tube; the body is placed."""
    obs = _observation()
    result = transform_observation(obs, _frames(), radius_mm=12.5)
    origin_in_workarea = _frames().camera_to_workarea.apply_to_position(
        (obs.x_mm, obs.y_mm, obs.z_mm))
    gap = math.dist(origin_in_workarea, result.workarea.centre_mm)
    assert gap > 100.0


def test_body_orientation_composes_the_axis_alignment_the_scene_applies():
    """R_body = R_model * align^-1, so the spawned mesh's +Z is the tube axis."""
    obs = _observation()
    body = body_orientation(obs)
    align = alignment_to_local_z(C5_AXIS)
    # The body's local +Z, taken back through align, is the model's task axis.
    model_axis = align.conjugate().rotate((0.0, 0.0, 1.0))
    assert model_axis == pytest.approx(
        tuple(v / math.sqrt(sum(a * a for a in C5_AXIS)) for v in C5_AXIS), abs=1e-9)
    assert body.axis("z") == pytest.approx(obs.tube_axis, abs=1e-9)


def test_axis_alignment_agrees_with_the_scene_builders_previous_numpy_version():
    numpy = pytest.importorskip("numpy")
    for vector in [C5_AXIS, (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1),
                   (0.3, 0.2, -0.9)]:
        source = numpy.asarray(vector, float)
        source = source / numpy.linalg.norm(source)
        target = numpy.array([0.0, 0.0, 1.0])
        cross = numpy.cross(source, target)
        dot = float(source @ target)
        if numpy.linalg.norm(cross) < 1e-9:
            expected = numpy.eye(3) if dot > 0 else numpy.diag([1.0, -1.0, -1.0])
        else:
            skew = numpy.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]],
                                [-cross[1], cross[0], 0]])
            expected = numpy.eye(3) + skew + skew @ skew * (1.0 / (1.0 + dot))
        actual = numpy.array(alignment_to_local_z(vector).to_matrix())
        assert numpy.abs(actual - expected).max() < 1e-9


def test_a_planar_observation_already_in_the_work_area_needs_only_one_link():
    obs = PhysicalObservation(
        observation_id="planar-1", x_mm=40.0, y_mm=-30.0, yaw_deg=30.0,
        object_type="cylindrical_proxy", source="camera", frame_id=WORKAREA_FRAME,
        diameter_mm=65, length_mm=215, measured_dof=("x", "y", "yaw"))
    result = transform_observation(obs, _frames(), radius_mm=32.5)
    assert result.camera is None
    # Height was not measured: the body rests on the plane, one radius up, and
    # the provenance says the height was assumed rather than measured.
    assert result.table.centre_mm == pytest.approx((540.0, -30.0, 32.5))
    assert result.pose_provenance["height_assumed_on_source_plane"] is True
    rgbd = transform_observation(_observation(), _frames(), radius_mm=12.5)
    assert rgbd.pose_provenance["height_assumed_on_source_plane"] is False
    # Yaw is the heading of the length: local +X of the planar frame.
    assert pick_yaw_deg(result.pose) == pytest.approx(30.0, abs=1e-6)


def test_pose_to_world_honours_a_full_orientation_and_round_trips():
    q = Orientation.from_yaw_deg(37.0).multiply(
        Orientation(0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)))
    pose = Pose(480.0, 10.0, 12.5, axis="x", frame="table", orientation=q.as_tuple())
    position, quaternion = pose_to_world(pose)
    assert quaternion == pytest.approx((q.w, q.x, q.y, q.z))
    back = world_to_pose(position, quaternion, "table")
    assert (back.x_mm, back.y_mm, back.z_mm) == pytest.approx((480.0, 10.0, 12.5))
    assert back.orientation is not None
    assert Orientation(*back.orientation).angle_to_deg(q) < 1e-6
    # Without an orientation the axis-aligned conversion is exactly what it was.
    plain = Pose(480.0, 10.0, 12.5, axis="x", frame="table")
    assert pose_to_world(plain)[1] == quaternion_for_axis("x")


def test_pick_yaw_is_the_length_heading_folded_to_a_line():
    assert pick_yaw_deg(Pose(0, 0, 0, axis="x")) == 0.0
    assert pick_yaw_deg(Pose(0, 0, 0, axis="y")) == 90.0
    assert pick_yaw_deg(Pose(0, 0, 0, axis="z")) == 0.0
    along_x = Orientation(0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
    for heading in (0.0, 30.0, 89.0, 91.0, 150.0, -20.0, -100.0):
        q = Orientation.from_yaw_deg(heading).multiply(along_x)
        yaw = pick_yaw_deg(Pose(0, 0, 0, orientation=q.as_tuple()))
        assert -90.0 < yaw <= 90.0
        assert math.isclose(math.cos(math.radians(yaw - heading)) ** 2, 1.0, abs_tol=1e-9)


# =========================================================================== #
# 2. Provenance: configured demo, never calibrated
# =========================================================================== #

def test_the_demo_transform_is_labelled_configured_and_never_calibrated():
    frames = load_workcell()
    assert frames.label == "Configured demo transform"
    assert "calibrat" not in frames.label.lower()
    assert "measured" not in frames.label.lower()
    spec = _spec(_batch(_observation()), frames)
    assert spec.transform_source == PROVENANCE_CONFIGURED_DEMO
    for source in spec.objects:
        assert source.provenance["transform_source"] == PROVENANCE_CONFIGURED_DEMO
        assert source.provenance["frame_chain"] == [
            CAMERA_OPTICAL_FRAME, WORKAREA_FRAME, TABLE_FRAME, "world"]


def test_a_measured_calibration_is_a_different_provenance():
    assert provenance_label(PROVENANCE_MEASURED_CALIBRATION) == "Measured calibration"
    assert provenance_label(PROVENANCE_CONFIGURED_DEMO) != provenance_label(
        PROVENANCE_MEASURED_CALIBRATION)
    assert provenance_label("") == "No transform"


def test_the_tracked_config_states_its_provenance_on_every_transform():
    text = _read(os.path.join(REPO, "config", "isaac_workcell.yaml"))
    assert "provenance: configured_demo" in text
    assert text.count("method: configured_demo") == 2
    assert "measured_calibration" in text          # named as the future value
    assert "NOT A CALIBRATION" in text


def test_a_config_whose_transform_disagrees_with_its_provenance_is_refused():
    doc = {"schema": "wisepack-isaac-workcell/1.0", "provenance": "configured_demo",
           "transforms": [{"parent_frame": WORKAREA_FRAME,
                           "child_frame": CAMERA_OPTICAL_FRAME,
                           "translation_mm": [0, 0, 500],
                           "method": "measured_calibration"}]}
    with pytest.raises(WorkcellConfigError, match="disagrees"):
        workcell_from_dict(doc)


def test_the_provenance_reaches_the_scene_command_and_the_acknowledgement():
    spec = _spec(_batch(_observation()))
    command = IsaacCommand.from_json(IsaacCommand(
        command=IsaacCommandType.RESET_SCENE, run_id="run-1",
        scenario_revision=2, scene=spec).to_json())
    assert command.scene.transform_source == PROVENANCE_CONFIGURED_DEMO
    ack = SceneAcknowledgement.from_dict(SceneAcknowledgement(
        scene_source=SCENE_SOURCE_PHYSICAL, observation_batch_id="fp-physical-1",
        transform_source=PROVENANCE_CONFIGURED_DEMO).to_dict())
    assert ack.transform_source == PROVENANCE_CONFIGURED_DEMO


# =========================================================================== #
# 3. No identity fallback
# =========================================================================== #

def test_a_camera_frame_pose_with_no_transform_is_refused_not_relabelled():
    frames = WorkcellFrames(error="workcell configuration not found")
    assert not frames.available
    with pytest.raises(SceneSyncRefused, match="NO identity transform"):
        transform_observation(_observation(), frames)


def test_a_transform_without_a_method_is_not_a_transform():
    doc = {"schema": "wisepack-isaac-workcell/1.0", "provenance": "configured_demo",
           "transforms": [{"parent_frame": WORKAREA_FRAME,
                           "child_frame": CAMERA_OPTICAL_FRAME,
                           "translation_mm": [0, 0, 500]}]}
    with pytest.raises(WorkcellConfigError, match="declares no method"):
        workcell_from_dict(doc)


def test_an_unknown_frame_is_refused():
    obs = _observation(frame_id="some_other_camera")
    with pytest.raises(SceneSyncRefused, match="no transform into the work area"):
        transform_observation(obs, _frames())


def test_a_missing_config_file_reports_unavailable_rather_than_identity(tmp_path):
    frames = load_workcell(path=str(tmp_path / "nothing.yaml"))
    assert not frames.available
    assert "not found" in frames.unavailable_reason
    assert frames.camera_to_workarea is None


def test_the_missing_second_link_is_refused_too():
    frames = _frames()
    frames.workarea_to_table = None
    with pytest.raises(SceneSyncRefused):
        transform_observation(_observation(), frames)


def test_synchronize_refuses_when_the_chain_is_unavailable():
    batch = _batch(_observation())
    with pytest.raises(SceneSyncRefused, match="transform is unavailable"):
        synchronize_scene(_items(batch), batch, WorkcellFrames(error="x"),
                          run_id="run-1", scenario_revision=2)


# =========================================================================== #
# 4. Refusals: invalid, non-finite, out of the work area, unreachable
# =========================================================================== #

def test_an_invalid_pose_refuses_synchronization():
    with pytest.raises(SceneSyncRefused, match="pose_valid is false"):
        transform_observation(_observation(pose_valid=False), _frames())


def test_a_non_finite_pose_refuses_synchronization():
    obs = _observation(model_origin=(float("nan"), 0.0, 500.0))
    with pytest.raises(SceneSyncRefused, match="not finite"):
        transform_observation(obs, _frames())


def test_an_observation_outside_the_work_area_bounds_is_refused_not_clamped():
    far = _observation(model_origin=(-222.164 + 900.0, 97.268, 594.551))
    with pytest.raises(SceneSyncRefused, match="outside the configured work-area"):
        transform_observation(far, _frames(), radius_mm=12.5)


def test_an_observation_floating_above_or_sunk_below_the_plane_is_refused():
    high = _observation(model_origin=(-222.164, 97.268, 594.551 - 200.0))
    with pytest.raises(SceneSyncRefused, match="floats the object centre"):
        transform_observation(high, _frames(), radius_mm=12.5)
    low = _observation(model_origin=(-222.164, 97.268, 594.551 + 80.0))
    with pytest.raises(SceneSyncRefused, match="below where it would rest"):
        transform_observation(low, _frames(), radius_mm=12.5)


def test_an_unreachable_object_is_refused_by_the_existing_reach_band():
    frames = _frames(bounds=WorkareaBounds(x_mm=(-2000, 2000), y_mm=(-2000, 2000)))
    frames.workarea_to_table = RigidTransform(
        parent_frame=TABLE_FRAME, child_frame=WORKAREA_FRAME,
        translation_mm=(1400.0, 0.0, 0.0), method=PROVENANCE_CONFIGURED_DEMO)
    with pytest.raises(SceneSyncRefused, match="outside the reachable band"):
        transform_observation(_observation(), frames, radius_mm=12.5)


def test_a_failed_batch_cannot_become_a_scene():
    batch = ObservationBatch.failed("fp-physical-9", "camera", "no frame")
    with pytest.raises(SceneSyncRefused, match="failed"):
        synchronize_scene([], batch, _frames(), run_id="run-1", scenario_revision=2)


def test_an_empty_batch_is_not_an_empty_scene():
    batch = _batch()
    with pytest.raises(SceneSyncRefused, match="no items"):
        synchronize_scene([], batch, _frames(), run_id="run-1", scenario_revision=2)


# =========================================================================== #
# 5. Replacement, not accumulation — and stale revisions
# =========================================================================== #

def test_the_scene_spec_holds_exactly_the_batch_objects():
    batch = _batch(_observation("obj-a"), _observation("obj-b", model_origin=(-150.0, 20.0, 594.0)))
    spec = _spec(batch)
    assert spec.scene_source == SCENE_SOURCE_PHYSICAL
    assert spec.observation_batch_id == "fp-physical-1"
    assert [o.observation_id for o in spec.objects] == ["obj-a", "obj-b"]
    assert [o.item_id for o in spec.objects] == ["item-001", "item-002"]


def test_a_second_batch_replaces_the_first_rather_than_adding_to_it():
    first = _spec(_batch(_observation("obj-a"), batch_id="fp-physical-1"), revision=2)
    second = _spec(_batch(_observation("obj-b", model_origin=(-150.0, 20.0, 594.0)),
                          batch_id="fp-physical-2"), revision=3)
    assert len(first.objects) == 1 and len(second.objects) == 1
    assert second.object_ids == ["item-001"]
    assert second.objects[0].observation_id == "obj-b"
    assert second.observation_batch_id != first.observation_batch_id
    # The simulator's item set is rebuilt from the spec alone.
    assert [i.item_id for i in scene_items(second)] == ["item-001"]


def test_the_simulator_removes_previous_objects_before_spawning_the_spec():
    src = _read(os.path.join(SIM, "scene.py"))
    reset = src[src.index("def reset_items"):src.index("def settle_items")]
    assert "stage.RemovePrim(path)" in reset
    assert reset.index("stage.RemovePrim(path)") < reset.index("self.build_items(scenario, scene=scene)")
    app = _read(os.path.join(SIM, "wisepack_isaac.py"))
    assert "self.scenario.items = scene_items(spec)" in app
    assert "self.scene.reset_items(self.scenario, scene=spec)" in app


def test_a_stale_spec_never_rewrites_a_newer_scene():
    older = _spec(_batch(_observation(), batch_id="fp-physical-1"), revision=2)
    assert older.is_stale_against("run-1", 3)
    assert not older.is_stale_against("run-1", 2)
    assert not older.is_stale_against("run-1", 1)
    # A different run is a different world; revisions are not comparable.
    assert not older.is_stale_against("run-2", 9)


def test_the_simulator_checks_staleness_before_any_scene_command():
    src = _read(os.path.join(SIM, "wisepack_isaac.py"))
    handler = src[src.index("def _on_command"):src.index("def _begin_run")]
    assert "_stale_scene_request(command)" in handler
    assert handler.index("_stale_scene_request(command)") < handler.index("self._reset_scene(command)")
    assert "IsaacState.RESET_FAILED" in handler
    stale = src[src.index("def _stale_scene_request"):src.index("def _robot_is_home")]
    assert "is_stale_against" in stale


def test_an_acknowledgement_for_an_older_batch_is_a_mismatch():
    ack = SceneAcknowledgement(
        run_id="run-1", scenario_revision=3, scene_source=SCENE_SOURCE_PHYSICAL,
        observation_batch_id="fp-physical-1", robot_home_verified=True,
        container_empty_verified=True)
    reasons = ack.mismatches(run_id="run-1", scenario_id="", revision=3, preset="",
                             seed=0, fingerprint="", object_count=0,
                             scene_source=SCENE_SOURCE_PHYSICAL,
                             observation_batch_id="fp-physical-2")
    assert any("fp-physical-1" in r and "fp-physical-2" in r for r in reasons)


def test_a_generated_scene_never_satisfies_a_physical_run():
    ack = SceneAcknowledgement(scene_source=SCENE_SOURCE_GENERATED,
                               robot_home_verified=True, container_empty_verified=True)
    reasons = ack.mismatches(run_id="", scenario_id="", revision=0, preset="", seed=0,
                             fingerprint="", object_count=0,
                             scene_source=SCENE_SOURCE_PHYSICAL)
    assert reasons and "generated" in reasons[0]


def test_the_fingerprint_covers_the_batch_and_the_pose():
    scenario = build_scenario("isaac_cylinders_smoke", seed=42)
    batch = _batch(_observation())
    scenario.items = _items(batch)
    spec = _spec(batch)
    generated = scene_fingerprint(scenario)
    physical = scene_fingerprint(scenario, scene=spec)
    assert physical != generated
    moved = _spec(_batch(_observation(model_origin=(-222.164 + 60.0, 97.268, 594.551))))
    assert scene_fingerprint(scenario, scene=moved) != physical
    other_batch = _spec(_batch(_observation(), batch_id="fp-physical-2"))
    assert scene_fingerprint(scenario, scene=other_batch) != physical
    assert scene_fingerprint(scenario, scene=_spec(batch)) == physical   # deterministic


# =========================================================================== #
# 6. Model-free stays model-free; the scene uses engineering geometry
# =========================================================================== #

def test_the_scene_object_is_engineering_cad_while_the_estimator_was_model_free():
    spec = _spec(_batch(_observation(method="foundationpose_rgbd_model_free")))
    source = spec.objects[0]
    assert source.perception_method == "foundationpose_rgbd_model_free"
    assert source.provenance["pose"]["estimator_geometry"] == "learned_representation"
    assert source.geometry_source == GEOMETRY_SOURCE_CAD_MESH
    assert source.model_id == "cylinder5"
    assert source.provenance["scene_geometry"]["source"] == "engineering_cad"
    assert "learned representation is never used" in source.provenance["scene_geometry"]["note"]
    assert source.dimensions == Dimensions(length_mm=342, outer_diameter_mm=25,
                                           inner_diameter_mm=19)


def test_the_synchronizer_never_touches_either_mesh_registry():
    """Import block only: the synchronizer copies identity, it resolves no mesh."""
    src = _read(os.path.join(CORE, "scene_sync.py"))
    imports = "\n".join(line for line in src.splitlines()
                        if line.startswith(("from ", "import ")))
    for forbidden in ("representation", "rgbd", "trimesh", "import os",
                      "import yaml"):
        assert forbidden not in imports, forbidden
    assert "load_object_registry" not in src
    assert "mesh_path(" not in src


def test_the_model_free_estimator_request_is_unchanged_by_the_scene_link():
    """The provider's model-free branch still sends the representation mesh only."""
    src = _read(os.path.join(REPO, "perception", "providers", "foundationpose_rgbd.py"))
    branch = src[src.index("if chosen.requires_representation:"):src.index("else:", src.index("if chosen.requires_representation:"))]
    assert "self.representations" in branch
    assert "model.resolved_path" not in branch
    assert "scene_sync" not in src and "isaac_workcell" not in src


def test_the_isaac_scene_resolves_cad_through_the_object_registry_only():
    src = _read(os.path.join(SIM, "scene.py"))
    assert "load_object_registry" in src
    assert "model.resolved_path(registry.root)" in src
    imports = "\n".join(line for line in src.splitlines()
                        if line.lstrip().startswith(("from ", "import ")))
    assert "representation" not in imports
    assert "load_representation_registry" not in src


def test_scene_items_carry_cad_identity_and_mass_for_the_simulator():
    spec = _spec(_batch(_observation()))
    items = scene_items(spec)
    assert len(items) == 1
    item = items[0]
    assert item.item_id == "item-001"
    assert item.geometry_source == GEOMETRY_SOURCE_CAD_MESH
    assert item.model_id == "cylinder5"
    assert (item.length_mm, item.outer_diameter_mm, item.inner_diameter_mm) == (342, 25, 19)
    assert item.weight_kg > 0.0
    assert item.source_position == Vec3(331, 10, 22)


# =========================================================================== #
# 7. The pick target derives from the observation, never from the row
# =========================================================================== #

def test_source_pose_for_returns_the_observation_pose_for_a_physical_scene():
    batch = _batch(_observation())
    items = _items(batch)
    spec = _spec(batch)
    pose = source_pose_for(spec, 0, items[0])
    assert pose == spec.objects[0].source_pose
    assert pose.orientation is not None
    row = table_pose_for_index(0, items[0])
    assert (pose.x_mm, pose.y_mm) != (row.x_mm, row.y_mm)
    # And the generated path is byte-for-byte what it was.
    assert source_pose_for(None, 0, items[0]) == row
    generated = SceneSpec(scene_source=SCENE_SOURCE_GENERATED)
    assert source_pose_for(generated, 0, items[0]) == row


def test_an_unobserved_item_has_no_pose_and_no_fallback():
    batch = _batch(_observation())
    spec = _spec(batch)
    scenario = build_scenario("isaac_cylinders_smoke", seed=42)
    unobserved = scenario.items[1]          # item-002: a generated item, never seen
    assert spec.object(unobserved.item_id) is None
    with pytest.raises(SourcePoseUnavailable, match="no generated pose is substituted"):
        source_pose_for(spec, 1, unobserved)


def test_the_bridge_does_not_import_the_generated_layout_at_all():
    src = _read(os.path.join(ORCH, "isaac_bridge.py"))
    imports = src[:src.index("LOG = ")]
    assert "table_pose_for_index" not in imports
    assert "source_pose_for" in imports
    build = src[src.index("def _build_command"):src.index("def _on_feedback")]
    assert "source_pose_for(self.scene_spec" in build
    assert "table_pose_for_index" not in build


def test_the_scene_builder_spawns_from_the_same_selector():
    src = _read(os.path.join(SIM, "scene.py"))
    assert "table_pose_for_index" not in src
    build = src[src.index("def build_items"):src.index("def _build_cad_item")]
    assert "source_pose_for(scene, index, item, self.layout)" in build


def test_the_grasp_yaw_follows_the_observed_heading_and_the_place_yaw_does_not():
    src = _read(os.path.join(SIM, "robot.py"))
    grasp = src[src.index("def _grasp_yaw"):src.index("def _place_yaw")]
    assert "pick_yaw_deg(self.command.source_pose)" in grasp
    place = src[src.index("def _place_yaw"):src.index("def _container_rim_z")]
    assert "_yaw_offset()" in place and "_grasp_yaw()" not in place


# =========================================================================== #
# 8. The bridge: hold on refusal, dispatch from the spec, ignore stale reports
# =========================================================================== #

class _FakeNode:
    def __init__(self):
        self.published = []
        self.execution_publishes = 0

    def create_publisher(self, *a, **k):
        node = self

        class P:
            def publish(self, msg):
                node.published.append(json.loads(msg.data))
        return P()

    def create_subscription(self, *a, **k): return None
    def get_logger(self): return self
    def info(self, *a): pass
    def warn(self, *a): pass
    def error(self, *a): pass
    def publish_execution(self): self.execution_publishes += 1


class _FakeEngine:
    def __init__(self, batch=None, physical=True):
        self.run_id = "run-1"
        self.scenario_revision = 2
        # The demonstration preset: its bin is long enough for the real part.
        self.scenario = build_scenario("isaac_cylinder5_physical", seed=42)
        self.observation_batch = batch
        if batch is not None:
            self.scenario.items = batch.to_waste_items()
        self.selected = None
        self.finished = False
        self.degraded = ""
        self.notes = []
        self.failed = []

        class Source:
            is_physical = physical
            value = "camera" if physical else "sim"

        class C:
            preset, seed = "isaac_cylinder5_physical", 42
            perception_source = Source()
        self.config = C()

    def note_physical_progress(self, *a, **k): self.notes.append((a, k))
    def enter_degraded(self, reason): self.degraded = reason; self.finished = True
    def fail_physical_item(self, placement, reason, details=None):
        self.failed.append((placement.item_id, reason))


def _bridge_module():
    """The real bridge, imported under the harness's ROS stubs."""
    from orchestrator_harness import (                             # noqa: PLC0415
        _install_stub_modules, _remove_stub_modules)
    _install_stub_modules()
    try:
        from wisepack_orchestration import isaac_bridge            # noqa: PLC0415
    finally:
        _remove_stub_modules()
    return isaac_bridge


def _bridge(frames=None):
    module = _bridge_module()
    node = _FakeNode()
    bridge = module.IsaacExecutionBridge(node, workcell=frames or _frames())
    bridge.simulator_ready = True
    return bridge, node


def test_the_bridge_sends_the_synchronized_spec_on_the_scene_command():
    bridge, node = _bridge()
    engine = _FakeEngine(_batch(_observation()))
    bridge.request_scene_reset(engine, engine.scenario_revision)
    command = IsaacCommand.from_dict(node.published[-1])
    assert command.command is IsaacCommandType.RESET_SCENE
    assert command.scene is not None and command.scene.is_physical
    assert command.scene.observation_batch_id == "fp-physical-1"
    assert command.scene.transform_source == PROVENANCE_CONFIGURED_DEMO
    assert command.scene.objects[0].item_id == "item-001"
    assert bridge.requested_fingerprint == scene_fingerprint(
        engine.scenario, bridge.layout, bridge.robot_id, scene=command.scene)
    assert bridge.scene_sync_status()["transform_label"] == "Configured demo transform"
    assert bridge.scene_sync_status()["status"] == "requested"


def test_a_generated_run_sends_no_scene_payload():
    bridge, node = _bridge()
    engine = _FakeEngine(None, physical=False)
    bridge.request_scene_reset(engine, engine.scenario_revision)
    command = IsaacCommand.from_dict(node.published[-1])
    assert command.scene is None
    assert bridge.scene_spec is None
    status = bridge.scene_sync_status()
    assert status["scene_source"] == SCENE_SOURCE_GENERATED
    assert status["synchronized"] is False


def test_a_refused_observation_holds_the_gate_and_publishes_nothing():
    bridge, node = _bridge()
    engine = _FakeEngine(_batch(_observation(pose_valid=False)))
    bridge.request_scene_reset(engine, engine.scenario_revision)
    assert node.published == []
    assert bridge.scene_sync_refusal
    assert not bridge.scene_ready
    assert bridge.scene_status == "refused"
    assert "pose_valid is false" in bridge.scene_block_reason()
    assert any(a[1] == "isaac_scene_sync_refused" for a, _ in engine.notes)
    assert bridge.scene_sync_status()["status"] == "refused"


def test_an_out_of_workarea_observation_refuses_execution_at_the_bridge():
    bridge, node = _bridge()
    far = _observation(model_origin=(-222.164 + 900.0, 97.268, 594.551))
    engine = _FakeEngine(_batch(far))
    bridge.request_scene_reset(engine, engine.scenario_revision)
    assert node.published == []
    assert "outside the configured work-area" in bridge.scene_block_reason()


def test_a_missing_transform_refuses_execution_at_the_bridge():
    bridge, node = _bridge(WorkcellFrames(error="workcell configuration not found"))
    engine = _FakeEngine(_batch(_observation()))
    bridge.request_scene_reset(engine, engine.scenario_revision)
    assert node.published == []
    assert "transform is unavailable" in bridge.scene_block_reason()
    assert bridge.scene_sync_status()["transform_available"] is False


def test_the_dispatched_source_pose_is_the_transformed_observation():
    from wisepack_core.packing import OptimizerConfig, pack_optimized   # noqa: PLC0415
    bridge, node = _bridge()
    batch = _batch(_observation())
    engine = _FakeEngine(batch)
    bridge.request_scene_reset(engine, engine.scenario_revision)
    plan = pack_optimized(engine.scenario, config=OptimizerConfig(seed=42, restarts=2))
    engine.selected = plan
    placement = plan.ordered_placements[0]
    item = engine.scenario.item(placement.item_id)
    container = plan.container(placement.container_id)
    command = bridge._build_command(engine, placement, item, container, 0)
    expected = transform_observation(batch.observations[0], bridge.workcell,
                                     bridge.layout, radius_mm=12.5)
    assert command.source_pose == expected.pose
    assert command.source_pose.frame == TABLE_FRAME
    assert command.source_pose.orientation is not None
    row = table_pose_for_index(0, item, bridge.layout)
    assert command.source_pose.distance_mm(row) > 50.0
    # The same numbers the simulator will spawn the body at.
    assert command.source_pose == bridge.scene_spec.objects[0].source_pose


def test_a_scene_ready_for_a_generated_scene_is_rejected_by_a_physical_run():
    bridge, node = _bridge()
    engine = _FakeEngine(_batch(_observation()))
    bridge.request_scene_reset(engine, engine.scenario_revision)
    ack = SceneAcknowledgement(
        run_id="run-1", scenario_id=engine.scenario.scenario_id,
        scenario_revision=2, preset="isaac_cylinder5_physical", seed=42,
        scene_fingerprint=bridge.requested_fingerprint, object_ids=["item-001"],
        object_count=1, robot_home_verified=True, container_empty_verified=True,
        scene_source=SCENE_SOURCE_GENERATED)
    bridge._on_reset_state(engine, IsaacFeedback(
        state=IsaacState.SCENE_READY, run_id="run-1", scenario_revision=2, scene=ack))
    assert not bridge.scene_ready
    assert "generated scene" in bridge.scene_mismatch


def test_a_matching_physical_acknowledgement_opens_the_gate():
    bridge, node = _bridge()
    engine = _FakeEngine(_batch(_observation()))
    bridge.request_scene_reset(engine, engine.scenario_revision)
    ack = SceneAcknowledgement(
        run_id="run-1", scenario_id=engine.scenario.scenario_id,
        scenario_revision=2, preset="isaac_cylinder5_physical", seed=42,
        scene_fingerprint=bridge.requested_fingerprint, object_ids=["item-001"],
        object_count=1, robot_home_verified=True, container_empty_verified=True,
        scene_source=SCENE_SOURCE_PHYSICAL, observation_batch_id="fp-physical-1",
        transform_source=PROVENANCE_CONFIGURED_DEMO,
        object_poses={"item-001": {"position_m": [0.481, 0.01, 0.42], "readable": True}})
    bridge._on_reset_state(engine, IsaacFeedback(
        state=IsaacState.SCENE_READY, run_id="run-1", scenario_revision=2, scene=ack))
    assert bridge.scene_ready
    status = bridge.scene_sync_status()
    assert status["synchronized"] is True and status["status"] == "synchronized"
    assert status["acknowledged_object_poses"]["item-001"]["readable"] is True


def test_a_stale_failure_report_does_not_hold_the_current_revision():
    bridge, node = _bridge()
    engine = _FakeEngine(_batch(_observation()))
    engine.scenario_revision = 3
    bridge.request_scene_reset(engine, 3)
    bridge._on_reset_state(engine, IsaacFeedback(
        state=IsaacState.RESET_FAILED, run_id="run-1", scenario_revision=2,
        message="stale scene request refused"))
    assert bridge.reset_failed_reason == ""
    assert bridge.reset_in_progress
    assert any(a[1] == "isaac_scene_stale_report_ignored" for a, _ in engine.notes)


# =========================================================================== #
# 9. Contract and dashboard wording
# =========================================================================== #

def test_the_contract_minor_bump_keeps_a_plain_command_identical_in_shape():
    command = IsaacCommand(command=IsaacCommandType.EXECUTE_ITEM, run_id="r",
                           sequence_index=0, item_id="item-001",
                           dimensions=Dimensions(342, 25, 19),
                           source_pose=Pose(1, 2, 3), target_pose=Pose(4, 5, 6))
    doc = command.to_dict()
    assert doc["scene"] is None
    assert "orientation" not in doc["source_pose"]
    assert IsaacCommand.from_dict(doc).scene is None


def test_a_scene_payload_on_an_item_command_is_refused():
    spec = _spec(_batch(_observation()))
    with pytest.raises(ContractError, match="does not carry a scene"):
        IsaacCommand(command=IsaacCommandType.EXECUTE_ITEM, run_id="r",
                     sequence_index=0, item_id="item-001",
                     dimensions=Dimensions(342, 25, 19),
                     source_pose=Pose(1, 2, 3), target_pose=Pose(4, 5, 6),
                     scene=spec)


def test_a_physical_spec_must_name_its_transform():
    with pytest.raises(ContractError, match="transform_source"):
        SceneSpec(scene_source=SCENE_SOURCE_PHYSICAL, objects=[SceneObject(
            "item-001", Dimensions(342, 25, 19), Pose(1, 2, 3))])


def test_a_non_unit_orientation_is_refused_on_the_wire():
    with pytest.raises(ContractError, match="not a unit quaternion"):
        Pose(1, 2, 3, orientation=(0.0, 0.0, 0.0, 2.0))


def test_the_dashboard_shows_the_four_lines_and_never_says_calibrated():
    html = _read(os.path.join(REPO, "web", "index.html"))
    for label in ("Scene source", "Workarea transform", "Isaac scene", "Object"):
        assert f'"{label}"' in html
    block = html[html.index("function renderSceneSync"):html.index("async function refreshPhysical() {")]
    assert "Not synchronized" in block and "Synchronized" in block
    assert "calibrat" not in block.lower()
    assert "warnbox" not in block          # not a banner
    app = _read(os.path.join(REPO, "web", "app.py"))
    assert "Work-area calibration is required before planning or execution" not in app
    assert "CONFIGURED DEMO transform" in app


def test_the_simulator_acknowledges_the_scene_it_built_not_the_request():
    src = _read(os.path.join(SIM, "wisepack_isaac.py"))
    ack = src[src.index("def _scene_acknowledgement"):src.index("def _object_poses_read_back")]
    assert "spec = self.scene_spec" in ack
    assert "command.scene" not in ack
    assert "object_poses=self._object_poses_read_back()" in ack
    pre = src[src.index("def _pre_pick_refusal"):src.index("def prepare_smoke_run")]
    assert "self.scene_spec.object(item) is None" in pre


# =========================================================================== #
# 10. The demonstration preset: a bin the real part fits, both arms can reach
# =========================================================================== #

def test_the_demo_preset_bin_fits_cylinder5_and_is_reachable_by_both_arms():
    from wisepack_core.execution import preset_physical_compatibility  # noqa: PLC0415
    from wisepack_core.isaac_transform import layout_for_robot         # noqa: PLC0415
    from wisepack_core.packing import OptimizerConfig, pack_optimized   # noqa: PLC0415
    from wisepack_core.robots import load_registry                     # noqa: PLC0415
    scenario = build_scenario("isaac_cylinder5_physical", seed=42)
    inner = scenario.container_template.inner_size
    assert inner.x >= 342 + 2 * 10          # the part plus release clearance
    batch = _batch(_observation())
    scenario.items = batch.to_waste_items()
    plan = pack_optimized(scenario, config=OptimizerConfig(seed=42, restarts=2))
    assert len(plan.ordered_placements) == 1
    assert plan.ordered_placements[0].axis.value == "x"
    registry = load_registry()
    for robot_id in ("panda", "xarm7"):
        profile = registry.get(robot_id)
        ok, reason = preset_physical_compatibility("isaac_cylinder5_physical", profile)
        assert ok, reason
        layout_for_robot(profile).validate(inner, 1, clearance_m=0.10)


def test_the_smoke_preset_is_untouched():
    scenario = build_scenario("isaac_cylinders_smoke", seed=42)
    assert len(scenario.items) == 4
    assert scenario.container_template.inner_size == Vec3(300, 220, 150)


# =========================================================================== #
# 11. The handshake follows the revision a batch bumps mid-run
# =========================================================================== #

def test_a_batch_adopted_mid_run_moves_the_requested_scene_with_it():
    """Measured live: the tree parks at the gate, the revision moves, the
    bridge kept asking for revision 1. tick() must re-request for the new one,
    and until it is acknowledged nothing may be approved or dispatched."""
    bridge, node = _bridge()
    engine = _FakeEngine(None, physical=True)          # camera run, no batch yet
    engine.scenario_revision = 1
    bridge.request_scene_reset(engine, 1)              # the generated placeholder
    assert IsaacCommand.from_dict(node.published[-1]).scene is None
    # ScanAndDetect adopts the batch: revision 2, items replaced.
    batch = _batch(_observation())
    engine.observation_batch = batch
    engine.scenario.items = batch.to_waste_items()
    engine.scenario_revision = 2
    assert bridge.scene_block_reason(engine).startswith(
        "the physical scene was requested for scenario revision 1")
    bridge.gate.adopt(engine.run_id)
    bridge.run_open = True
    bridge.tick(engine)
    command = IsaacCommand.from_dict(node.published[-1])
    assert command.command is IsaacCommandType.RESET_SCENE
    assert command.scenario_revision == 2
    assert command.scene is not None and command.scene.observation_batch_id == "fp-physical-1"
    assert bridge.required_revision == 2
    assert not bridge.scene_ready


def test_no_dispatch_while_the_requested_scene_is_not_this_batch():
    bridge, node = _bridge()
    engine = _FakeEngine(_batch(_observation(), batch_id="fp-physical-1"))
    bridge.request_scene_reset(engine, engine.scenario_revision)
    # A second batch arrives and is adopted before the first was acknowledged.
    second = _batch(_observation("obj-2", model_origin=(-150.0, 20.0, 594.0)),
                    batch_id="fp-physical-2")
    engine.observation_batch = second
    engine.scenario.items = second.to_waste_items()
    reason = bridge.spec_mismatch_reason(engine)
    assert "fp-physical-1" in reason and "fp-physical-2" in reason
    assert bridge.scene_block_reason(engine) == reason
    engine.selected = object()                         # would be dispatchable
    assert bridge._dispatch_next(engine) is True       # held, nothing sent
    assert all(IsaacCommand.from_dict(m).command is not IsaacCommandType.EXECUTE_ITEM
               for m in node.published)


def test_a_generated_placeholder_scene_never_serves_a_physical_batch():
    bridge, node = _bridge()
    engine = _FakeEngine(None, physical=True)
    bridge.request_scene_reset(engine, engine.scenario_revision)
    engine.observation_batch = _batch(_observation())
    engine.scenario.items = engine.observation_batch.to_waste_items()
    assert "generated placeholder" in bridge.spec_mismatch_reason(engine)


def test_the_orchestrator_syncs_the_scene_on_every_tick():
    src = _read(os.path.join(ORCH, "hitl_orchestrator.py"))
    tick = src[src.index("def _tick"):src.index("def _publish_event")]
    assert "self.sync_physical_scene()" in tick
    assert tick.index("self.sync_physical_scene()") < tick.index("self.tree.tick_once()")
    assert "self.isaac.scene_block_reason(self.engine)" in src
