"""6-DoF pose, symmetry, RGB-D framing and the object registry.

Every test here runs with NO GPU, NO depth camera, NO FoundationPose and NO
CUDA. That is deliberate and is the §34 requirement: the ordinary
`python3 -m pytest tests/ -q` must stay usable on a machine that has none of
those, so the parts of the RGB-D pipeline WISEPACK actually owns — the
representation, the validation and the symmetry reasoning — are testable
everywhere.

WHAT IS BEING PROTECTED. Each check below corresponds to a failure that
produces a CONFIDENT WRONG ANSWER rather than an error:

    an unnormalised quaternion       a rotation that is not a rotation
    an axial symmetry ignored        a fabricated angle published as measured
    intrinsics from another size     a pose wrong by a scale factor
    unaligned depth                  a mask indexing the wrong depth pixels
    a mesh in the wrong unit         a 1000x "fit" nobody notices
    a camera-frame pose relabelled   objects placed inside the camera
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORE = os.path.join(REPO, "wisepack_ws", "src", "wisepack_core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from wisepack_core.domain import PhysicalObservation                # noqa: E402
from wisepack_core.pose import (                                    # noqa: E402
    CAMERA_OPTICAL_FRAME, Orientation, PoseError, RigidTransform, Symmetry,
    SymmetryType, WORKAREA_FRAME, canonicalize,
)
from wisepack_core.rgbd import (                                    # noqa: E402
    ALIGNMENT_ALIGNED, ALIGNMENT_UNALIGNED, ALIGNMENT_UNKNOWN,
    CameraIntrinsics, ObjectModel, ObjectModelRegistry, RGBDError, RGBDFrame,
    load_object_registry,
)


# --------------------------------------------------------------------------- #
# 1. Orientation — the quaternion is the authority
# --------------------------------------------------------------------------- #


def test_a_zero_quaternion_is_rejected_rather_than_normalised():
    """The classic "nobody filled this in" value. It has no rotation to fix."""
    with pytest.raises(PoseError) as exc:
        Orientation(0.0, 0.0, 0.0, 0.0)
    assert "zero norm" in str(exc.value)


def test_a_badly_scaled_quaternion_is_rejected_not_silently_rescaled():
    """0.5 is not float drift — it is a different quantity."""
    with pytest.raises(PoseError):
        Orientation(0.0, 0.0, 0.0, 0.5)
    # ... but the producer that KNOWS it emits unnormalised output can say so.
    fixed = Orientation.normalized(0.0, 0.0, 0.0, 0.5)
    assert fixed.w == pytest.approx(1.0)


def test_float32_drift_is_repaired_silently():
    """Every producer in this pipeline computes in float32 somewhere."""
    drifted = Orientation(0.0, 0.0, 0.0, 1.0 + 1e-7)
    assert drifted.norm == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("yaw", [0.0, 45.0, 90.0, -31.0, 179.9, -179.9])
def test_a_planar_yaw_round_trips_through_the_quaternion(yaw):
    assert Orientation.from_yaw_deg(yaw).yaw_deg == pytest.approx(yaw, abs=1e-9)


def test_from_matrix_handles_the_180_degree_cases():
    """The naive trace formula divides by ~0 exactly here."""
    flip_z = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    assert abs(Orientation.from_matrix(flip_z).yaw_deg) == pytest.approx(180.0)
    flip_x = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    roll, _, _ = Orientation.from_matrix(flip_x).rpy_deg()
    assert abs(roll) == pytest.approx(180.0)


def test_angle_between_orientations_is_convention_free():
    a = Orientation.from_yaw_deg(10.0)
    b = Orientation.from_yaw_deg(40.0)
    assert a.angle_to_deg(b) == pytest.approx(30.0, abs=1e-6)
    # q and -q are the SAME rotation; the metric must not see a difference.
    negated = Orientation.normalized(-a.x, -a.y, -a.z, -a.w)
    assert a.angle_to_deg(negated) == pytest.approx(0.0, abs=1e-6)


def test_rpy_survives_gimbal_lock_instead_of_raising():
    """Pitch at exactly +/-90 deg: |sin| can exceed 1 by float noise."""
    straight_up = Orientation.normalized(0.0, 0.7071067811865476, 0.0,
                                         0.7071067811865476)
    _, pitch, _ = straight_up.rpy_deg()
    assert abs(pitch) == pytest.approx(90.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# 2. Symmetry — the fabricated-angle problem
# --------------------------------------------------------------------------- #


def test_an_axial_symmetry_names_the_unobservable_degree_of_freedom():
    symmetry = Symmetry(type=SymmetryType.AXIAL, axis="z")
    assert symmetry.ambiguous_dof == ["rotation_about_z"]
    assert symmetry.rotation_observable is False


def test_two_estimates_of_one_cylinder_differ_only_by_an_unobservable_spin():
    """THE POINT OF THE WHOLE SYMMETRY MACHINERY.

    Two poses of the same physical cylinder that differ only by rotation about
    its own axis are the SAME physical state. Compared raw they look 149 degrees
    apart, which would make a perfectly repeatable sensor look wildly unstable —
    the error is in the comparison, not the sensor.
    """
    lying_along_x = Orientation.normalized(0.0, 0.7071067811865476, 0.0,
                                           0.7071067811865476)
    first = lying_along_x.multiply(Orientation.from_yaw_deg(37.0))
    second = lying_along_x.multiply(Orientation.from_yaw_deg(-112.0))

    assert first.angle_to_deg(second) > 100.0, "raw comparison must disagree"

    symmetry = Symmetry(type=SymmetryType.AXIAL, axis="z")
    a = canonicalize(first, symmetry)
    b = canonicalize(second, symmetry)
    assert a.angle_to_deg(b) == pytest.approx(0.0, abs=1e-6)

    # And what SURVIVES canonicalisation is the observable part: where the axis
    # points. That must not be destroyed along with the spin.
    for value, expected in zip(a.axis("z"), lying_along_x.axis("z")):
        assert value == pytest.approx(expected, abs=1e-6)


def test_canonicalisation_leaves_an_asymmetric_object_untouched():
    """No symmetry means every rotational DoF is real. Nothing may be removed."""
    pose = Orientation.normalized(0.2, 0.3, 0.4, 0.85)
    assert canonicalize(pose, Symmetry()).angle_to_deg(pose) == pytest.approx(0.0)


def test_a_discrete_symmetry_wraps_into_one_sector():
    """A hex nut is the same at 0 and at 60 degrees."""
    symmetry = Symmetry(type=SymmetryType.DISCRETE, axis="z", fold=6)
    assert symmetry.ambiguous_dof == ["rotation_about_z_modulo_60deg"]
    a = canonicalize(Orientation.from_yaw_deg(10.0), symmetry)
    b = canonicalize(Orientation.from_yaw_deg(70.0), symmetry)
    assert a.angle_to_deg(b) == pytest.approx(0.0, abs=1e-4)


def test_a_discrete_symmetry_needs_a_fold():
    with pytest.raises(PoseError):
        Symmetry(type=SymmetryType.DISCRETE, axis="z")


def test_a_flip_symmetry_normalises_the_axis_DIRECTION_only():
    """An end-for-end identical tube: which end is 'up' is not observable."""
    symmetry = Symmetry(type=SymmetryType.FLIP, axis="z")
    down = Orientation.normalized(1.0, 0.0, 0.0, 0.0)          # z -> -z
    assert canonicalize(down, symmetry).axis("z")[2] >= -1e-9


# --------------------------------------------------------------------------- #
# 3. RigidTransform — a homography is not an extrinsic
# --------------------------------------------------------------------------- #


def test_a_transform_with_no_method_is_not_valid():
    """An identity default must never pass as a measured calibration.

    This is what stops a camera-frame pose being relabelled as a work-area pose
    by a transform nobody ever measured.
    """
    assert not RigidTransform(WORKAREA_FRAME, CAMERA_OPTICAL_FRAME).valid
    assert RigidTransform(WORKAREA_FRAME, CAMERA_OPTICAL_FRAME,
                          method="charuco_solvepnp").valid


def test_a_transform_round_trips_through_its_inverse():
    transform = RigidTransform(
        WORKAREA_FRAME, CAMERA_OPTICAL_FRAME,
        translation_mm=(120.0, -45.0, 700.0),
        rotation=Orientation.from_yaw_deg(37.0), method="charuco_solvepnp")
    point = (10.0, 20.0, 30.0)
    there = transform.apply_to_position(point)
    back = transform.inverse().apply_to_position(there)
    for value, expected in zip(back, point):
        assert value == pytest.approx(expected, abs=1e-9)


def test_a_transform_to_its_own_frame_is_refused():
    with pytest.raises(PoseError):
        RigidTransform(WORKAREA_FRAME, WORKAREA_FRAME, method="x")


def test_transform_provenance_survives_serialisation():
    transform = RigidTransform(
        WORKAREA_FRAME, CAMERA_OPTICAL_FRAME, method="charuco_solvepnp",
        revision="cal-0007", measured_at="2026-08-09T12:00:00Z",
        reprojection_error_px=0.31, sample_count=24)
    revived = RigidTransform.from_dict(
        json.loads(json.dumps(transform.to_dict())))
    assert revived.method == "charuco_solvepnp"
    assert revived.revision == "cal-0007"
    assert revived.reprojection_error_px == pytest.approx(0.31)
    assert revived.sample_count == 24
    assert revived.valid


def test_an_unmeasured_reprojection_error_is_none_not_zero():
    """None means "this method does not measure its own error". Zero would
    render as a perfect calibration."""
    assert RigidTransform(WORKAREA_FRAME, CAMERA_OPTICAL_FRAME,
                          method="fixed_mount").reprojection_error_px is None


# --------------------------------------------------------------------------- #
# 4. PhysicalObservation — additive, backward-compatible 6-DoF
# --------------------------------------------------------------------------- #


def test_a_planar_observation_still_behaves_exactly_as_before():
    """The whole backward-compatibility promise in one test."""
    observation = PhysicalObservation(observation_id="p", x_mm=82.4, y_mm=46.1,
                                      yaw_deg=-31.0)
    assert observation.yaw_deg == -31.0                 # EXACT, not 30.999999
    assert observation.pose_valid is True
    assert observation.orientation.yaw_deg == pytest.approx(-31.0, abs=1e-6)
    document = observation.to_dict()
    assert document["pose"]["yaw_deg"] == -31.0
    assert document["pose"]["x_mm"] == 82.4


def test_a_legacy_document_without_orientation_still_parses():
    """A batch recorded before 6-DoF existed must survive the upgrade."""
    legacy = {"observation_id": "old",
              "pose": {"x_mm": 5.0, "y_mm": 6.0, "yaw_deg": 12.0}}
    revived = PhysicalObservation.from_dict(legacy)
    assert revived.yaw_deg == 12.0
    assert revived.orientation.yaw_deg == pytest.approx(12.0, abs=1e-6)
    assert revived.symmetry is None


def test_a_six_dof_observation_round_trips_through_json():
    observation = PhysicalObservation(
        observation_id="c", x_mm=120.0, y_mm=-40.0, z_mm=310.0,
        orientation=Orientation.normalized(0.0, 0.7071067811865476, 0.0,
                                           0.7071067811865476),
        symmetry=Symmetry(type=SymmetryType.AXIAL, axis="z"),
        perception_method="foundationpose_rgbd",
        object_model_id="tutorial_bolt",
        frame_id=CAMERA_OPTICAL_FRAME,
        measured_dof=("x", "y", "z", "roll", "pitch"))
    revived = PhysicalObservation.from_dict(
        json.loads(json.dumps(observation.to_dict())))
    assert revived.frame_id == CAMERA_OPTICAL_FRAME
    assert revived.perception_method == "foundationpose_rgbd"
    assert revived.object_model_id == "tutorial_bolt"
    assert revived.symmetry.type is SymmetryType.AXIAL
    assert revived.orientation.angle_to_deg(observation.orientation) == \
        pytest.approx(0.0, abs=1e-6)
    assert set(revived.measured_dof) == {"x", "y", "z", "roll", "pitch"}


def test_yaw_and_quaternion_can_never_disagree():
    """One pose, two representations. A stale yaw beside a fresh quaternion is
    the bug this reconciliation exists to make impossible."""
    observation = PhysicalObservation(
        observation_id="r", x_mm=0.0, y_mm=0.0,
        yaw_deg=999.0,                                   # wrong on purpose
        orientation=Orientation.from_yaw_deg(-30.0))
    assert observation.yaw_deg == pytest.approx(-30.0, abs=1e-6)


def test_an_unusable_pose_is_carried_and_flagged_not_dropped():
    """A pose that must not be acted on is still evidence worth publishing."""
    observation = PhysicalObservation(
        observation_id="u", x_mm=1.0, y_mm=2.0, pose_valid=False,
        frame_id=CAMERA_OPTICAL_FRAME)
    assert observation.to_dict()["pose"]["valid"] is False


def test_the_serialised_pose_carries_human_readable_angles_too():
    observation = PhysicalObservation(
        observation_id="h", x_mm=0.0, y_mm=0.0,
        orientation=Orientation.from_yaw_deg(45.0))
    rpy = observation.to_dict()["pose"]["rpy_deg"]
    assert rpy is not None and rpy[2] == pytest.approx(45.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# 5. Intrinsics
# --------------------------------------------------------------------------- #


def test_intrinsics_for_the_wrong_resolution_are_rejected():
    """A principal point outside the image is a borrowed calibration."""
    with pytest.raises(RGBDError) as exc:
        CameraIntrinsics(fx=1386.0, fy=1386.0, cx=960.0, cy=540.0,
                         width=640, height=480)
    assert "different resolution" in str(exc.value)


def test_intrinsics_scale_correctly_with_the_image():
    """Halving the image halves every intrinsic. Getting this wrong is a
    factor-of-two pose error that looks entirely plausible."""
    full = CameraIntrinsics(1386.0, 1386.0, 960.0, 540.0, 1920, 1080,
                            source="device_sdk")
    half = full.scaled(960, 540)
    assert half.fx == pytest.approx(693.0)
    assert half.cx == pytest.approx(480.0)
    assert half.horizontal_fov_deg == pytest.approx(full.horizontal_fov_deg,
                                                    abs=1e-9)


def test_a_non_positive_focal_length_is_rejected():
    with pytest.raises(RGBDError):
        CameraIntrinsics(0.0, 500.0, 320.0, 240.0, 640, 480)


def test_the_tutorial_intrinsics_load_and_are_sane():
    """The one intrinsic set that genuinely exists in the reference assets."""
    k = CameraIntrinsics.from_matrix(
        [[554.2562509926996, 0.0, 320.0],
         [0.0, 554.2562509926996, 240.0],
         [0.0, 0.0, 1.0]], 640, 480, source="file:cam_K.txt")
    assert k.horizontal_fov_deg == pytest.approx(60.0, abs=0.1)
    assert k.deproject(320.0, 240.0, 500.0) == (0.0, 0.0, 500.0)


def test_deprojection_is_millimetres_in_millimetres_out():
    k = CameraIntrinsics(500.0, 500.0, 320.0, 240.0, 640, 480)
    x, y, z = k.deproject(420.0, 240.0, 1000.0)
    assert (x, y, z) == (pytest.approx(200.0), pytest.approx(0.0), 1000.0)


# --------------------------------------------------------------------------- #
# 6. RGBDFrame
# --------------------------------------------------------------------------- #


def _frame(**overrides) -> RGBDFrame:
    base = dict(
        intrinsics=CameraIntrinsics(500.0, 500.0, 320.0, 240.0, 640, 480,
                                    source="test"),
        rgb_available=True, depth_available=True,
        alignment=ALIGNMENT_ALIGNED, depth_scale_mm=1.0)
    base.update(overrides)
    return RGBDFrame(**base)


def test_a_complete_frame_is_usable():
    usable, reason = _frame().usability()
    assert usable and reason == ""


@pytest.mark.parametrize("overrides,expected", [
    ({"rgb_available": False}, "no colour image"),
    ({"depth_available": False}, "no depth image"),
    ({"intrinsics": None}, "no camera intrinsics"),
    ({"alignment": ALIGNMENT_UNALIGNED}, "wrong depth pixels"),
    ({"alignment": ALIGNMENT_UNKNOWN}, "wrong depth pixels"),
])
def test_every_way_a_frame_can_be_unusable_is_reported_with_a_reason(
        overrides, expected):
    usable, reason = _frame(**overrides).usability()
    assert not usable and expected in reason


def test_unknown_alignment_is_not_treated_as_aligned():
    """Nobody said is not the same as yes. Assuming yes indexes the wrong
    depth pixels and produces a pose that looks like sensor noise."""
    assert not _frame(alignment=ALIGNMENT_UNKNOWN).depth_aligned


def test_intrinsics_that_do_not_match_the_frame_size_are_caught():
    frame = _frame(width=1920, height=1080)
    usable, reason = frame.usability()
    assert not usable and "640x480" in reason and "1920x1080" in reason


def test_a_non_positive_depth_scale_is_refused():
    with pytest.raises(RGBDError):
        RGBDFrame(depth_scale_mm=0.0)


def test_a_frame_round_trips_through_json():
    frame = _frame(camera_model="Reference RGB-D", depth_invalid_fraction=0.03,
                   captured_at="2026-08-09T12:00:00Z")
    revived = RGBDFrame.from_dict(json.loads(json.dumps(frame.to_dict())))
    assert revived.usable
    assert revived.camera_model == "Reference RGB-D"
    assert revived.depth_invalid_fraction == pytest.approx(0.03)
    assert revived.intrinsics.fx == pytest.approx(500.0)


# --------------------------------------------------------------------------- #
# 7. Object model registry
# --------------------------------------------------------------------------- #


def test_the_shipped_registry_loads_without_error():
    registry = load_object_registry()
    assert registry.error == "", registry.error
    assert "tutorial_bolt" in registry.models
    assert "wisepack_cylinder_proxy" in registry.models


def test_mesh_units_are_declared_and_convert_to_millimetres():
    """Neither STL nor OBJ records a unit. A wrong one is a 1000x error."""
    registry = load_object_registry()
    assert registry.require("tutorial_bolt").mesh_scale_to_mm == 1000.0
    assert registry.require("wisepack_cylinder_proxy").mesh_scale_to_mm == 1.0


def test_an_unknown_mesh_unit_is_refused():
    with pytest.raises(RGBDError):
        ObjectModel(model_id="x", mesh_units="furlong")


def test_a_non_positive_scale_is_refused():
    with pytest.raises(RGBDError):
        ObjectModel(model_id="x", scale=0.0)


def test_the_cylinder_proxy_declares_its_axial_symmetry():
    """WISEPACK's own object is the one whose symmetry matters most."""
    model = load_object_registry().require("wisepack_cylinder_proxy")
    assert model.symmetry.type is SymmetryType.AXIAL
    assert model.symmetry.ambiguous_dof == ["rotation_about_z"]
    assert model.symmetry.rotation_observable is False


def test_the_regression_object_is_asymmetric_on_purpose():
    """A first regression must not let a wrong orientation hide behind a
    symmetry."""
    assert load_object_registry().require(
        "tutorial_bolt").symmetry.type is SymmetryType.NONE


def test_a_model_with_no_mesh_is_reported_unusable_with_a_reason():
    """The cylinder proxy has no CAD, and pretending otherwise would feed a
    fabricated shape to a model-based estimator."""
    registry = load_object_registry()
    usable, reason = registry.require("wisepack_cylinder_proxy").usability(
        registry.root)
    assert not usable and "declares no mesh" in reason


def test_a_missing_mesh_file_is_reported_rather_than_assumed_present():
    model = ObjectModel(model_id="m", mesh_path="nowhere/none.obj")
    usable, reason = model.usability("/tmp")
    assert not usable and "not found" in reason


def test_an_unknown_model_raises_and_names_what_is_configured():
    registry = load_object_registry()
    with pytest.raises(RGBDError) as exc:
        registry.require("no_such_model")
    assert "tutorial_bolt" in str(exc.value)


def test_duplicate_model_ids_are_refused():
    with pytest.raises(RGBDError):
        ObjectModelRegistry.from_dict(
            {"objects": [{"model_id": "dup"}, {"model_id": "dup"}]})


def test_a_broken_registry_reports_the_error_instead_of_looking_empty(tmp_path):
    """"No models configured" and "the configuration is wrong" are different
    facts, and rendering them the same way sends someone to the wrong place."""
    broken = tmp_path / "broken.yaml"
    broken.write_text("objects: [ {model_id: x, mesh_units: furlong} ]")
    registry = load_object_registry(str(broken))
    assert registry.error and "furlong" in registry.error
    assert registry.models == {}


def test_the_registry_is_the_only_place_mesh_paths_are_configured():
    """A path hard-coded in a provider is a second registry that agrees with
    this one only until someone edits it."""
    import pathlib
    import re
    # A FILENAME, not the substring ".obj" — which also occurs inside
    # `object_type`, `model.object_type` and every other identifier containing
    # "obj". The original check failed the build for a provider that mentioned
    # an object type, which is not a mesh path and not the thing being
    # prevented.
    mesh_file = re.compile(r"[\w./-]+\.(?:obj|stl|ply)\b", re.IGNORECASE)
    providers = pathlib.Path(REPO) / "perception" / "providers"
    for source in providers.glob("*.py"):
        found = mesh_file.findall(source.read_text())
        assert not found, (
            f"{source.name} names a mesh file directly ({found}) — mesh paths "
            "belong in config/perception_objects.yaml")
