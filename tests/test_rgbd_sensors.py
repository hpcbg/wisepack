"""The simulated sensor must be the D435, and must not pretend to be one.

Two backends, one declaration, and the difference between them stated rather
than blurred: a rendered frame is synthetic and a device frame is measured.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))

from wisepack_core.rgbd_sensors import (                         # noqa: E402
    BACKEND_ISAAC, BACKEND_REALSENSE, PROVENANCE_MEASURED,
    PROVENANCE_NOMINAL, SensorProfileError, load_sensor_profiles,
    sensor_profile)


def test_the_declared_sensor_is_a_d435():
    """The physical target is a D435. A D455 profile would be a different
    sensor: ~95 deg depth FOV and a 95 mm baseline against 87 deg and 50 mm."""
    profile = sensor_profile("d435")
    assert profile.model == "RealSense D435"
    assert profile.vendor == "Intel"


def test_no_d455_is_relabelled_as_a_d435():
    """NVIDIA ships D455/D457/D555 assets and no D435. Renaming one would put a
    different sensor behind the label."""
    text = open(os.path.join(REPO, "config", "rgbd_sensors.yaml"),
                encoding="utf-8").read()
    assert "sensor_id: d455" not in text
    camera = open(os.path.join(REPO, "simulators", "isaac", "rgbd_camera.py"),
                  encoding="utf-8").read()
    # The module may EXPLAIN why the D455 asset is not used; what it must not do
    # is load one.
    assert "rsd455.usd" not in camera
    assert "get_assets_root_path" not in camera


def test_the_nominal_intrinsics_are_derived_from_the_declared_field_of_view():
    profile = sensor_profile("d435")
    fx, fy = profile.colour.focal_lengths_px()
    # 1280 x 720 at 69.4 x 42.5 deg -> ~924 px, and square pixels.
    assert 900 < fx < 950
    assert abs(fx - fy) < 5
    k = profile.colour.intrinsics_matrix()
    assert k[0][2] == pytest.approx(640.0)
    assert k[1][2] == pytest.approx(360.0)


def test_the_profile_is_nominal_and_says_so():
    """Two devices of one model differ. A published specification is not a
    calibration of anything, and must never be presented as a measurement."""
    profile = sensor_profile("d435")
    assert profile.provenance == PROVENANCE_NOMINAL
    assert not profile.is_measured
    assert "not a measurement" in profile.provenance_note.lower()
    assert profile.serial_number == ""


def test_the_depth_module_is_wider_than_the_colour_camera():
    """A real D435's depth FOV exceeds its colour FOV, which is why aligned
    depth has invalid borders. Recorded so the simulated frame's cleanliness is
    not mistaken for the real sensor's behaviour."""
    profile = sensor_profile("d435")
    assert profile.depth.hfov_deg > profile.colour.hfov_deg
    assert profile.baseline_mm == pytest.approx(50.0)


def test_the_simulated_camera_is_never_called_a_realsense():
    profile = sensor_profile("d435")
    simulated = profile.describe_backend(BACKEND_ISAAC)
    assert simulated["camera_backend"] == "Isaac Sim"
    assert simulated["provenance"] == "synthetic"
    assert "compatible simulated" in simulated["camera_model"]
    assert "NOT the physical device" in simulated["note"]

    physical = profile.describe_backend(BACKEND_REALSENSE)
    assert physical["camera_backend"] == "RealSense"
    assert physical["provenance"] == "measured"
    assert physical["camera_model"] == "RealSense D435"


def test_what_the_simulation_cannot_reproduce_is_listed():
    """A clean synthetic depth image is not evidence about a real one."""
    profile = sensor_profile("d435")
    joined = " ".join(profile.limitations).lower()
    assert "stereo" in joined
    assert "ir projector" in joined or "projector" in joined


def test_an_unknown_sensor_is_refused_with_the_declared_ones_listed():
    with pytest.raises(SensorProfileError) as exc:
        sensor_profile("d455")
    assert "unknown sensor" in str(exc.value)


def test_the_simulated_camera_imports_no_camera_sdk():
    """§: pyrealsense2 belongs to the physical path alone. A synthetic frame
    must never travel a code path implying a device was read."""
    source = open(os.path.join(REPO, "simulators", "isaac", "rgbd_camera.py"),
                  encoding="utf-8").read()
    import ast
    for node in ast.walk(ast.parse(source)):
        modules = []
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        for module in modules:
            assert "realsense" not in module.lower(), module


def test_the_sensor_profile_module_imports_no_simulator_and_no_sdk():
    source = open(os.path.join(REPO, "wisepack_ws", "src", "wisepack_core",
                               "wisepack_core", "rgbd_sensors.py"),
                  encoding="utf-8").read()
    import ast
    for node in ast.walk(ast.parse(source)):
        modules = []
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        for module in modules:
            lowered = module.lower()
            for forbidden in ("isaacsim", "omni", "pxr", "pyrealsense"):
                assert forbidden not in lowered, module


def test_the_isaac_mask_source_is_labelled_as_ground_truth():
    source = open(os.path.join(REPO, "simulators", "isaac", "rgbd_camera.py"),
                  encoding="utf-8").read()
    assert 'MASK_SOURCE_ISAAC_GT = "isaac_instance_gt"' in source


def test_a_measured_profile_would_carry_a_serial():
    """The shape a future measured profile takes, so the distinction is real
    rather than aspirational."""
    from wisepack_core.rgbd_sensors import SensorProfile
    measured = SensorProfile(
        sensor_id="d435-serial123", model="RealSense D435",
        provenance=PROVENANCE_MEASURED, serial_number="123456789")
    assert measured.is_measured
    assert measured.describe_backend(BACKEND_REALSENSE)["provenance"] == "measured"
