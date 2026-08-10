"""The PHYSICAL RealSense, when one is attached. Skipped when none is.

WHY THIS IS A SEPARATE FILE
---------------------------
`tests/test_foundationpose_integration.py` states its own rule at the top: NO
GPU, NO CUDA, NO DOCKER, NO WEIGHTS, NO CAMERA. Its camera tests are about what
WISEPACK does when the device is ABSENT, and they inject that absence so they
test it on every machine. Letting them also accept a present camera would have
retired the failure path the day hardware arrived.

So the opposite claim — the camera is here and it is calibrated — lives here,
and it SKIPS rather than fails when no device is attached. Neither file's
outcome depends on the other's bench state.

IT ASKS THE WORKER, IT DOES NOT OPEN THE CAMERA ITSELF. The worker container
owns RGB-D acquisition (see perception/foundationpose/worker/camera.py), and
testing it from the host would mean a SECOND librealsense — which is exactly the
two-SDK arrangement that ownership exists to prevent. That is not hypothetical
here: this host carries librealsense 2.56.5 and the worker 2.58.3, they report
the device name differently ("Intel RealSense D435" against "RealSense D435"),
and the host build cannot pull frames from this camera at all. A test that
streamed on the host would therefore have measured the host's SDK rather than
the acquisition path WISEPACK actually uses.

WHAT IS ASSERTED IS WHAT WAS MEASURED. Model, serial, intrinsics and depth scale
are read from the device. Nothing is compared against `config/rgbd_sensors.yaml`,
whose numbers are `documented_nominal` — a published specification, not a
calibration of this unit.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))

from wisepack_core.rgbd import (ALIGNMENT_ALIGNED, CameraIntrinsics,  # noqa: E402
                                RGBDFrame)
from wisepack_core.rgbd_sensors import sensor_profile               # noqa: E402

CONTAINER = os.environ.get("WISEPACK_FP_CONTAINER",
                           "wisepack-foundationpose-worker")

#: Runs INSIDE the worker. One device description and one real frame, so the
#: camera is opened once however many tests read the result.
#:
#: THE FIRST OPEN AFTER ANOTHER OWNER RELEASED THE DEVICE OFTEN TIMES OUT. A
#: D4xx that was streaming for someone else a moment ago accepts
#: `pipeline.start()` and then delivers nothing to the first `wait_for_frames`;
#: the next attempt succeeds immediately. That is a bench condition, so it is
#: retried ONCE — and only once, because two consecutive failures are a fault
#: worth failing the test over rather than papering across.
#:
#: `warmup()` TAKES THE MODULE'S OWN COUNT and is not passed a smaller one. The
#: opening frames of a D4xx carry almost no depth while auto-exposure settles —
#: measured here at 8% valid pixels after 10 frames against 99.7% after 30 —
#: and a test that read one of those would be measuring the warmup, not the
#: camera.
PROBE = r'''
import json, sys, time
sys.path.insert(0, "/opt/wisepack-fp-worker")
from camera import available, describe, RGBDStream

usable, reason = available()
out = {"usable": usable, "reason": reason}
if usable:
    out["device"] = describe()
    last = None
    for attempt in range(2):
        try:
            with RGBDStream(align=True) as stream:
                stream.warmup()
                colour, depth, meta = stream.frame()
                out["state"] = stream.state()
            out["frame"] = {
                "colour_shape": list(colour.shape),
                "colour_dtype": str(colour.dtype),
                "depth_shape": list(depth.shape),
                "depth_dtype": str(depth.dtype),
                "depth_valid_fraction": float((depth > 0).mean()),
                "meta": meta,
            }
            out["stream_attempts"] = attempt + 1
            break
        except Exception as exc:
            last = exc
            time.sleep(2)
    else:
        raise SystemExit(f"the streams did not start in two attempts: {last}")
print(json.dumps(out))
'''


@pytest.fixture(scope="module")
def measured():
    """The worker's answer about the physical camera, or a skip saying why not.

    EVERY SKIP NAMES THE MISSING LAYER. "no camera" covers a stopped container,
    an unplugged cable and a device held by another process, and sending someone
    to re-seat a cable when the worker is simply not running is the failure this
    whole camera path is written to avoid.
    """
    if not shutil.which("docker"):
        pytest.skip("docker is not installed; the worker owns RGB-D acquisition")
    running = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER],
        capture_output=True, text=True)
    if running.returncode != 0 or running.stdout.strip() != "true":
        pytest.skip(f"the worker container {CONTAINER!r} is not running; start "
                    "it with ./scripts/setup_foundationpose.sh --no-build --run")
    probe = subprocess.run(["docker", "exec", "-i", CONTAINER, "python3", "-"],
                           input=PROBE, capture_output=True, text=True,
                           timeout=180)
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip().splitlines()[-1:]
        if any("busy" in line.lower() or "resource" in line.lower()
               for line in detail):
            pytest.skip(f"the device is held by another process: {detail}")
        pytest.fail(f"the worker could not probe the camera: {detail}")
    document = json.loads(probe.stdout.strip().splitlines()[-1])
    if not document["usable"]:
        pytest.skip(f"no physical RealSense attached: {document['reason']}")
    return document


def test_a_physical_camera_reports_itself_usable_with_no_reason(measured):
    """A usable camera explains nothing; only a refusal carries a reason."""
    assert measured["usable"] is True
    assert measured["reason"] == ""


def test_the_attached_device_is_the_sensor_wisepack_declares(measured):
    """The bench camera must be the sensor the configuration is written for.

    Not a tautology: a D415 or a D455 would enumerate perfectly well and would
    have a different depth FOV and baseline, so every nominal number in
    `config/rgbd_sensors.yaml` would silently describe the wrong hardware.

    CONTAINMENT, NOT EQUALITY, and for a reason: librealsense 2.56 reports
    "Intel RealSense D435" where 2.58 reports "RealSense D435". Pinning the
    exact string would make this a test of the SDK version.
    """
    device = measured["device"]
    declared = sensor_profile("d435", repo_root=REPO)
    assert declared.model in device["name"], (
        f"the attached device reports {device['name']!r}, which is not the "
        f"declared {declared.model!r}")
    assert device["serial_number"], "a device with no serial cannot be audited"
    assert device["firmware_version"]
    assert device["product_line"] == "D400"


def test_the_depth_scale_comes_from_the_device_in_millimetres(measured):
    """Millimetres per raw unit, read rather than assumed. A D4xx reports about
    1.0; what matters is that the value is the device's own and is positive."""
    device = measured["device"]
    scale = device["depth_scale_mm_per_unit"]
    assert scale is not None and scale > 0
    assert device["depth_scale_m_per_unit"] == pytest.approx(scale / 1000.0)


def test_the_profile_is_negotiated_from_what_the_device_offers(measured):
    """A resolution the device does not offer fails at `pipeline.start()`, so
    the pair actually used must appear in the device's own synchronised set."""
    state, options = measured["state"], measured["device"]["synchronised_profiles"]
    assert options, "no colour(BGR8)+depth(Z16) combination at a shared size"
    assert {"width": state["width"], "height": state["height"],
            "fps": state["fps"]} in options


def test_colour_and_depth_start_together_at_that_profile(measured):
    """Both streams, one bundle, one instant, at the negotiated size."""
    state, frame = measured["state"], measured["frame"]
    assert frame["colour_shape"][:2] == [state["height"], state["width"]]
    assert frame["depth_shape"] == [state["height"], state["width"]]
    assert frame["colour_dtype"] == "uint8"
    assert frame["depth_dtype"] == "uint16"
    assert frame["meta"]["device_timestamp_ms"] > 0


def test_the_measured_calibration_is_present_and_belongs_to_this_frame(measured):
    """Intrinsics are read from the device AND match the image they describe.

    Intrinsics for another resolution are the classic silent error: they produce
    poses wrong by a scale factor while looking entirely reasonable.
    """
    state = measured["state"]
    for key in ("colour_intrinsics", "depth_intrinsics"):
        intrinsics = state[key]
        assert intrinsics, f"{key} were not read from the device"
        assert intrinsics["fx"] > 0 and intrinsics["fy"] > 0
        assert intrinsics["width"] == state["width"]
        assert intrinsics["height"] == state["height"]
        # The principal point must lie INSIDE the image it belongs to.
        assert 0 <= intrinsics["cx"] <= intrinsics["width"]
        assert 0 <= intrinsics["cy"] <= intrinsics["height"]
        assert "distortion" in intrinsics["model"]
        assert len(intrinsics["coeffs"]) == 5


def test_depth_is_aligned_to_colour_and_that_is_verified(measured):
    """Verified against the colour intrinsics, not merely claimed."""
    state = measured["state"]
    assert state["aligned"] is True
    assert state["alignment_verified"] is True
    colour_k, depth_k = state["colour_intrinsics"], state["depth_intrinsics"]
    assert depth_k["fx"] == pytest.approx(colour_k["fx"], abs=1e-3)
    assert depth_k["cx"] == pytest.approx(colour_k["cx"], abs=1e-3)


def test_the_frame_carries_actual_depth_rather_than_an_empty_image(measured):
    """A stream that starts and returns all zeros is not a measurement. A tenth
    of the frame is a deliberately low bar — this is an acquisition check, not a
    judgement about what the camera is pointed at."""
    valid = measured["frame"]["depth_valid_fraction"]
    assert valid > 0.1, f"only {valid:.1%} of depth pixels carry a measurement"


def test_the_measured_device_produces_the_generic_rgbd_frame(measured):
    """The SAME contract the simulated path produces, from measured numbers.

    Nothing downstream may branch on real-versus-simulated; only the provenance
    differs, and it is carried in `source` rather than in a separate type.
    """
    state = measured["state"]
    intrinsics = state["colour_intrinsics"]
    frame = RGBDFrame(
        intrinsics=CameraIntrinsics(
            fx=intrinsics["fx"], fy=intrinsics["fy"],
            cx=intrinsics["cx"], cy=intrinsics["cy"],
            width=intrinsics["width"], height=intrinsics["height"],
            source=f"device_sdk librealsense {state['device']['name']} "
                   f"serial {state['device']['serial_number']}"),
        depth_scale_mm=state["depth_scale_mm_per_unit"],
        alignment=ALIGNMENT_ALIGNED,
        rgb_available=True, depth_available=True,
        camera_model=state["device"]["name"])
    usable, reason = frame.usability()
    assert usable is True, reason
    assert "device_sdk" in frame.intrinsics.source
    assert 40.0 < frame.intrinsics.horizontal_fov_deg < 120.0
