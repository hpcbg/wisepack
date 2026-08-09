"""The real detector, on a saved frame, against a recorded measurement.

WHY THIS EXISTS
---------------
The `fasterrcnn_bottle` provider was ADAPTED from a working implementation in
another repository so that WISEPACK could stop depending on that repository at
runtime. The failure mode of such a port is not a crash — it is a detector that
still runs, still looks right in the annotated image, and quietly measures
something slightly different. Every test in `test_perception.py` would still
pass, because they all feed the adapter JSON.

So this file runs the ACTUAL network on an ACTUAL frame and compares against the
measurement the pre-port implementation produced on that same frame with the same
weights:

    2 objects
      (86.782, 83.553) mm, -115.112 deg, confidence 0.999
      (41.008, 59.853) mm,   24.302 deg, confidence 0.997
    calibration: valid, all four markers in frame

SKIPPED, NOT FAILED, when the environment is absent. torch, OpenCV and a 159 MB
checkpoint are not present in ordinary CI, and a red suite on a laptop teaches
people to ignore red suites. Run it where the perception environment exists:

    .venv-perception/bin/python -m pytest tests/test_perception_detector_regression.py -v

TOLERANCES. Position and angle are compared to 0.01 mm / 0.01 deg — they are
derived from integer pixel centres through a homography and are reproducible
exactly. Confidence gets 1e-3, because it is a float32 network output and cuDNN
kernel selection moves the last few digits between runs and between devices.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"),
              os.path.join(REPO, "perception")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: The frame. Committed to WISEPACK (93 KB) precisely so this test needs no
#: camera, no network and no other repository. It shows the printed ArUco board
#: with all four corner markers, two bottles with caps and one without.
REFERENCE_IMAGE = os.path.join(REPO, "tests", "data", "perception",
                               "calibrated-scene.jpg")

#: The recorded measurement, produced by the implementation this provider was
#: adapted from, on the frame above, with /data/arise/models/best_model.pth.
REFERENCE_OBJECTS = [
    {"x": 86.78183234019883, "y": 83.5534700206283,
     "yaw": -115.11248512270495, "conf": 0.99905, "selected": True},
    {"x": 41.00769077715643, "y": 59.85301811445141,
     "yaw": 24.302357554761613, "conf": 0.99679, "selected": False},
]

POSITION_TOLERANCE_MM = 0.01
ANGLE_TOLERANCE_DEG = 0.01
CONFIDENCE_TOLERANCE = 1e-3


def _model_path():
    """The weights, without downloading anything for a test."""
    from wisepack_core.perception import resolve_model_path
    import model_store
    resolution = resolve_model_path(cache_dir=model_store.default_cache_dir())
    return resolution.path if resolution.available else None


def _requirements_or_skip():
    try:
        import cv2                                           # noqa: F401,PLC0415
        import torch                                         # noqa: F401,PLC0415
        import torchvision                                   # noqa: F401,PLC0415
    except ImportError as exc:
        pytest.skip(f"the perception environment is not active ({exc}); run "
                    "./scripts/setup_perception.sh and use "
                    ".venv-perception/bin/python")
    if not os.path.exists(REFERENCE_IMAGE):
        pytest.skip(f"the reference frame {REFERENCE_IMAGE} is missing")
    model = _model_path()
    if not model:
        pytest.skip("no detector weights are available on this host")
    return model


@pytest.fixture(scope="module")
def detection():
    """One real inference pass. Module-scoped: the model load costs ~30 s."""
    model = _requirements_or_skip()
    import cv2
    from perception_config import PerceptionConfig
    from providers import fasterrcnn_bottle as provider

    config = PerceptionConfig.from_env(model_path=model)
    detector = provider.build_detector(config)
    frame = cv2.imread(REFERENCE_IMAGE)
    assert frame is not None, f"{REFERENCE_IMAGE} could not be decoded"
    return detector.process_frame(frame.copy()), frame


def test_the_reference_frame_yields_the_recorded_object_count(detection):
    result, _frame = detection
    assert len(result["objects"]) == len(REFERENCE_OBJECTS), (
        "the ported detector reports a different number of objects than the "
        f"implementation it was adapted from: {result['objects']}")
    assert result["object_count"] == len(REFERENCE_OBJECTS)


def test_the_reference_frame_is_calibrated(detection):
    """All four markers are visible, so the plane must resolve for THIS frame."""
    result, _frame = detection
    assert result["calibration"]["status"] == "valid"
    assert result["calibration"]["markers_in_frame"] is True
    assert result["calibration"]["revision"], (
        "a valid calibration must carry a revision, so a recalibration is "
        "visible rather than silent")


@pytest.mark.parametrize("index", range(len(REFERENCE_OBJECTS)))
def test_each_measured_pose_matches_the_recorded_one(detection, index):
    result, _frame = detection
    measured = result["objects"][index]
    expected = REFERENCE_OBJECTS[index]
    assert measured["x"] == pytest.approx(expected["x"],
                                          abs=POSITION_TOLERANCE_MM)
    assert measured["y"] == pytest.approx(expected["y"],
                                          abs=POSITION_TOLERANCE_MM)
    assert measured["yaw"] == pytest.approx(expected["yaw"],
                                            abs=ANGLE_TOLERANCE_DEG)
    assert measured["conf"] == pytest.approx(expected["conf"],
                                             abs=CONFIDENCE_TOLERANCE)
    assert measured["selected"] is expected["selected"]


def test_exactly_one_object_is_selected_and_it_is_the_pick_pose(detection):
    result, _frame = detection
    selected = [o for o in result["objects"] if o["selected"]]
    assert len(selected) == 1
    assert result["pick_pose"]["x"] == pytest.approx(selected[0]["x"])
    assert result["pick_pose"]["y"] == pytest.approx(selected[0]["y"])
    assert result["pick_pose"]["rotation"] == pytest.approx(selected[0]["yaw"])


def test_an_object_without_a_cap_is_counted_but_not_reported(detection):
    """The reported count is "objects with a resolved orientation".

    The reference frame contains a third bottle whose cap the network does not
    match. Reporting it with yaw 0 would be a fabricated measurement; dropping it
    silently would make the count unexplainable. It is dropped AND counted.
    """
    result, _frame = detection
    assert result["objects_without_orientation"] >= 1
    assert result["caps_detected"] >= len(result["objects"])


def test_both_annotated_images_are_produced_and_differ_from_the_raw_frame(
        detection):
    """§14 asks for annotated-image generation, so it is verified, not assumed."""
    import numpy as np
    result, frame = detection
    for key in ("annotated_image", "detections_image"):
        image = result.get(key)
        assert image is not None, f"{key} was not produced"
        assert image.shape == frame.shape, f"{key} changed the frame geometry"
        assert not np.array_equal(image, frame), (
            f"{key} is identical to the raw frame — nothing was drawn")
    # The two views are different by construction: one carries the plane and the
    # measurements, the other only the network's boxes.
    assert not np.array_equal(result["annotated_image"],
                              result["detections_image"])


def test_the_annotated_image_encodes_as_jpeg(detection):
    """The service serves it over HTTP, so it has to survive `imencode`."""
    import cv2
    result, _frame = detection
    ok, buffer = cv2.imencode(".jpg", result["annotated_image"])
    assert ok and len(buffer.tobytes()) > 1024


def test_the_batch_built_from_the_real_result_is_domain_neutral(detection):
    """The whole point: measured physics in, generic observations out."""
    import json
    from providers import fasterrcnn_bottle as provider
    from wisepack_core.perception import BatchStatus, ProxyGeometry, WorkAreaFrame

    result, _frame = detection
    batch = provider.observations_from_detections(
        result, batch_id="regression-001",
        captured_at="2026-08-09T00:00:00.000Z",
        geometry=ProxyGeometry(),
        frame=WorkAreaFrame(width_mm=130, depth_mm=130),
        calibration_status=result["calibration"]["status"],
        calibration_revision=result["calibration"]["revision"])

    assert batch.status is BatchStatus.OK
    assert batch.count == len(REFERENCE_OBJECTS)
    assert batch.calibration_valid
    assert batch.observations[0].x_mm == pytest.approx(
        REFERENCE_OBJECTS[0]["x"], abs=POSITION_TOLERANCE_MM)
    assert all(o.object_type == "cylindrical_proxy" for o in batch.observations)

    # It must survive the JSON hop to the dashboard and to DDS — no numpy arrays,
    # no float32, no image bytes.
    document = json.dumps(batch.to_dict())
    assert "annotated_image" not in document
    assert len(document) < 20000


def test_the_real_detector_loaded_nothing_from_a_foreign_checkout(detection):
    """Runs AFTER a real inference pass, so it audits the loaded process.

    A text scan can miss a dynamic import; this cannot — the network has already
    run by the time it looks.
    """
    _result, _frame = detection
    foreign = sorted({getattr(module, "__file__", "") or ""
                      for module in sys.modules.values()}
                     - {""})
    offenders = [f for f in foreign if f.startswith("/data/arise/harmony")]
    assert offenders == [], (
        "the real detection pass loaded code from another project's checkout: "
        f"{offenders}")


def test_the_generated_calibration_sheet_measures_back_to_its_own_geometry(
        tmp_path):
    """WISEPACK prints the board AND measures it. The two must agree.

    A generator that placed markers at the wrong corners, mirrored the y axis or
    got the scale wrong would still produce a plausible-looking sheet, and every
    coordinate the system afterwards reported would be wrong by a fixed
    transform nobody could see. So the sheet is fed straight back through the
    calibration it is meant to satisfy.

    No camera and no model: this exercises geometry only.
    """
    try:
        import cv2                                           # noqa: PLC0415
    except ImportError as exc:                               # pragma: no cover
        pytest.skip(f"OpenCV is not available ({exc})")

    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from calibration import PlaneCalibration                 # noqa: PLC0415
    from generate_calibration_sheet import main              # noqa: PLC0415
    from perception_config import CalibrationBoard           # noqa: PLC0415

    sheet = tmp_path / "sheet.png"
    assert main(["--out", str(sheet)]) == 0
    image = cv2.imread(str(sheet))
    assert image is not None

    board = CalibrationBoard()
    calibration = PlaneCalibration(board)
    result = calibration.analyse(image)
    assert result.calibrated and result.seen_this_frame
    assert sorted(int(i) for i in result.ids.flatten()) == sorted(
        board.marker_ids)

    for expected, (pixel_x, pixel_y) in zip(board.corners_mm, result.markers):
        measured = calibration.to_plane(result.homography, pixel_x, pixel_y)
        assert measured[0] == pytest.approx(expected[0], abs=0.05)
        assert measured[1] == pytest.approx(expected[1], abs=0.05)

    # The middle of the printed square is the middle of the plane — which is
    # what catches a mirrored axis, the failure a corners-only check misses.
    height, width = image.shape[:2]
    centre = calibration.to_plane(result.homography, width / 2, height / 2)
    assert centre[0] == pytest.approx(65.0, abs=1.0)
    assert centre[1] == pytest.approx(65.0, abs=1.0)


def test_the_reference_frame_is_committed_and_small():
    """It is test DATA, not a model. Committed so this test needs nothing else."""
    path = pathlib.Path(REFERENCE_IMAGE)
    assert path.exists(), (
        "tests/data/perception/calibrated-scene.jpg is the whole reason this "
        "regression can run without a camera or another repository")
    assert path.stat().st_size < 1_000_000
