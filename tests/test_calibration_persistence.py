"""Saved planar calibration — detect without the calibration sheet.

THE BEHAVIOUR THIS PINS. Calibrating and detecting are separate activities. The
printed ArUco sheet is a CALIBRATION REFERENCE: it is shown once, the resulting
homography is saved, and every detection afterwards uses the saved one — over a
table covered in the objects being measured, and across restarts.

Before this, the homography was recomputed from markers on every frame and cached
only in memory, so the sheet had to stay in view during normal detection and a
restart put the operator back to fetching it.

The four states, and all four are tested here:

    saved calibration, no markers  -> metric detection, no sheet needed
    no saved calibration, markers  -> calibrate, save, then detect
    no saved calibration, no markers -> a clear "camera not calibrated" error
    restart                        -> the saved calibration is loaded and works

NO CAMERA AND NO DETECTOR. The plane is geometry; these drive `PlaneCalibration`
with synthesised frames and a stub ArUco detector, so they run in ordinary CI.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "perception"))

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from calibration import (                                        # noqa: E402
    CALIBRATION_SOURCE_MARKERS, CALIBRATION_SOURCE_SAVED,
    UNCALIBRATED_SENTINEL, PlaneCalibration, SavedCalibration,
    load_calibration, save_calibration)
from perception_config import CalibrationBoard                   # noqa: E402

#: Where the four markers sit in the synthetic frame, in board order. Chosen so
#: the homography is well conditioned and the arithmetic is checkable by hand:
#: 100 px maps to 130 mm.
MARKER_PIXELS = ((100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0))
FRAME_SIZE = (640, 480)          # (width, height)


def _frame(width=FRAME_SIZE[0], height=FRAME_SIZE[1]):
    return np.zeros((height, width, 3), dtype=np.uint8)


class _StubDetector:
    """Stands in for `cv2.aruco.ArucoDetector`.

    The real detector needs a printed sheet in a real image. What these tests
    are about is what happens WHEN MARKERS ARE OR ARE NOT SEEN, so that is the
    input: a switch, not a rendering pipeline.
    """

    def __init__(self, marker_ids, visible=True, pixels=None):
        self.marker_ids = list(marker_ids)
        self.visible = visible
        #: WHERE THE SHEET IS. A parameter rather than a module global, so a
        #: test that moves the sheet says so locally instead of mutating state
        #: every other test shares.
        self.pixels = tuple(pixels or MARKER_PIXELS)
        self.calls = 0

    def detectMarkers(self, _gray):                          # noqa: N802
        self.calls += 1
        if not self.visible:
            return (), None, ()
        corners = []
        for centre in self.pixels:
            x, y = centre
            corners.append(np.array(
                [[[x - 5, y - 5], [x + 5, y - 5], [x + 5, y + 5], [x - 5, y + 5]]],
                dtype=np.float32))
        ids = np.array([[m] for m in self.marker_ids], dtype=np.int32)
        return corners, ids, ()


def _plane(tmp_path, markers_visible=True, board=None, load_saved=True):
    board = board or CalibrationBoard()
    plane = PlaneCalibration(board=board,
                             store_path=str(tmp_path / "perception_calibration.json"),
                             load_saved=load_saved)
    plane._detector = _StubDetector(board.marker_ids, visible=markers_visible)
    return plane


# --------------------------------------------------------------------------- #
# 1. Saved calibration + no markers -> successful metric detection
# --------------------------------------------------------------------------- #


def test_a_saved_calibration_measures_without_any_markers(tmp_path):
    """THE HEADLINE. "Detect & plan" must not require the calibration sheet."""
    # Calibrate once, with the sheet in view.
    calibrating = _plane(tmp_path, markers_visible=True)
    first = calibrating.analyse(_frame())
    assert first.calibrated and first.seen_this_frame

    # A fresh session, sheet off the table.
    detecting = _plane(tmp_path, markers_visible=False)
    result = detecting.analyse(_frame())

    assert result.calibrated, "a saved calibration must not need markers"
    assert result.seen_this_frame is False
    assert result.source == CALIBRATION_SOURCE_SAVED
    # AND IT ACTUALLY MEASURES. "Calibrated" that yields the sentinel would be
    # a status with no measurement behind it.
    x_mm, y_mm = detecting.to_plane(result.homography, *MARKER_PIXELS[0])
    assert (x_mm, y_mm) != UNCALIBRATED_SENTINEL
    assert x_mm == pytest.approx(0.0, abs=1e-6)
    assert y_mm == pytest.approx(0.0, abs=1e-6)


def test_the_saved_calibration_reproduces_the_measured_one(tmp_path):
    """The point of storing the matrix: the same pixel gives the same
    millimetre, whether it was measured now or loaded from disk."""
    calibrating = _plane(tmp_path, markers_visible=True)
    measured = calibrating.analyse(_frame())
    detecting = _plane(tmp_path, markers_visible=False)
    loaded = detecting.analyse(_frame())

    for pixel in ((300.0, 300.0), (150.0, 480.0), (500.0, 100.0)):
        a = calibrating.to_plane(measured.homography, *pixel)
        b = detecting.to_plane(loaded.homography, *pixel)
        assert a[0] == pytest.approx(b[0], abs=1e-6)
        assert a[1] == pytest.approx(b[1], abs=1e-6)


# --------------------------------------------------------------------------- #
# 2. No calibration + markers -> calibrates, saves, then detects
# --------------------------------------------------------------------------- #


def test_markers_calibrate_and_persist_the_result(tmp_path):
    store = tmp_path / "perception_calibration.json"
    assert not store.exists()

    plane = _plane(tmp_path, markers_visible=True)
    result = plane.analyse(_frame())

    assert result.calibrated
    assert result.source == CALIBRATION_SOURCE_MARKERS
    assert store.exists(), "a measured calibration must be saved"

    document = json.loads(store.read_text())
    # §3: everything needed to operate later WITHOUT markers.
    assert len(document["homography"]) == 3
    assert all(len(row) == 3 for row in document["homography"])
    assert (document["width"], document["height"]) == FRAME_SIZE
    assert document["marker_ids"] == list(CalibrationBoard().marker_ids)
    assert len(document["corners_mm"]) == 4
    assert document["dictionary"] == CalibrationBoard().dictionary
    assert document["saved_at"]


def test_showing_the_sheet_again_recalibrates_and_replaces_the_saved_one(tmp_path):
    """§4. A fresh measurement always wins: it is the only input that reflects
    where the sheet is NOW."""
    plane = _plane(tmp_path, markers_visible=True)
    plane.analyse(_frame())
    first = json.loads((tmp_path / "perception_calibration.json").read_text())

    # The sheet moves: same marker ids, different pixels.
    plane._detector = _StubDetector(
        plane.board.marker_ids, visible=True,
        pixels=((120.0, 110.0), (520.0, 100.0), (515.0, 505.0), (105.0, 495.0)))
    second_result = plane.analyse(_frame())
    second = json.loads((tmp_path / "perception_calibration.json").read_text())

    assert second["homography"] != first["homography"]
    assert second_result.source == CALIBRATION_SOURCE_MARKERS
    # THE CHANGE IS VISIBLE IN THE AUDIT TRAIL. Without this, moving the sheet
    # silently changes every subsequent coordinate.
    assert second["revision"] != first["revision"]


# --------------------------------------------------------------------------- #
# 3. No calibration + no markers -> a clear error
# --------------------------------------------------------------------------- #


def test_no_calibration_and_no_markers_is_a_clear_error(tmp_path):
    plane = _plane(tmp_path, markers_visible=False)
    result = plane.analyse(_frame())

    assert not result.calibrated
    assert result.status == "invalid"
    # NAMED, NOT INFERRED. "invalid" alone leaves an operator guessing between a
    # missing sheet, a missing file and a resolution change.
    assert "not calibrated" in result.reason
    assert "calibration sheet" in result.reason
    # And it yields no measurement at all, rather than a plausible coordinate.
    assert plane.to_plane(result.homography, 300.0, 300.0) == UNCALIBRATED_SENTINEL


def test_the_error_survives_to_the_batch_reason():
    """The provider must report the detector's own reason, not a generic one."""
    from providers.fasterrcnn_bottle import observations_from_detections
    x, y = UNCALIBRATED_SENTINEL
    # THE SHAPE THE DETECTOR ACTUALLY PRODUCES — `process_frame`'s own mapping,
    # which is what the service passes through.
    batch = observations_from_detections(
        {"objects": [{"x": x, "y": y, "yaw": 0.0, "conf": 0.9}],
         "calibration": {"status": "invalid", "revision": "",
                         "markers_in_frame": False,
                         "reason": "the camera is not calibrated: no saved "
                                   "calibration is available"}},
        batch_id="b1")
    assert not batch.ok
    assert "not calibrated" in batch.error


# --------------------------------------------------------------------------- #
# 4. Restart -> the saved calibration is loaded and works
# --------------------------------------------------------------------------- #


def test_a_restart_loads_the_saved_calibration(tmp_path):
    """A new process, a new object, no markers — and still metric."""
    _plane(tmp_path, markers_visible=True).analyse(_frame())

    restarted = _plane(tmp_path, markers_visible=False)
    assert restarted.saved is not None, "the saved calibration was not loaded"
    assert restarted.saved_error == ""
    result = restarted.analyse(_frame())
    assert result.calibrated and result.source == CALIBRATION_SOURCE_SAVED
    assert restarted.to_plane(result.homography, 500.0, 500.0)[0] == pytest.approx(
        130.0, abs=1e-6)


def test_a_restart_with_no_saved_file_reports_that_and_not_a_crash(tmp_path):
    plane = _plane(tmp_path, markers_visible=False)
    assert plane.saved is None
    assert "no saved calibration" in plane.saved_error


# --------------------------------------------------------------------------- #
# A saved calibration is refused when it describes something else
# --------------------------------------------------------------------------- #


def test_a_calibration_from_another_resolution_is_refused(tmp_path):
    """A homography maps PIXELS. Reusing one across a resolution change would
    return coordinates that are wrong by a scale factor and look fine."""
    _plane(tmp_path, markers_visible=True).analyse(_frame())
    plane = _plane(tmp_path, markers_visible=False)
    result = plane.analyse(_frame(width=1280, height=960))
    assert not result.calibrated
    assert "resolution" in result.reason or "1280x960" in result.reason


def test_a_calibration_for_another_board_is_refused(tmp_path):
    """It measured a different work area, which is not a slightly-wrong
    calibration — it is a measurement of something else."""
    _plane(tmp_path, markers_visible=True).analyse(_frame())
    other = CalibrationBoard.square(200.0)
    plane = _plane(tmp_path, markers_visible=False, board=other)
    assert plane.saved is None
    assert "work area" in plane.saved_error or "markers" in plane.saved_error


def test_a_degenerate_saved_homography_is_refused(tmp_path):
    """A singular matrix projects every pixel onto one line: it would produce
    coordinates rather than an error, which is worse than failing."""
    store = tmp_path / "perception_calibration.json"
    board = CalibrationBoard()
    save_calibration(SavedCalibration(
        homography=[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]],
        width=FRAME_SIZE[0], height=FRAME_SIZE[1],
        marker_ids=board.marker_ids, corners_mm=board.corners_mm,
        dictionary=board.dictionary), str(store))
    plane = _plane(tmp_path, markers_visible=False)
    assert plane.saved is None
    assert "degenerate" in plane.saved_error


def test_a_corrupt_calibration_file_is_reported_not_raised(tmp_path):
    store = tmp_path / "perception_calibration.json"
    store.write_text("{not json")
    saved, reason = load_calibration(str(store))
    assert saved is None
    assert "could not be read" in reason


def test_saving_is_atomic_and_leaves_no_partial_file(tmp_path):
    """A crash mid-write must not leave a half-written calibration that loads as
    garbage."""
    store = tmp_path / "nested" / "perception_calibration.json"
    board = CalibrationBoard()
    path = save_calibration(SavedCalibration(
        homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        width=640, height=480, marker_ids=board.marker_ids,
        corners_mm=board.corners_mm, dictionary=board.dictionary), str(store))
    assert path and os.path.isfile(path)
    assert not os.path.exists(f"{store}.tmp")
    assert load_calibration(str(store))[0] is not None


# --------------------------------------------------------------------------- #
# The sheet is a calibration reference, not part of detection
# --------------------------------------------------------------------------- #


def test_detection_does_not_depend_on_the_detector_seeing_markers(tmp_path):
    """§6. Once saved, the marker detector's answer changes nothing about the
    measurement."""
    _plane(tmp_path, markers_visible=True).analyse(_frame())
    plane = _plane(tmp_path, markers_visible=False)
    without = plane.analyse(_frame())

    plane._detector.visible = True
    with_sheet = plane.analyse(_frame())

    for pixel in ((300.0, 300.0), (200.0, 400.0)):
        a = plane.to_plane(without.homography, *pixel)
        b = plane.to_plane(with_sheet.homography, *pixel)
        assert a[0] == pytest.approx(b[0], abs=1e-6)
        assert a[1] == pytest.approx(b[1], abs=1e-6)


def test_the_faster_rcnn_detector_is_untouched():
    """§7. This change is about geometry; the network is not involved."""
    source = open(os.path.join(REPO, "perception", "providers",
                               "fasterrcnn_bottle.py"), encoding="utf-8").read()
    # The detection pipeline still builds the same model and reads the same
    # class names; only the calibration STATUS block gained fields.
    assert "fasterrcnn_resnet50_fpn" in source
    assert "CLASS_NAMES" in source
