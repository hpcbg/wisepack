"""ArUco plane calibration — image pixels to millimetres on the work surface.

ADAPTED FROM HARMONY (`ai-bottle-detector-fiware/pipeline.py`, MIT): the marker
detection, the ordered corner correspondence, the homography, the marker cache
and the overlay drawing are the ones that project was validated with, extracted
here into a WISEPACK module with an explicit interface. The behaviour is
deliberately unchanged — including the two quirks documented below — because a
"cleaned up" calibration is a DIFFERENT calibration, and the whole point of
porting rather than rewriting is that the measured coordinates stay the same.

    frame ──> detect markers ──> order by board.marker_ids ──> homography H
                                                                    │
                              pixel (u, v) ──> H ──> (x_mm, y_mm) ──┘

TWO BEHAVIOURS THAT ARE PART OF THE CONTRACT, NOT ACCIDENTS
-----------------------------------------------------------
  * THE MARKER POSITIONS ARE CACHED AND PERSIST. Once all four corner markers
    have been seen, the plane stays calibrated even in frames where a bottle
    covers one of them — otherwise a detection would fail precisely when the
    table is busy. A later frame that shows all four REPLACES the cache. This is
    why `calibrated` can be true for a frame with no markers visible at all.

  * AN UNCALIBRATED FRAME YIELDS THE SENTINEL (1, 1). With no homography there
    is no measurement, and the sentinel is what the downstream provider detects
    to reject the batch as a calibration failure rather than plan from a pile of
    objects all at one point. It is NOT a position, and nothing may treat it as
    one.

This module imports OpenCV and numpy but no detector: it is about geometry, and
a future provider that finds objects some other way calibrates the same plane
the same way.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Tuple

from perception_config import CalibrationBoard, PerceptionConfigurationError

#: What `to_plane()` returns when there is no homography. Reproduced from the
#: detector's original behaviour so the downstream sentinel check keeps working
#: against the ported implementation exactly as it did against the original.
UNCALIBRATED_SENTINEL: Tuple[float, float] = (1.0, 1.0)

#: Calibration states, as reported on every observation and batch.
CALIBRATION_VALID = "valid"
CALIBRATION_INVALID = "invalid"

#: Overlay colours, BGR. Kept identical to the original so the annotated image an
#: operator has learned to read does not change meaning.
MARKER_OUTLINE = (0, 255, 255)
MARKER_CENTRE = (0, 0, 255)
MARKER_LABEL = (0, 150, 255)
PLANE_OUTLINE = (0, 255, 0)


class CalibrationResult:
    """What one frame said about the plane.

    `homography` is None when the plane has never been resolved. It is NOT None
    merely because this frame hid a marker — see the cache note above.
    """

    __slots__ = ("corners", "ids", "markers", "homography", "seen_this_frame")

    def __init__(self, corners: Any, ids: Any, markers: Any,
                 homography: Any, seen_this_frame: bool) -> None:
        self.corners = corners
        self.ids = ids
        self.markers = markers
        self.homography = homography
        self.seen_this_frame = seen_this_frame

    @property
    def calibrated(self) -> bool:
        return self.homography is not None

    @property
    def status(self) -> str:
        return CALIBRATION_VALID if self.calibrated else CALIBRATION_INVALID

    @property
    def revision(self) -> str:
        """A short digest of the cached marker positions, or "".

        Makes a RECALIBRATION VISIBLE. Without it, moving the sheet changes every
        subsequent coordinate and nothing in the audit trail records that the
        frame itself moved.
        """
        if self.markers is None:
            return ""
        positions = getattr(self.markers, "tolist", lambda: self.markers)()
        return hashlib.sha256(repr(positions).encode()).hexdigest()[:12]


class PlaneCalibration:
    """Resolves the measured plane from the printed board. One per process.

    Stateful on purpose: the marker cache is the state, and it is what makes a
    partially occluded board usable.
    """

    def __init__(self, board: Optional[CalibrationBoard] = None) -> None:
        import cv2                                           # noqa: PLC0415
        import numpy as np                                   # noqa: PLC0415

        self._cv2 = cv2
        self._np = np
        self.board = board or CalibrationBoard()

        dictionary_id = getattr(cv2.aruco, self.board.dictionary, None)
        if dictionary_id is None:
            raise PerceptionConfigurationError(
                f"unknown ArUco dictionary {self.board.dictionary!r}; see "
                "cv2.aruco.DICT_* for the available names")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._detector = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters())

        self._plane_points = np.array(self.board.corners_mm, dtype=np.float32)
        #: The cached, ordered corner-marker centres. None until all four have
        #: been seen together in one frame.
        self._markers = None

    # -- detection --------------------------------------------------------- #

    def detect_markers(self, frame) -> Tuple[Any, Any]:
        """Every ArUco marker in the frame — the board's and anyone else's."""
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = self._detector.detectMarkers(gray)
        return corners, ids

    def _plane_markers(self, corners: Any, ids: Any) -> Tuple[Any, bool]:
        """The four corner centres in board order, and whether THIS frame had them.

        Returns the cache when the frame does not show all four. Ordered by
        `board.marker_ids`, never sorted: the order is what pairs each marker
        with its millimetre coordinate.
        """
        if ids is None:
            return self._markers, False

        flat = ids.flatten()
        wanted = self.board.marker_ids
        found = {}
        for corner, marker_id in zip(corners, flat):
            if marker_id in wanted:
                found[int(marker_id)] = self._np.mean(corner[0], axis=0)

        if len(found) != len(wanted):
            return self._markers, False

        ordered = self._np.array([found[int(m)] for m in wanted],
                                 dtype=self._np.float32)
        self._markers = ordered
        return ordered, True

    def analyse(self, frame) -> CalibrationResult:
        """Detect the board in one frame and resolve the plane."""
        corners, ids = self.detect_markers(frame)
        markers, seen = self._plane_markers(corners, ids)
        homography = None
        if markers is not None:
            homography, _mask = self._cv2.findHomography(markers,
                                                         self._plane_points)
        return CalibrationResult(corners, ids, markers, homography, seen)

    # -- measurement ------------------------------------------------------- #

    def to_plane(self, homography: Any, x: float, y: float) -> Tuple[float, float]:
        """Pixel (x, y) -> millimetres on the calibrated plane.

        Returns `UNCALIBRATED_SENTINEL` when there is no homography, or when the
        projection is degenerate (a point on the horizon of the plane divides by
        a vanishing w). Both are "no measurement", and both must be reported as
        such rather than as a coordinate.
        """
        if homography is None:
            return UNCALIBRATED_SENTINEL
        try:
            pixel = self._np.array([[x, y, 1]], dtype=self._np.float32).T
            point = homography @ pixel
            point = point / point[2]
            measured = (float(point[0][0]), float(point[1][0]))
        except Exception:                                    # noqa: BLE001
            return UNCALIBRATED_SENTINEL
        if any(v != v or abs(v) == float("inf") for v in measured):
            return UNCALIBRATED_SENTINEL
        return measured

    # -- overlays ---------------------------------------------------------- #

    def annotate(self, image, result: CalibrationResult) -> None:
        """Draw the markers and the calibrated plane onto `image`, in place."""
        cv2 = self._cv2
        np = self._np

        if result.ids is not None:
            for corner, marker_id in zip(result.corners, result.ids.flatten()):
                points = corner[0].astype(int)
                cv2.polylines(image, [points], True, MARKER_OUTLINE, 2)
                centre = tuple(np.mean(points, axis=0).astype(int))
                cv2.circle(image, centre, 5, MARKER_CENTRE, -1)
                cv2.putText(image, f"{marker_id}",
                            (points[0][0], points[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, MARKER_LABEL, 2)

        if result.homography is None:
            return
        try:
            plane = np.float32(self._plane_points).reshape(-1, 1, 2)
            outline = cv2.perspectiveTransform(plane,
                                               np.linalg.inv(result.homography))
            cv2.polylines(image, [np.int32(outline)], True, PLANE_OUTLINE, 2)
        except Exception:                                    # noqa: BLE001
            # A singular homography cannot be inverted. The overlay is a
            # convenience; losing it must not lose the detection.
            pass


__all__ = [
    "PlaneCalibration", "CalibrationResult", "UNCALIBRATED_SENTINEL",
    "CALIBRATION_VALID", "CALIBRATION_INVALID",
]
