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
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from perception_config import CalibrationBoard, PerceptionConfigurationError

#: Where the saved calibration lives. A WISEPACK CONFIG FILE, following
#: HARMONY's approach of describing the work area in configuration rather than
#: rediscovering it every run — extended with the one thing HARMONY's template
#: does not hold, the COMPUTED HOMOGRAPHY, because that is what makes detection
#: work with the calibration sheet off the table.
CALIBRATION_FILE_ENV = "WISEPACK_PERCEPTION_CALIBRATION_FILE"
DEFAULT_CALIBRATION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "perception_calibration.json")


def calibration_file() -> str:
    return os.environ.get(CALIBRATION_FILE_ENV, DEFAULT_CALIBRATION_FILE)


class SavedCalibration:
    """A calibration read back from disk, and whether it may be used.

    WHY THE HOMOGRAPHY IS STORED AND NOT JUST THE BOARD. Recomputing it needs
    all four markers in the frame, which means the calibration sheet has to be
    on the table during every detection — including the ones where the table is
    covered in the objects being measured. Storing the computed matrix is what
    separates CALIBRATING from DETECTING: the sheet is a calibration reference,
    shown once, and normal detection never needs it again.

    WHAT IS STORED WITH IT, AND WHY EACH IS LOAD-BEARING:
      * `width`/`height` — a homography maps PIXELS. The same board at a
        different capture resolution gives a different matrix, so a saved one
        must be refused rather than silently misapplied.
      * the marker ids, corner coordinates and dictionary — these define WHICH
        work area was measured. A calibration for a 130 mm board must not be
        used after someone reconfigures a 200 mm one.
    """

    def __init__(self, homography: Any, width: int, height: int,
                 marker_ids: Tuple[int, ...],
                 corners_mm: Tuple[Tuple[float, float], ...],
                 dictionary: str, saved_at: str = "", revision: str = "") -> None:
        self.homography = homography
        self.width = int(width)
        self.height = int(height)
        self.marker_ids = tuple(int(m) for m in marker_ids)
        self.corners_mm = tuple(tuple(float(v) for v in c) for c in corners_mm)
        self.dictionary = str(dictionary)
        self.saved_at = saved_at
        self.revision = revision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "homography": [[float(v) for v in row] for row in self.homography],
            "width": self.width,
            "height": self.height,
            "marker_ids": list(self.marker_ids),
            "corners_mm": [list(c) for c in self.corners_mm],
            "dictionary": self.dictionary,
            "saved_at": self.saved_at,
            "revision": self.revision,
            "note": ("Computed by WISEPACK from the printed ArUco sheet. The "
                     "homography maps image pixels at the resolution above to "
                     "millimetres on the work area. Delete this file to force "
                     "recalibration."),
        }

    def usable_for(self, board: CalibrationBoard,
                   width: Optional[int] = None,
                   height: Optional[int] = None) -> Tuple[bool, str]:
        """(may this calibration be used, reason if not). NEVER raises."""
        rows = self.homography
        if rows is None or len(rows) != 3 or any(len(r) != 3 for r in rows):
            return False, "the saved homography is not a 3x3 matrix"
        flat = [float(v) for row in rows for v in row]
        if any(v != v or abs(v) == float("inf") for v in flat):
            return False, "the saved homography contains a non-finite value"
        if abs(_determinant_3x3(rows)) < 1e-12:
            # A degenerate matrix projects every pixel onto one line. It would
            # produce coordinates rather than an error, which is worse.
            return False, "the saved homography is degenerate (not invertible)"
        if tuple(self.marker_ids) != tuple(board.marker_ids):
            return False, (f"the saved calibration used markers "
                           f"{list(self.marker_ids)} but the configured board "
                           f"uses {list(board.marker_ids)}")
        configured = tuple(tuple(float(v) for v in c) for c in board.corners_mm)
        if self.corners_mm != configured:
            return False, ("the saved calibration measured a different work "
                           "area than the configured board")
        if self.dictionary != board.dictionary:
            return False, (f"the saved calibration used ArUco dictionary "
                           f"{self.dictionary} but the board is configured for "
                           f"{board.dictionary}")
        if width and height and (self.width, self.height) != (int(width), int(height)):
            return False, (f"the saved calibration was measured at "
                           f"{self.width}x{self.height} but frames are "
                           f"{int(width)}x{int(height)}; a homography maps "
                           "pixels, so it does not carry across a resolution "
                           "change")
        return True, ""

    @staticmethod
    def from_dict(document: Dict[str, Any]) -> "SavedCalibration":
        return SavedCalibration(
            homography=document["homography"],
            width=document["width"], height=document["height"],
            marker_ids=tuple(document.get("marker_ids") or ()),
            corners_mm=tuple(tuple(c) for c in document.get("corners_mm") or ()),
            dictionary=str(document.get("dictionary", "")),
            saved_at=str(document.get("saved_at", "")),
            revision=str(document.get("revision", "")))


def _determinant_3x3(m: Any) -> float:
    a, b, c = (float(v) for v in m[0])
    d, e, f = (float(v) for v in m[1])
    g, h, i = (float(v) for v in m[2])
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def load_calibration(path: str = "") -> Tuple[Optional[SavedCalibration], str]:
    """Read the saved calibration. ([None], reason) when there is none or it is
    unreadable — never an exception, because a missing calibration is an
    ordinary first-run state and a corrupt one must be reported, not crash the
    service."""
    resolved = path or calibration_file()
    if not os.path.isfile(resolved):
        return None, f"no saved calibration at {resolved}"
    try:
        with open(resolved, encoding="utf-8") as handle:
            return SavedCalibration.from_dict(json.load(handle)), ""
    except Exception as exc:                                 # noqa: BLE001
        return None, f"{resolved} could not be read: {exc}"


def save_calibration(calibration: SavedCalibration, path: str = "") -> str:
    """Persist it. Returns the path, or "" with the reason printed nowhere —
    saving is best-effort: a calibration that cannot be written is still usable
    for THIS session, and losing the session over a read-only config directory
    would be worse than losing the file."""
    resolved = path or calibration_file()
    try:
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        # Written via a temporary file and renamed, so a crash mid-write cannot
        # leave a half-written calibration that loads as garbage.
        temporary = f"{resolved}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(calibration.to_dict(), handle, indent=2)
        os.replace(temporary, resolved)
        return resolved
    except Exception:                                        # noqa: BLE001
        return ""

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

#: Where a resolved calibration came from.
CALIBRATION_SOURCE_MARKERS = "markers"
CALIBRATION_SOURCE_SAVED = "saved"


def _marker_revision(markers: Any) -> str:
    """A short digest of the marker positions a calibration was measured from.

    Makes a RECALIBRATION VISIBLE: moving the sheet changes every subsequent
    coordinate, and without this nothing in the audit trail records that the
    frame itself moved.
    """
    positions = getattr(markers, "tolist", lambda: markers)()
    return hashlib.sha256(repr(positions).encode()).hexdigest()[:12]


class CalibrationResult:
    """What one frame said about the plane.

    `homography` is None when the plane has never been resolved. It is NOT None
    merely because this frame hid a marker — see the cache note above.
    """

    __slots__ = ("corners", "ids", "markers", "homography", "seen_this_frame",
                 "source", "reason", "_revision")

    def __init__(self, corners: Any, ids: Any, markers: Any,
                 homography: Any, seen_this_frame: bool,
                 source: str = CALIBRATION_SOURCE_MARKERS,
                 reason: str = "", revision: str = "") -> None:
        self.corners = corners
        self.ids = ids
        self.markers = markers
        self.homography = homography
        self.seen_this_frame = seen_this_frame
        #: WHERE THIS CALIBRATION CAME FROM — markers in this frame, or the
        #: saved file. Reported so an operator can tell "measured just now" from
        #: "loaded from disk", which is the difference between a calibration
        #: that reflects the table as it is and one that reflects it as it was.
        self.source = source
        #: Why there is no calibration, when there is none.
        self.reason = reason
        self._revision = revision

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
        if self._revision:
            return self._revision
        if self.markers is None:
            return ""
        return _marker_revision(self.markers)


class PlaneCalibration:
    """Resolves the measured plane from the printed board. One per process.

    Stateful on purpose: the marker cache is the state, and it is what makes a
    partially occluded board usable.
    """

    def __init__(self, board: Optional[CalibrationBoard] = None,
                 store_path: str = "", load_saved: bool = True) -> None:
        import cv2                                           # noqa: PLC0415
        import numpy as np                                   # noqa: PLC0415

        self._cv2 = cv2
        self._np = np
        self.board = board or CalibrationBoard()
        self.store_path = store_path or calibration_file()

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

        # -- the SAVED calibration ----------------------------------------- #
        #
        # Loaded once, here, so a restart does not put the operator back to
        # "fetch the calibration sheet". This is the whole point of persisting
        # it: calibrating and detecting are separate activities, and only the
        # first one needs the board.
        self.saved: Optional[SavedCalibration] = None
        #: Why the saved calibration is unusable, when there is one and it is.
        self.saved_error: str = ""
        if load_saved:
            saved, reason = load_calibration(self.store_path)
            if saved is None:
                self.saved_error = reason
            else:
                usable, why = saved.usable_for(self.board)
                # REFUSED WITH THE REASON rather than repaired. A calibration
                # that describes a different board or a different resolution is
                # not a slightly-wrong calibration, it is a measurement of
                # something else.
                self.saved = saved if usable else None
                self.saved_error = "" if usable else why

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
        """Resolve the plane for one frame.

        THE ORDER MATTERS AND IS THE BEHAVIOUR THIS FILE EXISTS TO PROVIDE:

          1. markers visible in THIS frame  -> compute and SAVE. Showing the
             sheet again is how an operator recalibrates; it always wins,
             because it is the only input that is a fresh measurement.
          2. otherwise, a usable SAVED calibration -> use it. No markers are
             required, which is what lets "Detect & plan" run over a table
             covered in the objects being measured.
          3. otherwise, the in-memory marker cache from earlier in this session.
          4. otherwise, UNCALIBRATED, with a reason a human can act on.
        """
        corners, ids = self.detect_markers(frame)
        markers, seen = self._plane_markers(corners, ids)

        if seen and markers is not None:
            homography, _mask = self._cv2.findHomography(markers,
                                                         self._plane_points)
            if homography is not None:
                self._persist(homography, frame, markers)
                return CalibrationResult(corners, ids, markers, homography, True)

        if self.saved is not None:
            height, width = frame.shape[:2]
            usable, why = self.saved.usable_for(self.board, width, height)
            if usable:
                return CalibrationResult(
                    corners, ids, markers,
                    self._np.array(self.saved.homography, dtype=self._np.float64),
                    False, source=CALIBRATION_SOURCE_SAVED,
                    revision=self.saved.revision)
            # A saved calibration that does not fit THIS frame is dropped once
            # and the reason kept, so the next frame does not re-test it.
            self.saved, self.saved_error = None, why

        homography = None
        if markers is not None:
            homography, _mask = self._cv2.findHomography(markers,
                                                         self._plane_points)
        return CalibrationResult(corners, ids, markers, homography, seen,
                                 reason="" if homography is not None
                                 else self.not_calibrated_reason())

    def not_calibrated_reason(self) -> str:
        """Why there is no calibration, in terms an operator can act on."""
        detail = f" ({self.saved_error})" if self.saved_error else ""
        return (
            "the camera is not calibrated: no saved calibration is available"
            f"{detail}, and the calibration markers "
            f"{list(self.board.marker_ids)} are not all visible in this frame. "
            "Place the calibration sheet in view once and detect again — it is "
            "then saved and detection no longer needs it.")

    def _persist(self, homography: Any, frame: Any, markers: Any) -> None:
        """Save a freshly measured calibration, replacing any earlier one."""
        height, width = frame.shape[:2]
        saved = SavedCalibration(
            homography=[[float(v) for v in row] for row in homography],
            width=int(width), height=int(height),
            marker_ids=self.board.marker_ids,
            corners_mm=self.board.corners_mm,
            dictionary=self.board.dictionary,
            saved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            revision=_marker_revision(markers))
        self.saved = saved
        self.saved_error = ""
        self.saved_path = save_calibration(saved, self.store_path)

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
