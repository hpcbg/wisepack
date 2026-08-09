"""WISEPACK perception configuration — WISEPACK's own, and nobody else's.

THE POINT OF THIS FILE. Perception used to be configured by generating another
project's `config.json` in a cache directory and importing that project's modules
with the cache as the working directory. That made a foreign checkout a RUNTIME
DEPENDENCY of a WISEPACK feature: delete it and the camera stopped working.

WISEPACK now owns its perception runtime end to end, so it owns the settings too.
Everything the perception service and its providers need is described here, read
from `WISEPACK_PERCEPTION_*` environment variables, and passed as ordinary Python
objects. No file is generated, no working directory is changed, no external
template is read.

    PerceptionConfig
      ├── camera / width / height / set_resolution   -> perception/camera.py
      ├── board (CalibrationBoard)                   -> perception/calibration.py
      ├── model_path / confidence_threshold          -> perception/providers/*
      └── work_area()                                -> wisepack_core.perception

Deliberately importable WITHOUT torch, OpenCV or a camera: the launcher, the
tests and `--check` all read the resolved configuration, and none of them should
have to load a neural network to find out which device index is configured.

PROVENANCE. The calibration board's defaults — `DICT_ARUCO_ORIGINAL`, marker ids
11/10/15/16, a 130 mm square with the origin at marker 11 — are the geometry of
the printed sheet the Faster R-CNN bottle detector was developed and validated
against in the HARMONY project. They are reproduced here as WISEPACK defaults so
the measured coordinate frame is unchanged; see `scripts/generate_calibration_sheet.py`
for the WISEPACK generator that prints a matching sheet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Calibration board
# --------------------------------------------------------------------------- #

#: The ArUco dictionary the calibration sheet is printed from. A NAME, resolved
#: against `cv2.aruco` at use time, so this module stays importable without
#: OpenCV and an unknown name fails with "unknown dictionary" rather than with an
#: AttributeError from inside a detector.
DEFAULT_ARUCO_DICTIONARY = "DICT_ARUCO_ORIGINAL"

#: The four corner markers, IN PLANE ORDER: the first sits at the plane origin
#: and the rest walk the square. Order is what pairs a marker with a coordinate,
#: so it is part of the configuration and never sorted.
DEFAULT_CORNER_MARKERS: Tuple[int, ...] = (11, 10, 15, 16)

#: The side of the calibrated square in millimetres, matching the printed A4
#: sheet. Small for a realistic work area — which is exactly why it is
#: configurable (see `WISEPACK_PERCEPTION_CALIBRATION_EXTENT_MM`).
DEFAULT_CORNER_EXTENT_MM = 130.0


@dataclass(frozen=True)
class CalibrationBoard:
    """The printed ArUco sheet that defines the measured plane.

    `marker_ids[i]` is physically at `corners_mm[i]`. The homography is computed
    from those correspondences, so the two sequences must be the same length and
    are always read together.
    """

    marker_ids: Tuple[int, ...] = DEFAULT_CORNER_MARKERS
    corners_mm: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0), (DEFAULT_CORNER_EXTENT_MM, 0.0),
        (DEFAULT_CORNER_EXTENT_MM, DEFAULT_CORNER_EXTENT_MM),
        (0.0, DEFAULT_CORNER_EXTENT_MM))
    dictionary: str = DEFAULT_ARUCO_DICTIONARY

    def __post_init__(self) -> None:
        if len(self.marker_ids) != len(self.corners_mm):
            raise PerceptionConfigurationError(
                f"the calibration board declares {len(self.marker_ids)} marker "
                f"ids and {len(self.corners_mm)} corner coordinates; each "
                "marker needs exactly one coordinate")
        if len(self.marker_ids) != 4:
            raise PerceptionConfigurationError(
                "a homography needs exactly four corner markers, got "
                f"{len(self.marker_ids)}")
        if len(set(self.marker_ids)) != len(self.marker_ids):
            raise PerceptionConfigurationError(
                f"duplicate calibration marker ids: {list(self.marker_ids)}")

    @property
    def extent_mm(self) -> Tuple[float, float]:
        """(width, depth) of the declared plane, in millimetres."""
        xs = [float(c[0]) for c in self.corners_mm]
        ys = [float(c[1]) for c in self.corners_mm]
        return max(xs) - min(xs), max(ys) - min(ys)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marker_ids": list(self.marker_ids),
            "corners_mm": [list(c) for c in self.corners_mm],
            "dictionary": self.dictionary,
        }

    @staticmethod
    def square(extent_mm: float,
               marker_ids: Sequence[int] = DEFAULT_CORNER_MARKERS,
               dictionary: str = DEFAULT_ARUCO_DICTIONARY) -> "CalibrationBoard":
        """The common case: a square board with the origin at the first marker."""
        side = float(extent_mm)
        return CalibrationBoard(
            marker_ids=tuple(int(m) for m in marker_ids),
            corners_mm=((0.0, 0.0), (side, 0.0), (side, side), (0.0, side)),
            dictionary=dictionary)


class PerceptionConfigurationError(ValueError):
    """Raised when the perception configuration cannot be honoured.

    Distinct from `wisepack_core.perception.PerceptionConfigError`, which is the
    DOMAIN's error for an unusable perception SOURCE. This one belongs to the
    host-side service and never reaches the container.
    """


# --------------------------------------------------------------------------- #
# The service configuration
# --------------------------------------------------------------------------- #

#: Confidence below which a detection is discarded. The value the current
#: provider's model was tuned and demonstrated with; changing it changes what the
#: detector reports, so it is configuration rather than a constant in a provider.
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

#: Capture defaults. A resolution IS set by default because the calibration
#: markers are small in frame and a 640×480 default from the driver loses them.
DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 1200


@dataclass(frozen=True)
class PerceptionConfig:
    """Everything the WISEPACK perception service needs, resolved once.

    `camera` is deliberately `Any`: OpenCV accepts an index, a device path and a
    stream URL, and each is legitimate. See `camera_setting()` for why the
    distinction between `2` and `"2"` matters.
    """

    camera: Any = 0
    set_resolution: bool = True
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    model_path: str = ""
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    board: CalibrationBoard = field(default_factory=CalibrationBoard)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera": self.camera,
            "set_resolution": self.set_resolution,
            "width": self.width,
            "height": self.height,
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "calibration_board": self.board.to_dict(),
        }

    def work_area(self):
        """The WISEPACK work-area frame implied by the calibrated plane.

        Read back FROM the board rather than declared twice, so the frame the
        dashboard shows and the plane the detector measures on cannot drift
        apart. The frame id and any explicitly declared extent still come from
        `WorkAreaFrame.from_env()`; only the size defaults to the board's.
        """
        from wisepack_core.perception import WorkAreaFrame   # noqa: PLC0415

        width, depth = self.board.extent_mm
        base = WorkAreaFrame.from_env()
        environ = os.environ
        declared_width = str(
            environ.get("WISEPACK_PHYSICAL_WORKAREA_WIDTH_MM", "") or "").strip()
        declared_depth = str(
            environ.get("WISEPACK_PHYSICAL_WORKAREA_DEPTH_MM", "") or "").strip()
        return WorkAreaFrame(
            frame_id=base.frame_id,
            width_mm=(base.width_mm if declared_width
                      else max(1, int(round(width)))),
            depth_mm=(base.depth_mm if declared_depth
                      else max(1, int(round(depth)))),
            origin_x_mm=base.origin_x_mm,
            origin_y_mm=base.origin_y_mm)

    @staticmethod
    def from_env(model_path: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None) -> "PerceptionConfig":
        """Resolve the configuration from `WISEPACK_PERCEPTION_*`.

        NO CAMERA DEVICE IS HARDCODED beyond the documented default: which camera
        is plugged in is a property of the host and is decided in exactly one
        place, `WISEPACK_PERCEPTION_CAMERA`.
        """
        env = os.environ if env is None else env

        camera_raw = str(env.get("WISEPACK_PERCEPTION_CAMERA", "") or "").strip()
        camera = camera_setting(camera_raw) if camera_raw else 0

        return PerceptionConfig(
            camera=camera,
            set_resolution=_flag(env, "WISEPACK_PERCEPTION_SET_RESOLUTION", True),
            width=_positive_int(env, "WISEPACK_PERCEPTION_WIDTH", DEFAULT_WIDTH),
            height=_positive_int(env, "WISEPACK_PERCEPTION_HEIGHT", DEFAULT_HEIGHT),
            model_path=str(model_path or ""),
            confidence_threshold=_confidence(
                env, "WISEPACK_PERCEPTION_CONFIDENCE",
                DEFAULT_CONFIDENCE_THRESHOLD),
            board=board_from_env(env))


def camera_setting(raw: Any) -> Any:
    """A camera is an INDEX or a URL. `2` and `rtsp://…` both reach cv2 as-is.

    `/dev/video2` and an RTSP URL are strings; a bare number must become an int,
    because `cv2.VideoCapture("2")` opens a *file* called "2" and then reports a
    perfectly ordinary "no frame" that looks exactly like an unplugged camera.
    """
    text = str(raw).strip()
    try:
        return int(text)
    except ValueError:
        return text


def board_from_env(env: Optional[Dict[str, str]] = None) -> CalibrationBoard:
    """The calibration board for this host.

    A LARGER PRINTED BOARD IS A CONFIGURATION CHANGE, never a code change and
    never a silently redesigned layout: set the marker ids and the extent, print
    a matching sheet with `scripts/generate_calibration_sheet.py`, and the
    measured plane follows.
    """
    env = os.environ if env is None else env
    markers: Tuple[int, ...] = DEFAULT_CORNER_MARKERS
    raw_markers = str(
        env.get("WISEPACK_PERCEPTION_CALIBRATION_MARKERS", "") or "").strip()
    if raw_markers:
        parsed: List[int] = []
        for piece in raw_markers.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                parsed.append(int(piece))
            except ValueError as exc:
                raise PerceptionConfigurationError(
                    "WISEPACK_PERCEPTION_CALIBRATION_MARKERS must be a "
                    f"comma-separated list of marker ids, got {raw_markers!r}"
                ) from exc
        markers = tuple(parsed)

    extent = DEFAULT_CORNER_EXTENT_MM
    raw_extent = str(
        env.get("WISEPACK_PERCEPTION_CALIBRATION_EXTENT_MM", "") or "").strip()
    if raw_extent:
        try:
            extent = float(raw_extent)
        except ValueError as exc:
            raise PerceptionConfigurationError(
                "WISEPACK_PERCEPTION_CALIBRATION_EXTENT_MM must be a number of "
                f"millimetres, got {raw_extent!r}") from exc
        if extent <= 0:
            raise PerceptionConfigurationError(
                "WISEPACK_PERCEPTION_CALIBRATION_EXTENT_MM must be positive, "
                f"got {raw_extent!r}")

    dictionary = str(env.get("WISEPACK_PERCEPTION_CALIBRATION_DICTIONARY", "")
                     or DEFAULT_ARUCO_DICTIONARY).strip()
    return CalibrationBoard.square(extent, markers, dictionary)


# --------------------------------------------------------------------------- #
# Small typed readers. Every one of them REPORTS a bad value rather than
# defaulting silently: a typo that quietly restored the default would change what
# the detector measures without telling anyone.
# --------------------------------------------------------------------------- #


def _flag(env: Dict[str, str], name: str, default: bool) -> bool:
    raw = str(env.get(name, "") or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise PerceptionConfigurationError(
        f"{name} must be a boolean (1/0, true/false), got {raw!r}")


def _positive_int(env: Dict[str, str], name: str, default: int) -> int:
    raw = str(env.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise PerceptionConfigurationError(
            f"{name} must be a whole number of pixels, got {raw!r}") from exc
    if value <= 0:
        raise PerceptionConfigurationError(
            f"{name} must be positive, got {raw!r}")
    return value


def _confidence(env: Dict[str, str], name: str, default: float) -> float:
    raw = str(env.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise PerceptionConfigurationError(
            f"{name} must be a number between 0 and 1, got {raw!r}") from exc
    if not 0.0 <= value <= 1.0:
        raise PerceptionConfigurationError(
            f"{name} must be between 0 and 1, got {raw!r}")
    return value


__all__ = [
    "CalibrationBoard", "PerceptionConfig", "PerceptionConfigurationError",
    "DEFAULT_ARUCO_DICTIONARY", "DEFAULT_CORNER_MARKERS",
    "DEFAULT_CORNER_EXTENT_MM", "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_WIDTH", "DEFAULT_HEIGHT", "board_from_env", "camera_setting",
]
