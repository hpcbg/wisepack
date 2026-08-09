#!/usr/bin/env python3
"""Print WISEPACK's own ArUco calibration sheet.

    .venv-perception/bin/python scripts/generate_calibration_sheet.py
    .venv-perception/bin/python scripts/generate_calibration_sheet.py \
        --extent-mm 600 --paper A3 --out /tmp/board.png

WHY THIS EXISTS
---------------
Camera perception needs a printed board whose four ArUco markers sit at known
millimetre coordinates: that is what turns pixels into a measurement. WISEPACK
must be able to produce that board itself, from its own configuration, so a
demonstrator never depends on finding a PDF in somebody else's repository.

The defaults reproduce the geometry the current detector was validated with —
`DICT_ARUCO_ORIGINAL`, marker ids 11 / 10 / 15 / 16, a 130 mm square with the
origin at marker 11 — so a sheet printed from this script is interchangeable
with the one that work used. Print AT 100 % SCALE ("actual size", not "fit to
page"): every scaling error becomes a proportional error in every coordinate the
system afterwards reports.

The marker CENTRES are what the homography uses, so this places centres exactly
on the declared corners:

    marker_ids[0] -> corners_mm[0]   (the plane origin)
    marker_ids[1] -> corners_mm[1]
    ...

The printed sheet is a run artefact, not source: it lands in `results/` by
default and is regenerated whenever the board configuration changes.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "perception"))

from perception_config import (                                    # noqa: E402
    CalibrationBoard, DEFAULT_ARUCO_DICTIONARY, DEFAULT_CORNER_EXTENT_MM,
    DEFAULT_CORNER_MARKERS, PerceptionConfigurationError, board_from_env,
)

#: Paper sizes in millimetres, portrait.
PAPER_MM = {"A4": (210.0, 297.0), "A3": (297.0, 420.0), "A2": (420.0, 594.0)}

#: Printed dots per inch. 300 is what an ordinary office printer resolves and
#: keeps a 30 mm marker crisp enough for reliable detection.
DEFAULT_DPI = 300

#: Marker side length, millimetres. Large enough to detect across a table,
#: small enough that four of them plus margins fit a 130 mm square on A4.
DEFAULT_MARKER_MM = 30.0


def build(board: CalibrationBoard, paper: str, dpi: int, marker_mm: float,
          landscape: bool = False):
    """Render the sheet. Returns the image and the layout it used."""
    import cv2                                               # noqa: PLC0415
    import numpy as np                                       # noqa: PLC0415

    dictionary_id = getattr(cv2.aruco, board.dictionary, None)
    if dictionary_id is None:
        raise PerceptionConfigurationError(
            f"unknown ArUco dictionary {board.dictionary!r}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

    page_w_mm, page_h_mm = PAPER_MM[paper]
    if landscape:
        page_w_mm, page_h_mm = page_h_mm, page_w_mm

    def px(mm: float) -> int:
        return int(round(mm * dpi / 25.4))

    xs = [float(c[0]) for c in board.corners_mm]
    ys = [float(c[1]) for c in board.corners_mm]
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)

    # The board must fit WITH a half marker of bleed on each side, because the
    # markers are centred on the corners and therefore stick out.
    needed_w = span_x + marker_mm + 10.0
    needed_h = span_y + marker_mm + 10.0
    if needed_w > page_w_mm or needed_h > page_h_mm:
        raise PerceptionConfigurationError(
            f"a {span_x:.0f} x {span_y:.0f} mm board with {marker_mm:.0f} mm "
            f"markers needs {needed_w:.0f} x {needed_h:.0f} mm and does not fit "
            f"{paper} ({page_w_mm:.0f} x {page_h_mm:.0f} mm). Use a larger "
            "--paper, --landscape, or smaller --marker-mm.")

    image = np.full((px(page_h_mm), px(page_w_mm)), 255, dtype=np.uint8)

    # Centre the board on the page, and remember the offset so the printed
    # legend can state where the origin actually is.
    offset_x = (page_w_mm - span_x) / 2.0 - min(xs)
    offset_y = (page_h_mm - span_y) / 2.0 - min(ys)

    placed = []
    for marker_id, (corner_x, corner_y) in zip(board.marker_ids,
                                               board.corners_mm):
        marker = cv2.aruco.generateImageMarker(dictionary, int(marker_id),
                                               px(marker_mm))
        # PAGE Y GROWS DOWNWARD, PLANE Y GROWS AWAY FROM THE ORIGIN. Flipping
        # here is what makes the printed sheet match the coordinates the
        # detector reports; getting it wrong mirrors every measurement.
        centre_x_mm = offset_x + float(corner_x)
        centre_y_mm = page_h_mm - (offset_y + float(corner_y))
        top = px(centre_y_mm - marker_mm / 2.0)
        left = px(centre_x_mm - marker_mm / 2.0)
        size = marker.shape[0]
        image[top:top + size, left:left + size] = marker
        placed.append((int(marker_id), float(corner_x), float(corner_y)))

    # The legend. A sheet whose provenance and scale are printed on it is one an
    # operator can trust six months later.
    font, scale, thickness = 2, 0.6, 2       # cv2.FONT_HERSHEY_DUPLEX == 2
    lines = [
        "WISEPACK camera calibration sheet",
        f"dictionary {board.dictionary}   markers "
        + ", ".join(str(m) for m in board.marker_ids),
        f"plane {span_x:.0f} x {span_y:.0f} mm, origin at marker "
        f"{board.marker_ids[0]} (marker CENTRES are the corners)",
        f"marker side {marker_mm:.0f} mm   PRINT AT 100% SCALE - do not fit to page",
    ]
    y = px(12.0)
    for line in lines:
        cv2.putText(image, line, (px(10.0), y), font, scale, 0, thickness,
                    cv2.LINE_AA)
        y += px(6.0)

    return image, {"paper": paper, "dpi": dpi, "marker_mm": marker_mm,
                   "page_mm": (page_w_mm, page_h_mm), "markers": placed}


def main(argv=None) -> int:
    default_board = board_from_env()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(
        REPO, "results", "wisepack-calibration-sheet.png"),
        help="output image (.png or .jpg)")
    parser.add_argument("--extent-mm", type=float, default=None,
                        help=f"side of the square board in mm "
                             f"(default {DEFAULT_CORNER_EXTENT_MM:.0f}, or "
                             "WISEPACK_PERCEPTION_CALIBRATION_EXTENT_MM)")
    parser.add_argument("--markers", default=None,
                        help="comma-separated marker ids in plane order "
                             f"(default {','.join(str(m) for m in DEFAULT_CORNER_MARKERS)})")
    parser.add_argument("--dictionary", default=None,
                        help=f"ArUco dictionary (default {DEFAULT_ARUCO_DICTIONARY})")
    parser.add_argument("--paper", default="A4", choices=sorted(PAPER_MM))
    parser.add_argument("--landscape", action="store_true")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--marker-mm", type=float, default=DEFAULT_MARKER_MM)
    args = parser.parse_args(argv)

    board = default_board
    if args.extent_mm or args.markers or args.dictionary:
        markers = (tuple(int(m) for m in args.markers.split(","))
                   if args.markers else board.marker_ids)
        extent = args.extent_mm if args.extent_mm else max(
            float(c[0]) for c in board.corners_mm)
        board = CalibrationBoard.square(
            extent, markers, args.dictionary or board.dictionary)

    try:
        image, layout = build(board, args.paper, args.dpi, args.marker_mm,
                              args.landscape)
    except PerceptionConfigurationError as exc:
        print(f"generate_calibration_sheet: {exc}", file=sys.stderr)
        return 2

    import cv2                                               # noqa: PLC0415
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if not cv2.imwrite(args.out, image):
        print(f"generate_calibration_sheet: could not write {args.out}",
              file=sys.stderr)
        return 1

    print(f"wrote {args.out}")
    print(f"  paper       : {layout['paper']}"
          f"{' landscape' if args.landscape else ''} at {layout['dpi']} dpi")
    print(f"  marker side : {layout['marker_mm']:.0f} mm")
    for marker_id, x_mm, y_mm in layout["markers"]:
        print(f"  marker {marker_id:<3d} -> ({x_mm:.0f}, {y_mm:.0f}) mm")
    print("  PRINT AT 100% SCALE. Then set, if they are not the defaults:")
    print("    WISEPACK_PERCEPTION_CALIBRATION_MARKERS="
          + ",".join(str(m) for m in board.marker_ids))
    print("    WISEPACK_PERCEPTION_CALIBRATION_EXTENT_MM="
          + f"{max(float(c[0]) for c in board.corners_mm):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
