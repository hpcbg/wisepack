"""The physical D435 + one real CAD part, from the command line.

    ./scripts/physical_c5.sh --model cylinder5 --roi 255,70,445,719

A CLI OVER `perception/physical_pipeline.py`, WHICH HOLDS THE ONE
IMPLEMENTATION. The dashboard's *Acquire & estimate* button calls the same
`run()`, so a pose seen in a browser is produced by exactly the code this
command runs — two copies would agree only until somebody edited one.

What lives here is the parts a terminal needs and a browser does not: argument
parsing, the printed diagnostics, and `--segment-only`.

WHICH OBJECT IS IN VIEW IS AN INPUT, NEVER INFERRED. `--model` names the CAD
part an operator placed on the table; several WISEPACK tubes share an outer
diameter and differ only in length, which one view cannot resolve. `--roi` says
WHERE to look and never WHAT is there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "perception"))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))

from physical_pipeline import (                                  # noqa: E402
    OUT, PhysicalAcquisitionError, eligible_models, fetch_image, run, segment,
    ARTIFACTS, require_camera)


def say(message: str) -> None:
    print(f"[physical-c5] {message}", flush=True)


def report_segmentation(document: Dict[str, Any]) -> None:
    """Every diagnostic the method produced, flat, as the worker returns it."""
    print()
    print("  SEGMENTATION — depth_plane_foreground")
    for label, key in (
            ("roi px", "roi_px"),
            ("foreground FULL frame", "foreground_points_full_frame"),
            ("plane_detected", "plane_detected"),
            ("plane residual mm", "plane_residual_mm"),
            ("plane inlier fraction", "plane_inlier_fraction"),
            ("plane normal", "plane_normal"),
            ("valid depth pixels", "valid_depth_pixels"),
            ("foreground points", "foreground_points"),
            ("connected components", "components"),
            ("component areas px", "component_areas_px"),
            ("selected component", "selected_component"),
            ("mask pixels", "mask_pixels"),
            ("mask area fraction", "mask_area_fraction"),
            ("mask centroid px", "mask_centroid_px"),
            ("mask median range mm", "mask_median_range_mm"),
            ("mask extent long mm", "mask_extent_long_mm"),
            ("mask extent across mm", "mask_extent_across_mm"),
            ("depth inside mask", "valid_depth_fraction_in_mask"),
            ("mask_valid", "mask_valid")):
        if key in document:
            print(f"    {label:24} {document[key]}")
    if document.get("reason"):
        print(f"    {'reason':24} {document['reason']}")


def report(document: Dict[str, Any]) -> None:
    observation = document["observation"]
    pose = observation["pose"]
    spread = document["repeatability"]
    checks = document["plausibility"]

    print()
    print("  POSE — camera optical frame, from the physical D435")
    print(f"    frame_id                {observation['frame_id']}")
    print(f"    model_id                {observation['object_model_id']}")
    print(f"    perception_method       {observation['perception_method']}")
    print(f"    acquisition             {document['run_label']}")
    print(f"    x, y, z (mm)            {pose['x_mm']:.1f}, {pose['y_mm']:.1f}, "
          f"{pose['z_mm']:.1f}")
    print("    quaternion (x,y,z,w)    "
          + ", ".join(f"{float(pose['orientation'][k]):.5f}" for k in "xyzw"))
    print(f"    pose_valid              {pose['valid']}")
    print(f"    measured_dof            {', '.join(pose['measured_dof'])}")
    print(f"    workarea_pose_available {pose['workarea_pose_available']}   "
          "(no camera-to-work-area extrinsic has been measured)")

    print()
    print("  REPEATABILITY — stationary scene, NOT accuracy")
    for label, key in (("body centre", "object_centre"),
                       ("model origin", "model_frame_origin")):
        block = spread.get(key) or {}
        if "radial_mm" not in block:
            continue
        print(f"    {label}")
        for axis in ("x_mm", "y_mm", "z_mm"):
            s = block[axis]
            print(f"      {axis:18} sd {s['sd']:.3f}  range {s['range']:.3f}")
        print(f"      {'radial':18} max {block['radial_mm']['max']:.3f} mm")
    if "orientation_vs_first_deg" in spread:
        print(f"    orientation          max "
              f"{spread['orientation_vs_first_deg']['max']:.3f} deg "
              "(includes any declared end swap)")
    if spread.get("tube_axis_line_deg"):
        print(f"    tube axis as a LINE  max "
              f"{spread['tube_axis_line_deg']['max']:.3f} deg")

    print()
    print("  PLAUSIBILITY — can only refute, never confirm")
    for key, value in checks.items():
        print(f"    {key:24} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="cylinder5",
                        help="the CAD model an operator placed in view")
    parser.add_argument("--frames", type=int, default=5,
                        help="frames captured, and poses estimated, for spread")
    parser.add_argument("--refine-iterations", type=int, default=5)
    parser.add_argument("--plane-tolerance-mm", type=float, default=None)
    parser.add_argument("--min-height-mm", type=float, default=None)
    parser.add_argument("--max-height-mm", type=float, default=None)
    parser.add_argument("--roi-radius-mm", type=float, default=None)
    parser.add_argument(
        "--roi", default="", metavar="x0,y0,x1,y1",
        help="operator ROI in COLOUR-IMAGE PIXELS. It says WHERE to look, not "
             "WHAT is there: identity stays --model.")
    parser.add_argument("--component", default=None,
                        choices=["largest", "centre"])
    parser.add_argument("--segment-only", action="store_true",
                        help="capture and segment, then stop — no inference")
    parser.add_argument("--dataset", default="",
                        help="reuse an existing capture instead of taking one")
    parser.add_argument("--models", action="store_true",
                        help="list the eligible CAD models and stop")
    args = parser.parse_args()

    if args.models:
        for entry in eligible_models():
            print(f"  {entry['model_id']:24} {entry['label']}")
        return 0

    roi = None
    if args.roi:
        try:
            roi = [int(round(float(v)))
                   for v in args.roi.replace(" ", "").split(",")]
        except ValueError:
            say(f"--roi must be four numbers x0,y0,x1,y1; got {args.roi!r}")
            return 2
        if len(roi) != 4:
            say(f"--roi needs exactly four values, got {len(roi)}")
            return 2
        say(f"operator ROI {roi} px — WHERE to look. The object's identity is "
            f"--model {args.model}, which an ROI can never establish.")

    options = {k: v for k, v in (
        ("plane_tolerance_mm", args.plane_tolerance_mm),
        ("min_height_mm", args.min_height_mm),
        ("max_height_mm", args.max_height_mm),
        ("roi_radius_mm", args.roi_radius_mm),
        ("component", args.component)) if v is not None}

    # `--segment-only` STOPS BEFORE INFERENCE, so it drives the stages directly
    # rather than `run()`. Same functions, one step short.
    if args.segment_only:
        try:
            require_camera()
            from physical_pipeline import capture as capture_stage  # noqa: PLC0415
            if args.dataset:
                dataset = args.dataset
                say(f"replaying capture {dataset}")
            else:
                dataset = os.path.basename(
                    capture_stage(args.model, args.frames)["root"])
                say(f"captured {dataset}")
            if roi:
                options["roi_px"] = roi
            document = segment(dataset, 0, options)
        except PhysicalAcquisitionError as exc:
            say(f"{exc.stage}: {exc.reason}")
            return 4
        report_segmentation(document)
        for kind, name in ARTIFACTS:
            fetch_image(kind, os.path.join(OUT, name))
        print()
        say(f"look at the mask before trusting any pose: {OUT}/mask_overlay.jpg")
        if not document.get("mask_valid"):
            say("the mask is NOT usable. No mask is fabricated to get past this.")
            return 4
        say("--segment-only: stopping before inference, as asked")
        return 0

    try:
        # The CLI plans nothing, so it takes the document and leaves the
        # batch — the dashboard is the caller that hands it to the workflow.
        document = run(model_id=args.model, roi_px=roi, frames=args.frames,
                       refine_iterations=args.refine_iterations,
                       dataset=args.dataset, segmentation_options=options,
                       log=say).document
    except PhysicalAcquisitionError as exc:
        say(f"{exc.stage}: {exc.reason}")
        if exc.detail.get("segmentation"):
            report_segmentation(exc.detail["segmentation"])
            print()
            say("nothing was estimated from it. No mask is fabricated to get "
                "past this step.")
        return 4

    report_segmentation(document["segmentation"])
    report(document)
    print()
    say(f"wrote {os.path.join(OUT, 'physical_c5.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
