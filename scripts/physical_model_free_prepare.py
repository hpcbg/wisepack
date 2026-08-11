#!/usr/bin/env python3
"""Capture physical D435 frames and give each ONE mask both estimators will use.

    ./scripts/physical_model_free_prepare.py --frames 12 --roi 255,70,445,719

WHY THE MASK IS MADE HERE AND ONCE PER FRAME
--------------------------------------------
The experiment compares two estimators on the SAME observation. If each of them
segmented independently, a difference between their poses could come from the
masks rather than from the geometry each was given, and that is the one thing
this comparison must not confuse. So one mask per frame is computed here, saved,
and handed to both.

THE MASK IS PRODUCED WITHOUT ANY CAD MODEL, which is what makes it legitimate to
share with the model-free estimator. `depth_plane_foreground` takes a depth image
and the camera's intrinsics — no mesh, no model_id, no dimensions. It fits the
dominant plane and keeps what stands on it. Nothing about Cylinder5's shape can
reach the model-free path through it.

THIS IS THE WORKER'S OWN SEGMENTER, imported rather than reimplemented. It needs
no GPU and no weights, so it runs on the host, and its plane fit uses a fixed
seed — the same frame yields the same mask on every run.

THE OPERATOR ROI SAYS WHERE TO LOOK, NEVER WHAT IS THERE. On a bench holding
more than one object, `depth_plane_foreground` returns one component containing
several of them; the ROI restores the single-object precondition it needs. It is
given by whoever set the bench up and is NEVER derived from a model's
dimensions — selecting the region by the size of the part you expect is
recognising the object by its shape and then measuring its pose with that same
shape.

NO MASK IS EVER FABRICATED. A frame whose mask fails validation is reported and
excluded; it is not repaired, and its absence is recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "perception", "foundationpose", "worker"))

PORT = os.environ.get("WISEPACK_FP_PORT", "22201")
BASE = f"http://127.0.0.1:{PORT}"
CAPTURES = os.path.join(REPO, ".cache-perception", "rgbd-captures")
#: One raw depth unit is one millimetre on this device, as the capture metadata
#: records. It is passed explicitly because it cannot be read off an image.
DEPTH_SCALE_MM = 1.0


def say(message: str) -> None:
    print(f"[physical-prepare] {message}", flush=True)


def post(path: str, payload: dict, timeout: float = 600.0):
    request = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    import cv2
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="cylinder5")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--dataset", default="",
                        help="reuse an existing capture instead of acquiring")
    parser.add_argument("--roi", default="", metavar="x0,y0,x1,y1",
                        help="operator ROI in COLOUR-IMAGE PIXELS: where to "
                             "look. Never what is there.")
    parser.add_argument("--plane-tolerance-mm", type=float, default=None)
    parser.add_argument("--min-height-mm", type=float, default=None)
    parser.add_argument("--max-height-mm", type=float, default=None)
    parser.add_argument("--component", default=None, choices=["largest", "centre"])
    parser.add_argument("--out", required=True,
                        help="directory to materialise per-frame inputs into")
    args = parser.parse_args()

    from segmentation import segment                          # noqa: PLC0415

    options = {}
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
        options["roi_px"] = roi
        say(f"operator ROI {roi} px — WHERE to look. The object's identity is "
            f"--model-id {args.model_id}, which an ROI can never establish.")
    for key, value in (("plane_tolerance_mm", args.plane_tolerance_mm),
                       ("min_height_mm", args.min_height_mm),
                       ("max_height_mm", args.max_height_mm),
                       ("component", args.component)):
        if value is not None:
            options[key] = value

    # ---------------------------------------------------------------- capture
    if args.dataset:
        dataset = args.dataset
        say(f"reusing capture {dataset}")
        capture_meta = json.load(open(os.path.join(CAPTURES, dataset,
                                                   "metadata.json"),
                                     encoding="utf-8"))
    else:
        say(f"capturing {args.frames} frames from the physical D435")
        try:
            capture_meta = post("/camera/capture", {
                "model_id": args.model_id, "frames": args.frames,
                "width": 1280, "height": 720, "fps": 30, "align": True})
        except urllib.error.URLError as exc:
            say(f"capture failed: {exc}. Is the worker running? "
                "./scripts/setup_foundationpose.sh --no-build --run")
            return 1
        dataset = os.path.basename(capture_meta["root"])
    # A MASK DRAWN ON THE COLOUR IMAGE WOULD SELECT THE WRONG DEPTH PIXELS if
    # the streams were not aligned, so this is a refusal, not a warning.
    if not capture_meta.get("alignment_verified"):
        say("the capture does not carry verified depth/colour alignment")
        return 1

    root = os.path.join(CAPTURES, dataset)
    K = np.loadtxt(os.path.join(root, "cam_K.txt")).reshape(3, 3)
    intrinsics = [[float(v) for v in row] for row in K]
    rgb_files = sorted(os.listdir(os.path.join(root, "rgb")))
    say(f"{dataset}: {len(rgb_files)} frames, "
        f"fx {K[0][0]:.2f} fy {K[1][1]:.2f} cx {K[0][2]:.2f} cy {K[1][2]:.2f}")

    # ----------------------------------------------------------- segment each
    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out, exist_ok=True)
    kept, dropped = [], []
    for index, name in enumerate(rgb_files):
        rgb = cv2.imread(os.path.join(root, "rgb", name))
        raw = cv2.imread(os.path.join(root, "depth", name), -1)
        depth_mm = (raw.astype(np.float64) * DEPTH_SCALE_MM).astype(np.uint16)

        result = segment("depth_plane_foreground", depth_mm, intrinsics,
                         dict(options))
        document = result.to_dict()
        pixels = int(result.mask.sum())
        if not result.valid:
            say(f"  frame {index:02d}: MASK REJECTED — {result.reason}")
            dropped.append({"frame": index, "reason": result.reason,
                            "mask_pixels": pixels})
            continue

        fid = f"f{len(kept):02d}"
        fdir = os.path.join(args.out, fid)
        for sub in ("rgb", "depth", "masks"):
            os.makedirs(os.path.join(fdir, sub), exist_ok=True)
        cv2.imwrite(f"{fdir}/rgb/000000.png", rgb)
        cv2.imwrite(f"{fdir}/depth/000000.png", raw.astype(np.uint16))
        cv2.imwrite(f"{fdir}/masks/000000.png",
                    (result.mask.astype(np.uint8) * 255))
        np.savetxt(f"{fdir}/cam_K.txt", K, fmt="%.18e")
        # The mask ON the photograph, so it can be looked at rather than
        # trusted. A mask judged only by its pixel count is a mask nobody
        # has actually inspected.
        overlay = rgb.copy()
        overlay[result.mask] = (0.45 * overlay[result.mask]
                                + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out, f"{fid}_mask_overlay.jpg"), overlay)

        kept.append({
            "id": fid, "source_frame": index, "source_file": name,
            "mask_pixels": pixels,
            "mask_median_range_mm": document.get("mask_median_range_mm"),
            "mask_extent_long_mm": document.get("mask_extent_long_mm"),
            "mask_extent_across_mm": document.get("mask_extent_across_mm"),
            "plane_residual_mm": document.get("plane_residual_mm"),
            "components": document.get("components"),
            "valid_depth_fraction_in_mask":
                document.get("valid_depth_fraction_in_mask"),
        })
        say(f"  frame {index:02d} -> {fid}: {pixels} px, "
            f"range {document.get('mask_median_range_mm')} mm, "
            f"extent {document.get('mask_extent_long_mm')} x "
            f"{document.get('mask_extent_across_mm')} mm")

    manifest = {
        "source": "realsense_d435",
        "purpose": "physical_cad_vs_model_free_comparison",
        "model_id": args.model_id,
        "dataset": dataset,
        "capture_root": root,
        "frames_kept": len(kept), "frames_dropped": len(dropped),
        "intrinsics": intrinsics,
        "depth_scale_mm_per_unit": DEPTH_SCALE_MM,
        "alignment_verified": True,
        "mask_source": "depth_plane_foreground",
        "mask_options": options,
        "mask_is_cad_free": True,
        "mask_note": (
            "One mask per frame, produced from depth and intrinsics only and "
            "given identically to BOTH estimators. No CAD model, model_id or "
            "dimension takes part in producing it, which is what makes sharing "
            "it with the model-free estimator legitimate."),
        "physical_ground_truth": None,
        "ground_truth_note": (
            "No independently measured physical pose exists for this object. "
            "Repeatability and inter-method agreement only; no accuracy."),
        "device": capture_meta.get("device"),
        "frames": kept, "dropped": dropped,
    }
    with open(os.path.join(args.out, "prepare_manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    say(f"{len(kept)} usable frame(s), {len(dropped)} dropped -> {args.out}")
    if kept:
        extents = [f["mask_extent_long_mm"] for f in kept
                   if f["mask_extent_long_mm"]]
        if extents:
            say(f"measured mask long extent: {min(extents):.0f}-"
                f"{max(extents):.0f} mm (a DIAGNOSTIC from depth, not a "
                f"selection criterion)")
        say(f"LOOK AT THE MASKS before trusting any pose: {args.out}/f00_mask_overlay.jpg")
    if not kept:
        say("no usable masks; nothing to estimate from")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
