#!/usr/bin/env python3
"""Score physical D435 CAD-vs-model-free runs WITHOUT any ground truth.

THERE IS NO INDEPENDENTLY MEASURED PHYSICAL POSE FOR CYLINDER5, so there is no
physical accuracy here and none is computed. Every number below is one of two
things, and the distinction is the whole point of this file:

  REPEATABILITY  how tightly ONE method agrees with ITSELF across independent
                 frames of a stationary object. A method can be repeatable and
                 wrong; this says nothing about whether it is right.

  AGREEMENT      how closely the two methods land on each other, frame by frame.
                 Two estimators agreeing is evidence they are not failing in
                 unrelated ways. It is NOT evidence either one is accurate —
                 they share a camera, a frame, a mask and a depth map, so they
                 can be wrong together.

Neither is an error. Nothing in this file may be reported as accuracy.

THE AXIS IS UNDIRECTED, as everywhere else in this comparison: a straight tube
reversed end for end is the same tube in the same place. That also decides how
axes are averaged — the mean of a set of undirected axes is the dominant
eigenvector of the sum of their outer products, not the mean of the vectors,
which would cancel to nothing the moment two of them pointed opposite ways.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))


def _stats(values):
    import numpy as np
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std()), "min": float(a.min()),
            "max": float(a.max()), "n": int(a.size)}


def _mean_axis(axes):
    """The dominant direction of a set of UNDIRECTED axes.

    Averaging the vectors themselves is wrong here: `a` and `-a` are the same
    axis, so a set that happens to contain both would average towards zero and
    produce a meaningless "mean" with a huge apparent spread. The eigenvector
    of `sum(a a^T)` is invariant to those sign flips, which is exactly the
    symmetry the metric has.
    """
    import numpy as np
    scatter = np.zeros((3, 3))
    for a in axes:
        scatter += np.outer(a, a)
    values, vectors = np.linalg.eigh(scatter)
    return vectors[:, int(np.argmax(values))]


def _line_angle(a, b):
    import numpy as np
    return math.degrees(math.acos(min(1.0, abs(float(np.asarray(a) @ np.asarray(b))))))


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    with open(args.estimates, encoding="utf-8") as handle:
        estimates = json.load(handle)

    from wisepack_core.rgbd import load_object_registry
    registry = load_object_registry(repo_root=REPO)
    model = registry.models[estimates["model_id"]]

    # THE SAME REFERENCE POINT AND THE SAME AXIS THE SIMULATED BENCHMARK USED,
    # read from the same registry, so the physical and simulated numbers are
    # the same quantity measured on different data.
    mesh_centre_mm = np.asarray(estimates["model_center_mm"], dtype=np.float64)
    task_axis = tuple(model.task_axis_vector or ()) or model.task_axis
    unit = (np.asarray(task_axis, dtype=np.float64)
            if not isinstance(task_axis, str)
            else np.asarray({"x": (1, 0, 0), "y": (0, 1, 0),
                             "z": (0, 0, 1)}[task_axis], dtype=np.float64))
    unit = unit / np.linalg.norm(unit)

    def centre_of(T):
        """The tube's reference point in camera_color_optical_frame, in mm."""
        return T[:3, :3] @ mesh_centre_mm + T[:3, 3] * 1000.0

    def axis_of(T):
        v = T[:3, :3] @ unit
        return v / np.linalg.norm(v)

    methods = ("cad", "model_free")
    frames, per_method = [], {m: {"centres": [], "axes": [], "t": []} for m in methods}
    for entry in estimates["results"]:
        row = {"frame": entry["id"], "mask_pixels": entry.get("mask_pixels")}
        centres, axes = {}, {}
        for tag in methods:
            T = np.asarray(entry[tag]["matrix_m"], dtype=np.float64)
            centres[tag], axes[tag] = centre_of(T), axis_of(T)
            per_method[tag]["centres"].append(centres[tag])
            per_method[tag]["axes"].append(axes[tag])
            per_method[tag]["t"].append(entry[tag]["inference_seconds"])
            row[tag] = {
                "centre_mm_camera_optical": [float(v) for v in centres[tag]],
                "tube_axis_camera_optical": [float(v) for v in axes[tag]],
                "inference_seconds": entry[tag]["inference_seconds"],
            }
        # AGREEMENT, PER FRAME. Same frame, same mask, same intrinsics — the
        # only difference between the two numbers is the geometry each was given.
        row["agreement"] = {
            "centre_difference_mm":
                float(np.linalg.norm(centres["cad"] - centres["model_free"])),
            "tube_axis_line_difference_deg":
                _line_angle(axes["cad"], axes["model_free"]),
        }
        frames.append(row)

    repeatability = {}
    for tag in methods:
        centres = np.asarray(per_method[tag]["centres"], dtype=np.float64)
        mean_centre = centres.mean(axis=0)
        # DEVIATION FROM THE MEAN, not a per-axis standard deviation alone: the
        # quantity that matters for a pick is how far the reported point can be
        # from the point the method usually reports, in any direction.
        deviations = [float(np.linalg.norm(c - mean_centre)) for c in centres]
        axes = per_method[tag]["axes"]
        mean_axis = _mean_axis(axes)
        angles = [_line_angle(a, mean_axis) for a in axes]
        repeatability[tag] = {
            "frames": len(centres),
            "mean_centre_mm_camera_optical": [float(v) for v in mean_centre],
            "centre_deviation_from_mean_mm": _stats(deviations),
            "centre_per_axis_std_mm": {
                "x": float(centres[:, 0].std()), "y": float(centres[:, 1].std()),
                "z": float(centres[:, 2].std())},
            "mean_tube_axis_camera_optical": [float(v) for v in mean_axis],
            "tube_axis_line_deviation_deg": _stats(angles),
            "inference_seconds": _stats(per_method[tag]["t"]),
        }

    agreement = {
        "centre_difference_mm":
            _stats([f["agreement"]["centre_difference_mm"] for f in frames]),
        "tube_axis_line_difference_deg":
            _stats([f["agreement"]["tube_axis_line_difference_deg"] for f in frames]),
    }

    report = {
        "label": args.label or estimates.get("label", ""),
        "capture": estimates.get("capture_root"),
        "model_id": estimates["model_id"],
        "frame_id": "camera_color_optical_frame",
        "physical_ground_truth_available": False,
        "accuracy_reported": False,
        "interpretation_note": (
            "REPEATABILITY and AGREEMENT only. No independently measured "
            "physical pose for this object exists, so no position or "
            "orientation ERROR is computed and none may be quoted. Two "
            "estimators agreeing is not evidence that either is correct."),
        "model_free_representation_digest": estimates.get("reference_digest"),
        "per_frame": frames,
        "repeatability": repeatability,
        "cad_vs_model_free_agreement": agreement,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    title = f" — {report['label']}" if report["label"] else ""
    print(f"\nPHYSICAL D435, NO GROUND TRUTH{title}")
    print("  repeatability = self-consistency across frames; "
          "agreement = the two methods vs each other. Neither is accuracy.")

    print("\nPER FRAME (centre in camera_color_optical_frame, mm)")
    print("  %-8s %8s | %-28s | %-28s | %8s %8s" %
          ("frame", "px", "CAD centre x,y,z", "model-free centre x,y,z",
           "Δmm", "Δdeg"))
    for f in frames:
        c, m = f["cad"], f["model_free"]
        print("  %-8s %8s | %28s | %28s | %8.3f %8.3f" % (
            f["frame"], f["mask_pixels"],
            "%.1f, %.1f, %.1f" % tuple(c["centre_mm_camera_optical"]),
            "%.1f, %.1f, %.1f" % tuple(m["centre_mm_camera_optical"]),
            f["agreement"]["centre_difference_mm"],
            f["agreement"]["tube_axis_line_difference_deg"]))

    print("\nREPEATABILITY (spread about each method's own mean)")
    print("  %-11s %10s %10s %10s %10s" %
          ("", "centre mm", "centre max", "axis deg", "axis max"))
    for tag in methods:
        r = repeatability[tag]
        print("  %-11s %10.3f %10.3f %10.3f %10.3f" % (
            tag, r["centre_deviation_from_mean_mm"]["mean"],
            r["centre_deviation_from_mean_mm"]["max"],
            r["tube_axis_line_deviation_deg"]["mean"],
            r["tube_axis_line_deviation_deg"]["max"]))

    print("\nCAD vs MODEL-FREE AGREEMENT (not accuracy)")
    print("  %-22s %8s %8s %8s %8s %8s" %
          ("", "mean", "median", "std", "min", "max"))
    for key, label in (("centre_difference_mm", "centre mm"),
                       ("tube_axis_line_difference_deg", "axis deg")):
        s = agreement[key]
        print("  %-22s %8.3f %8.3f %8.3f %8.3f %8.3f" %
              (label, s["mean"], s["median"], s["std"], s["min"], s["max"]))

    print("\nINFERENCE TIME (s)")
    for tag in methods:
        s = repeatability[tag]["inference_seconds"]
        print("  %-11s mean %.3f  std %.3f  min %.3f  max %.3f"
              % (tag, s["mean"], s["std"], s["min"], s["max"]))
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
