#!/usr/bin/env python3
"""Score two already-computed pose estimates against simulator ground truth.

A SEPARATE STEP, ON PURPOSE. This is the only place ground truth is opened, and
it runs after `model_free_query.py` has written both estimates — so neither
estimator can have been influenced by the answer. The estimator container is not
even given the file.

THE METRIC IS THE ONE THE CAD PATH ALREADY USES: the position of the object's
reference point, and the UNDIRECTED tube-axis line
`acos(|dot(axis_est, axis_gt)|)`. A straight tube reversed end for end is the
same object in the same place, and its spin about its own axis is not a task
quantity — scoring either as error would report a correct pose as wrong.

BOTH ESTIMATES ARE IN THE SAME FRAME, which is what makes them comparable: the
reference views were registered in the CAD frame, so the Neural Object Field
reconstructed in that frame too, and `register()` returns `ob_in_cam` for
whichever mesh it was given.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimates", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.estimates, encoding="utf-8") as handle:
        estimates = json.load(handle)
    with open(args.ground_truth, encoding="utf-8") as handle:
        truth = json.load(handle)

    T_truth = np.asarray(truth["T_camera_object"], dtype=np.float64)
    centre_mm = np.asarray(truth["model_center_mm"], dtype=np.float64)
    # The tube axis in the model's own frame, from the registry — the same
    # quantity the CAD evaluation uses.
    from wisepack_core.rgbd import load_object_registry
    model = load_object_registry(repo_root=REPO).models[truth["model_id"]]
    task_axis = tuple(model.task_axis_vector or ()) or model.task_axis
    unit = (np.asarray(task_axis, dtype=np.float64)
            if not isinstance(task_axis, str)
            else np.asarray({"x": (1, 0, 0), "y": (0, 1, 0),
                             "z": (0, 0, 1)}[task_axis], dtype=np.float64))
    unit = unit / np.linalg.norm(unit)

    def locate(T):
        """Where a pose puts the object's reference point, in millimetres."""
        return T[:3, :3] @ centre_mm + T[:3, 3] * 1000.0

    def axis_of(T):
        v = T[:3, :3] @ unit
        return v / np.linalg.norm(v)

    def line_angle(a, b):
        """UNDIRECTED: end-for-end reversal is the same line."""
        return math.degrees(math.acos(min(1.0, abs(float(a @ b)))))

    truth_point, truth_axis = locate(T_truth), axis_of(T_truth)
    scored = {}
    for name in ("cad", "model_free"):
        T = np.asarray(estimates[name]["matrix_m"], dtype=np.float64)
        point, axis = locate(T), axis_of(T)
        delta = point - truth_point
        along = float(delta @ truth_axis)
        scored[name] = {
            "position_error_mm": float(np.linalg.norm(delta)),
            "along_axis_mm": along,
            "transverse_mm": math.sqrt(max(0.0, float(delta @ delta) - along**2)),
            "tube_axis_line_error_deg": line_angle(axis, truth_axis),
            "inference_seconds": estimates[name]["inference_seconds"],
            "mesh_path": estimates[name]["mesh_path"],
            "mesh_vertices": estimates[name]["mesh_vertices"],
            "reference_point_mm": [float(v) for v in point],
        }

    # THE TWO ESTIMATES AGAINST EACH OTHER, which is a different question from
    # either against truth: it says how far apart the methods are, without
    # reference to which is right.
    cad_T = np.asarray(estimates["cad"]["matrix_m"], dtype=np.float64)
    mf_T = np.asarray(estimates["model_free"]["matrix_m"], dtype=np.float64)
    between = {
        "centre_difference_mm": float(np.linalg.norm(locate(cad_T) - locate(mf_T))),
        "tube_axis_line_difference_deg": line_angle(axis_of(cad_T), axis_of(mf_T)),
    }

    report = {
        "query_dir": estimates.get("query_dir"),
        "query_mask_pixels": estimates.get("query_mask_pixels"),
        "ground_truth_source": args.ground_truth,
        "ground_truth_read_after_both_estimates": True,
        "reference_point_gt_mm": [float(v) for v in truth_point],
        "cad": scored["cad"],
        "model_free": scored["model_free"],
        "cad_vs_model_free": between,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n  %-14s %14s %14s" % ("", "CAD", "MODEL-FREE"))
    print("  %-14s %11.3f mm %11.3f mm"
          % ("position err", scored["cad"]["position_error_mm"],
             scored["model_free"]["position_error_mm"]))
    print("  %-14s %11.3f °  %11.3f °"
          % ("axis-line err", scored["cad"]["tube_axis_line_error_deg"],
             scored["model_free"]["tube_axis_line_error_deg"]))
    print("  %-14s %11.2f s  %11.2f s"
          % ("inference", scored["cad"]["inference_seconds"],
             scored["model_free"]["inference_seconds"]))
    print("  %-14s %11d    %11d"
          % ("mesh verts", scored["cad"]["mesh_vertices"],
             scored["model_free"]["mesh_vertices"]))
    print("\n  CAD vs model-free: centre %.3f mm, axis line %.3f deg"
          % (between["centre_difference_mm"],
             between["tube_axis_line_difference_deg"]))
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
