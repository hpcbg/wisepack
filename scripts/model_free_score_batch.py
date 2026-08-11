#!/usr/bin/env python3
"""Score a whole CAD-vs-model-free benchmark, per query and in aggregate.

THE ONLY PLACE GROUND TRUTH IS OPENED, and it runs after every estimate exists.
The estimator container is never given this directory.

THE METRICS ARE THE ONES THE CAD PATH ALREADY USES: reference-point position,
and the UNDIRECTED tube-axis line `acos(|dot(a, b)|)` — a straight tube reversed
end for end is the same object in the same place, and its spin about its own axis
is not a task quantity.

AGGREGATES ARE REPORTED WITH SPREAD, not just a mean. Ten queries is a small
sample, and a mean alone would hide whether a method is consistently close or
occasionally very wrong — which is the difference that matters for deciding
whether it can be trusted.
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
    return {
        "mean": float(a.mean()), "median": float(np.median(a)),
        # POPULATION SD over the sample we have; with n=10 the distinction from
        # the sample SD is small but stating which one it is costs nothing.
        "std": float(a.std()), "min": float(a.min()), "max": float(a.max()),
        # p90 ON TEN POINTS is an interpolation between the 9th and 10th, so it
        # is reported but should be read as "near the worst", not as a quantile
        # anyone should extrapolate from.
        "p90": float(np.percentile(a, 90)), "n": int(a.size),
    }


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimates", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.estimates, encoding="utf-8") as handle:
        estimates = json.load(handle)

    from wisepack_core.rgbd import load_object_registry
    registry = load_object_registry(repo_root=REPO)

    rows, per_method = [], {"cad": {"pos": [], "ang": [], "t": []},
                            "model_free": {"pos": [], "ang": [], "t": []}}
    for entry in estimates["results"]:
        qid = entry["id"]
        gt_path = os.path.join(args.ground_truth_dir, f"{qid}.json")
        with open(gt_path, encoding="utf-8") as handle:
            truth = json.load(handle)
        T_truth = np.asarray(truth["T_camera_object"], dtype=np.float64)
        centre_mm = np.asarray(truth["model_center_mm"], dtype=np.float64)
        model = registry.models[truth["model_id"]]
        task_axis = tuple(model.task_axis_vector or ()) or model.task_axis
        unit = (np.asarray(task_axis, dtype=np.float64)
                if not isinstance(task_axis, str)
                else np.asarray({"x": (1, 0, 0), "y": (0, 1, 0),
                                 "z": (0, 0, 1)}[task_axis], dtype=np.float64))
        unit = unit / np.linalg.norm(unit)

        def locate(T):
            return T[:3, :3] @ centre_mm + T[:3, 3] * 1000.0

        def axis_of(T):
            v = T[:3, :3] @ unit
            return v / np.linalg.norm(v)

        def line_angle(a, b):
            return math.degrees(math.acos(min(1.0, abs(float(a @ b)))))

        truth_point, truth_axis = locate(T_truth), axis_of(T_truth)
        row = {"id": qid, "mask_pixels": entry["mask_pixels"]}
        poses = {}
        for tag in ("cad", "model_free"):
            T = np.asarray(entry[tag]["matrix_m"], dtype=np.float64)
            poses[tag] = T
            pos = float(np.linalg.norm(locate(T) - truth_point))
            ang = line_angle(axis_of(T), truth_axis)
            secs = entry[tag]["inference_seconds"]
            row[tag] = {"position_error_mm": pos,
                        "tube_axis_line_error_deg": ang,
                        "inference_seconds": secs}
            per_method[tag]["pos"].append(pos)
            per_method[tag]["ang"].append(ang)
            per_method[tag]["t"].append(secs)
        row["cad_vs_model_free"] = {
            "centre_difference_mm":
                float(np.linalg.norm(locate(poses["cad"]) - locate(poses["model_free"]))),
            "tube_axis_line_difference_deg":
                line_angle(axis_of(poses["cad"]), axis_of(poses["model_free"])),
        }
        rows.append(row)

    aggregate = {
        tag: {"position_error_mm": _stats(v["pos"]),
              "tube_axis_line_error_deg": _stats(v["ang"]),
              "inference_seconds": _stats(v["t"])}
        for tag, v in per_method.items()
    }
    ratio_mean = (aggregate["model_free"]["position_error_mm"]["mean"]
                  / aggregate["cad"]["position_error_mm"]["mean"])
    ratio_median = (aggregate["model_free"]["position_error_mm"]["median"]
                    / aggregate["cad"]["position_error_mm"]["median"])
    report = {
        "queries": estimates["query_count"],
        "cad_mesh_vertices": estimates["cad_mesh_vertices"],
        "model_free_mesh_vertices": estimates["model_free_mesh_vertices"],
        "ground_truth_read_after_all_estimates": True,
        "per_query": rows,
        "aggregate": aggregate,
        "model_free_over_cad_translation_error": {
            "mean_ratio": ratio_mean, "median_ratio": ratio_median},
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nPER QUERY")
    print("  %-5s %7s | %9s %8s | %9s %8s | %8s %7s" %
          ("id", "px", "CAD mm", "CAD deg", "MF mm", "MF deg", "Δmm", "Δdeg"))
    for r in rows:
        print("  %-5s %7d | %9.3f %8.3f | %9.3f %8.3f | %8.3f %7.3f"
              % (r["id"], r["mask_pixels"],
                 r["cad"]["position_error_mm"], r["cad"]["tube_axis_line_error_deg"],
                 r["model_free"]["position_error_mm"],
                 r["model_free"]["tube_axis_line_error_deg"],
                 r["cad_vs_model_free"]["centre_difference_mm"],
                 r["cad_vs_model_free"]["tube_axis_line_difference_deg"]))

    print("\nAGGREGATE (n=%d)" % report["queries"])
    for metric, unit_label in (("position_error_mm", "mm"),
                               ("tube_axis_line_error_deg", "deg"),
                               ("inference_seconds", "s")):
        print("  %s (%s)" % (metric, unit_label))
        print("    %-11s %8s %8s %8s %8s %8s %8s" %
              ("", "mean", "median", "std", "min", "max", "p90"))
        for tag in ("cad", "model_free"):
            s = aggregate[tag][metric]
            print("    %-11s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
                  % (tag, s["mean"], s["median"], s["std"], s["min"],
                     s["max"], s["p90"]))
    print("\n  model-free / CAD translation error: mean %.2fx, median %.2fx"
          % (ratio_mean, ratio_median))
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
