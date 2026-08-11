#!/usr/bin/env python3
"""Run CAD and model-free FoundationPose over a whole query set.

RUNS INSIDE THE FOUNDATIONPOSE CONTAINER, which is given the query images and
the two meshes — and NOT the ground truth, which lives in a sibling directory
that is never mounted. Scoring is a separate host-side step.

ONE ESTIMATOR PER MESH, REUSED ACROSS QUERIES. `FoundationPose` holds the mesh
and its sampled model points; rebuilding it per query would re-pay that cost and,
worse, would make per-query inference times incomparable between the two methods.
The scorer, refiner and rasteriser context are shared for the same reason.

THE SAME QUERY REACHES BOTH. Each frame is loaded once and handed to both
estimators, so RGB, depth, intrinsics and mask are identical by construction
rather than by two loads agreeing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-dir", required=True)
    parser.add_argument("--cad-mesh", required=True)
    parser.add_argument("--cad-scale-to-metres", type=float, default=0.001)
    parser.add_argument("--model-free-mesh", required=True)
    parser.add_argument("--model-free-scale-to-metres", type=float, default=1.0)
    parser.add_argument("--refine-iterations", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, "/opt/foundationpose")
    sys.path.insert(0, "/opt/foundationpose/bundlesdf")

    import cv2
    import numpy as np
    import trimesh
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

    # GROUND TRUTH IS NOT REACHABLE, and this asserts it rather than assuming it:
    # if the answer were ever mounted alongside the queries, this would fail
    # loudly instead of quietly producing a flattered result.
    for root, _dirs, files in os.walk(args.queries_dir):
        for name in files:
            if "ground_truth" in name or name.endswith("_gt.json"):
                raise SystemExit(
                    f"ground truth is visible to the estimator: {root}/{name}")

    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    def build(mesh_path: str, scale: float):
        mesh = trimesh.load(mesh_path, force="mesh")
        if scale != 1.0:
            mesh.apply_scale(scale)
        estimator = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
            mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx)
        return estimator, int(len(mesh.vertices))

    cad_est, cad_verts = build(args.cad_mesh, args.cad_scale_to_metres)
    mf_est, mf_verts = build(args.model_free_mesh,
                             args.model_free_scale_to_metres)
    print(f"[bench] CAD mesh {cad_verts} verts | model-free mesh {mf_verts} verts",
          flush=True)

    query_ids = sorted(d for d in os.listdir(args.queries_dir)
                       if os.path.isdir(os.path.join(args.queries_dir, d)))
    results = []
    for qid in query_ids:
        qdir = os.path.join(args.queries_dir, qid)
        rgb = cv2.cvtColor(cv2.imread(f"{qdir}/rgb/000000.png"), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(f"{qdir}/depth/000000.png", -1) / 1e3
        mask = cv2.imread(f"{qdir}/masks/000000.png", -1) > 0
        K = np.loadtxt(f"{qdir}/cam_K.txt").reshape(3, 3)

        entry = {"id": qid, "mask_pixels": int(mask.sum())}
        for tag, estimator in (("cad", cad_est), ("model_free", mf_est)):
            started = time.time()
            pose = estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                      iteration=args.refine_iterations)
            entry[tag] = {
                "matrix_m": np.asarray(pose, dtype=np.float64).tolist(),
                "inference_seconds": round(time.time() - started, 3),
            }
        results.append(entry)
        print(f"[bench] {qid}: cad {entry['cad']['inference_seconds']:.2f}s  "
              f"model-free {entry['model_free']['inference_seconds']:.2f}s  "
              f"({entry['mask_pixels']} px)", flush=True)

    document = {
        "queries_dir": args.queries_dir,
        "query_count": len(results),
        "refine_iterations": args.refine_iterations,
        "cad_mesh": args.cad_mesh, "cad_mesh_vertices": cad_verts,
        "model_free_mesh": args.model_free_mesh,
        "model_free_mesh_vertices": mf_verts,
        "ground_truth_read": False,
        "shared_query_per_pair": True,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    print("MODEL-FREE-BENCH OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:                                    # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("MODEL-FREE-BENCH FAILED", flush=True)
        raise SystemExit(1)
