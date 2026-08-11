#!/usr/bin/env python3
"""One query frame, two estimators: FoundationPose CAD vs model-free.

RUNS INSIDE THE FOUNDATIONPOSE CONTAINER, like the representation build.

WHAT MAKES THIS A CONTROLLED COMPARISON
---------------------------------------
Both estimators are given the SAME query: the same RGB, the same depth, the same
intrinsics, the same instance mask, from the same camera pose, of the same object
at the same pose, rendered with the same `brushed_steel` material. One variable
differs, and it is the only one that is supposed to:

    A. CAD         model_pts / mesh = the exact Cylinder5 CAD
    B. model-free  model_pts / mesh = the mesh reconstructed by the Neural
                   Object Field from 15 rendered reference views

Both go through upstream's own `FoundationPose(...).register(...)` — the same
class, the same refiner, the same scorer, the same iteration count.

GROUND TRUTH IS NOT READABLE FROM HERE, AND THAT IS STRUCTURAL. This script
never opens `ground_truth.json`; it writes two poses and exits. Scoring happens
in a separate host-side step that runs afterwards, so neither estimate can have
been influenced by the answer — the ordering is enforced by which process can
see which file, not by remembering to look later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", required=True,
                        help="the query frame, in upstream dataset layout")
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

    # ---- THE ONE QUERY, loaded once and shared -----------------------------
    rgb = cv2.cvtColor(cv2.imread(f"{args.query_dir}/rgb/000000.png"),
                       cv2.COLOR_BGR2RGB)
    depth = cv2.imread(f"{args.query_dir}/depth/000000.png", -1) / 1e3
    mask = cv2.imread(f"{args.query_dir}/masks/000000.png", -1) > 0
    K = np.loadtxt(f"{args.query_dir}/cam_K.txt").reshape(3, 3)
    print(f"[query] rgb {rgb.shape} depth {depth.shape} "
          f"mask {int(mask.sum())} px", flush=True)

    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    def estimate(tag: str, mesh_path: str, scale: float) -> dict:
        mesh = trimesh.load(mesh_path, force="mesh")
        if scale != 1.0:
            mesh.apply_scale(scale)
        estimator = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
            mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx)
        started = time.time()
        pose = estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                  iteration=args.refine_iterations)
        elapsed = time.time() - started
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        print(f"[{tag}] {elapsed:.2f}s  mesh {len(vertices)} verts", flush=True)
        return {
            "matrix_m": np.asarray(pose, dtype=np.float64).tolist(),
            "inference_seconds": round(elapsed, 3),
            "mesh_path": mesh_path,
            "mesh_vertices": int(len(vertices)),
            "mesh_scale_to_metres": scale,
            "mesh_extents_m": [float(v) for v in
                               (vertices.max(axis=0) - vertices.min(axis=0))],
        }

    results = {
        "query_dir": args.query_dir,
        "query_mask_pixels": int(mask.sum()),
        "intrinsics": K.tolist(),
        "refine_iterations": args.refine_iterations,
        # THE SAME INPUTS, SAID EXPLICITLY, so the comparison can be checked
        # rather than trusted.
        "shared_query": True,
        "ground_truth_read": False,
        "cad": estimate("cad", args.cad_mesh, args.cad_scale_to_metres),
        "model_free": estimate("model-free", args.model_free_mesh,
                               args.model_free_scale_to_metres),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("MODEL-FREE-QUERY OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:                                    # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("MODEL-FREE-QUERY FAILED", flush=True)
        raise SystemExit(1)
