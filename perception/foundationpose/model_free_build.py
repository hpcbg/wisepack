#!/usr/bin/env python3
"""Build the FoundationPose model-free representation, through the official path.

RUNS INSIDE THE FOUNDATIONPOSE CONTAINER. It is copied in and executed there,
because the Neural Object Field needs the pinned CUDA runtime, the compiled
BundleSDF extensions and kaolin — none of which exist on the host.

WHAT IT CALLS, AND WHAT IT DOES NOT REIMPLEMENT
-----------------------------------------------
`bundlesdf/run_nerf.py::run_one_ob(base_dir, cfg)` — the upstream function, at
the pinned revision, with upstream's own `config_ycbv.yml`. Nothing about the
reconstruction is WISEPACK's: not the sampling, not the octree, not the mesh
extraction, not the texturing. This file arranges the inputs upstream documents
and records what came out.

`use_octree` IS LEFT AT THE OFFICIAL VALUE. Turning it off would make the build
import-clean without kaolin and would silently be a different method; the point
of the exercise is the official one.

NO CAD REACHES IT. The only inputs are the reference set's rendered RGB, depth,
masks, per-view camera poses and intrinsics. No mesh path is read, constructed
or passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

#: Upstream's own configuration for this workflow, used unmodified except for
#: the save directory. Copying values out of it into a WISEPACK file would be a
#: second configuration that agrees with upstream only until one is edited.
UPSTREAM_CONFIG = "/opt/foundationpose/bundlesdf/config_ycbv.yml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True,
                        help="the reference object directory (ob_XXXXXXX)")
    parser.add_argument("--out", required=True, help="where to write the report")
    args = parser.parse_args()

    sys.path.insert(0, "/opt/foundationpose")
    sys.path.insert(0, "/opt/foundationpose/bundlesdf")

    import yaml
    import numpy as np

    with open(UPSTREAM_CONFIG, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    print(f"[model-free] upstream config: {UPSTREAM_CONFIG}", flush=True)
    print(f"[model-free] use_octree={cfg.get('use_octree')} "
          f"i_embed={cfg.get('i_embed')} "
          f"mesh_resolution={cfg.get('mesh_resolution')}", flush=True)

    from bundlesdf.run_nerf import run_one_ob                # the official seam

    started = time.time()
    mesh = run_one_ob(base_dir=args.base_dir, cfg=cfg)
    elapsed = time.time() - started

    out_file = os.path.join(args.base_dir, "model", "model.obj")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    mesh.export(out_file)

    # WHAT CAME OUT, MEASURED — not assumed from the object it was built from.
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    centred = vertices - vertices.mean(axis=0)
    _u, singular, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    along = centred @ axis
    radial = np.linalg.norm(centred - np.outer(along, axis), axis=1)

    report = {
        "ok": True,
        "upstream_config": UPSTREAM_CONFIG,
        "use_octree": cfg.get("use_octree"),
        "i_embed": cfg.get("i_embed"),
        "build_seconds": round(elapsed, 1),
        "mesh_path": out_file,
        "mesh_bytes": os.path.getsize(out_file),
        "vertices": int(len(vertices)),
        "faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        # METRES: upstream reconstructs in the object frame the reference poses
        # were given in, at real scale, because the depth it consumed was metric.
        "units": "m",
        "aabb_extents_m": [float(v) for v in extents],
        "principal_axis": [float(v) for v in axis],
        "principal_length_m": float(along.max() - along.min()),
        "transverse_diameter_m": float(2.0 * np.percentile(radial, 95)),
        "centroid_m": [float(v) for v in vertices.mean(axis=0)],
        "volume_m3": float(mesh.volume) if mesh.is_watertight else None,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[model-free] built in {elapsed:.0f}s -> {out_file}", flush=True)
    print(f"[model-free] principal length {report['principal_length_m']*1000:.1f} mm, "
          f"transverse {report['transverse_diameter_m']*1000:.1f} mm", flush=True)
    print("MODEL-FREE-BUILD OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:                                    # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("MODEL-FREE-BUILD FAILED", flush=True)
        raise SystemExit(1)
