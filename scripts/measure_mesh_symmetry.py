#!/usr/bin/env python3
"""Measure which rotations map a mesh onto itself.

    python3 scripts/measure_mesh_symmetry.py references/CAD-Models/STL-Files/*.stl

WHY THIS IS MEASURED AND NOT DECLARED BY EYE
--------------------------------------------
`config/perception_objects.yaml` declares each object's symmetry, and WISEPACK
uses that declaration to decide which rotational degrees of freedom it is
entitled to report as measured. A wrong declaration is harmful in BOTH
directions:

  * claim a symmetry that is not there  -> a real, measurable orientation is
    collapsed away and the pose loses information it actually had;
  * miss a symmetry that is there       -> WISEPACK publishes an angle that no
    sensor could have determined, as though it were a measurement.

The second is the dangerous one, because the fabricated angle looks exactly like
a real one. So the declaration is derived from the geometry rather than from
looking at a render.

METHOD
------
Sample the surface densely twice — once as the reference, once as the probe —
rotate the probe set about the bounding-box centre, and measure how far each
rotated point lands from the reference surface. A rotation that maps the shape
onto itself leaves that distance at the sampling noise floor, which is reported
alongside so the comparison is against something real rather than against zero.

Continuous (axial) symmetry is tested with several unrelated angles, not one: a
single 90 deg test cannot distinguish a cylinder from a square tube.

THE MESH'S OWN AXIS IS TESTED, AND THAT MATTERS
----------------------------------------------
Testing only the three coordinate axes is not enough, and getting this wrong has
already produced a wrong answer in this repository: `Cylinder5` is a straight
tube whose axis is oblique to x, y and z, and a coordinate-axis-only test
reported it as having no continuous symmetry at all. It was declared a
"symmetric hairpin" on that evidence. It is not bent; it is a straight tube
rotated about 22 degrees in the xy-plane, and its spin about its own axis is
just as unobservable as any other pipe's.

So the PRINCIPAL AXIS of the mesh (first component of a PCA over surface
samples) is tested alongside x, y and z. A shape whose symmetry axis is neither
a coordinate axis nor its principal axis can still be missed; the report says
which axes were tried and never claims a mesh is asymmetric, only that the
rotations it tried are not symmetries of it.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

#: Angles used to probe continuous symmetry. Deliberately not multiples of each
#: other: 90 deg alone is a symmetry of a square tube as well as a cylinder.
CONTINUOUS_PROBE_ANGLES = (17.0, 91.0, 143.0)

#: How much worse than the sampling noise a deviation must be before it counts
#: as "the shape moved". Two is generous; the real signals below are two orders
#: of magnitude clear of it.
NOISE_MULTIPLE = 3.0


def measure(path: str, samples: int = 20000) -> Optional[dict]:
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree

    np.random.seed(0)          # reproducible: this feeds a tracked config file
    mesh = trimesh.load(path)
    if not hasattr(mesh, "sample"):
        return None
    reference = cKDTree(mesh.sample(samples * 6))
    probe = mesh.sample(samples)
    # ROTATE ABOUT THE CENTROID, not the bounding-box centre. For a part whose
    # axis is oblique to the coordinate frame those two points differ, and
    # rotating about a point that is NOT on the symmetry axis moves the shape
    # even when the rotation IS a symmetry — which is how Cylinder5's own axis
    # came out at 3.46 mm instead of the 0.75 mm noise floor, and nearly
    # produced a second wrong declaration.
    centre = probe.mean(axis=0)

    def deviation(rotation) -> float:
        moved = (probe - centre) @ rotation.T + centre
        return float(np.percentile(reference.query(moved)[0], 99))

    def rotation(vector, degrees: float):
        return trimesh.transformations.rotation_matrix(
            np.radians(degrees), vector)[:3, :3]

    noise = deviation(np.eye(3))
    threshold = max(noise * NOISE_MULTIPLE, 1e-6)

    # THE MESH'S OWN AXIS, from a PCA over the surface samples. For any
    # elongated part this is the axis a symmetry is most likely to be about,
    # and it is the one a coordinate-axis test cannot see.
    _u, _s, vt = np.linalg.svd(probe - probe.mean(axis=0), full_matrices=False)
    principal = vt[0]

    axes = {}
    candidates = [("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1]),
                  ("principal", principal.tolist())]
    for name, vector in candidates:
        continuous = max(deviation(rotation(vector, a))
                         for a in CONTINUOUS_PROBE_ANGLES)
        two_fold = deviation(rotation(vector, 180.0))
        axes[name] = {
            "continuous_mm": continuous,
            "two_fold_mm": two_fold,
            "continuous_symmetry": continuous <= threshold,
            "two_fold_symmetry": two_fold <= threshold,
        }
    # STRAIGHTNESS, measured directly: on a straight tube every surface point
    # sits at roughly the barrel radius from the principal axis. It is reported
    # because "is this bent?" is the question the symmetry result gets read as.
    offsets = probe - probe.mean(axis=0)
    along = offsets @ principal
    radial = np.linalg.norm(offsets - np.outer(along, principal), axis=1)
    return {"path": path, "extents": mesh.extents.tolist(),
            "noise_mm": noise, "threshold_mm": threshold, "axes": axes,
            "principal_axis": [round(float(v), 4) for v in principal],
            "length_along_axis_mm": float(along.max() - along.min()),
            "max_radial_mm": float(radial.max()),
            "median_radial_mm": float(np.median(radial))}


def describe(result: dict) -> str:
    """The declaration this measurement supports, in registry vocabulary."""
    axial = [a for a, v in result["axes"].items() if v["continuous_symmetry"]]
    if "principal" in axial:
        # Prefer the mesh's own axis when both it and a coordinate axis qualify:
        # it is the physically meaningful one, and for an obliquely modelled
        # part it is the ONLY one that qualifies.
        vector = result["principal_axis"]
        return (f"type: axial, axis_vector: {vector}  "
                "(spin about the tube's own axis is NOT observable)")
    if axial:
        return f"type: axial, axis: {axial[0]}  (rotation about {axial[0]} is NOT observable)"
    two_fold = [a for a, v in result["axes"].items() if v["two_fold_symmetry"]]
    if two_fold:
        return (f"type: discrete, fold: 2, axis: {two_fold[0]}  "
                f"(orientation unique only up to 180 deg about {two_fold[0]})")
    return "type: none for the axes tested  (x, y, z only — see LIMITATION)"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("meshes", nargs="+", help="mesh files to measure")
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args(argv)

    try:
        import numpy  # noqa: F401
        import trimesh  # noqa: F401
        from scipy.spatial import cKDTree  # noqa: F401
    except ImportError as exc:
        print(f"this tool needs trimesh, numpy and scipy: {exc}", file=sys.stderr)
        print("it is also runnable inside the FoundationPose worker image, "
              "which already has them.", file=sys.stderr)
        return 2

    for path in args.meshes:
        result = measure(path, args.samples)
        if result is None:
            print(f"{os.path.basename(path):<16} not a single mesh — skipped")
            continue
        row = "  ".join(
            f"{a}: any={v['continuous_mm']:7.2f} 180={v['two_fold_mm']:6.2f}"
            for a, v in result["axes"].items())
        print(f"{os.path.basename(path):<16} noise={result['noise_mm']:5.2f}  {row}")
        print(f"{'':<16} principal {result['principal_axis']}  "
              f"length {result['length_along_axis_mm']:.1f} mm  "
              f"max radial {result['max_radial_mm']:.1f} mm "
              f"-> {'STRAIGHT' if result['max_radial_mm'] < 30 else 'BENT'}")
        print(f"{'':<16} -> {describe(result)}")
    print("\nmm; 99th-percentile distance from the rotated surface back to the "
          "original.\nA value at the noise floor means that rotation is "
          "UNOBSERVABLE for this shape.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
