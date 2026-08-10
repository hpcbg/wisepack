#!/usr/bin/env python3
"""Run FoundationPose against an Isaac reference case and measure the error.

    python3 scripts/isaac_reference_regression.py --model-id cylinder5

WHAT MAKES THIS DIFFERENT FROM THE BOLT REGRESSION
--------------------------------------------------
The bolt dataset has no ground-truth pose, so it can only report repeatability
and plausibility. An Isaac case knows exactly where it put the object, so this
reports a real error:

    translation error   in millimetres
    orientation error   MODULO the object's declared symmetry

The second qualifier is not a detail. Every WISEPACK pipe section is a straight
tube, so its spin about its own axis was never observable; comparing an
estimated spin against a "true" spin measures nothing and would make a perfect
estimate look wrong. `symmetry_aware_angle_deg` discards exactly the degrees of
freedom the shape makes unobservable and no others.

STILL NOT ABSOLUTE ACCURACY ON REAL DATA. This is a rendering: no sensor noise,
no real depth artefacts, no real materials. It measures whether FoundationPose
recovers a pose WISEPACK already knows, which is the thing the bolt cannot do —
and is not a substitute for live RealSense validation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
sys.path.insert(0, os.path.join(REPO, "perception"))

DEFAULT_ROOT = os.path.join(REPO, ".cache-perception", "isaac-reference")


def main(argv=None) -> int:
    from wisepack_core.foundationpose_client import FoundationPoseClient
    from wisepack_core.pose import (Orientation, Symmetry, axis_line_angle_deg,
                                    symmetry_aware_angle_deg)
    from wisepack_core.rgbd import load_object_registry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="cylinder5")
    parser.add_argument("--dataset-root", default=DEFAULT_ROOT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--refine-iterations", type=int, default=5)
    args = parser.parse_args(argv)

    root = os.path.join(args.dataset_root, args.model_id)
    truth_path = os.path.join(root, "ground_truth.json")
    if not os.path.isfile(truth_path):
        print(f"no reference case at {root}. Generate it first:\n"
              f"  ./scripts/run_isaac_task.sh /tmp/gen.log 540 "
              f"simulators/isaac/generate_reference_dataset.py --model-id {args.model_id}",
              file=sys.stderr)
        return 2
    with open(truth_path, encoding="utf-8") as handle:
        truth = json.load(handle)

    client = FoundationPoseClient()
    usable, reason = client.capability()
    if not usable:
        print(f"the FoundationPose worker cannot estimate: {reason}", file=sys.stderr)
        return 1

    # The worker reads datasets from its read-only mount, so the case must be
    # reachable there. Named RELATIVE to that mount, never as a host path.
    relative = os.environ.get("WISEPACK_FP_ISAAC_DATASET",
                              f"isaac-reference/{args.model_id}")
    request = {
        "dataset": relative,
        "mesh_path": truth["mesh_path"],
        "mesh_scale_to_metres": {"mm": 0.001, "m": 1.0}[truth["mesh_units"]],
        "depth_scale_mm": truth["depth_scale_mm_per_unit"],
        "frame": 0,
        "refine_iterations": args.refine_iterations,
    }

    results = []
    for _ in range(max(1, args.repeats)):
        result, error = client.estimate(request)
        if result is None:
            print(f"estimate failed: {error}", file=sys.stderr)
            return 1
        results.append(result)

    symmetry = Symmetry.from_dict(truth.get("symmetry"))
    matrix = truth["T_camera_object"]
    truth_orientation = Orientation.from_matrix(
        [row[:3] for row in matrix[:3]])

    # POSITION IS COMPARED AT A POINT ON THE OBJECT, not at the mesh origin.
    #
    # A pose is a frame, and "translation error" is only meaningful once you say
    # WHICH POINT of the object you are locating. For a part drawn obliquely the
    # STL origin can sit far off the body — Cylinder5's is 98.8 mm off the tube
    # axis — and measuring there turns rotation about the axis into apparent
    # translation. That is not an estimator error; it is the wrong measuring
    # point. Both poses project onto the rendered object correctly (IoU 0.92 and
    # 0.87), which is the proof that neither frame is wrong.
    #
    # The AABB centre lies ON the axis, so spin cannot move it, and it is
    # FoundationPose's own `model_center` — the same definition on both sides,
    # not a compensation.
    centre_mm = truth.get("model_center_mm") or [0.0, 0.0, 0.0]

    def locate(rotation, translation_m):
        """Where a given pose puts the object's reference point, in MILLIMETRES.

        Everything here is millimetres: `centre_mm` already is, and the pose's
        translation is metres and is converted once. An earlier version rotated
        a metre-scale vector and added it to a millimetre translation, which
        silently reduced the reference point to almost nothing.
        """
        return [sum(rotation[r][c] * centre_mm[c] for c in range(3))
                + translation_m[r] * 1000.0 for r in range(3)]

    truth_rotation = [row[:3] for row in matrix[:3]]
    truth_position = locate(truth_rotation,
                            [matrix[0][3], matrix[1][3], matrix[2][3]])

    print(f"ISAAC REFERENCE REGRESSION — {args.model_id}")
    print(f"  symmetry declared: {symmetry.type.value}"
          + (f" fold {symmetry.fold}" if symmetry.fold else "")
          + f" about {symmetry.axis}")
    print(f"  ambiguous DoF    : {symmetry.ambiguous_dof or ['none']}")
    print()
    print(f"  reference point  : object AABB centre "
          f"{[round(v, 1) for v in centre_mm]} mm in the STL frame")
    print(f"  ground truth     : position {[round(v, 2) for v in truth_position]} mm")

    # THE TASK-LEVEL AXIS, from the registry. Separate from the geometric
    # symmetry above, and the primary number for a picking task.
    model = load_object_registry().models.get(args.model_id)
    task_axis = getattr(model, "task_axis", "z") if model else "z"
    # A measured vector wins over a coordinate-axis name whenever one exists.
    vector = tuple(getattr(model, "task_axis_vector", ()) or ()) if model else ()
    if vector:
        task_axis = vector
    task_equivalence = (getattr(model, "task_pose_equivalence", "exact")
                        if model else "exact")
    print(f"  task equivalence : {task_equivalence} about {task_axis}")
    print()

    translations, orientations, raw_orientations, axis_errors = [], [], [], []
    along_errors, across_errors = [], []
    for index, result in enumerate(results):
        quaternion = result["orientation"]
        estimated = Orientation(x=quaternion["x"], y=quaternion["y"],
                                z=quaternion["z"], w=quaternion["w"])
        # The SAME reference point, located by the estimated pose.
        estimated_matrix = result["matrix_m"]
        position = locate([row[:3] for row in estimated_matrix[:3]],
                          [estimated_matrix[0][3], estimated_matrix[1][3],
                           estimated_matrix[2][3]])
        delta = [p - t for p, t in zip(position, truth_position)]
        translation_error = math.sqrt(sum(d * d for d in delta))
        modulo = symmetry_aware_angle_deg(estimated, truth_orientation, symmetry)
        plain = estimated.angle_to_deg(truth_orientation)
        axis_error = axis_line_angle_deg(estimated, truth_orientation, task_axis)
        # ALONG vs ACROSS the tube: sliding a straight tube along its own length
        # is the weakly-constrained direction, and separating it says which kind
        # of error this is.
        unit = task_axis if not isinstance(task_axis, str) else {
            "x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0)}[task_axis]
        length = math.sqrt(sum(v * v for v in unit))
        unit = [v / length for v in unit]
        axis_cam = [sum(truth_rotation[r][c] * unit[c] for c in range(3))
                    for r in range(3)]
        along = sum(d * a for d, a in zip(delta, axis_cam))
        across = math.sqrt(max(0.0, translation_error ** 2 - along ** 2))
        translations.append(translation_error)
        orientations.append(modulo)
        raw_orientations.append(plain)
        axis_errors.append(axis_error)
        along_errors.append(along)
        across_errors.append(across)
        print(f"  run {index}: position {translation_error:6.2f} mm "
              f"(along {along:6.2f}, transverse {across:5.2f})"
              f"  |  AXIS-LINE {axis_error:6.3f} deg"
              f"  |  geometric {modulo:7.3f} deg")

    print()
    print("  -- TASK-LEVEL (what picking needs) --------------------------------")
    print("  AXIS-LINE ERROR     "
          f"mean {sum(axis_errors)/len(axis_errors):8.3f} deg  "
          f"min {min(axis_errors):8.3f}   max {max(axis_errors):8.3f}")
    print("  acos(|dot(axis_est, axis_gt)|): blind to circumferential spin and")
    print("  to an A1/A2 end swap, neither of which affects picking this tube.")
    print()
    print("  POSITION ERROR      "
          f"mean {sum(translations)/len(translations):8.2f} mm   "
          f"along-axis {sum(along_errors)/len(along_errors):7.2f}   "
          f"transverse {sum(across_errors)/len(across_errors):6.2f}")
    print()
    print("  These two are the ASSERTABLE regression metrics: measured stable")
    print("  across repeated runs, fresh processes and render noise.")
    print()
    print("  -- GEOMETRIC (what the CAD is) ------------------------------------")
    print("  NOT a regression assertion. Cylinder5's circumferential spin is")
    print("  constrained only by its saddle ends, weakly from this view: RGB")
    print("  render noise alone moves it by ~108 deg while the tube axis moves")
    print("  by 0.006 deg. See REFERENCE_ASSETS.md.")
    print("  ORIENTATION ERROR   "
          f"mean {sum(orientations)/len(orientations):8.3f} deg  "
          f"min {min(orientations):8.3f}   max {max(orientations):8.3f}"
          "   (modulo declared symmetry)")
    if symmetry.ambiguous_dof:
        print("  The raw angle below IGNORES the symmetry and is NOT an error: "
              "it includes\n  rotations the shape makes unobservable, and is "
              "shown only for contrast.")
        print(f"  raw angle           "
              f"mean {sum(raw_orientations)/len(raw_orientations):8.3f} deg")
    print()
    print("  Rendered ground truth: exact by construction, no sensor noise. "
          "NOT a\n  measurement of real-world accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
