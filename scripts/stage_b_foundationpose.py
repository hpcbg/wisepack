#!/usr/bin/env python3
"""Stage B — FoundationPose on the actual WISEPACK workcell frame.

    ./scripts/stage_b.sh          # acquire in Isaac, then run this

Takes the RGB, depth, intrinsics and exact instance mask produced by the running
`cad_cylinder5_single` workcell, sends them through the SAME WISEPACK-managed
FoundationPose worker the bolt and the reference regression use, and builds a
generic `PhysicalObservation` in the camera frame.

WHAT IS AND IS NOT USED
-----------------------
The worker receives ordinary serialised RGB-D — colour, depth, intrinsics, a
binary mask and a CAD path — exactly what the physical D435 path will send it.
It is never told where Isaac put the object.

Isaac's ground truth is used ONLY after the estimate exists, to evaluate it, and
only after being converted into the same frame FoundationPose reports in.

NO FALLBACK. If the worker fails, the failure is reported. There is no
substitution of ground truth, of an earlier result, or of planar perception:
a pose that did not come from this frame would be worse than no pose.
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

DATASET = "stage_a_workcell"
LOCAL = os.path.join(REPO, ".cache-perception", "isaac-reference", DATASET)
EVIDENCE = os.path.join(REPO, ".cache-perception", "stage-b")


def main(argv=None) -> int:
    import numpy as np
    from providers.foundationpose_rgbd import FoundationPoseProvider
    from wisepack_core.pose import (Orientation, Symmetry, axis_line_angle_deg,
                                    symmetry_aware_angle_deg)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine-iterations", type=int, default=5)
    args = parser.parse_args(argv)

    truth_path = os.path.join(LOCAL, "ground_truth.json")
    if not os.path.isfile(truth_path):
        print("No Stage A acquisition found. Acquire one first:\n"
              "  ./scripts/run_isaac_task.sh /tmp/stage_a.log 900 \\\n"
              "      simulators/isaac/stage_a_check.py", file=sys.stderr)
        return 2
    with open(truth_path, encoding="utf-8") as handle:
        truth = json.load(handle)

    os.makedirs(EVIDENCE, exist_ok=True)
    provider = FoundationPoseProvider()

    print("STAGE B — FoundationPose on the WISEPACK workcell frame")
    print(f"  acquisition  : {truth['acquisition_backend']} / "
          f"{truth['camera_profile']}")
    print(f"  mask source  : {truth['mask_source']} ({truth['mask_provenance']})")
    print(f"  object model : {truth['model_id']}  (exact CAD from the registry)")
    print()

    # ---- the estimate. NO GROUND TRUTH REACHES THIS CALL ------------------
    batch = provider.acquire_reference(
        dataset=DATASET, model_id=truth["model_id"],
        depth_scale_mm=truth["depth_scale_mm_per_unit"], frame=0,
        refine_iterations=args.refine_iterations,
        batch_id="stage-b-1")
    if not batch.ok:
        print(f"FOUNDATIONPOSE FAILED: {batch.error}", file=sys.stderr)
        print("  No fallback is applied — see §9.", file=sys.stderr)
        return 1

    observation = batch.observations[0]
    raw = provider.client.last_result()[0]
    estimate = np.asarray(raw["matrix_m"], dtype=np.float64)

    print("  PhysicalObservation")
    print(f"    frame_id            {observation.frame_id}")
    print(f"    pose_valid          {observation.pose_valid}")
    print(f"    position (mesh org) {[round(v, 2) for v in (observation.x_mm, observation.y_mm, observation.z_mm)]} mm")
    q = observation.orientation
    print(f"    quaternion (raw)    "
          f"{[round(v, 6) for v in (q.x, q.y, q.z, q.w)]}")
    print(f"    model_id            {observation.object_model_id}")
    print(f"    perception_method   {observation.perception_method}")
    print(f"    geometry            D{observation.diameter_mm} x L{observation.length_mm} mm")
    print(f"    geometric symmetry  {observation.symmetry.type.value} "
          f"fold={observation.symmetry.fold} about {observation.symmetry.axis}")
    print(f"    measured_dof        {list(observation.measured_dof)}")
    print(f"    workarea available  {observation.workarea_pose_available} "
          "(Stage C applies the camera->workarea transform)")
    print()

    # ---- evaluation. GROUND TRUTH ENTERS ONLY HERE ------------------------
    model = provider.registry.models[truth["model_id"]]
    task_axis = tuple(model.task_axis_vector or ()) or model.task_axis
    centre_mm = np.asarray(truth["model_center_mm"], dtype=np.float64)
    T_truth = np.asarray(truth["T_camera_object"], dtype=np.float64)

    def locate(T):
        """Where a pose puts the CAD reference point, in millimetres."""
        return T[:3, :3] @ centre_mm + T[:3, 3] * 1000.0

    truth_point, estimate_point = locate(T_truth), locate(estimate)
    delta = estimate_point - truth_point

    unit = (np.asarray(task_axis, dtype=np.float64)
            if not isinstance(task_axis, str)
            else np.asarray({"x": (1, 0, 0), "y": (0, 1, 0),
                             "z": (0, 0, 1)}[task_axis], dtype=np.float64))
    unit = unit / np.linalg.norm(unit)
    axis_truth = T_truth[:3, :3] @ unit
    axis_truth /= np.linalg.norm(axis_truth)
    axis_estimate = estimate[:3, :3] @ unit
    axis_estimate /= np.linalg.norm(axis_estimate)

    along = float(delta @ axis_truth)
    transverse = math.sqrt(max(0.0, float(delta @ delta) - along ** 2))
    orientation_truth = Orientation.from_matrix([r[:3] for r in T_truth[:3]])
    axis_error = axis_line_angle_deg(q, orientation_truth, task_axis)
    geometric = symmetry_aware_angle_deg(
        q, orientation_truth, Symmetry.from_dict(truth["symmetry"]))

    print("  EVALUATION against Isaac ground truth (diagnostics only)")
    print(f"    GT   reference point {np.round(truth_point, 2).tolist()} mm")
    print(f"    est  reference point {np.round(estimate_point, 2).tolist()} mm")
    print(f"    tube axis  GT  {np.round(axis_truth, 4).tolist()}")
    print(f"    tube axis  est {np.round(axis_estimate, 4).tolist()}")
    print()
    print(f"    POSITION ERROR        {np.linalg.norm(delta):8.2f} mm")
    print(f"      along the tube axis {along:8.2f} mm")
    print(f"      transverse          {transverse:8.2f} mm")
    print(f"    TUBE-AXIS-LINE ERROR  {axis_error:8.3f} deg")
    print()
    print(f"    full geometric orientation error {geometric:.3f} deg")
    print("      NOT the task metric: C5's circumferential/saddle orientation is")
    print("      weakly constrained from some views and is irrelevant to picking.")

    # ---- reprojection, against the exact mask -----------------------------
    iou_truth, centroid_truth = _reproject(T_truth, "gt")
    iou_estimate, centroid_estimate = _reproject(estimate, "estimate")
    print()
    print("  REPROJECTION onto the Stage A frame, versus the exact mask")
    print(f"    Isaac GT       IoU {iou_truth:6.4f}   centroid {centroid_truth:6.1f} px")
    print(f"    FoundationPose IoU {iou_estimate:6.4f}   centroid {centroid_estimate:6.1f} px")

    document = {
        "observation": observation.to_dict(),
        "batch": {k: v for k, v in batch.to_dict().items() if k != "observations"},
        "evaluation": {
            "position_error_mm": float(np.linalg.norm(delta)),
            "along_axis_mm": along, "transverse_mm": transverse,
            "tube_axis_line_error_deg": axis_error,
            "full_geometric_orientation_error_deg": geometric,
            "reprojection_iou_gt": iou_truth,
            "reprojection_iou_estimate": iou_estimate,
            "note": "Isaac ground truth is diagnostics only; it never entered "
                    "the perception path.",
        },
    }
    with open(os.path.join(EVIDENCE, "stage_b.json"), "w",
              encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, default=str)
    print(f"\n  wrote {EVIDENCE}/stage_b.json")
    return 0


def _reproject(T, tag):
    """Project the ORIGINAL CAD under a pose and compare to the exact mask."""
    import subprocess
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_reproject_cad.py")
    # THE OUTPUT DIRECTORY IS MOUNTED. Passing a host path to a script running
    # inside the container writes nowhere: that path does not exist in there.
    os.makedirs(EVIDENCE, exist_ok=True)
    result = subprocess.run(
        ["docker", "run", "--rm",
         "-v", "/data/jarvis/wisepack/references:/ref:ro",
         "-v", f"{LOCAL}:/frame:ro",
         "-v", f"{EVIDENCE}:/out",
         "-v", f"{script}:/tmp/r.py:ro",
         "wisepack-foundationpose:pinned", "python3", "/tmp/r.py",
         json.dumps([[float(v) for v in row] for row in T]), tag, "/out"],
        capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("REPROJECT"):
            _, iou, centroid = line.split()
            return float(iou), float(centroid)
    print(result.stdout[-500:], result.stderr[-500:], file=sys.stderr)
    return float("nan"), float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
