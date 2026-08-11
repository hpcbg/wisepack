#!/usr/bin/env python3
"""Stage B — FoundationPose on the actual WISEPACK workcell frame.

    ./scripts/stage_b.sh          # acquire a fresh frame, then run this
    ./scripts/stage_b.sh --reuse  # estimate from the last acquisition

A THIN CLI OVER `perception/simulated_rgbd_pipeline.py`. The estimate, the
evaluation and the artefact are the library's; this file decides what to print.
The dashboard's *Acquire & estimate* button calls the same functions, so the
number an operator sees in the browser is produced by the code this regression
runs — the reason the library exists at all.

WHAT IS AND IS NOT USED
-----------------------
The worker receives ordinary serialised RGB-D — colour, depth, intrinsics, a
binary mask and a CAD path — exactly what the physical D435 path sends it. It is
never told where Isaac put the object: `estimate()` is not given the scene.

Isaac's ground truth is used ONLY after the estimate exists, to evaluate it.

NO FALLBACK. If the worker fails, the failure is reported. There is no
substitution of ground truth, of an earlier result, or of planar perception: a
pose that did not come from this frame would be worse than no pose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
sys.path.insert(0, os.path.join(REPO, "perception"))

EVIDENCE = os.path.join(REPO, ".cache-perception", "stage-b")


def main(argv=None) -> int:
    import numpy as np
    from simulated_rgbd_pipeline import (SimulatedAcquisitionError,
                                         estimate as run_estimate,
                                         evaluate_camera_frame,
                                         frame_provenance, load_scene,
                                         STAGE_B_RESULT)
    from wisepack_core.rgbd import load_object_registry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine-iterations", type=int, default=5)
    args = parser.parse_args(argv)

    try:
        scene = load_scene()
    except SimulatedAcquisitionError as exc:
        print(f"{exc.reason}\n"
              "  ./scripts/run_isaac_task.sh /tmp/stage_a.log 900 \\\n"
              "      simulators/isaac/stage_a_check.py", file=sys.stderr)
        return 2

    os.makedirs(EVIDENCE, exist_ok=True)
    provenance = frame_provenance(scene)
    model_id = scene["model_id"]

    print("STAGE B — FoundationPose on the WISEPACK workcell frame")
    print(f"  acquisition  : {provenance['acquisition']} / "
          f"{provenance['acquisition_backend']} / "
          f"{provenance['camera_profile']}")
    print(f"  provenance   : {provenance['provenance']} — the FRAME is "
          "rendered; the ESTIMATOR is real")
    print(f"  mask source  : {provenance['mask_source']} "
          f"({provenance['mask_provenance']})")
    print(f"  object model : {model_id}  (exact CAD from the registry)")
    print()

    # ---- the estimate. NO GROUND TRUTH REACHES THIS CALL ------------------
    try:
        batch = run_estimate(model_id, args.refine_iterations,
                             batch_id="stage-b-1")
    except SimulatedAcquisitionError as exc:
        print(f"FOUNDATIONPOSE FAILED: {exc.reason}", file=sys.stderr)
        print("  No fallback is applied — see §9.", file=sys.stderr)
        return 1

    observation = batch.observations[0]
    print("  PhysicalObservation")
    print(f"    frame_id            {observation.frame_id}")
    print(f"    pose_valid          {observation.pose_valid}")
    print(f"    position (mesh org) "
          f"{[round(v, 2) for v in (observation.x_mm, observation.y_mm, observation.z_mm)]} mm")
    q = observation.orientation
    print(f"    quaternion (raw)    "
          f"{[round(v, 6) for v in (q.x, q.y, q.z, q.w)]}")
    print(f"    model_id            {observation.object_model_id}")
    print(f"    perception_method   {observation.perception_method}")
    print(f"    acquisition         {batch.acquisition}")
    print(f"    geometry            D{observation.diameter_mm} x "
          f"L{observation.length_mm} mm")
    print(f"    geometric symmetry  {observation.symmetry.type.value} "
          f"fold={observation.symmetry.fold} about {observation.symmetry.axis}")
    print(f"    measured_dof        {list(observation.measured_dof)}")
    print(f"    workarea available  {observation.workarea_pose_available} "
          "(Stage C applies the camera->workarea transform)")
    print()

    # ---- evaluation. GROUND TRUTH ENTERS ONLY HERE ------------------------
    model = load_object_registry(repo_root=REPO).models[model_id]
    evaluation = evaluate_camera_frame(observation, scene, model)

    print("  EVALUATION against Isaac ground truth (evaluation only)")
    print(f"    GT   reference point "
          f"{np.round(evaluation['reference_point_gt_mm'], 2).tolist()} mm")
    print(f"    est  reference point "
          f"{np.round(evaluation['reference_point_estimate_mm'], 2).tolist()} mm")
    print()
    print(f"    POSITION ERROR        {evaluation['position_error_mm']:8.2f} mm")
    print(f"      along the tube axis {evaluation['along_axis_mm']:8.2f} mm")
    print(f"      transverse          {evaluation['transverse_mm']:8.2f} mm")
    print(f"    TUBE-AXIS-LINE ERROR  "
          f"{evaluation['tube_axis_line_error_deg']:8.3f} deg")
    print()
    print("    full geometric orientation error "
          f"{evaluation['full_geometric_orientation_error_deg']:.3f} deg")
    print("      NOT the task metric: C5's circumferential/saddle orientation is")
    print("      weakly constrained from some views and is irrelevant to picking.")

    # ---- reprojection, against the exact mask -----------------------------
    truth_matrix = scene["T_camera_object"]
    estimate_matrix = _matrix_of(observation)
    iou_truth, centroid_truth = _reproject(truth_matrix, "gt")
    iou_estimate, centroid_estimate = _reproject(estimate_matrix, "estimate")
    print()
    print("  REPROJECTION onto the Stage A frame, versus the exact mask")
    print(f"    Isaac GT       IoU {iou_truth:6.4f}   centroid {centroid_truth:6.1f} px")
    print(f"    FoundationPose IoU {iou_estimate:6.4f}   centroid {centroid_estimate:6.1f} px")

    # THE LIBRARY ALREADY WROTE THE OBSERVATION AND THE EVALUATION. Only the
    # reprojection is this script's own, so only it is merged in.
    with open(STAGE_B_RESULT, encoding="utf-8") as handle:
        document = json.load(handle)
    document["evaluation"].update({
        "reprojection_iou_gt": iou_truth,
        "reprojection_iou_estimate": iou_estimate,
    })
    with open(STAGE_B_RESULT, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, default=str)
    print(f"\n  wrote {STAGE_B_RESULT}")
    return 0


def _matrix_of(observation) -> list:
    """The estimate as a 4x4, in metres — the form the reprojector takes."""
    import numpy as np
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(observation.orientation.to_matrix(),
                                dtype=np.float64)
    matrix[:3, 3] = np.asarray([observation.x_mm, observation.y_mm,
                                observation.z_mm], dtype=np.float64) / 1000.0
    return matrix.tolist()


def _reproject(T, tag):
    """Project the ORIGINAL CAD under a pose and compare to the exact mask."""
    import subprocess
    from simulated_rgbd_pipeline import FRAME

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_reproject_cad.py")
    # THE OUTPUT DIRECTORY IS MOUNTED. Passing a host path to a script running
    # inside the container writes nowhere: that path does not exist in there.
    os.makedirs(EVIDENCE, exist_ok=True)
    result = subprocess.run(
        ["docker", "run", "--rm",
         "-v", "/data/jarvis/wisepack/references:/ref:ro",
         "-v", f"{FRAME}:/frame:ro",
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
