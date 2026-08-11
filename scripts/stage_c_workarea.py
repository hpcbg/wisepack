#!/usr/bin/env python3
"""Stage C — express the FoundationPose estimate in the WISEPACK workarea.

    ./scripts/stage_c.sh

A THIN CLI OVER `perception/simulated_rgbd_pipeline.py`. The transform, the
transformed observation, the evaluation and the artefact are the library's; this
file decides what to print. The dashboard's *Acquire & estimate* button calls
`run()` in that same library, so the pose it shows and the pose this prints come
from one implementation.

WHAT IS RUNTIME AND WHAT IS EVALUATION
--------------------------------------
Runtime input is the FoundationPose estimate and nothing else. Isaac's
ground-truth object pose is never used to build the workarea observation; it is
read only afterwards, to score the result.

THE TRANSFORM IS DERIVED, NOT TYPED IN. `T_workarea_camera` is composed by the
scene from the camera prim's own pose and the layout's workarea origin. The
library reads it as data and applies it through
`wisepack_core.pose.RigidTransform` — the same abstraction the physical camera
will use once a measured extrinsic exists. Only the SOURCE of the numbers
changes.

WHY THE COMPARISON IS AGAINST THE SETTLED POSE. The tube moved ~6 mm while
physics settled, so the scenario's requested placement is not where the object
is. Scoring against the request would measure the scenario; scoring against the
settled pose measures the perception.
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


def main(argv=None) -> int:
    import numpy as np
    from simulated_rgbd_pipeline import (SimulatedAcquisitionError,
                                         STAGE_B_RESULT, STAGE_C_RESULT,
                                         evaluate_workarea, load_scene,
                                         to_workarea, workarea_transform)
    from wisepack_core.domain import PhysicalObservation
    from wisepack_core.rgbd import load_object_registry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    try:
        scene = load_scene()
    except SimulatedAcquisitionError as exc:
        print(f"{exc.reason}\n  Run ./scripts/stage_b.sh first.",
              file=sys.stderr)
        return 2
    if not os.path.isfile(STAGE_B_RESULT):
        print(f"No Stage B estimate at {STAGE_B_RESULT}.\n"
              "  Run ./scripts/stage_b.sh first.", file=sys.stderr)
        return 2
    with open(STAGE_B_RESULT, encoding="utf-8") as handle:
        stage_b = json.load(handle)

    observation = PhysicalObservation.from_dict(stage_b["observation"])
    model = load_object_registry(repo_root=REPO).models[
        observation.object_model_id]

    print("STAGE C — camera frame -> WISEPACK workarea")
    print(f"  runtime input : FoundationPose estimate "
          f"({observation.perception_method})")
    print(f"  model         : {observation.object_model_id}")
    print()

    transform = workarea_transform(scene)
    if transform is None:
        print("  the exported frame carries NO camera->workarea transform, so "
              "the pose\n  stays in the camera frame. It is not relabelled.",
              file=sys.stderr)
        return 1

    print("  CAMERA -> WORKAREA TRANSFORM")
    print(f"    method            {transform.method}")
    print(f"    provenance        "
          f"{(scene.get('workarea') or {}).get('provenance', '')}")
    print(f"    {transform.child_frame} -> {transform.parent_frame}")
    print(f"    translation       "
          f"{[round(v, 2) for v in transform.translation_mm]} mm")
    r = transform.rotation
    print(f"    rotation          "
          f"{[round(v, 6) for v in (r.x, r.y, r.z, r.w)]}")
    print(f"    valid             {transform.valid} "
          "(a transform with no method is the default, not a measurement)")
    print()

    transformed = to_workarea(observation, transform)
    camera_position = (observation.x_mm, observation.y_mm, observation.z_mm)
    workarea_position = (transformed.x_mm, transformed.y_mm, transformed.z_mm)

    print("  WORKAREA OBSERVATION")
    print(f"    frame_id                 {transformed.frame_id}")
    print(f"    pose_valid               {transformed.pose_valid}")
    print(f"    workarea_pose_available  {transformed.workarea_pose_available}")
    # TWO DIFFERENT PHYSICAL POINTS, and neither is called just "position".
    print("    -- CAD model frame (what FoundationPose reports) --")
    print(f"    model-frame origin (camera)   "
          f"{[round(v, 2) for v in camera_position]} mm")
    print(f"    model-frame origin (workarea) "
          f"{[round(v, 2) for v in workarea_position]} mm")
    print("    -- task level (what a grasp targets) --")
    print(f"    OBJECT CENTRE (workarea)      "
          f"{[round(v, 2) for v in transformed.object_center]} mm")
    separation = math.sqrt(sum(
        (a - b) ** 2 for a, b in zip(transformed.object_center,
                                     workarea_position)))
    print(f"    the two differ by {separation:.2f} mm — this part's CAD origin")
    print("      lies outside its body, so a grasp must use the CENTRE.")
    print(f"    tube axis (camera)       "
          f"{np.round(observation.tube_axis or (), 4).tolist()}")
    print(f"    tube axis (workarea)     "
          f"{np.round(transformed.tube_axis or (), 4).tolist()}")
    print()

    # ---- evaluation. GROUND TRUTH ENTERS ONLY HERE ------------------------
    evaluation = evaluate_workarea(transformed, scene, model, transform)
    stage_b_error = stage_b["evaluation"]["position_error_mm"]
    stage_b_axis = stage_b["evaluation"]["tube_axis_line_error_deg"]
    drift = abs(evaluation["position_error_mm"] - stage_b_error)

    print("  EVALUATION in the workarea (evaluation only)")
    print(f"    settled Cylinder5 (actual, after physics) "
          f"{np.round(evaluation['settled_position_mm'], 2).tolist()} mm")
    print(f"    estimated reference point                "
          f"{np.round(evaluation['estimated_object_center_mm'], 2).tolist()} mm")
    print()
    print(f"    POSITION ERROR        {evaluation['position_error_mm']:8.2f} mm")
    print(f"      along the tube axis {evaluation['along_axis_mm']:8.2f} mm")
    print(f"      transverse          {evaluation['transverse_mm']:8.2f} mm")
    print(f"    TUBE-AXIS-LINE ERROR  "
          f"{evaluation['tube_axis_line_error_deg']:8.3f} deg")
    print()
    print(f"    Stage B (camera frame) was {stage_b_error:.2f} mm / "
          f"{stage_b_axis:.3f} deg")
    print(f"    difference after the transform: {drift:.2f} mm")
    if drift > 2.0:
        print("    WARNING: the workarea error differs from the camera-frame "
              "error by more than\n    numerical roundoff. That indicates a "
              "frame convention problem, not a\n    perception problem — "
              "diagnose it rather than compensating.")

    # THE LIBRARY'S `run()` WRITES THIS FILE IN THE ORDINARY FLOW. Reached
    # through the CLI, Stage B and Stage C are separate invocations, so the
    # document is assembled here from the same functions and in the same shape.
    from simulated_rgbd_pipeline import (KNOWN_TRANSFORM_NOTE,
                                         frame_provenance)
    axis = tuple(model.task_axis_vector or ()) or model.task_axis
    document = {
        "model_id": observation.object_model_id,
        "perception_method": observation.perception_method,
        "acquisition": frame_provenance(scene),
        "run_mode": "reused_frame",
        "run_label": "SIMULATED RGB-D — RE-ESTIMATED FROM THE LAST RENDER",
        "run_note": ("FoundationPose estimated from the frame Isaac rendered. "
                     "Assembled by the Stage C CLI from the same library the "
                     "dashboard calls."),
        "camera_frame_id": observation.frame_id,
        "camera_frame_pose": {
            "position_mm": [float(v) for v in camera_position],
            "orientation": observation.orientation.to_dict(),
        },
        "camera_to_workarea_transform": transform.to_dict(),
        "camera_to_workarea_note": KNOWN_TRANSFORM_NOTE,
        "model_frame_pose": {
            "model_frame_origin_mm": [float(v) for v in workarea_position],
            "orientation": transformed.orientation.to_dict(),
            "note": ("the pose of the CAD model frame, as FoundationPose "
                     "reports it. Its origin can lie outside the body."),
        },
        "task_reference_point": {
            "object_center_mm": [float(v) for v in transformed.object_center],
            "tube_axis_line": [float(v) for v in (transformed.tube_axis or ())],
            "diameter_mm": transformed.diameter_mm,
            "length_mm": transformed.length_mm,
            "inner_diameter_mm": transformed.inner_diameter_mm,
            "note": ("the physical body centre and long axis. THIS is what a "
                     "grasp targets."),
        },
        "model_frame_to_object_center_mm": float(separation),
        "workarea_frame_id": transformed.frame_id,
        "workarea_pose_available": transformed.workarea_pose_available,
        "task_axis_in_model_frame": (list(axis) if not isinstance(axis, str)
                                     else axis),
        "observation": transformed.to_dict(),
        "camera_frame_evaluation": stage_b["evaluation"],
        "evaluation": {**evaluation,
                       "stage_b_position_error_mm": stage_b_error,
                       "difference_from_stage_b_mm": drift},
    }
    os.makedirs(os.path.dirname(STAGE_C_RESULT), exist_ok=True)
    with open(STAGE_C_RESULT, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, default=str)
    print(f"\n  wrote {STAGE_C_RESULT}  (Stage B result untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
