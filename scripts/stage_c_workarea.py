#!/usr/bin/env python3
"""Stage C — camera frame to WISEPACK workarea, through the generic transform.

    ./scripts/stage_c.sh

Takes the Stage B `PhysicalObservation` — a FoundationPose estimate in
`camera_color_optical_frame` — and expresses it in `wisepack_workarea` using the
SE(3) transform derived from the Isaac scene.

WHAT IS RUNTIME AND WHAT IS DIAGNOSTIC
--------------------------------------
Runtime input is the FoundationPose estimate and nothing else. Isaac's
ground-truth object pose is never used to build the workarea observation; it is
read only afterwards, to score the result.

THE TRANSFORM IS DERIVED, NOT TYPED IN. `T_workarea_camera` is composed by the
scene from the camera prim's own pose and the layout's workarea origin. This
module reads it as data and applies it through `wisepack_core.pose.RigidTransform`
— the same abstraction the physical camera will use once a measured extrinsic
exists. Only the SOURCE of the numbers changes.

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

FRAME = os.path.join(REPO, ".cache-perception", "isaac-reference",
                     "stage_a_workcell")
STAGE_B = os.path.join(REPO, ".cache-perception", "stage-b", "stage_b.json")
EVIDENCE = os.path.join(REPO, ".cache-perception", "stage-c")


def main(argv=None) -> int:
    import numpy as np
    from wisepack_core.domain import PhysicalObservation
    from wisepack_core.pose import (CAMERA_OPTICAL_FRAME, WORKAREA_FRAME,
                                    Orientation, RigidTransform,
                                    axis_line_angle_deg)
    from wisepack_core.rgbd import load_object_registry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    for path, what in ((os.path.join(FRAME, "ground_truth.json"),
                        "Stage A acquisition"), (STAGE_B, "Stage B estimate")):
        if not os.path.isfile(path):
            print(f"No {what} found at {path}.\n"
                  "  Run ./scripts/stage_b.sh first.", file=sys.stderr)
            return 2
    with open(os.path.join(FRAME, "ground_truth.json"), encoding="utf-8") as handle:
        scene = json.load(handle)
    with open(STAGE_B, encoding="utf-8") as handle:
        stage_b = json.load(handle)

    os.makedirs(EVIDENCE, exist_ok=True)
    observation = PhysicalObservation.from_dict(stage_b["observation"])
    registry = load_object_registry()
    model = registry.models[observation.object_model_id]
    task_axis = tuple(model.task_axis_vector or ()) or model.task_axis

    print("STAGE C — camera frame -> WISEPACK workarea")
    print(f"  runtime input : FoundationPose estimate "
          f"({observation.perception_method})")
    print(f"  model         : {observation.object_model_id}")
    print()

    # ---- the transform, read from the scene as DATA -----------------------
    workarea = scene["workarea"]
    matrix = np.asarray(workarea["T_workarea_camera"], dtype=np.float64)
    transform = RigidTransform(
        parent_frame=WORKAREA_FRAME,
        child_frame=CAMERA_OPTICAL_FRAME,
        translation_mm=tuple(float(v) * 1000.0 for v in matrix[:3, 3]),
        rotation=Orientation.from_matrix([row[:3] for row in matrix[:3]]),
        method=workarea["method"],
        notes=workarea["note"])
    print("  CAMERA -> WORKAREA TRANSFORM")
    print(f"    method            {transform.method}")
    print(f"    provenance        {workarea['provenance']}")
    print(f"    {transform.child_frame} -> {transform.parent_frame}")
    print(f"    translation       "
          f"{[round(v, 2) for v in transform.translation_mm]} mm")
    r = transform.rotation
    print(f"    rotation          "
          f"{[round(v, 6) for v in (r.x, r.y, r.z, r.w)]}")
    print(f"    valid             {transform.valid} "
          "(a transform with no method is the default, not a measurement)")
    print()

    # ---- the whole pose, not just the position ----------------------------
    camera_position = (observation.x_mm, observation.y_mm, observation.z_mm)
    workarea_position = transform.apply_to_position(camera_position)
    workarea_orientation = transform.apply_to_orientation(observation.orientation)

    transformed = PhysicalObservation(
        observation_id=observation.observation_id,
        x_mm=workarea_position[0], y_mm=workarea_position[1],
        z_mm=workarea_position[2],
        object_type=observation.object_type, source=observation.source,
        frame_id=WORKAREA_FRAME,
        detector=observation.detector, model_id=observation.model_id,
        captured_at=observation.captured_at,
        calibration_status=observation.calibration_status,
        diameter_mm=observation.diameter_mm, length_mm=observation.length_mm,
        inner_diameter_mm=observation.inner_diameter_mm,
        geometry_source=observation.geometry_source
        if hasattr(observation, "geometry_source") else "",
        orientation=workarea_orientation,
        symmetry=observation.symmetry,
        perception_method=observation.perception_method,
        object_model_id=observation.object_model_id,
        model_center_mm=observation.model_center_mm,
        task_axis_vector=observation.task_axis_vector,
        pose_valid=True,
        # A VALIDATED TRANSFORM NOW EXISTS, so the pose can be placed. The
        # camera-frame estimate was already valid; this is the separate question
        # of whether it can be moved into the work area, and now it can.
        workarea_transform_valid=True,
        measured_dof=observation.measured_dof,
        confidence=observation.confidence)

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

    def axis_of(orientation):
        unit = (np.asarray(task_axis, dtype=np.float64)
                if not isinstance(task_axis, str)
                else np.asarray({"x": (1, 0, 0), "y": (0, 1, 0),
                                 "z": (0, 0, 1)}[task_axis], dtype=np.float64))
        unit = unit / np.linalg.norm(unit)
        vector = np.asarray(orientation.to_matrix(), dtype=np.float64) @ unit
        return vector / np.linalg.norm(vector)

    camera_axis = axis_of(observation.orientation)
    workarea_axis = axis_of(workarea_orientation)
    print(f"    tube axis (camera)       {np.round(camera_axis, 4).tolist()}")
    print(f"    tube axis (workarea)     {np.round(workarea_axis, 4).tolist()}")
    print()

    # ---- evaluation. GROUND TRUTH ENTERS ONLY HERE ------------------------
    #
    # Against the SETTLED pose: the tube moved while physics resolved, and the
    # scenario's requested placement is not where it is.
    settled = np.asarray(scene["settled_workarea"]["position_mm"],
                         dtype=np.float64)
    truth_matrix = np.asarray(scene["T_camera_object"], dtype=np.float64)
    centre_mm = np.asarray(scene["model_center_mm"], dtype=np.float64)

    # The estimate locates the CAD reference point; the settled pose is the
    # prim's own centre, which is that same physical point.
    # THE SAME POINT the task representation exposes — derived once, on the
    # observation, rather than recomputed here under a second convention.
    estimate_point = np.asarray(transformed.object_center, dtype=np.float64)
    delta = estimate_point - settled

    truth_orientation_camera = Orientation.from_matrix(
        [row[:3] for row in truth_matrix[:3]])
    truth_orientation_workarea = transform.apply_to_orientation(
        truth_orientation_camera)
    truth_axis = axis_of(truth_orientation_workarea)
    along = float(delta @ truth_axis)
    transverse = math.sqrt(max(0.0, float(delta @ delta) - along ** 2))
    axis_error = axis_line_angle_deg(workarea_orientation,
                                     truth_orientation_workarea, task_axis)

    print("  EVALUATION in the workarea (diagnostics only)")
    print(f"    settled Cylinder5 (actual, after physics) "
          f"{np.round(settled, 2).tolist()} mm")
    print(f"    estimated reference point                "
          f"{np.round(estimate_point, 2).tolist()} mm")
    print()
    print(f"    POSITION ERROR        {np.linalg.norm(delta):8.2f} mm")
    print(f"      along the tube axis {along:8.2f} mm")
    print(f"      transverse          {transverse:8.2f} mm")
    print(f"    TUBE-AXIS-LINE ERROR  {axis_error:8.3f} deg")
    print()
    stage_b_error = stage_b["evaluation"]["position_error_mm"]
    stage_b_axis = stage_b["evaluation"]["tube_axis_line_error_deg"]
    print(f"    Stage B (camera frame) was {stage_b_error:.2f} mm / "
          f"{stage_b_axis:.3f} deg")
    drift = abs(float(np.linalg.norm(delta)) - stage_b_error)
    print(f"    difference after the transform: {drift:.2f} mm")
    if drift > 2.0:
        print("    WARNING: the workarea error differs from the camera-frame "
              "error by more than\n    numerical roundoff. That indicates a "
              "frame convention problem, not a\n    perception problem — "
              "diagnose it rather than compensating.")

    document = {
        "camera_frame_pose": {
            "position_mm": [float(v) for v in camera_position],
            "orientation": observation.orientation.to_dict(),
        },
        "camera_frame_id": CAMERA_OPTICAL_FRAME,
        "camera_to_workarea_transform": transform.to_dict(),
        # NAMED, so neither can be mistaken for the other.
        "model_frame_pose": {
            "model_frame_origin_mm": [float(v) for v in workarea_position],
            "orientation": workarea_orientation.to_dict(),
            "note": ("the pose of the CAD model frame, as FoundationPose "
                     "reports it. Its origin can lie outside the body."),
        },
        "task_reference_point": {
            "object_center_mm": [float(v) for v in transformed.object_center],
            "tube_axis_line": [float(v) for v in workarea_axis],
            "diameter_mm": transformed.diameter_mm,
            "length_mm": transformed.length_mm,
            "inner_diameter_mm": transformed.inner_diameter_mm,
            "note": ("the physical body centre and long axis. THIS is what a "
                     "grasp targets."),
        },
        "model_frame_to_object_center_mm": float(separation),
        "workarea_frame_id": WORKAREA_FRAME,
        "workarea_pose_available": transformed.workarea_pose_available,
        "model_id": observation.object_model_id,
        "perception_method": observation.perception_method,
        "acquisition": {
            "backend": scene["acquisition_backend"],
            "camera_profile": scene["camera_profile"],
            "mask_source": scene["mask_source"],
            "mask_provenance": scene["mask_provenance"],
        },
        "task_axis_in_model_frame": (list(task_axis)
                                     if not isinstance(task_axis, str)
                                     else task_axis),
        "observation": transformed.to_dict(),
        "evaluation": {
            "compared_against": "settled Isaac pose after physics, NOT the "
                                "scenario's requested placement",
            "settled_position_mm": [float(v) for v in settled],
            "position_error_mm": float(np.linalg.norm(delta)),
            "along_axis_mm": along,
            "transverse_mm": transverse,
            "tube_axis_line_error_deg": axis_error,
            "stage_b_position_error_mm": stage_b_error,
            "difference_from_stage_b_mm": drift,
            "note": "Isaac ground truth is diagnostics only; the workarea "
                    "observation was built from the FoundationPose estimate.",
        },
    }
    with open(os.path.join(EVIDENCE, "stage_c.json"), "w",
              encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, default=str)
    print(f"\n  wrote {EVIDENCE}/stage_c.json  (Stage B result untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
