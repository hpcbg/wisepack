#!/usr/bin/env python3
"""Render a deterministic set of QUERY observations for the model-free benchmark.

    ./scripts/run_isaac_task.sh /tmp/q.log 1800 \
        simulators/isaac/generate_model_free_queries.py [--queries 12]

WHAT THIS IS FOR. The first CAD-vs-model-free comparison used one frame. One
frame is an anecdote: it cannot separate "model-free is about twice as far off"
from "this particular view happened to suit one method". This renders a small
evaluation set so the difference can be stated with a spread rather than a
single number.

EVERY QUERY IS FRESH, AND NONE REPEATS THE REFERENCE SET. Both the object pose
and the camera pose differ from the 15 reference views: the object is moved and
re-oriented on the work surface, and the cameras sit at different distances,
azimuths and elevations. A query that duplicated a reference view would be
measuring memorisation, not generalisation.

GROUND TRUTH IS WRITTEN SOMEWHERE THE ESTIMATOR CANNOT REACH. The images go to
`queries/<id>/` and the answer goes to `ground_truth/<id>.json`, a sibling tree.
The benchmark container is given only the first. That is a mount boundary, not a
convention to remember.

EVERY QUERY'S GROUND TRUTH IS VERIFIED BEFORE IT IS WRITTEN, by projecting the
CAD mesh at the derived pose and requiring it to cover the rendered mask. An
earlier version of this file shipped ground truth that was wrong — the pose was
read several simulation steps before the frame was rendered, while the body was
still settling — and the benchmark reported that error as estimator error for
BOTH methods. The pose is now read at capture time, the body must be at rest
first, and a query whose reprojection disagrees with its own mask is discarded
rather than measured. A benchmark cannot validate its own reference; the
reference has to validate itself.

THE MATERIAL IS THE APPROVED ONE, unchanged, so the query surface matches the
surface the reference views were rendered with.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_ROOT = os.path.join(REPO, ".cache-perception", "model-free")


def _query_plan():
    """Deterministic CANDIDATE (object offset, yaw/tilt, camera spherical) poses.

    HAND-LAID RATHER THAN RANDOM, so the set is reproducible and its coverage is
    inspectable: translations spread across the usable work area, orientations
    swept through a half-turn of yaw with a few genuine tilts, and camera
    elevations from a near-side-on 30 deg to a steep 65 deg.

    MORE CANDIDATES THAN QUERIES ARE WANTED, because a candidate is rejected if
    it clips the frame, shows too little of the object, never comes to rest, or
    fails the ground-truth reprojection check. Taking the first N that pass
    keeps the requested sample size without weakening any of those tests — the
    alternative, loosening a gate to reach a count, would trade a known-good
    reference for a number.
    """
    offsets = [(0.00, 0.00), (0.08, -0.05), (-0.07, 0.06), (0.05, 0.09),
               (-0.09, -0.04), (0.11, 0.02), (-0.04, -0.10), (0.02, 0.12),
               (0.09, -0.09), (-0.11, 0.05), (0.06, 0.04), (-0.02, -0.07),
               (0.10, 0.07), (-0.06, -0.02), (0.03, -0.12), (-0.10, -0.08),
               (0.07, 0.11), (-0.03, 0.09)]
    yaws = [0, 35, 70, 105, 140, 175, 20, 55, 90,
            125, 160, 15, 45, 80, 115, 150, 30, 65]
    tilts = [0, 0, 8, 0, 12, 0, 5, 0, 15, 0, 7, 0, 0, 10, 0, 6, 0, 0]
    cams = [(35.0, 30.0, 0.62), (110.0, 45.0, 0.55), (200.0, 35.0, 0.68),
            (290.0, 55.0, 0.60), (75.0, 65.0, 0.52), (155.0, 40.0, 0.66),
            (245.0, 30.0, 0.58), (325.0, 50.0, 0.63), (15.0, 45.0, 0.70),
            (135.0, 60.0, 0.54), (215.0, 50.0, 0.61), (300.0, 35.0, 0.65),
            (60.0, 40.0, 0.64), (185.0, 55.0, 0.57), (265.0, 45.0, 0.67),
            (350.0, 35.0, 0.59), (95.0, 50.0, 0.69), (230.0, 60.0, 0.56)]
    return [{"index": i, "offset_m": offsets[i], "yaw_deg": yaws[i],
             "tilt_deg": tilts[i], "camera": cams[i]}
            for i in range(len(offsets))]


def main() -> int:
    from isaacsim import SimulationApp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--model-id", default="cylinder5")
    args = parser.parse_args()
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    try:
        return _run(app, args)
    except BaseException:                                    # noqa: BLE001
        import traceback
        print("MODEL-FREE-QUERIES FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        return 1


def _run(app, args) -> int:
    import numpy as np
    import cv2
    import isaacsim.core.experimental.utils.stage as stage_utils

    sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
    sys.path.insert(0, REPO)

    from wisepack_core.generator import build_scenario
    from wisepack_core.rgbd_sensors import sensor_profile
    from wisepack_core.isaac_transform import SceneLayout
    from simulators.isaac.config import PhysicsConfig
    from simulators.isaac.scene import WisepackScene
    from simulators.isaac.rgbd_camera import SimulatedRGBDCamera
    from simulators.isaac.materials import load_materials

    started = time.time()
    root = os.path.join(BENCH_ROOT, args.model_id, "benchmark")
    q_root, gt_root = os.path.join(root, "queries"), os.path.join(root, "ground_truth")
    # CLEARED, NOT MERGED. Query ids are positional, so a shorter run would
    # otherwise leave older queries behind under ids this run never wrote —
    # images from one generation scored against ground truth from another.
    import shutil
    for path in (q_root, gt_root):
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

    scenario = build_scenario("cad_cylinder5_single")
    layout = SceneLayout()
    scene = WisepackScene(layout, PhysicsConfig())
    scene.build(scenario, container_ids=["CNT-001"])
    item = scenario.items[0]
    material = load_materials().require()
    print(f"MODEL-FREE-QUERIES material {material.name}", flush=True)

    body = scene.items[item.item_id]
    home_p, home_q = scene.item_world_pose(item.item_id)
    offsets = scene.cad_mesh_offsets[item.item_id]
    align = np.asarray(offsets["align_to_local_z"], dtype=np.float64)
    centre_m = np.asarray(offsets["centre_m"], dtype=np.float64)

    import isaacsim.core.experimental.utils.app as app_utils
    app_utils.play()
    app.update()

    def _update(frames):
        for _ in range(frames):
            app.update()

    # SETTLE ONCE BEFORE ANY QUERY, so the first one starts from a resting body
    # rather than from the spawn pose.
    scene.settle_items(_update, frames=120)

    profile = sensor_profile("d435")
    stage = stage_utils.get_current_stage()
    camera = SimulatedRGBDCamera(
        stage, "/World/QueryCamera", profile,
        position_m=(float(home_p[0]) + 0.5, float(home_p[1]), float(home_p[2]) + 0.4),
        look_at_m=tuple(float(v) for v in home_p))
    camera.warmup()
    K = np.asarray(camera.intrinsics_matrix(), dtype=np.float64)

    def quat_mul(a, b):
        w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
        return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                         w1*x2 + x1*w2 + y1*z2 - z1*y2,
                         w1*y2 - x1*z2 + y1*w2 + z1*x2,
                         w1*z2 + x1*y2 - y1*x2 + z1*w2])

    def rotation_of(quat):
        w, x, y, z = [float(v) for v in np.asarray(quat).reshape(-1)]
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

    def object_pose_now():
        """T_world_object for the CAD frame, from the body's pose RIGHT NOW."""
        prim_p, prim_q = scene.item_world_pose(item.item_id)
        R = rotation_of(prim_q) @ align
        t = np.asarray(prim_p).reshape(-1)[:3] - R @ centre_m
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        return T, np.asarray(prim_p).reshape(-1)[:3]

    def settle_to_rest(max_rounds=12, frames=60,
                       tol_m=2e-4, tol_deg=0.20):
        """Step until the body STOPS MOVING, rather than for a fixed count.

        A FIXED SETTLE COUNT IS A GUESS. A tilted cylinder dropped onto the
        surface topples and rolls, and how long that takes depends on the pose
        it was given; 90 frames was enough for some queries and not others,
        which is exactly the kind of difference that produces ground truth
        that is right most of the time. This returns only once two consecutive
        reads agree, and reports honestly when they never do.

        THE ANGULAR TOLERANCE IS LOOSER THAN THE POSITIONAL ONE because the
        solver does not settle to a fixed quaternion: a body at a dead stop
        still reports about 0.05 deg of change between reads, so demanding
        less than that rejects resting bodies forever. It costs nothing — the
        pose used as ground truth is re-read at capture time with physics
        paused, so this test only has to establish that the body has stopped
        travelling, not to be the source of the number.
        """
        previous, _ = object_pose_now()
        for _ in range(max_rounds):
            _update(frames)
            current, _ = object_pose_now()
            moved = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
            # Rotation difference as an angle, via the trace of R_a^T R_b.
            cos = (np.trace(previous[:3, :3].T @ current[:3, :3]) - 1.0) / 2.0
            turned = np.degrees(np.arccos(float(np.clip(cos, -1.0, 1.0))))
            previous = current
            if moved <= tol_m and turned <= tol_deg:
                return True, moved, turned
        return False, moved, turned

    # THE MESH THE GROUND TRUTH WILL BE CHECKED WITH, in the same convention
    # `T_world_object` is expressed in: raw STL coordinates scaled to metres.
    # This is the geometry the scene itself was built from, read back through
    # the same registry, so the check cannot pass by using a different mesh.
    import trimesh
    from wisepack_core.rgbd import load_object_registry
    _registry = load_object_registry()
    _model = _registry.models[item.model_id]
    _mesh = trimesh.load(_model.resolved_path(_registry.root), force="mesh")
    check_vertices = (np.asarray(_mesh.vertices, dtype=np.float64)
                      * (_model.mesh_scale_to_mm / 1000.0))
    check_faces = np.asarray(_mesh.faces, dtype=np.int32)

    def reprojection_iou(T_camera_object, mask):
        """How well the CAD at the derived pose covers the RENDERED silhouette.

        THE ONE CHECK THAT CANNOT BE FOOLED BY A CONSISTENT MISTAKE. Every
        quantity here is independent of the arithmetic that produced the pose:
        the mask comes from the renderer's instance segmentation, the mesh from
        the registry, K from the camera profile. If the pose is wrong in any
        way — stale, mis-composed, wrong centre — the silhouettes separate.
        """
        cam = (T_camera_object[:3, :3] @ check_vertices.T).T + T_camera_object[:3, 3]
        if float(cam[:, 2].min()) <= 1e-6:
            return 0.0
        uv = (K @ cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        painted = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(painted, uv[check_faces].astype(np.int32), 1)
        projected = painted > 0
        union = int((projected | mask).sum())
        return float((projected & mask).sum()) / union if union else 0.0

    manifest, written, rejected = [], 0, []
    for spec in _query_plan():
        if written >= args.queries:
            break
        # --- move and re-orient the workpiece, then let physics settle -------
        dx, dy = spec["offset_m"]
        yaw, tilt = np.radians(spec["yaw_deg"]), np.radians(spec["tilt_deg"])
        q_yaw = np.array([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)])
        q_tilt = np.array([np.cos(tilt/2), np.sin(tilt/2), 0.0, 0.0])
        target_q = quat_mul(quat_mul(q_yaw, q_tilt), np.asarray(home_q, float))
        target_p = np.asarray(home_p, float) + np.array([dx, dy, 0.02])
        body.set_world_poses(positions=target_p.reshape(1, 3),
                             orientations=target_q.reshape(1, 4))
        scene.settle_items(_update, frames=60)
        at_rest, drift_m, drift_deg = settle_to_rest()
        if not at_rest:
            print(f"  q{spec['index']:02d}: skipped, never came to rest "
                  f"({drift_m*1000:.2f} mm, {drift_deg:.3f} deg still moving)",
                  flush=True)
            rejected.append({"index": spec["index"], "reason": "not_at_rest"})
            continue

        # PHYSICS IS PAUSED FOR THE CAPTURE. `rep.orchestrator.step`, which is
        # how both `warmup` and `capture` drive the renderer, advances the
        # simulation on a different schedule from `app.update` — measurably so:
        # with the body verified at rest under `app.update`, the capture steps
        # still moved it 14 to 39 mm, which is the whole of the error the first
        # benchmark attributed to the estimators. Pausing the timeline removes
        # the question instead of trying to time around it: while paused the
        # renderer still steps, and the body cannot move at all.
        # The extra update lets the paused state reach the physics stepping
        # before anything is read; without it the FIRST query of a run still
        # moved during its capture, while every later one did not.
        app_utils.pause()
        app.update()

        # THE POSE IT ACTUALLY SETTLED AT is the ground truth — not the pose it
        # was asked for. Scoring against the request would measure the request.
        T_world_ob, prim_p = object_pose_now()
        t_world_cad = T_world_ob[:3, 3]

        # --- the camera, at a viewpoint no reference view used ---------------
        az, el, radius = spec["camera"]
        a, e = np.radians(az), np.radians(el)
        eye = t_world_cad + radius * np.array(
            [np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
        if eye[2] <= layout.table_top_z_m + 0.05:
            print(f"  q{spec['index']:02d}: skipped, camera below the surface", flush=True)
            rejected.append({"index": spec["index"], "reason": "camera_below_surface"})
            app_utils.play()
            continue
        camera._place(camera.camera, tuple(float(v) for v in eye),
                      tuple(float(v) for v in t_world_cad))
        camera.warmup(6)
        frame = camera.capture()

        # THE POSE IS READ AGAIN, AFTER THE FRAME EXISTS, and the two reads must
        # agree. With the timeline paused they should agree exactly; this stays
        # because the pause is the reason they agree, and a check that only
        # passes while an assumption holds is worth keeping for the day the
        # assumption stops holding.
        T_after, prim_p = object_pose_now()
        shifted = float(np.linalg.norm(T_after[:3, 3] - T_world_ob[:3, 3]))
        app_utils.play()
        if shifted > 2e-4:
            print(f"  q{spec['index']:02d}: skipped, moved {shifted*1000:.2f} mm "
                  "between the pose read and the frame", flush=True)
            rejected.append({"index": spec["index"], "reason": "moved_during_capture"})
            continue
        T_world_ob, t_world_cad = T_after, T_after[:3, 3]
        try:
            mask = camera.mask_for(frame, args.model_id)
        except ValueError as exc:
            print(f"  q{spec['index']:02d}: skipped, {exc}", flush=True)
            rejected.append({"index": spec["index"], "reason": f"mask: {exc}"})
            continue
        # FAIR TEST, NOT AN EASY ONE: enough of the object must be visible for
        # the comparison to mean something, and nothing is cropped by the frame.
        if int(mask.sum()) < 2000:
            print(f"  q{spec['index']:02d}: skipped, only {int(mask.sum())} px", flush=True)
            rejected.append({"index": spec["index"], "reason": "too_few_pixels"})
            continue
        if (mask[0, :].any() or mask[-1, :].any()
                or mask[:, 0].any() or mask[:, -1].any()):
            print(f"  q{spec['index']:02d}: skipped, clipped by the frame", flush=True)
            rejected.append({"index": spec["index"], "reason": "clipped_by_frame"})
            continue

        T_camera_object = np.asarray(frame["camera_from_world"],
                                     dtype=np.float64) @ T_world_ob

        # THE GATE. A query is only usable if its own ground truth reprojects
        # onto its own mask. The threshold is high on purpose: the two
        # silhouettes come from the same geometry seen through the same camera,
        # so agreement should be limited by rasterisation and by the renderer's
        # anti-aliased mask edge, not by anything in the pose. The earlier,
        # defective ground truth scored a mean of 0.49 here.
        iou = reprojection_iou(T_camera_object, mask)
        if iou < 0.90:
            print(f"  q{spec['index']:02d}: REJECTED, ground truth reprojects "
                  f"at IoU {iou:.3f} against its own mask", flush=True)
            rejected.append({"index": spec["index"],
                             "reason": "ground_truth_reprojection",
                             "reprojection_iou": round(iou, 4)})
            continue

        qid = f"q{written:02d}"
        qdir = os.path.join(q_root, qid)
        for sub in ("rgb", "depth", "masks"):
            os.makedirs(os.path.join(qdir, sub), exist_ok=True)
        cv2.imwrite(f"{qdir}/rgb/000000.png",
                    cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{qdir}/depth/000000.png", frame["depth_mm"].astype(np.uint16))
        cv2.imwrite(f"{qdir}/masks/000000.png", (mask.astype(np.uint8) * 255))
        np.savetxt(f"{qdir}/cam_K.txt", K, fmt="%.18e")

        # --- THE ANSWER, in a sibling tree the estimator is never given ------
        with open(os.path.join(gt_root, f"{qid}.json"), "w", encoding="utf-8") as h:
            json.dump({"model_id": args.model_id,
                       "T_camera_object": T_camera_object.tolist(),
                       "model_center_mm": [float(v) for v in centre_m * 1000.0],
                       "reprojection_iou_against_own_mask": round(iou, 4),
                       "frame_id": "camera_color_optical_frame",
                       "note": "Simulator ground truth, read at capture time "
                               "from a body verified at rest and checked by "
                               "reprojection against this query's own mask. "
                               "Written OUTSIDE the query tree so the "
                               "estimator cannot read it."}, h, indent=2)

        manifest.append({
            "id": qid, "object_offset_m": [dx, dy],
            "object_yaw_deg": spec["yaw_deg"], "object_tilt_deg": spec["tilt_deg"],
            "camera_azimuth_deg": az, "camera_elevation_deg": el,
            "camera_radius_m": radius, "object_pixels": int(mask.sum()),
            "settled_position_m": [float(v) for v in prim_p],
            "reprojection_iou": round(iou, 4),
        })
        written += 1
        print(f"  {qid}: yaw {spec['yaw_deg']:3d} tilt {spec['tilt_deg']:2d} "
              f"cam az {az:5.1f} el {el:4.1f} r {radius:.2f}  "
              f"{int(mask.sum()):6d} px  GT IoU {iou:.3f}", flush=True)

    with open(os.path.join(root, "query_manifest.json"), "w", encoding="utf-8") as h:
        json.dump({
            "source": "isaac_simulated",
            "purpose": "foundationpose_model_free_pose_benchmark",
            "model_id": args.model_id,
            "material_profile": material.name,
            "query_count": written,
            "intrinsics": K.tolist(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generation_seconds": round(time.time() - started, 1),
            "ground_truth_location": "ground_truth/ (a sibling of queries/)",
            "ground_truth_note": (
                "Kept out of the query tree so the benchmark container, which "
                "mounts only queries/, cannot read it."),
            "distinct_from_reference_set": True,
            "ground_truth_verified_by_reprojection": True,
            "ground_truth_minimum_reprojection_iou": 0.90,
            "rejected": rejected,
            "queries": manifest}, h, indent=2)
    ious = [q["reprojection_iou"] for q in manifest]
    print(f"\nMODEL-FREE-QUERIES {written}/{args.queries} in "
          f"{time.time()-started:.0f}s -> {root}", flush=True)
    if ious:
        print(f"MODEL-FREE-QUERIES ground-truth reprojection IoU: "
              f"min {min(ious):.3f} mean {sum(ious)/len(ious):.3f} "
              f"max {max(ious):.3f}", flush=True)
    if rejected:
        print(f"MODEL-FREE-QUERIES rejected {len(rejected)}: "
              + ", ".join(f"#{r['index']} {r['reason']}" for r in rejected),
              flush=True)
    print("WROTE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
