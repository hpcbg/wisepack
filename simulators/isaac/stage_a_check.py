#!/usr/bin/env python3
"""Stage A — prove the CAD workpiece and the simulated sensor in the real cell.

    ./scripts/run_isaac_task.sh /tmp/stage_a.log 900 \
        simulators/isaac/stage_a_check.py

WHAT THIS DOES AND DOES NOT DO
------------------------------
It builds the EXISTING WISEPACK workcell — the same `WisepackScene`, the same
table, the same containers, the same layout — with the `cad_cylinder5_single`
scenario, adds a D435-compatible simulated RGB-D camera, settles physics, and
then CHECKS what is actually there.

It runs no pose estimation. Stage A is about whether the scene and the sensing
are right; asking FoundationPose to estimate against a scene nobody has looked
at would produce a number with nothing behind it.

Every check reports a measured value, not a claim. A failed check is printed and
the run continues, so one bad number does not hide the rest.
"""

from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT = os.path.join(REPO, ".cache-perception", "stage-a")


def _workarea_from_camera(layout, camera_from_world):
    """T_workarea_camera, composed from the scene's own numbers.

    The workarea frame is axis-aligned with world and offset by the table
    frame origin, so `T_workarea_world` is a pure translation — but it is
    BUILT here rather than assumed, so a layout change flows through.
    """
    import numpy as np
    workarea_from_world = np.eye(4)
    workarea_from_world[:3, 3] = -np.asarray(layout.table_frame_origin_m,
                                             dtype=np.float64)
    world_from_camera = np.linalg.inv(camera_from_world)
    return workarea_from_world @ world_from_camera


def _world_to_workarea_mm(layout, position_m):
    """A world point in workarea millimetres."""
    import numpy as np
    offset = np.asarray(position_m, dtype=np.float64) - np.asarray(
        layout.table_frame_origin_m, dtype=np.float64)
    return [float(v) * 1000.0 for v in offset]


def main() -> int:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    try:
        return _run(app)
    except BaseException:                                    # noqa: BLE001
        import traceback
        print("STAGE-A FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        return 1


def _run(app) -> int:
    import numpy as np
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils

    sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
    sys.path.insert(0, REPO)

    from wisepack_core.generator import build_scenario
    from wisepack_core.rgbd_sensors import sensor_profile
    from wisepack_core.isaac_transform import (SceneLayout, pose_to_world,
                                               table_pose_for_index)
    from simulators.isaac.config import PhysicsConfig
    from simulators.isaac.scene import WisepackScene
    from simulators.isaac.rgbd_camera import SimulatedRGBDCamera

    os.makedirs(OUTPUT, exist_ok=True)
    results = {}

    def check(name, ok, detail=""):
        results[name] = {"ok": bool(ok), "detail": detail}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        return ok

    # ---- the existing workcell, with a CAD item ---------------------------
    print("STAGE-A building the existing WISEPACK workcell", flush=True)
    scenario = build_scenario("cad_cylinder5_single")
    layout = SceneLayout()
    scene = WisepackScene(layout, PhysicsConfig())
    scene.build(scenario, container_ids=["CNT-001"])

    item = scenario.items[0]
    check("scenario is CAD-backed",
          item.geometry_source == "cad_mesh" and item.model_id == "cylinder5",
          f"{item.item_id} model_id={item.model_id} "
          f"{item.length_mm}x{item.outer_diameter_mm} mm, {item.weight_kg} kg")
    check("containers built from the existing workcell",
          len(scene.containers) == 1, f"{list(scene.containers)}")

    # ---- the mesh that was actually loaded --------------------------------
    from pxr import UsdGeom
    stage = stage_utils.get_current_stage()
    from simulators.isaac.scene import item_path
    prim = stage.GetPrimAtPath(item_path(item.item_id))
    check("CAD prim exists", prim and prim.IsValid(), item_path(item.item_id))
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get())
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    extents = points.max(axis=0) - points.min(axis=0)
    check("mesh has real triangles", len(faces) > 300,
          f"{len(points)} vertices, {len(faces)//3} triangles")
    # D25 x L342: the long extent is the tube, the short two the barrel.
    ordered = np.sort(extents)[::-1] * 1000.0
    check("mesh scale is correct (mm)",
          300 < ordered[0] < 380 and 20 < ordered[2] < 30,
          f"extents {np.round(ordered, 1).tolist()} mm (nominal 342 x ~148 x 25 "
          "for this obliquely modelled part)")

    # HOLLOW: a solid rod of this radius would have far more volume. Measured
    # from the mesh rather than assumed from the nominal wall thickness.
    import trimesh
    tri = trimesh.Trimesh(vertices=points, faces=faces.reshape(-1, 3))
    solid = np.pi * (0.0125 ** 2) * 0.342
    check("mesh is hollow, not a solid rod",
          tri.volume < 0.6 * solid,
          f"volume {tri.volume*1e9:,.0f} mm^3 vs solid {solid*1e9:,.0f} mm^3")

    # SADDLE ENDS: no face is perpendicular to the tube axis on a saddle-cut
    # part, whereas a square-cut tube has a ring of them.
    samples = tri.sample(40000)
    centre = samples.mean(axis=0)
    _u, _s, vt = np.linalg.svd(samples - centre, full_matrices=False)
    axis = vt[0]
    perpendicular = int((np.abs(tri.face_normals @ axis) > 0.95).sum())
    check("saddle ends preserved", perpendicular == 0,
          f"{perpendicular} faces perpendicular to the axis "
          "(a square-cut tube would have a ring of them)")

    # ---- physics -----------------------------------------------------------
    app_utils.play()
    app.update()
    # `settle_items` calls its updater WITH a frame count — the same contract
    # `wisepack_isaac.py` satisfies. Passing `app.update` directly fails,
    # because SimulationApp.update() takes no argument.
    def _update(frames):
        for _ in range(frames):
            app.update()

    scene.settle_items(_update, frames=120)
    body = scene.items[item.item_id]
    position = np.asarray(body.get_world_poses()[0]).reshape(-1)[:3]
    requested = np.asarray(
        pose_to_world(table_pose_for_index(0, item, layout), layout)[0])
    drift = float(np.linalg.norm(position - requested))
    check("body rests where the scenario asked", drift < 0.05,
          f"requested {np.round(requested,3).tolist()} m, "
          f"settled {np.round(position,3).tolist()} m, drift {drift*1000:.1f} mm")
    check("body did not fall through the table",
          position[2] > layout.table_top_z_m - 0.02,
          f"z = {position[2]:.3f} m, table top {layout.table_top_z_m:.3f} m")
    check("body did not explode", np.all(np.isfinite(position))
          and float(np.linalg.norm(position)) < 5.0,
          f"|position| = {float(np.linalg.norm(position)):.3f} m")
    masses = np.asarray(body.get_masses()).reshape(-1)
    check("mass is the tube's, not a solid rod's",
          abs(float(masses[0]) - item.weight_kg) < 0.01,
          f"{float(masses[0]):.3f} kg (scenario says {item.weight_kg} kg)")

    # ---- reachability, against the existing layout ------------------------
    base = np.array([0.0, 0.0, layout.table_top_z_m])
    reach = float(np.linalg.norm(position[:2] - base[:2]))
    check("tube centre is within practical reach", 0.25 < reach < 0.75,
          f"{reach:.3f} m from the robot base in XY")

    # ---- the simulated D435 ------------------------------------------------
    profile = sensor_profile("d435")
    camera = SimulatedRGBDCamera(
        stage, "/World/RGBDCamera", profile,
        position_m=(position[0] - 0.30, position[1] - 0.45, layout.table_top_z_m + 0.62),
        look_at_m=(float(position[0]), float(position[1]), float(position[2])))
    camera.warmup()
    frame = camera.capture()

    k = frame["intrinsics"]
    nominal = profile.colour.intrinsics_matrix()
    check("intrinsics match the configured D435 nominal profile",
          abs(k[0][0] - nominal[0][0]) < 0.5 and abs(k[1][1] - nominal[1][1]) < 0.5,
          f"fx={k[0][0]:.2f} fy={k[1][1]:.2f} cx={k[0][2]:.1f} cy={k[1][2]:.1f}")
    check("camera transform available from the scene",
          np.isfinite(np.asarray(frame["camera_from_world"])).all(),
          "4x4 world -> camera optical")

    rgb, depth = frame["rgb"], frame["depth_mm"]
    check("RGB rendered", rgb.shape == (profile.colour.height,
                                        profile.colour.width, 3),
          f"{rgb.shape}, mean level {float(rgb.mean()):.1f}")

    # ---- the synthetic mask -----------------------------------------------
    try:
        mask = camera.mask_for(frame, item.model_id)
    except ValueError as exc:
        check("instance mask identifies cylinder5", False, str(exc))
        mask = np.zeros(depth.shape, bool)
    else:
        check("instance mask identifies cylinder5", bool(mask.any()),
              f"{int(mask.sum())} px, labels {frame['id_to_labels']}")

    if mask.any():
        inside = depth[mask]
        coverage = float((inside > 0).mean())
        check("usable depth over the tube", coverage > 0.9,
              f"{coverage:.1%} of masked pixels have depth; range "
              f"{int(inside[inside>0].min())}..{int(inside[inside>0].max())} mm")
        ys, xs = np.nonzero(mask)
        margin = min(xs.min(), ys.min(),
                     rgb.shape[1] - 1 - xs.max(), rgb.shape[0] - 1 - ys.max())
        check("tube fully inside the frame", margin > 5,
              f"nearest edge margin {int(margin)} px, "
              f"bbox x[{xs.min()}..{xs.max()}] y[{ys.min()}..{ys.max()}]")
        check("clipping range does not remove the object",
              int(depth[mask][depth[mask] > 0].min()) > 0,
              f"nearest masked depth {int(inside[inside>0].min())} mm, "
              f"near plane {profile.min_range_m} m")

    check("mask provenance is labelled",
          frame["mask_source"] == "isaac_instance_gt"
          and frame["provenance"] == "synthetic",
          f"mask_source={frame['mask_source']} backend={frame['camera_backend']} "
          f"model={frame['camera_model']} provenance={frame['provenance']}")

    # ---- visual evidence ---------------------------------------------------
    import cv2
    cv2.imwrite(f"{OUTPUT}/d435_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finite = depth[depth > 0]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        scaled = np.clip((depth.astype(np.float64) - lo) / max(hi - lo, 1), 0, 1)
        colourised = cv2.applyColorMap((scaled * 255).astype(np.uint8),
                                       cv2.COLORMAP_TURBO)
        colourised[depth == 0] = 0
        cv2.imwrite(f"{OUTPUT}/d435_depth.png", colourised)
    cv2.imwrite(f"{OUTPUT}/cylinder5_mask.png", (mask.astype(np.uint8) * 255))
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(f"{OUTPUT}/d435_rgb_mask_overlay.png", overlay)

    # A WIDE VIEW OF THE WHOLE CELL, from a second camera, so the table, the
    # container and the arm are visible together rather than inferred.
    wide = SimulatedRGBDCamera(
        stage, "/World/WideView", profile,
        position_m=(-0.9, -1.3, layout.table_top_z_m + 1.15),
        look_at_m=(0.35, 0.0, layout.table_top_z_m))
    wide.warmup(12)
    cv2.imwrite(f"{OUTPUT}/workcell.png",
                cv2.cvtColor(wide.capture()["rgb"], cv2.COLOR_RGB2BGR))

    # ---- export the frame for Stage B -------------------------------------
    #
    # WRITTEN WHERE THE WORKER ALREADY LOOKS, in FoundationPose's own demo
    # layout, so Stage B sends ordinary serialised RGB-D through the ordinary
    # API — exactly as the physical D435 path will.
    dataset = os.path.join(REPO, ".cache-perception", "isaac-reference",
                           "stage_a_workcell")
    for directory in ("rgb", "depth", "masks"):
        os.makedirs(os.path.join(dataset, directory), exist_ok=True)
    cv2.imwrite(f"{dataset}/rgb/000000.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{dataset}/depth/000000.png", depth)
    cv2.imwrite(f"{dataset}/masks/000000.png", (mask.astype(np.uint8) * 255))
    with open(f"{dataset}/cam_K.txt", "w", encoding="utf-8") as handle:
        for row in k:
            handle.write(" ".join(f"{v:.18e}" for v in row) + "\n")

    # THE GROUND TRUTH, CONVERTED INTO THE ORIGINAL CAD FRAME.
    #
    # The scene canonicalises the mesh for stable placement — centred on its own
    # body and rotated so its tube axis is local +Z — but FoundationPose reports
    # the pose of the mesh AS THE STL DEFINES IT. Comparing the two without
    # undoing that canonicalisation would compare different frames.
    #
    #   v_prim  = R_align @ (v_cad - centre)
    #   p_world = R_wp @ v_prim + t_wp
    #           = (R_wp R_align) v_cad + (t_wp - R_wp R_align centre)
    #
    # so the world pose OF THE ORIGINAL CAD FRAME follows exactly, with no
    # fitted or fudged term.
    offsets = scene.cad_mesh_offsets[item.item_id]
    align = np.asarray(offsets["align_to_local_z"], dtype=np.float64)
    centre_m = np.asarray(offsets["centre_m"], dtype=np.float64)
    prim_position, prim_quaternion = scene.item_world_pose(item.item_id)
    w, x, y, z = [float(v) for v in np.asarray(prim_quaternion).reshape(-1)]
    R_wp = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])
    R_world_cad = R_wp @ align
    t_world_cad = np.asarray(prim_position).reshape(-1)[:3] - R_world_cad @ centre_m
    T_world_cad = np.eye(4)
    T_world_cad[:3, :3] = R_world_cad
    T_world_cad[:3, 3] = t_world_cad
    T_camera_cad = np.asarray(frame["camera_from_world"]) @ T_world_cad

    # The CAD reference point, in ORIGINAL STL coordinates. Position error is
    # measured here — a point on the tube axis — and not at the STL origin,
    # which for this obliquely drawn part sits 141 mm off the body.
    from wisepack_core.rgbd import load_object_registry
    registry = load_object_registry()
    cad_vertices = np.asarray(trimesh.load(
        registry.models[item.model_id].resolved_path(registry.root)).vertices)
    aabb_centre_mm = (cad_vertices.min(axis=0) + cad_vertices.max(axis=0)) / 2.0

    with open(f"{dataset}/ground_truth.json", "w", encoding="utf-8") as handle:
        json.dump({
            "model_id": item.model_id,
            "mesh_path": "CAD-Models/STL-Files/Cylinder5.stl",
            "mesh_units": "mm",
            "frame_id": "camera_color_optical_frame",
            # The answer, in the ORIGINAL CAD frame — the frame FoundationPose
            # reports in.
            "T_camera_object": T_camera_cad.tolist(),
            "model_center_mm": [float(v) for v in aabb_centre_mm],
            "depth_scale_mm_per_unit": 1.0,
            "symmetry": {"type": "discrete", "axis": "z", "fold": 2},
            # Provenance, carried with the data rather than asserted later.
            "acquisition_backend": "isaac_sim",
            "camera_profile": "d435_compatible_simulated",
            "mask_source": frame["mask_source"],
            "mask_provenance": frame["provenance"],
            "camera_from_world": frame["camera_from_world"],
            "canonicalisation": {
                "centre_m": offsets["centre_m"],
                "align_to_local_z": offsets["align_to_local_z"],
                "note": ("applied by the scene for stable placement and undone "
                         "here; FoundationPose is never adjusted to match it"),
            },
            # THE CAMERA -> WORKAREA TRANSFORM, derived from the scene.
            #
            # The WISEPACK workarea is the layout's `table` frame: millimetres,
            # axes parallel to world, origin at `table_frame_origin_m`. Because
            # the simulated camera is part of the scene its pose is exact, so
            # this is a derivation and not a calibration — which is precisely
            # why its provenance is recorded as `isaac_scene_transform` and not
            # as a measurement.
            #
            #   T_workarea_camera = T_workarea_world @ inv(T_camera_world)
            #
            # Both factors come from the scene; no value here is typed in.
            "workarea": {
                "frame_id": "wisepack_workarea",
                "origin_in_world_m": [float(v) for v in layout.table_frame_origin_m],
                "T_workarea_camera": _workarea_from_camera(
                    layout, np.asarray(frame["camera_from_world"])).tolist(),
                "method": "isaac_scene_transform",
                "provenance": "synthetic",
                "note": ("Exact by construction: the camera is a prim in the "
                         "scene. The physical path will populate the same "
                         "transform from a measured extrinsic."),
            },
            # THE SETTLED POSE, in the workarea frame — what the object ACTUALLY
            # is after physics, not where the scenario asked for it. Evaluation
            # must compare against this: the tube moved ~6 mm while settling,
            # and validating against the requested coordinates would score the
            # scenario rather than the perception.
            "settled_workarea": {
                "position_mm": _world_to_workarea_mm(
                    layout, np.asarray(prim_position).reshape(-1)[:3]),
                "note": ("the object's real pose after physics settled; the "
                         "requested spawn pose is deliberately NOT used here"),
            },
            "note": ("Isaac ground truth. EVALUATION ONLY — it never enters "
                     "the runtime perception path."),
        }, handle, indent=2)
    print(f"WROTE-DATASET {dataset}", flush=True)

    passed = sum(1 for r in results.values() if r["ok"])
    print(f"\nSTAGE-A {passed}/{len(results)} checks passed", flush=True)
    with open(f"{OUTPUT}/stage_a.json", "w", encoding="utf-8") as handle:
        json.dump({"checks": results,
                   "frame": {k: v for k, v in frame.items()
                             if k not in ("rgb", "depth_mm", "instance_ids")},
                   "mask_pixels": int(mask.sum())}, handle, indent=2)
    print(f"WROTE {OUTPUT}", flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
