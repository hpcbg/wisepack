#!/usr/bin/env python3
"""Render the model-free / few-shot REFERENCE set for a WISEPACK workpiece.

    ./scripts/run_isaac_task.sh /tmp/refs.log 1800 \
        simulators/isaac/generate_model_free_references.py [--views 16]

WHAT THIS PRODUCES, AND WHOSE FORMAT IT IS
------------------------------------------
The directory layout the PINNED upstream FoundationPose model-free stage reads
in `bundlesdf/run_nerf.py::run_one_ob`, and nothing of WISEPACK's own invention:

    <base_dir>/rgb/<i:07d>.png              colour
    <base_dir>/depth_enhanced/<i:07d>.png   uint16 millimetres (upstream /1e3 -> m)
    <base_dir>/mask/<i:07d>.png             binary object mask
    <base_dir>/cam_in_ob/<i:07d>.txt        4x4 camera-in-object pose
    <base_dir>/K.txt                        3x3 intrinsics
    <base_dir>/select_frames.yml            frame list upstream opens

WHAT THE SIMULATOR IS AND IS NOT ALLOWED TO CONTRIBUTE
------------------------------------------------------
ALLOWED, and used here: rendering the scene, rendering these reference views,
and — explicitly — supplying the REGISTERED REFERENCE POSES (`cam_in_ob`) that
the few-shot workflow requires. Upstream's own YCB-Video reference sets carry
exactly this, taken from that dataset's ground truth; a reference set without
registered poses is not a reference set. The manifest records that fact as
`simulator_ground_truth_used_for_reference_pose: true` so no reader has to infer
it.

FORBIDDEN, and not done anywhere in this file: giving the estimator a mesh. No
CAD path, no vertex, no nominal dimension and no symmetry declaration is written
into the reference set or passed to the Neural Object Field. The object frame
these poses are expressed in is the CAD frame — a CHOICE OF FRAME, so that a
model-free estimate is directly comparable with the CAD-mode estimate and with
ground truth — and a frame convention is not geometry: it says where the origin
is, not what shape sits at it.

THE MATERIAL IS THE DECLARED ONE. `config/isaac_materials.yaml` binds
`brushed_steel` to the workpiece, so the reference views and any later query
frame are the same surface by construction rather than by two renders happening
to agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Where reference sets live. Gitignored: they are large, regenerable, and
#: derived from a licensed third-party workflow's expectations.
REFERENCE_ROOT = os.path.join(REPO, ".cache-perception", "model-free")

#: Upstream names object directories `ob_<id:07d>`; `run_one_ob` is given the
#: directory directly, so the name only has to be stable and legible.
OBJECT_DIR = "ob_0000001"


def _viewpoints(count: int, radius_m: float, elevations_deg):
    """Deterministic camera positions on a sphere sector around the object.

    MEANINGFUL COVERAGE, NOT SIXTEEN NEAR-DUPLICATES. Azimuth is swept in equal
    steps at each of several elevations, so the set sees the tube end-on, along
    its length and obliquely. A ring at one elevation would reconstruct a band
    and leave the ends unobserved.

    A GOLDEN-ANGLE OFFSET per elevation ring stops the rings sharing azimuths,
    which would leave whole wedges of the object seen from only one height.
    """
    import numpy as np

    per_ring = max(1, count // len(elevations_deg))
    remainder = count - per_ring * len(elevations_deg)
    points = []
    for ring, elevation in enumerate(elevations_deg):
        n = per_ring + (1 if ring < remainder else 0)
        offset = 137.507 * ring                       # golden angle, degrees
        for i in range(n):
            azimuth = offset + i * (360.0 / n)
            az, el = np.radians(azimuth), np.radians(elevation)
            points.append((
                radius_m * np.cos(el) * np.cos(az),
                radius_m * np.cos(el) * np.sin(az),
                radius_m * np.sin(el),
                float(azimuth % 360.0), float(elevation)))
    return points


def _digest(base_dir: str) -> str:
    """A content digest of the reference set.

    WHAT THE CACHE KEY IS BUILT FROM. Every file upstream reads, hashed in a
    stable order: change a view, a mask, a pose or the intrinsics and the digest
    changes, so a representation built from the old set cannot be reused.
    """
    sha = hashlib.sha256()
    for folder in ("rgb", "depth_enhanced", "mask", "cam_in_ob"):
        directory = os.path.join(base_dir, folder)
        for name in sorted(os.listdir(directory)):
            sha.update(name.encode())
            with open(os.path.join(directory, name), "rb") as handle:
                sha.update(handle.read())
    with open(os.path.join(base_dir, "K.txt"), "rb") as handle:
        sha.update(handle.read())
    return sha.hexdigest()[:16]


def main() -> int:
    from isaacsim import SimulationApp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=int, default=16)
    # 0.55 m clipped the tube vertically at high elevation: a 342 mm object at
    # fy 925 needs ~0.63 m to fit inside 720 px with margin. 0.80 m keeps the
    # whole silhouette inside the frame from every viewpoint in the sector.
    parser.add_argument("--radius-m", type=float, default=0.80)
    parser.add_argument("--model-id", default="cylinder5")
    args = parser.parse_args()

    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    try:
        return _run(app, args)
    except BaseException:                                    # noqa: BLE001
        import traceback
        print("MODEL-FREE-REFS FAILED", flush=True)
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
    base_dir = os.path.join(REFERENCE_ROOT, args.model_id, "reference", OBJECT_DIR)
    for folder in ("rgb", "depth_enhanced", "mask", "cam_in_ob"):
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

    print("MODEL-FREE-REFS building the workcell", flush=True)
    scenario = build_scenario("cad_cylinder5_single")
    layout = SceneLayout()
    scene = WisepackScene(layout, PhysicsConfig())
    scene.build(scenario, container_ids=["CNT-001"])
    item = scenario.items[0]

    material = load_materials().require()
    print(f"  material profile : {material.name} "
          f"(metallic {material.metallic}, roughness {material.roughness})",
          flush=True)

    # ---- the object's world pose, in the ORIGINAL CAD frame ---------------
    #
    # Identical derivation to Stage A's, and for the identical reason: the scene
    # canonicalises the mesh for stable placement, so the prim frame is not the
    # frame a pose is reported in. This is a FRAME, not geometry.
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
    T_world_ob = np.eye(4)
    T_world_ob[:3, :3] = R_world_cad
    T_world_ob[:3, 3] = t_world_cad
    T_ob_world = np.linalg.inv(T_world_ob)

    # ---- the camera, re-placed per view ------------------------------------
    profile = sensor_profile("d435")
    stage = stage_utils.get_current_stage()
    camera = SimulatedRGBDCamera(
        stage, "/World/RefCamera", profile,
        position_m=(float(t_world_cad[0]) + 0.4, float(t_world_cad[1]),
                    float(t_world_cad[2]) + 0.4),
        look_at_m=tuple(float(v) for v in t_world_cad))
    camera.warmup()

    K = np.asarray(camera.intrinsics_matrix(), dtype=np.float64)
    np.savetxt(os.path.join(base_dir, "K.txt"), K, fmt="%.18e")

    views = _viewpoints(args.views, args.radius_m,
                        elevations_deg=(20.0, 40.0, 60.0, 75.0))
    written, manifest_views = 0, []
    for index, (dx, dy, dz, azimuth, elevation) in enumerate(views):
        eye = t_world_cad + np.array([dx, dy, dz], dtype=np.float64)
        # A VIEW BELOW THE TABLE SEES NOTHING. The object rests on a surface, so
        # the sector is the upper hemisphere; a rejected viewpoint is reported
        # rather than silently producing a frame of tabletop.
        if eye[2] <= layout.table_top_z_m + 0.05:
            print(f"  view {index}: skipped, below the work surface", flush=True)
            continue
        camera._place(camera.camera, tuple(float(v) for v in eye),
                      tuple(float(v) for v in t_world_cad))
        camera.warmup(6)
        frame = camera.capture()
        try:
            mask = camera.mask_for(frame, args.model_id)
        except ValueError as exc:
            print(f"  view {index}: skipped, {exc}", flush=True)
            continue
        if int(mask.sum()) < 500:
            print(f"  view {index}: skipped, only {int(mask.sum())} object px",
                  flush=True)
            continue
        # A CLIPPED SILHOUETTE IS BAD REFERENCE DATA. The Neural Object Field
        # reconstructs from these outlines; one cut off by the image border
        # asserts that the object ENDS at the edge of the frame, and the
        # reconstruction inherits the truncation. Rejected loudly rather than
        # quietly reconstructed from.
        if (mask[0, :].any() or mask[-1, :].any()
                or mask[:, 0].any() or mask[:, -1].any()):
            print(f"  view {index}: skipped, object touches the frame border "
                  "(move the camera back)", flush=True)
            continue

        # cam_in_ob: the camera OPTICAL frame expressed in the object frame.
        T_world_cam = np.linalg.inv(np.asarray(frame["camera_from_world"],
                                               dtype=np.float64))
        cam_in_ob = T_ob_world @ T_world_cam

        stem = f"{written:07d}"
        cv2.imwrite(os.path.join(base_dir, "rgb", stem + ".png"),
                    cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(base_dir, "depth_enhanced", stem + ".png"),
                    frame["depth_mm"].astype(np.uint16))
        cv2.imwrite(os.path.join(base_dir, "mask", stem + ".png"),
                    (mask.astype(np.uint8) * 255))
        np.savetxt(os.path.join(base_dir, "cam_in_ob", stem + ".txt"),
                   cam_in_ob, fmt="%.18e")
        manifest_views.append({
            "frame": stem, "azimuth_deg": round(azimuth, 2),
            "elevation_deg": round(elevation, 2),
            "camera_distance_m": round(float(np.linalg.norm(eye - t_world_cad)), 4),
            "object_pixels": int(mask.sum()),
        })
        written += 1
        print(f"  view {stem}: az {azimuth:6.1f} el {elevation:4.1f} "
              f"{int(mask.sum()):6d} px", flush=True)

    if written == 0:
        print("MODEL-FREE-REFS FAILED: no usable views", flush=True)
        return 1

    # `select_frames.yml` is opened by `run_one_ob` before anything else.
    import yaml
    with open(os.path.join(base_dir, "select_frames.yml"), "w",
              encoding="utf-8") as handle:
        yaml.safe_dump({"frames": [v["frame"] for v in manifest_views]}, handle)

    elapsed = time.time() - started
    digest = _digest(base_dir)
    manifest = {
        "source": "isaac_simulated",
        "purpose": "foundationpose_model_free_reference",
        "reference_object": args.model_id,
        "view_count": written,
        "requested_view_count": args.views,
        "material_profile": material.name,
        "material": material.to_dict(),
        "camera_profile": "d435_compatible_simulated",
        "intrinsics": K.tolist(),
        "object_frame": "original CAD frame of the model (a FRAME, not geometry)",
        "reference_set_digest": digest,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation_seconds": round(elapsed, 1),
        # THE DECLARATION §4 REQUIRES, stated as a fact rather than implied.
        "simulator_ground_truth_used_for_reference_pose": True,
        "simulator_ground_truth_note": (
            "Isaac's object pose was used ONLY to express the reference camera "
            "poses (cam_in_ob) in the object frame, which the upstream few-shot "
            "workflow requires of any reference set. It is NOT supplied to the "
            "Neural Object Field beyond those poses, and it is NOT available to "
            "any query estimate."),
        "cad_supplied_to_estimator": False,
        "cad_note": (
            "No mesh, vertex, nominal dimension or symmetry declaration is "
            "written into this reference set. The simulator used the CAD asset "
            "to RENDER the object, which is scene generation, not estimator "
            "input."),
        "views": manifest_views,
        "upstream_format": (
            "bundlesdf/run_nerf.py::run_one_ob — rgb/, depth_enhanced/ (uint16 "
            "mm), mask/, cam_in_ob/ (4x4), K.txt, select_frames.yml"),
    }
    with open(os.path.join(base_dir, "wisepack_reference_manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nMODEL-FREE-REFS {written}/{args.views} views in {elapsed:.0f}s",
          flush=True)
    print(f"  base_dir : {base_dir}", flush=True)
    print(f"  digest   : {digest}", flush=True)
    print("WROTE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
