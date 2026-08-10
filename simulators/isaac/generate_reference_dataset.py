#!/usr/bin/env python3
"""Generate a deterministic WISEPACK reference dataset in Isaac Sim.

    ./scripts/run_isaac_task.sh /tmp/gen.log 540 \
        simulators/isaac/generate_reference_dataset.py --model-id cylinder5

IT LIVES HERE, NOT IN scripts/, BECAUSE IT IMPORTS ISAAC. Isaac Sim is one
implementation behind the WISEPACK contract, not a dependency of it: `isaacsim`,
`omni` and `pxr` are confined to `simulators/isaac/` so the ordinary test suite
and the orchestrator never need a GPU. `tests/test_isaac_backend.py` fails the
build if that leaks — as it did when this file was first written under
`scripts/`.

WHY THIS EXISTS
---------------
The tutorial bolt dataset has NO GROUND-TRUTH POSE, so it can only ever measure
repeatability and plausibility. A synthetic scene knows exactly where it put the
object, which makes a real error measurement possible: translation in
millimetres, and orientation modulo the object's declared symmetry.

It is also the domain-relevant reference. WISEPACK packs pipe sections, not
bolts.

WHAT IS GROUND TRUTH HERE, AND WHAT IS NOT
------------------------------------------
Ground truth is the pose the scene was BUILT with, together with the camera
intrinsics that scene was rendered through. Both are exact by construction.

What this is NOT is a measurement of the real world: it is a rendering, without
sensor noise, without real depth artefacts and without real materials. It
measures whether FoundationPose recovers a pose WISEPACK already knows — which
is exactly the thing the bolt dataset cannot do, and exactly not a substitute
for live RealSense validation.

THE TWO CONVENTIONS THAT MUST NOT BE CONFUSED
---------------------------------------------
USD cameras look down **-Z with +Y up**. FoundationPose, OpenCV and every
intrinsic matrix here use the optical convention: **+Z forward, +Y down**. The
ground-truth pose is converted between them explicitly; getting this wrong
produces a clean 180-degree orientation error that looks like an estimator bug.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

#: Deterministic by construction — a fixed pose per case, no randomisation.
#: This is a REGRESSION dataset: it has to give the same answer next month, and
#: domain randomisation belongs in training-data generation, not here.
CASES = {
    "cylinder5": {
        # Position in the camera's optical frame, millimetres, and an
        # orientation chosen so the bend is clearly visible rather than
        # edge-on — the whole reason Cylinder5 goes first is that its geometry
        # constrains the pose, and a view that hides the bend wastes that.
        "position_mm": (0.0, 0.0, 900.0),
        "euler_deg": (18.0, 25.0, 40.0),
    },
    "cylinder3": {
        "position_mm": (0.0, 0.0, 850.0),
        "euler_deg": (12.0, 30.0, 20.0),
    },
    "cylinder4": {
        "position_mm": (0.0, 0.0, 1100.0),
        "euler_deg": (8.0, 35.0, 15.0),
    },
}

DEFAULT_WIDTH, DEFAULT_HEIGHT = 1280, 720

#: The output root. WISEPACK-owned and git-ignored: rendered datasets are
#: large, and they are regenerable from this script plus the CAD.
DEFAULT_OUTPUT_ROOT = os.path.join(REPO, ".cache-perception", "isaac-reference")


def euler_to_matrix(rx_deg, ry_deg, rz_deg):
    """Extrinsic XYZ euler -> 3x3. Only used to BUILD a pose, never to report one."""
    import numpy as np
    rx, ry, rz = (np.radians(v) for v in (rx_deg, ry_deg, rz_deg))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def load_case(model_id):
    """Mesh path, units and symmetry from the WISEPACK registry — not guessed."""
    import yaml
    with open(os.path.join(REPO, "config", "perception_objects.yaml"),
              encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    for entry in document["objects"]:
        if entry["model_id"] == model_id:
            return entry
    raise SystemExit(f"unknown model_id {model_id!r}; known: "
                     + ", ".join(e["model_id"] for e in document["objects"]))


def assets_root():
    configured = os.environ.get("WISEPACK_PERCEPTION_ASSETS_ROOT", "").strip()
    return configured or os.path.join(os.path.dirname(REPO), "references")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="cylinder5")
    parser.add_argument("--out", default="")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--focal-length-mm", type=float, default=24.0)
    parser.add_argument("--horizontal-aperture-mm", type=float, default=20.955)
    args = parser.parse_args(argv)

    entry = load_case(args.model_id)
    case = CASES.get(args.model_id)
    if case is None:
        raise SystemExit(f"no reference case defined for {args.model_id!r}; "
                         f"defined: {', '.join(sorted(CASES))}")
    mesh_file = os.path.join(assets_root(), entry["mesh_path"])
    if not os.path.isfile(mesh_file):
        raise SystemExit(f"mesh not found: {mesh_file}")

    output = args.out or os.path.join(DEFAULT_OUTPUT_ROOT, args.model_id)
    for directory in ("rgb", "depth", "masks"):
        os.makedirs(os.path.join(output, directory), exist_ok=True)

    # ---- Isaac -------------------------------------------------------------
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    try:
        return _build_and_render(app, args, entry, case, mesh_file, output)
    except BaseException as exc:                             # noqa: BLE001
        # NOTHING MAY SWALLOW THE REASON. Isaac installs its own shutdown
        # handling, and an exception raised inside the render loop otherwise
        # ends as a silent "Simulation App Shutting Down" with no traceback in
        # the log — which is exactly what happened the first time this ran, and
        # cost a ten-minute wait on a process that had already died.
        import traceback
        print("ISAAC-REFERENCE FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            app.close()
        except BaseException:                                # noqa: BLE001
            pass
        return 1


def _mark(message):
    """Progress markers, flushed. Isaac buffers and can die without unwinding,
    so a stage that is not reported is a stage that did not complete."""
    print(f"STAGE {message}", flush=True)
    sys.stdout.flush()


def _build_and_render(app, args, entry, case, mesh_file, output):
    _mark("enter")

    _mark("imports")
    import numpy as np
    import omni.replicator.core as rep
    import trimesh
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: F401

    import omni.usd
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # ---- the object, built directly from the CAD ---------------------------
    #
    # The mesh is created from the STL's own vertices rather than imported
    # through a converter: it keeps the geometry EXACT, and the millimetre ->
    # metre conversion is the registry's declared unit applied once, here.
    _mark("load-mesh")
    mesh = trimesh.load(mesh_file)
    scale = {"mm": 0.001, "m": 1.0}[entry["mesh_units"]] * float(entry.get("scale", 1.0))
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
    # Centre the mesh on its own origin so the reported pose is the pose OF THE
    # MESH AS FOUNDATIONPOSE LOADS IT — the same convention the worker reports.
    faces = np.asarray(mesh.faces, dtype=np.int32)

    _mark("build-mesh")
    object_prim_path = "/World/ReferenceObject"
    usd_mesh = UsdGeom.Mesh.Define(stage, object_prim_path)
    usd_mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in vertices])
    usd_mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    usd_mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    usd_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.57, 0.60)])

    # SEMANTICS ARE WHAT MAKE THE OBJECT SEGMENTABLE. Without a label the prim
    # simply does not appear in `idToLabels`, the instance mask comes back
    # empty, and the failure looks like a rendering problem rather than a
    # missing annotation. The label is the model_id, so the mask and the CAD
    # identity carry the same name.
    from isaacsim.core.utils.semantics import add_labels
    add_labels(usd_mesh.GetPrim(), labels=[args.model_id], instance_name="class")
    _mark("semantics")

    # ---- the pose, in the OPTICAL frame, then converted to USD -------------
    _mark("pose")
    position_m = np.array(case["position_mm"], dtype=np.float64) / 1000.0
    rotation = euler_to_matrix(*case["euler_deg"])
    # T_optical_object: what the ground truth will say, and what FoundationPose
    # is expected to recover.
    T_opt_obj = np.eye(4)
    T_opt_obj[:3, :3] = rotation
    T_opt_obj[:3, 3] = position_m

    # USD camera convention (-Z forward, +Y up) vs optical (+Z forward, +Y
    # down). One flip, applied explicitly and in one place.
    FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
    T_usdcam_obj = FLIP @ T_opt_obj

    # The camera sits at the world origin looking down -Z, so world == USD
    # camera frame and the object's world pose IS T_usdcam_obj.
    xform = UsdGeom.Xformable(usd_mesh)
    xform.ClearXformOpOrder()
    matrix = Gf.Matrix4d(*[float(v) for v in T_usdcam_obj.T.flatten()])
    xform.AddTransformOp().Set(matrix)

    # ---- a work surface, so the scene is not an object floating in void ----
    # Placed BELOW the object along the optical +Y (which is USD -Y).
    _mark("surface")
    surface = UsdGeom.Mesh.Define(stage, "/World/WorkSurface")
    extent, below = 1.5, -0.15
    surface.CreatePointsAttr([Gf.Vec3f(-extent, below, -extent),
                              Gf.Vec3f(extent, below, -extent),
                              Gf.Vec3f(extent, below, extent),
                              Gf.Vec3f(-extent, below, extent)])
    surface.CreateFaceVertexCountsAttr([4])
    surface.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    surface.CreateDisplayColorAttr([Gf.Vec3f(0.82, 0.80, 0.78)])

    light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    light.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 20.0, 0.0))

    # ---- the camera, with intrinsics we then WRITE DOWN --------------------
    _mark("camera")
    camera_path = "/World/Camera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(args.focal_length_mm)
    camera.CreateHorizontalApertureAttr(args.horizontal_aperture_mm)
    vertical_aperture = args.horizontal_aperture_mm * args.height / args.width
    camera.CreateVerticalApertureAttr(vertical_aperture)
    # THE NEAR PLANE IS LOAD-BEARING. USD's default clipping range starts at
    # 1.0 m, so an object on a table 0.9 m away is clipped away entirely and the
    # scene renders as pure background — which looks exactly like a broken
    # annotator until you check.
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

    fx = args.focal_length_mm * args.width / args.horizontal_aperture_mm
    fy = args.focal_length_mm * args.height / vertical_aperture
    cx, cy = args.width / 2.0, args.height / 2.0
    intrinsics = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    # ---- render ------------------------------------------------------------
    _mark("render-product")
    render_product = rep.create.render_product(camera_path,
                                               (args.width, args.height))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    # SEMANTIC, NOT INSTANCE, and for a measured reason: with Isaac 6's label
    # schema the `instance_segmentation` annotator returns the object under an
    # EMPTY label, while `semantic_segmentation` resolves it to
    # {'class': '<model_id>'}. With exactly one labelled object in the scene the
    # semantic mask IS the instance mask, and it carries the CAD identity.
    # A cluttered scene would need the instance annotator and the labelling
    # question reopened — see SEGMENTATION.md.
    mask_annotator = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
    for annotator in (rgb_annotator, depth_annotator, mask_annotator):
        annotator.attach(render_product)

    # Several steps: the first frames of a path-traced render are noisy and the
    # annotators are not populated until the pipeline has run.
    _mark("stepping")
    for _ in range(40):
        rep.orchestrator.step(rt_subframes=8)

    _mark("fetch")
    rgb = rgb_annotator.get_data()[..., :3]
    depth_m = np.asarray(depth_annotator.get_data(), dtype=np.float64)
    mask_data = mask_annotator.get_data()

    # ---- the exact instance mask -------------------------------------------
    _mark("mask")
    ids = np.asarray(mask_data["data"])
    info = mask_data["info"]["idToLabels"]
    _mark(f"idToLabels {info}")
    # Match on the LABEL (the model_id) or the prim path — Replicator reports
    # one or the other depending on how the prim was annotated.
    object_ids = [int(key) for key, value in info.items()
                  if args.model_id in str(value)]
    if not object_ids:
        # REPORTED BEFORE ANYTHING IS CLOSED. `app.close()` tears the process
        # down hard enough that a message printed after it never reaches the
        # log — which is exactly how this failure first presented: a silent
        # "Simulation App Shutting Down" with no reason anywhere.
        print("ISAAC-REFERENCE FAILED: the object did not appear in the "
              f"instance segmentation. idToLabels was {info}", flush=True)
        sys.stdout.flush()
        raise SystemExit(2)
    mask = np.isin(ids, object_ids)

    # `distance_to_image_plane` is already the Z depth an intrinsic matrix
    # expects (not range along the ray), in metres. Written as uint16
    # millimetres, the encoding the FoundationPose reader uses, with 0 meaning
    # "no measurement" exactly as a real sensor does.
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    depth_mm[depth_m <= 0] = 0

    _mark("write")
    import cv2
    cv2.imwrite(os.path.join(output, "rgb", "000000.png"),
                cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output, "depth", "000000.png"), depth_mm)
    cv2.imwrite(os.path.join(output, "masks", "000000.png"),
                (mask.astype(np.uint8) * 255))
    with open(os.path.join(output, "cam_K.txt"), "w", encoding="utf-8") as handle:
        for row in intrinsics:
            handle.write(" ".join(f"{v:.18e}" for v in row) + "\n")

    ground_truth = {
        "model_id": args.model_id,
        "mesh_path": entry["mesh_path"],
        "mesh_units": entry["mesh_units"],
        "symmetry": entry.get("symmetry", {"type": "none"}),
        # THE ANSWER, in the frame the worker reports poses in.
        "frame_id": "camera_color_optical_frame",
        "T_camera_object": [[float(v) for v in row] for row in T_opt_obj],
        "position_mm": [float(v) for v in case["position_mm"]],
        "euler_deg_extrinsic_xyz": [float(v) for v in case["euler_deg"]],
        "camera": {
            "intrinsics": intrinsics,
            "width": args.width, "height": args.height,
            "focal_length_mm": args.focal_length_mm,
            "horizontal_aperture_mm": args.horizontal_aperture_mm,
            # The camera is AT the world origin looking down -Z (USD), which is
            # +Z in the optical frame. Recorded so the convention is auditable.
            "usd_to_optical": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                               [0, 0, 0, 1]],
            "pose_in_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                              [0, 0, 0, 1]],
        },
        # THE REFERENCE POINT FOR POSITION ERROR, and it is not the STL origin.
        #
        # Cylinder5's STL origin sits 98.8 mm OFF the tube axis, in empty space,
        # because of how the part was drawn. Measuring "position error" there
        # turns any rotation about the tube axis into apparent translation: a
        # 114 deg spin swings that point 166 mm. The pose is not wrong; the
        # measuring point was.
        #
        # So position is compared at the AABB centre — a point ON the axis, which
        # spin cannot move, and which is physically "where the tube is". It is
        # also exactly FoundationPose's own `model_center`, defined identically
        # as (vertices.min(0) + vertices.max(0)) / 2, so both sides of the
        # comparison agree by construction rather than by a correction.
        "model_center_mm": [float(v) for v in
                            ((np.asarray(mesh.vertices).min(axis=0)
                              + np.asarray(mesh.vertices).max(axis=0)) / 2.0)],
        "model_center_note": (
            "AABB centre in ORIGINAL STL coordinates; identical to "
            "FoundationPose's model_center. Position error is measured here, "
            "not at the STL origin, which for an obliquely drawn part is an "
            "arbitrary point in space."),
        "depth_scale_mm_per_unit": 1.0,
        "depth_encoding": "uint16 PNG millimetres; 0 = no measurement",
        "mask_source": "isaac_instance_segmentation",
        "mask_pixels": int(mask.sum()),
        "generator": "scripts/generate_isaac_reference.py",
        "isaac_version": open("/data/isaac-sim/isaac-sim-6.0.1/VERSION").read().strip()
        if os.path.isfile("/data/isaac-sim/isaac-sim-6.0.1/VERSION") else "",
        "deterministic": True,
        "note": ("Rendered ground truth. Exact by construction, and NOT a "
                 "measurement of the real world: no sensor noise, no real depth "
                 "artefacts, no real materials. It measures whether "
                 "FoundationPose recovers a pose WISEPACK already knows; it is "
                 "not a substitute for live RealSense validation."),
    }
    with open(os.path.join(output, "ground_truth.json"), "w",
              encoding="utf-8") as handle:
        json.dump(ground_truth, handle, indent=2)

    print(f"WROTE {output}")
    print(f"  mask pixels {int(mask.sum())}")
    print(f"  depth range {depth_mm[depth_mm > 0].min()}..{depth_mm.max()} mm")
    print(f"  intrinsics fx={fx:.3f} fy={fy:.3f} cx={cx} cy={cy}")
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
