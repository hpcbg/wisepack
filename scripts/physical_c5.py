"""The first REAL-OBJECT FoundationPose run: physical D435 -> one Cylinder5.

    ./scripts/physical_c5.sh

WHAT THIS IS
------------
The physical counterpart of `stage_b.sh`. Same worker, same provider, same
`PhysicalObservation` — the only differences are where the pixels come from and
that there is no ground truth to compare against:

    physical D435 -> capture -> depth_plane_foreground -> FoundationPose -> pose

WHAT IT REFUSES TO DO
---------------------
* NO FABRICATED MASK. If segmentation produces nothing usable, this stops and
  says so. A hand-drawn or thresholded stand-in would put an invented object
  region into the input of a pose measurement.
* NO ACCURACY NUMBER. Nothing here knows where the tube actually is. Repeatedly
  estimating a STATIONARY scene measures REPEATABILITY — the spread of the
  estimator's own answers — and that is reported as repeatability and never as
  accuracy.
* NO WORK-AREA POSE. The physical camera has never been calibrated to the work
  area, so the pose stays in the camera optical frame and
  `workarea_pose_available` is False. Inventing an extrinsic here is how a
  plausible pose ends up in the wrong space.
* NO ROBOT. No plan, no IK, no motion.

WHICH OBJECT IS IN VIEW IS AN INPUT, NEVER INFERRED. `--model` names the CAD
part an operator placed on the table. Several WISEPACK tubes share an outer
diameter and differ only in length, which one view does not resolve.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "perception"))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))

WORKER = f"http://127.0.0.1:{os.environ.get('WISEPACK_FP_PORT', '22201')}"
OUT = os.path.join(REPO, ".cache-perception", "physical-c5")

#: The profile validated on this bench — see the physical-D435 milestone.
#: Stated rather than negotiated, so a silent renegotiation to another size
#: cannot change the intrinsics underneath a comparison.
WIDTH, HEIGHT, FPS = 1280, 720, 30

#: A uint16 millimetre depth PNG. The device reports 1.0000000475 mm per raw
#: unit and `capture_dataset` writes exactly those raw units.
DEPTH_SCALE_MM = 1.0


def say(message: str) -> None:
    print(f"[physical-c5] {message}", flush=True)


def post(path: str, payload: Dict[str, Any], timeout: float = 600.0
         ) -> Tuple[Optional[Dict[str, Any]], str]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{WORKER}{path}", data=body, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), ""
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        return None, f"HTTP {exc.code}: {detail}"
    except Exception as exc:                                 # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def get(path: str, timeout: float = 30.0) -> Tuple[Optional[Any], str]:
    try:
        with urllib.request.urlopen(f"{WORKER}{path}", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")), ""
    except Exception as exc:                                 # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


#: Where `capture_dataset` writes on the HOST — the same directory the worker
#: sees as /captures. Named here so the LOSSLESS originals can be kept beside
#: the worker's diagnostic JPEGs.
CAPTURES_ON_HOST = os.path.join(REPO, ".cache-perception", "rgbd-captures")


def copy_source_frames(dataset: str, index: int = 0) -> None:
    """The unmodified PNGs the estimate was computed from.

    THE JPEGS ARE DIAGNOSTICS, NOT DATA. The worker's images are re-encoded and
    colour-mapped for looking at; the uint16 depth PNG and the raw colour PNG
    are the actual inputs, and an evaluator asking "what did it see" must get
    those rather than a picture of them.
    """
    import shutil                                             # noqa: PLC0415
    root = os.path.join(CAPTURES_ON_HOST, dataset)
    for kind, name in (("rgb", "source_rgb.png"),
                       ("depth", "source_depth_aligned_uint16.png")):
        source = os.path.join(root, kind, f"{index:06d}.png")
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(OUT, name))
    intrinsics = os.path.join(root, "cam_K.txt")
    if os.path.isfile(intrinsics):
        shutil.copy2(intrinsics, os.path.join(OUT, "cam_K.txt"))


def fetch_image(kind: str, destination: str) -> bool:
    try:
        with urllib.request.urlopen(f"{WORKER}/image/{kind}", timeout=30) as r:
            data = r.read()
    except Exception:                                        # noqa: BLE001
        return False
    with open(destination, "wb") as handle:
        handle.write(data)
    return True


# --------------------------------------------------------------------------- #
# 1. the device
# --------------------------------------------------------------------------- #


def require_camera() -> Dict[str, Any]:
    """The physical D435, or a refusal naming the layer that is missing."""
    health, error = get("/health")
    if health is None:
        say(f"the FoundationPose worker is not answering at {WORKER}: {error}")
        say("start it:  ./scripts/setup_foundationpose.sh --no-build --run")
        raise SystemExit(2)
    if not health.get("rgbd_camera_available"):
        say("the worker has no RGB-D camera: "
            + "; ".join(health.get("blocked_by") or ["no reason given"]))
        say("diagnose it:  ./scripts/realsense_diagnose.sh")
        raise SystemExit(2)
    if not health.get("inference_available"):
        say("FoundationPose inference is unavailable: "
            + "; ".join(health.get("blocked_by") or ["no reason given"]))
        raise SystemExit(2)

    camera, error = get("/camera")
    if camera is None or not camera.get("available"):
        say(f"the camera could not be described: {error or camera.get('reason')}")
        raise SystemExit(2)
    device = camera["device"]
    say(f"device: {device['name']}  serial {device['serial_number']}  "
        f"firmware {device['firmware_version']}  USB {device['usb_type_descriptor']}")
    if {"width": WIDTH, "height": HEIGHT, "fps": FPS} not in \
            device.get("synchronised_profiles", []):
        say(f"the device does not offer {WIDTH}x{HEIGHT}@{FPS} for colour+depth")
        raise SystemExit(2)
    return device


# --------------------------------------------------------------------------- #
# 2. capture
# --------------------------------------------------------------------------- #


def capture(model_id: str, frames: int) -> Dict[str, Any]:
    """One dataset of `frames` synchronised, aligned RGB-D frames.

    ONE CAPTURE, SEVERAL FRAMES, ON PURPOSE. The scene is stationary, so the
    frames differ only by sensor noise — which is exactly what makes the spread
    of the poses estimated from them a measure of repeatability.
    """
    say(f"capturing {frames} frame(s) at {WIDTH}x{HEIGHT}@{FPS}, depth aligned "
        "to colour")
    document, error = post("/camera/capture", {
        "model_id": model_id, "frames": frames,
        "width": WIDTH, "height": HEIGHT, "fps": FPS, "align": True})
    if document is None:
        say(f"the capture failed: {error}")
        raise SystemExit(3)
    if not document.get("alignment_verified"):
        say("the capture does NOT carry verified alignment; a mask drawn on the "
            "colour image would select the wrong depth pixels")
        raise SystemExit(3)
    intrinsics = document["colour_intrinsics"]
    say(f"captured {os.path.basename(document['root'])}: "
        f"fx {intrinsics['fx']:.2f} fy {intrinsics['fy']:.2f} "
        f"cx {intrinsics['cx']:.2f} cy {intrinsics['cy']:.2f}  "
        f"depth scale {document['depth_scale_mm_per_unit']:.9f} mm/unit "
        "(both read from the device)")
    return document


# --------------------------------------------------------------------------- #
# 3. segmentation, LOOKED AT before anything is inferred
# --------------------------------------------------------------------------- #


def segment(dataset: str, frame: int, options: Dict[str, Any]
            ) -> Dict[str, Any]:
    document, error = post("/segment", {
        "dataset": dataset, "frame": frame, "method": "depth_plane_foreground",
        "depth_scale_mm": DEPTH_SCALE_MM, "segmentation": options})
    if document is None:
        say(f"segmentation failed: {error}")
        raise SystemExit(4)
    return document


def report_segmentation(document: Dict[str, Any]) -> None:
    """Every diagnostic the method produced, flat, as the worker returns it.

    `SegmentationResult.to_dict()` spreads its diagnostics at the top level
    beside `mask_source`/`mask_valid`/`reason`; reading them from a nested
    "diagnostics" key silently prints None for every number, which reads as
    "the method measured nothing" rather than "this reader looked in the wrong
    place".
    """
    print()
    print("  SEGMENTATION — depth_plane_foreground")
    for label, key in (
            ("roi px", "roi_px"),
            ("foreground FULL frame", "foreground_points_full_frame"),
            ("plane_detected", "plane_detected"),
            ("plane residual mm", "plane_residual_mm"),
            ("plane inlier fraction", "plane_inlier_fraction"),
            ("plane normal", "plane_normal"),
            ("valid depth pixels", "valid_depth_pixels"),
            ("foreground points", "foreground_points"),
            ("connected components", "components"),
            ("component areas px", "component_areas_px"),
            ("selected component", "selected_component"),
            ("mask pixels", "mask_pixels"),
            ("mask area fraction", "mask_area_fraction"),
            ("mask centroid px", "mask_centroid_px"),
            ("mask median range mm", "mask_median_range_mm"),
            ("mask extent long mm", "mask_extent_long_mm"),
            ("mask extent across mm", "mask_extent_across_mm"),
            ("depth inside mask", "valid_depth_fraction_in_mask"),
            ("mask_valid", "mask_valid")):
        if key in document:
            print(f"    {label:24} {document[key]}")
    if document.get("reason"):
        print(f"    {'reason':24} {document['reason']}")


# --------------------------------------------------------------------------- #
# 4. the pose, through the ordinary provider
# --------------------------------------------------------------------------- #


def estimate(dataset: str, model_id: str, frames: int, options: Dict[str, Any],
             refine_iterations: int) -> List[Any]:
    """One batch per frame, through the SAME provider the Isaac path uses."""
    from providers.foundationpose_rgbd import FoundationPoseProvider

    provider = FoundationPoseProvider()
    batches = []
    for index in range(frames):
        started = time.monotonic()
        batch = provider.acquire_physical(
            dataset=dataset, model_id=model_id,
            depth_scale_mm=DEPTH_SCALE_MM, frame=index,
            refine_iterations=refine_iterations,
            batch_id=f"physical-c5-{index + 1}",
            mask_source="depth_plane_foreground", segmentation=options)
        elapsed = (time.monotonic() - started) * 1000.0
        status = getattr(batch.status, "value", batch.status)
        if str(status) != "ok":
            say(f"frame {index}: estimation FAILED — {batch.error}")
        else:
            pose = pose_of(batch)
            say(f"frame {index}: x {pose['x_mm']:.1f}  y {pose['y_mm']:.1f}  "
                f"z {pose['z_mm']:.1f} mm   ({elapsed:.0f} ms)")
        batches.append(batch)
    return batches


def succeeded(batches: List[Any]) -> List[Any]:
    return [b for b in batches
            if str(getattr(b.status, "value", b.status)) == "ok"
            and b.observations]


def observation_of(batch: Any) -> Dict[str, Any]:
    """The SERIALISED observation — the same contract the dashboard renders.

    Read through `to_dict()` rather than off the object: the serialisation is
    what every consumer sees, and a script that reached past it could report a
    field the dashboard never shows.
    """
    return batch.observations[0].to_dict()


def pose_of(batch: Any) -> Dict[str, Any]:
    return observation_of(batch)["pose"]


# --------------------------------------------------------------------------- #
# 5. evidence, WITHOUT ground truth
# --------------------------------------------------------------------------- #


def repeatability(batches: List[Any]) -> Dict[str, Any]:
    """The spread of poses over a STATIONARY scene.

    THIS IS NOT ACCURACY, and the distinction is not pedantic: an estimator can
    return the same wrong pose every time. A tight spread means the measurement
    is reproducible; where the tube actually is remains unmeasured.
    """
    poses = [pose_of(b) for b in succeeded(batches)]
    if len(poses) < 2:
        return {"frames": len(poses),
                "note": "fewer than two successful estimates; no spread exists"}

    def spread(values: List[float]) -> Dict[str, float]:
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return {"mean": round(mean, 3), "sd": round(math.sqrt(variance), 3),
                "min": round(min(values), 3), "max": round(max(values), 3),
                "range": round(max(values) - min(values), 3)}

    centres = [(p["x_mm"], p["y_mm"], p["z_mm"]) for p in poses]
    mean = [sum(c[i] for c in centres) / len(centres) for i in range(3)]
    radial = [math.dist(c, mean) for c in centres]

    # ORIENTATION SPREAD AS AN ANGLE, not as four quaternion components: the
    # components move with an arbitrary sign convention, the angle does not.
    quats = [(p["orientation"]["x"], p["orientation"]["y"],
              p["orientation"]["z"], p["orientation"]["w"]) for p in poses]
    angles = []
    for i in range(1, len(quats)):
        dot = abs(sum(a * b for a, b in zip(quats[0], quats[i])))
        angles.append(math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot)))))

    # THE SAME SPREAD, ON THE QUANTITY THAT IS ACTUALLY OBSERVABLE.
    #
    # `pose.x/y/z` is the CAD model's OWN origin, and Cylinder5's sits 141 mm
    # outside its geometry. The part's two ends are interchangeable — the
    # registry declares `fold: 2` about z from a MEASURED 0.77 mm residual — so
    # the estimator may return either of two poses that describe the same
    # object in the same place. Under that swap the body centre barely moves
    # while the model origin swings through twice its offset, which shows up as
    # tens of millimetres of "spread" that is nothing of the sort.
    #
    # So both are reported. A large origin spread beside a small centre spread
    # is the end swap; both large is genuine instability.
    tasks = [observation_of(b).get("task") or {} for b in succeeded(batches)]
    bodies = [t["object_center_mm"] for t in tasks if t.get("object_center_mm")]
    body: Dict[str, Any] = {}
    if len(bodies) >= 2:
        body_mean = [sum(c[i] for c in bodies) / len(bodies) for i in range(3)]
        body_radial = [math.dist(c, body_mean) for c in bodies]
        body = {
            "x_mm": spread([c[0] for c in bodies]),
            "y_mm": spread([c[1] for c in bodies]),
            "z_mm": spread([c[2] for c in bodies]),
            "radial_mm": {"max": round(max(body_radial), 3),
                          "mean": round(sum(body_radial) / len(body_radial), 3)},
        }

    # THE TUBE AXIS AS A LINE, folded at 90 deg. Which way the axis points is
    # the end swap again; the LINE it lies along is what the task needs, and it
    # is the part that must be stable.
    axes = [t["tube_axis_line"] for t in tasks if t.get("tube_axis_line")]
    axis_deg: List[float] = []
    for other in axes[1:]:
        dot = abs(sum(a * b for a, b in zip(axes[0], other)))
        axis_deg.append(math.degrees(math.acos(max(-1.0, min(1.0, dot)))))

    return {
        "frames": len(poses),
        "model_frame_origin": {
            "x_mm": spread([c[0] for c in centres]),
            "y_mm": spread([c[1] for c in centres]),
            "z_mm": spread([c[2] for c in centres]),
            "radial_mm": {"max": round(max(radial), 3),
                          "mean": round(sum(radial) / len(radial), 3)},
            "note": ("the CAD model's own origin, 141 mm outside Cylinder5's "
                     "geometry — moves under the declared end swap"),
        },
        "object_centre": body or {"note": "not reported by the observation"},
        "orientation_vs_first_deg": {
            "max": round(max(angles), 3) if angles else 0.0,
            "mean": round(sum(angles) / len(angles), 3) if angles else 0.0,
            "note": ("includes the declared 180 deg end swap, which is NOT an "
                     "error: the registry records it as unobservable"),
        },
        "tube_axis_line_deg": {
            "max": round(max(axis_deg), 3) if axis_deg else 0.0,
            "mean": round(sum(axis_deg) / len(axis_deg), 3) if axis_deg else 0.0,
            "note": "axis as a LINE, folded at 90 deg — the task-relevant part",
        },
        "note": ("REPEATABILITY over a stationary scene. It is NOT accuracy: "
                 "no ground truth exists for the physical part, and an "
                 "estimator can be precisely wrong."),
    }


def plausibility(batch: Any, model, segmentation: Dict[str, Any],
                 intrinsics: Dict[str, float]) -> Dict[str, Any]:
    """Cheap physical checks that need no ground truth.

    Each one can only REFUTE a pose, never confirm it. Stated that way because
    a green tick beside "plausible" would be read as "correct". They are the
    checks that ARE available when nothing knows where the tube really is:
    does the CAD land where the pixels are, and is it the size and distance the
    depth image already measured independently.
    """
    mask_pixels = int(segmentation.get("mask_pixels", 0))
    pose = pose_of(batch)
    z_mm = float(pose["z_mm"])
    checks: Dict[str, Any] = {}

    # 1. Is the object where the depth image says something is?
    #
    # THE BODY CENTRE, NOT THE POSE ORIGIN. `pose.x/y/z` is where the CAD
    # model's OWN origin lands, and Cylinder5 is drawn obliquely with its
    # origin 141 mm outside the geometry — comparing that against a range read
    # off the tube's surface would fail by exactly that offset and look like a
    # bad pose. The domain already publishes the physical centre for this
    # reason; `task.object_center_mm` is what a grasp would target.
    task = observation_of(batch).get("task") or {}
    centre = task.get("object_center_mm")
    surface_mm = segmentation.get("mask_median_range_mm")
    checks["pose_origin_z_mm"] = round(z_mm, 1)
    checks["pose_reference_point"] = pose.get("reference_point")
    if centre:
        checks["object_centre_z_mm"] = round(float(centre[2]), 1)
    if centre and surface_mm and model.diameter_mm:
        # The depth image sees the NEAR SURFACE; the centre of a tube of
        # diameter D sits about D/2 behind it. Two independent measurements —
        # one from the depth image, one from the registered CAD — that must
        # agree to within the sensor's own noise if the pose is on this object.
        expected = float(surface_mm) + model.diameter_mm / 2.0
        error = float(centre[2]) - expected
        checks["centre_depth_expected_mm"] = round(expected, 1)
        checks["centre_depth_error_mm"] = round(error, 1)
        checks["centre_depth_plausible"] = abs(error) <= 25.0

    # 2. Does the CAD land ON the pixels that were segmented?
    #    The projected body centre against the mask centroid: a pose on a
    #    NEIGHBOURING object passes every scale check and fails this one.
    if centre and float(centre[2]) > 0:
        cx = intrinsics.get("cx")
        cy = intrinsics.get("cy")
        mask_centroid = segmentation.get("mask_centroid_px")
        if cx is not None and cy is not None and mask_centroid:
            u = intrinsics["fx"] * float(centre[0]) / float(centre[2]) + cx
            v = intrinsics["fy"] * float(centre[1]) / float(centre[2]) + cy
            offset = math.dist((u, v), (float(mask_centroid[0]),
                                        float(mask_centroid[1])))
            checks["cad_centroid_px"] = [round(u, 1), round(v, 1)]
            checks["mask_centroid_px"] = [round(float(v_), 1)
                                          for v_ in mask_centroid]
            checks["centroid_offset_px"] = round(offset, 1)
            checks["centroid_offset_mm"] = round(
                offset * float(centre[2]) / intrinsics["fx"], 1)
            # A tube is long and thin, so a centroid that agrees to a few
            # millimetres cannot be sitting on a different object.
            checks["centroid_plausible"] = offset <= 40.0

    # 3. Is the tube axis pointing along the tube in the image?
    axis = task.get("tube_axis_line")
    if axis:
        checks["tube_axis_line"] = [round(float(a), 4) for a in axis]

    checks["depth_plausible"] = 200.0 < z_mm < 3000.0

    # 2. Does the CAD, at that range, subtend roughly the mask's area?
    #    A pose a factor of ten out in depth fails this by a factor of a
    #    hundred in area — the classic units error, made visible.
    range_mm = float(segmentation.get("mask_median_range_mm") or z_mm)
    if model.length_mm and model.diameter_mm and range_mm > 0:
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        px_long = model.length_mm * fx / range_mm
        px_across = model.diameter_mm * fy / range_mm
        expected = px_long * px_across
        checks["cad_projected_area_px"] = round(expected, 1)
        checks["mask_pixels"] = mask_pixels
        ratio = mask_pixels / expected if expected > 0 else 0.0
        checks["mask_to_cad_area_ratio"] = round(ratio, 3)
        # A tube's silhouette is a rectangle only when it is side-on; fore-
        # shortening shrinks it and the mask carries depth-edge bleed. Wide
        # bounds on purpose: this is a units/scale check, not a fit metric.
        checks["area_plausible"] = 0.25 <= ratio <= 2.5
    return checks


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="cylinder5",
                        help="the CAD model an operator placed in view")
    parser.add_argument("--frames", type=int, default=5,
                        help="frames captured, and poses estimated, for spread")
    parser.add_argument("--refine-iterations", type=int, default=5)
    parser.add_argument("--plane-tolerance-mm", type=float, default=None)
    parser.add_argument("--min-height-mm", type=float, default=None)
    parser.add_argument("--max-height-mm", type=float, default=None)
    parser.add_argument("--roi-radius-mm", type=float, default=None)
    parser.add_argument(
        "--roi", default="",
        metavar="x0,y0,x1,y1",
        help="operator ROI in COLOUR-IMAGE PIXELS. It says WHERE to look, not "
             "WHAT is there: identity stays --model. Use it when the bench "
             "holds other objects, so the single-object rules apply to the "
             "region you meant.")
    parser.add_argument("--component", default=None,
                        choices=["largest", "centre"])
    parser.add_argument("--segment-only", action="store_true",
                        help="capture and segment, then stop — no inference")
    parser.add_argument("--dataset", default="",
                        help="reuse an existing capture instead of taking one")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    roi = None
    if args.roi:
        try:
            roi = [int(round(float(v))) for v in args.roi.replace(" ", "").split(",")]
        except ValueError:
            say(f"--roi must be four numbers x0,y0,x1,y1; got {args.roi!r}")
            return 2
        if len(roi) != 4:
            say(f"--roi needs exactly four values, got {len(roi)}")
            return 2
        say(f"operator ROI {roi} px — WHERE to look. The object's identity is "
            f"--model {args.model}, which an ROI can never establish.")
    options = {k: v for k, v in (
        ("roi_px", roi),
        ("plane_tolerance_mm", args.plane_tolerance_mm),
        ("min_height_mm", args.min_height_mm),
        ("max_height_mm", args.max_height_mm),
        ("roi_radius_mm", args.roi_radius_mm),
        ("component", args.component)) if v is not None}

    from wisepack_core.rgbd import load_object_registry
    registry = load_object_registry(repo_root=REPO)
    model = registry.models.get(args.model)
    if model is None:
        say(f"unknown object model {args.model!r}; known: "
            + ", ".join(sorted(registry.models)))
        return 2
    if not model.mesh_exists(registry.root):
        say(f"{args.model} has no mesh at {model.resolved_path(registry.root)}")
        return 2
    say(f"object: {args.model} — {model.resolved_path(registry.root)} "
        f"({model.mesh_units}, declared {model.diameter_mm}x{model.length_mm} mm)")

    device = require_camera()

    if args.dataset:
        dataset_name = args.dataset
        capture_document: Dict[str, Any] = {}
        say(f"reusing capture {dataset_name}")
    else:
        capture_document = capture(args.model, args.frames)
        dataset_name = os.path.basename(capture_document["root"])

    segmentation = segment(dataset_name, 0, options)
    report_segmentation(segmentation)
    for kind, name in (("rgb", "rgb.jpg"), ("depth", "depth_aligned.jpg"),
                       ("mask", "mask.jpg"),
                       ("segmentation", "mask_overlay.jpg")):
        fetch_image(kind, os.path.join(OUT, name))

    print()
    say(f"look at the mask before trusting any pose:  {OUT}/mask_overlay.jpg")

    if not segmentation.get("mask_valid"):
        say("the mask is NOT usable, so nothing is estimated from it. No mask "
            "is fabricated to get past this step.")
        say(f"reason: {segmentation.get('reason')}")
        return 4
    if args.segment_only:
        say("--segment-only: stopping before inference, as asked")
        return 0

    batches = estimate(dataset_name, args.model, args.frames, options,
                       args.refine_iterations)
    ok = succeeded(batches)
    if not ok:
        say("no frame produced a pose; see the errors above")
        return 5

    fetch_image("overlay", os.path.join(OUT, "pose_overlay.jpg"))
    copy_source_frames(dataset_name, segmentation.get("frame_index", 0))
    first = ok[0]
    observation = observation_of(first)
    pose = observation["pose"]
    spread = repeatability(ok)
    # THE INTRINSICS THIS FRAME WAS SEGMENTED WITH — the same matrix the worker
    # read from the device and wrote into the capture's cam_K.txt. Taken from
    # the segmentation response rather than the capture document so a `--dataset`
    # re-run, which has no capture document, checks against the same numbers.
    matrix = segmentation["intrinsics"]
    intrinsics = {"fx": matrix[0][0], "fy": matrix[1][1],
                  "cx": matrix[0][2], "cy": matrix[1][2]}
    checks = plausibility(first, model, segmentation, intrinsics)

    print()
    print("  POSE — camera optical frame, from the physical D435")
    print(f"    frame_id                {observation['frame_id']}")
    print(f"    model_id                {observation['object_model_id']}")
    print(f"    perception_method       {observation['perception_method']}")
    print(f"    acquisition             {first.acquisition}")
    quaternion = pose["orientation"]
    print(f"    x, y, z (mm)            {pose['x_mm']:.1f}, {pose['y_mm']:.1f}, "
          f"{pose['z_mm']:.1f}")
    print(f"    quaternion (x,y,z,w)    "
          + ", ".join(f"{float(quaternion[k]):.5f}" for k in "xyzw"))
    print(f"    pose_valid              {pose['valid']}")
    print(f"    measured_dof            {', '.join(pose['measured_dof'])}")
    print(f"    workarea_pose_available {pose['workarea_pose_available']}   "
          "(no camera-to-work-area extrinsic has been measured)")

    print()
    print("  REPEATABILITY — stationary scene, NOT accuracy")
    for label, key in (("body centre", "object_centre"),
                       ("model origin", "model_frame_origin")):
        block = spread.get(key) or {}
        if "radial_mm" not in block:
            continue
        print(f"    {label}")
        for axis in ("x_mm", "y_mm", "z_mm"):
            s = block[axis]
            print(f"      {axis:18} sd {s['sd']:.3f}  range {s['range']:.3f}")
        print(f"      {'radial':18} max {block['radial_mm']['max']:.3f} mm")
    if "orientation_vs_first_deg" in spread:
        print(f"    orientation          max "
              f"{spread['orientation_vs_first_deg']['max']:.3f} deg "
              "(includes the declared 180 deg end swap)")
    if spread.get("tube_axis_line_deg"):
        print(f"    tube axis as a LINE  max "
              f"{spread['tube_axis_line_deg']['max']:.3f} deg")

    print()
    print("  PLAUSIBILITY — can only refute, never confirm")
    for key, value in checks.items():
        print(f"    {key:24} {value}")

    # LIVE AND REPLAYED ARE DIFFERENT CLAIMS, and the difference is recorded
    # here rather than inferred by whoever reads the file. Both are PHYSICAL —
    # a replay reads frames a real D435 produced — but a recorded capture is not
    # evidence that the camera worked just now, and a dashboard that showed one
    # as the other would be claiming a live sensor on a machine with none.
    live = not args.dataset
    document = {
        "device": device,
        "dataset": dataset_name,
        "model_id": args.model,
        "selected_profile": {"width": WIDTH, "height": HEIGHT, "fps": FPS},
        "acquisition_backend": "realsense",
        "provenance": "measured",
        "run_mode": "live" if live else "replay",
        "run_label": ("LIVE PHYSICAL D435" if live
                      else "RECORDED PHYSICAL D435 DATA"),
        "run_note": ("Acquired from the physical D435 during this run."
                     if live else
                     "Replayed from frames a physical D435 recorded earlier. "
                     "Real sensor data, NOT simulation, and NOT a live camera."),
        "operator_roi_px": roi,
        "roi_note": ("The ROI says WHERE to look. Object identity is model_id, "
                     "stated by the operator, and is never inferred from it."),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "capture": capture_document,
        "segmentation": segmentation,
        "observation": observation,
        "batch": first.to_dict() if hasattr(first, "to_dict") else {},
        "repeatability": spread,
        "plausibility": checks,
        "images": {k: os.path.join(OUT, v) for k, v in (
            ("rgb", "rgb.jpg"), ("depth", "depth_aligned.jpg"),
            ("mask", "mask.jpg"), ("mask_overlay", "mask_overlay.jpg"),
            ("pose_overlay", "pose_overlay.jpg"))},
        "accuracy_note": (
            "No ground truth exists for the physical part. Repeatability is "
            "reported; accuracy is NOT measured and is not claimed."),
    }
    path = os.path.join(OUT, "physical_c5.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, default=str)
    print()
    say(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
