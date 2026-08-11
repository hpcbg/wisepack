"""The physical D435 -> FoundationPose pipeline, in ONE place.

WHY THIS MODULE EXISTS. The same acquisition has two callers — the
`./scripts/physical_c5.sh` CLI and the dashboard's *Acquire & estimate* button —
and two copies of "capture, segment, refuse, estimate" would agree only until
somebody edited one. A pose an operator sees in the browser must be produced by
exactly the code the CLI runs, or the two become different measurements wearing
one name.

WHAT IT DOES NOT DO, and these are the load-bearing absences:

* NO SECOND ESTIMATOR. Inference is the FoundationPose worker's, reached through
  the ordinary provider. Nothing here registers a mesh.
* NO FABRICATED MASK. If segmentation reports `mask_valid: false` this RAISES
  with the reason. It never falls back to a previous pose, to simulation, to
  Isaac ground truth or to the planar detector.
* NO WORK-AREA POSE. The result stays in `camera_color_optical_frame`. No
  extrinsic is invented, so `workarea_pose_available` stays false.
* NO OBJECT IDENTIFICATION. `model_id` is an input. The ROI says WHERE to look
  and can never say WHAT is there.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(REPO, "perception"),
              os.path.join(REPO, "wisepack_ws", "src", "wisepack_core")):
    if _path not in sys.path:                                 # pragma: no cover
        sys.path.insert(0, _path)

WORKER = f"http://127.0.0.1:{os.environ.get('WISEPACK_FP_PORT', '22201')}"

#: Where the result and its images live. One directory, so the dashboard panel
#: and the CLI read the same artefact.
OUT = os.path.join(REPO, ".cache-perception", "physical-c5")

#: Where `capture_dataset` writes on the HOST — the worker's /captures mount.
CAPTURES_ON_HOST = os.path.join(REPO, ".cache-perception", "rgbd-captures")

#: The profile validated on this bench. Stated rather than negotiated, so a
#: silent renegotiation cannot change the intrinsics underneath a comparison.
WIDTH, HEIGHT, FPS = 1280, 720, 30

#: A uint16 millimetre depth PNG, which is what `capture_dataset` writes.
DEPTH_SCALE_MM = 1.0

#: The images the panel and the CLI both show.
ARTIFACTS = (("rgb", "rgb.jpg"), ("depth", "depth_aligned.jpg"),
             ("mask", "mask.jpg"), ("segmentation", "mask_overlay.jpg"))


class PhysicalResult:
    """What one physical acquisition produced, for both of its callers.

    THE BATCH IS RETURNED, NOT REBUILT. The provider already constructed an
    `ObservationBatch` on the way to writing this document; handing the caller
    the document alone would force the dashboard to reconstruct one from JSON,
    and a second construction is a second place for the CAD geometry, the frame
    or the provenance to be assembled differently.

    `document` is the JSON-serialisable artefact the panel renders. `batch` is
    the live domain object the workflow plans from. They describe one
    acquisition and are never produced separately.
    """

    def __init__(self, document: Dict[str, Any], batch: Any) -> None:
        self.document = document
        self.batch = batch


class PhysicalAcquisitionError(RuntimeError):
    """A refusal, naming the STAGE that refused and why.

    The stage matters: "the camera is not there", "the mask is unusable" and
    "the estimator failed" send an operator to three different places, and one
    "acquisition failed" would send them to none of them.
    """

    def __init__(self, stage: str, reason: str,
                 detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason
        self.detail = detail or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": False, "stage": self.stage, "reason": self.reason,
                **self.detail}


def _noop(_message: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


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
        return None, f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}"
    except Exception as exc:                                 # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def get(path: str, timeout: float = 30.0) -> Tuple[Optional[Any], str]:
    try:
        with urllib.request.urlopen(f"{WORKER}{path}", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")), ""
    except Exception as exc:                                 # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def fetch_image(kind: str, destination: str) -> bool:
    try:
        with urllib.request.urlopen(f"{WORKER}/image/{kind}", timeout=30) as r:
            data = r.read()
    except Exception:                                        # noqa: BLE001
        return False
    with open(destination, "wb") as handle:
        handle.write(data)
    return True


def copy_source_frames(dataset: str, index: int = 0) -> None:
    """The unmodified PNGs the estimate was computed from.

    THE JPEGS ARE DIAGNOSTICS, NOT DATA: they are re-encoded and colour-mapped
    for looking at, and an evaluator asking "what did it see" must get the
    uint16 depth and the raw colour rather than a picture of them.
    """
    root = os.path.join(CAPTURES_ON_HOST, dataset)
    for kind, name in (("rgb", "source_rgb.png"),
                       ("depth", "source_depth_aligned_uint16.png")):
        source = os.path.join(root, kind, f"{index:06d}.png")
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(OUT, name))
    intrinsics = os.path.join(root, "cam_K.txt")
    if os.path.isfile(intrinsics):
        shutil.copy2(intrinsics, os.path.join(OUT, "cam_K.txt"))


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def require_camera() -> Dict[str, Any]:
    """The physical D435, or a refusal naming the layer that is missing."""
    health, error = get("/health")
    if health is None:
        raise PhysicalAcquisitionError(
            "worker",
            f"the FoundationPose worker is not answering at {WORKER}: {error}. "
            "Start it: ./scripts/setup_foundationpose.sh --no-build --run")
    if not health.get("rgbd_camera_available"):
        raise PhysicalAcquisitionError(
            "camera",
            "the worker has no RGB-D camera: "
            + ("; ".join(health.get("blocked_by") or ["no reason given"]))
            + ". Diagnose it: ./scripts/realsense_diagnose.sh")
    if not health.get("inference_available"):
        raise PhysicalAcquisitionError(
            "inference",
            "FoundationPose inference is unavailable: "
            + "; ".join(health.get("blocked_by") or ["no reason given"]))
    camera, error = get("/camera")
    if camera is None or not camera.get("available"):
        raise PhysicalAcquisitionError(
            "camera", f"the camera could not be described: "
                      f"{error or (camera or {}).get('reason')}")
    device = camera["device"]
    if {"width": WIDTH, "height": HEIGHT, "fps": FPS} not in \
            device.get("synchronised_profiles", []):
        raise PhysicalAcquisitionError(
            "profile",
            f"the device does not offer {WIDTH}x{HEIGHT}@{FPS} for colour+depth")
    return device


def capture(model_id: str, frames: int) -> Dict[str, Any]:
    """One dataset of `frames` synchronised, aligned RGB-D frames.

    ONE CAPTURE, SEVERAL FRAMES: the scene is stationary, so the frames differ
    only by sensor noise — which is what makes the spread of the poses estimated
    from them a measure of repeatability.
    """
    document, error = post("/camera/capture", {
        "model_id": model_id, "frames": frames,
        "width": WIDTH, "height": HEIGHT, "fps": FPS, "align": True})
    if document is None:
        raise PhysicalAcquisitionError("capture", f"the capture failed: {error}")
    if not document.get("alignment_verified"):
        raise PhysicalAcquisitionError(
            "capture",
            "the capture does not carry verified alignment; a mask drawn on the "
            "colour image would select the wrong depth pixels")
    return document


def segment(dataset: str, frame: int, options: Dict[str, Any]
            ) -> Dict[str, Any]:
    document, error = post("/segment", {
        "dataset": dataset, "frame": frame, "method": "depth_plane_foreground",
        "depth_scale_mm": DEPTH_SCALE_MM, "segmentation": options})
    if document is None:
        raise PhysicalAcquisitionError("segmentation",
                                       f"segmentation failed: {error}")
    return document


def estimate(dataset: str, model_id: str, frames: int, options: Dict[str, Any],
             refine_iterations: int, log: Callable[[str], None] = _noop
             ) -> List[Any]:
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
            batch_id=f"physical-{model_id}-{index + 1}",
            mask_source="depth_plane_foreground", segmentation=options)
        elapsed = (time.monotonic() - started) * 1000.0
        if str(getattr(batch.status, "value", batch.status)) != "ok":
            log(f"frame {index}: estimation FAILED — {batch.error}")
        else:
            pose = pose_of(batch)
            log(f"frame {index}: x {pose['x_mm']:.1f}  y {pose['y_mm']:.1f}  "
                f"z {pose['z_mm']:.1f} mm   ({elapsed:.0f} ms)")
        batches.append(batch)
    return batches


def succeeded(batches: List[Any]) -> List[Any]:
    return [b for b in batches
            if str(getattr(b.status, "value", b.status)) == "ok" and b.observations]


def observation_of(batch: Any) -> Dict[str, Any]:
    """The SERIALISED observation — the same contract the dashboard renders."""
    return batch.observations[0].to_dict()


def pose_of(batch: Any) -> Dict[str, Any]:
    return observation_of(batch)["pose"]


# --------------------------------------------------------------------------- #
# Evidence, without ground truth
# --------------------------------------------------------------------------- #


def repeatability(batches: List[Any]) -> Dict[str, Any]:
    """The spread of poses over a STATIONARY scene.

    NOT ACCURACY, and the distinction is not pedantic: an estimator can return
    the same wrong pose every time.
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
    quats = [(p["orientation"]["x"], p["orientation"]["y"],
              p["orientation"]["z"], p["orientation"]["w"]) for p in poses]
    angles = []
    for i in range(1, len(quats)):
        dot = abs(sum(a * b for a, b in zip(quats[0], quats[i])))
        angles.append(math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot)))))

    # THE SAME SPREAD ON THE QUANTITY THAT IS OBSERVABLE. `pose.x/y/z` is the
    # CAD model's own origin, which for an obliquely drawn part sits well
    # outside the geometry; under a declared end swap it moves while the body
    # centre does not. Both are reported so the difference is visible.
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
            "note": "the CAD model's own origin — moves under a declared end swap",
        },
        "object_centre": body or {"note": "not reported by the observation"},
        "orientation_vs_first_deg": {
            "max": round(max(angles), 3) if angles else 0.0,
            "mean": round(sum(angles) / len(angles), 3) if angles else 0.0,
            "note": "includes any declared end swap, which is NOT an error",
        },
        "tube_axis_line_deg": {
            "max": round(max(axis_deg), 3) if axis_deg else 0.0,
            "mean": round(sum(axis_deg) / len(axis_deg), 3) if axis_deg else 0.0,
            "note": "axis as a LINE, folded at 90 deg — the task-relevant part",
        },
        "note": ("REPEATABILITY over a stationary scene. It is NOT accuracy: no "
                 "ground truth exists for the physical part."),
    }


def plausibility(batch: Any, model, segmentation: Dict[str, Any],
                 intrinsics: Dict[str, float]) -> Dict[str, Any]:
    """Cheap physical checks that need no ground truth.

    Each can only REFUTE a pose, never confirm it.
    """
    mask_pixels = int(segmentation.get("mask_pixels", 0))
    pose = pose_of(batch)
    z_mm = float(pose["z_mm"])
    checks: Dict[str, Any] = {}

    task = observation_of(batch).get("task") or {}
    centre = task.get("object_center_mm")
    surface_mm = segmentation.get("mask_median_range_mm")
    checks["pose_origin_z_mm"] = round(z_mm, 1)
    checks["pose_reference_point"] = pose.get("reference_point")
    if centre:
        checks["object_centre_z_mm"] = round(float(centre[2]), 1)
    if centre and surface_mm and model.diameter_mm:
        # The depth image sees the NEAR SURFACE; the centre of a tube sits about
        # D/2 behind it. Two independent estimates that must agree.
        expected = float(surface_mm) + model.diameter_mm / 2.0
        error = float(centre[2]) - expected
        checks["centre_depth_expected_mm"] = round(expected, 1)
        checks["centre_depth_error_mm"] = round(error, 1)
        checks["centre_depth_plausible"] = abs(error) <= 25.0

    if centre and float(centre[2]) > 0:
        cx, cy = intrinsics.get("cx"), intrinsics.get("cy")
        mask_centroid = segmentation.get("mask_centroid_px")
        if cx is not None and cy is not None and mask_centroid:
            u = intrinsics["fx"] * float(centre[0]) / float(centre[2]) + cx
            v = intrinsics["fy"] * float(centre[1]) / float(centre[2]) + cy
            offset = math.dist((u, v), (float(mask_centroid[0]),
                                        float(mask_centroid[1])))
            checks["cad_centroid_px"] = [round(u, 1), round(v, 1)]
            checks["mask_centroid_px"] = [round(float(x), 1) for x in mask_centroid]
            checks["centroid_offset_px"] = round(offset, 1)
            checks["centroid_offset_mm"] = round(
                offset * float(centre[2]) / intrinsics["fx"], 1)
            checks["centroid_plausible"] = offset <= 40.0

    axis = task.get("tube_axis_line")
    if axis:
        checks["tube_axis_line"] = [round(float(a), 4) for a in axis]
    checks["depth_plausible"] = 200.0 < z_mm < 3000.0

    range_mm = float(segmentation.get("mask_median_range_mm") or z_mm)
    if model.length_mm and model.diameter_mm and range_mm > 0:
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        expected = (model.length_mm * fx / range_mm) * (model.diameter_mm * fy / range_mm)
        checks["cad_projected_area_px"] = round(expected, 1)
        checks["mask_pixels"] = mask_pixels
        ratio = mask_pixels / expected if expected > 0 else 0.0
        checks["mask_to_cad_area_ratio"] = round(ratio, 3)
        checks["area_plausible"] = 0.25 <= ratio <= 2.5
    return checks


# --------------------------------------------------------------------------- #
# The whole thing, for either caller
# --------------------------------------------------------------------------- #


def eligible_models(repo_root: str = REPO) -> List[Dict[str, Any]]:
    """Models that have a mesh AND declare `foundationpose_rgbd`.

    FROM THE REGISTRY, never a hard-coded list: a model-based estimator can only
    locate a shape it has, and which shapes exist is configuration.
    """
    from wisepack_core.rgbd import load_object_registry              # noqa: PLC0415
    registry = load_object_registry(repo_root=repo_root)
    listing = []
    for model in registry.models.values():
        if not model.supports("foundationpose_rgbd"):
            continue
        if not model.mesh_exists(registry.root):
            continue
        label = (f"{model.model_id} — D{model.diameter_mm} x L{model.length_mm} mm"
                 if model.diameter_mm and model.length_mm else model.model_id)
        listing.append({
            "model_id": model.model_id,
            # SAID IN THE OPTION ITSELF. A regression asset that reads like an
            # ordinary part is how one came to be selected for a live
            # acquisition of a tube.
            "label": label + (" (reference asset — not a WISEPACK part)"
                              if model.is_reference else ""),
            "diameter_mm": model.diameter_mm,
            "length_mm": model.length_mm,
            "object_type": model.object_type,
            "is_reference": model.is_reference,
        })
    # WORKPIECES FIRST, reference assets last. Ordering is not a substitute for
    # the declared default, but a list that opens on a bolt invites the mistake
    # the default exists to prevent.
    return sorted(listing, key=lambda m: (m["is_reference"], m["model_id"]))


def run(model_id: str, roi_px: Optional[List[int]] = None, frames: int = 5,
        refine_iterations: int = 5, dataset: str = "",
        segmentation_options: Optional[Dict[str, Any]] = None,
        log: Callable[[str], None] = _noop) -> "PhysicalResult":
    """Acquire, segment, refuse or estimate — and write the artefact.

    RAISES `PhysicalAcquisitionError` at the stage that refused. Returns the
    same document `.cache-perception/physical-c5/physical_c5.json` holds, which
    is what the dashboard panel renders.
    """
    from wisepack_core.rgbd import load_object_registry              # noqa: PLC0415

    os.makedirs(OUT, exist_ok=True)
    options: Dict[str, Any] = dict(segmentation_options or {})
    if roi_px:
        options["roi_px"] = [int(v) for v in roi_px]

    registry = load_object_registry(repo_root=REPO)
    model = registry.models.get(model_id)
    if model is None:
        raise PhysicalAcquisitionError(
            "model", f"unknown object model {model_id!r}; known: "
                     + (", ".join(sorted(registry.models)) or "(none)"))
    if not model.supports("foundationpose_rgbd"):
        raise PhysicalAcquisitionError(
            "model", f"{model_id} does not declare the foundationpose_rgbd method")
    if not model.mesh_exists(registry.root):
        raise PhysicalAcquisitionError(
            "model", f"{model_id} has no mesh at "
                     f"{model.resolved_path(registry.root) or '(unset)'}")

    device = require_camera()
    live = not dataset
    if live:
        capture_document = capture(model_id, frames)
        dataset = os.path.basename(capture_document["root"])
        log(f"captured {dataset}")
    else:
        capture_document = {}
        log(f"replaying capture {dataset}")

    segmentation = segment(dataset, 0, options)
    for kind, name in ARTIFACTS:
        fetch_image(kind, os.path.join(OUT, name))

    # THE REFUSAL. No mask is fabricated, and no previous result is substituted.
    if not segmentation.get("mask_valid"):
        raise PhysicalAcquisitionError(
            "segmentation",
            segmentation.get("reason") or "segmentation produced no usable mask",
            {"segmentation": segmentation, "dataset": dataset,
             "images": [k for k, _ in ARTIFACTS]})

    batches = estimate(dataset, model_id, frames, options, refine_iterations, log)
    ok = succeeded(batches)
    if not ok:
        raise PhysicalAcquisitionError(
            "estimation",
            "; ".join(b.error for b in batches if getattr(b, "error", ""))
            or "no frame produced a pose",
            {"segmentation": segmentation, "dataset": dataset})

    fetch_image("overlay", os.path.join(OUT, "pose_overlay.jpg"))
    copy_source_frames(dataset, int(segmentation.get("frame_index", 0)))

    first = ok[0]
    matrix = segmentation["intrinsics"]
    intrinsics = {"fx": matrix[0][0], "fy": matrix[1][1],
                  "cx": matrix[0][2], "cy": matrix[1][2]}
    document = {
        "device": device,
        "dataset": dataset,
        "model_id": model_id,
        "selected_profile": {"width": WIDTH, "height": HEIGHT, "fps": FPS},
        "acquisition_backend": "realsense",
        "provenance": "measured",
        # LIVE AND REPLAYED ARE DIFFERENT CLAIMS. Both are physical sensor data;
        # a recording is not evidence that the camera worked just now.
        "run_mode": "live" if live else "replay",
        "run_label": ("LIVE PHYSICAL D435" if live
                      else "RECORDED PHYSICAL D435 DATA"),
        "run_note": ("Acquired from the physical D435 during this run." if live
                     else "Replayed from frames a physical D435 recorded "
                          "earlier. Real sensor data, NOT simulation, and NOT a "
                          "live camera."),
        "operator_roi_px": options.get("roi_px"),
        "roi_note": ("The ROI says WHERE to look. Object identity is model_id, "
                     "stated by the operator, and is never inferred from it."),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "capture": capture_document,
        "segmentation": segmentation,
        "observation": observation_of(first),
        "batch": first.to_dict() if hasattr(first, "to_dict") else {},
        "repeatability": repeatability(ok),
        "plausibility": plausibility(first, model, segmentation, intrinsics),
        "images": {k: os.path.join(OUT, v) for k, v in
                   (*ARTIFACTS, ("overlay", "pose_overlay.jpg"))},
        "accuracy_note": ("No ground truth exists for the physical part. "
                          "Repeatability is reported; accuracy is NOT measured "
                          "and is not claimed."),
    }
    with open(os.path.join(OUT, "physical_c5.json"), "w", encoding="utf-8") as h:
        json.dump(document, h, indent=2, default=str)
    # THE BATCH RIDES ALONG. `first` is the batch the provider built from the
    # worker's response, with the CAD geometry the registry declares already on
    # its observation — which is exactly what the packing layer needs and what
    # a JSON round-trip would make somebody reassemble by hand.
    return PhysicalResult(document, first)


__all__ = ["PhysicalAcquisitionError", "PhysicalResult", "run", "eligible_models", "OUT",
           "WIDTH", "HEIGHT", "FPS", "DEPTH_SCALE_MM", "ARTIFACTS",
           "require_camera", "capture", "segment", "estimate", "succeeded",
           "observation_of", "pose_of", "repeatability", "plausibility",
           "fetch_image", "copy_source_frames"]
