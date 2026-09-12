"""The physical D435 -> whole-scene -> ObservationBatch pipeline, in ONE place.

THE SAME SHAPE AS `physical_pipeline.run`, for the same reason: the dashboard's
*Acquire scene* button and the `./scripts/physical_scene.sh` CLI must produce
one measurement by one code path. The capture is the FoundationPose worker's
(`/camera/capture`, warmed and aligned exactly as the single-object path); the
segmentation, classification and observation building run on the host from the
capture's own PNGs, because the worker's segmentation keeps one object by
design and the GPU is not needed for a footprint.

WHAT IT REFUSES, with the stage that refused: no worker, no camera, a capture
without verified alignment, a frame with no work plane, a scene with nothing
classifiable. Nothing is substituted for any of them.

WHAT IT LEAVES OUT, and says so: instances inside a configured ignore region,
footprints matching no demo class, and objects the CONFIGURED demo workcell
cannot place (outside the work-area bounds or the robot's reach band). Each is
listed in the document and in the batch's `detector_status` with its reason.
The batch carries only the objects the scene can be synchronized from, so the
downstream chain is exactly the single-object one, repeated N times.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(REPO, "perception"),
              os.path.join(REPO, "wisepack_ws", "src", "wisepack_core")):
    if _path not in sys.path:                                 # pragma: no cover
        sys.path.insert(0, _path)

from physical_pipeline import (CAPTURES_ON_HOST, FPS, HEIGHT, WIDTH,   # noqa: E402
                               PhysicalAcquisitionError, PhysicalResult,
                               WORKER, capture, get)

#: Where the scene result and its images live: the dashboard panel and the CLI
#: read the same artefact. Separate from the single-object `physical-c5`
#: directory so the two results can never be mistaken for one another.
OUT = os.path.join(REPO, ".cache-perception", "physical-scene")
RESULT = os.path.join(OUT, "physical_scene.json")

ARTIFACTS = (("rgb", "rgb.jpg"), ("depth", "depth_aligned.jpg"),
             ("mask", "mask_overlay.jpg"), ("overlay", "scene_overlay.jpg"))

#: The capture is named after what it is, not after one model.
CAPTURE_NAME = "scene"

#: A live capture whose depth image has fewer valid pixels than this is taken
#: again; the D435's depth needs a moment after the stream opens.
MIN_VALID_DEPTH_FRACTION = 0.90
MAX_CAPTURE_ATTEMPTS = 3


def _noop(_message: str) -> None:
    pass


def require_rgbd_camera() -> Dict[str, Any]:
    """The physical D435 behind the worker, or a refusal naming the layer."""
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


def _placeability(observations, layout=None) -> Dict[str, Dict[str, Any]]:
    """Which observations the CONFIGURED demo workcell can place, and why not.

    THE SAME CHAIN THE SYNCHRONIZER RUNS, called here so an object the robot
    cannot reach is reported at acquisition time with its reason rather than
    holding the whole scene at the gate later. The transform is the configured
    demo transform and the reach band is the selected layout's; neither is
    relaxed here.
    """
    from wisepack_core.isaac_transform import DEFAULT_LAYOUT          # noqa: PLC0415
    from wisepack_core.scene_sync import SceneSyncRefused, transform_observation  # noqa: PLC0415
    from wisepack_core.workcell import load_workcell                  # noqa: PLC0415

    frames = load_workcell(repo_root=REPO)
    layout = layout or DEFAULT_LAYOUT
    verdicts: Dict[str, Dict[str, Any]] = {}
    for obs in observations:
        radius = float(obs.diameter_mm or 0) / 2.0
        try:
            transformed = transform_observation(obs, frames, layout, radius_mm=radius)
        except SceneSyncRefused as exc:
            reason = str(exc)
            prefix = f"{obs.observation_id}: "
            if reason.startswith(prefix):
                reason = reason[len(prefix):]
            verdicts[obs.observation_id] = {"placeable": False, "reason": reason}
            continue
        verdicts[obs.observation_id] = {
            "placeable": True, "reason": "",
            "workarea_mm": [round(v, 1) for v in transformed.workarea.centre_mm],
            "table_mm": [round(v, 1) for v in transformed.table.centre_mm],
            "world_m": [round(v, 4) for v in transformed.world_position_m],
            "transform_source": transformed.transform_source,
        }
    return verdicts


def run_scene(roi_px: Optional[List[int]] = None, frames: int = 1,
              dataset: str = "", segmentation_options: Optional[Dict[str, Any]] = None,
              layout=None, log: Callable[[str], None] = _noop) -> PhysicalResult:
    """Capture (or replay), segment the whole scene, classify, build the batch.

    RAISES `PhysicalAcquisitionError` at the stage that refused. Returns the
    document `.cache-perception/physical-scene/physical_scene.json` holds and
    the live `ObservationBatch` the workflow plans from.
    """
    import cv2                                                        # noqa: PLC0415
    from providers.scene_depth_plane import build_batch, load_catalogue  # noqa: PLC0415
    from scene_segmentation import (SceneSegmentationError, load_capture,  # noqa: PLC0415
                                    render_depth, render_mask_overlay,
                                    render_overlay, segment_scene)

    started_at = time.monotonic()
    timing: Dict[str, Any] = {}
    os.makedirs(OUT, exist_ok=True)
    catalogue = load_catalogue()
    options: Dict[str, Any] = dict(segmentation_options or {})
    if roi_px:
        options["roi_px"] = [int(v) for v in roi_px]
    elif catalogue.scene_roi_px:
        options["roi_px"] = list(catalogue.scene_roi_px)
    options.setdefault("ignore_regions_px", catalogue.ignore_regions_px)

    live = not dataset
    stage = time.monotonic()
    attempts: List[Dict[str, Any]] = []
    if live:
        device = require_rgbd_camera()
        # A DEPTH FRAME THAT HAS NOT SETTLED IS NOT A MEASUREMENT OF THE BENCH.
        # The first frames after the stream opens can carry a quarter of the
        # pixels with no depth at all, and a plane fitted through those reads
        # every plate as flat. The capture is repeated — a bounded number of
        # times, each one a fresh warmed capture — until the depth is mostly
        # valid, and refused with the fractions seen if it never is.
        for attempt in range(1, MAX_CAPTURE_ATTEMPTS + 1):
            capture_document = capture(CAPTURE_NAME, frames)
            dataset = os.path.basename(capture_document["root"])
            root = os.path.join(CAPTURES_ON_HOST, dataset)
            try:
                _, depth_probe, _ = load_capture(root, 0)
            except SceneSegmentationError as exc:
                raise PhysicalAcquisitionError("capture", str(exc),
                                               {"dataset": dataset}) from exc
            fraction = float((depth_probe > 0).mean())
            attempts.append({"dataset": dataset, "valid_depth_fraction": round(fraction, 4)})
            log(f"captured {dataset} (valid depth {fraction:.1%})")
            if fraction >= MIN_VALID_DEPTH_FRACTION:
                break
        else:
            raise PhysicalAcquisitionError(
                "capture",
                f"the depth stream did not settle: {MAX_CAPTURE_ATTEMPTS} captures "
                f"had less than {MIN_VALID_DEPTH_FRACTION:.0%} valid depth",
                {"attempts": attempts})
    else:
        health, _ = get("/health")
        device = ((health or {}).get("probes") or {}).get("rgbd_camera", {}).get("detail", {})
        capture_document = {}
        log(f"replaying capture {dataset}")
    timing["capture_ms"] = round((time.monotonic() - stage) * 1000.0, 1)
    timing["capture_attempts"] = len(attempts) if live else 0

    root = os.path.join(CAPTURES_ON_HOST, dataset)
    stage = time.monotonic()
    try:
        bgr, depth_mm, K = load_capture(root, 0)
        segmentation = segment_scene(bgr, depth_mm, K, options)
    except SceneSegmentationError as exc:
        raise PhysicalAcquisitionError("segmentation", str(exc),
                                       {"dataset": dataset}) from exc
    timing["segmentation_ms"] = round((time.monotonic() - stage) * 1000.0, 1)
    log(f"plane residual {segmentation.plane.residual_mm:.1f} mm, "
        f"{len(segmentation.instances)} instances")

    intrinsics = {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                  "cx": float(K[0, 2]), "cy": float(K[1, 2])}
    captured_at = str((capture_document or {}).get("captured_at") or "")
    stage = time.monotonic()
    batch, report = build_batch(
        segmentation.instances, segmentation.plane, catalogue=catalogue,
        batch_id="physical-scene-1", captured_at=captured_at, dataset=dataset,
        intrinsics=intrinsics, repo_root=REPO)

    # WHAT THE CONFIGURED WORKCELL CAN PLACE. Objects it cannot are taken out
    # of the batch HERE, with their reason on record, so the scene the twin is
    # synchronized from is exactly the set of objects the robot can act on.
    verdicts = _placeability(batch.observations, layout=layout)
    kept = []
    for obs in batch.observations:
        verdict = verdicts.get(obs.observation_id, {})
        entry = next((c for c in report if c.observation_id == obs.observation_id), None)
        if verdict.get("placeable"):
            kept.append(obs)
        elif entry is not None:
            entry.status = "excluded"
            entry.reason = verdict.get("reason", "not placeable")
    batch.observations = kept
    if not kept:
        batch.status = batch.status.__class__("empty")
        batch.error = "no classified object lies inside the configured demo workcell"
    batch.detector_status["instances"] = [c.to_dict() for c in report]
    batch.detector_status["placeability"] = verdicts
    batch.detector_status["counts"] = {
        "measured": len(report),
        "observed": sum(1 for c in report if c.status == "observed"),
        "ignored": sum(1 for c in report if c.status == "ignored"),
        "unclassified": sum(1 for c in report if c.status == "unclassified"),
        "excluded": sum(1 for c in report if c.status == "excluded"),
    }
    timing["classification_ms"] = round((time.monotonic() - stage) * 1000.0, 1)

    # Pictures: the frame, the depth, the tinted masks and the labelled boxes.
    stage = time.monotonic()
    labels: Dict[int, str] = {}
    colours: Dict[int, Any] = {}
    for c in report:
        i = c.instance.index
        if c.status == "observed":
            labels[i] = f"{c.demo_class.demo_type} {c.instance.length_mm:.0f}x{c.instance.width_mm:.0f}"
            colours[i] = (0, 200, 0)
        elif c.status == "excluded":
            labels[i] = f"{c.demo_class.demo_type if c.demo_class else '?'} (out of reach)"
            colours[i] = (0, 165, 255)
        elif c.status == "ignored":
            labels[i] = "ignored"
            colours[i] = (0, 0, 220)
        else:
            labels[i] = "unclassified"
            colours[i] = (128, 128, 128)
    cv2.imwrite(os.path.join(OUT, "rgb.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(os.path.join(OUT, "depth_aligned.jpg"), render_depth(depth_mm))
    cv2.imwrite(os.path.join(OUT, "mask_overlay.jpg"),
                render_mask_overlay(bgr, segmentation.instances))
    cv2.imwrite(os.path.join(OUT, "scene_overlay.jpg"),
                render_overlay(bgr, segmentation.instances, labels, colours),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    timing["artifacts_ms"] = round((time.monotonic() - stage) * 1000.0, 1)
    timing["total_ms"] = round((time.monotonic() - started_at) * 1000.0, 1)

    if not kept:
        raise PhysicalAcquisitionError(
            "classification", batch.error,
            {"dataset": dataset, "instances": [c.to_dict() for c in report],
             "images": [k for k, _ in ARTIFACTS]})

    document = {
        "device": device,
        "dataset": dataset,
        "perception_method": batch.perception_method,
        "estimator_geometry": "",
        "selected_profile": {"width": WIDTH, "height": HEIGHT, "fps": FPS},
        "acquisition_backend": "realsense",
        "provenance": "measured",
        "run_mode": "live" if live else "replay",
        "run_label": ("LIVE PHYSICAL D435" if live
                      else "RECORDED PHYSICAL D435 DATA"),
        "run_note": ("Acquired from the physical D435 during this run." if live
                     else "Replayed from frames a physical D435 recorded "
                          "earlier. Real sensor data, NOT simulation, and NOT a "
                          "live camera."),
        "operator_roi_px": options.get("roi_px"),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "capture": capture_document,
        "capture_attempts": attempts,
        "timing_ms": timing,
        "segmentation": segmentation.to_dict(),
        "catalogue": catalogue.to_dict(),
        "objects": [c.to_dict() for c in report],
        "placeability": verdicts,
        "counts": dict(batch.detector_status["counts"]),
        "batch": batch.to_dict(),
        "images": {k: os.path.join(OUT, v) for k, v in ARTIFACTS},
        "identity_note": batch.detector_status["identity_note"],
        "pose_note": batch.detector_status["pose_note"],
        "accuracy_note": ("No ground truth exists for the physical parts. "
                          "Footprints are measured on the fitted plane; "
                          "accuracy is NOT measured and is not claimed."),
    }
    with open(RESULT, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, default=str)
    log(f"{len(kept)} object(s) in the batch; "
        f"{document['counts']['excluded']} excluded, "
        f"{document['counts']['ignored']} ignored, "
        f"{document['counts']['unclassified']} unclassified")
    return PhysicalResult(document, batch)


__all__ = ["OUT", "RESULT", "ARTIFACTS", "run_scene", "require_rgbd_camera"]
