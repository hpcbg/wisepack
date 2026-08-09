"""`fasterrcnn_bottle` — a WISEPACK perception provider.

THE ONLY MODULE IN WISEPACK THAT KNOWS WHAT A BOTTLE IS, and it knows it for
exactly one reason: to stop knowing it. Everything on the WISEPACK side of this
file sees a ``PhysicalObservation`` of a ``cylindrical_proxy`` and nothing else.

    camera frame -> [Faster R-CNN + ArUco] -> [this provider] -> ObservationBatch
                                                                      |
                                                    packing / workflow / validation
                                                    (future) Isaac scene synchronizer

PROVENANCE, NOT ARCHITECTURE
----------------------------
The detection pipeline below is ADAPTED FROM HARMONY's `ai-bottle-detector-fiware`
(`pipeline.py`, MIT) — the network construction, the confidence threshold, the
bottle/cap association, the yaw derivation and the annotated-image rendering are
that project's, reused because they are known to work on this hardware with these
weights. Once adapted, THIS IS WISEPACK CODE: it runs from the WISEPACK
repository, in the WISEPACK perception environment, and the HARMONY repository is
not a runtime dependency. Delete `/data/arise/harmony` and camera perception is
unaffected.

A second provider — YOLO/OBB, RGB-D pose, segmentation — is a sibling file, not a
change to anything above.

BOTTLES ARE PHYSICAL PROXIES for the cylindrical workpieces WISEPACK packages.
That is a fact about the objects on the table, and it stops at this file.

WHAT THE DETECTOR PRODUCES, PER OBJECT
--------------------------------------
    {"x": <mm>, "y": <mm>, "yaw": <deg>, "conf": <0..1>, "selected": <bool>}

`x`/`y` are millimetres on the ArUco-calibrated plane (see `calibration.py`).
`yaw` is degrees derived from the object-centre -> cap-centre vector.

TWO BEHAVIOURS OF THAT PIPELINE MATTER AND ARE HANDLED EXPLICITLY:

  * ONLY OBJECTS WITH A MATCHED CAP ARE REPORTED. Without the cap there is no
    orientation, so an unmatched bottle is skipped. The reported count is
    therefore "objects with a resolved orientation", not "objects present". That
    distinction is carried into the batch rather than smoothed over.

  * AN UNCALIBRATED FRAME YIELDS THE SENTINEL (1, 1) for every object. Those are
    not measurements, and a plan built from a pile of objects all at (1, 1) would
    be nonsense presented as physics. `_looks_uncalibrated` detects the signature
    and the batch is rejected as an error rather than parsed into confident
    garbage.

The adapter accepts every shape the detector or a transport emits — the
`process_frame` return value, a `result_json` document from a topic, and a bare
list of objects — because a provider's job is to absorb that variety, not to
push it downstream.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# `wisepack_core` is the DOMAIN, imported from the workspace source tree exactly
# as the ROS nodes and the dashboard import it. The provider depends on the core;
# the core has no idea this file exists.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PERCEPTION = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PERCEPTION)
_CORE = os.path.join(_REPO, "wisepack_ws", "src", "wisepack_core")
for _path in (_CORE, _PERCEPTION):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from wisepack_core.perception import (                             # noqa: E402
    BatchStatus, ObservationBatch, PerceptionSource, ProxyGeometry,
    WORKAREA_FRAME_ID, WorkAreaFrame,
)
from wisepack_core.domain import PhysicalObservation                # noqa: E402
from calibration import (                                           # noqa: E402
    CALIBRATION_INVALID, PlaneCalibration, UNCALIBRATED_SENTINEL,
)
from perception_config import (                                     # noqa: E402
    DEFAULT_CONFIDENCE_THRESHOLD, PerceptionConfig,
)

#: The provider's own identity, as selected by WISEPACK_PERCEPTION_DETECTOR.
PROVIDER_NAME = "fasterrcnn_bottle"

#: Where this implementation came from. Diagnostics and provenance only — it is
#: never an architectural label and never appears in an operator-facing
#: description of the perception subsystem.
IMPLEMENTATION_ORIGIN = "HARMONY"

#: Where the trained weights came from, for the same reason.
MODEL_ORIGIN = "HARMONY bottle detector / hpcbg/harmony-bottle-detector"

#: What the operator sees when the dashboard names the detector.
DISPLAY_NAME = "Faster R-CNN"

#: Identifies the detector in every observation's provenance. Specific enough to
#: reproduce a result years later; carried as DATA, never as a type.
DETECTOR_ID = "fasterrcnn_resnet50_fpn/bottle"

#: The detector class label the observations were derived from. Kept as
#: provenance so a later analyst can tell what physically stood on the table,
#: without any WISEPACK consumer having to care.
DETECTOR_CLASS = "bottle"

# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #

#: Background + bottle + cap. The head is built for exactly this many classes,
#: so it is a property of the CHECKPOINT and not a tunable.
NUM_CLASSES = 3

#: Class ids as trained.
CLASS_NAMES = {1: "bottle", 2: "cap"}

#: Fraction of a cap's area that must fall inside an object's box for the two to
#: be treated as the same physical item.
CAP_COVERAGE_THRESHOLD = 0.5

#: Overlay colours, BGR. Unchanged from the implementation this was adapted from,
#: so an annotated image an operator has learned to read keeps its meaning.
BOX_COLOURS = {1: (0, 255, 0), 2: (255, 0, 255)}
OBJECT_COLOUR = (0, 255, 0)
SELECTED_COLOUR = (255, 0, 255)
CAP_COLOUR = (0, 0, 255)


class BottleDetector:
    """Faster R-CNN inference plus the plane measurement, for one process.

    Heavy imports (`torch`, `torchvision`, `cv2`) happen in `__init__`, not at
    module import: naming the provider must not load a neural network, and the
    launcher, the tests and `--check` all import this module without one.
    """

    def __init__(self, model_path: str,
                 calibration: Optional[PlaneCalibration] = None,
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 device: Optional[str] = None) -> None:
        import cv2                                           # noqa: PLC0415
        import numpy as np                                   # noqa: PLC0415
        import torch                                         # noqa: PLC0415
        from torchvision.models.detection import (           # noqa: PLC0415
            fasterrcnn_resnet50_fpn,
        )
        from torchvision.models.detection.faster_rcnn import (  # noqa: PLC0415
            FastRCNNPredictor,
        )

        self._cv2 = cv2
        self._np = np
        self._torch = torch
        self.model_path = model_path
        self.confidence_threshold = float(confidence_threshold)
        self.calibration = calibration or PlaneCalibration()

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # NO PRETRAINED WEIGHTS ARE FETCHED. `weights=None, weights_backbone=None`
        # builds the same architecture without a torchvision download: every
        # parameter and buffer is replaced by the checkpoint below, so the result
        # is identical and the service starts on a host with no network.
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                          NUM_CLASSES)

        checkpoint = torch.load(model_path, map_location=self.device,
                                weights_only=False)
        # Training checkpoints wrap the weights; an exported state dict does not.
        # Both shapes are accepted because both exist in the wild for this model.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        self.model = model

    # -- inference --------------------------------------------------------- #

    def _predict(self, frame) -> Tuple[Any, Any, Any]:
        """Boxes, scores and labels above the confidence threshold."""
        cv2, torch = self._cv2, self._torch
        from torchvision.transforms import functional as F     # noqa: PLC0415

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(image_rgb)

        with torch.no_grad():
            predictions = self.model([tensor.to(self.device)])

        pred = predictions[0]
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()

        keep = scores >= self.confidence_threshold
        return boxes[keep], scores[keep], labels[keep]

    # -- the whole pass ---------------------------------------------------- #

    def process_frame(self, frame) -> Dict[str, Any]:
        """One frame in, measured objects and two annotated images out.

        Returns:
            objects           the detections WITH a resolved orientation
            pick_pose         the first such object, for a single-pick consumer
            calibration       {"status", "revision", "markers_in_frame"}
            annotated_image   markers, plane, measured poses  (the operator view)
            detections_image  raw boxes and scores only       (the model view)
        """
        cv2, np = self._cv2, self._np

        boxes, scores, labels = self._predict(frame)

        # THE MODEL VIEW, taken before any calibration overlay: it shows what the
        # network saw and nothing else, which is what makes it useful when the
        # measurement looks wrong but the detection looks right.
        detections_image = frame.copy()
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            colour = BOX_COLOURS.get(int(label), OBJECT_COLOUR)
            cv2.rectangle(detections_image, (x1, y1), (x2, y2), colour, 2)
            text = f"{CLASS_NAMES.get(int(label), f'class {label}')}: {score:.2f}"
            (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                 1, 2)
            cv2.rectangle(detections_image, (x1, y1 - height - 4),
                          (x1 + width, y1), colour, -1)
            cv2.putText(detections_image, text, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # THE OPERATOR VIEW starts from the raw frame and accumulates the plane
        # and the measurements.
        annotated = frame.copy()
        calibration = self.calibration.analyse(frame)
        self.calibration.annotate(annotated, calibration)

        image_h, image_w = frame.shape[:2]
        objects: List[Dict[str, Any]] = []
        caps: List[Dict[str, Any]] = []

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
            points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
                              ).astype(np.int32)
            mask = np.zeros((image_h, image_w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)

            entry = {
                "centre": (int(centre_x), int(centre_y)),
                "points": points,
                "mask": mask,
                "confidence": float(score),
                "coords": self.calibration.to_plane(calibration.homography,
                                                    centre_x, centre_y),
            }
            name = CLASS_NAMES.get(int(label))
            if name == "bottle":
                objects.append(entry)
            elif name == "cap":
                caps.append(entry)

        selected = None
        for entry in objects:
            entry["cap"] = None
            entry["yaw"] = 0.0
            for cap in caps:
                overlap = cv2.countNonZero(cv2.bitwise_and(entry["mask"],
                                                           cap["mask"]))
                area = cv2.countNonZero(cap["mask"])
                if not area:
                    continue
                # THE LAST CAP ABOVE THE THRESHOLD WINS, not the best one. That
                # is the adapted behaviour and it is kept deliberately: a bottle
                # overlaps at most one cap by more than half its area in this
                # setup, so "last above threshold" and "best" agree, and changing
                # it would silently change which measurements the demonstrator
                # produces.
                if overlap / area > CAP_COVERAGE_THRESHOLD:
                    entry["cap"] = cap["centre"]
                    entry["cap_coords"] = cap["coords"]

            if entry["cap"] is None:
                continue

            # YAW IS MEASURED ON THE PLANE, not in pixels: the vector runs from
            # the object centre to its cap centre in millimetres, so perspective
            # is already removed. +90 puts 0 deg along the plane's +y axis.
            vector = (entry["cap_coords"][0] - entry["coords"][0],
                      entry["cap_coords"][1] - entry["coords"][1])
            angle = math.degrees(math.atan2(vector[1], vector[0])) + 90
            entry["yaw"] = angle % 360 - 180

            if selected is None:
                selected = entry

        results: List[Dict[str, Any]] = []
        for entry in objects:
            if entry["cap"] is None:
                # NO CAP, NO ORIENTATION, NO REPORT. Counted below so the gap
                # between "objects seen" and "objects reported" is visible.
                continue

            colour = SELECTED_COLOUR if entry is selected else OBJECT_COLOUR
            cv2.drawContours(annotated, [entry["points"]], -1, colour, 3)
            cv2.line(annotated, entry["centre"], entry["cap"], OBJECT_COLOUR, 2)
            cv2.circle(annotated, entry["centre"], 5, OBJECT_COLOUR, -1)

            x_mm, y_mm = entry["coords"]
            yaw = entry["yaw"]
            origin_x, origin_y = entry["centre"]
            cv2.putText(annotated, f"({x_mm:.1f}, {y_mm:.1f})",
                        (origin_x + 5, origin_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, OBJECT_COLOUR, 2)
            cv2.putText(annotated, f"{int(yaw)} deg",
                        (origin_x + 5, origin_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, OBJECT_COLOUR, 2)
            cv2.putText(annotated, f"Bottle {entry['confidence']:.2f}",
                        (origin_x + 5, origin_y + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, OBJECT_COLOUR, 2)

            results.append({
                "x": float(x_mm),
                "y": float(y_mm),
                "yaw": float(yaw),
                "conf": float(entry["confidence"]),
                "selected": entry is selected,
            })

        for cap in caps:
            cv2.drawContours(annotated, [cap["points"]], -1, CAP_COLOUR, 2)
            cv2.circle(annotated, cap["centre"], 5, CAP_COLOUR, -1)

        pick_pose: Dict[str, float] = {}
        if selected is not None:
            pick_pose = {"x": float(selected["coords"][0]),
                         "y": float(selected["coords"][1]),
                         "rotation": float(selected["yaw"])}

        return {
            "objects": results,
            "pick_pose": pick_pose,
            "object_count": len(results),
            "objects_without_orientation": len(objects) - len(results),
            "caps_detected": len(caps),
            "calibration": {
                "status": calibration.status,
                "revision": calibration.revision,
                "markers_in_frame": calibration.seen_this_frame,
                # WHERE IT CAME FROM, and why detection did or did not need the
                # sheet. `markers_in_frame: false` with `source: saved` is the
                # ordinary, correct state for a working deployment.
                "source": calibration.source,
                "reason": calibration.reason,
            },
            "annotated_image": annotated,
            "detections_image": detections_image,
        }


def build_detector(config: PerceptionConfig,
                   calibration: Optional[PlaneCalibration] = None
                   ) -> BottleDetector:
    """The provider's constructor, as the service calls it.

    The SERVICE never names torch, a checkpoint format or an ArUco dictionary;
    it hands over the resolved configuration and gets something with
    `process_frame`. That is the whole provider contract.
    """
    return BottleDetector(
        model_path=config.model_path,
        calibration=calibration or PlaneCalibration(config.board),
        confidence_threshold=config.confidence_threshold)


# --------------------------------------------------------------------------- #
# Detector result -> domain-neutral batch
# --------------------------------------------------------------------------- #


def _objects_of(payload: Any) -> Optional[Sequence[Any]]:
    """Find the detected-object list in any of the shapes this provider sees."""
    if isinstance(payload, (list, tuple)):
        return payload
    if not isinstance(payload, Mapping):
        return None
    for key in ("objects", "bottles", "detections"):
        value = payload.get(key)
        if isinstance(value, (list, tuple)):
            return value
    return None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # NaN/inf reach here from a detector that divided by zero somewhere. They
    # would propagate silently through every arithmetic step downstream.
    return number if number == number and abs(number) != float("inf") else None


#: The only detector keys ever copied into `detector_status`. An ALLOWLIST, not
#: an exclusion list, because `process_frame` also returns `annotated_image` and
#: `detections_image` — numpy arrays of a whole camera frame. Copying the payload
#: wholesale put those into a field that is published as JSON over DDS and HTTP,
#: where they are both unserialisable and megabytes of binary nobody asked for.
#: Naming what travels is the fix.
_STATUS_KEYS = ("status", "object_count", "objects_without_orientation",
                "caps_detected", "bottleCount", "pickPose", "pick_pose")


def _detector_status(payload: Any) -> Dict[str, Any]:
    """The detector's own scalar outputs, kept verbatim as evidence."""
    if not isinstance(payload, Mapping):
        return {}
    return {key: payload[key] for key in _STATUS_KEYS if key in payload}


def _looks_uncalibrated(poses: Sequence[Any]) -> bool:
    """True when every reported coordinate is the uncalibrated sentinel.

    Requires at least one object: an empty result says nothing about
    calibration, and reporting "calibration invalid" for an empty table would be
    inventing a fault.
    """
    points = [p for p in poses if p is not None]
    return bool(points) and all(p == UNCALIBRATED_SENTINEL for p in points)


def _clamp_confidence(value: Optional[float]) -> Optional[float]:
    """Keep confidence inside [0, 1] without rejecting the whole detection.

    A detector reporting 1.0000001 is a float artefact, not a malformed result.
    A wildly out-of-range value is dropped to None rather than carried, so no
    consumer can read a fabricated number as a measurement.
    """
    if value is None:
        return None
    if -0.01 <= value <= 1.01:
        return min(1.0, max(0.0, value))
    return None


def observations_from_detections(
        payload: Any,
        *,
        batch_id: str,
        captured_at: str = "",
        requested_at: str = "",
        model_id: str = "",
        geometry: Optional[ProxyGeometry] = None,
        frame: Optional[WorkAreaFrame] = None,
        calibration_status: Optional[str] = None,
        calibration_revision: str = "",
        source: str = PerceptionSource.CAMERA.value,
) -> ObservationBatch:
    """Convert one detector result into a domain-neutral batch.

    NEVER RAISES for a bad payload. A malformed, unparseable or uncalibrated
    result comes back as ``BatchStatus.ERROR`` with a reason, because those have
    to be visible in the dashboard rather than become an exception somewhere up
    the call stack — and because a failed scan must not be able to masquerade as
    a successful empty one.

    ``calibration_status`` overrides the inference below when the caller knows
    better (the detector service does: it can see whether the ArUco plane was
    resolved for this very frame).
    """
    geometry = geometry or ProxyGeometry()
    frame = frame or WorkAreaFrame()

    def fail(reason: str) -> ObservationBatch:
        return ObservationBatch.failed(
            batch_id=batch_id, source=source, error=reason,
            frame_id=frame.frame_id, captured_at=captured_at,
            requested_at=requested_at,
            detector=DETECTOR_ID, model_id=model_id,
            calibration_status=calibration_status or "unknown",
            calibration_revision=calibration_revision,
            detector_status=_detector_status(payload))

    # -- an explicit failure from the detector wins over anything else ------ #
    if isinstance(payload, Mapping):
        status = str(payload.get("status", "")).upper()
        if status == "FAILED":
            return fail(str(payload.get("error")
                            or "the detector reported FAILED without a reason"))

    raw_objects = _objects_of(payload)
    if raw_objects is None:
        kind = type(payload).__name__
        return fail(f"malformed detector result: no object list found "
                    f"(payload was {kind})")

    observations: List[PhysicalObservation] = []
    raw_points: List[Any] = []
    malformed = 0
    outside_workarea = 0

    for index, entry in enumerate(raw_objects):
        if not isinstance(entry, Mapping):
            malformed += 1
            continue
        x = _number(entry.get("x"))
        y = _number(entry.get("y"))
        if x is None or y is None:
            # A detection without a usable position is not a detection. Counted
            # and reported, never silently defaulted to the origin.
            malformed += 1
            continue
        raw_points.append((x, y))
        yaw = _number(entry.get("yaw"))
        if yaw is None:
            yaw = _number(entry.get("rotation")) or 0.0
        confidence = _clamp_confidence(_number(entry.get("conf"))
                                       if entry.get("conf") is not None
                                       else _number(entry.get("confidence")))
        if not frame.contains(x, y, tolerance_mm=geometry.diameter_mm):
            outside_workarea += 1
        observations.append(PhysicalObservation(
            observation_id=f"physical-cylinder-{index + 1:03d}",
            x_mm=x, y_mm=y, yaw_deg=yaw,
            confidence=confidence,
            object_type="cylindrical_proxy",
            source=source,
            frame_id=frame.frame_id,
            detector=DETECTOR_ID,
            model_id=model_id,
            detector_class=str(entry.get("class") or DETECTOR_CLASS),
            detector_object_index=index,
            captured_at=captured_at,
            calibration_status=calibration_status or "unknown",
            calibration_revision=calibration_revision,
            diameter_mm=geometry.diameter_mm,
            length_mm=geometry.length_mm,
        ))

    # -- calibration -------------------------------------------------------- #
    resolved_calibration = calibration_status
    if resolved_calibration is None:
        if not observations:
            resolved_calibration = "unknown"
        elif _looks_uncalibrated(raw_points):
            resolved_calibration = CALIBRATION_INVALID
        else:
            resolved_calibration = "valid"
    if resolved_calibration == CALIBRATION_INVALID:
        # THE DETECTOR'S OWN REASON WHEN IT GAVE ONE. It knows whether a saved
        # calibration was missing, unreadable or measured at another resolution;
        # this layer only knows the coordinates came back as the sentinel, and
        # telling an operator to fetch the sheet when the real problem is a
        # resolution change sends them to the wrong fix.
        detail = ""
        if isinstance(payload, Mapping):
            detail = str((payload.get("calibration") or {}).get("reason") or "")
        return fail(detail or (
            "the camera is not calibrated: no saved calibration is available "
            "and the calibration markers are not visible in this frame — every "
            f"coordinate is the uncalibrated sentinel {UNCALIBRATED_SENTINEL}, "
            "not a measurement. Place the calibration sheet in view once and "
            "detect again; it is then saved and detection no longer needs it."))

    for obs in observations:
        obs.calibration_status = resolved_calibration
        obs.calibration_revision = calibration_revision

    if malformed and not observations:
        return fail(f"malformed detector result: all {malformed} entries lacked "
                    "a usable position")

    # The detector's own scalar outputs, as evidence. `object_count` is its count
    # of objects WITH a resolved orientation; a mismatch against ours is worth
    # seeing rather than hiding.
    detector_status = _detector_status(payload)
    if malformed:
        detector_status["malformed_entries"] = malformed
    if outside_workarea:
        # Reported, never clamped. An object measured outside the declared plane
        # is a calibration or work-area configuration question for the operator,
        # and moving the number would destroy the evidence for it.
        detector_status["outside_workarea"] = outside_workarea
        detector_status["workarea"] = frame.to_dict()

    return ObservationBatch(
        batch_id=batch_id,
        source=source,
        status=BatchStatus.OK if observations else BatchStatus.EMPTY,
        observations=observations,
        frame_id=frame.frame_id,
        captured_at=captured_at,
        requested_at=requested_at,
        detector=DETECTOR_ID,
        model_id=model_id,
        calibration_status=resolved_calibration,
        calibration_revision=calibration_revision,
        detector_status=detector_status,
    )


def parse_detection_json(text: str, **kwargs: Any) -> ObservationBatch:
    """Same as ``observations_from_detections``, from raw JSON on a transport.

    Invalid JSON is a failed batch, not an exception: a malformed message is one
    of the failures that has to render in the dashboard.
    """
    import json                                              # noqa: PLC0415
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as exc:
        batch_id = str(kwargs.get("batch_id", "batch-0"))
        return ObservationBatch.failed(
            batch_id=batch_id,
            source=str(kwargs.get("source", PerceptionSource.CAMERA.value)),
            error=f"malformed detector message: not valid JSON ({exc})",
            detector=DETECTOR_ID,
            frame_id=WORKAREA_FRAME_ID)
    return observations_from_detections(payload, **kwargs)


__all__ = [
    "PROVIDER_NAME", "IMPLEMENTATION_ORIGIN", "MODEL_ORIGIN", "DISPLAY_NAME",
    "DETECTOR_ID", "DETECTOR_CLASS", "UNCALIBRATED_SENTINEL", "NUM_CLASSES",
    "CLASS_NAMES", "CAP_COVERAGE_THRESHOLD", "BottleDetector", "build_detector",
    "observations_from_detections", "parse_detection_json",
]
