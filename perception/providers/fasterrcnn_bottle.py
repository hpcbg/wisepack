"""`fasterrcnn_bottle` — a WISEPACK perception provider.

THE ONLY MODULE IN WISEPACK THAT KNOWS WHAT A BOTTLE IS, and it knows it for
exactly one reason: to stop knowing it. Everything on the WISEPACK side of this
file sees a ``PhysicalObservation`` of a ``cylindrical_proxy`` and nothing else.

    camera frame -> [Faster R-CNN + ArUco] -> [this provider] -> ObservationBatch
                                                                      |
                                                    packing / workflow / validation
                                                    (future) Isaac scene synchronizer

PROVENANCE, NOT ARCHITECTURE. The detector implementation is the bottle detector
developed in HARMONY (`ai-bottle-detector-fiware`), reused rather than rewritten.
WISEPACK owns perception; HARMONY contributes one provider behind this boundary.
A second provider — YOLO/OBB, RGB-D pose, segmentation — is a sibling file, not
a change to anything above.

BOTTLES ARE PHYSICAL PROXIES for the cylindrical workpieces WISEPACK packages.
That is a fact about the objects on the table, and it stops at this file.

WHAT HARMONY ACTUALLY PRODUCES (verified against the repository, not assumed)
----------------------------------------------------------------------------
``ai-bottle-detector-fiware/pipeline.py`` returns, per detected bottle:

    {"x": <mm>, "y": <mm>, "yaw": <deg>, "conf": <0..1>, "selected": <bool>}

`x`/`y` are millimetres in the ArUco-calibrated plane (the shipped A4 sheet puts
the origin at marker 11 and the far corner at 130 mm). `yaw` is degrees derived
from the bottle-centre -> cap-centre vector.

TWO BEHAVIOURS OF THAT PIPELINE MATTER HERE AND ARE HANDLED EXPLICITLY:

  * ONLY BOTTLES WITH A MATCHED CAP ARE REPORTED. `pipeline.py` skips any bottle
    whose cap was not matched (`if b['cap'] is None: continue`), because without
    the cap there is no orientation. So the reported count is "objects with a
    resolved orientation", not "objects present". That distinction is carried
    into the batch rather than smoothed over — see `caps_missing`.

  * AN UNCALIBRATED FRAME YIELDS THE SENTINEL (1, 1). When no homography is
    available, the coordinate computation falls into a bare `except` that
    substitutes `coords = (1, 1)` for EVERY object. Those are not measurements,
    and a plan built from a pile of objects all at (1, 1) would be nonsense
    presented as physics. `_looks_uncalibrated` detects the signature and the
    batch is rejected as an error rather than parsed into confident garbage.

The adapter accepts every shape HARMONY emits — the ROS 2 `result_json`
document, the raw `process_frame` return value, and a bare list of objects —
because §7 says to reuse HARMONY's existing interfaces rather than require it to
grow a new one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import os
import sys

# `wisepack_core` is the DOMAIN, imported from the workspace source tree exactly
# as the ROS nodes and the dashboard import it. The provider depends on the core;
# the core has no idea this file exists.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CORE = os.path.join(_REPO, "wisepack_ws", "src", "wisepack_core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from wisepack_core.perception import (                             # noqa: E402
    BatchStatus, ObservationBatch, PerceptionSource, ProxyGeometry,
    WORKAREA_FRAME_ID, WorkAreaFrame,
)
from wisepack_core.domain import PhysicalObservation                # noqa: E402

#: The provider's own identity, as selected by WISEPACK_PERCEPTION_DETECTOR.
PROVIDER_NAME = "fasterrcnn_bottle"

#: Where this implementation came from. Diagnostics and provenance only — it is
#: never an architectural label and never appears in an operator-facing
#: description of the perception subsystem.
IMPLEMENTATION_ORIGIN = "HARMONY"

#: What the operator sees when the dashboard names the detector.
DISPLAY_NAME = "Faster R-CNN"

#: Identifies the detector in every observation's provenance (§11). Specific
#: enough to reproduce a result years later; carried as DATA, never as a type.
DETECTOR_ID = "fasterrcnn_resnet50_fpn/bottle"


#: The HARMONY class label the observations were derived from. Kept as
#: provenance so a later analyst can tell what physically stood on the table,
#: without any WISEPACK consumer having to care.
HARMONY_DETECTOR_CLASS = "bottle"

#: The coordinate HARMONY's `pipeline.py` substitutes when the ArUco homography
#: is unavailable. Not a position — a marker for "calibration missing".
UNCALIBRATED_SENTINEL = (1.0, 1.0)


def _objects_of(payload: Any) -> Optional[Sequence[Any]]:
    """Find the detected-object list in any of HARMONY's result shapes."""
    if isinstance(payload, (list, tuple)):
        return payload
    if not isinstance(payload, Mapping):
        return None
    for key in ("bottles", "objects", "detections"):
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


#: The only HARMONY keys ever copied into `detector_status`. An ALLOWLIST, not
#: an exclusion list, because `pipeline.process_frame` also returns
#: `ai_processed_image` and `processed_image` — numpy arrays of a whole camera
#: frame. Copying the payload wholesale put those into a field that is published
#: as JSON over DDS and HTTP, where they are both unserialisable and megabytes
#: of binary nobody asked for. Naming what travels is the fix.
_STATUS_KEYS = ("status", "bottleCount", "pickPose", "pick_pose")


def _detector_status(payload: Any) -> Dict[str, Any]:
    """HARMONY's own scalar outputs, kept verbatim as evidence."""
    if not isinstance(payload, Mapping):
        return {}
    return {key: payload[key] for key in _STATUS_KEYS if key in payload}


def _looks_uncalibrated(poses: Sequence[Any]) -> bool:
    """True when every reported coordinate is HARMONY's uncalibrated sentinel.

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


def observations_from_harmony(
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
    """Convert one HARMONY detector result into a domain-neutral batch.

    NEVER RAISES for a bad payload. A malformed, unparseable or uncalibrated
    result comes back as ``BatchStatus.ERROR`` with a reason, because §15
    requires those to be visible in the dashboard rather than to become an
    exception somewhere up the call stack — and because a failed scan must not
    be able to masquerade as a successful empty one.

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
                            or "HARMONY reported FAILED without a reason"))

    raw_objects = _objects_of(payload)
    if raw_objects is None:
        kind = type(payload).__name__
        return fail(f"malformed HARMONY result: no object list found "
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
            detector_class=str(entry.get("class") or HARMONY_DETECTOR_CLASS),
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
            resolved_calibration = "invalid"
        else:
            resolved_calibration = "valid"
    if resolved_calibration == "invalid":
        return fail(
            "HARMONY reported no valid ArUco calibration for this frame — every "
            "coordinate is the uncalibrated sentinel "
            f"{UNCALIBRATED_SENTINEL}, not a measurement. Place the calibration "
            "sheet in view of the camera and detect again.")

    for obs in observations:
        obs.calibration_status = resolved_calibration
        obs.calibration_revision = calibration_revision

    if malformed and not observations:
        return fail(f"malformed HARMONY result: all {malformed} entries lacked "
                    "a usable position")

    # HARMONY's own scalar outputs, as evidence. `bottleCount` is its count of
    # objects WITH a matched cap; a mismatch against ours is worth seeing rather
    # than hiding.
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


def parse_harmony_json(text: str, **kwargs: Any) -> ObservationBatch:
    """Same as ``observations_from_harmony``, from the raw JSON on a ROS topic.

    Invalid JSON is a failed batch, not an exception: a malformed ROS message is
    one of the failures §15 lists, and it has to render in the dashboard.
    """
    import json                                              # noqa: PLC0415
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as exc:
        batch_id = str(kwargs.get("batch_id", "batch-0"))
        return ObservationBatch.failed(
            batch_id=batch_id,
            source=str(kwargs.get("source", PerceptionSource.CAMERA.value)),
            error=f"malformed HARMONY message: not valid JSON ({exc})",
            detector=DETECTOR_ID,
            frame_id=WORKAREA_FRAME_ID)
    return observations_from_harmony(payload, **kwargs)


__all__ = [
    "PROVIDER_NAME", "IMPLEMENTATION_ORIGIN", "DISPLAY_NAME", "DETECTOR_ID",
    "HARMONY_DETECTOR_CLASS", "UNCALIBRATED_SENTINEL",
    "observations_from_harmony", "parse_harmony_json",
]
