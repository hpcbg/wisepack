"""`foundationpose_rgbd` — a WISEPACK perception provider.

Model-based 6-DoF pose estimation: a known CAD mesh, an RGB-D frame and a
segmentation mask in, a rigid transform in the CAMERA frame out.

    worker container  --HTTP-->  THIS PROVIDER  -->  ObservationBatch
                                                     of PhysicalObservation

WHERE THE BOUNDARY IS, AND WHY IT IS HERE
-----------------------------------------
This module is the ONLY place in WISEPACK that knows what a FoundationPose
response looks like. Above it there are observations with positions, quaternions
and declared symmetry; nothing above it has heard of a mesh, a mask, a rotation
grid, a refiner iteration count, or the worker's HTTP schema. The planner, the
Digital Twin, the DDS bridge and FIWARE all see the same `ObservationBatch` a
planar detection produces, differing only in which fields are populated and in
the provenance stamped on it.

That is the same contract `fasterrcnn_bottle` honours, and it is why adding this
method changes nothing above the provider boundary.

WHAT IT REFUSES TO DO
---------------------
* It does not transform into the work area. FoundationPose reports relative to
  the camera optical frame, and converting that to `wisepack_workarea` needs a
  measured SE(3) extrinsic which does not exist yet. The observation keeps the
  camera frame, is `pose_valid=True` — the estimate itself is sound where it
  lives — and carries `workarea_transform_valid=False`, from which
  `workarea_pose_available` derives False. That last flag is what prevents the
  pose being treated as a work-area or Isaac pose. Reusing the planar ArUco
  homography would be a 2-D planar map standing in for a 3-D transform, which is
  a different quantity wearing the same name.
* It does not plan. It builds observations; what happens to them is elsewhere.
* It does not invent a symmetry. The declared symmetry comes from the object
  registry, where it was MEASURED (`scripts/measure_mesh_symmetry.py`), and the
  canonicalisation applied here is the domain layer's.
* It does not call a score an accuracy. FoundationPose's score ranks pose
  hypotheses against each other; it is not a distance from a true pose, and no
  ground truth exists in any dataset here to make it one.

REFERENCE MODE IS NOT CAMERA MODE
---------------------------------
`acquire_reference()` runs the saved tutorial dataset through the whole path so
the provider, the serialisation and the dashboard can be exercised without a
depth camera. Every batch it produces is stamped `acquisition="reference"` and
carries a note saying so. The pose in it is a REAL, VALID estimate of a saved
frame — `pose_valid` is True — and it is not a measurement of the physical work
area. A run must never consume one believing a camera was involved, which is what
`acquisition` records: a separate question from either validity above.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_CORE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "wisepack_ws", "src", "wisepack_core")
if _CORE not in sys.path:                                    # pragma: no cover
    sys.path.insert(0, _CORE)

from wisepack_core.domain import PhysicalObservation           # noqa: E402
from wisepack_core.foundationpose_client import (              # noqa: E402
    FoundationPoseClient)
from wisepack_core.perception import (                         # noqa: E402
    BatchStatus, ObservationBatch, PerceptionMethod, PerceptionSource)
from wisepack_core.pose import (                               # noqa: E402
    CAMERA_OPTICAL_FRAME, Orientation, PoseError, canonicalize)
from wisepack_core.rgbd import ObjectModelRegistry, load_object_registry  # noqa: E402

#: This provider's identity, in the registry's vocabulary.
PROVIDER_NAME = "foundationpose_rgbd"
METHOD = PerceptionMethod.FOUNDATIONPOSE_RGBD.value

#: What produced the numbers, for `PhysicalObservation.detector`. Names the
#: METHOD and the upstream project, never a WISEPACK internal path.
ESTIMATOR_ID = "foundationpose/rgbd-6dof"

#: How a batch was acquired. Two values, and the difference matters more than
#: any other field on the batch.
ACQUISITION_LIVE = "live"
ACQUISITION_REFERENCE = "reference"

#: Stamped on every reference batch. Present in the payload so a consumer cannot
#: render one without it, in the same way the proxy disclosure is.
REFERENCE_NOTE = (
    "REFERENCE / OFFLINE REGRESSION — this pose was estimated from a saved "
    "dataset, not from a live camera. It exercises the FoundationPose worker, "
    "the WISEPACK provider and the dashboard; it is not a measurement of the "
    "physical work area and must not be planned against.")

#: Why a camera-frame pose is not yet actionable. One sentence, carried with the
#: observation rather than left for a reader to work out.
NO_EXTRINSIC_NOTE = (
    "Pose is expressed in the camera optical frame. No validated SE(3) "
    "camera-to-work-area extrinsic exists, so this pose is not placed in "
    "wisepack_workarea and is not actionable. The planar ArUco homography is a "
    "planar map, not a 3-D transform, and is not reused for this.")


class FoundationPoseProviderError(RuntimeError):
    """A provider failure with the reason an operator needs."""


# --------------------------------------------------------------------------- #
# Response validation
# --------------------------------------------------------------------------- #


def validate_response(result: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Check a worker response before any of it is believed.

    Returns `(validated, "")` or `({}, reason)`. NEVER raises and never repairs:
    a response missing its frame, or carrying a quaternion that is not a
    rotation, is a response that cannot be turned into a measurement, and
    guessing the missing part would produce a confident pose from a broken one.
    """
    if not isinstance(result, dict):
        return {}, f"the worker returned {type(result).__name__}, not an object"

    frame_id = str(result.get("frame_id") or "").strip()
    if not frame_id:
        # A POSE WITHOUT A FRAME IS THREE NUMBERS. Refusing here is what stops
        # them being placed somewhere by a consumer that assumed a default.
        return {}, ("the worker returned no frame_id; a pose without a "
                    "coordinate frame cannot be placed by any consumer")

    position = result.get("position_mm")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        return {}, "the worker returned no usable position_mm [x, y, z]"
    try:
        x_mm, y_mm, z_mm = (float(v) for v in position)
    except (TypeError, ValueError):
        return {}, f"position_mm is not numeric: {position!r}"

    quaternion = result.get("orientation")
    if not isinstance(quaternion, dict):
        return {}, "the worker returned no orientation quaternion"
    try:
        orientation = Orientation(x=float(quaternion["x"]), y=float(quaternion["y"]),
                                  z=float(quaternion["z"]), w=float(quaternion["w"]))
    except KeyError as exc:
        return {}, f"the orientation quaternion is missing component {exc}"
    except (TypeError, ValueError) as exc:
        return {}, f"the orientation quaternion is not numeric: {exc}"
    except PoseError as exc:
        # `Orientation` rejects a zero or non-unit quaternion. That is a
        # rotation that does not exist, and it must not become a pose.
        return {}, f"the orientation quaternion is not a rotation: {exc}"

    return {
        "frame_id": frame_id,
        "x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm,
        "orientation": orientation,
        "captured_at": str(result.get("estimated_at") or ""),
        "revision": str(result.get("foundationpose_revision") or ""),
        "duration_ms": result.get("duration_ms"),
        "mesh_path": str(result.get("mesh_path") or ""),
        "dataset": str(result.get("dataset") or ""),
        "frame_file": str(result.get("frame_file") or ""),
        "intrinsics": result.get("intrinsics"),
        "pose_of": str(result.get("pose_of") or ""),
    }, ""


# --------------------------------------------------------------------------- #
# The provider
# --------------------------------------------------------------------------- #


class FoundationPoseProvider:
    """Turns worker responses into domain observations. Owns no camera."""

    def __init__(self, client: Optional[FoundationPoseClient] = None,
                 registry: Optional[ObjectModelRegistry] = None) -> None:
        self.client = client or FoundationPoseClient()
        self._registry = registry

    # -- capability --------------------------------------------------------- #

    @property
    def registry(self) -> ObjectModelRegistry:
        if self._registry is None:
            self._registry = load_object_registry()
        return self._registry

    def models(self) -> List[Dict[str, Any]]:
        """Object models this METHOD can use — those that declare it and have a
        mesh. A model with no mesh is listed as unusable WITH THE REASON rather
        than hidden, so an operator can see that the entry exists and what it
        needs."""
        listing: List[Dict[str, Any]] = []
        for model in self.registry.models.values():
            if not model.supports(METHOD):
                continue
            document = model.to_dict()
            has_mesh = model.mesh_exists(self.registry.root)
            document["mesh_available"] = has_mesh
            document["usable"] = has_mesh
            document["reason"] = ("" if has_mesh else
                                  ("no mesh file at "
                                   f"{model.resolved_path(self.registry.root) or '(unset)'}"
                                   " — a model-based method cannot estimate the "
                                   "pose of a shape it does not have"))
            listing.append(document)
        return listing

    def capability(self, health: Optional[Dict[str, Any]] = None,
                   rgbd_camera_available: Optional[bool] = None
                   ) -> Dict[str, Any]:
        """The WHOLE inference chain, one field per prerequisite.

        `inference_ready` is their conjunction and is never True merely because
        a container exists. `rgbd_camera_available` is tri-state: None means
        "not determined from here", which is not the same as "no camera" —
        today it is False because no depth camera is attached, and the reason
        says exactly that rather than implying a broken worker.
        """
        document = self.client.health() if health is None else health
        worker_ok, worker_reason = self.client.capability(document)

        usable_models = [m for m in self.models() if m["usable"]]
        # THE WORKER OWNS THE CAMERA, so its answer is the default. The
        # parameter remains for tests and for a caller that genuinely knows
        # better; it is not a place to guess from.
        camera = (bool(rgbd_camera_available) if rgbd_camera_available is not None
                  else bool(document.get("rgbd_camera_available")))

        capability: Dict[str, Any] = {
            "method": METHOD,
            "worker_url": document.get("worker_url", self.client.url),
            "worker_reachable": bool(document.get("worker_reachable")),
            "worker_ready": bool(document.get("worker_ready")),
            "gpu_available": bool(document.get("gpu_available")),
            "foundationpose_runtime_available":
                bool(document.get("foundationpose_runtime_available")),
            "scorer_weights_available":
                bool(document.get("scorer_weights_available")),
            "refiner_weights_available":
                bool(document.get("refiner_weights_available")),
            "object_model_available": bool(usable_models),
            "rgbd_camera_available": camera,
            # RUNTIME READY vs LIVE INFERENCE READY. The distinction the
            # dashboard needs: today the runtime IS ready and live inference is
            # not, and reporting one number would hide which.
            "runtime_ready": worker_ok,
            "inference_ready": bool(worker_ok and camera and usable_models),
            "offline_regression_available": bool(worker_ok and usable_models),
            "models": usable_models,
            "blocked_by": [],
            "foundationpose_revision": document.get("foundationpose_revision", ""),
            "revision_matches_pin": bool(document.get("revision_matches_pin")),
            "versions": dict(document.get("versions") or {}),
            "licence_note": document.get("licence_note", ""),
        }

        blockers: List[str] = []
        if not worker_ok:
            blockers.append(worker_reason)
        if not usable_models:
            blockers.append(
                "no object model with a mesh declares the foundationpose_rgbd "
                "method; a model-based estimator needs the CAD geometry of the "
                "object it is looking at")
        if not camera:
            # THE WORKER'S OWN REASON when it gave one: it knows whether the
            # device is absent, unreadable or ambiguous, and this layer only
            # knows the flag came back false.
            reason = ((document.get("probes") or {}).get("rgbd_camera") or {}
                      ).get("reason") or ""
            blockers.append(reason or (
                "no RGB-D camera is available to the FoundationPose worker, so "
                "no live frame can be acquired. The runtime is unaffected and "
                "the offline reference regression still runs."))
        capability["blocked_by"] = blockers
        return capability

    # -- estimation --------------------------------------------------------- #

    def acquire_reference(self, dataset: str, model_id: str,
                          depth_scale_mm: float, frame: int = 0,
                          refine_iterations: int = 5,
                          batch_id: str = "fp-reference-1") -> ObservationBatch:
        """Run the SAVED reference dataset end to end. Not a camera acquisition.

        This is the offline regression path: worker -> provider ->
        PhysicalObservation -> serialisation -> dashboard, with no depth camera
        in existence. Every batch it returns says so in `acquisition` and in
        the batch note. The poses are VALID estimates in the camera frame; what
        makes them unusable for planning is `workarea_pose_available` being
        False and the acquisition being `reference` — never a false claim that
        the estimate failed.
        """
        return self._estimate(
            dataset=dataset, model_id=model_id, depth_scale_mm=depth_scale_mm,
            frame=frame, refine_iterations=refine_iterations,
            batch_id=batch_id, acquisition=ACQUISITION_REFERENCE)

    def _estimate(self, dataset: str, model_id: str, depth_scale_mm: float,
                  frame: int, refine_iterations: int, batch_id: str,
                  acquisition: str) -> ObservationBatch:
        requested_at = _utc_now()

        def failed(reason: str) -> ObservationBatch:
            return ObservationBatch.failed(
                batch_id=batch_id, source=PerceptionSource.CAMERA.value,
                error=reason, frame_id=CAMERA_OPTICAL_FRAME,
                requested_at=requested_at, detector=ESTIMATOR_ID,
                perception_method=METHOD, acquisition=acquisition,
                model_id=model_id)

        model = self.registry.models.get(model_id)
        if model is None:
            return failed(
                f"unknown object model {model_id!r}. FoundationPose estimates "
                "the pose OF A KNOWN SHAPE; it cannot run without one. Known: "
                + (", ".join(sorted(self.registry.models)) or "(registry empty)"))
        if not model.supports(METHOD):
            return failed(f"{model_id} does not declare the {METHOD} method")
        mesh_path = model.resolved_path(self.registry.root)
        if not model.mesh_exists(self.registry.root):
            return failed(
                f"{model_id} has no mesh file at {mesh_path or '(unset)'}")

        # The registry root and the worker's read-only mount are THE SAME
        # DIRECTORY, so a registry-relative mesh path is already meaningful
        # inside the container and no translation is needed. A request never
        # carries a host path.
        request = {
            "dataset": dataset,
            "mesh_path": os.path.relpath(mesh_path, self.registry.root),
            # UNITS ARE PASSED EXPLICITLY, both of them, because neither can be
            # read from the file. The mesh unit comes from the registry, where
            # it is declared; the depth scale comes from the caller, who knows
            # the sensor. The worker has no default for either.
            "mesh_scale_to_metres": model.mesh_scale_to_mm / 1000.0,
            "depth_scale_mm": float(depth_scale_mm),
            "frame": int(frame),
            "refine_iterations": int(refine_iterations),
        }
        result, reason = self.client.estimate(request)
        if result is None:
            return failed(reason)

        validated, problem = validate_response(result)
        if problem:
            return failed(f"the worker's response could not be used: {problem}")

        observation = self.observation_from(
            validated, model=model, acquisition=acquisition,
            observation_id=f"{batch_id}-obj-1")

        return ObservationBatch(
            batch_id=batch_id,
            source=PerceptionSource.CAMERA.value,
            status=BatchStatus.OK,
            observations=[observation],
            # THE BATCH KEEPS THE CAMERA FRAME. Not `wisepack_workarea`: no
            # validated extrinsic exists, and relabelling the frame is how a
            # pose ends up placed in the wrong space with total confidence.
            frame_id=validated["frame_id"],
            captured_at=validated["captured_at"] or requested_at,
            requested_at=requested_at,
            detector=ESTIMATOR_ID,
            perception_method=METHOD,
            acquisition=acquisition,
            model_id=model.model_id,
            # A 6-DoF camera-frame pose has no ArUco plane behind it. Saying
            # "valid" would claim a calibration that was never performed.
            calibration_status="not_applicable",
            detector_status={
                "foundationpose_revision": validated["revision"],
                "duration_ms": validated["duration_ms"],
                "dataset": validated["dataset"],
                "frame_file": validated["frame_file"],
                "mesh_path": validated["mesh_path"],
                "pose_of": validated["pose_of"],
                "intrinsics": validated["intrinsics"],
                "acquisition": acquisition,
                "note": (REFERENCE_NOTE
                         if acquisition == ACQUISITION_REFERENCE else ""),
                "frame_note": NO_EXTRINSIC_NOTE,
                # NOT "accuracy", and not silently omitted either.
                "accuracy_note": result.get("accuracy_note", ""),
            })

    def observation_from(self, validated: Dict[str, Any], model,
                         acquisition: str,
                         observation_id: str) -> PhysicalObservation:
        """One validated worker response -> one domain observation.

        SYMMETRY IS APPLIED HERE, from the registry's MEASURED declaration, and
        the estimator's untouched output is kept beside it. Canonicalising in
        place and discarding the raw value would destroy the evidence needed to
        tell "the estimator was wrong" from "this rotation was never
        observable".
        """
        raw = validated["orientation"]
        canonical = canonicalize(raw, model.symmetry)
        changed = not _same_rotation(raw, canonical)

        return PhysicalObservation(
            observation_id=observation_id,
            # `x_mm`/`y_mm`/`yaw_deg` remain what every existing consumer
            # reads. They are the projection of the 6-DoF pose, not a second
            # independent measurement, and `measured_dof` says which of them
            # this method actually determined.
            x_mm=validated["x_mm"],
            y_mm=validated["y_mm"],
            z_mm=validated["z_mm"],
            # NOT SET HERE. `PhysicalObservation` derives `yaw_deg` as the
            # planar projection of the authoritative quaternion, which is
            # exactly the reconciliation wanted — and computing an Euler angle
            # in the provider would make a diagnostic quantity look like an
            # input.
            object_type=model.object_type,
            source=PerceptionSource.CAMERA.value,
            frame_id=validated["frame_id"],
            detector=ESTIMATOR_ID,
            model_id=validated["revision"] or ESTIMATOR_ID,
            captured_at=validated["captured_at"],
            calibration_status="not_applicable",
            diameter_mm=model.diameter_mm,
            length_mm=model.length_mm,
            geometry_source="cad_model",
            orientation=canonical,
            # KEPT ONLY WHEN IT DIFFERS. A raw copy identical to the canonical
            # one is noise in every payload; a raw copy that differs is the
            # record of what symmetry removed.
            orientation_raw=raw if changed else None,
            symmetry=model.symmetry,
            perception_method=METHOD,
            object_model_id=model.model_id,
            # THE ESTIMATE IS VALID. It is a real, reproducible 6-DoF pose in
            # the frame it declares, and reporting otherwise would call a good
            # measurement a failed one.
            pose_valid=True,
            # WHAT IS MISSING IS THE WAY TO MOVE IT. No validated SE(3)
            # camera-to-work-area extrinsic exists, so the pose cannot be placed
            # in the work area — `workarea_pose_available` derives False from
            # this plus the camera frame, and that is the flag a planner or the
            # Isaac synchronizer must consult. Never an identity transform: an
            # unmeasured extrinsic is missing, not identity.
            workarea_transform_valid=False,
            # KEYED ON WHETHER ANY DoF IS AMBIGUOUS, not on whether the
            # symmetry is axial. A discrete symmetry leaves rotation about the
            # axis observable only MODULO the fold — Cylinder5's pose is
            # determined up to a 180 deg leg swap and no further — and calling
            # that "orientation" measured would be exactly the over-claim the
            # symmetry declaration exists to prevent.
            measured_dof=("x", "y", "z") + (
                ("orientation",) if not model.symmetry.ambiguous_dof
                else ("orientation_partial",)),
            # NO CONFIDENCE. FoundationPose's score ranks hypotheses against
            # each other; it is not a probability that the pose is right, and
            # `confidence` is rendered as one throughout the dashboard.
            confidence=None,
        )


def _same_rotation(a: Orientation, b: Orientation, tolerance: float = 1e-9) -> bool:
    """Whether two quaternions express the same rotation (q and -q both do)."""
    dot = abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w)
    return abs(dot - 1.0) <= tolerance


def _utc_now() -> str:
    import time                                              # noqa: PLC0415
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = ["FoundationPoseProvider", "FoundationPoseProviderError",
           "validate_response", "PROVIDER_NAME", "METHOD", "ESTIMATOR_ID",
           "ACQUISITION_LIVE", "ACQUISITION_REFERENCE", "REFERENCE_NOTE",
           "NO_EXTRINSIC_NOTE"]
