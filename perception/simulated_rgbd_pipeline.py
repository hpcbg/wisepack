"""The simulated RGB-D -> FoundationPose -> workarea pipeline, in ONE place.

WHY THIS MODULE EXISTS. Exactly the reason `physical_pipeline.py` exists, one
axis over: the same acquisition now has two callers — the Stage B/C CLIs and the
dashboard's *Acquire & estimate* button — and two copies of "render, estimate,
transform, evaluate" would agree only until somebody edited one. The estimate an
operator sees in the browser must be produced by the code the regression scripts
run, or the demonstration and the evidence become two different measurements
wearing one name.

WHAT MOVED HERE, AND FROM WHERE
-------------------------------
    Stage B   scripts/stage_b_foundationpose.py   ->  estimate() + evaluate_camera_frame()
    Stage C   scripts/stage_c_workarea.py         ->  to_workarea() + evaluate_workarea()

Both scripts are now thin CLIs over these functions. They keep their printed
output, because that is what makes them useful as deterministic regression
helpers; what they no longer keep is an implementation.

WHAT IT DOES NOT DO, and these are the load-bearing absences:

* NO SECOND ESTIMATOR. Inference is the FoundationPose worker's, reached through
  the ordinary `FoundationPoseProvider`. Nothing here registers a mesh, and
  nothing here re-implements a pose.
* NO GROUND TRUTH IN THE PERCEPTION PATH. `estimate()` does not read the scene's
  ground truth at all; it cannot, because it never receives it. The evaluation
  functions take an ALREADY-COMPUTED observation as an argument, so ground truth
  can only ever be applied after an estimate exists. That ordering is structural,
  not a convention to be remembered.
* NO INVENTED TRANSFORM. `to_workarea()` applies the transform the SCENE
  exported, through the same generic `RigidTransform` the physical camera will
  use once a measured extrinsic exists. Only the source of the numbers differs.
  Where no transform is exported, the observation stays in the camera frame and
  says so — it is never relabelled.
* NO CLAIM OF SENSOR FIDELITY. The frame is rendered by a camera built to the
  D435's PUBLISHED GEOMETRY (`config/rgbd_sensors.yaml`). It is
  `d435_compatible_simulated`, never "a D435 in Isaac": no stereo matching, no IR
  pattern, no dropouts on low-texture metal. `provenance` says `simulated` and
  `acquisition` says `isaac_simulated` on every artefact this module writes.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(REPO, "perception"),
              os.path.join(REPO, "wisepack_ws", "src", "wisepack_core")):
    if _path not in sys.path:                                 # pragma: no cover
        sys.path.insert(0, _path)

from wisepack_core.acquisition import (                          # noqa: E402
    ACQUISITION_ISAAC, acquisition_provenance)

WORKER = f"http://127.0.0.1:{os.environ.get('WISEPACK_FP_PORT', '22201')}"

#: The Isaac export Stage A writes and the worker reads. One name, so a fresh
#: render replaces the frame an estimate will be computed from rather than
#: accumulating datasets nobody can tell apart.
DATASET = "stage_a_workcell"
FRAME = os.path.join(REPO, ".cache-perception", "isaac-reference", DATASET)

#: The artefacts, at the paths the existing panels and `--reuse` already read.
#: Keeping these locations is what lets `stage_b.sh`/`stage_c.sh --reuse` and the
#: dashboard's image endpoints go on working against one set of files.
STAGE_A_OUT = os.path.join(REPO, ".cache-perception", "stage-a")
STAGE_B_OUT = os.path.join(REPO, ".cache-perception", "stage-b")
STAGE_C_OUT = os.path.join(REPO, ".cache-perception", "stage-c")
STAGE_B_RESULT = os.path.join(STAGE_B_OUT, "stage_b.json")
STAGE_C_RESULT = os.path.join(STAGE_C_OUT, "stage_c.json")

#: The renderer writes uint16 millimetres, the encoding a real depth sensor
#: produces. Declared, never detected: a depth image in millimetres is
#: indistinguishable from one in metres.
DEPTH_SCALE_MM = 1.0

#: How long one Isaac render may take before it is abandoned. Isaac start-up
#: dominates; the render itself is seconds.
ISAAC_TIMEOUT_S = 900

#: Said on every artefact. Present in the payload so a consumer cannot render a
#: simulated result without it, in the same way the proxy disclosure is.
SIMULATED_NOTE = (
    "SIMULATED ACQUISITION — colour and depth were RENDERED by Isaac Sim "
    "through a camera built to the D435's published geometry, not read from a "
    "physical sensor. There is no stereo matching, no IR pattern and no "
    "dropout on low-texture metal. The FoundationPose estimator is real and is "
    "the same one the physical path uses.")

#: Why the workarea pose is legitimately available here and not on the physical
#: path. Carried with the result rather than left for a reader to work out.
KNOWN_TRANSFORM_NOTE = (
    "The camera-to-workarea transform is EXACT AND KNOWN because the camera is "
    "part of the simulated scene, which exported it. It is applied through the "
    "same generic RigidTransform the physical camera will use once a measured "
    "extrinsic exists. This is not the physical path's situation and its "
    "workaround is not used here.")


class SimulatedResult:
    """What one simulated acquisition produced, for both of its callers.

    THE BATCH IS RETURNED, NOT REBUILT. The workarea batch is constructed once,
    here, on the way to writing the document; handing a caller the document
    alone would force the dashboard to reassemble a batch from JSON, and a
    second construction is a second place for the CAD geometry, the frame or the
    provenance to be assembled differently.
    """

    def __init__(self, document: Dict[str, Any], batch: Any,
                 camera_batch: Any) -> None:
        self.document = document
        #: The batch the WORKFLOW consumes — workarea frame when a transform
        #: exists, camera frame when it does not.
        self.batch = batch
        #: What FoundationPose actually reported, kept beside it. Evidence, not
        #: an alternative input.
        self.camera_batch = camera_batch


class SimulatedAcquisitionError(RuntimeError):
    """A failure with the STAGE that refused and the reason an operator needs.

    A renderer that never started and an estimator that found no pose are
    different problems; one "acquisition failed" would be neither.
    """

    def __init__(self, stage: str, reason: str,
                 detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason
        self.detail = detail or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "reason": self.reason,
                "acquisition": ACQUISITION_ISAAC,
                "provenance": acquisition_provenance(ACQUISITION_ISAAC),
                **self.detail}


def _noop(_message: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# Capability
# --------------------------------------------------------------------------- #


def isaac_available() -> Tuple[bool, str]:
    """Can a FRESH frame be rendered right now?

    A SEPARATE QUESTION FROM "is there a frame on disk". A previously exported
    frame makes re-estimation possible; it says nothing about whether the
    renderer can run again, and conflating the two is how an operator ends up
    pressing a button that cannot do what it says.
    """
    root = os.environ.get("WISEPACK_ISAAC_SIM_ROOT") or os.environ.get(
        "ISAAC_SIM_ROOT", "")
    if root:
        return (os.access(os.path.join(root, "python.sh"), os.X_OK),
                "" if os.access(os.path.join(root, "python.sh"), os.X_OK)
                else f"no executable python.sh under {root}")
    import glob                                                # noqa: PLC0415
    for candidate in sorted(glob.glob("/data/isaac-sim/isaac-sim-6.*"),
                            reverse=True):
        if os.access(os.path.join(candidate, "python.sh"), os.X_OK):
            return True, ""
    return False, ("no Isaac Sim 6.x with a bundled python.sh was found; set "
                   "ISAAC_SIM_ROOT to point at one")


def frame_available() -> bool:
    """Is there an exported Isaac frame an estimate can be computed from?"""
    return os.path.isfile(os.path.join(FRAME, "ground_truth.json"))


#: The images the panel shows, fetched from the worker after the estimate. The
#: OVERLAY especially: it is drawn from the pose that was just computed, so a
#: panel can never show a picture of an earlier estimate beside a newer number.
ARTIFACTS = (("rgb", "rgb.jpg"), ("depth", "depth.jpg"), ("mask", "mask.jpg"),
             ("overlay", "pose_overlay.jpg"))


def fetch_images(destination: str = STAGE_B_OUT) -> Dict[str, str]:
    """The worker's own renderings of the frame it just estimated from.

    DIAGNOSTICS, NOT DATA. These are re-encoded for looking at; the uint16 depth
    and the raw colour stay in the exported dataset. A missing image is reported
    as missing rather than substituted from an earlier run.
    """
    import urllib.request                                      # noqa: PLC0415
    os.makedirs(destination, exist_ok=True)
    written: Dict[str, str] = {}
    for kind, name in ARTIFACTS:
        path = os.path.join(destination, name)
        try:
            with urllib.request.urlopen(f"{WORKER}/image/{kind}",
                                        timeout=30) as response:
                data = response.read()
        except Exception:                                      # noqa: BLE001
            # STALE IMAGES ARE WORSE THAN NONE. If this estimate produced no
            # picture of a given kind, an older file left in place would be
            # shown beside a newer pose as though it belonged to it.
            if os.path.isfile(path):
                os.remove(path)
            continue
        with open(path, "wb") as handle:
            handle.write(data)
        written[kind] = path
    return written


def require_worker() -> Dict[str, Any]:
    """The FoundationPose worker, or a refusal naming the missing layer.

    NO RGB-D CAMERA IS REQUIRED, and that is the whole point of the acquisition
    axis: a simulated run needs no RealSense, and demanding one made a working
    Isaac run report itself blocked on hardware it never used.
    """
    import urllib.request                                      # noqa: PLC0415
    try:
        with urllib.request.urlopen(f"{WORKER}/health", timeout=30) as response:
            health = json.loads(response.read().decode("utf-8"))
    except Exception as exc:                                   # noqa: BLE001
        raise SimulatedAcquisitionError(
            "worker",
            f"the FoundationPose worker is not answering at {WORKER}: "
            f"{type(exc).__name__}: {exc}. Start it: "
            "./scripts/setup_foundationpose.sh --no-build --run") from exc
    if not health.get("inference_available"):
        raise SimulatedAcquisitionError(
            "inference",
            "FoundationPose inference is unavailable: "
            + "; ".join(str(b) for b in (health.get("blocked_by")
                                         or ["no reason given"])))
    return health


# --------------------------------------------------------------------------- #
# Stage A — the simulated acquisition
# --------------------------------------------------------------------------- #


def acquire_frame(log: Callable[[str], None] = _noop,
                  timeout_s: int = ISAAC_TIMEOUT_S) -> Dict[str, Any]:
    """Render one RGB-D frame of the workcell in Isaac. A SUBPROCESS, of course.

    Isaac Sim runs under its own bundled interpreter with its own extensions; it
    cannot be imported into the dashboard process or into `.venv-perception`.
    So this invokes the SAME runner the Stage A script invokes, with the SAME
    scene script — one renderer, reached one way. It is not a shell-out to
    another stage script, and it parses nothing from the log: the runner reports
    by exit code and the frame reports by existing.
    """
    ok, reason = isaac_available()
    if not ok:
        raise SimulatedAcquisitionError("renderer", reason)
    runner = os.path.join(REPO, "scripts", "run_isaac_task.sh")
    scene = os.path.join(REPO, "simulators", "isaac", "stage_a_check.py")
    log_path = os.path.join(STAGE_A_OUT, "isaac.log")
    os.makedirs(STAGE_A_OUT, exist_ok=True)
    log("rendering the workcell in Isaac Sim (this takes about a minute)")
    started = time.monotonic()
    completed = subprocess.run(
        [runner, log_path, str(timeout_s), scene],
        capture_output=True, text=True, check=False)
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise SimulatedAcquisitionError(
            "renderer",
            f"the Isaac acquisition failed (exit {completed.returncode}). "
            f"The full log is at {log_path}.",
            {"log_path": log_path, "elapsed_s": round(elapsed, 1)})
    if not frame_available():
        raise SimulatedAcquisitionError(
            "renderer",
            "Isaac exited successfully but exported no frame; there is nothing "
            "to estimate from.", {"log_path": log_path})
    log(f"rendered in {elapsed:.0f}s")
    return {"log_path": log_path, "elapsed_s": round(elapsed, 1)}


def load_scene() -> Dict[str, Any]:
    """The Stage A export: provenance, the transform, and the ground truth.

    ONE READ, and the caller decides what to do with each part. Nothing in this
    function separates them, because the separation that matters is WHEN each is
    used, and that is enforced by which function receives it.
    """
    path = os.path.join(FRAME, "ground_truth.json")
    if not os.path.isfile(path):
        raise SimulatedAcquisitionError(
            "frame",
            "no simulated RGB-D frame has been exported. Acquire one first — "
            "from the dashboard, or with ./scripts/stage_a.sh")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def frame_provenance(scene: Dict[str, Any]) -> Dict[str, Any]:
    """How the frame was produced, stated rather than inferred downstream."""
    return {
        "acquisition": ACQUISITION_ISAAC,
        "acquisition_backend": scene.get("acquisition_backend", "isaac_sim"),
        "camera_profile": scene.get("camera_profile",
                                    "d435_compatible_simulated"),
        "provenance": acquisition_provenance(ACQUISITION_ISAAC),
        "mask_source": scene.get("mask_source", ""),
        "mask_provenance": scene.get("mask_provenance", "synthetic"),
        "simulated_note": SIMULATED_NOTE,
    }


# --------------------------------------------------------------------------- #
# Stage B — the estimate. NO GROUND TRUTH REACHES THIS.
# --------------------------------------------------------------------------- #


def estimate(model_id: str, refine_iterations: int = 5,
             batch_id: str = "simulated-rgbd-1",
             provider: Optional[Any] = None) -> Any:
    """FoundationPose on the rendered frame, in the CAMERA optical frame.

    THE SIGNATURE IS THE GUARD. This function is not given the scene, so it
    cannot read a ground-truth pose even by accident; what it receives is the
    dataset name, the CAD identity and an iteration count — exactly what the
    physical path sends for a real capture.
    """
    if provider is None:
        from providers.foundationpose_rgbd import (              # noqa: PLC0415
            FoundationPoseProvider)
        provider = FoundationPoseProvider()
    batch = provider.acquire_simulated(
        dataset=DATASET, model_id=model_id, depth_scale_mm=DEPTH_SCALE_MM,
        frame=0, refine_iterations=refine_iterations, batch_id=batch_id)
    if not batch.ok or not batch.observations:
        raise SimulatedAcquisitionError(
            "estimation",
            batch.error or "FoundationPose produced no pose from this frame. "
                           "No fallback is applied: neither ground truth, nor "
                           "an earlier result, nor planar perception is "
                           "substituted.")
    return batch


# --------------------------------------------------------------------------- #
# Stage C — the known transform. STILL NO GROUND TRUTH.
# --------------------------------------------------------------------------- #


def workarea_transform(scene: Dict[str, Any]) -> Optional[Any]:
    """The scene's exported camera->workarea SE(3), as a generic transform.

    READ AS DATA, NEVER TYPED IN. The scene composed it from the camera prim's
    own pose and the layout's workarea origin. None when the export carries
    none, which is a real answer: the observation then stays in the camera
    frame rather than being relabelled into one nobody measured.
    """
    import numpy as np                                         # noqa: PLC0415
    from wisepack_core.pose import (                           # noqa: PLC0415
        CAMERA_OPTICAL_FRAME, WORKAREA_FRAME, Orientation, RigidTransform)

    workarea = scene.get("workarea") or {}
    matrix = workarea.get("T_workarea_camera")
    if not matrix:
        return None
    array = np.asarray(matrix, dtype=np.float64)
    return RigidTransform(
        parent_frame=WORKAREA_FRAME, child_frame=CAMERA_OPTICAL_FRAME,
        translation_mm=tuple(float(v) * 1000.0 for v in array[:3, 3]),
        rotation=Orientation.from_matrix([row[:3] for row in array[:3]]),
        method=workarea.get("method", ""),
        notes=workarea.get("note", ""))


def to_workarea(observation: Any, transform: Any) -> Any:
    """One camera-frame observation, expressed in the work area.

    THE WHOLE POSE, not just the position: an orientation left in the camera
    frame beside a transformed position is two different measurements wearing
    one name.
    """
    from wisepack_core.domain import PhysicalObservation        # noqa: PLC0415
    from wisepack_core.pose import WORKAREA_FRAME               # noqa: PLC0415

    position = transform.apply_to_position(
        (observation.x_mm, observation.y_mm, observation.z_mm))
    orientation = transform.apply_to_orientation(observation.orientation)
    return PhysicalObservation(
        observation_id=observation.observation_id,
        x_mm=position[0], y_mm=position[1], z_mm=position[2],
        object_type=observation.object_type, source=observation.source,
        frame_id=WORKAREA_FRAME,
        detector=observation.detector, model_id=observation.model_id,
        captured_at=observation.captured_at,
        calibration_status=observation.calibration_status,
        diameter_mm=observation.diameter_mm, length_mm=observation.length_mm,
        inner_diameter_mm=observation.inner_diameter_mm,
        geometry_source=getattr(observation, "geometry_source", ""),
        orientation=orientation, symmetry=observation.symmetry,
        perception_method=observation.perception_method,
        object_model_id=observation.object_model_id,
        model_center_mm=observation.model_center_mm,
        task_axis_vector=observation.task_axis_vector,
        pose_valid=True,
        # A VALIDATED TRANSFORM EXISTS, so the pose can be placed. The
        # camera-frame estimate was already valid; this is the separate question
        # of whether it can be MOVED into the work area, and here it can.
        workarea_transform_valid=True,
        measured_dof=observation.measured_dof,
        confidence=observation.confidence)


def workarea_batch(camera_batch: Any, observation: Any) -> Any:
    """The batch the workflow consumes, built from the transformed observation.

    GENERIC AND UNREMARKABLE, on purpose. Nothing on it says "simulated" beyond
    the `acquisition` field every batch carries; there is no simulated-only
    schema, and the orchestrator adopts it exactly as it adopts a planar or a
    physical one.
    """
    from wisepack_core.perception import (                      # noqa: PLC0415
        BatchStatus, ObservationBatch, PerceptionSource)
    return ObservationBatch(
        batch_id=camera_batch.batch_id,
        source=PerceptionSource.CAMERA.value, status=BatchStatus.OK,
        observations=[observation], frame_id=observation.frame_id,
        captured_at=camera_batch.captured_at,
        requested_at=camera_batch.requested_at,
        detector=camera_batch.detector,
        perception_method=camera_batch.perception_method,
        acquisition=ACQUISITION_ISAAC,
        model_id=camera_batch.model_id,
        calibration_status="not_applicable",
        detector_status=dict(camera_batch.detector_status or {}))


# --------------------------------------------------------------------------- #
# Evaluation — GROUND TRUTH ENTERS ONLY HERE, AND ONLY WITH AN ESTIMATE IN HAND
# --------------------------------------------------------------------------- #


def _task_axis(model) -> Any:
    return tuple(model.task_axis_vector or ()) or model.task_axis


def _unit(axis: Any) -> Any:
    import numpy as np                                         # noqa: PLC0415
    vector = (np.asarray(axis, dtype=np.float64)
              if not isinstance(axis, str)
              else np.asarray({"x": (1, 0, 0), "y": (0, 1, 0),
                               "z": (0, 0, 1)}[axis], dtype=np.float64))
    return vector / np.linalg.norm(vector)


def evaluate_camera_frame(observation: Any, scene: Dict[str, Any],
                          model) -> Dict[str, Any]:
    """Score a camera-frame estimate against the simulator's own pose.

    THE ESTIMATE IS AN ARGUMENT. This function cannot run before one exists,
    which is the structural reason ground truth cannot leak into perception.

    THE ANGULAR METRIC IS THE TUBE-AXIS LINE, undirected:
    `acos(|dot(axis_est, axis_gt)|)`. A straight tube's end-for-end reversal
    describes the same object, and its spin about its own axis is not a task
    quantity — scoring either as error would report a correct pose as wrong.
    """
    import numpy as np                                         # noqa: PLC0415
    from wisepack_core.pose import (Orientation,                # noqa: PLC0415
                                    axis_line_angle_deg,
                                    symmetry_aware_angle_deg, Symmetry)

    axis = _task_axis(model)
    centre_mm = np.asarray(scene["model_center_mm"], dtype=np.float64)
    truth_matrix = np.asarray(scene["T_camera_object"], dtype=np.float64)
    truth_orientation = Orientation.from_matrix(
        [row[:3] for row in truth_matrix[:3]])

    truth_point = truth_matrix[:3, :3] @ centre_mm + truth_matrix[:3, 3] * 1000.0
    estimate_matrix = np.asarray(observation.orientation.to_matrix(),
                                 dtype=np.float64)
    estimate_point = (estimate_matrix @ centre_mm
                      + np.asarray([observation.x_mm, observation.y_mm,
                                    observation.z_mm], dtype=np.float64))
    delta = estimate_point - truth_point

    truth_axis = truth_matrix[:3, :3] @ _unit(axis)
    truth_axis = truth_axis / np.linalg.norm(truth_axis)
    along = float(delta @ truth_axis)
    return {
        "reference_point_gt_mm": [float(v) for v in truth_point],
        "reference_point_estimate_mm": [float(v) for v in estimate_point],
        "position_error_mm": float(np.linalg.norm(delta)),
        "along_axis_mm": along,
        "transverse_mm": math.sqrt(max(0.0, float(delta @ delta) - along ** 2)),
        "tube_axis_line_error_deg": axis_line_angle_deg(
            observation.orientation, truth_orientation, axis),
        # REPORTED, AND NAMED AS NOT THE TASK METRIC. A straight tube's spin is
        # weakly constrained from some views and irrelevant to picking, so this
        # number is evidence about observability, not about error.
        "full_geometric_orientation_error_deg": symmetry_aware_angle_deg(
            observation.orientation, truth_orientation,
            Symmetry.from_dict(scene["symmetry"])),
        "note": ("Isaac ground truth is EVALUATION ONLY; it never entered the "
                 "perception path and never reaches the planner."),
    }


def evaluate_workarea(observation: Any, scene: Dict[str, Any], model,
                      transform: Any) -> Dict[str, Any]:
    """Score the transformed estimate against where physics actually left it.

    AGAINST THE SETTLED POSE, not the scenario's requested placement: the tube
    moved while physics resolved, so scoring against the request would measure
    the scenario rather than the perception.
    """
    import numpy as np                                         # noqa: PLC0415
    from wisepack_core.pose import (Orientation,                # noqa: PLC0415
                                    axis_line_angle_deg)

    axis = _task_axis(model)
    settled = np.asarray(scene["settled_workarea"]["position_mm"],
                         dtype=np.float64)
    estimate_point = np.asarray(observation.object_center, dtype=np.float64)
    delta = estimate_point - settled

    truth_camera = Orientation.from_matrix(
        [row[:3] for row in np.asarray(scene["T_camera_object"],
                                       dtype=np.float64)[:3]])
    truth_workarea = transform.apply_to_orientation(truth_camera)
    truth_axis = (np.asarray(truth_workarea.to_matrix(), dtype=np.float64)
                  @ _unit(axis))
    truth_axis = truth_axis / np.linalg.norm(truth_axis)
    along = float(delta @ truth_axis)
    return {
        "compared_against": ("settled Isaac pose after physics, NOT the "
                             "scenario's requested placement"),
        "settled_position_mm": [float(v) for v in settled],
        "estimated_object_center_mm": [float(v) for v in estimate_point],
        "position_error_mm": float(np.linalg.norm(delta)),
        "along_axis_mm": along,
        "transverse_mm": math.sqrt(max(0.0, float(delta @ delta) - along ** 2)),
        "tube_axis_line_error_deg": axis_line_angle_deg(
            observation.orientation, truth_workarea, axis),
        "note": ("Isaac ground truth is EVALUATION ONLY; the workarea "
                 "observation was built from the FoundationPose estimate."),
    }


# --------------------------------------------------------------------------- #
# The whole pipeline
# --------------------------------------------------------------------------- #


def run(model_id: str = "", refine_iterations: int = 5, acquire: bool = True,
        batch_id: str = "simulated-rgbd-1",
        log: Callable[[str], None] = _noop) -> "SimulatedResult":
    """Render (optionally), estimate, transform, evaluate — and write artefacts.

    RAISES `SimulatedAcquisitionError` at the stage that refused. Returns the
    same document `.cache-perception/stage-c/stage_c.json` holds, which is what
    the dashboard panel renders and what `--reuse` replays.

    `acquire=False` re-estimates from the frame already exported. THE ESTIMATE
    IS STILL FRESH — the worker runs again — and the document says which of the
    two happened rather than letting a reused frame read as a new render.
    """
    from wisepack_core.rgbd import load_object_registry          # noqa: PLC0415

    require_worker()
    rendered: Dict[str, Any] = {}
    if acquire:
        rendered = acquire_frame(log)
    elif not frame_available():
        raise SimulatedAcquisitionError(
            "frame",
            "no simulated RGB-D frame has been exported, so there is nothing "
            "to re-estimate from. Acquire one instead of reusing.")

    scene = load_scene()
    # THE MODEL IS THE OPERATOR'S OR THE SCENE'S, never guessed from the image.
    model_id = model_id or str(scene.get("model_id") or "")
    registry = load_object_registry(repo_root=REPO)
    model = registry.models.get(model_id)
    if model is None:
        raise SimulatedAcquisitionError(
            "model", f"unknown object model {model_id!r}; known: "
                     + (", ".join(sorted(registry.models)) or "(none)"))

    log(f"estimating {model_id} with FoundationPose")
    camera_batch = estimate(model_id, refine_iterations, batch_id)
    camera_observation = camera_batch.observations[0]

    os.makedirs(STAGE_B_OUT, exist_ok=True)
    os.makedirs(STAGE_C_OUT, exist_ok=True)

    # THE PICTURES OF THIS ESTIMATE, fetched before anything else can run and
    # replace them.
    images = fetch_images()

    # ---- evaluation of the camera-frame estimate, AFTER it exists ----------
    camera_evaluation = evaluate_camera_frame(camera_observation, scene, model)
    with open(STAGE_B_RESULT, "w", encoding="utf-8") as handle:
        json.dump({"observation": camera_observation.to_dict(),
                   "batch": {k: v for k, v in camera_batch.to_dict().items()
                             if k != "observations"},
                   "acquisition": frame_provenance(scene),
                   "evaluation": camera_evaluation},
                  handle, indent=2, default=str)

    # ---- the known transform, and the batch the workflow consumes ----------
    transform = workarea_transform(scene)
    if transform is None:
        # NO TRANSFORM, NO WORKAREA POSE. The camera-frame batch is returned as
        # it is; relabelling its frame to get a placeable-looking pose is the
        # one thing this module must never do.
        document = {
            "model_id": model_id,
            "acquisition": frame_provenance(scene),
            "run_mode": "acquired" if acquire else "reused_frame",
            "camera_frame_id": camera_observation.frame_id,
            "workarea_pose_available": False,
            "workarea_note": ("the exported frame carries no camera-to-workarea "
                              "transform, so the pose stays in the camera frame"),
            "observation": camera_observation.to_dict(),
            "evaluation": camera_evaluation,
            "images": dict(images),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(STAGE_C_RESULT, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, default=str)
        return SimulatedResult(document, camera_batch, camera_batch)

    observation = to_workarea(camera_observation, transform)
    batch = workarea_batch(camera_batch, observation)
    workarea_evaluation = evaluate_workarea(observation, scene, model, transform)

    axis_vector = _task_axis(model)
    document = {
        "model_id": model_id,
        "perception_method": camera_observation.perception_method,
        "acquisition": frame_provenance(scene),
        # ACQUIRED AND REUSED ARE DIFFERENT CLAIMS, and both are simulated. A
        # reused frame is not evidence that the renderer ran just now.
        "run_mode": "acquired" if acquire else "reused_frame",
        "run_label": ("SIMULATED RGB-D — RENDERED THIS RUN" if acquire
                      else "SIMULATED RGB-D — RE-ESTIMATED FROM THE LAST RENDER"),
        "run_note": (
            "Isaac rendered the frame during this run and FoundationPose "
            "estimated from it." if acquire else
            "FoundationPose estimated again from the frame Isaac rendered "
            "earlier. The estimate is fresh; the frame is not."),
        "isaac_render": rendered,
        "camera_frame_id": camera_observation.frame_id,
        "camera_frame_pose": {
            "position_mm": [camera_observation.x_mm, camera_observation.y_mm,
                            camera_observation.z_mm],
            "orientation": camera_observation.orientation.to_dict(),
        },
        "camera_to_workarea_transform": transform.to_dict(),
        "camera_to_workarea_note": KNOWN_TRANSFORM_NOTE,
        # NAMED, so neither can be mistaken for the other.
        "model_frame_pose": {
            "model_frame_origin_mm": [observation.x_mm, observation.y_mm,
                                      observation.z_mm],
            "orientation": observation.orientation.to_dict(),
            "note": ("the pose of the CAD model frame, as FoundationPose "
                     "reports it. Its origin can lie outside the body."),
        },
        "task_reference_point": {
            "object_center_mm": [float(v) for v in observation.object_center],
            "tube_axis_line": [float(v) for v in (observation.tube_axis or ())],
            "diameter_mm": observation.diameter_mm,
            "length_mm": observation.length_mm,
            "inner_diameter_mm": observation.inner_diameter_mm,
            "note": ("the physical body centre and long axis. THIS is what a "
                     "grasp targets."),
        },
        "workarea_frame_id": observation.frame_id,
        "workarea_pose_available": observation.workarea_pose_available,
        "task_axis_in_model_frame": (list(axis_vector)
                                     if not isinstance(axis_vector, str)
                                     else axis_vector),
        "observation": observation.to_dict(),
        "camera_frame_evaluation": camera_evaluation,
        "evaluation": workarea_evaluation,
        "images": dict(images),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(STAGE_C_RESULT, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, default=str)
    return SimulatedResult(document, batch, camera_batch)


__all__ = ["SimulatedAcquisitionError", "SimulatedResult", "run", "estimate",
           "to_workarea", "workarea_batch", "workarea_transform",
           "evaluate_camera_frame", "evaluate_workarea", "acquire_frame",
           "load_scene", "frame_provenance", "frame_available",
           "isaac_available", "require_worker", "DATASET", "FRAME",
           "DEPTH_SCALE_MM", "STAGE_B_RESULT", "STAGE_C_RESULT",
           "SIMULATED_NOTE", "KNOWN_TRANSFORM_NOTE"]
