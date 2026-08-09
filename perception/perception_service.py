#!/usr/bin/env python3
"""WISEPACK perception service — the host process that owns the camera.

WHAT THIS IS
------------
The WISEPACK-owned perception service. It is generic: it captures frames,
answers health, runs one-shot detections, keeps the raw and annotated images,
and returns `ObservationBatch` documents. It knows nothing about neural
networks, bottles, or how a pose is measured.

Everything detector-specific lives behind the PROVIDER boundary in
`perception/providers/`. Today that is `fasterrcnn_bottle`; selecting a
different provider — YOLO/OBB, RGB-D pose, segmentation — is
`WISEPACK_PERCEPTION_DETECTOR`, and changes nothing in this file.

WISEPACK OWNS ITS PERCEPTION RUNTIME
------------------------------------
Every executable line this service needs is in THIS repository:

    perception/perception_config.py   the settings, WISEPACK's own
    perception/camera.py              capture
    perception/calibration.py         ArUco plane -> millimetres
    perception/model_store.py         weights: resolve, cache, fetch
    perception/providers/*.py         the detectors

An earlier revision imported a HARMONY checkout at runtime — its `pipeline.py`,
its `camera.py`, its `config.json`, its `torch_venv`. That made another
repository a hard runtime dependency of a WISEPACK feature. It no longer is: the
current provider's detection pipeline is ADAPTED FROM HARMONY (MIT, see NOTICE)
and now lives here, so deleting `/data/arise/harmony` changes nothing.

WHY IT RUNS ON THE HOST
-----------------------
The host owns the camera, the GPU, torch, torchvision, OpenCV and the model
weights. The WISEPACK ROS 2 / Vulcanexus / Fast DDS stack runs in its CONTAINER,
where none of those exist. So perception is a host process and the container
reaches it over HTTP at `WISEPACK_PERCEPTION_SERVICE_URL`.

HTTP ONLY, DELIBERATELY. This service publishes no DDS and imports no rclpy: it
would otherwise need a host ROS 2 installation, and WISEPACK's validated
middleware is the CONTAINERIZED Vulcanexus runtime — the one whose TypeObject
propagation the Orion-LD DDS Enabler depends on. Duplicating middleware on the
host to publish from here would create a second, unvalidated path. The
orchestrator (inside the container, on Vulcanexus) fetches batches over HTTP and
publishes `/wisepack/perception/*` itself, so the DDS/NGSI-LD contract is
unchanged and still travels the validated route.

ONE OWNER OF THE CAMERA. `cv2.VideoCapture` does not survive two processes on
one device, so this is the only process that opens it. The dashboard never does.

INTERFACES
----------
    GET  /health                            liveness + every §5 health field
    GET  /api/v1/camera/snapshot            one JPEG frame (no inference)
    GET  /api/v1/camera/live                MJPEG preview   (no inference)
    POST /api/v1/detect                     ONE-SHOT inference -> ObservationBatch
    GET  /api/v1/camera/last-detection      the last batch, unchanged
    GET  /api/v1/detection/image/annotated  the detector's annotated result
    GET  /api/v1/detection/image/raw        the exact frame that was analysed

INFERENCE IS ONE-SHOT (§6). The MJPEG preview is raw frames only; the detector
runs when someone asks for a detection and not otherwise. A 50 ms preview loop
driving a half-second inference would produce a plan that changes under the
operator while they read it.

    .venv-perception/bin/python perception/perception_service.py --port 22101

Normally you do not run this by hand: the launcher starts it when
`WISEPACK_PERCEPTION_SOURCE=camera`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

for _path in (os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"), HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from wisepack_core.events import utc_now_iso                       # noqa: E402
from wisepack_core.perception import (                             # noqa: E402
    ObservationBatch, PerceptionHealth, PerceptionSource, ProxyGeometry,
    resolve_detector,
)
from perception_config import PerceptionConfig                     # noqa: E402
from calibration import PlaneCalibration                           # noqa: E402
from model_store import default_cache_dir, ensure_model            # noqa: E402
# THE ONLY detector-aware import in this file. One provider exists today; when a
# second is added this becomes a lookup on `resolve_detector()` and nothing else
# in the service changes.
from providers import fasterrcnn_bottle as PROVIDER                # noqa: E402

#: The port this service listens on. Deliberately NOT 22001: that is the port the
#: original HARMONY NGSI-v2 detector service uses, and both must be able to run
#: on one host without colliding.
DEFAULT_PORT = 22101


# --------------------------------------------------------------------------- #
# Detector runtime
# --------------------------------------------------------------------------- #


class DetectorRuntime:
    """Owns the camera, the calibration and the provider. One per process.

    EVERY FAILURE IS A REPORTED STATE, NOT AN EXCEPTION OUT OF THE SERVICE (§5,
    §15). A missing camera, missing weights, an unimportable torch or a failed
    inference each leave the process running and answering /health with the
    reason. The dashboard's job is to show that reason; it cannot show a
    traceback that killed the server.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._camera = None
        self._detector = None
        self._batch_counter = 0
        # WHICH PROVIDER. Resolved once, so an unknown name fails at start-up
        # rather than at the first detection.
        self.detector_name = resolve_detector()
        self.model = ensure_model(log=lambda message: print(message,
                                                            file=sys.stderr))
        self.config = PerceptionConfig.from_env(model_path=self.model.path or "")
        self.work_area = self.config.work_area()
        self.geometry = ProxyGeometry.from_env()
        # The plane lives on the RUNTIME, not inside the provider, because it is
        # the same geometry whichever detector finds the objects — and because
        # its marker cache must survive a provider that fails to load.
        self.calibration = PlaneCalibration(self.config.board)
        self.last_error: str = self.model.message if not self.model.available else ""
        self.last_inference_at: str = ""
        self.last_batch: Optional[ObservationBatch] = None
        self.last_annotated: Optional[bytes] = None
        self.last_raw: Optional[bytes] = None
        self.model_loaded = False
        self.calibration_status: str = "unknown"
        self.calibration_revision: str = ""

    # -- lazy heavy imports ------------------------------------------------ #

    def _ensure_detector(self) -> None:
        """Build the provider. Once, on the first detection.

        Deferred because it costs tens of seconds and ~159 MB of weights, and
        because the camera preview and /health must work on a host where the
        model has not arrived yet.
        """
        if self._detector is not None:
            return
        if not self.model.available:
            raise RuntimeError(self.model.message
                               or "detector weights are not available")
        try:
            self._detector = PROVIDER.build_detector(self.config,
                                                     self.calibration)
        except ImportError as exc:
            raise RuntimeError(
                f"the {self.detector_name} provider could not be imported "
                f"({exc}). It needs torch, torchvision and OpenCV — create the "
                "WISEPACK perception environment with "
                "./scripts/setup_perception.sh") from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"the detector could not load its weights: {exc}. "
                + (self.model.message or "")) from exc
        self.model_loaded = True

    def _ensure_camera(self) -> None:
        if self._camera is not None:
            return
        from camera import Camera                             # noqa: PLC0415
        self._camera = Camera(self.config.camera,
                              self.config.set_resolution,
                              self.config.width,
                              self.config.height)

    # -- frames ------------------------------------------------------------ #

    def frame(self, timeout_s: float = 3.0):
        """The newest camera frame, or None. Never raises for an absent camera."""
        try:
            self._ensure_camera()
        except Exception as exc:                             # noqa: BLE001
            self.last_error = f"camera unavailable: {exc}"
            return None
        frame = self._camera.wait_for_frame(timeout_s)
        if frame is None:
            self.last_error = self._camera.last_error or (
                f"camera {self.config.camera!r} is configured but delivered no "
                "frame — is it connected and not held by another process?")
        return frame

    def jpeg(self, frame) -> Optional[bytes]:
        try:
            import cv2                                       # noqa: PLC0415
            ok, buffer = cv2.imencode(".jpg", frame)
            return buffer.tobytes() if ok else None
        except Exception:                                    # noqa: BLE001
            return None

    @property
    def camera_available(self) -> bool:
        return self._camera is not None and self._camera.get_frame() is not None

    # -- the one-shot detection -------------------------------------------- #

    def detect(self) -> ObservationBatch:
        """Capture one frame, run inference, return a WISEPACK batch.

        Serialised by the lock: two concurrent detections would fight over one
        camera and one CUDA context, and the second would replace the first's
        images halfway through being served.
        """
        with self._lock:
            self._batch_counter += 1
            batch_id = f"batch-{self._batch_counter:03d}"
            source = PerceptionSource.CAMERA.value
            # TWO DIFFERENT INSTANTS, AND THEY ARE NOT INTERCHANGEABLE.
            #
            # `requested_at` is now. `captured_at` is whenever the camera
            # actually hands over a frame — which is AFTER the first-call model
            # load (~30 s cold) and after the frame wait. Stamping the request
            # time as the capture time made every batch look up to half a minute
            # older than the scene it described, and `captured_at` is the
            # instant a staleness check and the future Isaac synchronizer both
            # treat as "when the world looked like this".
            requested_at = utc_now_iso()

            try:
                self._ensure_detector()
            except Exception as exc:                         # noqa: BLE001
                self.last_error = str(exc)
                return ObservationBatch.failed(
                    batch_id=batch_id, source=source, error=str(exc),
                    # NO capture time: no frame was acquired. Inventing one would
                    # assert a measurement that never happened.
                    captured_at="", requested_at=requested_at,
                    detector=PROVIDER.DETECTOR_ID,
                    frame_id=self.work_area.frame_id)

            frame = self.frame()
            if frame is None:
                return ObservationBatch.failed(
                    batch_id=batch_id, source=source,
                    error=self.last_error or "no camera frame available",
                    captured_at="", requested_at=requested_at,
                    detector=PROVIDER.DETECTOR_ID,
                    model_id=self.model.path or "",
                    frame_id=self.work_area.frame_id)

            # THE FRAME IS IN HAND — this is the measurement instant. Stamped
            # before inference, because inference describes THIS frame and the
            # scene may have moved on while the network ran.
            captured_at = utc_now_iso()

            try:
                result = self._detector.process_frame(frame.copy())
            except Exception as exc:                         # noqa: BLE001
                self.last_error = f"inference failed: {exc}"
                return ObservationBatch.failed(
                    batch_id=batch_id, source=source, error=self.last_error,
                    captured_at=captured_at, requested_at=requested_at,
                    detector=PROVIDER.DETECTOR_ID,
                    model_id=self.model.path or "",
                    frame_id=self.work_area.frame_id)

            # THE PROVIDER REPORTS THE CALIBRATION FOR THIS VERY FRAME, so the
            # service never has to re-run marker detection (which would disturb
            # the plane cache) or guess from the coordinates.
            calibration = result.get("calibration") or {}
            self.calibration_status = str(calibration.get("status", "unknown"))
            self.calibration_revision = str(calibration.get("revision", ""))

            # `annotated_image` carries the calibration overlay and the measured
            # coordinates; `detections_image` carries only the raw boxes. The
            # former is what §9 asks the dashboard to show.
            self.last_annotated = self.jpeg(result.get("annotated_image"))
            self.last_raw = self.jpeg(frame)
            self.last_inference_at = captured_at

            batch = PROVIDER.observations_from_detections(
                result,
                batch_id=batch_id,
                captured_at=captured_at,
                requested_at=requested_at,
                model_id=self.model.path or "",
                geometry=self.geometry,
                frame=self.work_area,
                calibration_status=self.calibration_status,
                calibration_revision=self.calibration_revision,
                source=source)
            self.last_batch = batch
            self.last_error = batch.error if not batch.ok else ""
            return batch

    # -- health ------------------------------------------------------------ #

    def health(self) -> Dict[str, Any]:
        batch = self.last_batch
        health = PerceptionHealth(
            source=PerceptionSource.CAMERA.value,
            service_reachable=True,
            camera_configured=self.config.camera is not None,
            camera_available=self.camera_available,
            model_available=self.model.available,
            model_loaded=self.model_loaded,
            model_path=self.model.path or "",
            model_origin=self.model.origin,
            calibration_status=(batch.calibration_status if batch
                                else self.calibration_status),
            last_inference_at=self.last_inference_at,
            last_error=self.last_error,
            detected_objects=(batch.count if batch and batch.ok else None),
            detector=PROVIDER.DETECTOR_ID,
        ).to_dict()
        health.update({
            # THE PROVIDER, named generically. `implementation_origin` and
            # `model_origin_note` are PROVENANCE for diagnostics — never the
            # architectural identity of the perception subsystem, and the
            # dashboard shows the display name rather than either of them.
            "provider": self.detector_name,
            "detector_display_name": PROVIDER.DISPLAY_NAME,
            "implementation_origin": PROVIDER.IMPLEMENTATION_ORIGIN,
            "model_origin_note": PROVIDER.MODEL_ORIGIN,
            "provider_module": PROVIDER.__name__,
            "model": self.model.to_dict(),
            "model_cache": default_cache_dir(),
            "proxy_geometry": self.geometry.to_dict(),
            "work_area": self.work_area.to_dict(),
            "calibration_board": self.config.board.to_dict(),
            "camera": self.config.camera,
            "last_batch_status": (batch.status.value if batch else "none"),
            "batches": self._batch_counter,
            "note": ("Physical bottles are currently used as proxies for "
                     "cylindrical workpieces. Detector confidence is not a "
                     "detection rate."),
        })
        return health

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._camera is not None:
                self._camera.release()
        self._camera = None


# --------------------------------------------------------------------------- #
# No middleware here — on purpose
# --------------------------------------------------------------------------- #
#
# THIS SERVICE PUBLISHES NO DDS AND IMPORTS NO rclpy.
#
# An earlier revision started a ROS 2 node here and sourced a ROS environment
# for the subprocess. That was wrong for this deployment: WISEPACK's validated
# middleware is the CONTAINERIZED Vulcanexus / Fast DDS runtime, the one the
# Orion-LD DDS Enabler needs for TypeObject propagation. Publishing from the
# host would have required a SECOND middleware installation there and would have
# created a parallel, unvalidated path to the same NGSI-LD attributes — and
# plain host ROS 2 is not equivalent to Vulcanexus for that purpose.
#
# So the WISEPACK-domain topics are published where the validated stack already
# lives: `wisepack_orchestration.hitl_orchestrator`, inside the container, reads
# batches from this service over HTTP and publishes
# `/wisepack/perception/objects_json` and `/wisepack/perception/status_json`
# itself. One authority, one middleware, unchanged DDS contract.
#
# It also means `./run_wisepack_dashboard.sh sim` needs no ROS anywhere.

# --------------------------------------------------------------------------- #
# HTTP application
# --------------------------------------------------------------------------- #


def create_app(runtime: Optional[DetectorRuntime] = None):
    """Build the FastAPI application. FastAPI is imported HERE, not at module
    scope, so the helpers above stay importable (and testable) without it."""
    from fastapi import FastAPI, HTTPException                # noqa: PLC0415
    from fastapi.responses import Response, StreamingResponse  # noqa: PLC0415

    runtime = runtime or DetectorRuntime()

    app = FastAPI(title="WISEPACK perception service")

    @app.on_event("shutdown")
    def _shutdown() -> None:
        runtime.close()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return runtime.health()

    @app.get("/api/v1/camera/snapshot")
    def snapshot():
        frame = runtime.frame()
        if frame is None:
            raise HTTPException(503, runtime.last_error or "camera not ready")
        image = runtime.jpeg(frame)
        if image is None:
            raise HTTPException(503, "frame could not be encoded")
        return Response(image, media_type="image/jpeg")

    @app.get("/api/v1/camera/live")
    def live():
        """RAW MJPEG preview. No inference — see the module docstring (§6)."""
        def frames():
            while True:
                frame = runtime.frame(timeout_s=1.0)
                if frame is None:
                    time.sleep(0.5)
                    continue
                image = runtime.jpeg(frame)
                if image is None:
                    time.sleep(0.5)
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + image + b"\r\n")
                time.sleep(0.05)

        return StreamingResponse(
            frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/v1/detect")
    def detect() -> Dict[str, Any]:
        """ONE-SHOT detection. 200 even for a failed batch, with the reason.

        A failure is a RESULT here, not a transport error: the operator asked
        whether the detector can see the table and the answer "no, because the
        calibration sheet is out of frame" is exactly as useful as a list of
        objects. An HTTP 500 would lose it.
        """
        return runtime.detect().to_dict()

    @app.get("/api/v1/camera/last-detection")
    def last_detection() -> Dict[str, Any]:
        if runtime.last_batch is None:
            return {"status": "none",
                    "message": "no detection has been requested yet"}
        return runtime.last_batch.to_dict()

    @app.get("/api/v1/detection/image/annotated")
    def annotated():
        if runtime.last_annotated is None:
            raise HTTPException(404, "no annotated result yet — run a detection")
        return Response(runtime.last_annotated, media_type="image/jpeg")

    @app.get("/api/v1/detection/image/raw")
    def raw():
        if runtime.last_raw is None:
            raise HTTPException(404, "no analysed frame yet — run a detection")
        return Response(runtime.last_raw, media_type="image/jpeg")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(
        os.environ.get("WISEPACK_PERCEPTION_SERVICE_PORT", DEFAULT_PORT)))
    parser.add_argument("--check", action="store_true",
                        help="print the resolved configuration and exit")
    args = parser.parse_args()

    runtime = DetectorRuntime()
    if args.check:
        print(json.dumps(runtime.health(), indent=2))
        return
    if not runtime.model.available:
        # A WARNING, not an exit. The preview and /health still work, and an
        # operator debugging a camera should not be blocked on the weights.
        print(f"[perception] WARNING: {runtime.model.message}", file=sys.stderr)

    import uvicorn                                            # noqa: PLC0415
    uvicorn.run(create_app(runtime), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
