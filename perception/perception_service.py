#!/usr/bin/env python3
"""WISEPACK perception service — the host process that owns the camera.

WHAT THIS IS
------------
The WISEPACK-owned perception service. It is generic: it captures frames,
answers health, runs one-shot detections, keeps the raw and annotated images,
and returns `ObservationBatch` documents. It knows nothing about neural
networks, bottles, or where any detector came from.

Everything detector-specific lives behind the PROVIDER boundary in
`perception/providers/`. Today that is `fasterrcnn_bottle`, which reuses the
Faster R-CNN + ArUco implementation developed in HARMONY rather than rewriting
it. Selecting a different provider — YOLO/OBB, RGB-D pose, segmentation — is
`WISEPACK_PERCEPTION_DETECTOR`, and changes nothing in this file.

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

    <detector-python> perception/perception_service.py --port 22101

Normally you do not run this by hand: the launcher starts it when
`WISEPACK_PERCEPTION_SOURCE=camera`.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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
    WorkAreaFrame, resolve_detector, resolve_model_path,
)
# THE ONLY detector-aware import in this file. One provider exists today; when a
# second is added this becomes a lookup on `resolve_detector()` and nothing else
# in the service changes.
from providers import fasterrcnn_bottle as PROVIDER                 # noqa: E402

#: Where the HARMONY checkout lives. A DOCUMENTED DEFAULT for the ARISE
#: demonstrator host, overridable, and never assumed to exist: absence is
#: reported as a health field, not raised as an import error.
DEFAULT_HARMONY_PATH = "/data/arise/harmony/ai-bottle-detector-fiware"

#: The port this service listens on. Deliberately NOT 22001: that is HARMONY's
#: own NGSI-v2 service, and both must be able to run on one host.
DEFAULT_PORT = 22101


def harmony_path() -> str:
    return os.path.abspath(os.environ.get("WISEPACK_HARMONY_PATH",
                                          DEFAULT_HARMONY_PATH))


# --------------------------------------------------------------------------- #
# HARMONY runtime configuration
# --------------------------------------------------------------------------- #
#
# HARMONY's `pipeline.py` and `camera.py` read `config/config.json` RELATIVE TO
# THE WORKING DIRECTORY, and `config/config.json` is git-ignored in HARMONY (only
# the `.tpl` is tracked). Writing into someone else's checkout to configure our
# own run would be rude and would collide with a HARMONY operator's settings.
#
# So WISEPACK generates its own HARMONY-shaped config in a WISEPACK-owned runtime
# directory and imports the HARMONY modules with that directory as the working
# directory. HARMONY is used verbatim; only where it reads its settings from
# changes, and the settings themselves come from the template plus WISEPACK
# environment overrides.

def runtime_dir() -> str:
    """A WRITABLE directory for the generated HARMONY config.

    Deliberately NOT inside the repository. `results/` holds run artefacts and
    is frequently owned by another user or mounted read-only; a machine-local
    cache is what this is, so it lives where machine-local caches live. The
    temporary-directory fallback keeps the service usable on a host with no
    writable HOME at all (a container running as an unmapped uid, say), because
    failing to start a camera preview over a config file's location would be an
    absurd way to lose a demo.
    """
    configured = os.environ.get("WISEPACK_HARMONY_RUNTIME_DIR", "").strip()
    if configured:
        return os.path.abspath(configured)
    cache = (os.environ.get("XDG_CACHE_HOME", "").strip()
             or os.path.join(os.path.expanduser("~"), ".cache"))
    candidate = os.path.join(cache, "wisepack", "harmony_runtime")
    try:
        os.makedirs(candidate, exist_ok=True)
        return candidate
    except OSError:
        import tempfile                                  # noqa: PLC0415
        return os.path.join(tempfile.gettempdir(), "wisepack_harmony_runtime")


def _template_config(harmony: str) -> Dict[str, Any]:
    """HARMONY's own config template, or its documented defaults if absent."""
    template = os.path.join(harmony, "config", "config.json.tpl")
    with contextlib.suppress(OSError, ValueError):
        with open(template, encoding="utf-8") as fh:
            return json.load(fh)
    return {"CAMERA": 0, "SET_RESOLUTION": True, "WIDTH": 1600, "HEIGHT": 1200,
            "MODEL_PATH": "models/best_model.pth",
            "CORNER_MARKERS": [11, 10, 15, 16],
            "CORNER_COORDINATES": [[0, 0], [130, 0], [130, 130], [0, 130]]}


def _camera_setting(raw: str) -> Any:
    """A camera is an INDEX or a URL. `2` and `rtsp://…` both reach cv2 as-is.

    `/dev/video2` and an RTSP URL are strings; a bare number must become an int,
    because `cv2.VideoCapture("2")` opens a *file* called "2" and then reports a
    perfectly ordinary "no frame" that looks exactly like an unplugged camera.
    """
    raw = str(raw).strip()
    try:
        return int(raw)
    except ValueError:
        return raw


def build_harmony_config(model_path: Optional[str] = None,
                         env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """HARMONY's configuration for THIS run: its template + WISEPACK overrides.

    NO CAMERA DEVICE IS HARDCODED beyond the template's documented default —
    `WISEPACK_PERCEPTION_CAMERA` (or HARMONY's own `AI_CAMERA`, honoured so an
    operator's existing habit keeps working) selects it.
    """
    env = os.environ if env is None else env
    config = _template_config(harmony_path())
    camera = (env.get("WISEPACK_PERCEPTION_CAMERA")
              or env.get("AI_CAMERA")
              or "").strip()
    if camera:
        config["CAMERA"] = _camera_setting(camera)
    if model_path:
        # Absolute, so HARMONY's relative-path resolution is bypassed and the
        # weights that get loaded are unambiguously the ones we resolved.
        config["MODEL_PATH"] = os.path.abspath(model_path)
    for key, name in (("WIDTH", "WISEPACK_PERCEPTION_WIDTH"),
                      ("HEIGHT", "WISEPACK_PERCEPTION_HEIGHT")):
        raw = str(env.get(name, "") or "").strip()
        if raw.isdigit():
            config[key] = int(raw)
    # The calibrated plane. Overridable so a LARGER printed board is a
    # configuration change, exactly as §13 requires — never a code change and
    # never a silently redesigned layout.
    markers = str(env.get("WISEPACK_HARMONY_CORNER_MARKERS", "") or "").strip()
    if markers:
        with contextlib.suppress(ValueError):
            config["CORNER_MARKERS"] = [int(m) for m in markers.split(",")]
    extent = str(env.get("WISEPACK_HARMONY_CORNER_EXTENT_MM", "") or "").strip()
    if extent:
        with contextlib.suppress(ValueError):
            side = float(extent)
            config["CORNER_COORDINATES"] = [[0, 0], [side, 0], [side, side],
                                            [0, side]]
    return config


def write_runtime_config(model_path: Optional[str] = None,
                         directory: Optional[str] = None) -> str:
    """Materialise the HARMONY config and return the directory holding it."""
    directory = directory or runtime_dir()
    os.makedirs(os.path.join(directory, "config"), exist_ok=True)
    config = build_harmony_config(model_path)
    with open(os.path.join(directory, "config", "config.json"), "w",
              encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    return directory


def work_area_from_config(config: Dict[str, Any]) -> WorkAreaFrame:
    """The WISEPACK work-area frame implied by HARMONY's calibrated plane.

    §13: WISEPACK ADOPTS HARMONY's coordinate system rather than redesigning it.
    The corner coordinates in HARMONY's config define the plane's extent in
    millimetres with the first marker at the origin; that is read back here so
    the declared work area and the actual calibration cannot drift apart.
    """
    corners = config.get("CORNER_COORDINATES") or []
    try:
        xs = [float(c[0]) for c in corners]
        ys = [float(c[1]) for c in corners]
        width, depth = int(round(max(xs))), int(round(max(ys)))
    except (TypeError, ValueError, IndexError):
        width = depth = 130
    base = WorkAreaFrame.from_env()
    return WorkAreaFrame(frame_id=base.frame_id,
                         width_mm=max(1, width), depth_mm=max(1, depth),
                         origin_x_mm=0.0, origin_y_mm=0.0)


# --------------------------------------------------------------------------- #
# Detector runtime
# --------------------------------------------------------------------------- #


class DetectorRuntime:
    """Owns the camera and the HARMONY pipeline. One instance per process.

    EVERY FAILURE IS A REPORTED STATE, NOT AN EXCEPTION OUT OF THE SERVICE (§5,
    §15). A missing camera, missing weights, an unimportable torch or a failed
    inference each leave the process running and answering /health with the
    reason. The dashboard's job is to show that reason; it cannot show a
    traceback that killed the server.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._camera = None
        self._process_frame = None
        self._pipeline = None
        self._batch_counter = 0
        # WHICH PROVIDER. Resolved once, so an unknown name fails at start-up
        # rather than at the first detection.
        self.detector = resolve_detector()
        self.model = resolve_model_path(harmony_dir=harmony_path())
        self.config = build_harmony_config(self.model.path)
        self.work_area = work_area_from_config(self.config)
        self.geometry = ProxyGeometry.from_env()
        self.last_error: str = self.model.message if not self.model.available else ""
        self.last_inference_at: str = ""
        self.last_batch: Optional[ObservationBatch] = None
        self.last_annotated: Optional[bytes] = None
        self.last_raw: Optional[bytes] = None
        self.model_loaded = False

    # -- lazy heavy imports ------------------------------------------------ #

    def _ensure_pipeline(self) -> None:
        """Import HARMONY's pipeline with its own config in reach. Once.

        The chdir is scoped to the import and restored in a `finally`: HARMONY
        resolves `config/config.json` against the working directory, and the
        alternative — editing HARMONY to accept a path — is a change to a
        repository that is not ours to change.
        """
        if self._process_frame is not None:
            return
        harmony = harmony_path()
        if not os.path.isdir(harmony):
            raise RuntimeError(
                f"HARMONY detector not found at {harmony}. Clone "
                "https://github.com/hpcbg/harmony and set WISEPACK_HARMONY_PATH "
                "to its ai-bottle-detector-fiware directory.")
        if not self.model.available:
            raise RuntimeError(self.model.message
                               or "HARMONY detector weights are not available")
        directory = write_runtime_config(self.model.path)
        previous_cwd = os.getcwd()
        if harmony not in sys.path:
            sys.path.insert(0, harmony)
        try:
            os.chdir(directory)
            import pipeline                                  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "the HARMONY detector pipeline could not be imported "
                f"({exc}). It needs torch, torchvision and opencv — install "
                "them, or run this service with HARMONY's detector venv "
                "python.") from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"the HARMONY detector could not load its weights: {exc}. "
                + (self.model.message or "")) from exc
        finally:
            os.chdir(previous_cwd)
        self._pipeline = pipeline
        self._process_frame = pipeline.process_frame
        self.model_loaded = True

    def _ensure_camera(self) -> None:
        if self._camera is not None:
            return
        harmony = harmony_path()
        if harmony not in sys.path:
            sys.path.insert(0, harmony)
        from camera import Camera                            # noqa: PLC0415
        device = self.config.get("CAMERA", 0)
        camera = Camera(device,
                        bool(self.config.get("SET_RESOLUTION", False)),
                        int(self.config.get("WIDTH", 1600)),
                        int(self.config.get("HEIGHT", 1200)))
        self._camera = camera

    # -- frames ------------------------------------------------------------ #

    def frame(self, timeout_s: float = 3.0):
        """The newest camera frame, or None. Never raises for an absent camera."""
        try:
            self._ensure_camera()
        except Exception as exc:                             # noqa: BLE001
            self.last_error = f"camera unavailable: {exc}"
            return None
        deadline = time.time() + timeout_s
        frame = self._camera.get_frame()
        while frame is None and time.time() < deadline:
            time.sleep(0.05)
            frame = self._camera.get_frame()
        if frame is None:
            self.last_error = (
                f"camera {self.config.get('CAMERA')!r} is configured but "
                "delivered no frame — is it connected and not held by another "
                "process?")
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

    # -- calibration ------------------------------------------------------- #

    def _calibration(self) -> tuple:
        """(status, revision), read from HARMONY's own marker cache.

        `pipeline._last_valid_plane_markers` is HARMONY's cache of the four
        corner markers; it is None until a frame containing all four has been
        seen, and it deliberately PERSISTS afterwards (HARMONY's README: "the
        markers positions will be cached and updated only if there are present
        again"). Reading it is how we know whether coordinates are real without
        re-running marker detection and disturbing that cache.

        The revision is a short digest of the cached marker positions, so a
        recalibration is visible as a changed revision rather than invisible.
        """
        markers = getattr(self._pipeline, "_last_valid_plane_markers", None)
        if markers is None:
            return "invalid", ""
        digest = hashlib.sha256(
            repr(getattr(markers, "tolist", lambda: markers)()).encode()
        ).hexdigest()[:12]
        return "valid", digest

    # -- the one-shot detection -------------------------------------------- #

    def detect(self) -> ObservationBatch:
        """Capture one frame, run HARMONY inference, return a WISEPACK batch.

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
            captured_at = ""

            try:
                self._ensure_pipeline()
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
                    model_id=self.model.path or "", frame_id=self.work_area.frame_id)

            # THE FRAME IS IN HAND — this is the measurement instant. Stamped
            # before inference, because inference describes THIS frame and the
            # scene may have moved on while the network ran.
            captured_at = utc_now_iso()

            try:
                result = self._process_frame(frame.copy())
            except Exception as exc:                         # noqa: BLE001
                self.last_error = f"inference failed: {exc}"
                return ObservationBatch.failed(
                    batch_id=batch_id, source=source, error=self.last_error,
                    captured_at=captured_at, requested_at=requested_at,
                    detector=PROVIDER.DETECTOR_ID,
                    model_id=self.model.path or "",
                    frame_id=self.work_area.frame_id)

            calibration_status, calibration_revision = self._calibration()
            # `processed_image` carries the calibration overlay and the measured
            # coordinates; `ai_processed_image` carries only the raw boxes. The
            # former is what §9 asks the dashboard to show.
            self.last_annotated = self.jpeg(result.get("processed_image"))
            self.last_raw = self.jpeg(frame)
            self.last_inference_at = captured_at

            batch = PROVIDER.observations_from_harmony(
                result,
                batch_id=batch_id,
                captured_at=captured_at,
                requested_at=requested_at,
                model_id=self.model.path or "",
                geometry=self.geometry,
                frame=self.work_area,
                calibration_status=calibration_status,
                calibration_revision=calibration_revision,
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
            camera_configured=self.config.get("CAMERA") is not None,
            camera_available=self.camera_available,
            model_available=self.model.available,
            model_loaded=self.model_loaded,
            model_path=self.model.path or "",
            model_origin=self.model.origin,
            calibration_status=(batch.calibration_status if batch else "unknown"),
            last_inference_at=self.last_inference_at,
            last_error=self.last_error,
            detected_objects=(batch.count if batch and batch.ok else None),
            detector=PROVIDER.DETECTOR_ID,
        ).to_dict()
        health.update({
            # THE PROVIDER, named generically. `implementation_origin` is
            # provenance for diagnostics — it is never the architectural
            # identity of the perception subsystem, and the dashboard shows the
            # display name rather than this.
            "provider": self.detector,
            "detector_display_name": PROVIDER.DISPLAY_NAME,
            "implementation_origin": PROVIDER.IMPLEMENTATION_ORIGIN,
            "provider_path": harmony_path(),
            "provider_available": os.path.isdir(harmony_path()),
            "model": self.model.to_dict(),
            "proxy_geometry": self.geometry.to_dict(),
            "work_area": self.work_area.to_dict(),
            "camera": self.config.get("CAMERA"),
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
