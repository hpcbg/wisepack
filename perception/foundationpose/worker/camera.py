"""RGB-D acquisition, owned by the FoundationPose worker.

ONE COHERENT ACQUISITION, AND THAT IS THE WHOLE REASON THIS IS HERE
-------------------------------------------------------------------
FoundationPose needs three things that must agree with each other: a colour
image, a depth image ALIGNED to that colour, and the intrinsics both share.
Acquiring them on the host and inferring in the container would mean two SDK
versions, two alignment implementations and two places for a depth scale to be
assumed — and whichever pair disagreed would produce a plausible, wrong pose
rather than an error. So the camera lives in the same process as the estimator.

WHAT THIS IS NOT
----------------
* NOT the planar camera. That is an ordinary UVC webcam owned by the perception
  service on the host, through OpenCV, and nothing here touches it.
* NOT `cv2.VideoCapture`. A RealSense presents several `/dev/video*` nodes
  (depth, colour, infrared, metadata); opening one by number gives raw frames
  with no intrinsics, no depth scale and no alignment — every calibrated
  quantity FoundationPose depends on. Streams are selected by ROLE through
  librealsense, which is also why a device index can never collide with the
  planar camera's.

NOTHING IS ASSUMED ABOUT THE DEVICE. The model, serial, stream profiles,
intrinsics and depth scale are all read from the hardware. Alignment is
performed by the SDK and then VERIFIED rather than trusted.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

#: Frames discarded before the first capture, while auto-exposure settles. The
#: opening frames of a D4xx are dark, and a black frame is not a measurement of
#: the scene.
WARMUP_FRAMES = 30

#: Where captured datasets are written inside the container. Mounted from a
#: WISEPACK-owned host directory so a capture survives the container.
CAPTURES_DIR = os.environ.get("WISEPACK_FP_CAPTURES_DIR", "/captures")


class CameraUnavailable(RuntimeError):
    """No usable RGB-D device, with the reason an operator can act on."""


def _sdk():
    try:
        import pyrealsense2 as rs                            # noqa: PLC0415
    except ImportError as exc:                               # pragma: no cover
        raise CameraUnavailable(
            f"pyrealsense2 is not available in this image ({exc}). Rebuild the "
            "FoundationPose worker: RGB-D acquisition is the worker's job.")
    return rs


def available() -> Tuple[bool, str]:
    """(an RGB-D camera is usable, reason if not). NEVER raises.

    Called by the capability probe on every /health, so it must be cheap and it
    must not throw: a missing camera is a state to report, and the worker keeps
    answering with the reason.
    """
    try:
        found = list(_sdk().context().query_devices())
    except CameraUnavailable as exc:
        return False, str(exc)
    except Exception as exc:                                 # noqa: BLE001
        return False, f"the RealSense SDK failed to enumerate devices: {exc}"
    if not found:
        return False, (
            "no RealSense device is connected to the worker. librealsense "
            "enumerates RealSense hardware only, so this is not about "
            "/dev/video* numbering — the planar webcam is a separate UVC device "
            "and is never used for depth. Check that the camera is on the USB "
            "bus and that the container was started with USB access.")
    return True, ""


def require_device(serial: str = "") -> Any:
    """The device to use, or a refusal naming what was actually seen."""
    rs = _sdk()
    usable, reason = available()
    if not usable:
        raise CameraUnavailable(reason)
    found = list(rs.context().query_devices())
    if serial:
        for device in found:
            if device.get_info(rs.camera_info.serial_number) == serial:
                return device
        raise CameraUnavailable(
            f"no RealSense with serial {serial!r}; connected: "
            + ", ".join(d.get_info(rs.camera_info.serial_number) for d in found))
    if len(found) > 1:
        # AMBIGUITY IS REFUSED. With two cameras attached, "the first one" is
        # whichever enumerated this boot, and a dataset captured from the wrong
        # one is indistinguishable from a correct one.
        raise CameraUnavailable(
            "more than one RealSense is connected; name one with `serial`: "
            + ", ".join(f"{d.get_info(rs.camera_info.name)} "
                        f"({d.get_info(rs.camera_info.serial_number)})"
                        for d in found))
    return found[0]


def _info(device: Any, field: str) -> str:
    rs = _sdk()
    try:
        return device.get_info(getattr(rs.camera_info, field))
    except Exception:                                        # noqa: BLE001
        return ""


def describe(serial: str = "") -> Dict[str, Any]:
    """Model, serial, firmware, USB speed, every stream profile, depth scale."""
    device = require_device(serial)
    document: Dict[str, Any] = {
        "name": _info(device, "name"),
        "serial_number": _info(device, "serial_number"),
        "firmware_version": _info(device, "firmware_version"),
        "product_line": _info(device, "product_line"),
        # USB SPEED IS DIAGNOSTIC, NOT COSMETIC: a D4xx on USB 2 silently loses
        # its higher resolutions and frame rates, which reads as a broken camera
        # rather than as a cable in the wrong socket.
        "usb_type_descriptor": _info(device, "usb_type_descriptor"),
        "sensors": [],
        "depth_scale_m_per_unit": None,
        "depth_scale_mm_per_unit": None,
    }
    for sensor in device.query_sensors():
        entry: Dict[str, Any] = {"name": sensor.get_info(_sdk().camera_info.name),
                                 "streams": []}
        if sensor.is_depth_sensor():
            scale = float(sensor.as_depth_sensor().get_depth_scale())
            document["depth_scale_m_per_unit"] = scale
            # BOTH WISEPACK AND FOUNDATIONPOSE WANT MILLIMETRES PER UNIT. The
            # conversion happens once, here, from the device's own value —
            # never from an assumed constant.
            document["depth_scale_mm_per_unit"] = scale * 1000.0
        seen = set()
        for profile in sensor.get_stream_profiles():
            try:
                video = profile.as_video_stream_profile()
                key = (profile.stream_name(), video.width(), video.height(),
                       str(profile.format()), profile.fps())
            except Exception:                                # noqa: BLE001
                continue
            if key in seen:
                continue
            seen.add(key)
            entry["streams"].append({
                "stream": profile.stream_name(),
                "width": video.width(), "height": video.height(),
                "format": str(profile.format()), "fps": profile.fps(),
            })
        entry["streams"].sort(key=lambda s: (s["stream"], -s["width"], -s["fps"]))
        document["sensors"].append(entry)
    document["synchronised_profiles"] = compatible_profiles(document)
    return document


def compatible_profiles(described: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Colour/depth profiles that can run SYNCHRONISED at one size and rate.

    Chosen from what the device reports rather than hard-coded: a D415, a D435
    and a D455 do not offer the same sets, and asking for a combination the
    device cannot deliver fails at `pipeline.start()` with a message about
    neither stream.
    """
    colour: Dict[Tuple[int, int, int], bool] = {}
    depth: Dict[Tuple[int, int, int], bool] = {}
    for sensor in described.get("sensors", []):
        for stream in sensor.get("streams", []):
            key = (stream["width"], stream["height"], stream["fps"])
            name = str(stream["stream"]).lower()
            if name == "color" and "bgr8" in str(stream["format"]).lower():
                colour[key] = True
            elif name == "depth" and "z16" in str(stream["format"]).lower():
                depth[key] = True
    shared = sorted(set(colour) & set(depth),
                    key=lambda k: (-(k[0] * k[1]), -k[2]))
    return [{"width": w, "height": h, "fps": f} for w, h, f in shared]


def _intrinsics_of(profile: Any) -> Dict[str, Any]:
    """The device's own intrinsics for one video stream profile."""
    video = profile.as_video_stream_profile()
    i = video.get_intrinsics()
    return {
        "width": i.width, "height": i.height,
        "fx": float(i.fx), "fy": float(i.fy),
        "cx": float(i.ppx), "cy": float(i.ppy),
        "model": str(i.model), "coeffs": [float(c) for c in i.coeffs],
        "matrix": [[float(i.fx), 0.0, float(i.ppx)],
                   [0.0, float(i.fy), float(i.ppy)],
                   [0.0, 0.0, 1.0]],
    }


class RGBDStream:
    """A started, synchronised colour+depth pipeline. Closed on exit.

    ALIGNMENT IS APPLIED HERE AND VERIFIED HERE. `rs.align(rs.stream.color)`
    reprojects depth into the colour camera's geometry; the result is checked
    against the colour intrinsics before any frame is used, because a dataset
    that claims alignment it did not get is a silent reprojection error in every
    pose computed from it.
    """

    def __init__(self, serial: str = "", width: int = 0, height: int = 0,
                 fps: int = 0, align: bool = True) -> None:
        rs = _sdk()
        device = require_device(serial)
        self.serial = device.get_info(rs.camera_info.serial_number)
        self.device_name = device.get_info(rs.camera_info.name)
        self.firmware = device.get_info(rs.camera_info.firmware_version)

        if not (width and height and fps):
            # NEGOTIATED FROM THE DEVICE rather than defaulted to a resolution
            # this particular model may not offer.
            options = compatible_profiles(describe(self.serial))
            if not options:
                raise CameraUnavailable(
                    "the device offers no colour(BGR8)+depth(Z16) combination "
                    "at a shared resolution and frame rate")
            chosen = options[0]
            width, height, fps = chosen["width"], chosen["height"], chosen["fps"]
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.align = bool(align)

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height,
                             rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height,
                             rs.format.z16, self.fps)
        try:
            self._profile = self._pipeline.start(config)
        except Exception as exc:                             # noqa: BLE001
            raise CameraUnavailable(
                f"could not start {self.width}x{self.height}@{self.fps} on "
                f"{self.device_name}: {exc}") from exc
        self._aligner = rs.align(rs.stream.color) if align else None
        self.depth_scale_m = float(
            self._profile.get_device().first_depth_sensor().get_depth_scale())
        self.colour_intrinsics: Optional[Dict[str, Any]] = None
        self.depth_intrinsics: Optional[Dict[str, Any]] = None
        self.alignment_verified = False

    def __enter__(self) -> "RGBDStream":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:                                    # noqa: BLE001
            pass

    def warmup(self, frames: int = WARMUP_FRAMES) -> None:
        for _ in range(max(0, frames)):
            self._pipeline.wait_for_frames()

    def frame(self):
        """One synchronised (colour, depth) pair as numpy arrays.

        The two come from ONE `wait_for_frames()` bundle, so they describe the
        same instant. Fetching them separately would pair a colour image with a
        depth image from a different moment, which on a moving scene is a pose
        error with no symptom.
        """
        import numpy as np                                   # noqa: PLC0415
        bundle = self._pipeline.wait_for_frames()
        if self._aligner is not None:
            bundle = self._aligner.process(bundle)
        colour = bundle.get_color_frame()
        depth = bundle.get_depth_frame()
        if not colour or not depth:
            raise CameraUnavailable("the device produced an incomplete frame")
        if self.colour_intrinsics is None:
            self.colour_intrinsics = _intrinsics_of(colour.get_profile())
            self.depth_intrinsics = _intrinsics_of(depth.get_profile())
            self.alignment_verified = bool(
                self.align
                and (colour.get_width(), colour.get_height())
                == (depth.get_width(), depth.get_height())
                and abs(self.depth_intrinsics["fx"]
                        - self.colour_intrinsics["fx"]) < 1e-3
                and abs(self.depth_intrinsics["cx"]
                        - self.colour_intrinsics["cx"]) < 1e-3)
        return (np.asanyarray(colour.get_data()),
                np.asanyarray(depth.get_data()),
                {"device_timestamp_ms": float(bundle.get_timestamp()),
                 "frame_number": int(bundle.get_frame_number())})

    def state(self) -> Dict[str, Any]:
        return {
            "device": {"name": self.device_name, "serial_number": self.serial,
                       "firmware_version": self.firmware},
            "width": self.width, "height": self.height, "fps": self.fps,
            "aligned": self.align,
            "alignment_method": ("librealsense rs.align(rs.stream.color)"
                                 if self.align else
                                 "none — depth is NOT aligned to colour"),
            "alignment_verified": self.alignment_verified,
            "colour_intrinsics": self.colour_intrinsics,
            "depth_intrinsics": self.depth_intrinsics,
            "depth_scale_m_per_unit": self.depth_scale_m,
            "depth_scale_mm_per_unit": self.depth_scale_m * 1000.0,
        }


def capture_dataset(model_id: str, frames: int = 30, name: str = "",
                    serial: str = "", width: int = 0, height: int = 0,
                    fps: int = 0, align: bool = True) -> Dict[str, Any]:
    """Capture a controlled dataset for ONE known CAD part.

    Written in FoundationPose's own demo layout — `cam_K.txt`, `rgb/`, `depth/`,
    `masks/` — reused verbatim rather than reinvented, so the same reader loads
    it and no conversion step can introduce the unit or alignment errors this
    whole path exists to avoid.

    `model_id` IS AN INPUT AND IS NEVER INFERRED. Which CAD part is on the table
    is known because an operator put it there and said so. Inferring it from
    appearance is precisely what could not be done for the older Tubes-Capture
    material, and is why that data remains unusable for pose estimation.

    THE MASK IS NOT PRODUCED HERE. `register()` needs one, and producing it means
    segmenting the scene — a perception decision, not a capture one. Guessing it
    from a depth threshold would put an invented segmentation into the input of a
    pose measurement. The dataset records `masks_present: false` and reports
    itself incomplete until one exists.
    """
    import cv2                                               # noqa: PLC0415

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    root = os.path.join(CAPTURES_DIR, name or f"{model_id}-{stamp}")
    for directory in ("rgb", "depth", "masks"):
        os.makedirs(os.path.join(root, directory), exist_ok=True)

    captured: List[Dict[str, Any]] = []
    with RGBDStream(serial=serial, width=width, height=height, fps=fps,
                    align=align) as stream:
        stream.warmup()
        for index in range(max(1, frames)):
            colour, depth, meta = stream.frame()
            filename = f"{index:06d}.png"
            cv2.imwrite(os.path.join(root, "rgb", filename), colour)
            cv2.imwrite(os.path.join(root, "depth", filename), depth)
            captured.append({"file": filename, **meta})
        state = stream.state()

    with open(os.path.join(root, "cam_K.txt"), "w", encoding="utf-8") as handle:
        for row in state["colour_intrinsics"]["matrix"]:
            handle.write(" ".join(f"{v:.18e}" for v in row) + "\n")

    metadata = {
        "model_id": model_id,
        "root": root,
        "frames": captured,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "depth_encoding": "uint16 PNG; multiply by depth_scale_mm_per_unit for mm",
        "invalid_depth_value": 0,
        # THE TWO THINGS CAPTURE CANNOT ESTABLISH, named rather than left as an
        # absence for a later reader to discover.
        "masks_present": False,
        "masks_note": ("register() needs a mask of the object. Segmentation is "
                       "a perception decision, not a capture one, and is not "
                       "guessed here."),
        "camera_to_workarea_extrinsic": None,
        "extrinsic_note": ("No validated SE(3) camera-to-work-area transform. "
                           "Poses from this dataset stay in the camera optical "
                           "frame until one is measured."),
        **state,
    }
    with open(os.path.join(root, "metadata.json"), "w", encoding="utf-8") as handle:
        import json                                          # noqa: PLC0415
        json.dump(metadata, handle, indent=2)
    return metadata


__all__ = ["CameraUnavailable", "available", "require_device", "describe",
           "compatible_profiles", "RGBDStream", "capture_dataset",
           "CAPTURES_DIR", "WARMUP_FRAMES"]
