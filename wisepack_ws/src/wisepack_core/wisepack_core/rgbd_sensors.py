"""RGB-D sensor profiles — one declaration, two acquisition backends.

    simulated D435  -> Isaac camera built from this geometry  -> RGBDFrame
    physical  D435  -> librealsense reads the real device     -> RGBDFrame

Both produce the same generic `RGBDFrame`, and nothing downstream branches on
which one ran. What differs is PROVENANCE, and this module carries it:

    documented_nominal   a published specification. Good enough to develop
                         against; it is not a calibration of anything.
    measured             read from a specific device, with its serial.

THE DISTINCTION IS LOAD-BEARING. Two cameras of the same model have different
intrinsics — that is what factory calibration is for — so a nominal profile must
never be presented as a measurement, and a simulated sensor must never be
described as the physical one.

NO SIMULATOR AND NO CAMERA SDK IS IMPORTED HERE. This is a description of a
sensor, readable by the dashboard, the tests and both backends alike.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: Where the profiles live, relative to the repository root.
SENSOR_REGISTRY_PATH = os.path.join("config", "rgbd_sensors.yaml")

PROVENANCE_NOMINAL = "documented_nominal"
PROVENANCE_MEASURED = "measured"

#: How a frame was acquired. Named so the dashboard can state it plainly and so
#: a synthetic frame can never be reported as a measurement of the world.
BACKEND_ISAAC = "isaac_sim"
BACKEND_REALSENSE = "realsense"


class SensorProfileError(ValueError):
    """A sensor profile that cannot be used, with the reason."""


@dataclass
class StreamProfile:
    """One video stream's geometry."""

    width: int
    height: int
    fps: int = 30
    format: str = ""
    hfov_deg: Optional[float] = None
    vfov_deg: Optional[float] = None

    def focal_lengths_px(self) -> Tuple[float, float]:
        """(fx, fy) implied by the field of view and the resolution.

        A pinhole projection is fully determined by those two, which is what
        lets a simulated camera reproduce a real one's geometry without any
        vendor-specific asset.
        """
        if not self.hfov_deg or not self.vfov_deg:
            raise SensorProfileError(
                "this stream declares no field of view, so no focal length can "
                "be derived from it")
        fx = (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        fy = (self.height / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)
        return fx, fy

    def intrinsics_matrix(self) -> List[List[float]]:
        """The nominal 3x3 K. PRINCIPAL POINT AT THE CENTRE, deliberately.

        A real device's principal point is not exactly centred and the device
        reports where it actually is. Assuming the centre is correct for a
        NOMINAL profile and wrong for a measured one, which is why a measured
        profile carries the device's own matrix instead of calling this.
        """
        fx, fy = self.focal_lengths_px()
        return [[fx, 0.0, self.width / 2.0],
                [0.0, fy, self.height / 2.0],
                [0.0, 0.0, 1.0]]

    def to_dict(self) -> Dict[str, Any]:
        return {"width": self.width, "height": self.height, "fps": self.fps,
                "format": self.format, "hfov_deg": self.hfov_deg,
                "vfov_deg": self.vfov_deg}


@dataclass
class SensorProfile:
    """One RGB-D sensor, as declared rather than as measured."""

    sensor_id: str
    model: str
    vendor: str = ""
    provenance: str = PROVENANCE_NOMINAL
    provenance_note: str = ""
    colour: Optional[StreamProfile] = None
    depth: Optional[StreamProfile] = None
    depth_scale_mm_per_unit: float = 1.0
    baseline_mm: Optional[float] = None
    min_range_m: Optional[float] = None
    max_range_m: Optional[float] = None
    limitations: Tuple[str, ...] = ()
    #: Set only for a measured profile — which device this came from.
    serial_number: str = ""

    @property
    def is_measured(self) -> bool:
        return self.provenance == PROVENANCE_MEASURED

    def describe_backend(self, backend: str) -> Dict[str, Any]:
        """What the dashboard should say about a frame from this backend.

        THE SIMULATED CAMERA IS NEVER CALLED A REALSENSE. It is a simulated
        RGB-D camera configured to a D435's published geometry, which is a
        different claim and is worded as one.
        """
        if backend == BACKEND_ISAAC:
            return {
                "camera_backend": "Isaac Sim",
                "camera_model": f"{self.model}-compatible simulated RGB-D",
                "provenance": "synthetic",
                "note": ("Rendered by Isaac Sim using this sensor's published "
                         "geometry. It is NOT the physical device and carries "
                         "none of its noise, and its intrinsics are nominal."),
            }
        if backend == BACKEND_REALSENSE:
            return {
                "camera_backend": "RealSense",
                "camera_model": self.model,
                "provenance": "measured",
                "note": ("Acquired from the physical device; intrinsics and "
                         "depth scale are read from it."),
            }
        raise SensorProfileError(f"unknown acquisition backend {backend!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id, "vendor": self.vendor,
            "model": self.model, "provenance": self.provenance,
            "provenance_note": self.provenance_note,
            "serial_number": self.serial_number,
            "colour": self.colour.to_dict() if self.colour else None,
            "depth": self.depth.to_dict() if self.depth else None,
            "depth_scale_mm_per_unit": self.depth_scale_mm_per_unit,
            "baseline_mm": self.baseline_mm,
            "min_range_m": self.min_range_m, "max_range_m": self.max_range_m,
            "limitations": list(self.limitations),
        }

    @staticmethod
    def from_dict(document: Dict[str, Any]) -> "SensorProfile":
        def stream(key: str) -> Optional[StreamProfile]:
            raw = document.get(key)
            if not raw:
                return None
            return StreamProfile(
                width=int(raw["width"]), height=int(raw["height"]),
                fps=int(raw.get("fps", 30)), format=str(raw.get("format", "")),
                hfov_deg=raw.get("hfov_deg"), vfov_deg=raw.get("vfov_deg"))

        depth_raw = document.get("depth") or {}
        return SensorProfile(
            sensor_id=str(document["sensor_id"]),
            model=str(document.get("model", document["sensor_id"])),
            vendor=str(document.get("vendor", "")),
            provenance=str(document.get("provenance", PROVENANCE_NOMINAL)),
            provenance_note=str(document.get("provenance_note", "")),
            serial_number=str(document.get("serial_number", "")),
            colour=stream("colour"), depth=stream("depth"),
            depth_scale_mm_per_unit=float(
                depth_raw.get("depth_scale_mm_per_unit", 1.0)),
            baseline_mm=depth_raw.get("baseline_mm"),
            min_range_m=depth_raw.get("min_range_m"),
            max_range_m=depth_raw.get("max_range_m"),
            limitations=tuple(document.get("limitations") or ()))


def load_sensor_profiles(path: str = "", repo_root: str = ""
                         ) -> Dict[str, SensorProfile]:
    """Every declared sensor. A BROKEN file is reported, never silently empty."""
    root = repo_root or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    resolved = path or os.path.join(root, SENSOR_REGISTRY_PATH)
    if not os.path.isfile(resolved):
        return {}
    import yaml                                              # noqa: PLC0415
    with open(resolved, encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    profiles: Dict[str, SensorProfile] = {}
    for entry in document.get("sensors") or []:
        profile = SensorProfile.from_dict(entry)
        if profile.sensor_id in profiles:
            raise SensorProfileError(
                f"duplicate sensor profile {profile.sensor_id!r}")
        profiles[profile.sensor_id] = profile
    return profiles


def sensor_profile(sensor_id: str = "d435", **kw: Any) -> SensorProfile:
    profiles = load_sensor_profiles(**kw)
    if sensor_id not in profiles:
        raise SensorProfileError(
            f"unknown sensor {sensor_id!r}; declared: "
            + (", ".join(sorted(profiles)) or "(none)"))
    return profiles[sensor_id]


__all__ = ["SensorProfile", "StreamProfile", "SensorProfileError",
           "load_sensor_profiles", "sensor_profile", "SENSOR_REGISTRY_PATH",
           "PROVENANCE_NOMINAL", "PROVENANCE_MEASURED", "BACKEND_ISAAC",
           "BACKEND_REALSENSE"]
