"""RGB-D frames, camera intrinsics, and the object-model registry.

WHAT THIS IS FOR
----------------
A model-based pose estimator needs four things that a 2-D detector does not:
colour, depth, the intrinsics that relate them to metres, and the CAD geometry
of the thing being located. Each has a failure mode that produces a *plausible
wrong answer* rather than an error, so each is represented explicitly and
validated here rather than passed around as a bare array:

    depth in the wrong unit      -> an object a thousand times too far away
    depth not aligned to colour  -> a mask that selects the wrong depth pixels
    intrinsics for another size  -> a pose off by a scale factor
    a mesh in metres, not mm     -> a "fit" against a model 1000x too small

None of those raise. All of them are caught by the checks in this module.

NO IMAGE DATA LIVES HERE. `RGBDFrame` carries metadata and *references*, not
pixels: the WISEPACK side describes and validates a frame, and the worker that
owns the camera keeps the arrays. That is what lets `wisepack_core` stay free of
numpy, OpenCV and any camera SDK, exactly as it is free of torch today.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .pose import Orientation, PoseError, RigidTransform, Symmetry, SymmetryType

# --------------------------------------------------------------------------- #
# Intrinsics
# --------------------------------------------------------------------------- #


class RGBDError(ValueError):
    """Raised when a frame, intrinsic or object model cannot be trusted."""


@dataclass(frozen=True)
class CameraIntrinsics:
    """A pinhole camera's `K`, together with the image size it belongs to.

    THE SIZE IS PART OF THE INTRINSIC, and omitting it is the classic error.
    `fx` is in pixels, so intrinsics measured at 1920x1080 are wrong by exactly
    a factor of two at 960x540 — and produce poses that are wrong by a factor of
    two while looking entirely reasonable. Carrying the size lets a consumer
    check, and lets `scaled()` do the conversion once and correctly.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    #: Where these numbers came from: `device_sdk`, `charuco_calibration`,
    #: `simulation`, `file:cam_K.txt`. NEVER left empty for a real camera —
    #: an intrinsic with no provenance cannot be audited.
    source: str = ""

    def __post_init__(self) -> None:
        for name in ("fx", "fy", "cx", "cy"):
            value = float(getattr(self, name))
            if value != value or abs(value) == float("inf"):
                raise RGBDError(f"intrinsic {name} is not finite")
            object.__setattr__(self, name, value)
        if self.fx <= 0 or self.fy <= 0:
            raise RGBDError(
                f"focal lengths must be positive, got fx={self.fx}, fy={self.fy}")
        for name in ("width", "height"):
            value = int(getattr(self, name))
            if value <= 0:
                raise RGBDError(f"image {name} must be positive, got {value}")
            object.__setattr__(self, name, value)
        # The principal point must be INSIDE the image. Outside it means the
        # intrinsics belong to a different resolution or a different camera —
        # the single most common way a borrowed calibration goes unnoticed.
        if not (0 <= self.cx <= self.width and 0 <= self.cy <= self.height):
            raise RGBDError(
                f"principal point ({self.cx}, {self.cy}) lies outside the "
                f"{self.width}x{self.height} image — these intrinsics belong to "
                "a different resolution or a different camera")

    @property
    def matrix(self) -> List[List[float]]:
        """`K`, row major."""
        return [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0]]

    @property
    def horizontal_fov_deg(self) -> float:
        """Sanity handle for a human: a plausible lens is roughly 40-120 deg."""
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.fx)))

    def scaled(self, width: int, height: int) -> "CameraIntrinsics":
        """These intrinsics, for a resized image. The ONLY correct way to do it."""
        if width <= 0 or height <= 0:
            raise RGBDError("target size must be positive")
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(
            fx=self.fx * sx, fy=self.fy * sy,
            cx=self.cx * sx, cy=self.cy * sy,
            width=width, height=height,
            source=f"{self.source} scaled {self.width}x{self.height}->{width}x{height}")

    def deproject(self, u: float, v: float, depth_mm: float
                  ) -> Tuple[float, float, float]:
        """Pixel + depth -> a point in the camera optical frame, in millimetres."""
        return ((u - self.cx) * depth_mm / self.fx,
                (v - self.cy) * depth_mm / self.fy,
                float(depth_mm))

    def to_dict(self) -> Dict[str, Any]:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "width": self.width, "height": self.height,
                "source": self.source,
                "horizontal_fov_deg": round(self.horizontal_fov_deg, 2)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CameraIntrinsics":
        return CameraIntrinsics(
            fx=float(d["fx"]), fy=float(d["fy"]),
            cx=float(d["cx"]), cy=float(d["cy"]),
            width=int(d["width"]), height=int(d["height"]),
            source=str(d.get("source", "")))

    @staticmethod
    def from_matrix(rows, width: int, height: int, source: str = ""
                    ) -> "CameraIntrinsics":
        """From a 3x3 `K` — the form `cam_K.txt` and every SDK reports."""
        return CameraIntrinsics(fx=float(rows[0][0]), fy=float(rows[1][1]),
                                cx=float(rows[0][2]), cy=float(rows[1][2]),
                                width=width, height=height, source=source)


# --------------------------------------------------------------------------- #
# Depth and frames
# --------------------------------------------------------------------------- #


#: Whether the depth image is registered to the colour image.
#:
#: `aligned` — every depth pixel corresponds to the colour pixel at the same
#:             index. This is what a mask drawn on the colour image requires,
#:             and the estimator assumes it without being able to check.
#: `unaligned` — depth is in its own frame. A colour mask indexes the WRONG
#:             depth pixels, quietly, and the resulting pose is wrong in a way
#:             that looks like sensor noise.
#: `unknown` — nobody said. Treated as unusable rather than assumed aligned.
ALIGNMENT_ALIGNED = "aligned"
ALIGNMENT_UNALIGNED = "unaligned"
ALIGNMENT_UNKNOWN = "unknown"

#: Depth units, as a multiplier to MILLIMETRES — WISEPACK's unit everywhere.
#: A 16-bit depth PNG in millimetres has scale 1.0; a float32 depth in metres
#: has 1000.0.
DEPTH_SCALE_MM_PER_UNIT_DEFAULT = 1.0


@dataclass
class RGBDFrame:
    """One coherent colour + depth observation, described but not carried.

    COHERENT MEANS ONE VIEW AT ONE INSTANT. Two images from different moments,
    or from cameras that have not been registered to each other, are not an
    RGB-D frame however convenient it is to treat them as one — and a pose
    estimated from such a pair is wrong in a way no downstream check can see.
    """

    intrinsics: Optional[CameraIntrinsics] = None
    width: int = 0
    height: int = 0
    #: How many MILLIMETRES one raw depth unit represents.
    depth_scale_mm: float = DEPTH_SCALE_MM_PER_UNIT_DEFAULT
    #: The raw value meaning "no measurement here". 0 for every sensor WISEPACK
    #: has met; carried rather than assumed.
    depth_invalid_value: float = 0.0
    alignment: str = ALIGNMENT_UNKNOWN
    rgb_available: bool = False
    depth_available: bool = False
    rgb_frame_id: str = ""
    depth_frame_id: str = ""
    captured_at: str = ""
    camera_model: str = ""
    #: Fraction of depth pixels with no measurement, when the producer counted
    #: them. `None` means not measured — different from zero.
    depth_invalid_fraction: Optional[float] = None
    last_error: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.intrinsics, dict):
            self.intrinsics = CameraIntrinsics.from_dict(self.intrinsics)
        self.depth_scale_mm = float(self.depth_scale_mm)
        if self.depth_scale_mm <= 0:
            raise RGBDError(
                f"depth_scale_mm must be positive, got {self.depth_scale_mm} — "
                "a non-positive scale would invert or annihilate every depth")
        if self.alignment not in (ALIGNMENT_ALIGNED, ALIGNMENT_UNALIGNED,
                                  ALIGNMENT_UNKNOWN):
            raise RGBDError(f"unknown depth alignment {self.alignment!r}")
        if self.intrinsics is not None:
            if not self.width:
                self.width = self.intrinsics.width
            if not self.height:
                self.height = self.intrinsics.height

    # -- usability --------------------------------------------------------- #

    @property
    def intrinsics_available(self) -> bool:
        return self.intrinsics is not None

    @property
    def depth_aligned(self) -> bool:
        return self.alignment == ALIGNMENT_ALIGNED

    def usability(self) -> Tuple[bool, str]:
        """(usable for model-based pose estimation, reason if not).

        EVERY CONDITION IS CHECKED BEFORE INFERENCE, not after, because each one
        produces a confident wrong pose rather than an error.
        """
        if not self.rgb_available:
            return False, "no colour image"
        if not self.depth_available:
            return False, "no depth image"
        if not self.intrinsics_available:
            return False, ("no camera intrinsics — a pose cannot be expressed "
                           "in metres without them")
        if self.alignment != ALIGNMENT_ALIGNED:
            return False, (
                f"depth alignment is {self.alignment!r}: a mask drawn on the "
                "colour image would select the wrong depth pixels")
        if (self.intrinsics.width != self.width
                or self.intrinsics.height != self.height):
            return False, (
                f"intrinsics are for {self.intrinsics.width}x"
                f"{self.intrinsics.height} but the frame is {self.width}x"
                f"{self.height}")
        return True, ""

    @property
    def usable(self) -> bool:
        return self.usability()[0]

    def to_dict(self) -> Dict[str, Any]:
        usable, reason = self.usability()
        return {
            "width": self.width, "height": self.height,
            "rgb_available": self.rgb_available,
            "depth_available": self.depth_available,
            "depth_aligned": self.depth_aligned,
            "alignment": self.alignment,
            "intrinsics_available": self.intrinsics_available,
            "intrinsics": (self.intrinsics.to_dict()
                           if self.intrinsics else None),
            "depth_scale_mm": self.depth_scale_mm,
            "depth_invalid_value": self.depth_invalid_value,
            "depth_invalid_fraction": self.depth_invalid_fraction,
            "rgb_frame_id": self.rgb_frame_id,
            "depth_frame_id": self.depth_frame_id,
            "captured_at": self.captured_at,
            "camera_model": self.camera_model,
            "usable": usable,
            "unusable_reason": reason,
            "last_error": self.last_error,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RGBDFrame":
        return RGBDFrame(
            intrinsics=(CameraIntrinsics.from_dict(d["intrinsics"])
                        if d.get("intrinsics") else None),
            width=int(d.get("width", 0)), height=int(d.get("height", 0)),
            depth_scale_mm=float(d.get("depth_scale_mm",
                                       DEPTH_SCALE_MM_PER_UNIT_DEFAULT)),
            depth_invalid_value=float(d.get("depth_invalid_value", 0.0)),
            alignment=str(d.get("alignment", ALIGNMENT_UNKNOWN)),
            rgb_available=bool(d.get("rgb_available", False)),
            depth_available=bool(d.get("depth_available", False)),
            rgb_frame_id=str(d.get("rgb_frame_id", "")),
            depth_frame_id=str(d.get("depth_frame_id", "")),
            captured_at=str(d.get("captured_at", "")),
            camera_model=str(d.get("camera_model", "")),
            depth_invalid_fraction=d.get("depth_invalid_fraction"),
            last_error=str(d.get("last_error", "")))


# --------------------------------------------------------------------------- #
# Object model registry
# --------------------------------------------------------------------------- #

#: Mesh units, as a multiplier to MILLIMETRES. CAD is authored in millimetres
#: and most estimators work in metres, so this is the single most likely
#: silent-failure point in the whole pipeline — a factor of 1000 in a number
#: nobody prints.
MESH_UNITS = {"mm": 1.0, "millimetre": 1.0, "millimeter": 1.0,
              "m": 1000.0, "metre": 1000.0, "meter": 1000.0,
              "cm": 10.0, "centimetre": 10.0, "centimeter": 10.0}


@dataclass
class ObjectModel:
    """The CAD geometry and physical facts of one object WISEPACK can locate.

    DECLARED, NOT DISCOVERED. Mesh units cannot be read from an STL or an OBJ —
    both formats are unitless — so the unit is configuration, and a mesh whose
    declared unit disagrees with its measured size is rejected rather than
    scaled by a guess.
    """

    model_id: str
    #: The WISEPACK object type this model instantiates. What planning sees.
    object_type: str = "cylindrical_proxy"
    mesh_path: str = ""
    mesh_units: str = "mm"
    #: Extra scale beyond the unit conversion. 1.0 normally; anything else needs
    #: a reason, because it is indistinguishable from a units mistake.
    scale: float = 1.0
    symmetry: Symmetry = field(default_factory=Symmetry)
    #: The packing dimensions, in millimetres. DECLARED here rather than derived
    #: from the mesh: the planner's arithmetic is integer millimetres and must
    #: not move when someone re-exports a mesh.
    diameter_mm: Optional[int] = None
    length_mm: Optional[int] = None
    #: The mesh's own bounding-box extents in millimetres, MEASURED from the
    #: mesh (see scripts/measure_mesh_symmetry.py) and recorded so a mesh that
    #: is silently replaced by a differently-scaled export is detectable. Not
    #: used for planning — `diameter_mm`/`length_mm` are, and they are declared.
    extents_mm: Tuple[float, ...] = ()
    #: Which perception methods can use this model. FoundationPose needs a mesh;
    #: the planar detector needs none, so a model may legitimately list one.
    methods: Tuple[str, ...] = ()
    description: str = ""
    #: Where the mesh came from, for attribution and reproducibility.
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.model_id:
            raise RGBDError("an object model needs a model_id")
        if isinstance(self.symmetry, dict):
            self.symmetry = Symmetry.from_dict(self.symmetry)
        unit = str(self.mesh_units).lower()
        if unit not in MESH_UNITS:
            raise RGBDError(
                f"{self.model_id}: unknown mesh unit {self.mesh_units!r}; "
                f"known: {', '.join(sorted(set(MESH_UNITS)))}")
        self.mesh_units = unit
        self.scale = float(self.scale)
        if self.scale <= 0:
            raise RGBDError(f"{self.model_id}: scale must be positive")
        self.methods = tuple(str(m) for m in (self.methods or ()))
        self.extents_mm = tuple(float(e) for e in (self.extents_mm or ()))

    @property
    def mesh_scale_to_mm(self) -> float:
        """One mesh unit, in millimetres, including any extra `scale`."""
        return MESH_UNITS[self.mesh_units] * self.scale

    def mesh_exists(self, root: str = "") -> bool:
        return bool(self.mesh_path) and os.path.isfile(self.resolved_path(root))

    def resolved_path(self, root: str = "") -> str:
        if not self.mesh_path:
            return ""
        if os.path.isabs(self.mesh_path):
            return self.mesh_path
        return os.path.join(root, self.mesh_path) if root else self.mesh_path

    def usability(self, root: str = "") -> Tuple[bool, str]:
        """(usable by a model-based method, reason if not)."""
        if not self.mesh_path:
            return False, f"{self.model_id} declares no mesh"
        if not self.mesh_exists(root):
            return False, (f"{self.model_id}: mesh not found at "
                           f"{self.resolved_path(root)}")
        return True, ""

    def supports(self, method: str) -> bool:
        """An empty `methods` means "any" — a model is not method-specific by
        default, and listing every method on every model would rot."""
        return not self.methods or method in self.methods

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "object_type": self.object_type,
            "mesh_path": self.mesh_path,
            "mesh_units": self.mesh_units,
            "scale": self.scale,
            "mesh_scale_to_mm": self.mesh_scale_to_mm,
            "symmetry": self.symmetry.to_dict(),
            "diameter_mm": self.diameter_mm,
            "length_mm": self.length_mm,
            "extents_mm": list(self.extents_mm),
            "methods": list(self.methods),
            "description": self.description,
            "provenance": self.provenance,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ObjectModel":
        return ObjectModel(
            model_id=str(d["model_id"]),
            object_type=str(d.get("object_type", "cylindrical_proxy")),
            mesh_path=str(d.get("mesh_path", "")),
            mesh_units=str(d.get("mesh_units", "mm")),
            scale=float(d.get("scale", 1.0)),
            symmetry=Symmetry.from_dict(d.get("symmetry")),
            diameter_mm=d.get("diameter_mm"),
            length_mm=d.get("length_mm"),
            extents_mm=tuple(d.get("extents_mm") or ()),
            methods=tuple(d.get("methods") or ()),
            description=str(d.get("description", "")),
            provenance=str(d.get("provenance", "")))


@dataclass
class ObjectModelRegistry:
    """Every object WISEPACK knows the geometry of, loaded from configuration.

    ONE PLACE, like the robot registry: a mesh path hard-coded in a provider is
    a second registry that agrees with this one only until someone edits it.
    """

    models: Dict[str, ObjectModel] = field(default_factory=dict)
    root: str = ""
    error: str = ""

    def get(self, model_id: str) -> Optional[ObjectModel]:
        return self.models.get(str(model_id))

    def require(self, model_id: str) -> ObjectModel:
        model = self.get(model_id)
        if model is None:
            raise RGBDError(
                f"unknown object model {model_id!r}; configured: "
                + (", ".join(sorted(self.models)) or "none"))
        return model

    def for_method(self, method: str) -> List[ObjectModel]:
        return [m for m in self.models.values() if m.supports(method)]

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "models": [
                {**m.to_dict(),
                 "mesh_available": m.mesh_exists(self.root),
                 "unusable_reason": m.usability(self.root)[1]}
                for m in sorted(self.models.values(), key=lambda m: m.model_id)
            ],
            "root": self.root,
            "error": self.error,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any], root: str = "") -> "ObjectModelRegistry":
        models: Dict[str, ObjectModel] = {}
        for entry in (d or {}).get("objects", []) or []:
            model = ObjectModel.from_dict(entry)
            if model.model_id in models:
                raise RGBDError(f"duplicate object model {model.model_id!r}")
            models[model.model_id] = model
        return ObjectModelRegistry(models=models, root=root)


#: Where the registry lives, relative to the repository root.
OBJECT_REGISTRY_PATH = os.path.join("config", "perception_objects.yaml")


#: Where the CAD meshes and reference datasets live. NOT inside the repository:
#: they are third-party reference material that is already on disk beside it,
#: and copying a 183 MB tree in so a path could be repo-relative would buy
#: nothing but a second copy to drift. The FoundationPose worker mounts this
#: same directory read-only at /datasets, so a mesh path is meaningful on both
#: sides of the boundary without translation.
OBJECT_ASSETS_ROOT_ENV = "WISEPACK_PERCEPTION_ASSETS_ROOT"


def object_assets_root(repo_root: str = "") -> str:
    """The directory `mesh_path` entries are relative to."""
    configured = os.environ.get(OBJECT_ASSETS_ROOT_ENV, "").strip()
    if configured:
        return configured.rstrip("/")
    root = repo_root or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    return os.path.join(os.path.dirname(root.rstrip("/")), "references")


def load_object_registry(path: Optional[str] = None, repo_root: str = ""
                         ) -> ObjectModelRegistry:
    """Load the registry. A BROKEN registry is reported, never silently empty.

    An empty registry and an unreadable one are different facts: the first means
    "no models are configured", the second means "the configuration is wrong",
    and rendering them the same way sends an operator to look at the wrong
    thing.
    """
    root = repo_root or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    resolved = path or os.path.join(root, OBJECT_REGISTRY_PATH)
    # THE REGISTRY FILE is tracked in the repository; the MESHES it names are
    # not. Two different roots, deliberately.
    assets = object_assets_root(root)
    if not os.path.isfile(resolved):
        return ObjectModelRegistry(models={}, root=assets, error="")
    try:
        import yaml                                            # noqa: PLC0415
        with open(resolved, encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        registry = ObjectModelRegistry.from_dict(document, root=assets)
        return registry
    except Exception as exc:                                   # noqa: BLE001
        return ObjectModelRegistry(models={}, root=assets,
                                   error=f"{resolved}: {exc}")


__all__ = [
    "RGBDError", "CameraIntrinsics", "RGBDFrame", "ObjectModel",
    "ObjectModelRegistry", "load_object_registry", "OBJECT_REGISTRY_PATH",
    "ALIGNMENT_ALIGNED", "ALIGNMENT_UNALIGNED", "ALIGNMENT_UNKNOWN",
    "MESH_UNITS", "DEPTH_SCALE_MM_PER_UNIT_DEFAULT",
    "object_assets_root", "OBJECT_ASSETS_ROOT_ENV",
]
