"""The `rgbd_scene_depth_plane` method: footprints on the plane -> observations.

WHAT THIS PROVIDER DOES. It takes the instances `scene_segmentation` measured
on one physical RGB-D frame, assigns each one a DEMO OBJECT CLASS from
`config/scene_demo_classes.yaml` by its footprint, and builds one
`PhysicalObservation` per classified object in `camera_color_optical_frame`:
the engineering model's origin, an orientation that lays the model's task axis
along the measured heading and its thin axis along the plane normal, and the
CAD dimensions the registry declares for that model. The result is an ordinary
`ObservationBatch`, which is what the packing layer, the Digital Twin and the
Isaac scene synchronizer consume — nothing downstream knows this method exists.

WHAT IT DOES NOT DO, stated so nobody has to discover it:

* NO RECOGNITION. Identity is assigned by footprint size against a configured
  table. Two parts with the same footprint are the same class here.
* NO 6-DoF ESTIMATOR. The pose is planar: x, y and heading measured on the
  fitted plane, z from the plane and the part's own radius, tilt assumed zero
  because the part rests on the bench. `measured_dof` says exactly that.
* NO CAD TO ANY ESTIMATOR. The CAD model is looked up AFTER classification, to
  instantiate the object in the twin; nothing here registers a mesh to pixels.
* NOTHING GUESSED. An unclassified footprint, an ignored region and an object
  the configured workcell cannot place are reported with their reason and are
  left out of the batch. The count of what was left out travels with the batch.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from wisepack_core.acquisition import ACQUISITION_REALSENSE
from wisepack_core.domain import PhysicalObservation
from wisepack_core.perception import (BatchStatus, ObservationBatch,
                                      PerceptionMethod, PerceptionSource)
from wisepack_core.pose import CAMERA_OPTICAL_FRAME, Orientation, Symmetry
from wisepack_core.rgbd import load_object_registry

from scene_segmentation import Instance, Plane

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSES_PATH = os.path.join(REPO, "config", "scene_demo_classes.yaml")

METHOD = PerceptionMethod.RGBD_SCENE_DEPTH_PLANE.value
DETECTOR_ID = "scene/depth_plane_footprint"
DETECTOR_REVISION = "depth_plane_footprint/1.0"
GEOMETRY_MEASURED_FOOTPRINT = "measured_footprint"


# --------------------------------------------------------------------------- #
# The class catalogue
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DemoClass:
    demo_type: str
    model_id: str
    length_mm: Tuple[float, float]
    width_mm: Tuple[float, float]
    min_height_mm: Optional[float] = None
    max_height_mm: Optional[float] = None
    proxy_diameter_mm: Optional[int] = None

    def matches(self, length: float, width: float, height: float) -> bool:
        if not self.length_mm[0] <= length <= self.length_mm[1]:
            return False
        if not self.width_mm[0] <= width <= self.width_mm[1]:
            return False
        if self.min_height_mm is not None and height < self.min_height_mm:
            return False
        if self.max_height_mm is not None and height > self.max_height_mm:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {"demo_type": self.demo_type, "model_id": self.model_id,
                "length_mm": list(self.length_mm), "width_mm": list(self.width_mm),
                "min_height_mm": self.min_height_mm,
                "max_height_mm": self.max_height_mm,
                "proxy_diameter_mm": self.proxy_diameter_mm}


@dataclass
class DemoCatalogue:
    classes: List[DemoClass]
    ignore_regions_px: List[Dict[str, Any]] = field(default_factory=list)
    scene_roi_px: Optional[List[int]] = None
    provenance: str = "configured_demo"
    path: str = ""

    def classify(self, instance: Instance) -> Optional[DemoClass]:
        for cls in self.classes:
            if cls.matches(instance.length_mm, instance.width_mm, instance.height_mm):
                return cls
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "provenance": self.provenance,
                "classes": [c.to_dict() for c in self.classes],
                "ignore_regions_px": list(self.ignore_regions_px),
                "scene_roi_px": list(self.scene_roi_px or [])}


def load_catalogue(path: str = CLASSES_PATH) -> DemoCatalogue:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    classes: List[DemoClass] = []
    for entry in raw.get("classes") or []:
        proxy = entry.get("proxy") or {}
        classes.append(DemoClass(
            demo_type=str(entry["demo_type"]),
            model_id=str(entry.get("model_id") or ""),
            length_mm=tuple(float(v) for v in entry["length_mm"]),
            width_mm=tuple(float(v) for v in entry["width_mm"]),
            min_height_mm=entry.get("min_height_mm"),
            max_height_mm=entry.get("max_height_mm"),
            proxy_diameter_mm=(int(proxy["diameter_mm"])
                               if proxy.get("diameter_mm") else None)))
    if not classes:
        raise ValueError(f"{path}: no demo classes declared")
    roi = raw.get("scene_roi_px")
    return DemoCatalogue(classes=classes,
                         ignore_regions_px=list(raw.get("ignore_regions_px") or []),
                         scene_roi_px=[int(v) for v in roi] if roi else None,
                         provenance=str(raw.get("provenance", "configured_demo")),
                         path=path)


# --------------------------------------------------------------------------- #
# Orientation from a footprint
# --------------------------------------------------------------------------- #


def _unit(v: Sequence[float]) -> Tuple[float, float, float]:
    norm = math.sqrt(sum(float(a) * float(a) for a in v))
    if norm < 1e-12:
        raise ValueError("zero vector")
    return tuple(float(a) / norm for a in v)  # type: ignore[return-value]


def _cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def model_axes(task_axis: Sequence[float], extents_mm: Sequence[float]
               ) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
    """(length axis, second axis, thin axis) of a model, orthonormal, right-handed.

    The thin axis is the coordinate axis with the smallest mesh extent among
    those perpendicular to the task axis — the plate's thickness, and for a
    tube any perpendicular, which is fine because a tube does not care.
    """
    a = _unit(task_axis)
    candidates = []
    for k in range(3):
        basis = [0.0, 0.0, 0.0]
        basis[k] = 1.0
        # Component perpendicular to the task axis.
        dot = sum(basis[i] * a[i] for i in range(3))
        perp = [basis[i] - dot * a[i] for i in range(3)]
        norm = math.sqrt(sum(p * p for p in perp))
        if norm < 0.3:
            continue
        extent = float(extents_mm[k]) if len(extents_mm) == 3 else 1.0
        candidates.append((extent, k, tuple(p / norm for p in perp)))
    if not candidates:
        raise ValueError("no axis perpendicular to the task axis")
    candidates.sort()
    c = _unit(candidates[0][2])
    b = _unit(_cross(c, a))
    c = _unit(_cross(a, b))
    return a, b, c


def orientation_on_plane(heading: Sequence[float], plane_normal_up: Sequence[float],
                         task_axis: Sequence[float], extents_mm: Sequence[float]
                         ) -> Orientation:
    """The model rotation that lays `task_axis` along `heading` on the plane
    and the model's thin axis along the plane normal (pointing up)."""
    n = _unit(plane_normal_up)
    d = [float(v) for v in heading]
    dot = sum(d[i] * n[i] for i in range(3))
    d = _unit([d[i] - dot * n[i] for i in range(3)])         # in-plane heading
    e = _unit(_cross(n, d))
    a, b, c = model_axes(task_axis, extents_mm)
    # R = T M^T with M = [a b c], T = [d e n]: R a = d, R b = e, R c = n.
    T = [[d[0], e[0], n[0]], [d[1], e[1], n[1]], [d[2], e[2], n[2]]]
    M = [[a[0], b[0], c[0]], [a[1], b[1], c[1]], [a[2], b[2], c[2]]]
    R = [[sum(T[i][k] * M[j][k] for k in range(3)) for j in range(3)]
         for i in range(3)]
    return Orientation.from_matrix(R)


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #


@dataclass
class ClassifiedInstance:
    instance: Instance
    demo_class: Optional[DemoClass]
    status: str                   # observed | ignored | unclassified | excluded
    reason: str = ""
    observation_id: str = ""
    model_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {**self.instance.to_dict(),
                "demo_type": self.demo_class.demo_type if self.demo_class else "",
                "model_id": self.model_id,
                "status": self.status, "reason": self.reason,
                "observation_id": self.observation_id}


def build_batch(instances: Sequence[Instance], plane: Plane, *,
                catalogue: Optional[DemoCatalogue] = None,
                batch_id: str = "physical-scene-1",
                captured_at: str = "", dataset: str = "",
                intrinsics: Optional[Dict[str, float]] = None,
                repo_root: str = REPO) -> Tuple[ObservationBatch, List[ClassifiedInstance]]:
    """Classify every instance and build the batch of the classified ones."""
    catalogue = catalogue or load_catalogue()
    registry = load_object_registry(repo_root=repo_root)
    up = plane.normal                                   # toward the camera
    requested_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    observations: List[PhysicalObservation] = []
    report: List[ClassifiedInstance] = []
    for instance in instances:
        if instance.ignored:
            report.append(ClassifiedInstance(instance, None, "ignored", instance.ignored))
            continue
        cls = catalogue.classify(instance)
        if cls is None:
            report.append(ClassifiedInstance(
                instance, None, "unclassified",
                f"footprint {instance.length_mm:.0f} x {instance.width_mm:.0f} mm, "
                f"height {instance.height_mm:.0f} mm matches no demo class"))
            continue
        index = len(observations) + 1
        observation_id = f"{batch_id}-obj-{index}"
        model = registry.models.get(cls.model_id) if cls.model_id else None
        if cls.model_id and model is None:
            report.append(ClassifiedInstance(
                instance, cls, "unclassified",
                f"class {cls.demo_type} names model {cls.model_id!r}, which the "
                "object registry does not declare"))
            continue

        if model is not None:
            task_axis = tuple(model.task_axis_vector or ()) or {
                "x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)
            }[model.task_axis]
            extents = tuple(model.extents_mm or ())
            diameter = int(model.diameter_mm or round(instance.width_mm))
            length = int(model.length_mm or round(instance.length_mm))
            inner = model.inner_diameter_mm
            thickness = float(diameter)
            if str(model.object_type) == "plate" and len(extents) == 3:
                thickness = float(min(extents))
            model_center = tuple(model.model_center_mm or (0.0, 0.0, 0.0))
            symmetry = model.symmetry
            object_type = model.object_type
            geometry_source = "cad_model"
        else:
            task_axis = (1.0, 0.0, 0.0)
            extents = ()
            diameter = int(cls.proxy_diameter_mm or max(4, min(14, round(instance.width_mm))))
            length = max(10, int(round(instance.length_mm)))
            inner = None
            thickness = float(diameter)
            model_center = (0.0, 0.0, 0.0)
            symmetry = Symmetry(type="axial", axis="x")
            object_type = cls.demo_type
            geometry_source = GEOMETRY_MEASURED_FOOTPRINT

        orientation = orientation_on_plane(instance.axis_line, up, task_axis, extents)
        # The body centre sits half a thickness above the plane; the model
        # ORIGIN is that centre minus the rotated model centre, so that
        # `object_center = t + R c` lands on the body exactly as it does for a
        # FoundationPose observation.
        centre = tuple(instance.centre_mm[i] + up[i] * thickness / 2.0 for i in range(3))
        rotated_centre = orientation.rotate(model_center)
        origin = tuple(centre[i] - rotated_centre[i] for i in range(3))

        observation = PhysicalObservation(
            observation_id=observation_id,
            x_mm=origin[0], y_mm=origin[1], z_mm=origin[2],
            object_type=str(object_type),
            source=PerceptionSource.CAMERA.value,
            frame_id=CAMERA_OPTICAL_FRAME,
            detector=DETECTOR_ID,
            model_id=DETECTOR_REVISION,
            detector_class=cls.demo_type,
            detector_object_index=instance.index,
            captured_at=captured_at or requested_at,
            calibration_status="not_applicable",
            diameter_mm=diameter, length_mm=length, inner_diameter_mm=inner,
            geometry_source=geometry_source,
            orientation=orientation,
            symmetry=symmetry,
            perception_method=METHOD,
            object_model_id=cls.model_id,
            pose_valid=True,
            workarea_transform_valid=False,
            measured_dof=("x", "y", "z", "yaw"),
            model_center_mm=tuple(float(v) for v in model_center),
            task_axis_vector=tuple(float(v) for v in task_axis),
            confidence=None)
        observations.append(observation)
        report.append(ClassifiedInstance(instance, cls, "observed", "",
                                         observation_id, cls.model_id))

    status = BatchStatus.OK if observations else BatchStatus.EMPTY
    batch = ObservationBatch(
        batch_id=batch_id,
        source=PerceptionSource.CAMERA.value,
        status=status,
        observations=observations,
        frame_id=CAMERA_OPTICAL_FRAME,
        captured_at=captured_at or requested_at,
        requested_at=requested_at,
        detector=DETECTOR_ID,
        perception_method=METHOD,
        acquisition=ACQUISITION_REALSENSE,
        model_id="",
        calibration_status="not_applicable",
        error="" if observations else "no instance matched a demo class",
        detector_status={
            "acquisition": ACQUISITION_REALSENSE,
            "dataset": dataset,
            "intrinsics": intrinsics or {},
            "plane": plane.to_dict(),
            "catalogue": catalogue.path,
            "catalogue_provenance": catalogue.provenance,
            "estimator_geometry": "",
            "pose_note": ("planar pose on the fitted work plane: x, y, z and "
                          "heading measured; tilt assumed zero (the part rests "
                          "on the bench). No 6-DoF estimator and no CAD supplied "
                          "to one."),
            "identity_note": ("object class assigned by FOOTPRINT SIZE against "
                              "config/scene_demo_classes.yaml, not recognised."),
            "instances": [c.to_dict() for c in report],
            "counts": {
                "measured": len(report),
                "observed": sum(1 for c in report if c.status == "observed"),
                "ignored": sum(1 for c in report if c.status == "ignored"),
                "unclassified": sum(1 for c in report if c.status == "unclassified"),
            },
            "frame_note": ("pose in camera_color_optical_frame; the Isaac scene "
                           "synchronizer applies the configured demo transform"),
        })
    return batch, report


__all__ = ["METHOD", "DETECTOR_ID", "DemoClass", "DemoCatalogue", "ClassifiedInstance",
           "load_catalogue", "model_axes", "orientation_on_plane", "build_batch",
           "GEOMETRY_MEASURED_FOOTPRINT"]
