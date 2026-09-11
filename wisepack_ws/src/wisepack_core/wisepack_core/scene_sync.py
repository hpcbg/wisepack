"""ObservationBatch -> synchronized Isaac source scene. The ONE frame chain.

    ObservationBatch
      -> scene objects (identity, geometry, pose provenance)
      -> work-area pose            camera_color_optical_frame -> wisepack_workarea
      -> Isaac workcell pose       wisepack_workarea -> table   (mm, robot base)
      -> Isaac world pose          table -> world               (metres, layout)
      -> SceneSpec                 what the simulator spawns, what the robot picks

This module is where a physical observation becomes a source object in the
Isaac workcell. Every transform it applies is a `wisepack_core.pose.
RigidTransform` loaded from `config/isaac_workcell.yaml` through
`wisepack_core.workcell`, with its provenance attached; every unit conversion
and frame origin it uses comes from `wisepack_core.isaac_transform`. There is
no arithmetic on frames anywhere else — not in the simulator, not in the
bridge, not in the dashboard.

WHAT IS REFUSED, AND WHY EACH ONE STOPS THE RUN
-----------------------------------------------
Each of these produces a plausible-looking wrong scene rather than an error if
it is let through, so each is a `SceneSyncRefused` with the reason, and the
bridge holds the scene gate shut on it:

    pose_valid is False          the estimate itself failed
    frame unknown / no transform an identity transform would place the object
                                 inside the camera, and call it a measurement
    non-finite result            a NaN pose spawns nothing and reports success
    outside the work-area bounds a transform error that lands a tube a metre
                                 away must stop, not move the tube
    unreachable for the robot    differential IK converges short and the
                                 gripper closes on air
    stale batch / revision       a later batch is the authoritative scene

Nothing here snaps, clamps or corrects a pose. A measured number is carried or
refused; it is never quietly improved.

TWO GEOMETRIES, KEPT APART
--------------------------
The pose comes from the estimator — CAD-based or model-free, the estimator's
business. The GEOMETRY of the spawned body is the ENGINEERING CAD part from the
object registry (`model_id`, `geometry_source == cad_mesh`), for both methods.
A model-free observation's learned representation is never handed to the scene,
and the CAD mesh is never handed back to the model-free estimator: the two
registries are separate on disk and this module reads neither — it copies the
identity the observation already carries and names its source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .domain import (GEOMETRY_SOURCE_CAD_MESH, GEOMETRY_SOURCE_GENERATED,
                     Axis, GeometryType, PhysicalObservation, Vec3, WasteItem)
from .isaac_contract import (SCENE_SOURCE_GENERATED, SCENE_SOURCE_PHYSICAL,
                             Dimensions, Pose, SceneObject, SceneSpec)
from .isaac_transform import (DEFAULT_LAYOUT, SceneLayout, alignment_to_local_z,
                              axis_from_quaternion, mm_to_m, pose_to_world,
                              quaternion_from_orientation)
from .pose import CAMERA_OPTICAL_FRAME, WORKAREA_FRAME, Orientation, RigidTransform
from .workcell import TABLE_FRAME, WorkcellFrames

#: What the scene geometry of a synchronized object IS, stated on every object
#: so no consumer has to infer it from a model id.
SCENE_GEOMETRY_ENGINEERING_CAD = "engineering_cad"
SCENE_GEOMETRY_CONFIGURED_PROXY = "configured_proxy"


class SceneSyncRefused(ValueError):
    """The batch cannot become a scene. The reason is the message."""


@dataclass(frozen=True)
class FramePose:
    """One pose in one named frame: body centre (mm) and body orientation."""

    frame: str
    centre_mm: Tuple[float, float, float]
    orientation: Orientation
    #: The tube axis LINE in this frame, for a human checking the chain.
    axis_line: Tuple[float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {"frame": self.frame,
                "centre_mm": [round(v, 3) for v in self.centre_mm],
                "orientation": self.orientation.to_dict(),
                "rpy_deg": [round(v, 3) for v in self.orientation.rpy_deg()],
                "axis_line": [round(v, 6) for v in self.axis_line]}


@dataclass
class TransformedObservation:
    """One observation, expressed in every frame of the chain, with evidence."""

    observation_id: str
    camera: Optional[FramePose]
    workarea: FramePose
    table: FramePose
    world_position_m: Tuple[float, float, float]
    world_quaternion_wxyz: Tuple[float, float, float, float]
    pose: Pose
    transform_source: str
    transform_revision: str
    pose_provenance: Dict[str, Any] = field(default_factory=dict)

    def provenance(self) -> Dict[str, Any]:
        return {
            "frame_chain": ([CAMERA_OPTICAL_FRAME] if self.camera else [])
                           + [WORKAREA_FRAME, TABLE_FRAME, "world"],
            "transform_source": self.transform_source,
            "transform_revision": self.transform_revision,
            "camera_pose": self.camera.to_dict() if self.camera else None,
            "workarea_pose": self.workarea.to_dict(),
            "table_pose": self.table.to_dict(),
            "world_pose": {
                "frame": "world",
                "position_m": [round(v, 6) for v in self.world_position_m],
                "quaternion_wxyz": [round(v, 9) for v in self.world_quaternion_wxyz],
            },
            "pose": dict(self.pose_provenance),
        }


def _finite(values: Sequence[float]) -> bool:
    return all(v == v and abs(v) != float("inf") for v in values)


def _axis_line(orientation: Orientation) -> Tuple[float, float, float]:
    """The body's local +Z — its length — in the parent frame."""
    return tuple(float(v) for v in orientation.axis("z"))


def body_orientation(observation: PhysicalObservation) -> Orientation:
    """The BODY-frame orientation an observation implies, in its own frame.

    The estimator reports the rotation of the MODEL frame (CAD or learned
    representation — for Cylinder5 the two coincide by construction, because
    the representation was built from CAD-frame reference poses). The Isaac
    body has its length along local +Z, so the model rotation is composed with
    the inverse of the axis alignment the scene builder applies to the mesh:

        R_body = R_model ∘ align(axis)^-1

    which leaves the body's +Z pointing exactly along the observed tube axis.
    A planar observation declares no axis vector; its yaw IS the heading of the
    length in the plane, so its model axis is local +X by definition.
    """
    if observation.orientation is None:
        raise SceneSyncRefused(
            f"{observation.observation_id}: the observation carries no orientation")
    axis = tuple(observation.task_axis_vector or ()) or (1.0, 0.0, 0.0)
    try:
        align = alignment_to_local_z(axis)
    except ValueError as exc:
        raise SceneSyncRefused(f"{observation.observation_id}: {exc}") from exc
    return observation.orientation.multiply(align.conjugate())


def transform_observation(observation: PhysicalObservation,
                          frames: WorkcellFrames,
                          layout: SceneLayout = DEFAULT_LAYOUT,
                          radius_mm: float = 0.0) -> TransformedObservation:
    """Carry ONE observation through the whole chain, or refuse with the reason.

    Accepts an observation in ``camera_color_optical_frame`` (the physical
    RGB-D case) or already in ``wisepack_workarea`` (the planar case). Any other
    frame is refused: there is no transform for it and none is assumed.
    """
    oid = observation.observation_id
    if not observation.pose_valid:
        raise SceneSyncRefused(
            f"{oid}: pose_valid is false — the estimate itself is not usable")
    if observation.orientation is None:
        raise SceneSyncRefused(f"{oid}: the observation carries no orientation")

    centre = tuple(float(v) for v in observation.object_center)
    orientation = body_orientation(observation)
    if not _finite(centre) or not _finite(orientation.as_tuple()):
        raise SceneSyncRefused(f"{oid}: the observed pose is not finite")

    # A HEIGHT THAT WAS NEVER MEASURED IS NOT CARRIED AS ONE. A planar method
    # declares which degrees of freedom it measured; when height is not among
    # them the object is taken to REST ON THE SOURCE PLANE, one radius up, and
    # the provenance says so. An RGB-D method measures z and is never touched.
    # An observation that declares nothing is treated as fully measured.
    assumed_height = False
    measured = tuple(observation.measured_dof or ())
    if measured and "z" not in measured and observation.frame_id == WORKAREA_FRAME:
        centre = (centre[0], centre[1],
                  frames.bounds.source_plane_z_mm + float(radius_mm))
        assumed_height = True

    camera_pose: Optional[FramePose] = None
    if observation.frame_id == CAMERA_OPTICAL_FRAME:
        if not frames.available:
            raise SceneSyncRefused(
                f"{oid}: no {CAMERA_OPTICAL_FRAME} -> {WORKAREA_FRAME} transform "
                f"is available ({frames.unavailable_reason}); the pose stays in "
                "the camera frame and NO identity transform is assumed")
        camera_to_workarea: RigidTransform = frames.camera_to_workarea  # type: ignore[assignment]
        camera_pose = FramePose(CAMERA_OPTICAL_FRAME, centre, orientation,
                                _axis_line(orientation))
        centre = camera_to_workarea.apply_to_position(centre)
        orientation = camera_to_workarea.apply_to_orientation(orientation)
    elif observation.frame_id == WORKAREA_FRAME:
        pass
    else:
        raise SceneSyncRefused(
            f"{oid}: pose is in frame {observation.frame_id!r}, for which no "
            "transform into the work area is configured")

    if not _finite(centre) or not _finite(orientation.as_tuple()):
        raise SceneSyncRefused(f"{oid}: the work-area pose is not finite")
    workarea_pose = FramePose(WORKAREA_FRAME, centre, orientation,
                              _axis_line(orientation))

    refusal = frames.bounds.refusal(centre[0], centre[1], centre[2], radius_mm)
    if refusal:
        raise SceneSyncRefused(f"{oid}: {refusal}")

    if frames.workarea_to_table is None or not frames.workarea_to_table.valid:
        raise SceneSyncRefused(
            f"{oid}: no {WORKAREA_FRAME} -> {TABLE_FRAME} transform is "
            f"available ({frames.unavailable_reason})")
    workarea_to_table: RigidTransform = frames.workarea_to_table
    table_centre = workarea_to_table.apply_to_position(centre)
    table_orientation = workarea_to_table.apply_to_orientation(orientation)
    if not _finite(table_centre) or not _finite(table_orientation.as_tuple()):
        raise SceneSyncRefused(f"{oid}: the workcell pose is not finite")
    table_pose = FramePose(TABLE_FRAME, table_centre, table_orientation,
                           _axis_line(table_orientation))

    reach = math.hypot(table_centre[0], table_centre[1]) / 1000.0
    if not layout.robot_min_reach_m <= reach <= layout.robot_max_reach_m:
        raise SceneSyncRefused(
            f"{oid}: the object centre is {reach:.3f} m from the robot base, "
            f"outside the reachable band [{layout.robot_min_reach_m}, "
            f"{layout.robot_max_reach_m}] m of the selected layout")

    quaternion = quaternion_from_orientation(table_orientation)
    pose = Pose(x_mm=table_centre[0], y_mm=table_centre[1], z_mm=table_centre[2],
                axis=axis_from_quaternion(quaternion).value, frame=TABLE_FRAME,
                orientation=table_orientation.as_tuple())
    world_position, world_quaternion = pose_to_world(pose, layout)
    if not _finite(world_position) or not _finite(world_quaternion):
        raise SceneSyncRefused(f"{oid}: the Isaac world pose is not finite")

    return TransformedObservation(
        observation_id=oid,
        camera=camera_pose,
        workarea=workarea_pose,
        table=table_pose,
        world_position_m=tuple(float(v) for v in world_position),
        world_quaternion_wxyz=tuple(float(v) for v in world_quaternion),
        pose=pose,
        transform_source=frames.provenance,
        transform_revision=(frames.camera_to_workarea.revision
                            if frames.camera_to_workarea else ""),
        pose_provenance={
            "observation_id": oid,
            "perception_method": observation.perception_method,
            "estimator_geometry": _estimator_geometry(observation),
            "object_model_id": observation.object_model_id,
            "detector": observation.detector,
            "model_id": observation.model_id,
            "captured_at": observation.captured_at,
            "pose_valid": bool(observation.pose_valid),
            "source_frame": observation.frame_id,
            "measured_dof": list(observation.measured_dof),
            "height_assumed_on_source_plane": assumed_height,
            "symmetry": (observation.symmetry.to_dict()
                         if observation.symmetry else None),
        })


def _estimator_geometry(observation: PhysicalObservation) -> str:
    method = str(observation.perception_method or "")
    if method.endswith("model_free"):
        return "learned_representation"
    if method.startswith("foundationpose"):
        return "cad"
    if method:
        return "none"
    return ""


def synchronize_scene(items: Sequence[WasteItem], batch: Any,
                      frames: WorkcellFrames, *, run_id: str,
                      scenario_revision: int,
                      layout: SceneLayout = DEFAULT_LAYOUT) -> SceneSpec:
    """The authoritative source objects for ONE scene revision, from ONE batch.

    ``items`` are the planning items the engine built from ``batch`` (each
    carrying its ``observation``); the spec is built from THEM so the ids the
    plan uses are the ids the scene spawns. Every item must be observed — a
    generated item mixed into a physical batch is refused, not laid out in a
    row beside the real one.
    """
    if batch is None:
        raise SceneSyncRefused("no observation batch has been applied to this run")
    if not getattr(batch, "ok", False):
        raise SceneSyncRefused(
            f"the observation batch failed: {getattr(batch, 'error', '') or 'unknown'}")
    if not items:
        raise SceneSyncRefused(
            "the observation batch produced no items; an empty scene is not "
            "a synchronized scene")
    if not frames.available:
        raise SceneSyncRefused(
            f"the work-area transform is unavailable: {frames.unavailable_reason}")

    objects: List[SceneObject] = []
    for item in items:
        observation = getattr(item, "observation", None)
        if observation is None:
            raise SceneSyncRefused(
                f"{item.item_id} carries no observation; a synchronized scene "
                "contains observed objects only")
        transformed = transform_observation(
            observation, frames, layout,
            radius_mm=float(item.outer_diameter_mm) / 2.0)
        cad_backed = (item.geometry_source == GEOMETRY_SOURCE_CAD_MESH
                      and bool(item.model_id))
        provenance = transformed.provenance()
        provenance["scene_geometry"] = {
            "source": (SCENE_GEOMETRY_ENGINEERING_CAD if cad_backed
                       else SCENE_GEOMETRY_CONFIGURED_PROXY),
            "model_id": item.model_id if cad_backed else "",
            "note": ("engineering CAD from config/perception_objects.yaml; the "
                     "estimator's learned representation is never used as "
                     "scene geometry" if cad_backed else
                     "configured proxy cylinder; no CAD model exists"),
        }
        objects.append(SceneObject(
            item_id=item.item_id,
            dimensions=Dimensions(
                length_mm=int(item.length_mm),
                outer_diameter_mm=int(item.outer_diameter_mm),
                inner_diameter_mm=(None if item.inner_diameter_mm is None
                                   else int(item.inner_diameter_mm))),
            source_pose=transformed.pose,
            model_id=item.model_id if cad_backed else "",
            geometry_source=(GEOMETRY_SOURCE_CAD_MESH if cad_backed
                             else GEOMETRY_SOURCE_GENERATED),
            observation_id=observation.observation_id,
            perception_method=observation.perception_method,
            weight_kg=float(item.weight_kg),
            material=str(item.material),
            provenance=provenance))

    return SceneSpec(
        scene_source=SCENE_SOURCE_PHYSICAL,
        run_id=run_id,
        scenario_revision=int(scenario_revision),
        observation_batch_id=str(getattr(batch, "batch_id", "") or ""),
        captured_at=str(getattr(batch, "captured_at", "") or ""),
        transform_source=frames.provenance,
        transform_revision=(frames.camera_to_workarea.revision
                            if frames.camera_to_workarea else ""),
        objects=objects)


def scene_items(scene: SceneSpec) -> List[WasteItem]:
    """The domain items a synchronized scene consists of — for the SIMULATOR.

    The simulator has no engine and no batch; it receives the spec and must
    spawn bodies with the right geometry, mass and identity. This rebuilds the
    same ``WasteItem`` shape the scene builder already consumes, from the spec
    alone, so the builder has one code path for both scene sources.
    """
    items: List[WasteItem] = []
    for source in scene.objects:
        pose = source.source_pose
        item = WasteItem(
            item_id=source.item_id,
            length_mm=int(source.dimensions.length_mm),
            outer_diameter_mm=int(source.dimensions.outer_diameter_mm),
            geometry_type=GeometryType.TUBE,
            inner_diameter_mm=source.dimensions.inner_diameter_mm,
            material=source.material or "carbon_steel",
            weight_kg=float(source.weight_kg),
            source_position=Vec3(int(round(pose.x_mm)), int(round(pose.y_mm)),
                                 int(round(pose.z_mm))),
            permitted_axes=(Axis.X, Axis.Y),
        )
        item.geometry_source = source.geometry_source
        item.model_id = source.model_id
        items.append(item)
    return items


def describe_chain(frames: WorkcellFrames, layout: SceneLayout = DEFAULT_LAYOUT
                   ) -> Dict[str, Any]:
    """The configured chain, for the dashboard and the evidence report."""
    return {
        **frames.to_dict(),
        "table_to_world": {
            "parent_frame": "world", "child_frame": TABLE_FRAME,
            "origin_m": [round(v, 6) for v in layout.table_frame_origin_m],
            "note": "millimetres to metres; axes parallel; from the selected "
                    "robot's SceneLayout",
        },
        "robot_base_table_mm": [0.0, 0.0, 0.0],
        "table_top_z_m": layout.table_top_z_m,
        "container_outer_xy_m": list(layout.container_outer_xy_m),
    }


__all__ = [
    "SceneSyncRefused", "FramePose", "TransformedObservation",
    "transform_observation", "body_orientation", "synchronize_scene",
    "scene_items", "describe_chain", "SCENE_GEOMETRY_ENGINEERING_CAD",
    "SCENE_GEOMETRY_CONFIGURED_PROXY",
]
