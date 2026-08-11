"""Procedural WISEPACK scene for Isaac Sim 6.0.1.

EVERYTHING IS GENERATED. There is no USD asset in this repository and there does
not need to be: a table, an open-top bin and a handful of pipe segments are boxes
and cylinders, and a committed binary would be one more thing to keep in step
with the scenario definition. The one asset that IS loaded — the SELECTED ROBOT
— comes from the Isaac Sim asset root at runtime, resolved through
``isaacsim.storage.native.get_assets_root_path`` rather than copied here, and it
is loaded by the robot adapter rather than by this module. Nothing here knows
which arm is in the cell.

THE ITEMS COME FROM WISEPACK, NOT FROM THIS FILE
------------------------------------------------
``build_items`` takes the ``Scenario`` produced by
``wisepack_core.build_scenario(preset, seed)`` — the identical call the
orchestrator makes — so the object ids, lengths, diameters and masses in the
physical scene are the same ones the optimizer planned against, by construction.
There is no second hard-coded object list to fall out of step. If the two ever
disagree it is because they were given different (preset, seed), which the
launcher passes through explicitly and the run reports.

API NOTE — Isaac Sim 6.0.1
--------------------------
This uses the ``isaacsim.core.experimental.*`` API. In 6.0.1 the older
``isaacsim.core.api`` has moved to ``extsDeprecated/``; it still loads, but
writing new code against a deprecated tree is how an integration ages badly
before it ships. Verified against the shipped standalone examples under
``standalone_examples/api/isaacsim.core.experimental.api/``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.materials import RigidBodyMaterial
from isaacsim.core.experimental.objects import Cube, Cylinder, DomeLight, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from pxr import Gf, UsdGeom, UsdLux

from wisepack_core.domain import (GEOMETRY_SOURCE_CAD_MESH, Scenario, Vec3,
                                  WasteItem)
from wisepack_core.isaac_transform import (
    SceneLayout, mm_to_m, pose_to_world, table_pose_for_index,
)

from .config import LOG_SCENE, PhysicsConfig

#: Prim path roots. Collected here so nothing else in the codebase has to know
#: the stage layout, and so `item_path` is the single naming rule.
#:
#: THE ROBOT'S PATH IS NOT HERE, and used to be. It belongs to the selected
#: robot's profile (``root_prim_path`` in config/isaac_robots.yaml), because a
#: stage constant naming one arm is exactly the kind of assumption that makes a
#: second arm a rewrite rather than a selection.
WORLD = "/World"
TABLE_PATH = f"{WORLD}/Table"
ITEMS_ROOT = f"{WORLD}/Items"
CONTAINERS_ROOT = f"{WORLD}/Containers"
MATERIALS_ROOT = f"{WORLD}/PhysicsMaterials"


def item_path(item_id: str) -> str:
    """USD path for one waste item. ``item-001`` -> ``/World/Items/item_001``.

    Hyphens are legal in WISEPACK ids (they travel into NGSI-LD urns) but not in
    USD prim names, so exactly one substitution happens and it happens here.
    """
    return f"{ITEMS_ROOT}/{item_id.replace('-', '_')}"


def container_path(container_id: str) -> str:
    return f"{CONTAINERS_ROOT}/{container_id.replace('-', '_')}"


def _alignment_to_local_z(model: Any) -> Any:
    """Rotation taking a model's own tube axis onto local +Z.

    The registry declares that axis — measured, and a VECTOR for parts modelled
    obliquely — so this reads a declaration rather than re-deriving geometry.
    Returns identity when the axis is already +Z.
    """
    vector = tuple(getattr(model, "task_axis_vector", ()) or ())
    if not vector:
        vector = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0),
                  "z": (0.0, 0.0, 1.0)}[getattr(model, "task_axis", "z")]
    source = np.asarray(vector, dtype=np.float64)
    source = source / np.linalg.norm(source)
    target = np.array([0.0, 0.0, 1.0])

    cross = np.cross(source, target)
    dot = float(source @ target)
    if np.linalg.norm(cross) < 1e-9:
        # Parallel or antiparallel: identity, or a half turn about any
        # perpendicular. For a tube either end is equivalent, so the choice of
        # perpendicular does not matter.
        return np.eye(3) if dot > 0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array([[0.0, -cross[2], cross[1]],
                     [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    return (np.eye(3) + skew
            + skew @ skew * (1.0 / (1.0 + dot)))


class WisepackScene:
    """Builds and owns the procedural scene, and reads rigid-body state back."""

    def __init__(self, layout: SceneLayout, physics: PhysicsConfig) -> None:
        self.layout = layout
        self.physics = physics
        self.items: Dict[str, RigidPrim] = {}
        self.item_specs: Dict[str, WasteItem] = {}
        self.item_index: Dict[str, int] = {}
        #: For CAD-backed items: the offset applied when the mesh was centred on
        #: its own body, in metres. Kept so the EVALUATION path can express the
        #: simulator's ground-truth pose in the ORIGINAL CAD frame — the frame
        #: FoundationPose reports in — instead of the centred one. It is never
        #: used by the runtime path, which learns object poses from perception.
        self.cad_mesh_offsets: Dict[str, Any] = {}
        self.containers: Dict[str, Vec3] = {}
        self._item_material: Optional[RigidBodyMaterial] = None
        self._static_material: Optional[RigidBodyMaterial] = None

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def build(self, scenario: Scenario, container_ids: List[str]) -> None:
        """Create the whole scene: ground, light, table, containers, items."""
        print(f"{LOG_SCENE} creating stage (metres, Z up)")
        stage_utils.create_new_stage()
        # Stated explicitly rather than relied upon. Every number that reaches
        # this module has already been converted to metres by
        # wisepack_core.isaac_transform, and a stage in centimetres would scale
        # all of them silently.
        stage_utils.set_stage_units(meters_per_unit=1.0)

        GroundPlane(f"{WORLD}/GroundPlane")
        DomeLight(f"{WORLD}/DomeLight").set_intensities(900)
        self._add_key_light()
        self._build_materials()
        self._build_table()

        for container_id in container_ids:
            inner = (scenario.container_template.inner_size
                     if scenario.container_template else Vec3(300, 220, 150))
            self.build_container(container_id, inner)

        self.build_items(scenario)
        self._add_camera()
        print(f"{LOG_SCENE} scene ready: {len(self.items)} items, "
              f"{len(self.containers)} container(s)")

    def _add_key_light(self) -> None:
        """A directional key light so the GUI view is readable.

        The dome light alone renders the scene flat and shadowless, which makes
        it genuinely hard to see whether a cylinder is inside the bin or resting
        on its rim — the one thing a person watching this demonstration is
        trying to judge.
        """
        stage = stage_utils.get_current_stage(backend="usd")
        light = UsdLux.DistantLight.Define(stage, f"{WORLD}/KeyLight")
        light.CreateIntensityAttr(1800.0)
        light.CreateAngleAttr(1.5)
        UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(
            Gf.Vec3f(-40.0, 0.0, 35.0))

    def _build_materials(self) -> None:
        p = self.physics
        self._item_material = RigidBodyMaterial(
            f"{MATERIALS_ROOT}/Item",
            static_frictions=p.item_static_friction,
            dynamic_frictions=p.item_dynamic_friction,
            restitutions=p.item_restitution)
        self._static_material = RigidBodyMaterial(
            f"{MATERIALS_ROOT}/Static",
            static_frictions=p.table_static_friction,
            dynamic_frictions=p.table_dynamic_friction,
            restitutions=p.container_restitution)

    def _build_table(self) -> None:
        """A static box whose TOP surface is at ``layout.table_top_z_m``.

        The robot — whichever one — is mounted on that surface and every item
        rests on it, so the top is the datum. The box is positioned by its
        centre, hence the half height offset: the one place that arithmetic
        appears.
        """
        size = self.layout.table_size_m
        cx, cy = self.layout.table_centre_xy_m
        centre_z = self.layout.table_top_z_m - size[2] / 2.0
        Cube(paths=TABLE_PATH, sizes=1.0, scales=np.array([size]),
             positions=np.array([[cx, cy, centre_z]]), colors="grey")
        table = GeomPrim(paths=TABLE_PATH, apply_collision_apis=True)
        table.apply_physics_materials(self._static_material)
        # No RigidPrim wrapper: without a rigid-body API the box is static
        # geometry, which is what a table is. Making it dynamic and then
        # constraining it would be a heavier way to reach the same place.
        print(f"{LOG_SCENE} table {size[0]}x{size[1]}x{size[2]} m, "
              f"top at z={self.layout.table_top_z_m} m")

    def build_container(self, container_id: str, inner_mm: Vec3) -> None:
        """An OPEN-TOP bin: one floor and four walls, all static colliders.

        Built from five boxes rather than a hollow mesh so every face is a
        primitive collider. PhysX resolves box contacts exactly; a concave mesh
        would need convex decomposition, and a badly decomposed bin lets items
        fall through a wall — which looks like a placement bug and is not one.

        The INNER volume matches the WISEPACK container spec exactly, because
        that inner volume is the frame every placement is expressed in.
        """
        inner = (mm_to_m(inner_mm.x), mm_to_m(inner_mm.y), mm_to_m(inner_mm.z))
        t = self.physics.container_wall_thickness
        ox, oy, oz = self.layout.container_outer_origin_for(container_id)
        root = container_path(container_id)
        # Define the grouping Xform through USD directly. The experimental
        # XformPrim wrapper ASSERTS the prim already exists — it wraps, it does
        # not create — so calling it here failed with "Specified paths must
        # correspond to existing prims".
        UsdGeom.Xform.Define(stage_utils.get_current_stage(backend="usd"), root)

        # Each entry: (name, size, centre) in world metres. The inner cavity
        # spans [ox+t, ox+t+inner_x] etc., which is exactly what
        # SceneLayout.container_origin_for returns as the frame origin.
        panels: List[Tuple[str, Tuple[float, float, float],
                           Tuple[float, float, float]]] = [
            ("Floor", (inner[0] + 2 * t, inner[1] + 2 * t, t),
             (ox + inner[0] / 2 + t, oy + inner[1] / 2 + t, oz + t / 2)),
            ("WallXMin", (t, inner[1] + 2 * t, inner[2]),
             (ox + t / 2, oy + inner[1] / 2 + t, oz + t + inner[2] / 2)),
            ("WallXMax", (t, inner[1] + 2 * t, inner[2]),
             (ox + inner[0] + 1.5 * t, oy + inner[1] / 2 + t,
              oz + t + inner[2] / 2)),
            ("WallYMin", (inner[0], t, inner[2]),
             (ox + inner[0] / 2 + t, oy + t / 2, oz + t + inner[2] / 2)),
            ("WallYMax", (inner[0], t, inner[2]),
             (ox + inner[0] / 2 + t, oy + inner[1] + 1.5 * t,
              oz + t + inner[2] / 2)),
        ]
        for name, size, centre in panels:
            path = f"{root}/{name}"
            Cube(paths=path, sizes=1.0, scales=np.array([size]),
                 positions=np.array([centre]), colors="cyan")
            panel = GeomPrim(paths=path, apply_collision_apis=True)
            panel.apply_physics_materials(self._static_material)

        self.containers[container_id] = inner_mm
        print(f"{LOG_SCENE} container {container_id}: inner "
              f"{inner_mm.x}x{inner_mm.y}x{inner_mm.z} mm, inner origin at "
              f"{tuple(round(v, 3) for v in self.layout.container_origin_for(container_id))} m")

    def build_items(self, scenario: Scenario) -> None:
        """Spawn one dynamic cylinder per WISEPACK waste item, on the table.

        Mass is taken from the item, not from a density: the generator already
        computed it from the real alloy density and the hollow tube's material
        volume, and a solid cylinder of the same outer diameter would be several
        times heavier than the pipe it represents.
        """
        for index, item in enumerate(scenario.items):
            pose = table_pose_for_index(index, item, self.layout)
            position, orientation = pose_to_world(pose, self.layout)
            path = item_path(item.item_id)

            # TWO GEOMETRY PATHS, and the item says which. Neither replaces the
            # other: a generated tube is a parametric cylinder and stays exactly
            # what it always was, while a CAD-backed item is the REAL part —
            # hollow bore, saddle ends and all — because for the perception and
            # sim-to-real work the exact geometry is the entire point.
            #
            # The branch is here, in the simulator adapter, and nowhere else.
            # Planning code never learns which path an item took.
            if item.geometry_source == GEOMETRY_SOURCE_CAD_MESH:
                self._build_cad_item(item, path, position, orientation)
            else:
                Cylinder(
                    paths=path,
                    radii=mm_to_m(item.outer_diameter_mm) / 2.0,
                    heights=mm_to_m(item.length_mm),
                    axes="Z",                   # matches quaternion_for_axis()
                    positions=np.array([position]),
                    orientations=np.array([orientation]),
                    colors="orange")
            geom = GeomPrim(paths=path, apply_collision_apis=True)
            geom.apply_physics_materials(self._item_material)

            body = RigidPrim(paths=path)
            body.set_masses(np.array([max(item.weight_kg, 0.05)]))
            # SLEEPING IS LEFT ENABLED, deliberately, and an earlier version of
            # this file disabling it was a real bug.
            #
            # The worry was that a body might fall asleep mid-settle and report
            # exactly zero velocity while still resolving contacts. It cannot:
            # PhysX only sleeps a body that has stayed below threshold for a
            # sustained period, which is the same criterion SettleMonitor
            # applies — a sleeping body genuinely is at rest.
            #
            # What forcing the threshold to zero DID do was stop resting items
            # ever sleeping, so a cylinder spawned exactly touching the table
            # crept for as long as the scene idled. Measured: the identical
            # pre-grasp goal converged when commanded one second after play and
            # timed out when commanded after an operator approval, because by
            # then the target had rolled out of reach.

            self.items[item.item_id] = body
            self.item_specs[item.item_id] = item
            self.item_index[item.item_id] = index
            print(f"{LOG_SCENE} item {item.item_id}: "
                  f"{item.length_mm}x{item.outer_diameter_mm} mm, "
                  f"{item.weight_kg} kg at "
                  f"{tuple(round(v, 3) for v in position)} m")

    def _build_cad_item(self, item: WasteItem, path: str,
                        position: Any, orientation: Any) -> None:
        """Load the item's real CAD mesh into USD, at the given pose.

        THE MESH IS RESOLVED THROUGH THE SHARED REGISTRY, not from a path
        written here: `config/perception_objects.yaml` is the one place CAD
        metadata lives, and it is the same lookup the FoundationPose provider
        uses. Duplicating it in the Isaac adapter would create a second source
        of truth that agrees only until one of them is edited — and the two
        would then disagree about which geometry FoundationPose was matching.

        THE STL IS PARSED HERE AND ONLY HERE. This file is already
        Isaac-specific and already imports `pxr`; the domain layer neither
        imports a simulator nor reads triangles.
        """
        import trimesh                                       # noqa: PLC0415
        from wisepack_core.rgbd import load_object_registry   # noqa: PLC0415

        registry = load_object_registry()
        model = registry.models.get(item.model_id)
        if model is None:
            raise ValueError(
                f"{item.item_id}: no object model {item.model_id!r} in the "
                "registry; a CAD-backed item names a part, it does not define "
                "one")
        mesh_file = model.resolved_path(registry.root)
        if not model.mesh_exists(registry.root):
            raise ValueError(
                f"{item.item_id}: mesh not found at {mesh_file or '(unset)'}")

        mesh = trimesh.load(mesh_file)
        # UNITS COME FROM THE REGISTRY'S DECLARATION, applied once. An STL
        # records no unit, and a millimetre mesh consumed as metres would drop a
        # 342 mm tube into the scene 342 metres long.
        scale = model.mesh_scale_to_mm / 1000.0
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
        # CANONICALISED INTO THE LAYOUT'S CONVENTION, in two steps, and both
        # are recorded so the evaluation path can undo them.
        #
        # 1. CENTRED ON ITS OWN BODY. The STL origin of an obliquely modelled
        #    part can sit far outside the geometry — Cylinder5's is 141 mm away
        #    — so placing the prim by that origin would put the tube somewhere
        #    other than where the layout asked for.
        centre = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
        vertices = vertices - centre

        # 2. TUBE AXIS ROTATED ONTO LOCAL +Z. `quaternion_for_axis` lays a
        #    Z-ALIGNED cylinder along a world axis — that is the convention the
        #    generated path satisfies by construction, because `Cylinder(axes=
        #    "Z")` builds one. A CAD part satisfies nothing by construction:
        #    Cylinder5's tube axis is (0.928, -0.372, 0) in its own file, so
        #    applying the layout's orientation left it lying at ~22 degrees to
        #    the intended axis, and it rolled 130 mm before settling.
        #
        #    The axis comes from the registry, where it was MEASURED, rather
        #    than being recomputed here from the triangles.
        align = _alignment_to_local_z(model)
        vertices = vertices @ align.T
        faces = np.asarray(mesh.faces, dtype=np.int32)

        usd_mesh = UsdGeom.Mesh.Define(stage_utils.get_current_stage(), path)
        usd_mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in vertices])
        usd_mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        usd_mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
        usd_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        # A PHYSICALLY BASED SURFACE, not a flat colour. This was
        # `DisplayColor(0.62, 0.64, 0.67)` — a light grey with no material at
        # all, which rendered a steel tube as a white plastic rod on a light
        # table. Appearance is an INPUT to the FoundationPose model-free method,
        # which builds its object representation from rendered reference
        # imagery, so the surface is declared once in
        # `config/isaac_materials.yaml` and bound here.
        #
        # NOTHING ELSE MOVES: the mesh, its dimensions, its pose, its declared
        # symmetry and its semantics are exactly as they were, and the instance
        # mask comes from those semantics rather than from any pixel colour.
        # The display colour is kept as a fallback for viewers that ignore
        # materials; it is not what the RGB-D camera renders.
        usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.64, 0.67)])
        self._bind_surface(usd_mesh.GetPrim(), model)

        xform = UsdGeom.Xformable(usd_mesh)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))
        # `orientation` is (w, x, y, z), the convention pose_to_world produces.
        xform.AddOrientOp().Set(Gf.Quatf(float(orientation[0]),
                                         Gf.Vec3f(float(orientation[1]),
                                                  float(orientation[2]),
                                                  float(orientation[3]))))

        # SEMANTICS, so Isaac can produce an exact instance mask for this part.
        # Labelled with the model_id, so the mask and the CAD identity carry the
        # same name — and so the evaluation path can never pair a mask with the
        # wrong mesh.
        try:
            from isaacsim.core.utils.semantics import add_labels  # noqa: PLC0415
            add_labels(usd_mesh.GetPrim(), labels=[item.model_id],
                       instance_name="class")
        except Exception as exc:                             # noqa: BLE001
            print(f"{LOG_SCENE} WARNING: no semantics on {item.item_id} "
                  f"({exc}); Isaac instance masks will not identify it")

        # THE COLLISION SHAPE IS DECLARED, not left to a fallback. PhysX cannot
        # use a raw triangle mesh for a DYNAMIC body and silently substitutes a
        # convex hull with an error in the log — which for a tube fills the bore
        # and rounds off the saddle notches. A convex DECOMPOSITION keeps the
        # concave features that matter for resting and grasping.
        from pxr import UsdPhysics                            # noqa: PLC0415
        collision = UsdPhysics.MeshCollisionAPI.Apply(usd_mesh.GetPrim())
        collision.CreateApproximationAttr().Set("convexDecomposition")

        self.cad_mesh_offsets[item.item_id] = {
            "centre_m": [float(v) for v in centre],
            "align_to_local_z": [[float(v) for v in row] for row in align],
        }
        print(f"{LOG_SCENE} item {item.item_id}: CAD {item.model_id} "
              f"({len(faces)} tris) from {mesh_file}")

    def _bind_surface(self, prim, model) -> None:
        """Give a CAD workpiece its declared physically based surface.

        THE PROFILE IS CONFIGURATION, not an argument invented here: the object
        model may name one, and otherwise the registry's declared default
        applies. Reference views and query frames therefore render the same
        surface by construction, which is what makes a model-free experiment
        comparable at all.

        A FAILURE IS REPORTED, NOT SWALLOWED. Rendering a workpiece with no
        material would silently reproduce the flat-grey appearance this replaced,
        and the reference imagery would be wrong in a way nothing downstream
        could detect.
        """
        from .materials import bind, load_materials             # noqa: PLC0415

        registry = load_materials()
        profile = registry.require(getattr(model, "material_profile", ""))
        bind(prim, profile, stage=stage_utils.get_current_stage(),
             material_root=f"{WORLD}/Looks")
        print(f"{LOG_SCENE} {model.model_id}: surface {profile.name} "
              f"(metallic {profile.metallic}, roughness {profile.roughness})")

    def _add_camera(self) -> None:
        """A viewpoint that frames the table, the pick row, the bin and the arm.

        TAKEN FROM THE LAYOUT, not written here, because the layout is what
        moves when a different robot is selected. The xArm 7 works a bin 80 mm
        nearer its base than the Panda does; a camera hard-coded for the Panda
        workcell would frame the same shot around furniture that is no longer
        where it was, and an operator watching the Simulator View would see the
        arm working at the edge of frame.
        """
        stage = stage_utils.get_current_stage(backend="usd")
        camera = UsdGeom.Camera.Define(stage, f"{WORLD}/DemoCamera")
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*self.layout.camera_position_m))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*[
            float(v) for v in self.layout.camera_rotation_deg]))
        camera.CreateFocalLengthAttr(20.0)
        print(f"{LOG_SCENE} DemoCamera at "
              f"{tuple(round(v, 2) for v in self.layout.camera_position_m)} m "
              f"framing the "
              f"{self.layout.robot_id or 'default'} workcell")

    # ------------------------------------------------------------------ #
    # Scene reset
    # ------------------------------------------------------------------ #

    def reset_items(self, scenario: Scenario) -> None:
        """Put the world back to the start of a NEW scenario.

        REMOVES every previous item rather than repositioning it. A new scenario
        can have different ids, different dimensions and a different count, so
        "move the old cylinders back" is not the same world — and any item the
        new scenario does not contain would otherwise be left lying in the
        container, where the packer has no model of it.

        Container CONTENTS are cleared by construction: the containers are
        static geometry and hold nothing of their own, so removing the items
        empties them.
        """
        stage = stage_utils.get_current_stage(backend="usd")
        removed = 0
        for item_id in list(self.items):
            path = item_path(item_id)
            if stage.GetPrimAtPath(path):
                stage.RemovePrim(path)
                removed += 1
        self.items.clear()
        self.item_specs.clear()
        self.item_index.clear()
        print(f"{LOG_SCENE} removed {removed} item(s) from the previous scenario")
        self.build_items(scenario)

    def settle_items(self, updater, frames: int = 90) -> None:
        """Let freshly spawned bodies resolve contact before anything is picked.

        Also zeroes their velocities first: a body created where another one
        used to be can otherwise inherit an impulse and creep away from the
        pose the plan was built against.
        """
        for item_id, body in self.items.items():
            try:
                body.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
            except Exception:                            # noqa: BLE001
                # Velocity reset is best-effort BEFORE the first physics step,
                # where the tensor view may not be populated yet; the settle
                # frames below achieve the same thing.
                pass
        updater(frames)

    # ------------------------------------------------------------------ #
    # Read-back
    # ------------------------------------------------------------------ #

    def has_item(self, item_id: str) -> bool:
        return item_id in self.items

    def item_world_pose(self, item_id: str
                        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(position_m, quaternion_wxyz) of one item, or None if it is gone."""
        body = self.items.get(item_id)
        if body is None:
            return None
        positions, orientations = body.get_world_poses()
        return (np.asarray(positions.numpy()[0], dtype=float),
                np.asarray(orientations.numpy()[0], dtype=float))

    def wake_item(self, item_id: str) -> bool:
        """WAKE A RELEASED BODY. Call this the instant a grasp is let go.

        NOT housekeeping — without it the item does not fall.

        PhysX sleeps a rigid body that has stayed below its velocity thresholds
        for long enough, and a body held by the temporary grasp joint qualifies:
        while the arm holds still at the release pose waiting for the gripper to
        open, the item is rigidly constrained and effectively motionless in the
        solver's terms. Deleting the joint does not wake it. The result is a
        cylinder frozen in mid-air at the release height, with exactly zero
        velocity — which SettleMonitor then reports as "settled" on its first
        check, and the run records a 160 mm placement error for an item that
        never left the gripper's frame.

        Measured exactly that way on the first xArm 7 run: released 10 mm above
        the container rim, settled 10 mm above the container rim, 0.24 s later.
        The Panda had been getting away with it because its fingers slide 40 mm
        apart against the held cylinder and the contact impulses kept the body
        awake — luck, not design, and the same failure was one tuning change
        away from appearing there too.

        Setting the velocities is what wakes it: writing to a sleeping actor's
        velocity wakes it in PhysX. Zero rather than some nudge, because the
        item must fall under GRAVITY from where it was let go — adding an
        impulse here would be this code deciding where the object goes, which is
        the one thing the release must not do.
        """
        body = self.items.get(item_id)
        if body is None:
            return False
        try:
            body.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
            return True
        except Exception as exc:                         # noqa: BLE001
            # Named, never silent. A body that could not be woken is a body that
            # will not fall, and the resulting "placement" would be fiction.
            print(f"{LOG_SCENE} WARNING: could not wake {item_id} on release: "
                  f"{exc!r} — if it does not fall, this is why")
            return False

    def item_velocities(self, item_id: str
                        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(linear_mps, angular_radps) of one item, or None if it is gone."""
        body = self.items.get(item_id)
        if body is None:
            return None
        linear, angular = body.get_velocities()
        return (np.asarray(linear.numpy()[0], dtype=float),
                np.asarray(angular.numpy()[0], dtype=float))

    def summary(self) -> Dict[str, Any]:
        return {
            "items": sorted(self.items),
            "containers": {k: v.to_dict() for k, v in self.containers.items()},
            "table_top_z_m": self.layout.table_top_z_m,
        }


__all__ = [
    "WisepackScene", "item_path", "container_path", "TABLE_PATH",
    "ITEMS_ROOT", "CONTAINERS_ROOT", "WORLD",
]
