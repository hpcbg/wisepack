"""THE ONE PLACE WISEPACK COORDINATES BECOME ISAAC WORLD COORDINATES.

There are exactly two coordinate conventions in play and they disagree about
both units and origin:

    WISEPACK   integer MILLIMETRES. A placement's ``position`` is the MIN CORNER
               of the item's axis-aligned bounding box, expressed in the target
               container's INNER frame (origin at the inner min corner). An
               item's orientation is one of three axis-aligned choices, named by
               which world axis its LENGTH points along.

    Isaac Sim  floating-point METRES, Z up, one shared world frame. A rigid body
               is posed by the CENTRE of its geometry plus a quaternion.

Every conversion between them lives in this module and nowhere else. That is a
hard rule, not a style preference: axis swaps and unit factors sprinkled through
scene code, robot code and validation code are individually plausible and
collectively unfalsifiable, and the failure they produce — a robot placing items
in a mirrored container — looks like a physics problem rather than an arithmetic
one. Concentrating them here means the whole conversion is covered by tests that
need no GPU and no simulator (``tests/test_isaac_backend.py``).

THREE FRAMES, NAMED
-------------------
``world``            Isaac's world. Metres, Z up, robot base at
                     (0, 0, ``table_top_z_m``).
``table``            Where items are picked from. Millimetres, axes parallel to
                     world, origin at ``table_frame_origin_m``. This is the
                     frame the contract's ``source_pose`` uses.
``container:<id>``   Millimetres, axes parallel to world, origin at the
                     container's inner min corner. This is exactly the frame
                     ``domain.Placement.position`` is already expressed in, so a
                     placement needs no reinterpretation — only a corner-to-centre
                     shift and a unit conversion.

WHY THE PICK LAYOUT IS NOT ``WasteItem.source_position``
--------------------------------------------------------
The generator scatters ``source_position`` through a 900x700x250 mm volume: a
loose bin, which is the right abstraction for a perception problem and the wrong
one for a Panda with a 80 mm parallel gripper and no vision. This module lays the
same items out in a deterministic single row on the table instead, derived from
the item's index in the scenario. The layout is ground truth by construction and
is documented as such — see ``simulators/isaac/README.md``. When real perception
arrives it replaces ``table_pose_for_index``, and nothing else changes.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from .domain import Axis, Container, Placement, Vec3, WasteItem
from .isaac_contract import Pose

#: Quaternions are (w, x, y, z) throughout — Isaac's convention for
#: ``set_world_poses``. Written out rather than assumed, because the other
#: common convention (x, y, z, w) produces a plausible-looking wrong rotation.
Quaternion = Tuple[float, float, float, float]

_SQRT_HALF = math.sqrt(0.5)

#: A USD cylinder's axis is Z by default, so the rotation needed to lay the
#: item's length along each world axis is fixed and enumerable. Keeping it as a
#: literal table rather than composing rotations at call sites means the values
#: can be asserted directly in a test.
_AXIS_QUATERNION: Dict[Axis, Quaternion] = {
    #: identity — the cylinder already points along +Z
    Axis.Z: (1.0, 0.0, 0.0, 0.0),
    #: +90 deg about Y maps local +Z onto world +X
    Axis.X: (_SQRT_HALF, 0.0, _SQRT_HALF, 0.0),
    #: -90 deg about X maps local +Z onto world +Y
    Axis.Y: (_SQRT_HALF, -_SQRT_HALF, 0.0, 0.0),
}


def mm_to_m(value: float) -> float:
    return float(value) / 1000.0


def m_to_mm(value: float) -> float:
    return float(value) * 1000.0


def quaternion_for_axis(axis: Axis) -> Quaternion:
    """The world orientation that lays a Z-aligned cylinder along ``axis``."""
    return _AXIS_QUATERNION[Axis(axis)]


def axis_from_quaternion(quaternion: Sequence[float]) -> Axis:
    """Which world axis a cylinder's own axis is closest to, given (w,x,y,z).

    Used on the way BACK: after PhysX has settled a dropped item, its
    orientation is arbitrary, and the honest way to report "which way is it
    lying" is to rotate the body's local +Z into world coordinates and pick the
    nearest world axis. This is a measurement, not a claim that the item landed
    exactly axis-aligned; the residual angle is reported separately by
    ``axis_deviation_deg``.
    """
    w, x, y, z = (float(v) for v in quaternion)
    # Third column of the rotation matrix: R @ (0, 0, 1).
    local_z = (2.0 * (x * z + w * y),
               2.0 * (y * z - w * x),
               1.0 - 2.0 * (x * x + y * y))
    magnitudes = (abs(local_z[0]), abs(local_z[1]), abs(local_z[2]))
    return (Axis.X, Axis.Y, Axis.Z)[magnitudes.index(max(magnitudes))]


def axis_deviation_deg(quaternion: Sequence[float], axis: Axis) -> float:
    """Angle in degrees between the body's own axis and ``axis``, folded to [0, 90].

    Folded because a cylinder is symmetric end-for-end: lying along +X and along
    -X are the same physical placement, and reporting 180 deg of "error" for a
    perfect result would be nonsense.
    """
    w, x, y, z = (float(v) for v in quaternion)
    local_z = (2.0 * (x * z + w * y),
               2.0 * (y * z - w * x),
               1.0 - 2.0 * (x * x + y * y))
    target = {Axis.X: (1.0, 0.0, 0.0), Axis.Y: (0.0, 1.0, 0.0),
              Axis.Z: (0.0, 0.0, 1.0)}[Axis(axis)]
    dot = sum(a * b for a, b in zip(local_z, target))
    return math.degrees(math.acos(min(1.0, abs(dot))))


# --------------------------------------------------------------------------- #
# Scene layout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SceneLayout:
    """Where everything sits in the Isaac world. Metres, Z up.

    The defaults are sized for a Franka Emika Panda (roughly 0.855 m reach from
    its base) working on a table, with the pick row in front of the robot and the
    container to its left. Both regions are inside reach at the heights the
    sequence actually visits — ``validate()`` checks that rather than trusting it,
    because an unreachable target does not fail loudly in Isaac: the IK simply
    converges to the closest achievable pose and the gripper closes on air.
    """

    #: Table top surface. The robot base is mounted ON the table at (0, 0).
    table_top_z_m: float = 0.40
    table_size_m: Tuple[float, float, float] = (1.20, 1.30, 0.40)
    #: Table centre in the world XY plane. Offset forward and left so the table
    #: covers both the pick row and the container without the robot standing on
    #: its edge.
    table_centre_xy_m: Tuple[float, float] = (0.22, 0.06)

    #: Pick row. Items lie with their LENGTH along world X, spaced along Y, so a
    #: top-down gripper whose fingers close along world Y closes across the
    #: cylinder rather than along it.
    pick_row_x_m: float = 0.48
    pick_row_y_start_m: float = -0.32
    pick_row_y_pitch_m: float = 0.11

    #: Outer bottom-left corner of the first container, ON the table top. The
    #: INNER origin — the frame every placement is expressed in — is derived from
    #: this plus the wall thickness, so the two can never be set inconsistently.
    #:
    #: Positioned so the bin's CENTRE sits around 0.46 m from the base — the
    #: middle of the Panda's usable shell. A manipulator workspace is bounded at
    #: both ends: pushing the bin out puts its far corner past the reach limit,
    #: and pulling it in puts its near corner inside the folded-arm region where
    #: the IK has no comfortable solution. validate() checks all four corners
    #: against both bounds rather than assuming which one is worst.
    #: Moved outward from an earlier (0.07, 0.29) after a measured failure: the
    #: pre-place pose sits above the bin at rim + clearance, which for a close-in
    #: bin is HIGH and NEAR — the region where the Panda must fold its elbow and
    #: damped least squares converges slowly. The carry timed out there while the
    #: pick and lift succeeded. Reachability is bounded at both ends and this is
    #: the far end of "too close", not the near end of "too far".
    container_outer_xy_m: Tuple[float, float] = (0.09, 0.32)
    #: Additional containers are laid out along +X at this pitch. The smoke
    #: scenario uses one; a plan that opens a second must still be renderable.
    container_pitch_x_m: float = 0.55
    container_wall_thickness_m: float = 0.008

    #: Robot reach envelope, measured from the base at (0, 0, table_top_z_m).
    #: Conservative: the nominal Panda reach is ~0.855 m and the last few
    #: centimetres are only achievable at unhelpful wrist configurations.
    #:
    #: THESE ARE THE PANDA'S NUMBERS and the defaults above are the Panda's
    #: layout, because it is the arm this workcell was written for. A robot with
    #: a different envelope does not inherit them silently — it declares the
    #: difference in its profile's `workcell` block and `layout_for_robot`
    #: builds its layout from that. See wisepack_core.robots.
    robot_max_reach_m: float = 0.78
    robot_min_reach_m: float = 0.20

    #: The spectator camera (``/World/DemoCamera``). Part of the LAYOUT rather
    #: than of the scene builder, so that a robot which moves the workcell also
    #: moves the viewpoint that frames it — otherwise selecting a shorter arm
    #: silently reframes the shot around furniture that is no longer there.
    camera_position_m: Tuple[float, float, float] = (1.55, -1.15, 1.45)
    camera_rotation_deg: Tuple[float, float, float] = (63.0, 0.0, 52.0)
    #: Which robot this layout was built for. "" means the shared default.
    #: Carried so a layout can be reported and fingerprinted without the caller
    #: having to remember which robot it asked for.
    robot_id: str = ""

    @property
    def table_frame_origin_m(self) -> Tuple[float, float, float]:
        """World position of the ``table`` frame origin.

        The origin is the robot base projected onto the table top, so a table
        pose of (0, 0, 0) mm is "at the robot's feet" and the numbers in a
        ``source_pose`` read naturally against the robot.
        """
        return (0.0, 0.0, self.table_top_z_m)

    def container_outer_origin_for(self, container_id: str
                                   ) -> Tuple[float, float, float]:
        """World position of one container's OUTER bottom-left corner."""
        slot = container_slot(container_id)
        x, y = self.container_outer_xy_m
        return (x + slot * self.container_pitch_x_m, y, self.table_top_z_m)

    def container_origin_for(self, container_id: str) -> Tuple[float, float, float]:
        """World position of one container's INNER min corner.

        This is the origin of the ``container:<id>`` frame — the frame the
        optimizer's placements are already expressed in — so it is the single
        anchor the whole placement conversion hangs off.
        """
        x, y, z = self.container_outer_origin_for(container_id)
        t = self.container_wall_thickness_m
        return (x + t, y + t, z + t)

    def validate(self, container_inner_mm: Vec3, item_count: int,
                 clearance_m: float = 0.15) -> None:
        """Raise if the layout puts anything outside the robot's usable envelope.

        ``clearance_m`` is the height above the container rim the sequence rises
        to before moving laterally, and the container check is done at THAT
        height rather than at table level, where everything trivially fits.

        This is checked rather than assumed because an unreachable target does
        not fail loudly in Isaac. Differential IK converges to the nearest
        achievable pose and the gripper closes on air, which looks like a grasp
        tuning problem and is actually a layout arithmetic problem.
        """
        problems = []
        for index in range(max(1, item_count)):
            x, y = self.pick_xy_m(index)
            reach = math.hypot(x, y)
            if not self.robot_min_reach_m <= reach <= self.robot_max_reach_m:
                problems.append(
                    f"pick slot {index} at ({x:.3f}, {y:.3f}) is {reach:.3f} m "
                    f"from the base, outside "
                    f"[{self.robot_min_reach_m}, {self.robot_max_reach_m}]")

        ox, oy, oz = self.container_origin_for("CNT-01")
        inner_x = mm_to_m(container_inner_mm.x)
        inner_y = mm_to_m(container_inner_mm.y)
        height = oz + mm_to_m(container_inner_mm.z) + clearance_m - self.table_top_z_m
        # All four corners: which one is worst depends on where the bin sits
        # relative to the base, and hard-coding "the far one" was wrong as soon
        # as the bin moved behind the robot's Y axis.
        for cx, cy in ((ox, oy), (ox + inner_x, oy),
                       (ox, oy + inner_y), (ox + inner_x, oy + inner_y)):
            reach = math.sqrt(cx ** 2 + cy ** 2 + height ** 2)
            if reach > self.robot_max_reach_m:
                problems.append(
                    f"container corner ({cx:.3f}, {cy:.3f}) at retreat height is "
                    f"{reach:.3f} m from the base, beyond "
                    f"{self.robot_max_reach_m} m — the arm will not reach it and "
                    "the IK will silently settle short")
            elif math.hypot(cx, cy) < self.robot_min_reach_m:
                problems.append(
                    f"container corner ({cx:.3f}, {cy:.3f}) is only "
                    f"{math.hypot(cx, cy):.3f} m from the base, inside the "
                    f"{self.robot_min_reach_m} m folded-arm region")
        if problems:
            raise ValueError(
                "Isaac scene layout is not reachable:\n  - " + "\n  - ".join(problems))

    # -- pick side ---------------------------------------------------------- #

    def pick_xy_m(self, index: int) -> Tuple[float, float]:
        """World XY of the ``index``-th pick slot."""
        return (self.pick_row_x_m,
                self.pick_row_y_start_m + index * self.pick_row_y_pitch_m)


DEFAULT_LAYOUT = SceneLayout()


def layout_for_robot(profile, base: SceneLayout = DEFAULT_LAYOUT) -> SceneLayout:
    """The workcell as the given robot needs it.

    THE WORKCELL MOVES FOR THE ROBOT. Dropping a shorter arm into coordinates
    sized for a longer one does not fail loudly — differential IK converges to
    the nearest achievable pose and the gripper opens over nothing — so a robot
    whose envelope differs states the difference in its profile and this builds
    the layout from it.

    ``profile`` is a ``wisepack_core.robots.RobotProfile``; it is duck-typed
    rather than imported so this module keeps no dependency on the registry, and
    ``None`` returns the shared default unchanged. Every override the profile
    does NOT set is inherited, so a robot that only differs in reach says only
    that.

    BOTH ENDS MUST CALL THIS WITH THE SAME PROFILE. The orchestrator converts
    plan poses through the layout and the simulator builds the scene from it; if
    they disagreed the arm would be sent to coordinates nobody spawned anything
    at. That is exactly what ``scene_fingerprint`` — which hashes the layout and
    the robot id together — exists to catch.
    """
    if profile is None:
        return base
    overrides = getattr(profile, "workcell", None)
    changes: Dict[str, object] = {"robot_id": getattr(profile, "robot_id", "")}
    for field_name in ("robot_max_reach_m", "robot_min_reach_m",
                       "container_outer_xy_m", "pick_row_x_m",
                       "pick_row_y_start_m", "pick_row_y_pitch_m",
                       "table_top_z_m", "camera_position_m",
                       "camera_rotation_deg"):
        value = getattr(overrides, field_name, None) if overrides else None
        if value is not None:
            changes[field_name] = value
    return dataclasses.replace(base, **changes)


def container_slot(container_id: str) -> int:
    """0-based layout slot for a container id such as ``"CNT-1"``.

    Unparseable ids fall back to slot 0 rather than raising: a container whose id
    does not end in a number is a naming change, not a reason to refuse to draw
    the scene. The ids the generator issues always do.
    """
    suffix = str(container_id).rsplit("-", 1)[-1]
    if suffix.isdigit():
        return max(0, int(suffix) - 1)
    return 0


# --------------------------------------------------------------------------- #
# WISEPACK -> contract poses (millimetres, named frames)
# --------------------------------------------------------------------------- #


def table_pose_for_index(index: int, item: WasteItem,
                         layout: SceneLayout = DEFAULT_LAYOUT) -> Pose:
    """The deterministic pick pose of the ``index``-th item, in the table frame.

    GROUND TRUTH, BY CONSTRUCTION. No perception produced this; the item is
    spawned exactly here and the robot is told exactly here. That is the
    documented limitation of this first iteration (see the module docstring).

    The item lies along world X and rests on the table, so its centre sits one
    radius above the surface.
    """
    x_m, y_m = layout.pick_xy_m(index)
    origin = layout.table_frame_origin_m
    radius_mm = item.outer_diameter_mm / 2.0
    return Pose(
        x_mm=m_to_mm(x_m - origin[0]),
        y_mm=m_to_mm(y_m - origin[1]),
        z_mm=radius_mm,
        axis=Axis.X.value,
        frame="table",
    )


def placement_pose(placement: Placement) -> Pose:
    """A planned placement as a contract pose, in its container's inner frame.

    The only arithmetic here is min-corner -> centre. The frame, the units and
    the axis are already what the domain model uses, which is precisely why the
    contract was defined in millimetres rather than converting at the boundary.
    """
    return Pose(
        x_mm=placement.position.x + placement.size.x / 2.0,
        y_mm=placement.position.y + placement.size.y / 2.0,
        z_mm=placement.position.z + placement.size.z / 2.0,
        axis=placement.axis.value,
        frame=f"container:{placement.container_id}",
    )


@dataclass
class ReleaseClearance:
    """How much room a released object is given, in millimetres.

    All three are configurable because they trade against each other: more wall
    clearance means a released cylinder is less likely to strike a wall on the
    way down, but it also moves the release point further from the planned pose,
    and past a point there is not enough interior left to keep objects apart.
    """

    #: Between the object's footprint and every interior wall.
    wall_mm: float = 25.0
    #: Between this object's footprint and one already placed.
    object_mm: float = 15.0


DEFAULT_RELEASE_CLEARANCE = ReleaseClearance()


def safe_release_pose(planned: Pose, dimensions, container_inner: Vec3,
                      *, clearance: ReleaseClearance = DEFAULT_RELEASE_CLEARANCE,
                      occupied: Optional[Sequence[Tuple[float, float]]] = None
                      ) -> Tuple[Pose, float]:
    """WHERE TO LET GO, given where the plan wants the object to end up.

    Returns (release pose, how far it was moved in mm).

    THE PROBLEM THIS SOLVES. A good packing plan puts items *flush* against the
    container walls: that is what makes it dense, and it is the whole point of
    the optimizer. But a flush target is a bad place to release an object from.
    The gripper cannot descend to the planned depth without scraping the wall,
    so the release happens above the rim and the object falls the rest of the
    way. Falling beside a wall it can clip the rim, rotate, and end up somewhere
    neither the plan nor the physics intended. In the first measured run the two
    largest errors, 67 mm and 60 mm, came with 42 and 36 degrees of axis error,
    which is what a wall strike looks like.

    So the release point is CLAMPED into the interior region where the object's
    own footprint clears every wall, and nudged away from footprints already
    occupied. An already-interior target is left exactly where it is: this only
    moves a release that would otherwise happen against a wall.

    It does NOT move the object after release, and it does not change the plan.
    The planned pose remains the reference the final settled pose is measured
    against, so this can only be judged by whether the measured error improves.
    """
    half_len = float(dimensions.length_mm) / 2.0
    radius = float(dimensions.outer_diameter_mm) / 2.0
    axis = str(getattr(planned, "axis", "x")).lower()

    # The footprint half-extents in container X/Y, from the cylinder's axis.
    if axis == "x":
        half_x, half_y = half_len, radius
    elif axis == "y":
        half_x, half_y = radius, half_len
    else:                                   # upright: a circle either way
        half_x, half_y = radius, radius

    def _clamp(value: float, half: float, span: float) -> float:
        low = half + clearance.wall_mm
        high = span - half - clearance.wall_mm
        if low > high:                      # too big to clear both walls: centre
            return span / 2.0
        return min(max(value, low), high)

    x = _clamp(planned.x_mm, half_x, float(container_inner.x))
    y = _clamp(planned.y_mm, half_y, float(container_inner.y))

    # Keep away from what is already down there. One pass, smallest push that
    # separates: a released object landing on another one is the other way a
    # placement goes wrong.
    for other_x, other_y in (occupied or ()):
        dx, dy = x - other_x, y - other_y
        need = clearance.object_mm
        if abs(dx) < (2 * half_x + need) and abs(dy) < (2 * half_y + need):
            if abs(dx) >= abs(dy):
                x = _clamp(other_x + math.copysign(2 * half_x + need, dx or 1.0),
                           half_x, float(container_inner.x))
            else:
                y = _clamp(other_y + math.copysign(2 * half_y + need, dy or 1.0),
                           half_y, float(container_inner.y))

    moved = math.hypot(x - planned.x_mm, y - planned.y_mm)
    return Pose(x_mm=x, y_mm=y, z_mm=planned.z_mm, axis=planned.axis,
                frame=planned.frame), moved


def dimensions_for(item: WasteItem):
    """The contract ``Dimensions`` for a domain item."""
    from .isaac_contract import Dimensions            # local: keeps the cycle out
    return Dimensions(
        length_mm=item.length_mm,
        outer_diameter_mm=item.outer_diameter_mm,
        inner_diameter_mm=item.inner_diameter_mm,
    )


# --------------------------------------------------------------------------- #
# Contract poses -> Isaac world (metres)
# --------------------------------------------------------------------------- #


def pose_to_world(pose: Pose, layout: SceneLayout = DEFAULT_LAYOUT
                  ) -> Tuple[Tuple[float, float, float], Quaternion]:
    """Convert any contract pose into an Isaac world (position_m, quaternion).

    Dispatches on the pose's declared frame. An unknown frame raises: guessing
    would place a physical object somewhere nobody specified.
    """
    if pose.frame == "table":
        origin = layout.table_frame_origin_m
    elif pose.frame.startswith("container:"):
        origin = layout.container_origin_for(pose.frame.split(":", 1)[1])
    else:
        raise ValueError(
            f"unknown pose frame {pose.frame!r}; expected 'table' or "
            "'container:<id>'")
    position = (origin[0] + mm_to_m(pose.x_mm),
                origin[1] + mm_to_m(pose.y_mm),
                origin[2] + mm_to_m(pose.z_mm))
    return position, quaternion_for_axis(Axis(pose.axis))


def world_to_pose(position_m: Sequence[float], quaternion: Sequence[float],
                  frame: str, layout: SceneLayout = DEFAULT_LAYOUT) -> Pose:
    """The exact inverse of :func:`pose_to_world`, for reporting a MEASURED pose.

    This is what turns "where PhysX left the cylinder" back into the frame the
    optimizer planned in, so target and outcome can be subtracted meaningfully.
    """
    if frame == "table":
        origin = layout.table_frame_origin_m
    elif frame.startswith("container:"):
        origin = layout.container_origin_for(frame.split(":", 1)[1])
    else:
        raise ValueError(
            f"unknown pose frame {frame!r}; expected 'table' or 'container:<id>'")
    return Pose(
        x_mm=m_to_mm(float(position_m[0]) - origin[0]),
        y_mm=m_to_mm(float(position_m[1]) - origin[1]),
        z_mm=m_to_mm(float(position_m[2]) - origin[2]),
        axis=axis_from_quaternion(quaternion).value,
        frame=frame,
    )


def container_frame(container_id: str) -> str:
    return f"container:{container_id}"


# --------------------------------------------------------------------------- #
# Containment check (frame-local, so it is testable without Isaac)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ContainmentVerdict:
    """Did the settled item end up where a container can hold it?"""

    inside_footprint: bool
    above_floor: bool
    below_rim_overflow: bool
    in_scene: bool
    detail: str = ""

    @property
    def ok(self) -> bool:
        return (self.inside_footprint and self.above_floor
                and self.below_rim_overflow and self.in_scene)


def check_containment(actual: Pose, container_inner: Vec3,
                      dimensions_length_mm: float,
                      diameter_mm: float,
                      footprint_tolerance_mm: float = 30.0,
                      overflow_allowance_mm: float = 250.0) -> ContainmentVerdict:
    """Verify a settled item against its container, in the container's own frame.

    Deliberately generous and deliberately explicit about being so. This is a
    check that the object is IN THE BOX, not that it landed on the millimetre:
    a released cylinder rolls, and a first-iteration drop that ends 20 mm from
    the planned centre is a success, not a failure. The pose ERROR is reported
    separately and is never rounded away (see ``IsaacFeedback``).

    ``footprint_tolerance_mm`` allows the item's centre to sit slightly outside
    the strict inner footprint — an item leaning on a wall genuinely is in the
    container. ``overflow_allowance_mm`` catches the opposite failure: an item
    perched ON the rim or on top of the pile rather than inside.
    """
    half_length = max(dimensions_length_mm, diameter_mm) / 2.0
    reasons = []

    inside = (-footprint_tolerance_mm <= actual.x_mm
              <= container_inner.x + footprint_tolerance_mm
              and -footprint_tolerance_mm <= actual.y_mm
              <= container_inner.y + footprint_tolerance_mm)
    if not inside:
        reasons.append(
            f"centre ({actual.x_mm:.0f}, {actual.y_mm:.0f}) mm is outside the "
            f"{container_inner.x}x{container_inner.y} mm footprint "
            f"(+/-{footprint_tolerance_mm:.0f} mm tolerance)")

    # The floor is z = 0 in the container frame. A centre below its own radius
    # means the body is intersecting or has fallen through the floor.
    above_floor = actual.z_mm >= diameter_mm / 2.0 - footprint_tolerance_mm
    if not above_floor:
        reasons.append(
            f"centre z {actual.z_mm:.0f} mm is below the container floor")

    below_overflow = actual.z_mm <= container_inner.z + overflow_allowance_mm
    if not below_overflow:
        reasons.append(
            f"centre z {actual.z_mm:.0f} mm is more than "
            f"{overflow_allowance_mm:.0f} mm above the {container_inner.z} mm "
            "rim — the item is resting on top, not inside")

    # A body that fell out of the world entirely: PhysX reports enormous
    # coordinates rather than deleting it, so an explicit sanity bound is the
    # only way to distinguish "gone" from "badly placed".
    in_scene = (abs(actual.x_mm) < 1e5 and abs(actual.y_mm) < 1e5
                and -1e4 < actual.z_mm < 1e5 and half_length > 0)
    if not in_scene:
        reasons.append("item is not in the scene any more")

    return ContainmentVerdict(
        inside_footprint=inside, above_floor=above_floor,
        below_rim_overflow=below_overflow, in_scene=in_scene,
        detail="; ".join(reasons))


def scene_fingerprint(scenario, layout: SceneLayout = DEFAULT_LAYOUT,
                      robot_id: Optional[str] = None) -> str:
    """A deterministic digest of the PHYSICAL scene a scenario implies.

    Computed on BOTH sides from the same function, so "the simulator built what
    this run planned" becomes a string comparison instead of an inference. The
    orchestrator computes it from the scenario it planned; Isaac computes it from
    the scenario it built; SCENE_READY carries Isaac's.

    WHAT IT COVERS, and why each part is load-bearing:

      * preset and seed — the generator inputs. Same preset, different seed is a
        different set of objects wearing the same names.
      * object ids — which items exist.
      * dimensions — an id can survive a regeneration while the geometry under
        it changes, and the plan is written against the geometry.
      * initial source poses — WHERE the arm will reach for each item. Two scenes
        with identical ids and dimensions but a different pick row are different
        worlds to a robot.
      * the container specification — where things are put, and how much fits.
      * THE ROBOT. A scene built for an xArm 7 is not the scene an approved
        Panda plan was written against: the bin sits 80 mm nearer the base, the
        reach envelope the plan was validated against is different, and the arm
        standing in it is a different machine. Without this facet a Panda
        SCENE_READY would satisfy an xArm run's gate — an acknowledgement about
        one world authorising motion in another. ``robot_id`` defaults to
        ``layout.robot_id`` so a caller that already built the layout for a
        robot cannot forget to pass it.

    What it deliberately does NOT cover: anything that legitimately differs
    between two correct builds of the same scenario (timestamps, prim paths,
    material names, camera placement). A fingerprint that changes when nothing
    physical changed would reject correct scenes, which is worse than useless —
    it would train an operator to ignore the check.

    Rounded to micrometres before hashing: floating-point recomputation of the
    same pose can differ in the last bits, and a digest that flips on that would
    be non-deterministic in exactly the way this exists to prevent.
    """
    import hashlib                                        # noqa: PLC0415
    import json as _json                                  # noqa: PLC0415

    def _round(value: float) -> float:
        return round(float(value), 6)

    parts: Dict[str, object] = {
        "preset": getattr(scenario, "preset", ""),
        "seed": int(getattr(scenario, "seed", 0) or 0),
        "robot": str(layout.robot_id if robot_id is None else robot_id),
        "layout": [_round(layout.table_top_z_m), _round(layout.pick_row_x_m),
                   [_round(v) for v in layout.container_outer_xy_m]],
        "items": [],
        "container": None,
    }
    for index, item in enumerate(getattr(scenario, "items", []) or []):
        pose = table_pose_for_index(index, item, layout)
        position, orientation = pose_to_world(pose, layout)
        parts["items"].append({
            "id": item.item_id,
            "length_mm": _round(item.length_mm),
            "outer_diameter_mm": _round(item.outer_diameter_mm),
            "inner_diameter_mm": _round(getattr(item, "inner_diameter_mm", 0.0) or 0.0),
            "position_m": [_round(v) for v in position],
            "orientation": [_round(v) for v in orientation],
        })
    template = getattr(scenario, "container_template", None)
    if template is not None:
        inner = template.inner_size
        parts["container"] = {
            "inner_mm": [_round(inner.x), _round(inner.y), _round(inner.z)],
            "max_payload_kg": _round(getattr(template, "max_payload_kg", 0.0) or 0.0),
        }
    blob = _json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def container_inner_size(container: Optional[Container]) -> Vec3:
    return container.inner_size if container else Vec3(0, 0, 0)


__all__ = [
    "Quaternion", "SceneLayout", "DEFAULT_LAYOUT", "ContainmentVerdict",
    "mm_to_m", "m_to_mm", "quaternion_for_axis", "axis_from_quaternion",
    "axis_deviation_deg", "container_slot", "table_pose_for_index",
    "placement_pose", "dimensions_for", "pose_to_world", "world_to_pose",
    "container_frame", "check_containment", "container_inner_size",
    "layout_for_robot", "scene_fingerprint",
]
