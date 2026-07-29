"""The pick-and-place sequence, as an explicit state machine. ROBOT-NEUTRAL.

    HOME -> PRE_GRASP -> GRASP -> ATTACH -> LIFT -> PRE_PLACE
         -> PLACE_ORIENTATION -> RELEASE -> DETACH -> RETREAT
         -> WAIT_FOR_SETTLE -> NEXT_ITEM

CONSERVATIVE ON PURPOSE. Every motion is a single Cartesian servo goal with a
convergence tolerance and a frame budget, and the states are traversed in a fixed
order with no re-planning, no blending and no shortcuts. That makes the whole
sequence readable in one sitting and makes a failure attributable to a named
state — which is worth more in a first integration than a smoother trajectory.

NOTHING HERE KNOWS WHICH ARM IT IS DRIVING
------------------------------------------
Every robot-specific fact — the asset, the prim paths, the joint names and
order, the home configuration, the gripper travel, the tool-centre-point offset,
the reach envelope — reaches this file through an ``IsaacRobotAdapter`` and its
profile. There is no branch on robot identity anywhere below, and there must not
be one: a robot-specific condition in a state machine is how two arms become two
state machines.

CONTROLLER
----------
Whatever the selected robot's profile names, which today is damped least-squares
differential IK over the articulation Jacobian for both supported arms — see
``adapters/kinematics.py``.

Differential IK is ITERATIVE: one call moves the end effector a fraction of the
way to the goal. So every state calls it once per physics frame and watches for
convergence, rather than commanding a pose and assuming arrival. It is NOT a
motion planner: there is no collision model and no trajectory, and the one
clearance rule below is enforced by choosing waypoints, not by planning.

THE ONE RULE THE PATH MUST NOT BREAK
------------------------------------
A held item is never moved laterally below the container rim. Every lateral
motion happens at ``rim_z + container_clearance``, and the descent to the drop
height is purely vertical at the target XY. Dragging a cylinder through a wall
would be resolved by PhysX as a spectacular impulse, or — worse, and more likely
— as a quiet penetration that still ends with the item "in" the bin.

MoveIt is deliberately not used. The repository has no existing MoveIt
integration to reuse, and introducing one to move a gripper between four known
poses would add a planning stack, a URDF pipeline and a second source of truth
for the robot model, in exchange for capability this sequence does not need.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from wisepack_core.domain import Axis, Vec3
from wisepack_core.isaac_contract import IsaacCommand, IsaacState, Pose
from wisepack_core.isaac_transform import (
    SceneLayout, mm_to_m, pose_to_world, world_to_pose, safe_release_pose,
)

from .config import LOG_ROBOT, MotionConfig
from .result import PlacementOutcome, SettleMonitor, evaluate_placement
from .scene import WisepackScene, item_path


class SequenceState(str, Enum):
    """The item sequence. Values are the log/telemetry names."""

    IDLE = "IDLE"
    HOME = "HOME"
    PRE_GRASP = "PRE_GRASP"
    GRASP = "GRASP"
    ATTACH = "ATTACH"
    LIFT = "LIFT"
    PRE_PLACE = "PRE_PLACE"
    PLACE_ORIENTATION = "PLACE_ORIENTATION"
    RELEASE = "RELEASE"
    DETACH = "DETACH"
    RETREAT = "RETREAT"
    WAIT_FOR_SETTLE = "WAIT_FOR_SETTLE"
    NEXT_ITEM = "NEXT_ITEM"
    FAILED = "FAILED"


#: Which physical feedback state each sequence state reports to WISEPACK. Several
#: sequence states share one reported state — ATTACH is part of grasping, DETACH
#: part of releasing — because the contract describes the PHYSICAL phase, not
#: this implementation's internal steps.
_REPORTED: Dict[SequenceState, Optional[IsaacState]] = {
    SequenceState.HOME: None,
    SequenceState.PRE_GRASP: IsaacState.MOVING_TO_PICK,
    SequenceState.GRASP: IsaacState.GRASPING,
    SequenceState.ATTACH: None,                       # still GRASPING
    SequenceState.LIFT: IsaacState.LIFTING,
    SequenceState.PRE_PLACE: IsaacState.MOVING_TO_CONTAINER,
    SequenceState.PLACE_ORIENTATION: None,            # still MOVING_TO_CONTAINER
    SequenceState.RELEASE: IsaacState.RELEASING,
    SequenceState.DETACH: None,                       # still RELEASING
    SequenceState.RETREAT: None,
    SequenceState.WAIT_FOR_SETTLE: IsaacState.SETTLING,
    SequenceState.NEXT_ITEM: None,
}

#: Quaternion for "gripper pointing straight down", (w, x, y, z). 180 deg about
#: X: the hand's approach axis maps onto world -Z.
_DOWN = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)


def down_orientation(yaw_deg: float) -> np.ndarray:
    """Top-down gripper orientation, rotated ``yaw_deg`` about world Z.

    Yaw is what aims the finger closing direction across the cylinder's axis, so
    it is the only orientation degree of freedom this sequence uses.
    """
    half = math.radians(yaw_deg) / 2.0
    yaw = np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)
    w0, x0, y0, z0 = yaw
    w1, x1, y1, z1 = _DOWN
    return np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ], dtype=float)


class PlacementSequence:
    """Executes one EXECUTE_ITEM command, one physics frame at a time.

    Callbacks rather than return values, because a state machine advancing inside
    a render loop has no natural place to return a result to:

        ``on_state(IsaacState, message, detail)``   physical progress
        ``on_done(item_id, PlacementOutcome)``      the item is finished
    """

    def __init__(self, scene: WisepackScene, layout: SceneLayout,
                 motion: MotionConfig, physics_dt: float,
                 on_state: Callable[[IsaacState, str, Dict[str, Any]], None],
                 on_done: Callable[[str, PlacementOutcome], None]) -> None:
        self.scene = scene
        self.layout = layout
        self.motion = motion
        self.physics_dt = physics_dt
        self.on_state = on_state
        self.on_done = on_done

        #: The IsaacRobotAdapter, set by attach_robot(). The ONLY channel
        #: through which this sequence touches a robot.
        self.robot = None
        self.state = SequenceState.IDLE
        self.command: Optional[IsaacCommand] = None
        self.sim_time = 0.0

        self._frames_in_state = 0
        self._dwell = 0
        self._goal_position: Optional[np.ndarray] = None
        self._goal_orientation: Optional[np.ndarray] = None
        self._settle: Optional[SettleMonitor] = None
        self._failure: str = ""
        self._axis_approximated = False
        #: CLEARANCE-AWARE RELEASE. Where the arm actually lets go, which is not
        #: always where the plan wants the item to end up: a dense plan puts
        #: items flush against the walls, and releasing there lets the falling
        #: object clip the rim. Computed per item in `begin()`.
        self._release_pose: Optional[Pose] = None
        self._release_offset_mm: float = 0.0
        #: Release footprints already used in this run, so two objects are not
        #: dropped onto the same spot.
        self._released_xy: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------ #
    # Wiring
    # ------------------------------------------------------------------ #

    def reset_release_history(self) -> None:
        """Forget where previous items were released. Called for a new run."""
        self._released_xy = []

    def attach_robot(self, robot) -> None:
        """Bind the IsaacRobotAdapter this sequence drives."""
        self.robot = robot

    @property
    def busy(self) -> bool:
        return self.state not in (SequenceState.IDLE, SequenceState.NEXT_ITEM,
                                  SequenceState.FAILED)

    @property
    def holding(self) -> Optional[str]:
        """The item currently welded to the hand, or None. Adapter-owned."""
        return None if self.robot is None else self.robot.holding

    def _release(self) -> None:
        """Drop whatever is held. Safe before a robot has been attached."""
        if self.robot is not None:
            self.robot.release_object()

    @property
    def tcp_offset_m(self) -> float:
        """Fingertip standoff of the SELECTED robot. 0.0 before one is attached."""
        return 0.0 if self.robot is None else self.robot.tool_centre_point_m

    # ------------------------------------------------------------------ #
    # Geometry for the current item
    # ------------------------------------------------------------------ #

    def _grasp_yaw(self) -> float:
        """Yaw that aims the fingers ACROSS the cylinder, for THIS robot.

        A property of the shipped asset's tool frame, so it comes from the robot
        profile. The environment override in MotionConfig still wins when it is
        set to something other than its default, because correcting a finger
        axis at the point of use must not require editing a config file.
        """
        if self.motion.grasp_yaw_offset_deg:
            return self.motion.grasp_yaw_offset_deg
        return 0.0 if self.robot is None else self.robot.grasp_yaw_offset_deg

    def _place_yaw(self) -> Tuple[float, bool]:
        """(yaw degrees, axis_was_approximated) for the planned target axis.

        The item is picked lying along world X, so a target axis of X needs no
        extra yaw and Y needs a quarter turn.

        A target axis of Z cannot be reached by a top-down parallel gripper
        without a regrasp, which this iteration does not implement. Rather than
        fail the item, it is placed along X and the deviation is REPORTED — the
        measured axis error will be ~90 deg and will say so. The Isaac smoke
        preset restricts its items to the X/Y axes precisely so this path is not
        exercised by the supported scenario.
        """
        assert self.command is not None and self.command.target_pose is not None
        axis = Axis(self.command.target_pose.axis)
        if axis is Axis.Y:
            return self._grasp_yaw() + 90.0, False
        if axis is Axis.Z:
            return self._grasp_yaw(), True
        return self._grasp_yaw(), False

    def _container_rim_z(self) -> float:
        assert self.command is not None
        inner = self.command.container_inner_mm or {}
        origin = self.layout.container_origin_for(self.command.container_id or "CNT-01")
        return origin[2] + mm_to_m(float(inner.get("z", 0)))

    def _item_radius(self) -> float:
        assert self.command is not None and self.command.dimensions is not None
        return mm_to_m(self.command.dimensions.outer_diameter_mm) / 2.0

    def _source_world(self) -> np.ndarray:
        assert self.command is not None and self.command.source_pose is not None
        position, _ = pose_to_world(self.command.source_pose, self.layout)
        return np.array(position, dtype=float)

    def _target_world(self) -> np.ndarray:
        """WHERE THE ARM GOES, which is not always where the plan wants the item.

        A dense plan puts items flush against the container walls. Releasing
        there means the object falls beside a wall and can clip the rim, rotate
        and land somewhere nobody planned. So the arm is sent to a
        clearance-aware release point instead, computed once per item in
        `begin()`.

        The PLAN is untouched: `command.target_pose` is still what the settled
        pose is measured against in `_finish_item`, so this can only be judged
        by whether the measured error improves, never by moving the goalposts.
        """
        assert self.command is not None and self.command.target_pose is not None
        pose = self._release_pose or self.command.target_pose
        position, _ = pose_to_world(pose, self.layout)
        return np.array(position, dtype=float)

    def _live_item_world(self) -> Optional[np.ndarray]:
        """Where the item actually is right now, rather than where it was spawned.

        Matters because a neighbouring pick can nudge an item before its turn
        comes. Grasping the commanded pose regardless would close the fingers
        next to it.
        """
        assert self.command is not None
        pose = self.scene.item_world_pose(self.command.item_id or "")
        return None if pose is None else pose[0]

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def begin(self, command: IsaacCommand) -> bool:
        """Start executing one item. False if the scene has no such object."""
        if not self.scene.has_item(command.item_id or ""):
            self._failure = (
                f"{command.item_id} is not in the Isaac scene — it was not part "
                f"of preset/seed the scene was built from")
            print(f"{LOG_ROBOT} cannot start {command.item_id}: {self._failure}")
            return False
        self.command = command
        self._axis_approximated = False
        self._failure = ""
        self._release()
        self._release_pose = None
        self._release_offset_mm = 0.0
        inner = command.container_inner_mm or {}
        if (command.target_pose is not None and command.dimensions is not None
                and inner.get("x") and inner.get("y")):
            self._release_pose, self._release_offset_mm = safe_release_pose(
                command.target_pose, command.dimensions,
                Vec3(int(inner["x"]), int(inner["y"]), int(inner.get("z", 0))),
                clearance=self.motion.release_clearance,
                occupied=list(self._released_xy))
            self._released_xy.append(
                (self._release_pose.x_mm, self._release_pose.y_mm))
        self._enter(SequenceState.HOME)
        moved = (f", release moved {self._release_offset_mm:.0f} mm inward for "
                 "wall clearance" if self._release_offset_mm > 0.5 else "")
        print(f"{LOG_ROBOT} item {command.item_id} #{command.sequence_index}: "
              f"pick {self._source_world().round(3)} -> place "
              f"{self._target_world().round(3)} (axis "
              f"{command.target_pose.axis if command.target_pose else '?'}{moved})")
        return True

    def abort(self, reason: str) -> None:
        """Stop immediately, dropping whatever is held."""
        if self.state is SequenceState.IDLE:
            return
        print(f"{LOG_ROBOT} aborting {self.command.item_id if self.command else '-'}"
              f": {reason}")
        self._release()
        if self.robot is not None:
            self.robot.open_gripper()
        self.state = SequenceState.IDLE
        self.command = None

    def _enter(self, state: SequenceState) -> None:
        self.state = state
        self._frames_in_state = 0
        self._dwell = 0
        reported = _REPORTED.get(state)
        if reported is not None and self.command is not None:
            self.on_state(reported, f"{state.value}", {"sequence_state": state.value})

    def _fail(self, reason: str) -> None:
        """End the item as a physical failure, naming the state it died in."""
        item = self.command.item_id if self.command else "-"
        self._failure = f"{self.state.value}: {reason}"
        print(f"{LOG_ROBOT} FAILED {item} in {self.state.value}: {reason}")
        self._release()
        if self.robot is not None:
            self.robot.open_gripper()
        outcome = PlacementOutcome(
            ok=False, reasons=[self._failure], notes=[], actual_pose=None,
            target_pose=self.command.target_pose if self.command else None,
            position_error_mm=None, axis_error_deg=None, settled=False,
            timed_out=False,
            detail={"sequence_state": self.state.value, "reason": reason})
        self.state = SequenceState.FAILED
        if self.command is not None:
            self.on_done(self.command.item_id or "", outcome)
        self.command = None
        self.state = SequenceState.IDLE

    # ------------------------------------------------------------------ #
    # Servo helpers
    # ------------------------------------------------------------------ #

    def _servo(self, tcp_position: np.ndarray, yaw_deg: float) -> bool:
        """Drive the end effector one IK step towards a TCP goal. True if reached.

        ``tcp_position`` is where the FINGERTIPS should be. Every adapter servos
        its end-effector LINK, so the goal is offset along the approach axis by
        that robot's tool-centre-point distance — 0.103 m for the Panda's hand
        frame, 0.162 m for the xArm gripper's base link. Getting it wrong is not
        subtle in its effect and is very easy to miss in code: a 70 mm error put
        every grasp descent that far above the object, on every item. Hence one
        helper, used by every state, reading one number from the profile.
        """
        assert self.robot is not None
        goal = (np.array(tcp_position, dtype=float)
                + np.array([0.0, 0.0, self.robot.tool_centre_point_m]))
        orientation = down_orientation(yaw_deg)
        self._goal_position, self._goal_orientation = goal, orientation
        self.robot.command_tcp_pose(position=goal, orientation=orientation)

        current, _ = self.robot.get_tcp_pose()
        error = float(np.linalg.norm(np.asarray(current, dtype=float) - goal))
        if error <= self.motion.goal_tolerance:
            self._dwell += 1
        else:
            # Restart the dwell: a single in-tolerance frame can be the arm
            # passing through the goal rather than arriving at it.
            self._dwell = 0
        return self._dwell >= self.motion.dwell_frames

    def _budget_exceeded(self) -> bool:
        return self._frames_in_state > self.motion.max_frames_per_goal

    # ------------------------------------------------------------------ #
    # The step
    # ------------------------------------------------------------------ #

    def step(self) -> None:
        """Advance one physics frame. Safe to call when idle."""
        self.sim_time += self.physics_dt
        if self.state in (SequenceState.IDLE, SequenceState.NEXT_ITEM,
                          SequenceState.FAILED):
            return
        if self.command is None or self.robot is None:  # pragma: no cover
            return
        self._frames_in_state += 1
        handler = getattr(self, f"_step_{self.state.value.lower()}")
        handler()

    # -- states ------------------------------------------------------------- #

    def _step_home(self) -> None:
        """Rise to a neutral height above the item before approaching it.

        Not a joint-space reset: teleporting the arm home between items would
        also teleport a held object, and would look nothing like a robot.
        """
        live = self._live_item_world()
        if live is None:
            self._fail("the item disappeared from the scene before the approach")
            return
        goal = np.array([live[0], live[1],
                         self.layout.table_top_z_m + self.motion.lift_height])
        self.robot.open_gripper()
        if self._servo(goal, self._grasp_yaw()) or self._budget_exceeded():
            self._enter(SequenceState.PRE_GRASP)

    def _step_pre_grasp(self) -> None:
        live = self._live_item_world()
        if live is None:
            self._fail("the item disappeared from the scene before the approach")
            return
        goal = np.array([live[0], live[1], live[2] + self.motion.pre_grasp_height])
        if self._servo(goal, self._grasp_yaw()):
            self._enter(SequenceState.GRASP)
        elif self._budget_exceeded():
            # Report WHERE it got to and WHERE the item actually is. "Could not
            # reach the pre-grasp pose" alone is unactionable — it cannot
            # distinguish an unreachable goal from an item that has rolled away
            # from the pose the plan was built against, and those need opposite
            # fixes.
            reached = np.asarray(self.robot.get_tcp_pose()[0], dtype=float)
            commanded = self._source_world()
            self._fail(
                f"could not reach the pre-grasp pose within "
                f"{self.motion.max_frames_per_goal} frames: goal "
                f"{np.round(goal, 3).tolist()}, hand reached "
                f"{np.round(reached, 3).tolist()}, item now "
                f"{np.round(live, 3).tolist()} (commanded from "
                f"{np.round(commanded, 3).tolist()})")

    def _step_grasp(self) -> None:
        """Descend onto the item, then close the fingers.

        The descent and the close are one state because they are one physical
        act; splitting them would add a state whose only job is to wait.
        """
        live = self._live_item_world()
        if live is None:
            self._fail("the item disappeared during the grasp")
            return
        descended = self._servo(np.array(live, dtype=float), self._grasp_yaw())
        if descended or self._frames_in_state > self.motion.max_frames_per_goal // 2:
            self.robot.close_gripper()
            if self._frames_in_state > self.motion.gripper_frames:
                self._enter(SequenceState.ATTACH)
        if self._budget_exceeded():
            self._fail(f"could not reach the grasp pose within "
                       f"{self.motion.max_frames_per_goal} frames")

    def _step_attach(self) -> None:
        """Weld the item to the hand. See grasp.py — this is an approximation."""
        item = self.command.item_id or ""
        item_pose = self.scene.item_world_pose(item)
        if item_pose is None:
            self._fail("the item disappeared before it could be attached")
            return
        # The adapter owns the weld because the frame the item is welded to is a
        # robot-specific prim, and this file must not know one arm's link names.
        self.robot.attach_object(
            item_path=item_path(item), item_id=item,
            item_position=item_pose[0], item_orientation=item_pose[1])
        self._enter(SequenceState.LIFT)

    def _step_lift(self) -> None:
        """Raise vertically before any lateral motion, clearing the pick row."""
        source = self._source_world()
        goal = np.array([source[0], source[1],
                         self.layout.table_top_z_m + self.motion.lift_height])
        if self._servo(goal, self._grasp_yaw()) or self._budget_exceeded():
            self._enter(SequenceState.PRE_PLACE)

    def _step_pre_place(self) -> None:
        """Move above the target, staying clear of the rim throughout.

        The lateral move happens at rim + clearance, which is the rule that keeps
        the held cylinder from being dragged through a container wall.
        """
        target = self._target_world()
        safe_z = self._container_rim_z() + self.motion.container_clearance
        goal = np.array([target[0], target[1], safe_z])
        if self._servo(goal, self._grasp_yaw()):
            self._enter(SequenceState.PLACE_ORIENTATION)
        elif self._budget_exceeded():
            self._fail("could not reach the pre-place pose above the container")

    def _step_place_orientation(self) -> None:
        """Turn the held item to the planned axis, then descend to drop height.

        The turn happens ABOVE the rim and the descent is purely vertical at the
        target XY, so the item never sweeps sideways inside the container.
        """
        yaw, approximated = self._place_yaw()
        self._axis_approximated = approximated
        target = self._target_world()
        rim_clear_z = self._container_rim_z() + self.motion.container_clearance

        # Two phases in one state, and the order is the safety property: turn
        # while still above the rim, and only then descend straight down. The
        # turn is gated on a frame count rather than a pose check because a yaw
        # error does not appear in the servo's POSITION tolerance at all — the
        # arm would report "arrived" with the item still crosswise.
        if self._frames_in_state <= self.motion.dwell_frames * 4:
            self._servo(np.array([target[0], target[1], rim_clear_z]), yaw)
            return

        # LOWER ONLY WHILE THE WHOLE OBJECT IS CLEAR OF THE WALLS.
        #
        # This clamp is the rule, not an optimisation. A good packing plan puts
        # items FLUSH against the container walls — that is what makes it a good
        # plan — so a placement's footprint frequently touches a wall exactly.
        # Descending into the bin with an item in that position scrapes the wall,
        # and PhysX resolves the contact by refusing to move the arm: measured as
        # "could not reach the release pose inside the container" after the pick,
        # lift and carry had all succeeded.
        #
        # So the descent stops at the rim plus the item's own radius. The item is
        # released from there and falls the rest of the way. That trades
        # placement accuracy for a placement that is physically possible at all,
        # and the resulting error is MEASURED and reported per item rather than
        # hidden — see result.evaluate_placement.
        #
        # Doing this properly in the next iteration means clearance-aware
        # placement: the optimizer would leave a few millimetres beside each wall
        # for the gripper, at a small cost in density.
        rim_z = self._container_rim_z()
        goal_z = max(target[2] + self.motion.drop_height,
                     rim_z + self._item_radius())
        if self._servo(np.array([target[0], target[1], goal_z]), yaw):
            self._enter(SequenceState.RELEASE)
        elif self._budget_exceeded():
            self._fail("could not reach the release pose inside the container")

    def _step_release(self) -> None:
        """Open the fingers. The item is still welded — DETACH does the drop."""
        self.robot.open_gripper()
        if self._frames_in_state >= self.motion.gripper_frames:
            self._enter(SequenceState.DETACH)

    def _step_detach(self) -> None:
        """Remove the weld. FROM HERE, GRAVITY AND PHYSX DECIDE.

        Nothing after this point sets the item's pose. It is not teleported into
        the container, nudged towards the plan, or corrected: it falls, hits
        whatever is below it, rolls, and stops where it stops.
        """
        self.robot.release_object()
        # WAKE IT. A body that has been rigidly held may be asleep, and PhysX
        # does not wake it just because the constraint holding it was deleted.
        # See scene.wake_item — without this the item hangs where it was let go
        # and reports itself settled there.
        self.scene.wake_item(self.command.item_id or "")
        self._settle = SettleMonitor(
            linear_threshold=self.motion.linear_velocity_threshold,
            angular_threshold=self.motion.angular_velocity_threshold,
            stable_time=self.motion.settle_stable_time,
            timeout=self.motion.settle_timeout)
        self._settle.start(self.sim_time)
        self._enter(SequenceState.RETREAT)

    def _step_retreat(self) -> None:
        """Withdraw vertically before any lateral motion, clear of the rim."""
        target = self._target_world()
        goal = np.array([target[0], target[1],
                         self._container_rim_z() + self.motion.retreat_height])
        if self._servo(goal, self._place_yaw()[0]) or self._budget_exceeded():
            self._enter(SequenceState.WAIT_FOR_SETTLE)

    def _step_wait_for_settle(self) -> None:
        """Wait for the rigid body to come to rest, then judge the outcome.

        An item is NOT complete because the gripper opened. This is the state
        that decides, and it decides from the body's measured velocities and its
        measured final pose.
        """
        assert self._settle is not None and self.command is not None
        item = self.command.item_id or ""
        velocities = self.scene.item_velocities(item)
        if velocities is None:
            self._finish_item(None, None, settled=False, timed_out=False)
            return

        settled, timed_out = self._settle.update(
            self.sim_time, velocities[0], velocities[1])
        if not settled and not timed_out:
            return

        pose = self.scene.item_world_pose(item)
        self._finish_item(pose, self._settle.to_dict(self.sim_time),
                          settled=settled, timed_out=timed_out)

    # ------------------------------------------------------------------ #
    # Completion
    # ------------------------------------------------------------------ #

    def _finish_item(self, pose: Optional[Tuple[np.ndarray, np.ndarray]],
                     settle_detail: Optional[Dict[str, Any]],
                     *, settled: bool, timed_out: bool) -> None:
        assert self.command is not None
        command = self.command
        item = command.item_id or ""
        frame = (command.target_pose.frame if command.target_pose
                 else f"container:{command.container_id}")

        actual_pose: Optional[Pose] = None
        quaternion: Optional[np.ndarray] = None
        if pose is not None:
            quaternion = pose[1]
            actual_pose = world_to_pose(pose[0], pose[1], frame, self.layout)

        inner = command.container_inner_mm or {}
        outcome = evaluate_placement(
            actual=actual_pose,
            target=command.target_pose,
            actual_quaternion=quaternion,
            container_inner=Vec3(int(inner.get("x", 0)), int(inner.get("y", 0)),
                                 int(inner.get("z", 0))),
            length_mm=command.dimensions.length_mm if command.dimensions else 0,
            diameter_mm=(command.dimensions.outer_diameter_mm
                         if command.dimensions else 0),
            settled=settled, timed_out=timed_out,
            settle_detail=settle_detail or {})
        if self._axis_approximated:
            outcome.notes.append(
                "target axis 'z' is unreachable for a top-down gripper without a "
                "regrasp; placed horizontally and the axis error reports it")
            outcome.detail["axis_approximated"] = True
        outcome.detail["sequence_state"] = SequenceState.WAIT_FOR_SETTLE.value
        outcome.detail["grasp"] = "temporary fixed joint (secure-grasp approximation)"

        print(f"{LOG_ROBOT} {item}: {outcome.message}")
        self.state = SequenceState.NEXT_ITEM
        self.command = None
        self.on_done(item, outcome)
        self.state = SequenceState.IDLE


__all__ = ["PlacementSequence", "SequenceState", "down_orientation"]
