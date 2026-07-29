"""THE ROBOT-NEUTRAL EXECUTION CONTRACT.

Everything above this line — the scene, the WISEPACK bridge, the placement
sequence, the run gating, the workflow — talks to an ``IsaacRobotAdapter`` and
knows nothing about which arm is behind it. Everything below it is one robot's
particulars. That boundary is the whole point of this package: the alternative,
copying the Panda implementation into an xArm file, produces two state machines
that drift and a fleet of ``if robot == "panda"`` branches in code that has no
business knowing what a robot is.

WHAT AN ADAPTER OWES ITS CALLER
-------------------------------
1. ``load()`` puts the robot on the stage and places its base. It does not
   command anything.
2. ``validate_model()`` proves the robot it loaded is the robot the profile
   describes — asset resolved, articulation valid, every configured joint
   present in the discovered order, end effector resolvable, gripper present —
   and raises ``RobotModelError`` naming exactly what disagreed. A partially
   loaded robot must never reach execution, so this runs before ``initialise``
   and its failure is terminal for the run.
3. Every motion method is ITERATIVE and non-blocking. ``command_tcp_pose``
   performs ONE differential-IK step towards the goal and returns; the caller
   servos it once per physics frame and watches for convergence. An adapter
   that blocked until arrival would freeze the render loop and make a stuck goal
   indistinguishable from a hung process.
4. ``get_diagnostics()`` reports what was MEASURED — discovered joints, resolved
   prims, home verification — never what was configured. A diagnostic that
   echoes its own configuration back cannot detect the case it exists for.

WHAT AN ADAPTER MUST NOT DO
---------------------------
Decide anything. It does not choose an item, re-plan, judge a placement, or
decide that a goal is a good idea. The approval gate, the pre-pick sanity
checks and the outcome measurement are upstream and stay upstream.

WHAT THIS IS NOT
----------------
Not a motion planner. Every adapter here servos a Cartesian goal with damped
least squares over the articulation Jacobian, one step per frame, with no
collision awareness and no trajectory. `RobotProfile.is_motion_planner` says so
and is always False. Not a hardware driver either: an ``IsaacRobotAdapter``
drives a simulated articulation in PhysX. A real xArm needs a different adapter
against the same interface, which is why the interface is here and not inside
the Isaac state machine.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from wisepack_core.robots import RobotProfile


class RobotModelError(RuntimeError):
    """The loaded robot is not the robot the profile describes.

    Raised by ``load`` and ``validate_model`` for a missing asset, an absent
    prim, a joint set that does not match the articulation, an unresolvable end
    effector, or a preset the selected robot does not support. The caller
    reports ``ROBOT_MODEL_INVALID``, holds the backend DEGRADED and disables
    approval — never a best-effort continuation.

    ``detail`` carries the machine-readable facts the diagnostics panel renders,
    so an operator sees "expected joint1..joint7, discovered panda_joint1..7"
    rather than a one-line exception string.
    """

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.detail: Dict[str, Any] = dict(detail or {})


class IsaacRobotAdapter(abc.ABC):
    """One simulated manipulator, behind an interface nothing else looks past."""

    def __init__(self, profile: RobotProfile) -> None:
        self.profile = profile
        #: Populated by load()/validate_model(); read by get_diagnostics(). Kept
        #: as plain data so a failure BEFORE the articulation exists still has
        #: something to report — a diagnostics call that itself raises tells the
        #: operator nothing.
        self._resolved_asset: str = ""
        self._discovered_dof_names: List[str] = []
        self._discovered_link_names: List[str] = []
        self._articulation_root: str = ""
        self._end_effector_resolved: str = ""
        self._model_valid: bool = False
        self._last_error: str = ""

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #

    def get_robot_id(self) -> str:
        return self.profile.robot_id

    @property
    def tool_centre_point_m(self) -> float:
        """Fingertip standoff from the end-effector link, along the tool axis."""
        return self.profile.tool_centre_point_m

    @property
    def grasp_yaw_offset_deg(self) -> float:
        return self.profile.grasp_yaw_offset_deg

    @property
    def model_valid(self) -> bool:
        return self._model_valid

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def load(self, *, base_position: Sequence[float],
             base_orientation: Sequence[float]) -> None:
        """Reference the asset in and place the base. Commands nothing."""

    @abc.abstractmethod
    def initialise(self) -> None:
        """Acquire the physics views. Called after the timeline is playing."""

    @abc.abstractmethod
    def validate_model(self, *, preset: str = "") -> None:
        """Raise RobotModelError unless the loaded robot matches the profile."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Return to a known state: home pose, open gripper, nothing held."""

    @abc.abstractmethod
    def stop_motion(self) -> None:
        """Hold position immediately. Used on abort and on a safe hold."""

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def get_joint_state(self) -> np.ndarray:
        """Current positions of every DOF, arm first, in articulation order."""

    @abc.abstractmethod
    def get_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """(position_m, quaternion_wxyz) of the END-EFFECTOR LINK, measured.

        The link, not the fingertips: the tool-centre-point offset is applied by
        the caller so that exactly one place in the codebase knows about it.
        """

    @abc.abstractmethod
    def is_home(self) -> bool:
        """MEASURED: read the joints and compare against the profile's home."""

    # ------------------------------------------------------------------ #
    # Motion
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def command_home(self) -> None:
        """Command the profile's home configuration. One call, not a loop."""

    @abc.abstractmethod
    def command_joint_positions(self, positions: Sequence[float]) -> None:
        """Command arm joint targets directly, bypassing kinematics."""

    @abc.abstractmethod
    def command_tcp_pose(self, position: Sequence[float],
                         orientation: Sequence[float]) -> None:
        """ONE differential-IK step towards a Cartesian goal for the EE link."""

    @abc.abstractmethod
    def open_gripper(self) -> None:
        ...

    @abc.abstractmethod
    def close_gripper(self) -> None:
        ...

    # ------------------------------------------------------------------ #
    # Grasp
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def attach_object(self, item_path: str, item_id: str,
                      item_position: np.ndarray,
                      item_orientation: np.ndarray) -> None:
        """Hold ``item_id``. TEMPORARY FIXED JOINT — see grasp.py.

        The adapter owns this because the frame the item is welded to is a
        robot-specific prim (``panda_hand``, ``xarm_gripper_base_link``) and
        nothing above this line may know those names.
        """

    @abc.abstractmethod
    def release_object(self) -> None:
        """Let go. Idempotent — releasing twice is not an error."""

    @property
    @abc.abstractmethod
    def holding(self) -> Optional[str]:
        """The item id currently welded to the hand, or None."""

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def get_diagnostics(self) -> Dict[str, Any]:
        """What was MEASURED about this robot, for the Diagnostics panel.

        Never raises: this is called on the failure path, including from a
        handler for the exception that left the model invalid, and a diagnostics
        call that itself throws replaces a useful report with a traceback.
        """
        profile = self.profile
        try:
            home_verified = bool(self._model_valid and self.is_home())
        except Exception as exc:                            # noqa: BLE001
            home_verified = False
            self._last_error = self._last_error or f"home check failed: {exc!r}"
        return {
            "robot_id": profile.robot_id,
            "display_name": profile.display_name,
            "manufacturer": profile.manufacturer,
            "model": profile.model,
            "implementation_status": profile.implementation_status,
            "robot_profile_revision": profile.revision,
            "adapter": type(self).__name__,
            "asset_resolved": self._resolved_asset or "",
            "articulation_valid": bool(self._articulation_root),
            "articulation_root": self._articulation_root or "",
            "expected_arm_joints": list(profile.arm_joint_names),
            "discovered_arm_joints": list(self._discovered_dof_names[
                :len(profile.arm_joint_names)]),
            "discovered_dof_names": list(self._discovered_dof_names),
            "discovered_link_names": list(self._discovered_link_names),
            "expected_gripper_joints": list(profile.gripper_joint_names),
            "end_effector_resolved": self._end_effector_resolved or "",
            "gripper_ready": bool(self._model_valid and profile.gripper_joint_names),
            "home_verified": home_verified,
            "kinematics": profile.kinematics,
            # Stated, not inferred. A differential IK controller is not a motion
            # planner and the dashboard must never imply that it is.
            "kinematics_ready": bool(self._model_valid),
            "motion_planning": profile.is_motion_planner,
            "tool_centre_point_m": profile.tool_centre_point_m,
            "model_valid": self._model_valid,
            "last_robot_error": self._last_error,
        }

    def note_error(self, message: str) -> None:
        self._last_error = message


__all__ = ["IsaacRobotAdapter", "RobotModelError"]
