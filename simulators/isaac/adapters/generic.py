"""ONE implementation of the adapter contract, driven entirely by a profile.

Both supported robots run through this class. That is deliberate and it is the
answer to "do not copy the complete current Panda implementation into an
xArm-specific file": every behaviour that differs between a Panda and an xArm 7
— asset URL, prim paths, joint names and order, home configuration, gripper
travel, tool-centre-point, reach — is DATA in ``config/isaac_robots.yaml``, so
there is nothing left to copy. What the concrete subclasses in ``panda.py`` and
``xarm7.py`` add is small on purpose, and they are honest about being small: they
exist to name the robot in an error message and to be the seam where a genuinely
robot-specific behaviour lands when one appears, rather than that behaviour
being bolted on here behind an ``if``.

VALIDATE BEFORE COMMANDING, ALWAYS
----------------------------------
``validate_model`` is not a formality. Isaac does not fail loudly when a
configuration is wrong for an asset: an absent joint name resolves to an empty
index list and the command silently does nothing, a wrong end-effector link
gives a Jacobian for some other body and the arm converges confidently to the
wrong place, and a tool-centre-point that is too short puts every grasp descent
above the object. All three read as physics problems and none of them are. So
every claim the profile makes is checked against the articulation that was
actually loaded, and a disagreement is terminal.

API NOTE — Isaac Sim 6.0.1
--------------------------
``isaacsim.core.experimental.*`` throughout. The older ``isaacsim.core.api`` has
moved to ``extsDeprecated/`` in 6.0.1; it still imports, and writing new code
against it is how an integration ages badly before it ships.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim
from isaacsim.storage.native import get_assets_root_path

from wisepack_core.robots import RobotProfile

from ..config import LOG_ROBOT
from ..grasp import GraspJoint
from .base import IsaacRobotAdapter, RobotModelError
from .kinematics import DEFAULT_DAMPING, DEFAULT_SCALE, differential_ik_step


class GenericArticulationAdapter(IsaacRobotAdapter):
    """A profile-driven manipulator: generic articulation, generic kinematics."""

    def __init__(self, profile: RobotProfile) -> None:
        super().__init__(profile)
        self.articulation: Optional[Articulation] = None
        self.end_effector: Optional[RigidPrim] = None
        self.grasp = GraspJoint()
        #: Resolved during validation, never assumed.
        self._arm_indices: List[int] = []
        self._gripper_indices: List[int] = []
        self._ee_link_index: int = -1
        self._jacobian_row: int = -1
        self._ee_prim_path: str = ""

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _resolve_asset(self) -> str:
        """The first configured asset path that exists under the asset root.

        Reports EVERY path it tried when none resolve. "Could not find the
        robot asset" without the list is the least useful possible message for
        a failure whose cause is almost always a path that moved between Isaac
        releases.
        """
        try:
            root = get_assets_root_path()
        except Exception as exc:                            # noqa: BLE001
            raise RobotModelError(
                f"the Isaac Sim asset root is not reachable, so "
                f"{self.profile.display_name} cannot be loaded: {exc}. The "
                "robot assets are fetched from NVIDIA's asset server at "
                "runtime and are not committed to this repository — check "
                "outbound HTTPS.",
                {"robot_id": self.profile.robot_id}) from exc

        tried = []
        for candidate in self.profile.asset_path_candidates:
            url = candidate if "://" in candidate else root.rstrip("/") + candidate
            tried.append(url)
            if _asset_exists(url):
                return url
        raise RobotModelError(
            f"none of the configured assets for "
            f"{self.profile.display_name} exist under the asset root:\n  - "
            + "\n  - ".join(tried)
            + f"\nEdit asset_path_candidates for {self.profile.robot_id!r} in "
              "config/isaac_robots.yaml.",
            {"robot_id": self.profile.robot_id, "tried": tried})

    def load(self, *, base_position: Sequence[float],
             base_orientation: Sequence[float]) -> None:
        profile = self.profile
        url = self._resolve_asset()
        self._resolved_asset = url
        print(f"{LOG_ROBOT} loading {profile.display_name} from {url}")
        kwargs: Dict[str, Any] = {"usd_path": url, "path": profile.root_prim_path}
        if profile.asset_variants:
            kwargs["variants"] = [(k, v) for k, v in profile.asset_variants.items()]
        try:
            stage_utils.add_reference_to_stage(**kwargs)
        except Exception as exc:                            # noqa: BLE001
            raise RobotModelError(
                f"{profile.display_name} could not be referenced into the "
                f"stage at {profile.root_prim_path}: {exc}",
                {"robot_id": profile.robot_id, "asset": url}) from exc

        # THE BASE GOES ON THE REFERENCE XFORM, not on the articulation.
        #
        # These are not always the same prim. The xArm 7 asset carries
        # PhysicsArticulationRootAPI on <root>/root_joint, so Articulation()
        # resolves to that joint and writing a world pose to it asserts with
        # "Undefined 'xformOp:translate' property". Writing to the reference
        # Xform is correct for both robots and needs no special case.
        XformPrim(profile.root_prim_path, reset_xform_op_properties=True
                  ).set_world_poses(
            positions=np.array([list(base_position)], dtype=float),
            orientations=np.array([list(base_orientation)], dtype=float))

        self.articulation = Articulation(profile.root_prim_path)
        resolved = [str(p) for p in self.articulation.paths]
        self._articulation_root = resolved[0] if resolved else ""

    def initialise(self) -> None:
        """Read the articulation once the timeline is playing.

        Separate from ``load`` because the physics tensor views do not exist
        until play(), and every name and index below comes from those views
        rather than from the USD — the discovered order is what the commands
        will actually address.
        """
        if self.articulation is None:                       # pragma: no cover
            raise RobotModelError("initialise() called before load()")
        self._discovered_dof_names = list(self.articulation.dof_names)
        self._discovered_link_names = list(self.articulation.link_names)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate_model(self, *, preset: str = "") -> None:
        profile = self.profile
        problems: List[str] = []
        detail: Dict[str, Any] = {"robot_id": profile.robot_id}

        if preset:
            refusal = profile.preset_refusal(preset)
            if refusal:
                # Checked FIRST and reported on its own: an incompatible
                # robot/preset pair is a selection mistake, and burying it under
                # joint diagnostics sends the operator looking at the asset.
                raise RobotModelError(refusal,
                                      {"robot_id": profile.robot_id,
                                       "preset": preset,
                                       "supported_presets":
                                           list(profile.supported_presets)})

        if self.articulation is None:
            raise RobotModelError(
                f"{profile.display_name} has no articulation — load() did not "
                "complete", detail)
        if not self._discovered_dof_names:
            self.initialise()

        if profile.articulation_root and self._articulation_root \
                and self._articulation_root != profile.articulation_root:
            problems.append(
                f"the articulation root resolved to {self._articulation_root} "
                f"but the profile expects {profile.articulation_root}")

        discovered = self._discovered_dof_names
        missing = [n for n in profile.arm_joint_names if n not in discovered]
        if missing:
            problems.append(
                f"configured arm joint(s) {missing} are not in the "
                f"articulation; it has {discovered}")
        missing_grip = [n for n in profile.gripper_joint_names
                        if n not in discovered]
        if missing_grip:
            problems.append(
                f"configured gripper joint(s) {missing_grip} are not in the "
                f"articulation; it has {discovered}")
        missing_mimic = [n for n in profile.gripper_mimic_joint_names
                         if n not in discovered]
        if missing_mimic:
            problems.append(
                f"configured gripper mimic joint(s) {missing_mimic} are not in "
                "the articulation")

        if not missing:
            # THE ORDER, not just the membership. Commands are issued by INDEX,
            # so an articulation that contains the right joints in a different
            # order would accept every command and move the wrong axes.
            self._arm_indices = [discovered.index(n)
                                 for n in profile.arm_joint_names]
            if self._arm_indices != sorted(self._arm_indices):
                problems.append(
                    f"the configured arm joint order {profile.arm_joint_names} "
                    f"is not the articulation's order "
                    f"{[discovered[i] for i in sorted(self._arm_indices)]}")
        if not missing_grip:
            self._gripper_indices = [discovered.index(n)
                                     for n in profile.gripper_joint_names]

        if profile.end_effector_link not in self._discovered_link_names:
            problems.append(
                f"the end-effector link {profile.end_effector_link!r} is not in "
                f"the articulation; it has {self._discovered_link_names}")
        else:
            self._ee_link_index = int(self.articulation.get_link_indices(
                profile.end_effector_link).list()[0])
            paths = self.articulation.link_paths
            flat = [str(p) for p in (paths[0] if paths and isinstance(paths[0], list)
                                     else paths)]
            index = self._discovered_link_names.index(profile.end_effector_link)
            self._ee_prim_path = flat[index] if index < len(flat) else ""
            if profile.end_effector_prim and self._ee_prim_path \
                    and self._ee_prim_path != profile.end_effector_prim:
                problems.append(
                    f"the end-effector prim resolved to {self._ee_prim_path} "
                    f"but the profile expects {profile.end_effector_prim}")
            if self._ee_prim_path:
                try:
                    self.end_effector = RigidPrim(self._ee_prim_path)
                    self.end_effector.get_world_poses()
                except Exception as exc:                    # noqa: BLE001
                    problems.append(
                        f"the end-effector prim {self._ee_prim_path} exists but "
                        f"its pose cannot be read: {exc!r}")

        # WHICH ROW OF THE JACOBIAN belongs to the end effector depends on
        # whether the base is fixed. A fixed-base articulation has num_links - 1
        # rows and the link's row is index - 1; a floating base has num_links.
        # Getting this wrong yields a valid-looking Jacobian for a different
        # body, and the arm then converges confidently to the wrong pose.
        try:
            shape = self.articulation.get_jacobian_matrices().numpy().shape
            rows = int(shape[1])
            links = int(self.articulation.num_links)
            if rows == links - 1:
                self._jacobian_row = self._ee_link_index - 1
            elif rows == links:
                self._jacobian_row = self._ee_link_index
            else:
                problems.append(
                    f"the Jacobian has {rows} row(s) for {links} link(s), which "
                    "matches neither a fixed nor a floating base")
            detail["jacobian_shape"] = list(shape)
            if self._jacobian_row >= 0 and int(shape[3]) < profile.arm_dof:
                problems.append(
                    f"the Jacobian has {shape[3]} column(s) but the profile "
                    f"configures {profile.arm_dof} arm joint(s)")
        except Exception as exc:                            # noqa: BLE001
            problems.append(f"the Jacobian is not readable: {exc!r}")

        try:
            dof = self.get_joint_state()
            if not np.all(np.isfinite(dof)):
                problems.append("the articulation reports non-finite joint "
                                "positions")
        except Exception as exc:                            # noqa: BLE001
            problems.append(f"the joint positions are not readable: {exc!r}")

        detail.update({
            "expected_arm_joints": list(profile.arm_joint_names),
            "discovered_dof_names": list(discovered),
            "discovered_link_names": list(self._discovered_link_names),
            "articulation_root": self._articulation_root,
            "end_effector_prim": self._ee_prim_path,
            "asset": self._resolved_asset,
        })
        if problems:
            self._model_valid = False
            self._last_error = "; ".join(problems)
            raise RobotModelError(
                f"{profile.display_name} does not match its profile "
                f"({profile.robot_id}, revision {profile.revision}):\n  - "
                + "\n  - ".join(problems), detail)

        self._end_effector_resolved = self._ee_prim_path
        self._model_valid = True
        self._last_error = ""
        print(f"{LOG_ROBOT} {profile.display_name} validated: "
              f"{profile.arm_dof} arm joint(s) {profile.arm_joint_names}, "
              f"gripper {profile.gripper_joint_names}, end effector "
              f"{profile.end_effector_link} at {self._ee_prim_path}, TCP offset "
              f"{profile.tool_centre_point_m:.3f} m")

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    def get_joint_state(self) -> np.ndarray:
        assert self.articulation is not None
        return np.asarray(self.articulation.get_dof_positions().numpy()[0],
                          dtype=float)

    def get_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.end_effector is not None
        position, orientation = self.end_effector.get_world_poses()
        return (np.asarray(position.numpy()[0], dtype=float),
                np.asarray(orientation.numpy()[0], dtype=float))

    def is_home(self) -> bool:
        """MEASURED against the profile's home, within its own tolerance."""
        try:
            dof = self.get_joint_state()
        except Exception:                                   # noqa: BLE001
            return False
        home = np.asarray(self.profile.home_joint_positions, dtype=float)
        if len(dof) < len(home):
            return False
        arm = dof[self._arm_indices] if self._arm_indices else dof[:len(home)]
        return bool(np.max(np.abs(arm - home)) < self.profile.home_tolerance_rad)

    # ------------------------------------------------------------------ #
    # Motion
    # ------------------------------------------------------------------ #

    def command_home(self) -> None:
        """Put the arm at home and HOLD it there.

        Both ``set_dof_positions`` and ``set_dof_position_targets``, and that is
        not redundant. Referencing an asset RECORDS a default state but never
        commands it, so at play() the drives hold whatever the USD authored and
        the arm drifts there over the following seconds. Measured on the Panda:
        an identical pre-grasp goal converged when the command arrived
        immediately and timed out when it arrived after an operator approval,
        because by then the arm had drifted somewhere else. Setting the
        positions places it; setting the targets keeps it.
        """
        assert self.articulation is not None
        home = list(self.profile.home_joint_positions)
        indices = self._arm_indices or list(range(len(home)))
        values = np.array([home], dtype=float)
        self.articulation.set_dof_positions(values, dof_indices=indices)
        self.articulation.set_dof_position_targets(values, dof_indices=indices)
        self.open_gripper()

    def command_joint_positions(self, positions: Sequence[float]) -> None:
        assert self.articulation is not None
        indices = self._arm_indices or list(range(len(positions)))
        self.articulation.set_dof_position_targets(
            np.array([list(positions)], dtype=float), dof_indices=indices)

    def command_tcp_pose(self, position: Sequence[float],
                         orientation: Sequence[float]) -> None:
        assert self.articulation is not None
        if not self._model_valid:                           # pragma: no cover
            raise RobotModelError(
                f"refusing to command {self.profile.display_name}: its model "
                "did not validate")
        current = self.articulation.get_dof_positions().numpy()
        ee_position, ee_orientation = self.end_effector.get_world_poses()
        jacobian = self.articulation.get_jacobian_matrices().numpy()
        arm = self.profile.arm_dof
        options = self.profile.kinematics_options
        # STABILISATION IS PER ROBOT AND OFF BY DEFAULT. Every term below is
        # zero unless the profile asks for it, so a robot that does not need one
        # behaves exactly as it did. See adapters/kinematics.py for the measured
        # justification of each.
        delta = differential_ik_step(
            jacobian[:, self._jacobian_row, :, :arm],
            ee_position.numpy(), ee_orientation.numpy(),
            position, orientation,
            damping=float(options.get("damping", DEFAULT_DAMPING)),
            scale=float(options.get("scale", DEFAULT_SCALE)),
            positions=current[0, :arm],
            # The profile's HOME is the preferred posture. It is already the
            # configuration the arm is validated to hold, so there is no second
            # number to keep in step with it.
            rest_posture=self.profile.home_joint_positions,
            rest_gain=float(options.get("rest_gain", 0.0)),
            singular_damping=float(options.get("singular_damping", 0.0)),
            manipulability_threshold=float(
                options.get("manipulability_threshold", 0.0)),
            max_joint_step=float(options.get("max_joint_step_rad", 0.0)))
        indices = self._arm_indices or list(range(arm))
        self.articulation.set_dof_position_targets(
            current[:, :arm] + delta, dof_indices=indices)

    def stop_motion(self) -> None:
        """Hold wherever the arm is now, by commanding its measured position."""
        if self.articulation is None:
            return
        try:
            dof = self.get_joint_state()
        except Exception:                                   # noqa: BLE001
            return
        indices = self._arm_indices or list(range(self.profile.arm_dof))
        self.articulation.set_dof_position_targets(
            np.array([[dof[i] for i in indices]], dtype=float),
            dof_indices=indices)

    # ------------------------------------------------------------------ #
    # Gripper
    # ------------------------------------------------------------------ #

    def _command_gripper(self, positions: Sequence[float]) -> None:
        assert self.articulation is not None
        indices = self._gripper_indices
        if not indices:
            return
        # ONLY the DRIVEN joints. The xArm gripper's other five carry
        # PhysxMimicJointAPI and follow the driven one in physics; commanding
        # them would be fighting the mimic constraint rather than helping it.
        self.articulation.set_dof_position_targets(
            np.array([list(positions)], dtype=float), dof_indices=indices)

    def open_gripper(self) -> None:
        self._command_gripper(self.profile.open_gripper_positions)

    def close_gripper(self) -> None:
        self._command_gripper(self.profile.closed_gripper_positions)

    # ------------------------------------------------------------------ #
    # Grasp
    # ------------------------------------------------------------------ #

    def attach_object(self, item_path: str, item_id: str,
                      item_position: np.ndarray,
                      item_orientation: np.ndarray) -> None:
        hand_position, hand_orientation = self.get_tcp_pose()
        self.grasp.attach(
            hand_path=self._ee_prim_path or self.profile.end_effector_prim,
            item_path=item_path, item_id=item_id,
            hand_position=hand_position, hand_orientation=hand_orientation,
            item_position=np.asarray(item_position, dtype=float),
            item_orientation=np.asarray(item_orientation, dtype=float))

    def release_object(self) -> None:
        self.grasp.detach()

    @property
    def holding(self) -> Optional[str]:
        return self.grasp.attached_item

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Drop anything held, open the gripper, return home. In that order.

        The order is the safety property: the weld is removed BEFORE the arm is
        moved, so a reset cannot drag a held object across the scene, and before
        any body is deleted, so it cannot leave a joint pointing at a prim that
        no longer exists.
        """
        self.release_object()
        self.open_gripper()
        self.command_home()


def _asset_exists(url: str) -> bool:
    """Does this USD path resolve? Never raises.

    Uses omni.client, which understands the omniverse://, http(s):// and file://
    schemes the asset root can take. A resolution failure must produce the
    "none of these paths exist" diagnostic, not a traceback from inside the
    client library.
    """
    try:
        import omni.client                                  # noqa: PLC0415
        result, _ = omni.client.stat(url)
        return result == omni.client.Result.OK
    except Exception:                                       # noqa: BLE001
        # Cannot tell. Treat as present and let add_reference_to_stage produce
        # the real error rather than refusing a robot over a probe that failed.
        return True


__all__ = ["GenericArticulationAdapter"]
