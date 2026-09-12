"""The Franka Emika Panda, as WISEPACK's regression backend.

DELIBERATELY THIN, and that is the result rather than an omission. Everything
that used to make the Panda implementation Panda-shaped — the asset URL, the
``panda_hand`` prim, the nine-DOF layout, the ready pose written out in two
places, the 0.103 m hand-to-fingertip offset — is now in
``config/isaac_robots.yaml`` and is executed by
``GenericArticulationAdapter``. What is left here is what is genuinely specific:
one extra validation, and a name to put in an error message.

WHY NOT ``isaacsim.robot.experimental.manipulators.examples.franka.Franka``
--------------------------------------------------------------------------
It was used before this refactor and it is a good helper. But it is Franka-only,
so keeping it would have meant the Panda running NVIDIA's differential IK and
the xArm running a second copy of the same maths — two implementations of one
behaviour, which is the thing this package exists to avoid. The generic path
uses the identical damped-least-squares formulation (see ``kinematics.py``), the
same asset with the same variant selections, and the same recorded ready pose,
so what changed for the Panda is which code computes the step, not what the step
is. Validation stage E re-runs a physical pick to show that.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence



from ..tool import GripperCutterTool, ToolSpec
from .base import RobotModelError
from .generic import GenericArticulationAdapter


class PandaRobotAdapter(GenericArticulationAdapter):
    """Franka Emika Panda: 7-DOF arm, two-finger parallel gripper.

    WITH THE COMBINED END EFFECTOR when the profile declares one: the stock
    parallel-jaw gripper stays the grasp mechanism, and a shear is authored on
    the same hand at the profile's fixed grasp->cut offset (`simulators/isaac/
    tool.py`). Both are actuated independently through this adapter — the
    fingers as articulation joints, the blades as their own drives — and the
    sequence in `robot.py` never learns which prim is which.
    """

    def __init__(self, profile) -> None:
        super().__init__(profile)
        self.tool: Optional[GripperCutterTool] = None

    def load(self, *, base_position: Sequence[float],
             base_orientation: Sequence[float]) -> None:
        super().load(base_position=base_position, base_orientation=base_orientation)
        if not self.profile.has_cutter:
            return
        # AUTHORED BEFORE PLAY as VISUAL geometry under the hand link: no rigid
        # body, no collider, no joint — PhysX never sees the tool (tool.py
        # explains why, with the measurement that forced it).
        spec = ToolSpec.from_dict(self.profile.end_effector_tool,
                                  tcp_m=self.profile.tool_centre_point_m)
        self.tool = GripperCutterTool(spec)
        self.tool.build(self.profile.end_effector_prim)

    # -- the combined tool ------------------------------------------------ #

    @property
    def has_cutter(self) -> bool:
        return self.tool is not None and self.tool.built

    @property
    def grasp_to_cut_offset_m(self) -> float:
        return float(self.tool.spec.cut_offset_m[0]) if self.tool is not None else 0.0

    def open_cutter(self) -> None:
        if self.tool is None:
            raise RobotModelError(f"{self.profile.display_name} carries no cutter")
        self.tool.open_cutter()

    def close_cutter(self) -> None:
        if self.tool is None:
            raise RobotModelError(f"{self.profile.display_name} carries no cutter")
        self.tool.close_cutter()

    def cutter_closed(self) -> bool:
        return bool(self.tool is not None and self.tool.cutter_closed())

    def tool_diagnostics(self) -> Dict[str, Any]:
        return self.tool.diagnostics() if self.tool is not None else {}

    def tick_tool(self) -> None:
        if self.tool is not None:
            self.tool.tick()

    def reset(self) -> None:
        super().reset()
        if self.tool is not None:
            self.tool.open_cutter()

    def validate_model(self, *, preset: str = "") -> None:
        super().validate_model(preset=preset)
        if self.profile.has_cutter and not self.has_cutter:
            self._model_valid = False
            self._last_error = "the profile declares a gripper+cutter tool but none was built"
            raise RobotModelError(self._last_error, {"robot_id": self.profile.robot_id})
        if self.tool is not None and not self.tool.physics_free():
            self._model_valid = False
            self._last_error = ("the gripper+cutter tool carries a physics schema; "
                                "it must be visual-only (see tool.py)")
            raise RobotModelError(self._last_error, {"robot_id": self.profile.robot_id})

        # THE TWO FINGERS MUST BE COMMANDED AS A PAIR. The Panda's fingers are
        # two independently driven prismatic joints with no mimic relationship,
        # so a profile that drives only one produces a gripper that closes
        # lopsidedly and grasps at an offset the TCP does not describe. The
        # generic validator checks that the configured joints EXIST; this checks
        # that enough of them are configured to be a gripper at all.
        problems: List[str] = []
        if len(self.profile.gripper_joint_names) != 2:
            problems.append(
                f"the Panda gripper has two independently driven finger joints, "
                f"but {len(self.profile.gripper_joint_names)} are configured "
                f"({self.profile.gripper_joint_names}); both must be commanded "
                "or the fingers close asymmetrically")
        if self.profile.gripper_mimic_joint_names:
            problems.append(
                f"the Panda gripper has no mimic joints, but "
                f"{self.profile.gripper_mimic_joint_names} are configured")
        if problems:
            self._model_valid = False
            self._last_error = "; ".join(problems)
            raise RobotModelError(
                "Franka Emika Panda gripper configuration is wrong:\n  - "
                + "\n  - ".join(problems),
                {"robot_id": self.profile.robot_id,
                 "gripper_joint_names": list(self.profile.gripper_joint_names)})


__all__ = ["PandaRobotAdapter"]
