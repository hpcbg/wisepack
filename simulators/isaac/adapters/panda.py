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

from typing import List

from .base import RobotModelError
from .generic import GenericArticulationAdapter


class PandaRobotAdapter(GenericArticulationAdapter):
    """Franka Emika Panda: 7-DOF arm, two-finger parallel gripper."""

    def validate_model(self, *, preset: str = "") -> None:
        super().validate_model(preset=preset)

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
