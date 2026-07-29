"""The UFACTORY xArm 7, in Isaac Sim.

THIS IS A SIMULATION ADAPTER. It drives the shipped xArm 7 articulation in
PhysX. It is not a driver for a physical xArm and shares no code with one; a
real-hardware adapter would implement the same ``IsaacRobotAdapter``-shaped
contract against the UFACTORY SDK and would have entirely different failure
modes. Nothing produced by this file may be described as a real-robot result.

WHAT THE SHIPPED ASSET ACTUALLY IS — measured, not assumed
----------------------------------------------------------
``/Isaac/Robots/Ufactory/xarm7/xarm7.usd``, 13 DOF and 15 links:

  * seven revolute arm joints, ``joint1``..``joint7``, in that articulation
    order, all driven;
  * a UFACTORY parallel gripper INCLUDED in the asset through its
    ``Variant_Set = xarm_gripper`` selection — no separate gripper asset has to
    be composed in;
  * ONE driven gripper joint, ``drive_joint`` (0.0 open .. 0.85 closed,
    radians). The other five gripper joints carry ``PhysxMimicJointAPI:rotX``
    and follow it in physics. Commanding them directly fights the mimic
    constraint, so the profile lists them separately and they are never driven;
  * the end effector is ``xarm_gripper_base_link``, link index 8, and the
    fingertips sit 0.162 m along its approach axis. That number was MEASURED
    from the finger meshes' world bounding box. Using the finger LINK origins
    instead gives 0.10 m, because those origins are at the knuckles — and a
    72 mm-too-high tool centre point is not subtle: it put every one of the four
    grasp descents exactly that far short, on every attempt.

TWO THINGS THAT ARE NOT LIKE THE PANDA
--------------------------------------
1. ``PhysicsArticulationRootAPI`` is on ``<root>/root_joint``, not on the
   referenced Xform. ``Articulation()`` therefore resolves to the joint prim,
   and writing a world pose there raises "Undefined 'xformOp:translate'". The
   generic adapter writes the base transform to the reference Xform, which is
   correct for both robots.
2. The reach is 0.71 m against the Panda's 0.78 m usable, and the workcell was
   laid out for the Panda. The bin's far inner corner sits 0.766 m from an xArm
   base at retreat height — outside it. Measured: at the Panda's bin position
   two of the twelve corner poses fail to converge. So the profile moves the
   bin, and ``isaac_transform.layout_for_robot`` applies that to BOTH ends of
   the contract. Nothing here silently reaches for something it cannot get.
"""

from __future__ import annotations

from typing import List

from .base import RobotModelError
from .generic import GenericArticulationAdapter


class XArm7RobotAdapter(GenericArticulationAdapter):
    """UFACTORY xArm 7: 7-DOF arm, one driven gripper joint, five mimics."""

    def validate_model(self, *, preset: str = "") -> None:
        super().validate_model(preset=preset)

        problems: List[str] = []
        # ONE DRIVEN GRIPPER JOINT. Configuring more is not a harmless
        # over-specification: the mimic joints have no PhysX drive, so a
        # position target on them is silently discarded, and a reader of the
        # profile would reasonably conclude the gripper is six-DOF commanded
        # when it is one.
        if len(self.profile.gripper_joint_names) != 1:
            problems.append(
                f"the xArm gripper has exactly one driven joint, but "
                f"{self.profile.gripper_joint_names} are configured as driven; "
                "the rest follow through PhysxMimicJointAPI and belong in "
                "gripper_mimic_joint_names")
        if not self.profile.gripper_mimic_joint_names:
            problems.append(
                "no gripper mimic joints are configured; the shipped asset has "
                "five, and listing them is what lets validation prove the "
                "gripper actually came with the arm rather than the variant "
                "having resolved to a bare wrist")

        # THE GRIPPER IS PART OF THE ASSET, and this proves it rather than
        # trusting the variant selection. An asset whose default variant changes
        # would otherwise produce an arm with no hand that validates cleanly and
        # then closes nothing on every item.
        missing = [name for name in ("left_finger", "right_finger")
                   if name not in self._discovered_link_names]
        if missing:
            problems.append(
                f"the gripper finger link(s) {missing} are absent — the "
                f"'{self.profile.asset_variants.get('Variant_Set', '?')}' "
                "variant did not resolve to a gripper. Discovered links: "
                f"{self._discovered_link_names}")

        if problems:
            self._model_valid = False
            self._last_error = "; ".join(problems)
            raise RobotModelError(
                "UFACTORY xArm 7 gripper configuration is wrong:\n  - "
                + "\n  - ".join(problems),
                {"robot_id": self.profile.robot_id,
                 "discovered_link_names": list(self._discovered_link_names),
                 "gripper_joint_names": list(self.profile.gripper_joint_names),
                 "gripper_mimic_joint_names":
                     list(self.profile.gripper_mimic_joint_names)})


__all__ = ["XArm7RobotAdapter"]
