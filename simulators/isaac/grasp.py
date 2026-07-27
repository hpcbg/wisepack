"""The temporary fixed-joint grasp — a SECURE-GRASP APPROXIMATION.

BE CLEAR ABOUT WHAT THIS IS
---------------------------
When the gripper has reached the item and closed, this module welds the item to
``panda_hand`` with a USD fixed joint. The joint is removed the instant the
gripper opens to release it.

So the CARRY is idealised: the item cannot slip, rotate in the fingers, or be
dropped by a marginal grasp, because for the duration of the carry it is rigidly
attached rather than held by friction. A real parallel-jaw grasp of a smooth
steel pipe is exactly the case where friction modelling matters most, and this
first iteration does not model it.

WHAT IS NOT APPROXIMATED
------------------------
Everything after the release. The joint is destroyed before the item falls, so
the drop, the impact, the roll, the contacts with the container walls and with
items already placed, and the final resting pose are all resolved by PhysX with
no assistance. The reported outcome — settled pose, distance from the planned
pose, containment — is a measurement of that physics, not of this shortcut.

WHY A JOINT RATHER THAN FRICTION
--------------------------------
Because the alternative for a first iteration is worse in a specific way: a
friction-only grasp that drops items intermittently produces run-to-run variation
that is indistinguishable from a bug in the WISEPACK integration under test. The
point of this iteration is to prove the plan -> physical execution -> feedback
loop, and a stochastic grasp would obscure exactly that. Replacing this with a
real friction grasp is the first item on the second-iteration list.

WHY `excludeFromArticulation` MATTERS
-------------------------------------
Without it, PhysX tries to fold the item into the Panda's articulation, which
means changing an articulation's topology while it is simulating. Setting the
attribute makes this a maximal-coordinate joint between two bodies instead —
created and destroyed at runtime without disturbing the arm.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import isaacsim.core.experimental.utils.stage as stage_utils
from pxr import Gf, UsdPhysics

from .config import LOG_ROBOT

#: One joint at a time: the robot has one gripper. A fixed path also means a
#: leaked joint from a previous item is overwritten rather than accumulating.
GRASP_JOINT_PATH = "/World/WisepackGraspJoint"


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=float)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q`` (w, x, y, z)."""
    w, x, y, z = (float(c) for c in q)
    u = np.array([x, y, z], dtype=float)
    return (v * (w * w - u.dot(u)) + 2.0 * u * u.dot(v) + 2.0 * w * np.cross(u, v))


class GraspJoint:
    """Creates and destroys the temporary weld between the hand and one item."""

    def __init__(self) -> None:
        self.attached_item: Optional[str] = None

    @property
    def is_attached(self) -> bool:
        return self.attached_item is not None

    def attach(self, hand_path: str, item_path: str, item_id: str,
               hand_position: np.ndarray, hand_orientation: np.ndarray,
               item_position: np.ndarray, item_orientation: np.ndarray) -> None:
        """Weld ``item_path`` to ``hand_path`` at their CURRENT relative pose.

        The relative pose is computed and written into the joint's local frames
        rather than left at identity. With identity local frames the joint would
        snap the item's origin onto the hand's origin — a visible teleport of a
        few centimetres at the moment of grasping, which is precisely the kind of
        thing this integration must not do.
        """
        self.detach()
        stage = stage_utils.get_current_stage(backend="usd")

        # Item pose expressed in the hand frame: q_rel = conj(q_hand) * q_item,
        # p_rel = rotate(conj(q_hand), p_item - p_hand).
        hand_conj = _quat_conjugate(np.asarray(hand_orientation, dtype=float))
        rel_position = _quat_rotate(
            hand_conj,
            np.asarray(item_position, dtype=float)
            - np.asarray(hand_position, dtype=float))
        rel_orientation = _quat_multiply(
            hand_conj, np.asarray(item_orientation, dtype=float))

        joint = UsdPhysics.FixedJoint.Define(stage, GRASP_JOINT_PATH)
        joint.CreateBody0Rel().SetTargets([hand_path])
        joint.CreateBody1Rel().SetTargets([item_path])
        joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(*(float(v) for v in rel_position)))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(
            float(rel_orientation[0]), Gf.Vec3f(float(rel_orientation[1]),
                                                float(rel_orientation[2]),
                                                float(rel_orientation[3]))))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        # See the module docstring: this keeps the item OUT of the Panda's
        # articulation, so the weld does not re-topologise a simulating arm.
        joint.CreateExcludeFromArticulationAttr().Set(True)

        self.attached_item = item_id
        print(f"{LOG_ROBOT} attached {item_id} to panda_hand "
              f"(secure-grasp approximation; offset "
              f"{tuple(round(float(v), 4) for v in rel_position)} m)")

    def detach(self) -> None:
        """Remove the weld. Idempotent — releasing twice is not an error."""
        stage = stage_utils.get_current_stage(backend="usd")
        if stage.GetPrimAtPath(GRASP_JOINT_PATH):
            stage.RemovePrim(GRASP_JOINT_PATH)
            if self.attached_item:
                print(f"{LOG_ROBOT} detached {self.attached_item} — the drop "
                      "and settling from here are PhysX, not this code")
        self.attached_item = None


__all__ = ["GraspJoint", "GRASP_JOINT_PATH"]
