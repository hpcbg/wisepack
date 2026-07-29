"""Differential inverse kinematics, robot-neutral and numpy-only.

DAMPED LEAST SQUARES over the end-effector Jacobian. Identical maths to the one
in Isaac's shipped Franka helper, lifted here so both adapters use one
implementation rather than the Panda getting NVIDIA's and the xArm getting a
second copy that drifts from it.

DIFFERENTIAL IK IS ITERATIVE. One call moves the end effector a FRACTION of the
way to the goal; it does not arrive. Callers issue one step per physics frame
and watch for convergence with a tolerance and a frame budget. Nothing here
knows about goals, tolerances or budgets — that is the sequence's job.

WHAT THIS IS NOT
----------------
A motion planner. There is no collision model, no trajectory, no re-planning and
no guarantee that the straight line the end effector takes is free. The sequence
above it enforces the one clearance rule that matters (never move laterally
below the container rim) by choosing WAYPOINTS, not by planning around obstacles.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: Damping (lambda) for the least-squares inverse. Larger is more stable near
#: singularities and slower to converge; 0.05 is Isaac's own default and is what
#: the Panda has always run with.
DEFAULT_DAMPING = 0.05
#: Step scale. 1.0 takes the full damped step each frame.
DEFAULT_SCALE = 1.0


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.atleast_2d(np.asarray(q, dtype=float))
    return np.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], axis=-1)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (w, x, y, z) quaternions, batched."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def pose_error(current_position: np.ndarray, current_orientation: np.ndarray,
               goal_position: np.ndarray, goal_orientation: np.ndarray
               ) -> np.ndarray:
    """The 6-vector (linear, angular) error, shaped for the Jacobian solve.

    The angular part is the vector part of ``goal * conj(current)``, sign-fixed
    by the scalar part. Without that sign fix the solver takes the long way
    round a rotation that is a few degrees away — a wrist that spins 350 degrees
    to reach a 10-degree correction, which looks exactly like a broken IK.
    """
    q = quaternion_multiply(goal_orientation,
                            quaternion_conjugate(current_orientation))
    linear = np.atleast_2d(goal_position) - np.atleast_2d(current_position)
    angular = q[:, 1:] * np.sign(q[:, [0]])
    return np.expand_dims(np.concatenate([linear, angular], axis=-1), axis=2)


def damped_least_squares(jacobian: np.ndarray, error: np.ndarray, *,
                         damping: float = DEFAULT_DAMPING,
                         scale: float = DEFAULT_SCALE) -> np.ndarray:
    """delta_q for one step: scale * J^T (J J^T + lambda^2 I)^-1 * error."""
    transpose = np.swapaxes(jacobian, 1, 2)
    lam = np.eye(jacobian.shape[1]) * (float(damping) ** 2)
    inverse = np.linalg.inv(jacobian @ transpose + lam)
    return (float(scale) * transpose @ inverse @ error).squeeze(-1)


def differential_ik_step(jacobian: np.ndarray,
                         current_position: np.ndarray,
                         current_orientation: np.ndarray,
                         goal_position: Sequence[float],
                         goal_orientation: Sequence[float], *,
                         damping: float = DEFAULT_DAMPING,
                         scale: float = DEFAULT_SCALE) -> np.ndarray:
    """One damped-least-squares step towards a Cartesian goal.

    ``jacobian`` is [N, 6, arm_dof] for the END-EFFECTOR link only — selecting
    the right slice out of the articulation's full Jacobian tensor is the
    adapter's job, because which row an end-effector link occupies depends on
    whether the articulation has a fixed or a floating base.
    """
    goal_position = np.atleast_2d(np.asarray(goal_position, dtype=float))
    goal_orientation = np.atleast_2d(np.asarray(goal_orientation, dtype=float))
    error = pose_error(np.asarray(current_position, dtype=float),
                       np.asarray(current_orientation, dtype=float),
                       goal_position, goal_orientation)
    return damped_least_squares(jacobian, error, damping=damping, scale=scale)


__all__ = [
    "DEFAULT_DAMPING", "DEFAULT_SCALE", "differential_ik_step",
    "damped_least_squares", "pose_error", "quaternion_conjugate",
    "quaternion_multiply",
]
