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

import math
from typing import Optional, Sequence

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


def adaptive_damping(jacobian: np.ndarray, *, base: float,
                     singular: float = 0.0,
                     threshold: float = 0.0) -> float:
    """Damping that grows as the arm approaches a singularity.

    MEASURED, not assumed. Instrumenting the xArm 7 through one pick-and-place
    showed manipulability — sqrt(det(J Jᵀ)) — collapsing from 0.05–0.07 in the
    pick poses to 0.0013–0.0020 over the container: a 35–50x drop. A FIXED
    damping term is far too small there, so the solver returns enormous joint
    steps for a small task error, and those steps then saturate at the joint
    velocity limit. That is the visible oscillation.

    Chiaverini's variable damping: below the threshold, add a term that grows
    smoothly to ``singular`` as manipulability goes to zero. Above it, nothing
    changes — so a well-conditioned arm behaves exactly as before, and the
    Panda, which never approaches these configurations in this workcell, is
    unaffected unless its profile opts in.
    """
    if singular <= 0.0 or threshold <= 0.0:
        return base
    jjt = jacobian @ np.swapaxes(jacobian, -1, -2)
    det = float(np.linalg.det(jjt[0] if jjt.ndim == 3 else jjt))
    w = math.sqrt(max(det, 0.0))
    if w >= threshold:
        return base
    ratio = w / threshold
    return float(math.sqrt(base ** 2 + (singular ** 2) * (1.0 - ratio ** 2)))


def null_space_step(jacobian: np.ndarray, positions: np.ndarray,
                    rest: np.ndarray, gain: float,
                    damping: float) -> np.ndarray:
    """Pull the redundant degrees of freedom toward a preferred posture.

    WHY A 7-DOF ARM NEEDS THIS AND A 6-DOF ONE DOES NOT. The task constrains six
    degrees of freedom; a seven-joint arm has one left over, and damped least
    squares says nothing about where it should be. Nothing anchors it, so the
    null-space coordinate WANDERS between ticks.

    Measured on the xArm 7 over one placement: joint 7 travelled 5.89 rad to
    achieve 0.28 rad of net change — 21x — and joint 2 travelled 3.13 rad for
    0.22 rad, reversing direction 26 times. The end effector was converging the
    whole time. That is not a tracking error; it is the redundancy drifting.

    The projector ``(I - J⁺J)`` maps the posture correction into the null space,
    so it changes the arm's shape WITHOUT moving the end effector.
    """
    if gain <= 0.0:
        return np.zeros_like(positions)
    j = jacobian[0] if jacobian.ndim == 3 else jacobian
    lam = np.eye(j.shape[0]) * (damping ** 2)
    pinv = j.T @ np.linalg.inv(j @ j.T + lam)
    projector = np.eye(j.shape[1]) - pinv @ j
    return projector @ (gain * (rest - positions))


def clamp_step(delta: np.ndarray, max_step: float) -> np.ndarray:
    """Scale a joint step so no joint exceeds ``max_step``. Direction preserved.

    NOT a safety limiter — PhysX already enforces the joint velocity limits.
    This exists because the controller was routinely asking for MORE than the
    limit allows: measured joint deltas sat at 0.0524 rad every tick, which is
    exactly 3.14 rad/s at a 60 Hz step. When the commanded step is larger than
    the achievable one the arm moves open-loop toward a target it never reaches,
    and by the next tick the solver has changed its mind. Clamping below the
    limit makes commanded and achieved motion agree again.

    Scaled, not clipped per joint: clipping changes the DIRECTION of the step
    and would steer the end effector somewhere the solver did not ask for.
    """
    if max_step <= 0.0:
        return delta
    peak = float(np.max(np.abs(delta)))
    if peak <= max_step or peak == 0.0:
        return delta
    return delta * (max_step / peak)


def differential_ik_step(jacobian: np.ndarray,
                         current_position: np.ndarray,
                         current_orientation: np.ndarray,
                         goal_position: Sequence[float],
                         goal_orientation: Sequence[float], *,
                         damping: float = DEFAULT_DAMPING,
                         scale: float = DEFAULT_SCALE,
                         positions: Optional[np.ndarray] = None,
                         rest_posture: Optional[Sequence[float]] = None,
                         rest_gain: float = 0.0,
                         singular_damping: float = 0.0,
                         manipulability_threshold: float = 0.0,
                         max_joint_step: float = 0.0) -> np.ndarray:
    """One step towards a Cartesian goal, optionally stabilised.

    ``jacobian`` is [N, 6, arm_dof] for the END-EFFECTOR link only — selecting
    the right slice out of the articulation's full Jacobian tensor is the
    adapter's job, because which row an end-effector link occupies depends on
    whether the articulation has a fixed or a floating base.

    Every stabilisation term is OFF by default and is switched on per robot
    through its profile, so a robot that does not need one is unaffected. Each
    exists because an instrumented run measured the problem it solves — see the
    individual helpers.
    """
    goal_position = np.atleast_2d(np.asarray(goal_position, dtype=float))
    goal_orientation = np.atleast_2d(np.asarray(goal_orientation, dtype=float))
    error = pose_error(np.asarray(current_position, dtype=float),
                       np.asarray(current_orientation, dtype=float),
                       goal_position, goal_orientation)
    lam = adaptive_damping(jacobian, base=damping, singular=singular_damping,
                           threshold=manipulability_threshold)
    delta = damped_least_squares(jacobian, error, damping=lam, scale=scale)
    if rest_posture is not None and rest_gain > 0.0 and positions is not None:
        q = np.asarray(positions, dtype=float).reshape(-1)
        rest = np.asarray(rest_posture, dtype=float).reshape(-1)
        if q.shape == rest.shape:
            delta = delta + null_space_step(jacobian, q, rest, rest_gain, lam)
    return clamp_step(delta, max_joint_step)


__all__ = [
    "DEFAULT_DAMPING", "DEFAULT_SCALE", "adaptive_damping", "clamp_step",
    "null_space_step", "differential_ik_step",
    "damped_least_squares", "pose_error", "quaternion_conjugate",
    "quaternion_multiply",
]
