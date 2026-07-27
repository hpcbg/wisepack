"""Tunables for the WISEPACK Isaac Sim execution backend.

Deliberately free of Isaac imports so it can be read, validated and unit-tested
by the ordinary test suite on a machine with no GPU and no simulator.

Every value here is a PHYSICAL choice with a consequence, so each one says what
goes wrong if it is set badly. Numbers whose only justification would be "it
looked about right" are the ones that later get blamed on physics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

LOG_SCENE = "[isaac-scene]"
LOG_ROBOT = "[isaac-robot]"
LOG_BRIDGE = "[isaac-bridge]"
LOG_RESULT = "[isaac-result]"
LOG_APP = "[isaac-app]"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{name}={raw!r} is not a number; refusing to run a physical "
            "simulation with an unparseable tuning value") from exc


@dataclass
class MotionConfig:
    """Heights, tolerances and settle criteria for the pick-and-place sequence.

    All distances are METRES (Isaac world units). The WISEPACK plan arrives in
    millimetres and is converted in exactly one place — see
    ``wisepack_core.isaac_transform`` — so nothing here needs a unit factor.
    """

    #: How far above the item the gripper stops before descending. Too small and
    #: the fingers clip the item on the way in (PhysX resolves that as an impulse
    #: that flicks it across the table); too large and the descent wastes frames.
    pre_grasp_height: float = 0.12
    #: How far the item is raised after the gripper closes, before any lateral
    #: motion. Must clear the neighbouring items in the pick row, or the held
    #: item sweeps them off the table.
    lift_height: float = 0.25
    #: Vertical clearance kept above the container RIM while moving laterally.
    #: This is what stops a held cylinder being dragged through a wall — the
    #: single most likely way to produce a physically impossible "success".
    #:
    #: 0.10 rather than 0.15: the transit pose sits at rim + clearance directly
    #: above the bin, and every extra centimetre pushes the wrist into a higher,
    #: more folded configuration where the IK converges slowly. 0.10 still clears
    #: the widest item's radius (35 mm) with a wide margin.
    container_clearance: float = 0.10
    #: Height above the planned target pose at which the gripper opens. Small,
    #: because the drop is the part physics decides: a larger value trades
    #: placement accuracy for clearance. It is NOT zero — releasing at the exact
    #: target means the fingers are inside the space the item must occupy.
    drop_height: float = 0.06
    #: Retreat height above the rim after releasing, before moving laterally.
    retreat_height: float = 0.20

    #: End-effector position error under which a servo goal counts as reached.
    #: Differential IK converges asymptotically, so without a tolerance the
    #: sequence would wait forever for an exact match.
    goal_tolerance: float = 0.015
    #: Physics frames a servo goal may take before the sequence gives up and
    #: reports ITEM_FAILED. Differential IK converges asymptotically, so this is
    #: a generous ceiling rather than an expected duration — a goal that is
    #: reachable usually converges in a third of it, and one that is not should
    #: fail rather than hang forever.
    #:
    #: 360 (~6 s) was too tight and was MEASURED to be: a pre-grasp descent that
    #: had converged comfortably in one run timed out in another, from the same
    #: pose to the same target. A budget that fails intermittently is worse than
    #: a slow one, because it reads as a physics problem.
    max_frames_per_goal: int = 900
    #: Frames the goal must stay within tolerance before the state advances.
    #: A single in-tolerance frame can be a fly-through rather than an arrival.
    dwell_frames: int = 8
    #: Frames held while the gripper opens or closes. The Panda's fingers are
    #: position-controlled, so this is a settling allowance, not a trajectory.
    gripper_frames: int = 45

    #: Settling. The item is NOT reported complete when the gripper opens — only
    #: when the rigid body has actually come to rest, or the timeout expires.
    settle_timeout: float = 6.0
    linear_velocity_threshold: float = 0.01        # m/s
    angular_velocity_threshold: float = 0.10       # rad/s
    #: How long the body must stay below BOTH thresholds to count as settled.
    #: An instantaneous zero happens at the top of a bounce.
    settle_stable_time: float = 0.35

    #: How far an item may be from the source pose the plan was built against
    #: before a pick is REFUSED. Items settle and can be nudged by a neighbour,
    #: so a few centimetres is normal; a large drift means the physical scene
    #: and the plan disagree, and moving toward it would be uncontrolled motion
    #: rather than a pick.
    max_source_drift_m: float = 0.08

    #: Yaw offset, in degrees, applied to the top-down gripper orientation when
    #: grasping. The pick row lays every item along world X and the Panda hand's
    #: fingers must close ACROSS that axis. 0 means the default downward
    #: orientation already closes along world Y; it is exposed because the finger
    #: axis is a property of the shipped asset's hand frame, not of this code,
    #: and correcting it must not require editing the state machine.
    grasp_yaw_offset_deg: float = 0.0

    def validate(self) -> None:
        problems = []
        if self.drop_height <= 0.0:
            problems.append("drop_height must be > 0: releasing at the exact "
                            "target puts the fingers inside the item's own space")
        if self.container_clearance < self.drop_height:
            problems.append(
                f"container_clearance ({self.container_clearance}) is below "
                f"drop_height ({self.drop_height}); the item would be lowered "
                "past the rim before it is above the target")
        if self.lift_height <= self.pre_grasp_height:
            problems.append(
                f"lift_height ({self.lift_height}) must exceed pre_grasp_height "
                f"({self.pre_grasp_height}) or the item is never raised clear")
        if self.settle_timeout <= self.settle_stable_time:
            problems.append("settle_timeout must exceed settle_stable_time")
        if self.linear_velocity_threshold <= 0 or self.angular_velocity_threshold <= 0:
            problems.append("velocity thresholds must be > 0")
        if problems:
            raise ValueError("Isaac motion configuration is inconsistent:\n  - "
                             + "\n  - ".join(problems))

    @staticmethod
    def from_env() -> "MotionConfig":
        """Build from WISEPACK_ISAAC_* environment overrides.

        Overriding is per-value and explicit. A whole config file would be one
        more thing to keep in step with the code for a handful of numbers.
        """
        cfg = MotionConfig(
            pre_grasp_height=_env_float("WISEPACK_ISAAC_PRE_GRASP_HEIGHT", 0.12),
            lift_height=_env_float("WISEPACK_ISAAC_LIFT_HEIGHT", 0.25),
            container_clearance=_env_float("WISEPACK_ISAAC_CONTAINER_CLEARANCE", 0.10),
            drop_height=_env_float("WISEPACK_ISAAC_DROP_HEIGHT", 0.06),
            retreat_height=_env_float("WISEPACK_ISAAC_RETREAT_HEIGHT", 0.20),
            goal_tolerance=_env_float("WISEPACK_ISAAC_GOAL_TOLERANCE", 0.015),
            settle_timeout=_env_float("WISEPACK_ISAAC_SETTLE_TIMEOUT", 6.0),
            linear_velocity_threshold=_env_float(
                "WISEPACK_ISAAC_LINEAR_VELOCITY_THRESHOLD", 0.01),
            angular_velocity_threshold=_env_float(
                "WISEPACK_ISAAC_ANGULAR_VELOCITY_THRESHOLD", 0.10),
            settle_stable_time=_env_float("WISEPACK_ISAAC_SETTLE_STABLE_TIME", 0.35),
            grasp_yaw_offset_deg=_env_float("WISEPACK_ISAAC_GRASP_YAW_OFFSET_DEG", 0.0),
            max_source_drift_m=_env_float("WISEPACK_ISAAC_MAX_SOURCE_DRIFT", 0.08),
        )
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pre_grasp_height": self.pre_grasp_height,
            "lift_height": self.lift_height,
            "container_clearance": self.container_clearance,
            "drop_height": self.drop_height,
            "retreat_height": self.retreat_height,
            "goal_tolerance": self.goal_tolerance,
            "settle_timeout": self.settle_timeout,
            "linear_velocity_threshold": self.linear_velocity_threshold,
            "angular_velocity_threshold": self.angular_velocity_threshold,
            "settle_stable_time": self.settle_stable_time,
            "grasp_yaw_offset_deg": self.grasp_yaw_offset_deg,
            "max_source_drift_m": self.max_source_drift_m,
        }


@dataclass
class PhysicsConfig:
    """Material and solver properties for the procedural scene.

    Friction values are ordinary engineering figures for dry steel-on-steel and
    steel-on-plastic. They are not measurements of anything, and nothing in the
    reported results depends on their exact value — they exist so items rest
    where they land instead of sliding indefinitely.
    """

    physics_dt: float = 1.0 / 60.0
    #: Steel offcuts on a steel table.
    item_static_friction: float = 0.7
    item_dynamic_friction: float = 0.55
    #: Low: a bouncing cylinder takes far longer to settle and adds nothing.
    item_restitution: float = 0.05
    #: kg/m^3. The generator already computes each item's mass from its real
    #: alloy density, and that mass is applied directly, so this is only the
    #: fallback for geometry with no WISEPACK item behind it.
    default_density: float = 7850.0
    table_static_friction: float = 0.8
    table_dynamic_friction: float = 0.6
    container_static_friction: float = 0.8
    container_dynamic_friction: float = 0.6
    container_restitution: float = 0.02
    #: Wall and floor thickness of the procedurally built container, metres.
    container_wall_thickness: float = 0.008
    #: CPU physics by default: this scene is four rigid bodies and one
    #: articulation, where GPU pipeline setup costs more than it saves, and CPU
    #: keeps the run reproducible across machines.
    device: str = "cpu"

    @staticmethod
    def from_env() -> "PhysicsConfig":
        return PhysicsConfig(
            device=os.environ.get("WISEPACK_ISAAC_PHYSICS_DEVICE", "cpu"))


@dataclass
class AppConfig:
    """Everything one Isaac run needs."""

    preset: str = "isaac_cylinders_smoke"
    seed: int = 42
    headless: bool = False
    #: Exit after the run reaches RUN_COMPLETED / RUN_FAILED. Used by the smoke
    #: validator; the dashboard demonstration keeps the window open instead.
    exit_on_complete: bool = False
    #: Hard ceiling on wall-clock seconds. 0 disables it. The validator sets it
    #: so a wedged run fails the script instead of hanging CI forever.
    max_runtime_s: float = 0.0
    motion: MotionConfig = field(default_factory=MotionConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset,
            "seed": self.seed,
            "headless": self.headless,
            "exit_on_complete": self.exit_on_complete,
            "max_runtime_s": self.max_runtime_s,
            "motion": self.motion.to_dict(),
            "physics_device": self.physics.device,
        }


__all__ = [
    "LOG_APP", "LOG_BRIDGE", "LOG_RESULT", "LOG_ROBOT", "LOG_SCENE",
    "AppConfig", "MotionConfig", "PhysicsConfig",
]
