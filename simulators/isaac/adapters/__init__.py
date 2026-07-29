"""Robot adapters for the Isaac execution backend, and the factory that picks one.

The factory is the ONLY place a profile's ``adapter`` name becomes a class. It
refuses an unknown name rather than falling back to a default: a config file
that asks for an implementation nothing provides must stop the process, not
quietly run a different robot than the one the operator selected.

Isaac imports stay inside this package — importing ``build_adapter`` pulls in
``isaacsim.*``, which is why the registry, the contract and the tests import
``wisepack_core.robots`` instead.
"""

from __future__ import annotations

from typing import Callable, Dict

from wisepack_core.robots import RobotProfile

from .base import IsaacRobotAdapter, RobotModelError


def build_adapter(profile: RobotProfile) -> IsaacRobotAdapter:
    """The adapter for one robot profile.

    Imported lazily so that naming a robot does not load every adapter, and so
    that a syntax error in one adapter cannot stop an unrelated robot starting.
    """
    from .panda import PandaRobotAdapter                    # noqa: PLC0415
    from .xarm7 import XArm7RobotAdapter                    # noqa: PLC0415

    registry: Dict[str, Callable[[RobotProfile], IsaacRobotAdapter]] = {
        "panda": PandaRobotAdapter,
        "xarm7": XArm7RobotAdapter,
    }
    factory = registry.get(profile.adapter)
    if factory is None:
        raise RobotModelError(
            f"robot {profile.robot_id!r} asks for adapter "
            f"{profile.adapter!r}, which is not implemented; available "
            f"adapters: {sorted(registry)}. Fix `adapter:` in "
            "config/isaac_robots.yaml.",
            {"robot_id": profile.robot_id, "adapter": profile.adapter,
             "available": sorted(registry)})
    return factory(profile)


__all__ = ["IsaacRobotAdapter", "RobotModelError", "build_adapter"]
