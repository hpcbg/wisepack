"""Make wisepack_core importable from the tests without installing it.

The ROS workspace is not built when the pure-Python tests run (that is the whole
point of Phase 1), so the core package is added to sys.path directly.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CORE = os.path.join(REPO, "wisepack_ws", "src", "wisepack_core")
if CORE not in sys.path:
    sys.path.insert(0, CORE)

# The repository root, so the Isaac adapter is importable as
# `simulators.isaac.<module>`. Only its Isaac-FREE modules (config, result) may
# be imported here — `simulators/isaac/__init__.py` deliberately imports nothing,
# so this does not drag isaacsim into the ordinary test run.
if REPO not in sys.path:
    sys.path.insert(0, REPO)
