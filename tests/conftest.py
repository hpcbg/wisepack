"""Make wisepack_core importable from the tests without installing it.

The ROS workspace is not built when the pure-Python tests run (that is the whole
point of Phase 1), so the core package is added to sys.path directly.
"""
import os
import sys

CORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "wisepack_ws", "src", "wisepack_core")
if CORE not in sys.path:
    sys.path.insert(0, CORE)
