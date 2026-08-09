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


# THE SAVED CALIBRATION IS REDIRECTED FOR THE WHOLE TEST SESSION.
#
# `PlaneCalibration` persists a measured homography, and two tests run the real
# detector over a real frame that contains the calibration sheet — so without
# this, running the suite WROTE A CALIBRATION INTO `config/`. A test run must
# not leave a measurement of somebody's table in the repository, and one that
# did was caught only because an unrelated test scans tracked files.
#
# Set before any test imports `calibration`, and pointed at a path under the
# pytest temp root so each run is independent.
import tempfile                                                  # noqa: E402

_CALIBRATION_TMP = tempfile.mkdtemp(prefix="wisepack-calibration-")
os.environ.setdefault(
    "WISEPACK_PERCEPTION_CALIBRATION_FILE",
    os.path.join(_CALIBRATION_TMP, "perception_calibration.json"))
