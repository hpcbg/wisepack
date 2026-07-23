#!/usr/bin/env bash
# Build the WISEPACK ROS 2 workspace. Runs INSIDE the Vulcanexus container.
#
# Mirrors TEMPO's build_tempo_ws.sh, which mirrors HARMONY's colcon sequence.
set -euo pipefail

VULCANEXUS_SETUP="${VULCANEXUS_SETUP:-/opt/vulcanexus/jazzy/setup.bash}"
WS_DIR="${WS_DIR:-/wisepack/wisepack_ws}"

if [[ ! -f "$VULCANEXUS_SETUP" ]]; then
    echo "ERROR: $VULCANEXUS_SETUP not found — are you inside the Vulcanexus image?" >&2
    echo "       Use ./run_vulcanexus_wisepack.sh, which provides it." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$VULCANEXUS_SETUP"
cd "$WS_DIR"

# rosdep is best-effort: the image already carries every declared dependency and
# rosdep needs network. Never fail the build on it.
if command -v rosdep >/dev/null 2>&1; then
    echo "==> rosdep (best-effort)"
    rosdep install --from-paths src --ignore-src -r -y 2>/dev/null \
        || echo "    rosdep skipped (offline or already satisfied)"
fi

echo "==> colcon build --symlink-install"
colcon build --symlink-install

# shellcheck disable=SC1091
source "$WS_DIR/install/setup.bash"
echo "==> build OK — workspace sourced from $WS_DIR/install/setup.bash"
