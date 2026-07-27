#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_wisepack_isaac.sh — start the Isaac Sim physical execution backend.
#
#     ./scripts/run_wisepack_isaac.sh                    # GUI if a display exists
#     WISEPACK_ISAAC_HEADLESS=1 ./scripts/run_wisepack_isaac.sh
#     ISAAC_SIM_ROOT=/opt/isaac-sim ./scripts/run_wisepack_isaac.sh
#     ./scripts/run_wisepack_isaac.sh --self-test        # build, report READY, exit
#
# Isaac runs on the HOST in its own bundled Python. WISEPACK may run in Docker
# with host networking; the two meet on the DDS wire, on a shared ROS_DOMAIN_ID,
# and nowhere else.
#
# WHY THIS SCRIPT SCRUBS THE ROS ENVIRONMENT
# ------------------------------------------
# Isaac Sim ships its OWN ROS 2 Jazzy build, compiled against its own Python ABI.
# Enabling the isaacsim.ros2.bridge extension puts that build on the path and
# `import rclpy` resolves to it. If the calling shell has sourced
# /opt/ros/jazzy/setup.bash (or Vulcanexus), PYTHONPATH carries a SECOND,
# ABI-incompatible rclpy ahead of Isaac's, and the failure is an import-time
# crash deep inside rclpy's C extension — which reads as "Isaac is broken".
#
# ROS_DOMAIN_ID is the one ROS variable that is deliberately kept and passed
# through: it is what puts the simulator on the same DDS domain as WISEPACK.
# ---------------------------------------------------------------------------
set -u

REPO="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
LOG="[isaac-launch]"

# Host-specific settings (the SSH port used in the port-forward hint). Resolved,
# never guessed — see scripts/lib_local_env.sh for the precedence order.
# shellcheck source=scripts/lib_local_env.sh
source "$REPO/scripts/lib_local_env.sh"
wisepack_load_local_env "$REPO"
wisepack_resolve_ssh_port || true

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
PRESET="${WISEPACK_PRESET:-isaac_cylinders_smoke}"
SEED="${WISEPACK_SEED:-42}"

# --- Isaac Sim discovery ----------------------------------------------------
# Explicit ISAAC_SIM_ROOT wins. Otherwise search known locations NEWEST FIRST,
# and never silently settle for an older major version: a 5.x install has a
# different core API and would fail with import errors that look like our bugs.
find_isaac_root() {
    if [ -n "${ISAAC_SIM_ROOT:-}" ]; then
        printf '%s' "$ISAAC_SIM_ROOT"
        return
    fi
    local candidates=(
        /data/isaac-sim/isaac-sim-6.0.1
        /data/isaac-sim/isaac-sim-6.0
        "$HOME/isaacsim"
        "$HOME/.local/share/ov/pkg/isaac-sim-6.0.1"
        /opt/isaac-sim
        /isaac-sim
    )
    local path
    for path in "${candidates[@]}"; do
        [ -x "$path/python.sh" ] && { printf '%s' "$path"; return; }
    done
    # Last resort: any isaac-sim-6.* under /data/isaac-sim, highest version first.
    for path in $(ls -d /data/isaac-sim/isaac-sim-6.* 2>/dev/null | sort -Vr); do
        [ -x "$path/python.sh" ] && { printf '%s' "$path"; return; }
    done
    printf ''
}

ISAAC_ROOT="$(find_isaac_root)"
if [ -z "$ISAAC_ROOT" ]; then
    echo "$LOG ERROR: no Isaac Sim installation found." >&2
    echo "$LOG   set ISAAC_SIM_ROOT=/path/to/isaac-sim, or install Isaac Sim 6.0.1." >&2
    echo "$LOG   searched: /data/isaac-sim/isaac-sim-6.*, ~/isaacsim, /opt/isaac-sim" >&2
    exit 4
fi

# Verify the bundled launcher BEFORE starting anything. A root that exists but
# has no python.sh produces a confusing 'command not found' several steps later.
if [ ! -x "$ISAAC_ROOT/python.sh" ]; then
    echo "$LOG ERROR: $ISAAC_ROOT has no executable python.sh." >&2
    echo "$LOG   that is not an Isaac Sim installation root." >&2
    exit 4
fi

ISAAC_VERSION="unknown"
[ -f "$ISAAC_ROOT/VERSION" ] && ISAAC_VERSION="$(cat "$ISAAC_ROOT/VERSION")"
case "$ISAAC_VERSION" in
    6.*) ;;
    *)  echo "$LOG WARNING: $ISAAC_ROOT reports version '$ISAAC_VERSION'." >&2
        echo "$LOG   this backend is written against Isaac Sim 6.0.1 and uses the" >&2
        echo "$LOG   isaacsim.core.experimental API. Older majors will fail to import." >&2
        ;;
esac

# --- display / headless -----------------------------------------------------
HEADLESS="${WISEPACK_ISAAC_HEADLESS:-}"
if [ -z "$HEADLESS" ]; then
    # GUI is the normal default for the dashboard demonstration, but only when a
    # display actually exists. Defaulting to GUI over SSH produces a crash inside
    # the renderer rather than a useful message.
    if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
        HEADLESS=0
    else
        HEADLESS=1
        echo "$LOG no DISPLAY — falling back to headless mode"
    fi
fi

ISAAC_ARGS=(--preset "$PRESET" --seed "$SEED")
[ "$HEADLESS" = "1" ] && ISAAC_ARGS+=(--headless)
ISAAC_ARGS+=("$@")

# --- WebRTC livestreaming (opt-in) -----------------------------------------
STREAMING="${WISEPACK_ISAAC_STREAMING:-0}"
STREAM_HOST="${WISEPACK_ISAAC_STREAM_HOST:-127.0.0.1}"
SIGNAL_PORT="${WISEPACK_ISAAC_SIGNAL_PORT:-49100}"
STREAM_PORT="${WISEPACK_ISAAC_STREAM_PORT:-47998}"
VIEWER_PORT="${WISEPACK_ISAAC_VIEWER_PORT:-0}"

if [ "$STREAMING" = "1" ]; then
    # Verify the extensions EXIST before launching. Kit reports a missing
    # livestream extension as a warning buried in a few thousand startup lines
    # and then runs perfectly happily with no stream at all, so the operator
    # waits for a viewer URL that is never coming.
    missing=""
    for ext in omni.kit.livestream.app omni.kit.livestream.webrtc; do
        ls -d "$ISAAC_ROOT"/ext*/"$ext"* >/dev/null 2>&1 || missing="$missing $ext"
    done
    if [ -n "$missing" ]; then
        echo "$LOG ERROR: Isaac Sim at $ISAAC_ROOT is missing required" >&2
        echo "$LOG   livestream extension(s):$missing" >&2
        echo "$LOG   run without WISEPACK_ISAAC_STREAMING=1, or install them." >&2
        exit 6
    fi

    # Refuse to start a SECOND stream server on an occupied signal port. Kit
    # silently falls back to another free port, so the URL published here would
    # point at a different, older stream — which shows a picture of the wrong
    # thing, the hardest kind of wrong to notice.
    if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${SIGNAL_PORT} "; then
        echo "$LOG ERROR: signal port ${SIGNAL_PORT} is already in use." >&2
        echo "$LOG   another Isaac stream is probably running. Stop it, or set" >&2
        echo "$LOG   WISEPACK_ISAAC_SIGNAL_PORT to a free port." >&2
        exit 6
    fi

    VIEWER_URL="${WISEPACK_ISAAC_STREAM_URL:-http://${STREAM_HOST}:$([ "$VIEWER_PORT" != "0" ] && echo "$VIEWER_PORT" || echo "$SIGNAL_PORT")}"
    export WISEPACK_ISAAC_STREAMING=1
    export WISEPACK_ISAAC_STREAM_HOST="$STREAM_HOST"
    export WISEPACK_ISAAC_SIGNAL_PORT="$SIGNAL_PORT"
    export WISEPACK_ISAAC_STREAM_PORT="$STREAM_PORT"
    export WISEPACK_ISAAC_VIEWER_PORT="$VIEWER_PORT"
    export WISEPACK_ISAAC_STREAM_URL="$VIEWER_URL"
    echo "$LOG streaming   : WebRTC signal ${STREAM_HOST}:${SIGNAL_PORT}, media UDP ${STREAM_PORT}"
    echo "$LOG viewer URL  : $VIEWER_URL"
    echo "$LOG remote view : ssh -p \"\${WISEPACK_SSH_PORT}\" -L ${SIGNAL_PORT}:127.0.0.1:${SIGNAL_PORT} <user>@<this-host>"
    hint="$(wisepack_ssh_port_hint)"
    [ -n "$hint" ] && echo "$LOG   note: $hint" >&2
    # The stream is unauthenticated. Say so once, loudly, at the point of use.
    case "$STREAM_HOST" in
        127.0.0.1|localhost|::1) ;;
        *) echo "$LOG WARNING: the WebRTC stream has NO authentication and NO" >&2
           echo "$LOG   encryption, and you have bound it to '$STREAM_HOST'." >&2
           echo "$LOG   Prefer loopback + SSH port forwarding, or restrict the" >&2
           echo "$LOG   ports by firewall to a single client address." >&2 ;;
    esac
else
    echo "$LOG streaming   : disabled (WISEPACK_ISAAC_STREAMING=1 to enable)"
fi

echo "$LOG Isaac Sim   : $ISAAC_ROOT (version $ISAAC_VERSION)"
echo "$LOG scenario    : preset=$PRESET seed=$SEED"
echo "$LOG ROS_DOMAIN_ID: $ROS_DOMAIN_ID   headless=$HEADLESS"
echo "$LOG results     : ${WISEPACK_RESULTS_DIR:-<unset>}"

# --- environment isolation --------------------------------------------------
# Everything ROS-related is removed except ROS_DOMAIN_ID. See the header for why:
# a host rclpy ahead of Isaac's on PYTHONPATH crashes inside its C extension.
unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PYTHON_VERSION
unset LD_LIBRARY_PATH
unset ROS_PACKAGE_PATH

# ...and then Isaac's OWN ROS environment is put back, using Isaac's own script.
#
# This is not a contradiction of the isolation rule, it is the point of it.
# Scrubbing alone is not enough: Isaac's internally-built ROS 2 libraries live in
# exts/isaacsim.ros2.core/<distro>/lib and are NOT on any default search path, so
# with LD_LIBRARY_PATH empty the bridge fails to load librmw_implementation.so
# and `import rclpy` then raises ModuleNotFoundError. Measured on this machine:
#
#     [Error] [isaacsim.ros2.core.impl.extension] ROS2 Bridge startup failed
#     Could not load .../librmw_implementation.so.
#     Error: libament_index_cpp.so: cannot open shared object file
#     ModuleNotFoundError: No module named 'rclpy'
#
# setup_ros_env.sh is shipped BY Isaac Sim and only acts when ROS_DISTRO is
# unset — which is exactly the state the scrub above leaves. It sets ROS_DISTRO
# from the Ubuntu release, adds Isaac's own lib directory, and defaults
# RMW_IMPLEMENTATION to Fast DDS. It never touches /opt/ros.
#
# `set +u` around it for the same reason run_wisepack_dashboard.sh drops nounset
# around Vulcanexus's setup.bash: the vendor script tests `[ -z "$ROS_DISTRO" ]`
# on a variable it has not defined, which under `set -u` aborts before any of the
# environment is applied. Measured here as:
#     setup_ros_env.sh: line 37: ROS_DISTRO: unbound variable
set +u
# shellcheck source=/dev/null
source "$ISAAC_ROOT/setup_ros_env.sh"
set -u

# Fast DDS on both ends: WISEPACK's Vulcanexus image and Isaac's bundled bridge
# must speak the same RMW or they simply never discover each other.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
echo "$LOG ROS runtime : Isaac-internal ${ROS_DISTRO:-?} (${RMW_IMPLEMENTATION})"
# UNBUFFERED, and this is not cosmetic. Python block-buffers stdout when it is a
# pipe or a file rather than a terminal, so the [isaac-*] progress lines sit in a
# 8 KiB buffer indefinitely. Both callers that matter read this output as a
# stream — run_wisepack_dashboard.sh greps the log for the READY line before it
# allows physical execution, and validate_isaac_sim.sh greps for the SMOKE-*
# markers — and both would have waited out their whole timeout on a simulator
# that was working perfectly. Measured while bringing this up: 23 kB of Kit's own
# (unbuffered, C++) logging with not one of our own lines among it.
export PYTHONUNBUFFERED=1
export WISEPACK_PRESET="$PRESET"
export WISEPACK_SEED="$SEED"
[ -n "${WISEPACK_RESULTS_DIR:-}" ] && export WISEPACK_RESULTS_DIR

exec "$ISAAC_ROOT/python.sh" "$REPO/simulators/isaac/wisepack_isaac.py" \
    "${ISAAC_ARGS[@]}"
