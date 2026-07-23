#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_vulcanexus_wisepack.sh
#
# Runs a WISEPACK validation inside the WISEPACK Vulcanexus image. Direct
# counterpart of TEMPO's run_vulcanexus_tempo.sh and HARMONY's
# run_vulcanexus_dds_validation.sh, and the same deal: the host needs Docker and
# nothing else.
#
#     ./run_vulcanexus_wisepack.sh validate_wisepack_e2e.sh
#     ./run_vulcanexus_wisepack.sh validate_fiware_action_log.sh
#     ./run_vulcanexus_wisepack.sh measure_dds_fiware_latency.sh
#
# The target is REQUIRED. For the full acceptance demo use ./run_wisepack_demo.sh.
#
# The container runs with host networking + host IPC, which is what makes the DDS
# result mean anything: Fast DDS discovery and shared-memory transport behave as
# they would on the cell, not as they would across a bridged Docker network. It
# is also what lets the container's ROS nodes reach an Orion-LD running on the
# host's port 1026.
#
# The repo is mounted at the SAME absolute path it has on the host, so
# realpath-based paths resolve identically in both places (HARMONY does this for
# the same reason).
# ---------------------------------------------------------------------------
set -u

REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
IMAGE="${WISEPACK_IMAGE:-wisepack:jazzy}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

TARGET="${1:-}"
shift || true

if [ -z "$TARGET" ]; then
    echo "[host] ERROR: no validation script named." >&2
    echo "" >&2
    echo "usage: ./run_vulcanexus_wisepack.sh <script.sh> [args...]" >&2
    echo "" >&2
    echo "available:" >&2
    for s in "$REPO"/validate_*.sh "$REPO"/measure_*.sh "$REPO"/generate_*.sh; do
        [ -f "$s" ] && echo "    $(basename "$s")" >&2
    done
    echo "" >&2
    echo "for the full acceptance demo use: ./run_wisepack_demo.sh" >&2
    exit 2
fi

if [ ! -f "$REPO/$TARGET" ]; then
    echo "[host] ERROR: '$TARGET' not found in $REPO" >&2
    exit 1
fi

# --ipc=host means orphaned DDS segments from a previous run can hang every
# ros2 command with no error message. Clear them first.
"$REPO/scripts/clean_dds_shm.sh"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[host] image $IMAGE not present — building it (Vulcanexus Jazzy + py_trees/fastapi) ..."
    docker build -f "$REPO/Dockerfile.wisepack" -t "$IMAGE" "$REPO" || exit 1
fi

TTY_FLAG=""
[ -t 1 ] && TTY_FLAG="-t"

# Results must land on the mounted repo: the container's /tmp is ephemeral and
# the artefacts would vanish with the container.
RESULTS_DIR="${WISEPACK_RESULTS_DIR:-$REPO/results}"
mkdir -p "$RESULTS_DIR"

echo "[host] launching $IMAGE (host net + host ipc) — running ./$TARGET $* ..."
exec docker run --rm -i $TTY_FLAG \
    --net=host \
    --ipc=host \
    --privileged \
    -e "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" \
    -e "WISEPACK_RESULTS_DIR=$RESULTS_DIR" \
    -e ORION \
    -e DDS_LATENCY_WARMUP \
    -e DDS_LATENCY_SAMPLES \
    -e DDS_LATENCY_TIMEOUT_SEC \
    -e WISEPACK_PRESET \
    -e WISEPACK_SEED \
    -v "$REPO:$REPO" \
    -w "$REPO" \
    "$IMAGE" \
    bash -lc '
        source /opt/vulcanexus/jazzy/setup.bash
        WS="wisepack_ws"
        if [ ! -f "$WS/install/setup.bash" ]; then
            echo "[container] $WS not built — running colcon build --symlink-install ..."
            ( cd "$WS" && colcon build --symlink-install ) || {
                echo "[container] ERROR: colcon build failed." >&2
                exit 1
            }
        fi
        exec ./"$TARGET" "$@"
    ' _ "$@"
