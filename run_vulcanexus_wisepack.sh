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

# Exit codes, so a caller can tell a broken invocation from a failed validation:
#   2  no target named / invalid target (traversal, absolute path)
#   3  target file does not exist
#   4  target exists but is not executable
#   N  whatever the validation script itself returned
EXIT_USAGE=2
EXIT_MISSING=3
EXIT_NOT_EXEC=4

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
    exit "$EXIT_USAGE"
fi

# The target is executed as ./"$target" INSIDE the container, relative to the
# mounted repo. Anything that could escape that root, or that is not a plain
# relative path, is rejected here rather than producing a confusing failure
# three layers down.
case "$TARGET" in
    /*)
        echo "[host] ERROR: target must be a path relative to the repository root," >&2
        echo "               not an absolute path: $TARGET" >&2
        exit "$EXIT_USAGE" ;;
    *..*)
        echo "[host] ERROR: target must not contain '..': $TARGET" >&2
        exit "$EXIT_USAGE" ;;
    "" | */)
        echo "[host] ERROR: target is not a file: $TARGET" >&2
        exit "$EXIT_USAGE" ;;
esac

if [ -d "$REPO/$TARGET" ]; then
    echo "[host] ERROR: '$TARGET' is a directory, not a validation script." >&2
    exit "$EXIT_USAGE"
fi
if [ ! -f "$REPO/$TARGET" ]; then
    echo "[host] ERROR: '$TARGET' not found in $REPO" >&2
    exit "$EXIT_MISSING"
fi
if [ ! -x "$REPO/$TARGET" ]; then
    echo "[host] ERROR: '$TARGET' is not executable. Fix with:" >&2
    echo "               chmod +x $TARGET" >&2
    exit "$EXIT_NOT_EXEC"
fi


# ── Refuse to run alongside another WISEPACK stack on the same DDS domain ────
# Two orchestrators on one ROS_DOMAIN_ID both publish the canonical topics, and
# a reader then sees whichever wrote last. Measured symptom: the validation
# reported stage NEXT_ITEM *before* approval, with progress 0% — two different
# runs answering the same question. This is the "clean stale ROS processes"
# rule, done by CONTAINER NAME so it can never touch another project's stack.
wisepack_reap_stale() {
    local own="$1"
    docker rm -f "$own" >/dev/null 2>&1 || true

    local others
    others="$(docker ps --filter "ancestor=${IMAGE}" --format '{{.Names}}' \
              | grep -v "^${own}$" || true)"
    if [ -n "$others" ]; then
        if [ "${WISEPACK_REAP_OTHERS:-1}" = "1" ]; then
            echo "[host] stopping other WISEPACK containers on ROS_DOMAIN_ID=$ROS_DOMAIN_ID:"
            printf '         %s\n' $others
            echo "$others" | xargs -r docker rm -f >/dev/null 2>&1 || true
        else
            echo "[host] WARNING: other WISEPACK containers are running and will" >&2
            echo "                share ROS_DOMAIN_ID=$ROS_DOMAIN_ID:" >&2
            printf '                %s\n' $others >&2
            echo "                results will be ambiguous. WISEPACK_REAP_OTHERS=1 to stop them." >&2
        fi
    fi
}

CONTAINER_NAME="${WISEPACK_CONTAINER:-wisepack-validate}"
wisepack_reap_stale "$CONTAINER_NAME"

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

echo "[host] resolved target : $REPO/$TARGET"
echo "[host] arguments      : ${*:-<none>}"
echo "[host] launching $IMAGE (host net + host ipc) ..."
exec docker run --rm -i $TTY_FLAG \
    --name "$CONTAINER_NAME" \
    --net=host \
    --ipc=host \
    --privileged \
    -e "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" \
    -e "WISEPACK_RESULTS_DIR=$RESULTS_DIR" \
    -e WISEPACK_SKIP_BUILD \
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
        # NO `set -u` before this line: Vulcanexus setup.bash dereferences
        # COLCON_TRACE while it is still unset, so nounset aborts before the ROS
        # environment is even sourced.
        source /opt/vulcanexus/jazzy/setup.bash
        set -u

        # TARGET is a HOST variable. It has to arrive here as a positional
        # argument: this script body is single-quoted, so the host shell never
        # expands it, and inside the container it is simply unset. Relying on
        # the environment made `exec ./"$TARGET"` become `exec ./`, which is the
        # repository directory — "cannot execute: Is a directory".
        target="$1"
        shift

        if [ -z "$target" ]; then
            echo "[container] ERROR: no target was passed into the container." >&2
            exit 2
        fi
        if [ ! -f "./$target" ]; then
            echo "[container] ERROR: ./$target is missing inside the container." >&2
            exit 3
        fi
        if [ ! -x "./$target" ]; then
            echo "[container] ERROR: ./$target is not executable." >&2
            exit 4
        fi

        WS="wisepack_ws"
        # Always build. An incremental colcon build is a few seconds when the
        # workspace is current, and silently validating a STALE install is how a
        # green run ends up proving nothing about the code in front of you.
        if [ "${WISEPACK_SKIP_BUILD:-0}" = "1" ]; then
            echo "[container] WISEPACK_SKIP_BUILD=1 — using the existing install/ as-is"
            if [ ! -f "$WS/install/setup.bash" ]; then
                echo "[container] ERROR: nothing built and the build was skipped." >&2
                exit 1
            fi
        else
            echo "[container] colcon build --symlink-install (incremental) ..."
            ( cd "$WS" && colcon build --symlink-install ) || {
                echo "[container] ERROR: colcon build failed." >&2
                exit 1
            }
        fi

        echo "[container] running ./$target $*"
        exec "./$target" "$@"
    ' _ "$TARGET" "$@"
