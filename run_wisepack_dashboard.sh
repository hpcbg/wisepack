#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_wisepack_dashboard.sh — one command: the WISEPACK stack AND its dashboard.
#
#     ./run_wisepack_dashboard.sh           # live: ROS 2/DDS stack + dashboard
#     ./run_wisepack_dashboard.sh fiware    # live + Orion-LD, state read from FIWARE
#     ./run_wisepack_dashboard.sh sim       # presentation only: no ROS, no FIWARE
#
# LIVE MODE brings the whole loop up inside the Vulcanexus image (orchestrator +
# perception simulator + Digital Twin validator) and attaches the dashboard. The
# dashboard observes every state topic and publishes ONLY the two operator
# topics, so an approval travels the same ROS 2 -> DDS path an external NGSI-LD
# client would use.
#
# SIM MODE runs the FastAPI app against the same wisepack_core workflow engine,
# with no ROS, no FIWARE and no Docker if the host has fastapi. It is NOT a
# separate animation: it is the identical domain logic and the identical
# optimizer, with a different transport. The header badge says SIMULATED.
#
# Port 8080 by default, because the container runs with --net=host so a clash is
# a hard bind failure, not a fallback. Override: PORT=9000 ./run_wisepack_dashboard.sh
# ---------------------------------------------------------------------------
set -u

REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
IMAGE="${WISEPACK_IMAGE:-wisepack:jazzy}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
PORT="${PORT:-8080}"
MODE="${1:-ros}"
PRESET="${WISEPACK_PRESET:-mixed_pipes_dense}"
SEED="${WISEPACK_SEED:-42}"
URL="http://127.0.0.1:${PORT}"
DDS_DIR="$REPO/wisepack_ws/src/wisepack_fiware/dds"

open_browser() {
    ( sleep 3
      command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 && exit 0
      command -v open     >/dev/null 2>&1 && open     "$URL" >/dev/null 2>&1 && exit 0
      true ) &
}

if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    echo "[dashboard] ERROR: port ${PORT} is already in use." >&2
    echo "[dashboard] pick another:  PORT=9000 ./run_wisepack_dashboard.sh $MODE" >&2
    exit 1
fi

# ---- sim mode: no ROS, and no Docker if the host can run it directly --------
if [ "$MODE" = "sim" ]; then
    # No colcon build here on purpose: sim mode imports wisepack_core straight
    # from wisepack_ws/src, so it is ALWAYS the current source. There is no
    # install/ to go stale.
    echo "[dashboard] sim mode — no ROS, no FIWARE, no hardware."
    echo "[dashboard] Packing figures are MEASURED by the same optimizer the live"
    echo "[dashboard] stack runs. Execution outcomes are SIMULATED and labelled so."
    echo "[dashboard] open $URL"

    if python3 -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
        open_browser
        exec python3 "$REPO/web/app.py" --source sim --port "$PORT" \
            --preset "$PRESET" --seed "$SEED"
    fi

    # No fastapi on the host. Rather than pip-install into a system interpreter
    # Ubuntu 24.04 marks externally-managed (PEP 668) — which fails, and fails at
    # demo time — borrow the container, which already has it.
    echo "[dashboard] host has no fastapi; running sim inside $IMAGE instead"
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        docker build -f "$REPO/Dockerfile.wisepack" -t "$IMAGE" "$REPO" || exit 1
    fi
    open_browser
    exec docker run --rm -i $([ -t 1 ] && echo -t) \
        --name wisepack-dashboard-sim \
        --net=host \
        -e "WISEPACK_DASH_PORT=$PORT" -e "WISEPACK_PRESET=$PRESET" -e "WISEPACK_SEED=$SEED" \
        -v "$REPO:$REPO" -w "$REPO" \
        "$IMAGE" \
        bash -lc 'exec python3 web/app.py --source sim --port "${WISEPACK_DASH_PORT}" \
                    --preset "${WISEPACK_PRESET}" --seed "${WISEPACK_SEED}"'
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

wisepack_reap_stale "wisepack-dashboard"

# ---- live modes -------------------------------------------------------------
"$REPO/scripts/clean_dds_shm.sh" || true

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[dashboard] image $IMAGE not present — building it ..."
    docker build -f "$REPO/Dockerfile.wisepack" -t "$IMAGE" "$REPO" || exit 1
fi

SOURCE="ros"
if [ "$MODE" = "fiware" ]; then
    SOURCE="fiware"
    echo "[dashboard] starting the Orion-LD DDS broker ..."
    ( cd "$DDS_DIR" && python3 generate_config.py --domain "$ROS_DOMAIN_ID" >/dev/null \
        && docker compose -f docker-compose.dds.yml up -d ) || {
        echo "[dashboard] ERROR: could not start the Orion-LD DDS stack." >&2
        exit 1
    }
    for i in $(seq 1 40); do
        curl -s --max-time 2 "${ORION:-http://localhost:1026}/version" 2>/dev/null \
            | grep -qi orionld && break
        sleep 1
    done
fi

echo "[dashboard] live mode ($SOURCE) — WISEPACK ROS 2/DDS stack + dashboard"
echo "[dashboard] open $URL   (Ctrl-C to stop everything)"
open_browser

exec docker run --rm -i $([ -t 1 ] && echo -t) \
    --name wisepack-dashboard \
    --net=host \
    --ipc=host \
    --privileged \
    -e "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" \
    -e "WISEPACK_DASH_PORT=$PORT" \
    -e "WISEPACK_SOURCE=$SOURCE" \
    -e "WISEPACK_PRESET=$PRESET" \
    -e "WISEPACK_SEED=$SEED" \
    -e WISEPACK_SKIP_BUILD \
    -e ORION \
    -v "$REPO:$REPO" \
    -w "$REPO" \
    "$IMAGE" \
    bash -lc '
        # No `set -u` here: Vulcanexus setup.bash dereferences COLCON_TRACE while
        # it is still unset, so nounset aborts before the ROS env is even sourced.
        source /opt/vulcanexus/jazzy/setup.bash

        # Always build. An incremental colcon build is a few seconds when the
        # workspace is current, and demonstrating a STALE install is how a
        # "working" dashboard ends up showing code from a previous session.
        if [ "${WISEPACK_SKIP_BUILD:-0}" = "1" ]; then
            echo "[container] WISEPACK_SKIP_BUILD=1 — using the existing install/"
            [ -f wisepack_ws/install/setup.bash ] || {
                echo "[container] ERROR: nothing built and the build was skipped." >&2
                exit 1
            }
        else
            echo "[container] colcon build --symlink-install (incremental) ..."
            ( cd wisepack_ws && colcon build --symlink-install ) || exit 1
        fi
        source wisepack_ws/install/setup.bash

        # Kill only what THIS launcher started. `pkill -f wisepack_` would also
        # match another checkout of this project running on the same machine,
        # and a broader pattern would take out unrelated ROS stacks entirely.
        cleanup() {
            if [ -n "${LAUNCH_PID:-}" ]; then
                kill -TERM "-$LAUNCH_PID" 2>/dev/null \
                    || kill "$LAUNCH_PID" 2>/dev/null || true
                wait "$LAUNCH_PID" 2>/dev/null || true
            fi
        }
        trap cleanup EXIT INT TERM

        echo "[container] launching WISEPACK (orchestrator + perception + twin) ..."
        setsid ros2 launch wisepack_bringup demo.launch.py \
            preset:="${WISEPACK_PRESET}" seed:="${WISEPACK_SEED}" \
            > /tmp/wisepack_stack.log 2>&1 &
        LAUNCH_PID=$!

        # The dashboard must not attach before the nodes exist, or its first
        # snapshot is all defaults and the tiles look dead on arrival.
        for i in $(seq 1 40); do
            if ros2 topic list 2>/dev/null | grep -q /wisepack/execution/state; then break; fi
            sleep 1
        done
        echo "[container] WISEPACK stack up — attaching dashboard (source=${WISEPACK_SOURCE})"

        exec python3 web/app.py --source "${WISEPACK_SOURCE}" \
            --port "${WISEPACK_DASH_PORT}"
    '
