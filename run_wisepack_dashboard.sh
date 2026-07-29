#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_wisepack_dashboard.sh — one command: the WISEPACK stack AND its dashboard.
#
#     ./run_wisepack_dashboard.sh              # live: ROS 2/DDS stack + dashboard
#     ./run_wisepack_dashboard.sh fiware       # live + Orion-LD, state from FIWARE
#     ./run_wisepack_dashboard.sh sim          # presentation only: no ROS, no FIWARE
#     ./run_wisepack_dashboard.sh isaac        # live + Isaac Sim PHYSICAL execution
#     ./run_wisepack_dashboard.sh isaac-fiware # isaac + Orion-LD, state from FIWARE
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
# `sim` is unchanged by the Isaac work and remains presentation-only.
#
# ISAAC MODES add a PHYSICAL EXECUTION BACKEND. Two different concepts, and the
# dashboard shows them separately:
#
#     data source        sim | ros | fiware      where the dashboard READS state
#     execution backend  simulated | isaac       who MOVES the item
#
# So `isaac` is the ordinary live ROS stack with the placements executed by a
# SELECTED manipulator in Isaac Sim instead of by the seeded robot model, and
# `isaac-fiware` is the same thing observed through Orion-LD. Isaac is NOT a
# replacement for the ROS data source.
#
# Isaac Sim runs on the HOST in its own bundled Python — never inside the
# WISEPACK image — and reaches the stack over DDS on the shared ROS_DOMAIN_ID.
# Ctrl-C stops the stack and the Isaac process THIS invocation started, by PID,
# and nothing else.
#
#     WISEPACK_ISAAC_HEADLESS=1 ./run_wisepack_dashboard.sh isaac   # over SSH
#     ISAAC_SIM_ROOT=/opt/isaac-sim ./run_wisepack_dashboard.sh isaac
#
# In the Isaac modes WISEPACK_PRESET defaults to the small physical smoke
# scenario (isaac_cylinders_smoke, 4 cylinders) rather than the 40-item packing
# benchmark, because forty robotic pick-and-place cycles is not a demonstration.
# An explicitly supplied WISEPACK_PRESET is always honoured.
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
SEED="${WISEPACK_SEED:-42}"
URL="http://127.0.0.1:${PORT}"
DDS_DIR="$REPO/wisepack_ws/src/wisepack_fiware/dds"

#: Seconds to wait for Isaac Sim to build its scene and report READY. Generous:
#: a first launch compiles shaders, which can take several minutes and is then
#: cached in ~/.cache/ov.
ISAAC_READY_TIMEOUT="${WISEPACK_ISAAC_READY_TIMEOUT:-300}"

case "$MODE" in
    -h|--help|help)
        sed -n '2,47p' "$0"
        exit 0 ;;
    ros|fiware|sim|isaac|isaac-fiware) ;;
    *)
        echo "[dashboard] unknown mode: $MODE" >&2
        echo "[dashboard] expected one of: ros | fiware | sim | isaac | isaac-fiware" >&2
        echo "[dashboard] try: ./run_wisepack_dashboard.sh --help" >&2
        exit 2 ;;
esac

# WHICH BACKEND EXECUTES. Kept separate from the data source throughout — see
# the header. `simulated` preserves the existing behaviour exactly.
EXECUTION_BACKEND="simulated"
case "$MODE" in
    isaac|isaac-fiware) EXECUTION_BACKEND="isaac" ;;
esac

# The physical smoke scenario is the default ONLY in the Isaac modes, and only
# when the operator did not name one. Everything else keeps the benchmark.
if [ "$EXECUTION_BACKEND" = "isaac" ]; then
    PRESET="${WISEPACK_PRESET:-isaac_cylinders_smoke}"
else
    PRESET="${WISEPACK_PRESET:-mixed_pipes_dense}"
fi

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
if [ "$MODE" = "fiware" ] || [ "$MODE" = "isaac-fiware" ]; then
    SOURCE="fiware"
    echo "[dashboard] starting the Orion-LD DDS broker ..."
    ( cd "$DDS_DIR" && python3 generate_config.py --domain "$ROS_DOMAIN_ID" >/dev/null \
        && docker compose -f docker-compose.dds.yml up -d ) || {
        echo "[dashboard] ERROR: could not start the Orion-LD DDS stack." >&2
        exit 1
    }
    # BOUNDED HEALTH GATE, and its RESULT IS CHECKED.
    #
    # This loop previously waited and then fell through regardless. When Orion
    # never came up the launcher still announced `source=fiware`, the dashboard
    # found no FIWARE data and honestly fell back to ROS per panel, and the
    # screen ended up asserting three contradictory things at once: the command
    # was `isaac-fiware`, the terminal said `source=fiware`, the header badge
    # said `ROS 2 / DDS`, and the FIWARE pill said `unreachable`. Waiting is not
    # the same as verifying.
    FIWARE_TIMEOUT="${WISEPACK_FIWARE_READY_TIMEOUT:-90}"
    FIWARE_OK=0
    echo "[dashboard] waiting up to ${FIWARE_TIMEOUT}s for Orion-LD on ${ORION:-http://localhost:1026} ..."
    for _ in $(seq 1 "$FIWARE_TIMEOUT"); do
        if curl -s --max-time 2 "${ORION:-http://localhost:1026}/version" 2>/dev/null \
                | grep -qi orionld; then
            FIWARE_OK=1
            break
        fi
        sleep 1
    done

    if [ "$FIWARE_OK" -eq 1 ]; then
        echo "[dashboard] Orion-LD is healthy — dashboard source: FIWARE"
    elif [ "${WISEPACK_FIWARE_DEGRADED:-0}" = "1" ]; then
        # EXPLICIT degraded mode: the source is downgraded to ROS and both the
        # operator and the dashboard are told so. Never "both".
        SOURCE="ros"
        echo "[dashboard] WARNING: Orion-LD did not become healthy within" >&2
        echo "[dashboard]   ${FIWARE_TIMEOUT}s. WISEPACK_FIWARE_DEGRADED=1 is set, so" >&2
        echo "[dashboard]   continuing with the ROS 2 / DDS source. The audit trail" >&2
        echo "[dashboard]   is NOT being read back from FIWARE in this run." >&2
    else
        echo "[dashboard] ERROR: Orion-LD did not become healthy within ${FIWARE_TIMEOUT}s." >&2
        echo "[dashboard]   '$MODE' explicitly asks for the FIWARE source, so this is a" >&2
        echo "[dashboard]   failure rather than something to paper over." >&2
        echo "[dashboard]   check: docker compose -f $DDS_DIR/docker-compose.dds.yml ps" >&2
        echo "[dashboard]   or re-run with WISEPACK_FIWARE_DEGRADED=1 to fall back to ROS." >&2
        exit 7
    fi
fi

# ── Isaac Sim: started on the HOST, owned by PID, waited for explicitly ─────
ISAAC_PID=""
ISAAC_LOG=""
ISAAC_PIDFILE=""

# How many PROCESSES are still in the group. Deliberately counts processes, not
# the threads htop shows by default — Kit is heavily threaded and every one of
# those rows is a TID inside a single process, not another simulator.
isaac_group_size() {
    ps -eo pgid=,pid= 2>/dev/null | awk -v g="$1" '$1 == g' | wc -l
}

isaac_cleanup() {
    # OWNERSHIP IS BY PROCESS GROUP, and only the group this invocation created.
    #
    # `setsid` below puts Isaac in its own process group whose PGID equals the
    # child's PID, so signalling `-$ISAAC_PID` reaches the whole tree — the
    # python.sh wrapper, the Kit process it execs, and the WebRTC service Kit
    # owns as part of that process. A `pkill -f isaac` pattern would instead
    # match another project's simulator on a shared machine, which is exactly
    # the rule the container reaping already follows by name rather than pattern.
    #
    # NOTE FOR ANYONE READING htop: Isaac shows dozens of rows because Kit is
    # heavily threaded and htop lists THREADS by default (press H to hide them).
    # Those are TIDs inside one process, not multiple Isaac instances. The check
    # below counts process-group members, which is the question that matters.
    [ -z "$ISAAC_PID" ] && return 0
    kill -0 "$ISAAC_PID" 2>/dev/null || return 0

    echo "[isaac-launch] stopping Isaac Sim process group $ISAAC_PID"
    kill -TERM "-$ISAAC_PID" 2>/dev/null || true

    # WAIT ON GROUP MEMBERSHIP, NOT ON THE LEADER. Kit spawns children that
    # outlive their parent during shutdown, so the leader can be gone while the
    # simulator is still holding the GPU. Measured exactly that way: keying the
    # wait on `kill -0 $LEADER` reported success with three processes still in
    # the group, and the KILL escalation was skipped because the leader had
    # already exited.
    local remaining
    for _ in $(seq 1 30); do
        remaining="$(isaac_group_size "$ISAAC_PID")"
        [ "$remaining" -eq 0 ] && break
        sleep 0.5
    done
    if [ "$(isaac_group_size "$ISAAC_PID")" -gt 0 ]; then
        echo "[isaac-launch] group $ISAAC_PID did not exit on TERM — sending KILL"
        kill -KILL "-$ISAAC_PID" 2>/dev/null || true
        for _ in $(seq 1 10); do
            [ "$(isaac_group_size "$ISAAC_PID")" -eq 0 ] && break
            sleep 0.5
        done
    fi

    # VERIFY, do not assume. A launcher that says "stopped" while the GPU is
    # still held is worse than one that says nothing.
    remaining="$(isaac_group_size "$ISAAC_PID")"
    if [ "${remaining:-0}" -gt 0 ]; then
        echo "[isaac-launch] WARNING: $remaining process(es) remain in group $ISAAC_PID" >&2
        ps -eo pgid=,pid=,cmd= 2>/dev/null | awk -v g="$ISAAC_PID" '$1 == g' | head -5 >&2
    else
        echo "[isaac-launch] process group $ISAAC_PID is gone"
    fi
    [ -n "${ISAAC_PIDFILE:-}" ] && rm -f "$ISAAC_PIDFILE"
    return 0
}

if [ "$EXECUTION_BACKEND" = "isaac" ]; then
    trap 'isaac_cleanup' EXIT INT TERM

    ISAAC_LOG="$(mktemp -t wisepack-isaac-XXXXXX.log)"
    ISAAC_PIDFILE="$(mktemp -t wisepack-isaac-pid-XXXXXX)"
    # BOTH ENDS GET THE SAME ROBOT, from the same variable. The orchestrator
    # resolves it too (see the `robot:=` launch argument below), and passing it
    # here rather than letting each side resolve independently is what stops the
    # simulator standing up one arm while the orchestrator plans for another.
    echo "[isaac-launch] starting Isaac Sim on the host (log: $ISAAC_LOG)"
    echo "[isaac-launch] robot       : ${WISEPACK_ISAAC_ROBOT:-<configured default>}"

    # THE CHILD REPORTS ITS OWN GROUP ID, and that is not pedantry.
    #
    # `setsid` FORKS when it is not already a process-group leader, so the `$!`
    # the shell records is setsid's short-lived parent, which exits immediately.
    # Measured here: `$!` was already dead one second later while the simulator
    # ran happily in a three-member group — so a cleanup keyed on `$!` found
    # nothing to kill and Ctrl-C left Isaac holding the GPU.
    #
    # The new session leader writes its own PID, which IS the process-group id,
    # so ownership is recorded from inside the group rather than guessed from
    # outside it.
    setsid bash -c '
        echo $$ > "$1"
        shift
        exec "$@"
    ' _ "$ISAAC_PIDFILE" \
        env ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
            WISEPACK_PRESET="$PRESET" \
            WISEPACK_SEED="$SEED" \
            ${WISEPACK_ISAAC_ROBOT:+WISEPACK_ISAAC_ROBOT="$WISEPACK_ISAAC_ROBOT"} \
            ${WISEPACK_RESULTS_DIR:+WISEPACK_RESULTS_DIR="$WISEPACK_RESULTS_DIR"} \
            "$REPO/scripts/run_wisepack_isaac.sh" > "$ISAAC_LOG" 2>&1 &

    # Wait briefly for the group leader to announce itself.
    ISAAC_PID=""
    for _ in $(seq 1 40); do
        if [ -s "$ISAAC_PIDFILE" ]; then
            ISAAC_PID="$(cat "$ISAAC_PIDFILE")"
            break
        fi
        sleep 0.25
    done
    if [ -z "$ISAAC_PID" ]; then
        echo "[isaac-launch] ERROR: Isaac Sim never reported its process group." >&2
        exit 5
    fi
    echo "[isaac-launch] Isaac Sim process group: $ISAAC_PID"

    echo "[isaac-launch] waiting up to ${ISAAC_READY_TIMEOUT}s for Isaac to report READY"
    echo "[isaac-launch] (first launch compiles shaders and is slow; it caches afterwards)"
    ISAAC_READY=0
    for _ in $(seq 1 "$ISAAC_READY_TIMEOUT"); do
        # Propagate a startup failure immediately rather than waiting out the
        # whole timeout for a process that has already died.
        if ! kill -0 "$ISAAC_PID" 2>/dev/null; then
            echo "[isaac-launch] ERROR: Isaac Sim exited during startup." >&2
            echo "[isaac-launch] last 40 lines of $ISAAC_LOG:" >&2
            tail -40 "$ISAAC_LOG" | sed 's/^/    /' >&2
            exit 5
        fi
        if grep -q '\[isaac-app\] READY' "$ISAAC_LOG" 2>/dev/null; then
            ISAAC_READY=1
            break
        fi
        sleep 1
    done

    if [ "$ISAAC_READY" -ne 1 ]; then
        echo "[isaac-launch] ERROR: Isaac Sim did not report READY within ${ISAAC_READY_TIMEOUT}s." >&2
        echo "[isaac-launch] check: GPU/driver, DISPLAY (use WISEPACK_ISAAC_HEADLESS=1)," >&2
        echo "[isaac-launch]        and outbound HTTPS for the Panda asset download." >&2
        echo "[isaac-launch] last 40 lines of $ISAAC_LOG:" >&2
        tail -40 "$ISAAC_LOG" | sed 's/^/    /' >&2
        exit 5
    fi
    echo "[isaac-launch] Isaac Sim READY (pid $ISAAC_PID) — physical execution enabled"
fi

echo "[dashboard] live mode ($SOURCE) — WISEPACK ROS 2/DDS stack + dashboard"
echo "[dashboard] execution backend: $EXECUTION_BACKEND"
echo "[dashboard] open $URL   (Ctrl-C to stop everything)"
open_browser

# NOT `exec` when Isaac is running: this shell must survive the container so its
# EXIT trap can stop the simulator it started.
DOCKER_RUN=(docker run --rm -i $([ -t 1 ] && echo -t) \
    --name wisepack-dashboard \
    --net=host \
    --ipc=host \
    --privileged \
    -e "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" \
    -e "WISEPACK_DASH_PORT=$PORT" \
    -e "WISEPACK_SOURCE=$SOURCE" \
    -e "WISEPACK_PRESET=$PRESET" \
    -e "WISEPACK_SEED=$SEED" \
    -e "WISEPACK_EXECUTION_BACKEND=$EXECUTION_BACKEND" \
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
        echo "[container] execution backend: ${WISEPACK_EXECUTION_BACKEND}"
        setsid ros2 launch wisepack_bringup demo.launch.py \
            preset:="${WISEPACK_PRESET}" seed:="${WISEPACK_SEED}" \
            execution_backend:="${WISEPACK_EXECUTION_BACKEND}" \
            robot:="${WISEPACK_ISAAC_ROBOT:-}" \
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
    ')

if [ "$EXECUTION_BACKEND" = "isaac" ]; then
    # Foreground, not exec: the EXIT trap installed above has to run when the
    # container stops, so that Ctrl-C takes the simulator down with the stack.
    "${DOCKER_RUN[@]}"
    STATUS=$?
    isaac_cleanup
    trap - EXIT INT TERM
    exit "$STATUS"
fi

exec "${DOCKER_RUN[@]}"
