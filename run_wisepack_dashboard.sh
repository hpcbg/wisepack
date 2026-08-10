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
# STARTUP IS CONCURRENT, NOT SEQUENTIAL. Isaac is started first and then WATCHED;
# the ROS stack and the dashboard come up immediately alongside it. Waiting for
# Isaac to report READY before starting the container — which is what this used
# to do — left port 8080 closed for as long as the simulator took to compile
# shaders, and an operator looking at a dead port cannot tell a slow simulator
# from a broken launcher. Isaac's progress is now visible IN the dashboard.
#
# Starting the UI early authorises nothing. Approval still requires an active
# run and a SCENE_READY correlated to it, to its scenario revision, to its
# fingerprint and to its robot — see wisepack_orchestration.isaac_bridge.
#
# WHICH ROBOT is resolved to a concrete id BEFORE anything starts, and the same
# value is given to the host simulator, the container, the orchestrator and the
# dashboard. See scripts/resolve_robot.py.
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

# HOST-SPECIFIC SETTINGS, from config/local.env (git-ignored, allowlist-parsed,
# never sourced — see scripts/lib_local_env.sh). Loaded HERE and not only in the
# Isaac launcher because the perception settings — which camera is plugged in,
# where the detector weights live, how big the printed calibration board and the
# physical objects are — are needed by THIS process and by the container it
# starts. A documented variable that silently does nothing is worse than an
# undocumented one. An explicit export always wins over the file.
# shellcheck source=scripts/lib_local_env.sh
. "$REPO/scripts/lib_local_env.sh"
wisepack_load_local_env "$REPO"

# ONE cleanup mechanism for every host child this launcher owns — Isaac Sim and
# the perception service — because two `trap ... EXIT` installations would
# silently replace one another and leak whichever was registered first.
# shellcheck source=scripts/lib_host_processes.sh
. "$REPO/scripts/lib_host_processes.sh"
# shellcheck source=scripts/lib_perception_service.sh
. "$REPO/scripts/lib_perception_service.sh"

# --- the CAD/reference tree, mounted at the SAME PATH it has on the host ------
#
# THE MESHES ARE NOT IN THE REPOSITORY, deliberately: they are third-party
# reference material that already lives BESIDE it, and `mesh_path` entries in
# config/perception_objects.yaml are relative to that sibling tree. The
# containerised dashboard mounted the repo and nothing else, so every
# `mesh_exists()` was False inside it and the FoundationPose provider correctly
# reported that no object model had a mesh — which reads as an empty registry
# rather than as a missing mount. Six models were eligible the whole time.
#
# MOUNTED AT THE IDENTICAL PATH, exactly as `$REPO` is, so a mesh path resolves
# to the same string on the host, in this container and in the FoundationPose
# worker (which mounts the same tree at /datasets). No translation table, and
# nothing is copied into the repository.
#
# READ-ONLY. WISEPACK reads this material and never writes it.
ASSETS_ROOT="${WISEPACK_PERCEPTION_ASSETS_ROOT:-$(dirname "$REPO")/references}"
ASSETS_MOUNT=()
[ -d "$ASSETS_ROOT" ] && ASSETS_MOUNT=(-v "$ASSETS_ROOT:$ASSETS_ROOT:ro")

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
        # The whole header block, delimited rather than counted: a hard-coded
        # line range silently drops the end of --help the moment the header
        # grows, which is exactly what happened when startup was documented.
        awk 'NR>2 && /^# -{20,}/{exit} NR>1{sub(/^# ?/, ""); print}' "$0"
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

# --- WHICH ROBOT, resolved to a concrete id before anything is started ------
#
# This used to print "robot : <configured default>" and hand an EMPTY string to
# both the simulator and the orchestrator, letting each resolve independently.
# That is how a placeholder reached `ros2 launch` as the malformed argument
# `robot:=`, which killed the whole ROS stack while its container stayed Up.
#
# So the answer is computed ONCE, here, on the host, before a process exists,
# and the same value goes to every consumer. An unresolvable robot, an
# incompatible robot/preset pair or a literal placeholder stops the launch here
# rather than being discovered by an operator staring at an IDLE dashboard.
ROBOT_ID=""
ROBOT_SOURCE=""
ROBOT_REVISION=""
ROBOT_REGISTRY=""
ROBOT_REGISTRY_DEFAULT=""
# WHERE THE STARTUP STATUS GOES. The results directory is the natural home and
# is usually right, but it is frequently owned by root — the container writes
# into it — and the host user then cannot create a file there. That failure used
# to be silent, which cost the diagnostics table its entire host half without
# saying so. So the location is CHOSEN, announced, and handed to the reader.
STATUS_DIR="${WISEPACK_RESULTS_DIR:-$REPO/results}"
mkdir -p "$STATUS_DIR" 2>/dev/null || true
HOST_STATUS_OWNED_DIR=""
if [ -w "$STATUS_DIR" ]; then
    HOST_STATUS="$STATUS_DIR/startup-host.json"
else
    # A DEDICATED directory, not a bare file in /tmp. The status writer replaces
    # the file atomically (write-temp-then-rename), which changes its inode —
    # and a FILE bind-mount follows the inode, so the container would keep
    # reading the original for the life of the run and never see an update.
    # Measured exactly that way: the launcher reported Isaac dead and
    # Diagnostics still showed it running. Mounting the directory fixes it, and
    # a directory this launcher owns is safe to mount where /tmp is not.
    HOST_STATUS_OWNED_DIR="${TMPDIR:-/tmp}/wisepack-startup-$$"
    mkdir -p "$HOST_STATUS_OWNED_DIR"
    HOST_STATUS="$HOST_STATUS_OWNED_DIR/startup-host.json"
    echo "[dashboard] note: $STATUS_DIR is not writable by $(id -un);"
    echo "[dashboard]       startup status goes to $HOST_STATUS instead"
fi
# The container runs as root and can always write the results directory, so the
# stack half stays where an operator would look for it.
STACK_STATUS="$STATUS_DIR/startup-stack.json"
export WISEPACK_HOST_STATUS="$HOST_STATUS"

if [ "$EXECUTION_BACKEND" = "isaac" ]; then
    if ! ROBOT_LINE="$(python3 "$REPO/scripts/resolve_robot.py" --preset "$PRESET")"; then
        echo "[isaac-launch] ERROR: the robot could not be resolved; nothing was started." >&2
        echo "[isaac-launch]   fix config/isaac_robots.yaml, or set WISEPACK_ISAAC_ROBOT" >&2
        echo "[isaac-launch]   to one of the configured ids." >&2
        exit 5
    fi
    IFS=$'\t' read -r ROBOT_ID ROBOT_SOURCE ROBOT_REVISION ROBOT_REGISTRY \
        ROBOT_REGISTRY_DEFAULT <<<"$ROBOT_LINE"
    # Belt and braces: resolve_robot.py already refuses a placeholder, and this
    # refuses one that somehow survived. A `<...>` reaching a launch argument is
    # precisely the regression this replaces.
    case "$ROBOT_ID" in
        ""|*"<"*|*">"*|*" "*)
            echo "[isaac-launch] ERROR: unresolved robot id '${ROBOT_ID}'." >&2
            exit 5 ;;
    esac
    # ONE value, exported once, reaching the host simulator, the container, the
    # orchestrator, the scene request and the dashboard API.
    export WISEPACK_ISAAC_ROBOT="$ROBOT_ID"
fi

python3 "$REPO/scripts/startup_status.py" init --out "$HOST_STATUS" \
    --scope host --mode "$MODE" \
    --robot "$ROBOT_ID" --robot-source "$ROBOT_SOURCE" \
    --robot-revision "$ROBOT_REVISION" --registry-path "$ROBOT_REGISTRY" \
    --registry-default "$ROBOT_REGISTRY_DEFAULT" 2>/dev/null || true
# A stale stack status from a previous run must not be read as this run's.
rm -f "$STACK_STATUS" 2>/dev/null || true

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

# ---- optional physical perception ------------------------------------------
#
# A THIRD AXIS, independent of the mode above and of the execution backend. With
# `WISEPACK_PERCEPTION_SOURCE=sim` — the default, and the value when the variable
# is absent — NOTHING below happens: no interpreter is resolved, no process is
# started, and the perception environment is not even looked at.
#
# Done HERE, before every `exec` path, so all five modes get it: `sim` execs the
# dashboard a few lines down, and the live modes exec Docker at the end of the
# script. Starting the detector after either of those would orphan it.
#
# THE SERVICE IS THE SAME IN EVERY MODE. It speaks HTTP only and needs no
# middleware on the host: `sim` reaches it from the dashboard, and the live modes
# reach it from the orchestrator inside the CONTAINERIZED Vulcanexus stack, which
# is where the WISEPACK-domain perception topics are published. Nothing here
# installs or sources a host ROS 2.
# TWO WAYS TO ASK FOR THE CAMERA, and they mean different things.
#
#   WISEPACK_PERCEPTION_SOURCE=camera   start this session ON the camera. The
#                                       first run detects, and a camera that
#                                       cannot be made available ABORTS the
#                                       launch (exit 6) rather than quietly
#                                       becoming a generated scenario.
#   WISEPACK_PERCEPTION_ENABLE=1        have the camera READY. The session
#                                       starts on the preset workflow and the
#                                       operator switches to the camera from the
#                                       dashboard whenever they like. A camera
#                                       that fails to start is a WARNING here,
#                                       not a failure: nothing has been claimed
#                                       yet, and preset runs are unaffected.
#
# Neither removes the other source, and neither has to be set again to switch:
# the object source is chosen per run, at runtime, in the dashboard. The service
# can also be started by hand at any time — the dashboard notices it without a
# restart, because availability is a live question, not a launch decision.
PERCEPTION_WANTED="${WISEPACK_PERCEPTION_SOURCE:-sim}"
if [ "$PERCEPTION_WANTED" = "camera" ]; then
    if ! perception_ensure_service "$REPO"; then
        echo "[perception] ABORTING: camera perception was requested as the" >&2
        echo "[perception]           INITIAL object source and no detector is" >&2
        echo "[perception]           available. WISEPACK will not start with a" >&2
        echo "[perception]           generated scenario instead — that would" >&2
        echo "[perception]           claim a camera it does not have." >&2
        echo "[perception]           Start on presets with" >&2
        echo "[perception]           WISEPACK_PERCEPTION_SOURCE=sim (the" >&2
        echo "[perception]           default), or make the camera optional with" >&2
        echo "[perception]           WISEPACK_PERCEPTION_ENABLE=1." >&2
        exit 6
    fi
elif [ "${WISEPACK_PERCEPTION_ENABLE:-0}" = "1" ]; then
    echo "[perception] camera capability requested — the session starts on the"
    echo "[perception] preset workflow; switch the dashboard's Object source to"
    echo "[perception] Physical camera whenever you like."
    if ! perception_ensure_service "$REPO"; then
        # NOT FATAL. Nothing has claimed a camera yet: preset scenarios work
        # exactly as they always have, and the dashboard reports the camera
        # source as unavailable with the reason instead of breaking WISEPACK.
        echo "[perception] WARNING: the camera could not be made available." >&2
        echo "[perception]          Preset scenarios are unaffected; the" >&2
        echo "[perception]          dashboard will show Physical camera as" >&2
        echo "[perception]          unavailable until a service is running." >&2
    fi
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
        # `host_run_foreground`, not `exec`: with a launcher-owned detector this
        # shell has to survive the dashboard so its EXIT trap can stop it. With
        # no host child it still execs, exactly as before.
        host_run_foreground python3 "$REPO/web/app.py" --source sim --port "$PORT" \
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
    # Built as an ARRAY so the container command can be handed to
    # host_run_foreground, which decides between `exec` and foreground+cleanup.
    # `--net=host` is what lets the containerised dashboard reach the detector
    # on the HOST at 127.0.0.1 — the service is never moved into the container.
    SIM_DOCKER=(docker run --rm -i)
    [ -t 1 ] && SIM_DOCKER+=(-t)
    SIM_DOCKER+=(
        --name wisepack-dashboard-sim
        --net=host
        -e "WISEPACK_DASH_PORT=$PORT" -e "WISEPACK_PRESET=$PRESET" -e "WISEPACK_SEED=$SEED"
        -e WISEPACK_PERCEPTION_SOURCE -e WISEPACK_PERCEPTION_ENABLE -e WISEPACK_PERCEPTION_SERVICE_URL
        -e WISEPACK_PHYSICAL_PROXY_DIAMETER_MM -e WISEPACK_PHYSICAL_PROXY_LENGTH_MM
        -e WISEPACK_PHYSICAL_PROXY_WALL_MM -e WISEPACK_PHYSICAL_PROXY_MATERIAL
        -e WISEPACK_PHYSICAL_PROXY_GROUP -e WISEPACK_PHYSICAL_FRAME_ID
        -e WISEPACK_PHYSICAL_WORKAREA_WIDTH_MM -e WISEPACK_PHYSICAL_WORKAREA_DEPTH_MM
        -e WISEPACK_PERCEPTION_ASSETS_ROOT
        -v "$REPO:$REPO" "${ASSETS_MOUNT[@]}" -w "$REPO"
        "$IMAGE"
        bash -lc 'exec python3 web/app.py --source sim --port "${WISEPACK_DASH_PORT}" \
                    --preset "${WISEPACK_PRESET}" --seed "${WISEPACK_SEED}"')
    host_run_foreground "${SIM_DOCKER[@]}"
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
    # OWNERSHIP IS BY SESSION, and only the session this invocation created.
    #
    # The supervisor is started with `setsid`, so its session contains it and
    # every Isaac process group it has ever started. Signalling that session
    # reaches the whole tree — the python.sh wrapper, the Kit process it execs,
    # and the WebRTC service Kit owns as part of that process. A `pkill -f
    # isaac` pattern would instead match another project's simulator on a shared
    # machine, which is exactly the kind of collateral a demonstration host does
    # not need.
    #
    # NOTE FOR ANYONE READING htop: Isaac shows dozens of rows because Kit is
    # heavily threaded and htop lists THREADS by default (press H to hide them).
    # Those are TIDs inside one process, not multiple Isaac instances.

    # Only what this launcher created, and only when it created it.
    if [ -n "${HOST_STATUS_OWNED_DIR:-}" ]; then
        rm -rf -- "$HOST_STATUS_OWNED_DIR"
        HOST_STATUS_OWNED_DIR=""
    fi
    # The watcher first, so it cannot report the shutdown it is being shut down
    # by as a failure.
    if [ -n "${ISAAC_WATCHER_PID:-}" ]; then
        kill -TERM "$ISAAC_WATCHER_PID" 2>/dev/null || true
        wait "$ISAAC_WATCHER_PID" 2>/dev/null || true
        ISAAC_WATCHER_PID=""
    fi

    # THE SUPERVISOR OWNS ISAAC, so it is asked to stop and given time to take
    # its process group with it. Killing Isaac directly here would race the
    # supervisor's own shutdown and could leave it starting a replacement.
    if [ -n "${ISAAC_SUPERVISOR_PID:-}" ]; then
        echo "[isaac-launch] stopping the Isaac supervisor ($ISAAC_SUPERVISOR_PID)"
        kill -TERM "$ISAAC_SUPERVISOR_PID" 2>/dev/null || true
        for _ in $(seq 1 90); do
            kill -0 "$ISAAC_SUPERVISOR_PID" 2>/dev/null || break
            sleep 0.5
        done
        if kill -0 "$ISAAC_SUPERVISOR_PID" 2>/dev/null; then
            echo "[isaac-launch] supervisor did not exit on TERM — sending KILL" >&2
            kill -KILL "$ISAAC_SUPERVISOR_PID" 2>/dev/null || true
        fi
        ISAAC_SUPERVISOR_PID=""
    fi

    # VERIFY, do not assume. A launcher that says "stopped" while the GPU is
    # still held is worse than one that says nothing.
    if [ -n "${ISAAC_SUPERVISOR_PGID:-}" ]; then
        remaining="$(ps -eo pgid= 2>/dev/null | tr -d " " \
            | grep -c "^${ISAAC_SUPERVISOR_PGID}$" || true)"
        if [ "${remaining:-0}" -gt 0 ]; then
            echo "[isaac-launch] $remaining process(es) remain in session" \
                 "$ISAAC_SUPERVISOR_PGID — sending KILL" >&2
            kill -KILL "-$ISAAC_SUPERVISOR_PGID" 2>/dev/null || true
            sleep 1
        fi
        remaining="$(ps -eo pgid= 2>/dev/null | tr -d " " \
            | grep -c "^${ISAAC_SUPERVISOR_PGID}$" || true)"
        if [ "${remaining:-0}" -gt 0 ]; then
            echo "[isaac-launch] WARNING: $remaining process(es) still in group" \
                 "$ISAAC_SUPERVISOR_PGID" >&2
            ps -eo pgid=,pid=,cmd= 2>/dev/null \
                | awk -v g="$ISAAC_SUPERVISOR_PGID" '$1 == g' | head -5 >&2
        else
            echo "[isaac-launch] supervisor session $ISAAC_SUPERVISOR_PGID is gone"
        fi
        ISAAC_SUPERVISOR_PGID=""
    fi

    # The control directory is this launcher's, and it holds only control
    # documents and per-generation logs.
    if [ -n "${ISAAC_CONTROL_DIR:-}" ]; then
        rm -rf -- "$ISAAC_CONTROL_DIR"
        ISAAC_CONTROL_DIR=""
    fi
    return 0
}

if [ "$EXECUTION_BACKEND" = "isaac" ]; then
    # REGISTERED, not trapped directly. A bare `trap 'isaac_cleanup' EXIT` here
    # would replace the trap the perception service installs when the launcher
    # owns one, and camera+Isaac mode would leak the detector on Ctrl-C. The
    # registry runs both hooks under one trap; each stops only what it owns.
    host_register_cleanup isaac_cleanup

    # THE LAUNCHER NO LONGER STARTS ISAAC DIRECTLY.
    #
    # It starts a SUPERVISOR that owns the Isaac process group, because the
    # robot is chosen when the Isaac process starts — the adapter is built from
    # the profile and the USD model is referenced into the stage — and neither
    # can be changed afterwards. Switching robots is therefore a PROCESS
    # operation, and something has to be able to stop one simulator and start
    # another. The orchestrator runs in a container and must not be able to.
    #
    # The supervisor accepts exactly one verb, `switch_robot`, through a
    # request file in the directory below. No Docker socket, no host shell, no
    # command string, no signal, no pid. See scripts/isaac_supervisor.py.
    ISAAC_CONTROL_DIR="${TMPDIR:-/tmp}/wisepack-control-$$"
    mkdir -p "$ISAAC_CONTROL_DIR/logs" || {
        echo "[isaac-launch] ERROR: cannot create $ISAAC_CONTROL_DIR" >&2
        exit 7
    }
    # The container WRITES requests here, so this one directory is read-write.
    # It contains nothing but control documents.
    export WISEPACK_CONTROL_DIR="$ISAAC_CONTROL_DIR"
    ISAAC_LOG="$ISAAC_CONTROL_DIR/logs"

    echo "[isaac-launch] starting the Isaac supervisor (logs: $ISAAC_LOG)"
    echo "[isaac-launch] robot        : $ROBOT_ID"
    echo "[isaac-launch] robot source : $ROBOT_SOURCE"
    echo "[isaac-launch] robot profile: $ROBOT_REVISION"
    echo "[isaac-launch] robot registry: $ROBOT_REGISTRY (default $ROBOT_REGISTRY_DEFAULT)"
    echo "[isaac-launch] control dir  : $ISAAC_CONTROL_DIR (switch_robot only)"

    setsid env ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
        WISEPACK_PRESET="$PRESET" \
        WISEPACK_SEED="$SEED" \
        ${WISEPACK_RESULTS_DIR:+WISEPACK_RESULTS_DIR="$WISEPACK_RESULTS_DIR"} \
        python3 "$REPO/scripts/isaac_supervisor.py" \
            --control-dir "$ISAAC_CONTROL_DIR" \
            --robot "$ROBOT_ID" \
            --log-dir "$ISAAC_CONTROL_DIR/logs" \
        > "$ISAAC_CONTROL_DIR/supervisor.log" 2>&1 &
    ISAAC_SUPERVISOR_PID=$!

    # Its process group, so stopping it stops the Isaac group it owns.
    for _ in $(seq 1 40); do
        ISAAC_SUPERVISOR_PGID="$(ps -o pgid= -p "$ISAAC_SUPERVISOR_PID" 2>/dev/null | tr -d ' ')"
        [ -n "$ISAAC_SUPERVISOR_PGID" ] && break
        sleep 0.25
    done
    if [ -z "${ISAAC_SUPERVISOR_PGID:-}" ]; then
        echo "[isaac-launch] ERROR: the Isaac supervisor did not start." >&2
        tail -20 "$ISAAC_CONTROL_DIR/supervisor.log" 2>/dev/null | sed 's/^/    /' >&2
        exit 5
    fi
    echo "[isaac-launch] supervisor pid $ISAAC_SUPERVISOR_PID (group $ISAAC_SUPERVISOR_PGID)"
    python3 "$REPO/scripts/startup_status.py" proc --out "$HOST_STATUS" \
        --name isaac-supervisor --pid "$ISAAC_SUPERVISOR_PID" \
        --expected 1 --running 1 2>/dev/null || true

    # THE DASHBOARD DOES NOT WAIT FOR THIS. Readiness is watched and reported as
    # state; every authorisation gate upstream is unchanged.
    echo "[isaac-launch] not blocking on Isaac: the dashboard starts now and shows"
    echo "[isaac-launch] its progress (first launch compiles shaders and is slow)"

    (
        # Bounded, quiet, and it never restarts anything. It mirrors the
        # supervisor's own status into the startup table an operator reads.
        announced_ready=0
        deadline=$(( $(date +%s) + ISAAC_READY_TIMEOUT ))
        STATUS_JSON="$ISAAC_CONTROL_DIR/supervisor-status.json"
        while :; do
            if ! kill -0 "$ISAAC_SUPERVISOR_PID" 2>/dev/null; then
                reason="the Isaac supervisor (pid $ISAAC_SUPERVISOR_PID) exited"
                echo "[isaac-launch] ERROR: $reason" >&2
                tail -25 "$ISAAC_CONTROL_DIR/supervisor.log" 2>/dev/null \
                    | sed 's/^/    /' >&2
                python3 "$REPO/scripts/startup_status.py" proc \
                    --out "$HOST_STATUS" --name isaac-supervisor --running 0 \
                    --error "$reason" 2>/dev/null || true
                python3 "$REPO/scripts/startup_status.py" degrade \
                    --out "$HOST_STATUS" --reason "$reason" 2>/dev/null || true
                break
            fi
            phase="$(python3 -c "
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception:
    print('')
else:
    print(f\"{d.get('phase','')}|{d.get('robot_id','')}|{d.get('simulator_generation',0)}|{d.get('simulator_ready')}|{d.get('last_error','')}\")
" "$STATUS_JSON" 2>/dev/null)"
            IFS='|' read -r p_phase p_robot p_gen p_ready p_err <<<"${phase:-|||}"
            if [ "$announced_ready" -eq 0 ] && [ "$p_ready" = "True" ]; then
                announced_ready=1
                echo "[isaac-launch] Isaac READY — $p_robot, generation $p_gen"
                python3 "$REPO/scripts/startup_status.py" proc \
                    --out "$HOST_STATUS" --name isaac-sim --expected 1 --running 1 \
                    2>/dev/null || true
            fi
            if [ "$p_phase" = "ROBOT_SWITCH_FAILED" ]; then
                python3 "$REPO/scripts/startup_status.py" proc \
                    --out "$HOST_STATUS" --name isaac-sim --running 0 \
                    --error "${p_err:-the simulator failed}" 2>/dev/null || true
                python3 "$REPO/scripts/startup_status.py" degrade \
                    --out "$HOST_STATUS" \
                    --reason "${p_err:-the Isaac simulator failed}" \
                    2>/dev/null || true
                announced_ready=2
            elif [ "$announced_ready" -eq 1 ] && [ "$p_ready" != "True" ]; then
                # A switch is under way: no longer ready, not a failure.
                announced_ready=0
            fi
            if [ "$announced_ready" -eq 0 ] && [ "$(date +%s)" -gt "$deadline" ]; then
                reason="Isaac did not report READY within ${ISAAC_READY_TIMEOUT}s"
                echo "[isaac-launch] WARNING: $reason" >&2
                python3 "$REPO/scripts/startup_status.py" degrade \
                    --out "$HOST_STATUS" --reason "$reason" 2>/dev/null || true
                announced_ready=2
            fi
            python3 "$REPO/scripts/startup_status.py" beat \
                --out "$HOST_STATUS" --name isaac-watcher 2>/dev/null || true
            sleep 2
        done
    ) &
    ISAAC_WATCHER_PID=$!
    python3 "$REPO/scripts/startup_status.py" proc --out "$HOST_STATUS" \
        --name isaac-watcher --pid "$ISAAC_WATCHER_PID" --expected 1 --running 1 \
        2>/dev/null || true
fi

# MOUNT THE HOST STATUS ONLY IF IT IS BOTH OUTSIDE THE REPO AND ALREADY A FILE.
# Docker creates a DIRECTORY for a bind source that does not exist, and a
# directory where the reader expects JSON is a worse failure than a missing
# file. Inside the repo it is already visible through the main bind mount.
HOST_STATUS_MOUNT=()
if [ -n "$HOST_STATUS_OWNED_DIR" ] && [ -d "$HOST_STATUS_OWNED_DIR" ]; then
    # The DIRECTORY, so an atomic replace inside it is visible to the reader.
    HOST_STATUS_MOUNT=(-v "$HOST_STATUS_OWNED_DIR:$HOST_STATUS_OWNED_DIR:ro")
fi
# THE ONE READ-WRITE PATH THE CONTAINER IS GIVEN, and it holds nothing but
# control documents. Not the Docker socket, not a host shell, not /proc, not any
# other host state. The supervisor accepts one verb from it and validates the
# argument against the tracked registry before it stops anything.
CONTROL_MOUNT=()
if [ -n "${ISAAC_CONTROL_DIR:-}" ] && [ -d "${ISAAC_CONTROL_DIR:-}" ]; then
    CONTROL_MOUNT=(-v "$ISAAC_CONTROL_DIR:$ISAAC_CONTROL_DIR:rw")
fi

echo "[dashboard] live mode ($SOURCE) — WISEPACK ROS 2/DDS stack + dashboard"
echo "[dashboard] execution backend: $EXECUTION_BACKEND"
echo "[dashboard] initial object source: ${WISEPACK_PERCEPTION_SOURCE:-sim} (switchable in the dashboard)"
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
    -e "WISEPACK_ISAAC_ROBOT=$ROBOT_ID" \
    -e "WISEPACK_ROBOT_SOURCE=$ROBOT_SOURCE" \
    -e "WISEPACK_STARTUP_STATUS=$STACK_STATUS" \
    -e "WISEPACK_HOST_STATUS=$HOST_STATUS" \
    -e "WISEPACK_CONTROL_DIR=${ISAAC_CONTROL_DIR:-}" \
    -e "WISEPACK_MODE=$MODE" \
    -e WISEPACK_SKIP_BUILD \
    -e ORION \
    -e WISEPACK_PERCEPTION_SOURCE -e WISEPACK_PERCEPTION_ENABLE -e WISEPACK_PERCEPTION_SERVICE_URL \
    -e WISEPACK_PHYSICAL_PROXY_DIAMETER_MM -e WISEPACK_PHYSICAL_PROXY_LENGTH_MM \
    -e WISEPACK_PHYSICAL_PROXY_WALL_MM -e WISEPACK_PHYSICAL_PROXY_MATERIAL \
    -e WISEPACK_PHYSICAL_PROXY_GROUP -e WISEPACK_PHYSICAL_FRAME_ID \
    -e WISEPACK_PHYSICAL_WORKAREA_WIDTH_MM -e WISEPACK_PHYSICAL_WORKAREA_DEPTH_MM \
    -e WISEPACK_PERCEPTION_ASSETS_ROOT \
    -v "$REPO:$REPO" \
    "${ASSETS_MOUNT[@]}" \
    "${HOST_STATUS_MOUNT[@]}" \
    "${CONTROL_MOUNT[@]}" \
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
            if [ -n "${HEARTBEAT_PID:-}" ]; then
                kill -TERM "$HEARTBEAT_PID" 2>/dev/null || true
            fi
            if [ -n "${LAUNCH_PID:-}" ]; then
                kill -TERM "-$LAUNCH_PID" 2>/dev/null \
                    || kill "$LAUNCH_PID" 2>/dev/null || true
                wait "$LAUNCH_PID" 2>/dev/null || true
            fi
        }
        trap cleanup EXIT INT TERM

        STATUS="${WISEPACK_STARTUP_STATUS:-}"
        status() {
            [ -n "$STATUS" ] || return 0
            python3 scripts/startup_status.py "$@" --out "$STATUS" 2>/dev/null || true
        }
        status init --scope stack --mode "${WISEPACK_MODE:-ros}"             --robot "${WISEPACK_ISAAC_ROBOT:-}"             --robot-source "${WISEPACK_ROBOT_SOURCE:-}"

        echo "[container] launching WISEPACK (orchestrator + perception + twin) ..."
        echo "[container] execution backend: ${WISEPACK_EXECUTION_BACKEND}"
        echo "[container] robot            : ${WISEPACK_ISAAC_ROBOT:-none (logical simulator)}"

        # BUILT AS AN ARRAY so an empty value is OMITTED rather than emitted.
        #
        # `robot:="${WISEPACK_ISAAC_ROBOT:-}"` expanded to the literal `robot:=`
        # whenever the variable was unset — and it was ALWAYS unset, because the
        # launcher never passed it through `docker run`. ros2 launch rejects
        # that as "malformed launch argument" and exits on its first line, in
        # every container-backed mode. The wrapper then waited out a fixed
        # timeout for a topic that was never coming, announced "WISEPACK stack
        # up" and started the dashboard against nothing. That is the whole IDLE
        # regression: an empty string in a shell expansion.
        LAUNCH_ARGS=(preset:="${WISEPACK_PRESET}" seed:="${WISEPACK_SEED}"
                     execution_backend:="${WISEPACK_EXECUTION_BACKEND}"
                     perception_source:="${WISEPACK_PERCEPTION_SOURCE:-sim}")
        if [ -n "${WISEPACK_ISAAC_ROBOT:-}" ]; then
            LAUNCH_ARGS+=(robot:="${WISEPACK_ISAAC_ROBOT}")
        fi

        setsid ros2 launch wisepack_bringup demo.launch.py "${LAUNCH_ARGS[@]}"             > /tmp/wisepack_stack.log 2>&1 &
        LAUNCH_PID=$!
        status proc --name ros-launch --pid "$LAUNCH_PID" --expected 1 --running 1

        # WAIT FOR THE TOPICS, BUT WATCH THE PROCESS.
        #
        # The old loop only asked "is the topic there yet" and then declared
        # success either way. Liveness is now checked every iteration, so a
        # launch that dies is reported in about a second instead of being
        # papered over after forty.
        STACK_UP=0
        for i in $(seq 1 60); do
            if ros2 topic list 2>/dev/null | grep -q /wisepack/execution/state; then
                STACK_UP=1
                break
            fi
            if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
                wait "$LAUNCH_PID" 2>/dev/null
                RC=$?
                echo "[container] ERROR: ros2 launch exited with status $RC after ${i}s." >&2
                echo "[container] last 25 lines of /tmp/wisepack_stack.log:" >&2
                tail -25 /tmp/wisepack_stack.log | sed "s/^/    /" >&2
                status proc --name ros-launch --running 0 --exit-code "$RC"                     --error "$(tail -3 /tmp/wisepack_stack.log | tr "\n" " " | cut -c1-400)"
                status degrade --reason "ros2 launch exited with status $RC"
                break
            fi
            sleep 1
        done

        if [ "$STACK_UP" -eq 1 ]; then
            echo "[container] WISEPACK stack up — attaching dashboard (source=${WISEPACK_SOURCE})"
            status proc --name ros-launch --running 1
            # PER-NODE liveness from the ROS graph, so "the launch process is
            # alive" and "the orchestrator actually came up" stay separable. A
            # launch that starts three of its four nodes is a real state and
            # reads identically to a healthy one without this.
            NODES="$(ros2 node list 2>/dev/null || true)"
            for pair in "orchestrator:/wisepack_hitl_orchestrator"                         "perception-sim:/wisepack_perception_sim"                         "twin-validator:/wisepack_twin_validator"                         "anomaly-simulator:/wisepack_anomaly_simulator"; do
                nm="${pair%%:*}"; node="${pair#*:}"
                if printf "%s\n" "$NODES" | grep -qx "$node"; then
                    status proc --name "$nm" --expected 1 --running 1
                else
                    status proc --name "$nm" --expected 1 --running 0                         --error "not present in the ROS graph after startup"
                fi
            done
        else
            # THE DASHBOARD STILL STARTS, and it must: it is where the operator
            # reads WHY the stack is not there. What must not happen — and used
            # to — is claiming the stack is up. Nothing is restarted; a launch
            # that failed once will fail the same way again, and a restart loop
            # would replace one clear diagnosis with a scrolling one.
            echo "[container] WISEPACK stack is NOT running — starting the dashboard" >&2
            echo "[container] anyway so Diagnostics can show why. Execution is DEGRADED." >&2
            status degrade --reason "the ROS stack did not come up"
        fi

        # A background heartbeat, so a launch that dies LATER is also surfaced
        # rather than leaving a stale "running" in the status file forever.
        (
            while kill -0 "$LAUNCH_PID" 2>/dev/null; do
                status beat --name ros-launch
                sleep 5
            done
            wait "$LAUNCH_PID" 2>/dev/null
            RC=$?
            status proc --name ros-launch --running 0 --exit-code "$RC"                 --error "the ROS launch process exited during the run"
            status degrade --reason "ros2 launch exited with status $RC during the run"
        ) &
        HEARTBEAT_PID=$!

        status proc --name dashboard --pid "$$" --expected 1 --running 1
        exec python3 web/app.py --source "${WISEPACK_SOURCE}"             --port "${WISEPACK_DASH_PORT}"
    ')

python3 "$REPO/scripts/startup_status.py" proc --out "$HOST_STATUS" \
    --name wisepack-container --expected 1 --running 1 2>/dev/null || true

# Foreground when this shell owns a host child — Isaac Sim, the WISEPACK
# perception service, or both — so the EXIT trap runs when the container stops
# and Ctrl-C takes every owned process down with the stack. Plain `exec` when it
# owns nothing, which is what every mode did before either existed.
host_run_foreground "${DOCKER_RUN[@]}"
