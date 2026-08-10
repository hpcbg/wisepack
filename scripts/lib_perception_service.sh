#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# lib_perception_service.sh — the optional WISEPACK perception service, as a
# HOST process owned by the launcher.
#
#     . "$REPO/scripts/lib_host_processes.sh"
#     . "$REPO/scripts/lib_perception_service.sh"
#     perception_ensure_service "$REPO" || exit 1
#
# WHAT THIS OWNS, AND WHAT IT DELIBERATELY DOES NOT
# -------------------------------------------------
# The detector stays a HOST process and remains the single owner of the physical
# camera. It is NOT moved into the WISEPACK container: the container has no
# camera device, no CUDA context and no torch, and putting it there would mean
# two owners the moment anyone also ran it on the host.
#
# The WISEPACK dashboard — on the host or in the container — reaches it over
# WISEPACK_PERCEPTION_SERVICE_URL and nothing else.
#
# NO MIDDLEWARE IS INSTALLED OR SOURCED FOR IT
# --------------------------------------------
# The service speaks HTTP only. WISEPACK's validated middleware is the
# CONTAINERIZED Vulcanexus / Fast DDS runtime, and nothing here installs,
# sources or requires a host ROS 2 — see perception/perception_service.py.
#
# THE INTERPRETER IS WISEPACK'S OWN, NEVER THIS SHELL'S AND NEVER ANOTHER
# PROJECT'S
# -----------------------------------------------------------------------
# `perception_service.py` imports torch, torchvision, cv2, fastapi and uvicorn
# through its provider. None of those are in the system Python on a normal host,
# and running the service with `python3` produced exactly:
#
#     ModuleNotFoundError: No module named 'uvicorn'
#
# So WISEPACK owns an environment for them: `.venv-perception/` in the working
# directory, created by scripts/setup_perception.sh and git-ignored. An earlier
# revision reached into a HARMONY checkout's `torch_venv` instead, which made
# another repository a runtime dependency of a WISEPACK feature — deleting that
# clone broke the camera. It no longer can.
#
# The interpreter is invoked DIRECTLY. The venv is never `activate`d into the
# launcher shell: activation would put that interpreter and its site-packages
# ahead of everything this launcher afterwards runs.
#
# There is NO fall back to the system python. A camera mode that quietly starts
# an interpreter which cannot import its own dependencies fails later, in a less
# obvious place, with a worse message.
#
# FIRST RUN BOOTSTRAPS ITSELF. If the environment is missing, this library runs
# scripts/setup_perception.sh once and continues. `WISEPACK_PERCEPTION_AUTO_SETUP=0`
# turns that off and turns the miss into one printed command instead.
# ---------------------------------------------------------------------------

#: The PID/PGID of the service THIS launcher started. Empty when none was
#: started — either because an external one was already healthy, or because the
#: perception source is `sim`. Cleanup keys on this and nothing else: an empty
#: value means "own nothing, stop nothing".
PERCEPTION_OWNED_PID=""
PERCEPTION_OWNED_PGID=""

#: Where the resolved interpreter and the reason for a failure are published by
#: `perception_resolve_python`, for the caller to print.
PERCEPTION_PYTHON=""
PERCEPTION_PYTHON_ORIGIN=""
PERCEPTION_PYTHON_ERROR=""

PERCEPTION_LOG="${WISEPACK_PERCEPTION_LOG:-${TMPDIR:-/tmp}/wisepack_perception.log}"

#: Seconds to wait for /health. Generous: the first request imports torch and
#: loads a ~159 MB Faster R-CNN, which is tens of seconds on CPU and slower on a
#: cold page cache. Bounded, though — a hang must not become an indefinite wait.
PERCEPTION_READY_TIMEOUT="${WISEPACK_PERCEPTION_READY_TIMEOUT:-180}"

#: The documented default, matching web/perception_client.py.
PERCEPTION_DEFAULT_URL="http://127.0.0.1:22101"


perception_service_url() {
    printf '%s' "${WISEPACK_PERCEPTION_SERVICE_URL:-$PERCEPTION_DEFAULT_URL}"
}

#: Intel's USB vendor id — a RealSense, which is NOT the planar camera.
PERCEPTION_INTEL_VENDOR_ID="8086"

#: Is there a planar (UVC) camera on this host at all?
#:
#: WHY DISCOVERY RATHER THAN A FLAG. Whether a webcam is plugged in is a
#: property of the machine, and the machine can be asked. Requiring an operator
#: to export `WISEPACK_PERCEPTION_ENABLE=1` to find out made the runtime
#: selector depend on a launch decision: the dashboard reported "camera
#: unavailable" on a host with a working camera, because of a missing shell
#: variable rather than missing hardware.
#:
#: THE REALSENSE IS EXCLUDED, BY DEVICE AND NOT BY INDEX. A D4xx presents
#: several /dev/video* nodes and none of them is the planar camera; treating one
#: as a webcam would hand the RGB-D device to a planar detector and take it away
#: from the FoundationPose worker that owns it. Ownership is decided here by
#: walking up from each node to its USB device and skipping Intel's.
#:
#: NOTHING IS OPENED. This reads sysfs only: a probe that grabbed the device
#: would be a second owner of the very camera the service is about to take.
perception_planar_camera_node() {
    local node name vendor sysfs
    for node in /dev/video*; do
        [ -e "$node" ] || continue
        name="$(basename "$node")"
        sysfs="/sys/class/video4linux/$name"
        [ -d "$sysfs" ] || continue
        # Up from the video node to the USB device that owns it.
        vendor="$(cat "$(readlink -f "$sysfs/device")/../idVendor" 2>/dev/null)"
        [ "$vendor" = "$PERCEPTION_INTEL_VENDOR_ID" ] && continue
        # THE LOWEST-NUMBERED NON-INTEL NODE, which is what OpenCV index 0
        # opens and therefore what the service will use. A UVC webcam also
        # presents a metadata node beside its capture node; this does not try to
        # tell them apart, because it does not need to — whether the chosen
        # device yields frames is answered by the service actually opening it,
        # and reported as its own health rather than guessed here.
        printf '%s' "$node"
        return 0
    done
    return 1
}

perception_planar_camera_present() {
    [ -n "$(perception_planar_camera_node)" ]
}

# Parse the URL with Python's urllib rather than with `cut`/`sed`: an IPv6
# literal, an explicit scheme and a missing port are all normal here and a
# hand-rolled split gets at least one of them wrong.
perception_url_field() {
    python3 - "$1" "$2" <<'PY' 2>/dev/null
import sys
from urllib.parse import urlparse
parsed = urlparse(sys.argv[1])
field = sys.argv[2]
if field == "hostname":
    print(parsed.hostname or "")
elif field == "port":
    print(parsed.port or (443 if parsed.scheme == "https" else 80))
PY
}

#: True when a COMPATIBLE service answers /health. Compatible, not merely
#: listening: something else on port 22101 that returns 200 is not a detector,
#: and reusing it would report a healthy camera that does not exist.
perception_health_ok() {
    python3 - "$1" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1].rstrip("/") + "/health", timeout=3) as r:
    body = json.loads(r.read().decode("utf-8"))
# `source` and `detector` are this service's own health fields.
raise SystemExit(0 if isinstance(body, dict)
                 and body.get("source") and body.get("detector") else 1)
PY
}

#: Which provider the running service is using, for the start-up banner.
perception_health_provider() {
    python3 - "$1" <<'PY' 2>/dev/null
import json, sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1].rstrip("/") + "/health", timeout=3) as r:
        body = json.loads(r.read().decode("utf-8"))
except Exception:
    print("unknown"); raise SystemExit(0)
name = body.get("detector_display_name") or body.get("detector") or "unknown"
print(body.get("provider", "unknown") + " (" + name + ")")
PY
}

perception_is_local_host() {
    case "$1" in
        127.0.0.1|localhost|::1|0.0.0.0|"") return 0 ;;
        *) return 1 ;;
    esac
}

# --- the WISEPACK perception environment ------------------------------------

#: Where the WISEPACK-owned perception environment lives. Overridable only to
#: move it (a bigger disk, a shared read-only image); it is always a WISEPACK
#: environment created by scripts/setup_perception.sh, never another project's.
perception_venv_dir() {
    printf '%s' "${WISEPACK_PERCEPTION_VENV:-$1/.venv-perception}"
}

#: Usable == exists AND can import everything the service imports. A venv whose
#: torch installation is broken must fail HERE, with the import error, not
#: sixty seconds later inside the first detection.
perception_env_usable() {
    [ -x "$1" ] || return 1
    "$1" -c 'import torch, torchvision, cv2, cv2.aruco, numpy, PIL, fastapi, uvicorn' \
        >/dev/null 2>&1
}

perception_resolve_python() {
    local repo="${1:-$REPO}"
    PERCEPTION_PYTHON=""
    PERCEPTION_PYTHON_ORIGIN=""
    PERCEPTION_PYTHON_ERROR=""

    local venv candidate setup
    venv="$(perception_venv_dir "$repo")"
    candidate="$venv/bin/python"
    setup="$repo/scripts/setup_perception.sh"

    # 1. The WISEPACK perception environment, ready to go.
    if perception_env_usable "$candidate"; then
        PERCEPTION_PYTHON="$candidate"
        PERCEPTION_PYTHON_ORIGIN="WISEPACK perception environment"
        return 0
    fi

    # 2. Missing or incomplete: build it, once, and try again. This is what
    #    keeps `./run_wisepack_dashboard.sh sim` the only command an operator
    #    ever types. Nothing is installed into the system Python.
    if [ "${WISEPACK_PERCEPTION_AUTO_SETUP:-1}" != "0" ] && [ -x "$setup" ]; then
        echo "[perception] perception environment : missing — creating it once"
        echo "[perception] setup                  : $setup"
        if "$setup"; then
            if perception_env_usable "$candidate"; then
                PERCEPTION_PYTHON="$candidate"
                PERCEPTION_PYTHON_ORIGIN="WISEPACK perception environment (created just now)"
                return 0
            fi
        fi
    fi

    # 3. Fail, naming the ONE command that fixes it. NEVER the system python —
    #    see the header.
    PERCEPTION_PYTHON_ERROR="$(cat <<EOF
the WISEPACK perception environment is not usable.

Looked for: $candidate

The perception service needs torch, torchvision, cv2, fastapi and uvicorn. The
system python3 has none of them, so it is deliberately NOT used as a fallback,
and nothing is ever installed into it.

Create the environment (this is the only command needed):

    ./scripts/setup_perception.sh

then start WISEPACK as usual. To see what is wrong with an existing one:

    ./scripts/setup_perception.sh --check
EOF
)"
    return 1
}

# --- lifecycle --------------------------------------------------------------

perception_log_tail() {
    [ -f "$PERCEPTION_LOG" ] || return 0
    echo "[perception] --- last lines of $PERCEPTION_LOG ---" >&2
    tail -n 25 "$PERCEPTION_LOG" >&2 2>/dev/null || true
    echo "[perception] --- end of log ---" >&2
}

perception_cleanup() {
    # OWNERSHIP IS BY PID/SESSION, and only the one this invocation created.
    # There is deliberately no `pkill`, no `pgrep -f`, and no matching on the
    # process name: an operator debugging the detector by hand in another
    # terminal must not have it killed by an unrelated WISEPACK run, and a
    # pattern like `pkill -f perception_service` would do exactly that.
    [ -n "${PERCEPTION_OWNED_PID:-}" ] || return 0

    echo "[perception] stopping the detector service this launcher started" \
         "(pid $PERCEPTION_OWNED_PID)"
    if [ -n "${PERCEPTION_OWNED_PGID:-}" ]; then
        kill -TERM "-$PERCEPTION_OWNED_PGID" 2>/dev/null || true
    fi
    kill -TERM "$PERCEPTION_OWNED_PID" 2>/dev/null || true

    local waited=0
    while [ "$waited" -lt 40 ]; do
        kill -0 "$PERCEPTION_OWNED_PID" 2>/dev/null || break
        sleep 0.25
        waited=$((waited + 1))
    done
    if kill -0 "$PERCEPTION_OWNED_PID" 2>/dev/null; then
        echo "[perception] service did not exit on TERM — sending KILL" >&2
        [ -n "${PERCEPTION_OWNED_PGID:-}" ] \
            && kill -KILL "-$PERCEPTION_OWNED_PGID" 2>/dev/null || true
        kill -KILL "$PERCEPTION_OWNED_PID" 2>/dev/null || true
    fi

    PERCEPTION_OWNED_PID=""
    PERCEPTION_OWNED_PGID=""
    return 0
}

#: Start the service in its OWN session, so cleanup can signal the whole tree
#: without ever touching a process this launcher did not create.
#:
#: $1 repo root, $2 bind host, $3 bind port.
perception_start_service() {
    local repo="$1" host="$2" port="$3"
    local service="$repo/perception/perception_service.py"

    if [ ! -f "$service" ]; then
        echo "[perception] ERROR: $service is missing" >&2
        return 1
    fi

    : > "$PERCEPTION_LOG" 2>/dev/null || true

    # The child inherits the exported WISEPACK_PERCEPTION_* / WISEPACK_PHYSICAL_*
    # settings from this shell — they are exported by scripts/lib_local_env.sh
    # and by any inline override — so there is no second list here to fall out
    # of step with the allowlist.
    #
    # NO MIDDLEWARE IS SOURCED, because the service speaks HTTP only: no host
    # ROS 2, no host Vulcanexus, no host Fast DDS. WISEPACK's validated
    # middleware is the CONTAINERIZED Vulcanexus runtime, and the
    # WISEPACK-domain perception topics are published from inside it by the
    # orchestrator. `setsid` gives the service its own session so cleanup can
    # signal exactly the tree this launcher created, and nothing else.
    setsid "$PERCEPTION_PYTHON" "$service" --host "$host" --port "$port" \
        >"$PERCEPTION_LOG" 2>&1 &

    PERCEPTION_OWNED_PID=$!
    # Read the real group back rather than assuming it equals the PID: `setsid`
    # forks when the caller is already a process-group leader, and then it does
    # not.
    PERCEPTION_OWNED_PGID="$(ps -o pgid= -p "$PERCEPTION_OWNED_PID" 2>/dev/null \
        | tr -d ' ')"
    [ -n "$PERCEPTION_OWNED_PGID" ] || PERCEPTION_OWNED_PGID="$PERCEPTION_OWNED_PID"

    # Registered the moment it exists, so a failure during the readiness wait —
    # or a Ctrl-C in the middle of it — still stops what was started.
    host_register_cleanup perception_cleanup
    return 0
}

perception_wait_ready() {
    local url="$1"
    local waited=0
    while [ "$waited" -lt "$PERCEPTION_READY_TIMEOUT" ]; do
        # LIVENESS FIRST. A service that died two seconds in must be reported as
        # dead now, not after the full readiness timeout has elapsed.
        if ! kill -0 "$PERCEPTION_OWNED_PID" 2>/dev/null; then
            echo "[perception] ERROR: the detector service exited during start-up." >&2
            perception_log_tail
            return 1
        fi
        if perception_health_ok "$url"; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo "[perception] ERROR: the detector service did not become healthy within" \
         "${PERCEPTION_READY_TIMEOUT}s at $url." >&2
    perception_log_tail
    return 1
}

# --- the entry point --------------------------------------------------------

#: $1 repo root.
#:
#: Returns 0 when a healthy service is available (started here or already
#: running), non-zero otherwise. A non-zero return must ABORT the launch: §15
#: forbids continuing into simulated perception once a camera was asked for.
perception_ensure_service() {
    local repo="$1"
    local url host port
    url="$(perception_service_url)"
    host="$(perception_url_field "$url" hostname)"
    port="$(perception_url_field "$url" port)"

    echo "[perception] perception source : camera"
    echo "[perception] service URL       : $url"

    # ALREADY RUNNING? Reuse it, and say so. An operator who started the
    # detector by hand — to watch its log, or to keep the model loaded across
    # WISEPACK restarts — must not have it restarted underneath them, and must
    # not have it stopped when this launcher exits.
    if perception_health_ok "$url"; then
        echo "[perception] detector service  : already running (external; will not be stopped)"
        echo "[perception] provider          : $(perception_health_provider "$url")"
        return 0
    fi

    # Only a LOCAL service can be started from here. Saying so beats binding
    # 127.0.0.1 and then failing to reach the configured host.
    if ! perception_is_local_host "$host"; then
        echo "[perception] ERROR: no detector service is answering at $url, and" >&2
        echo "[perception]        $host is not this machine — this launcher can" >&2
        echo "[perception]        only start a LOCAL service. Start it on $host," >&2
        echo "[perception]        or point WISEPACK_PERCEPTION_SERVICE_URL at a" >&2
        echo "[perception]        service that is running." >&2
        return 1
    fi

    if ! perception_resolve_python "$repo"; then
        echo "[perception] ERROR: $PERCEPTION_PYTHON_ERROR" >&2
        return 1
    fi

    echo "[perception] detector service  : starting"
    echo "[perception] perception python : $PERCEPTION_PYTHON ($PERCEPTION_PYTHON_ORIGIN)"
    echo "[perception] camera            : ${WISEPACK_PERCEPTION_CAMERA:-0 (default)}"
    echo "[perception] service log       : $PERCEPTION_LOG"

    perception_start_service "$repo" "$host" "$port" || return 1

    if ! perception_wait_ready "$url"; then
        # Stop ONLY what was started here before returning the failure.
        perception_cleanup
        return 1
    fi

    echo "[perception] detector service  : ready"
    echo "[perception] provider          : $(perception_health_provider "$url")"
    return 0
}
