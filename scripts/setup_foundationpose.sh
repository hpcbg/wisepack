#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_foundationpose.sh — build the WISEPACK-managed FoundationPose worker.
#
#     ./scripts/setup_foundationpose.sh              # build the image
#     ./scripts/setup_foundationpose.sh --weights    # ... and fetch the weights
#     ./scripts/setup_foundationpose.sh --check      # report, change nothing
#     ./scripts/setup_foundationpose.sh --run        # build if needed, then run
#     ./scripts/setup_foundationpose.sh --stop       # stop the container WE own
#
# THIS IS OPT-IN AND IT IS NOT PART OF ORDINARY WISEPACK.
# Preset scenarios and the planar camera provider never touch any of it. Nothing
# in `./run_wisepack_dashboard.sh` builds this image.
#
# LICENCE, BEFORE ANYTHING ELSE
# -----------------------------
# FoundationPose is third-party software released by NVIDIA under the NVIDIA
# Source Code License — NON-COMMERCIAL RESEARCH USE ONLY. WISEPACK is MIT and
# vendors none of it: the image is built from a recipe that clones a pinned
# upstream revision, and the network weights are fetched separately from the
# official source. Building this image opts you into that licence.
#
# WHY A CONTAINER RATHER THAN .venv-perception
# --------------------------------------------
# The planar provider's environment works and is small. FoundationPose needs a
# CUDA toolchain, compiled CUDA extensions, pytorch3d and nvdiffrast; putting
# that into `.venv-perception` would risk the working detector for the sake of
# an optional one. Separate GPU container, separate lifetime, no shared state.
# ---------------------------------------------------------------------------

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

#: The image and the container this script owns. DETERMINISTIC NAMES, because
#: ownership is by exact name here and nothing else: no `docker kill $(...)`, no
#: pattern matching, no pkill. A container WISEPACK did not create is never
#: touched.
IMAGE="${WISEPACK_FP_IMAGE:-wisepack-foundationpose:pinned}"
CONTAINER="${WISEPACK_FP_CONTAINER:-wisepack-foundationpose-worker}"
PORT="${WISEPACK_FP_PORT:-22201}"

#: Host paths. Both WISEPACK-owned and git-ignored; mounted READ-ONLY.
WEIGHTS_DIR="${WISEPACK_FP_WEIGHTS_DIR:-$REPO/.cache-perception/foundationpose/weights}"
#: The reference datasets. `references/` already lives beside the repository, so
#: it is MOUNTED rather than copied — duplicating a 183 MB tree to satisfy a
#: container layout would be waste, and a second copy is a second thing to drift.
DATASETS_DIR="${WISEPACK_FP_DATASETS_DIR:-$(cd "$REPO/.." 2>/dev/null && pwd)/references}"

DO_BUILD=1
DO_WEIGHTS=0
DO_RUN=0
DO_STOP=0
CHECK_ONLY=0

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --weights) DO_WEIGHTS=1 ;;
        --run)     DO_RUN=1 ;;
        --stop)    DO_STOP=1; DO_BUILD=0 ;;
        --check)   CHECK_ONLY=1; DO_BUILD=0 ;;
        --no-build) DO_BUILD=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "setup_foundationpose.sh: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

say() { echo "[foundationpose] $*"; }

# --- preconditions ----------------------------------------------------------

have_docker() { command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; }

#: Does GPU passthrough ACTUALLY WORK, rather than merely appear configured?
#:
#: `docker info` lists whatever runtimes daemon.json names, and daemon.json can
#: name a `nvidia-container-runtime` binary that is not installed — which is
#: exactly the state this host was found in. Trusting that listing produced a
#: `--gpus all` that failed and stopped the worker from starting at all, which
#: is the opposite of the rule this worker is built around: a missing capability
#: must be REPORTED by a running worker, never turned into a dead container.
#:
#: So the question is answered by trying it, once, with a trivial image.
GPU_PASSTHROUGH_CACHE=""
have_gpu_passthrough() {
    if [ -n "$GPU_PASSTHROUGH_CACHE" ]; then
        [ "$GPU_PASSTHROUGH_CACHE" = "yes" ]
        return
    fi
    if docker run --rm --gpus all "$IMAGE" true >/dev/null 2>&1; then
        GPU_PASSTHROUGH_CACHE="yes"
        return 0
    fi
    GPU_PASSTHROUGH_CACHE="no"
    return 1
}

#: Configuration only — what daemon.json claims. Useful for the diagnostic
#: message, never as the decision.
nvidia_runtime_declared() {
    docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia'
}

# --- GPU access without the NVIDIA Container Toolkit -------------------------
#
# WHAT THE TOOLKIT ACTUALLY DOES, when you strip away the plumbing, is two
# things: expose /dev/nvidia* to the container, and put the host's user-space
# driver libraries inside it at their sonames. Both are ordinary `docker run`
# arguments. Installing the toolkit needs root on the host; this does not.
#
# THIS IS A FALLBACK, NOT A RECOMMENDATION. The toolkit is the supported
# mechanism and handles cases this does not (MIG, device enumeration, IPC,
# capability devices). It is used here only when the toolkit is absent, it is
# VERIFIED by actually running torch against it rather than assumed to work, and
# when it fails the worker still starts and still reports gpu_available=false.
#
# The libraries are bind-mounted READ-ONLY and version-matched to the running
# driver: a libcuda from a different driver than the loaded kernel module fails
# with an error that names neither.
DRIVER_LIB_DIR="${WISEPACK_FP_DRIVER_LIB_DIR:-/usr/lib/x86_64-linux-gnu}"

driver_version() {
    nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
        | head -1 | tr -d '[:space:]'
}

#: Emits the `docker run` arguments for manual passthrough, or nothing.
manual_gpu_args() {
    local version; version="$(driver_version)"
    [ -n "$version" ] || return 1
    [ -e /dev/nvidiactl ] || return 1

    local args=()
    local device
    for device in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
        [ -e "$device" ] && args+=(--device "$device")
    done

    # `soname:filename-stem` — the name the loader looks for, and the versioned
    # file that provides it. libcuda is the only mandatory one; the rest are
    # needed for NVML queries and for PTX JIT, which nvdiffrast performs.
    local pair stem soname source
    for pair in "libcuda.so.1:libcuda" \
                "libnvidia-ml.so.1:libnvidia-ml" \
                "libnvidia-ptxjitcompiler.so.1:libnvidia-ptxjitcompiler" \
                "libnvidia-nvvm.so.4:libnvidia-nvvm"; do
        soname="${pair%%:*}"; stem="${pair##*:}"
        source="${DRIVER_LIB_DIR}/${stem}.so.${version}"
        if [ -f "$source" ]; then
            args+=(-v "${source}:${DRIVER_LIB_DIR}/${soname}:ro")
        elif [ "$stem" = "libcuda" ]; then
            return 1   # without this one there is no point continuing
        fi
    done
    # Loaded by libcuda under its own versioned name on recent drivers, so it is
    # mounted where it is looked for rather than at a soname.
    source="${DRIVER_LIB_DIR}/libnvidia-gpucomp.so.${version}"
    [ -f "$source" ] && args+=(-v "${source}:${source}:ro")

    printf '%s\n' "${args[@]}"
}

#: Does manual passthrough actually give torch a usable device? Tried, not
#: assumed — the same rule as have_gpu_passthrough().
MANUAL_GPU_CACHE=""
have_manual_gpu() {
    if [ -n "$MANUAL_GPU_CACHE" ]; then
        [ "$MANUAL_GPU_CACHE" = "yes" ]; return
    fi
    local args=()
    mapfile -t args < <(manual_gpu_args) || true
    if [ ${#args[@]} -eq 0 ]; then
        MANUAL_GPU_CACHE="no"; return 1
    fi
    if docker run --rm "${args[@]}" "$IMAGE" \
            python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
            >/dev/null 2>&1; then
        MANUAL_GPU_CACHE="yes"; return 0
    fi
    MANUAL_GPU_CACHE="no"; return 1
}

image_exists() { docker image inspect "$IMAGE" >/dev/null 2>&1; }

#: Only a container with OUR exact name, and only if WE labelled it. The label
#: is what stops this script adopting somebody else's container that happens to
#: share a name.
owned_container_id() {
    docker ps -a --filter "name=^/${CONTAINER}$" \
                 --filter "label=wisepack.owned=true" \
                 --format '{{.ID}}' 2>/dev/null | head -1
}

weights_state() {
    local refiner="$WEIGHTS_DIR/2023-10-28-18-33-37/model_best.pth"
    local scorer="$WEIGHTS_DIR/2024-01-11-20-02-45/model_best.pth"
    local have=0
    [ -s "$refiner" ] && have=$((have + 1))
    [ -s "$scorer" ] && have=$((have + 1))
    echo "$have"
}

# --- check ------------------------------------------------------------------

if [ "$CHECK_ONLY" = "1" ]; then
    echo "image        : $IMAGE $(image_exists && echo '(present)' || echo '(not built)')"
    echo "container    : $CONTAINER $( [ -n "$(owned_container_id)" ] && echo '(exists, WISEPACK-owned)' || echo '(none)')"
    echo "docker       : $(have_docker && echo usable || echo 'NOT usable')"
    echo "nvidia rt    : $(nvidia_runtime_declared && echo 'declared in daemon.json' || echo 'not declared')"
    if image_exists; then
        if have_gpu_passthrough; then
            echo "gpu passthru : WORKS (NVIDIA Container Toolkit)"
        elif have_manual_gpu; then
            echo "gpu passthru : WORKS via manual device+driver-library mounts"
            echo "               (the NVIDIA Container Toolkit is NOT installed;"
            echo "                driver $(driver_version))"
        else
            echo "gpu passthru : DOES NOT WORK — no toolkit and no usable driver"
        fi
    else
        echo "gpu passthru : unknown (build the image first)"
    fi
    echo "weights dir  : $WEIGHTS_DIR"
    echo "weights      : $(weights_state)/2 checkpoints present"
    echo "datasets dir : $DATASETS_DIR $( [ -d "$DATASETS_DIR" ] && echo '(present)' || echo '(MISSING)')"
    exit 0
fi

# --- stop -------------------------------------------------------------------

if [ "$DO_STOP" = "1" ]; then
    id="$(owned_container_id)"
    if [ -z "$id" ]; then
        say "no WISEPACK-owned worker container to stop"
        exit 0
    fi
    say "stopping the worker this project created ($id)"
    docker stop "$id" >/dev/null 2>&1 || true
    docker rm "$id" >/dev/null 2>&1 || true
    exit 0
fi

# --- weights ----------------------------------------------------------------
#
# FETCHED FROM THE OFFICIAL SOURCE ONLY. Community mirrors of these checkpoints
# exist; using one would mean WISEPACK results depended on weights nobody can
# attest to. When the official source is unavailable the correct behaviour is to
# WAIT, and to say so.

if [ "$DO_WEIGHTS" = "1" ]; then
    say "resolving FoundationPose weights (official source only)"
    mkdir -p "$WEIGHTS_DIR"
    if ! python3 "$REPO/scripts/foundationpose_weights.py" --dir "$WEIGHTS_DIR"; then
        say "WARNING: the weights are not available yet."
        say "         The image still builds and the worker still starts; it"
        say "         will report inference_available=false with the reason."
    fi
fi

# --- build ------------------------------------------------------------------

if [ "$DO_BUILD" = "1" ]; then
    if ! have_docker; then
        echo "[foundationpose] ERROR: Docker is not usable by this user." >&2
        exit 1
    fi
    if ! nvidia_runtime_declared; then
        # NOT FATAL AT BUILD TIME. The image builds fine without the runtime;
        # it is running with a GPU that needs it, and the worker reports that
        # precisely rather than failing here with a confusing message.
        say "WARNING: the NVIDIA container runtime is not declared to Docker."
        say "         The image will build, but the worker will report"
        say "         gpu_available=false until the toolkit is installed."
    fi
    say "building $IMAGE (this takes a while: CUDA extensions are compiled)"
    docker build \
        -f "$REPO/perception/foundationpose/Dockerfile" \
        -t "$IMAGE" \
        "$REPO/perception/foundationpose" || {
            echo "[foundationpose] ERROR: image build failed." >&2
            exit 1
        }
    say "built $IMAGE"
fi

# --- run --------------------------------------------------------------------

if [ "$DO_RUN" = "1" ]; then
    if ! image_exists; then
        echo "[foundationpose] ERROR: $IMAGE does not exist — build it first." >&2
        exit 1
    fi
    existing="$(owned_container_id)"
    if [ -n "$existing" ]; then
        say "removing the previous WISEPACK-owned worker ($existing)"
        docker rm -f "$existing" >/dev/null 2>&1 || true
    fi

    # GPU ARGS ONLY IF THEY ACTUALLY WORK. A `--gpus all` that the daemon
    # rejects does not degrade the worker — it prevents it from starting, and a
    # container that cannot start cannot tell anyone why. Starting without the
    # GPU leaves a worker that answers `gpu_available: false` with the reason,
    # which is the diagnosable outcome.
    GPU_ARGS=()
    if have_gpu_passthrough; then
        GPU_ARGS=(--gpus all)
        say "GPU: --gpus all (NVIDIA Container Toolkit)"
    elif have_manual_gpu; then
        mapfile -t GPU_ARGS < <(manual_gpu_args)
        say "GPU: the NVIDIA Container Toolkit is NOT installed; using verified"
        say "     manual passthrough (driver $(driver_version), /dev/nvidia*"
        say "     plus read-only driver libraries). Installing the toolkit is"
        say "     the supported route and would replace this:"
        say "     https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
    else
        say "WARNING: no GPU access works on this host — starting the worker"
        say "         WITHOUT a GPU. It will answer /health with"
        say "         gpu_available=false and inference_available=false."
        say "         Install the NVIDIA Container Toolkit to enable inference:"
        say "         https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
    fi

    MOUNTS=()
    [ -d "$WEIGHTS_DIR" ] && MOUNTS+=(-v "$WEIGHTS_DIR:/weights:ro")
    [ -d "$DATASETS_DIR" ] && MOUNTS+=(-v "$DATASETS_DIR:/datasets:ro")

    say "starting $CONTAINER on port $PORT"
    # `--label wisepack.owned=true` is the ownership record: cleanup keys on the
    # exact name AND this label, so a container this project did not create can
    # never be stopped by it.
    #
    # NO DOCKER SOCKET IS MOUNTED anywhere, and no --privileged.
    docker run -d \
        --name "$CONTAINER" \
        --label wisepack.owned=true \
        --label wisepack.component=foundationpose-worker \
        "${GPU_ARGS[@]}" \
        -p "127.0.0.1:${PORT}:22201" \
        "${MOUNTS[@]}" \
        "$IMAGE" >/dev/null || {
            echo "[foundationpose] ERROR: could not start the worker." >&2
            exit 1
        }
    say "worker started. Capability:"
    sleep 3
    docker logs "$CONTAINER" 2>&1 | head -12 | sed 's/^/[foundationpose] /'
    say "health: curl -s http://127.0.0.1:${PORT}/health"
fi

exit 0
