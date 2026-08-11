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
#: Where the worker writes controlled RGB-D captures. WISEPACK-owned and
#: git-ignored: these are measurements, not source, and they are large.
CAPTURES_DIR="${WISEPACK_FP_CAPTURES_DIR:-$REPO/.cache-perception/rgbd-captures}"
#: Deterministic reference cases rendered in Isaac Sim, with ground-truth poses.
ISAAC_REFERENCE_DIR="${WISEPACK_FP_ISAAC_DIR:-$REPO/.cache-perception/isaac-reference}"

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

# --- RealSense USB passthrough ----------------------------------------------
#
# THE WORKER OWNS RGB-D ACQUISITION, so the container needs access to the
# camera's USB node — and to NOTHING ELSE. `--privileged` would work and is
# refused: it grants every device on the host to a container that needs one.
#
# NARROWEST THING THAT WORKS: `--device` for the nodes THIS Intel device owns,
# found by its Intel vendor id in sysfs — the USB node
# `/dev/bus/usb/<bus>/<dev>` AND the `/dev/video*` nodes that hang off this same
# device's interfaces. `--privileged` would work and is refused.
#
# WHY /dev/video* IS NEEDED, having once been argued against here: nothing in
# WISEPACK opens a video node by index — but librealsense's Linux backend is
# V4L2. It enumerates through /sys/class/video4linux and then OPENS
# /dev/videoN. With the USB node alone, `rs.context().query_devices()` returns
# an EMPTY list inside the container while the camera is plainly on the bus —
# which reads as a broken camera rather than as a missing device node.
#
# THE PLANAR WEBCAM IS STILL NOT EXPOSED. The video nodes are discovered by
# walking down from the Intel USB device in sysfs, never by index, so the
# webcam's own nodes (a different USB device) are not passed and cannot be.
#
# THE TRADE-OFF, STATED: a USB node number changes when the camera is replugged,
# so the worker must be restarted after a replug. That is the price of not
# mounting the whole USB bus, and it is the right way round for a camera that is
# plugged in once and left.
INTEL_VENDOR_ID="8086"

#: Echoes `--device <node>` for every node each RealSense owns, or nothing.
realsense_usb_args() {
    local sysfs bus dev vendor node v4l
    for sysfs in /sys/bus/usb/devices/*/; do
        vendor="$(cat "${sysfs}idVendor" 2>/dev/null)"
        [ "$vendor" = "$INTEL_VENDOR_ID" ] || continue
        bus="$(cat "${sysfs}busnum" 2>/dev/null)"
        dev="$(cat "${sysfs}devnum" 2>/dev/null)"
        [ -n "$bus" ] && [ -n "$dev" ] || continue
        node="$(printf '/dev/bus/usb/%03d/%03d' "$bus" "$dev")"
        [ -e "$node" ] && printf -- '--device\n%s\n' "$node"
        # The V4L2 nodes of THIS device: <usb device>/<interface>/video4linux/videoN
        for v4l in "${sysfs}"*/video4linux/video*; do
            [ -e "/dev/$(basename "$v4l")" ] || continue
            printf -- '--device\n/dev/%s\n' "$(basename "$v4l")"
        done
    done
}

host_sees_realsense() {
    [ -n "$(realsense_usb_args)" ]
}

#: Does GPU passthrough ACTUALLY WORK, rather than merely appear configured?
#:
#: `docker info` lists whatever runtimes daemon.json names, and daemon.json can
# --- GPU access ------------------------------------------------------------
#
# SHARED WITH THE MODEL-FREE BUILD. This host has no NVIDIA Container Toolkit,
# so reaching the GPU takes a verified fallback rather than `--gpus all`; the
# model-free Neural Object Field build needs exactly the same answer, and two
# copies would agree only until one was edited.
. "$REPO/scripts/lib_foundationpose_gpu.sh"

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
    echo "captures dir : $CAPTURES_DIR"
    echo "realsense    : $(host_sees_realsense && echo "host sees $(( $(realsense_usb_args | wc -l) / 2 )) Intel device node(s)" || echo 'NOT present on the host USB bus')"
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
    # CAPTURES ARE WRITABLE — they are the one thing the worker produces that
    # must outlive the container.
    mkdir -p "$CAPTURES_DIR" 2>/dev/null || true
    [ -d "$CAPTURES_DIR" ] && MOUNTS+=(-v "$CAPTURES_DIR:/captures")
    # The Isaac-generated reference cases, mounted INSIDE the dataset tree so a
    # request names them exactly like any other dataset. Read-only: they are
    # generated artefacts and the worker never writes to them.
    # A SIBLING of /datasets, not a child: Docker cannot create a mountpoint
    # inside a read-only mount, and the worker searches both roots.
    [ -d "$ISAAC_REFERENCE_DIR" ] \
        && MOUNTS+=(-v "$ISAAC_REFERENCE_DIR:/isaac-reference:ro")

    USB_ARGS=()
    if host_sees_realsense; then
        mapfile -t USB_ARGS < <(realsense_usb_args)
        say "RealSense: passing through $(( ${#USB_ARGS[@]} / 2 )) device node(s)"
    else
        say "RealSense: no Intel USB device on this host — the worker starts"
        say "           without camera access and reports rgbd_camera_available"
        say "           false. Plug the camera in, then re-run --run."
    fi

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
        "${USB_ARGS[@]}" \
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
