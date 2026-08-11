#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# lib_foundationpose_gpu.sh — how a FoundationPose container reaches the GPU.
#
#     IMAGE="wisepack-foundationpose:pinned"
#     . "$REPO/scripts/lib_foundationpose_gpu.sh"
#     GPU_ARGS=(); foundationpose_gpu_args GPU_ARGS || echo "no GPU"
#
# WHY THIS IS A LIBRARY. Two things run FoundationPose containers now: the
# long-lived worker (`setup_foundationpose.sh`) and the model-free Neural Object
# Field build (`model_free_build.sh`), a batch job with different mounts. Both
# need the same answer to "how does a container on THIS host reach the GPU",
# and this host has no NVIDIA Container Toolkit — so the answer is a verified
# fallback rather than `--gpus all`. Two copies of that would agree only until
# one was edited, and the failure mode is a build silently running on the CPU.
#
# `IMAGE` must be set before sourcing: every probe here answers by RUNNING the
# image, never by reading configuration.
# ---------------------------------------------------------------------------

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


#: The `docker run` arguments for this host, returned by name. Non-zero when no
#: GPU can be offered at all — which the caller reports rather than ignoring.
foundationpose_gpu_args() {
    local -n _out="$1"
    _out=()
    if have_gpu_passthrough; then
        _out=(--gpus all)
        return 0
    fi
    if have_manual_gpu; then
        mapfile -t _out < <(manual_gpu_args)
        return 0
    fi
    return 1
}
