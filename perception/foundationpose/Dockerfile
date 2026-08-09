# WISEPACK-managed FoundationPose worker.
#
# WHY A SEPARATE IMAGE AT ALL
# ---------------------------
# The planar provider lives in `.venv-perception`, a small virtualenv that
# reuses whatever torch the host already has. FoundationPose cannot: it needs a
# CUDA toolchain, custom CUDA extensions, pytorch3d and nvdiffrast, all compiled
# against one specific CUDA. Installing that into `.venv-perception` would put
# the working planar detector at the mercy of a much heavier dependency set, so
# it gets its own GPU container and the planar environment stays untouched.
#
# WHAT THIS IMAGE CONTAINS, AND WHAT IT DELIBERATELY DOES NOT
# -----------------------------------------------------------
# CONTAINS: the runtime — CUDA, torch, pytorch3d, nvdiffrast, a pinned checkout
# of FoundationPose, and the small WISEPACK worker that wraps it.
#
# DOES NOT CONTAIN: the network weights. They are ~1 GB, are distributed under
# the FoundationPose licence, and are fetched to a host-side WISEPACK cache and
# mounted READ-ONLY at run time. Baking them in would make the image
# undistributable and would tie a licensed artefact to a build.
#
# LICENCE. FoundationPose is released under the NVIDIA Source Code License —
# NON-COMMERCIAL RESEARCH USE ONLY. WISEPACK is MIT and vendors none of it: this
# file is a build recipe that clones a pinned upstream revision. Building this
# image opts you into that licence; nothing in a default WISEPACK install does.
# See NOTICE and perception/foundationpose/UPSTREAM_NOTES.md.
#
#     ./scripts/setup_foundationpose.sh            # build
#     ./scripts/setup_foundationpose.sh --weights  # ... and fetch the weights
#
# WHY THIS PIN AND NOT UPSTREAM'S
# -------------------------------
# Upstream's own dockerfile targets nvidia/cudagl:11.3 + conda + python 3.8.
# CUDA 11.3 predates Ada (SM 8.9), so the CUDA extensions would build without an
# architecture this RTX 4090 can execute — it would fall back to PTX JIT or fail
# outright. CUDA 12.4 supports SM 8.9 natively and still has broad wheel
# coverage for the torch/pytorch3d/nvdiffrast combination this needs.

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

# The revision this image is built from. Recorded in the image and reported by
# /health, so a result can always be traced to the code that produced it.
ARG FOUNDATIONPOSE_REF=a1b694b83e633c2cb6115b9063d940a687759392
ARG NVDIFFRAST_REF=v0.3.3
ARG PYTORCH3D_REF=V0.7.8
ARG TORCH_VERSION=2.4.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # BUILD FOR THIS GPU AND NOTHING ELSE. Ada is SM 8.9; naming it keeps the
    # CUDA extension builds short and guarantees native cubins rather than a
    # silent PTX-JIT fallback that only shows up as a slow first inference.
    TORCH_CUDA_ARCH_LIST="8.9" \
    CUDA_HOME=/usr/local/cuda \
    FOUNDATIONPOSE_DIR=/opt/foundationpose \
    FOUNDATIONPOSE_REF=${FOUNDATIONPOSE_REF} \
    # Where the read-only mounts land. Both are WISEPACK-owned host paths.
    WISEPACK_FP_WEIGHTS_DIR=/weights \
    WISEPACK_FP_DATASETS_DIR=/datasets

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ninja-build pkg-config \
        python3 python3-dev python3-pip \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libeigen3-dev libboost-all-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

RUN python -m pip install --upgrade pip setuptools wheel ninja

# --- torch first, so every native build below links against ONE torch -------
RUN python -m pip install \
        torch==${TORCH_VERSION} torchvision --index-url ${TORCH_INDEX}

# --- the worker's own (light) dependencies ----------------------------------
COPY worker/requirements.txt /tmp/worker-requirements.txt
RUN python -m pip install -r /tmp/worker-requirements.txt

# --- pytorch3d and nvdiffrast: compiled, no wheels exist for this combination
#
# `--no-build-isolation` is REQUIRED, not a preference: with isolation pip
# installs a fresh torch into the build environment and the extension links
# against that one instead of the torch above, producing an import error at
# run time that names neither package.
RUN python -m pip install --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_REF}"

RUN python -m pip install --no-build-isolation \
        "git+https://github.com/NVlabs/nvdiffrast.git@${NVDIFFRAST_REF}"

# --- FoundationPose itself: cloned at a PINNED revision, never vendored -----
RUN git clone https://github.com/NVlabs/FoundationPose.git ${FOUNDATIONPOSE_DIR} \
    && cd ${FOUNDATIONPOSE_DIR} \
    && git checkout ${FOUNDATIONPOSE_REF} \
    && git rev-parse HEAD > /opt/foundationpose.revision

RUN python -m pip install -r ${FOUNDATIONPOSE_DIR}/requirements.txt || true

# The upstream CUDA extension still declares c++14, which modern nvcc rejects
# against this torch. Upstream's own install notes carry the same patch.
RUN sed -i 's/c++14/c++17/g' ${FOUNDATIONPOSE_DIR}/bundlesdf/mycuda/setup.py || true

# BUILT WITH `|| true`, AND THAT IS DELIBERATE. A failed extension build must
# still produce a startable container: the worker probes each capability at run
# time and reports precisely which one is missing (§2). An image that refuses to
# build teaches nothing; a container that starts and says
# "foundationpose_runtime_available: false, last_error: <the real reason>" is
# diagnosable from the dashboard.
RUN cd ${FOUNDATIONPOSE_DIR}/bundlesdf/mycuda \
    && (python -m pip install --no-build-isolation -e . || \
        echo "mycuda extension build FAILED — worker will report it") \
    && true

# --- mycpp: REQUIRED, despite upstream treating it as optional --------------
#
# `Utils.py` wraps `import mycpp` in a try/except and leaves it None on failure,
# so a container without it imports `estimater` perfectly happily — and then dies
# inside `register()` at estimater.py:120, where `mycpp.cluster_poses` prunes the
# rotation grid. Optional at import, mandatory in practice. It is built here and
# probed by name at run time, so a missing one is reported as a missing
# capability rather than discovered as a crash mid-estimate.
#
# It is a CMake/pybind11 module rather than a pip package, hence the separate
# build; `pybind11` is needed only to locate its CMake config.
RUN python -m pip install pybind11     && cd ${FOUNDATIONPOSE_DIR}/mycpp     && mkdir -p build && cd build     && (cmake -DCMAKE_PREFIX_PATH="$(python -c 'import pybind11;print(pybind11.get_cmake_dir())')" ..         && make -j"$(nproc)"         || echo "mycpp extension build FAILED — worker will report it")     && true

# WEIGHTS ARE STILL NOT IN THE IMAGE — this is a symlink, and it dangles until
# the host mount appears. `ScorePredictor` and `PoseRefinePredictor` build their
# checkpoint paths as `<code_dir>/weights/<run_name>/model_best.pth` with no
# override parameter, so the read-only mount has to be visible there. Pointing
# the expected path at /weights keeps the checkpoints on the host, outside the
# image, under their own licence — copying them in would put NVIDIA-licensed
# weights inside a WISEPACK artefact.
RUN ln -sfn /weights ${FOUNDATIONPOSE_DIR}/weights

# --- RealSense runtime deps -------------------------------------------------
#
# `pyrealsense2` ships the librealsense runtime in its wheel; these are the
# system libraries that wheel links against for USB access. Installed here
# rather than on the host: the container owns RGB-D acquisition.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libusb-1.0-0 udev \
    && rm -rf /var/lib/apt/lists/*

# --- the WISEPACK worker ----------------------------------------------------
COPY worker /opt/wisepack-fp-worker
# `mycpp/build` is where CMake leaves that module. Upstream appends it to
# sys.path from Utils.py; the worker needs it importable directly, because it
# probes for it BY NAME before accepting any estimate request.
ENV PYTHONPATH=${FOUNDATIONPOSE_DIR}:${FOUNDATIONPOSE_DIR}/mycpp/build:/opt/wisepack-fp-worker

WORKDIR /opt/wisepack-fp-worker

# Read-only in practice: the run helper mounts the weights and datasets with
# `:ro`. Declared here so the intent survives a hand-written `docker run`.
VOLUME ["/weights", "/datasets"]

EXPOSE 22201

# NO HEALTHCHECK THAT DEPENDS ON WEIGHTS. The container is healthy when the
# worker answers; whether it can infer is a FIELD in that answer, not a
# liveness condition.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:22201/health', timeout=4).status==200 else 1)"

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "22201"]
