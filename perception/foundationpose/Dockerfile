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

# --- Kaolin: REQUIRED BY THE MODEL-FREE / FEW-SHOT PATH ONLY ----------------
#
# Upstream says so itself (readme "Optional — Kaolin (only for the model-free
# setup; version must match your PyTorch/CUDA)"). The CAD path never touches it;
# the model-free path does, through `Utils.py`'s `OctreeManager`, which the NeRF
# uses for ray tracing because the official configs set `use_octree: 1`:
#
#     Utils.py:908-944   kaolin.ops.spc.quantize_points / unbatched_points_to_octree
#                        / scan_octrees / generate_points / unbatched_make_dual
#                        / unbatched_make_trinkets / unbatched_query
#     nerf_runner.py:303 near/far bounds from octree_m.ray_trace(...)
#
# AN OFFICIAL WHEEL, NOT A SOURCE BUILD, and one that matches the runtime
# already pinned above EXACTLY — python 3.10, torch 2.4.1, CUDA 12.4. Nothing
# above this line moves: not the FoundationPose revision, not torch, not CUDA,
# not pytorch3d, not nvdiffrast. Building Kaolin from source would compile
# against the same torch for no benefit.
#
# PLACED HERE, AFTER THE COMPILED LAYERS, so adding it re-runs no CUDA build.
ARG KAOLIN_VERSION=0.17.0
ARG KAOLIN_INDEX=https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu124.html
# TORCH IS PINNED THROUGH A CONSTRAINT, not left to pip's resolver. Kaolin
# declares a torch dependency, and a resolver free to satisfy it could pull a
# different build — which would silently unlink every CUDA extension compiled
# above. The constraint makes "torch does not move" a property of the build
# rather than something to verify afterwards and hope for.
RUN printf 'torch==%s\ntorchvision\n' "${TORCH_VERSION}" > /tmp/pin.txt \
    && python -m pip install -c /tmp/pin.txt \
        kaolin==${KAOLIN_VERSION} -f ${KAOLIN_INDEX} \
    && rm -f /tmp/pin.txt

# --- trimesh: PINNED TO WHAT THE PINNED REVISION CAN ACTUALLY CALL ----------
#
# Upstream's `requirements.txt` lists `trimesh` with NO version, so the build
# resolved whatever was current — 5.0.0 — while the pinned FoundationPose
# revision was written against trimesh 3.x. The model-free texturing stage then
# died on an API that no longer exists:
#
#     nerf_runner.py:1138   mesh.remove_duplicate_faces()
#     AttributeError: 'Trimesh' object has no attribute 'remove_duplicate_faces'
#
# Everything else in this file is pinned precisely so the runtime matches the
# revision; trimesh floating was the omission, not a deliberate choice.
#
# 4.8.3 IS MEASURED, NOT GUESSED. The symbol was checked in each release:
# present in 3.23.5, 4.0.10, 4.4.9 and 4.8.3; absent in 4.12.2 and 5.0.0. So
# 4.8.3 is the NEWEST version the pinned code can call, which keeps the drift
# from the current runtime as small as the incompatibility allows.
#
# NO SOURCE PATCH. FoundationPose is used as published; the dependency is what
# moves, because the dependency is what was wrong.
#
# numpy IS IN THE CONSTRAINT because trimesh would otherwise be free to pull
# numpy 2.x back in and break kaolin, which requires <2.0.
ARG TRIMESH_VERSION=4.8.3
# xatlas IS THE UV-UNWRAP BACKEND `trimesh.unwrap()` IMPORTS, and the model-free
# texturing stage calls it one line after `remove_duplicate_faces`:
#
#     nerf_runner.py:1139   mesh = mesh.unwrap()
#     trimesh/base.py:2353    import xatlas
#
# Upstream declares it in neither `requirements.txt` nor `environment.yml` — the
# same omission as the unpinned trimesh, and the last thing between a trained
# Neural Object Field and a textured mesh. A prebuilt manylinux wheel, ~260 kB,
# MIT-licensed, so it changes nothing about the NVIDIA licence position.
ARG XATLAS_VERSION=0.0.11
# PyOpenGL 3.1.0 CANNOT DRIVE PYRENDER'S OWN OSMESA BACKEND. `pyrender`'s
# `platforms/osmesa.py` imports `OSMesaCreateContextAttribs`, and 3.1.0 ships
# only `OSMesaCreateContext`. Checked per release: absent in 3.1.0, present in
# 3.1.5 and 3.1.7.
#
# THE PIN THIS VIOLATES IS ITSELF THE DEFECT. The published pyrender 0.1.45
# wheel records `Requires-Dist: PyOpenGL (==3.1.0)` — verified byte-for-byte in
# the installed dist-info — pinning the one version whose OSMesa support its own
# OSMesa backend cannot use. FoundationPose's own requirement is `pyOpenGL>=3.1`,
# which 3.1.7 satisfies; the environment was already inconsistent anyway, with
# pyopengl-accelerate at 3.1.10 against pyopengl 3.1.0.
#
# `pip check` therefore reports pyrender's metadata constraint. That is expected
# and is left alone: pyrender is NOT changed to silence it, because the metadata
# is what is wrong, not the runtime. `--no-deps` keeps the change to PyOpenGL
# alone — nothing else is allowed to move to make version numbers look tidy.
ARG PYOPENGL_VERSION=3.1.7
# rtree IS TRIMESH'S SPATIAL INDEX, and the texture bake reaches it:
#
#     nerf_runner.py:1161   trimesh.proximity.closest_point(mesh, pts)
#     trimesh/proximity.py:52  nearby_faces -> triangles_tree -> util.bounds_tree
#     ModuleNotFoundError: No module named 'rtree'
#
# trimesh declares it only as an optional extra (`rtree; extra == "easy"`) and
# FoundationPose's requirements.txt does not list it — the same omission as
# xatlas. A prebuilt manylinux wheel with libspatialindex bundled, so no apt
# package and no source build.
ARG RTREE_VERSION=1.4.1
RUN printf 'torch==%s\ntorchvision\nnumpy<2.0\n' "${TORCH_VERSION}" > /tmp/pin.txt \
    && python -m pip install -c /tmp/pin.txt \
        trimesh==${TRIMESH_VERSION} xatlas==${XATLAS_VERSION} \
    && python -m pip install -c /tmp/pin.txt --no-deps \
        PyOpenGL==${PYOPENGL_VERSION} \
    && python -m pip install -c /tmp/pin.txt --no-deps \
        rtree==${RTREE_VERSION} \
    && rm -f /tmp/pin.txt

# --- OSMesa: HEADLESS GL FOR THE MODEL-FREE TEXTURE BAKE --------------------
#
# The last stage of the official model-free build renders the reconstructed mesh
# from the training views to bake a texture:
#
#     nerf_runner.py:1143   ModelRendererOffscreen(...)
#     offscreen_renderer.py:47   pyrender.OffscreenRenderer(W, H)
#
# pyrender defaults to opening an X11 window through pyglet. Upstream never hits
# this because their `docker/run_container.sh` runs with a real display mounted
# in; THIS WORKER IS DELIBERATELY HEADLESS, and a container that needs an X
# server to finish a build is a container that cannot run unattended.
#
# OSMesa is pyrender's own supported offscreen backend and needs no X server, no
# display and no extra driver mounts — one apt package. It changes WHICH GL
# BACKEND the official texture-baking code renders through, and nothing about
# the NeRF, the reconstruction or the pose algorithm. FoundationPose itself is
# untouched.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# CHOSEN AT IMPORT TIME, so it is declared here rather than set late in Python.
# PyOpenGL picks its platform the first time it is imported, and anything that
# imports `pyrender` — or `OpenGL` directly, or `offscreen_renderer` — fixes the
# choice for the whole process. Setting it inside FoundationPose source would be
# both a patch and, in most import orders, too late.
ENV PYOPENGL_PLATFORM=osmesa

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
# `bundlesdf` ITSELF IS ON THE PATH, and that is the whole mycuda fix. The
# extensions were always built and always worked — `.so` files for `common` and
# `gridencoder` are in the image and export the symbols the NeRF calls — but
# upstream imports the same package two ways:
#
#     Utils.py:52          from bundlesdf.mycuda import common     (namespace pkg,
#                                                                   already resolved)
#     nerf_helpers.py:204  from mycuda.torch_ngp_grid_encoder.grid import GridEncoder
#     Utils.py:973         from mycuda import common
#
# The second form needs the bundlesdf DIRECTORY on sys.path. Nothing is copied
# and nothing is rebuilt: this is import visibility for binaries that already
# exist at their official upstream locations.
ENV PYTHONPATH=${FOUNDATIONPOSE_DIR}:${FOUNDATIONPOSE_DIR}/bundlesdf:${FOUNDATIONPOSE_DIR}/mycpp/build:/opt/wisepack-fp-worker

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
