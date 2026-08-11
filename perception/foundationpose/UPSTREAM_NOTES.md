# Upstream FoundationPose: what WISEPACK pins, and why

Two independent implementations of FoundationPose exist. WISEPACK uses one of
them as its numeric reference and treats the other as a comparison target only.
This file records what was inspected and the reasoning, so the choice can be
re-evaluated rather than re-guessed.

---

## 1. Official NVLabs FoundationPose — the numeric reference

* `https://github.com/NVlabs/FoundationPose`
* `main` @ **`a1b694b83e633c2cb6115b9063d940a687759392`**
* Licence: **NVIDIA Source Code License — non-commercial research use only.**

Runtime: `ScorePredictor` + `PoseRefinePredictor` (PyTorch), `nvdiffrast`,
`pytorch3d`, and custom CUDA extensions built by `build_all.sh`.

Weights: **`.pth` checkpoints distributed only via Google Drive** — refiner
`2023-10-28-18-33-37`, scorer `2024-01-11-20-02-45`. There is no versioned
public URL, so they cannot be baked into an image reproducibly.

**Pose convention.** `est.register()` returns `ob_in_cam`, a 4×4 for the mesh
**as loaded**, in the camera frame. `run_demo.py` separately computes
`center_pose = pose @ inv(to_origin)` — where `to_origin` comes from
`trimesh.bounds.oriented_bounds(mesh)` — purely to draw the 3-D box. Note that
`oriented_bounds` is an **oriented** bounding box: it carries a rotation as well
as a centre.

## 2. NVIDIA Isaac ROS FoundationPose — comparison target, not the reference

* `https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation`
* `main` @ **`ab389aea93376402cfd9bb093dd3dd38d12574b6`** (release 4.5, Jul 2026)
* A C++/TensorRT reimplementation consuming ONNX exports of the same networks.

### The reported mesh-centre sign discrepancy — checked in source

The concern is that the C++ decoder's mesh-centre translation disagrees in sign
with the NVLabs Python implementation. Read directly, at this revision:

`isaac_ros_foundationpose/src/foundationpose_impl/mesh_loader.cpp`
```cpp
mesh_data_->mesh_model_center = (max_v + min_v) / 2.0f;          // AXIS-ALIGNED centre
vertices.push_back(mesh->mVertices[v].x - mesh_data_->mesh_model_center[0]);
```
The vertices handed to the network are **pre-centred**: `v' = v − c`.

`isaac_ros_foundationpose/src/foundationpose_impl/pose_decoder.cpp`
```cpp
tf_to_center.block<3, 1>(0, 3) = -mesh_data->mesh_model_center;   // T(−c)
Eigen::Matrix4f corrected = pose_matrix * tf_to_center;
```

**The sign is self-consistent at this revision.** If `P` is the pose of the
*centred* model, a point of the original mesh maps as
`X_cam = P·(v − c) = P·v − P·c`, so the pose of the **original mesh frame** is
`Q = P · T(−c)` — exactly what the code computes. A `+c` here would be the bug;
`−c` is right, *given the pre-centring two files away*. The two are only correct
as a pair, which is presumably how a sign error crept in historically.

### What DOES still differ, and it is not a sign

| | NVLabs Python | Isaac ROS 4.5 |
|---|---|---|
| Centre used | `trimesh.bounds.oriented_bounds` — **oriented** box (centre **and** rotation) | `(max_v + min_v)/2` — **axis-aligned** centre, no rotation |
| Reported pose refers to | the **mesh as loaded** (`ob_in_cam`); the bbox-centre pose is computed separately for drawing | the **original mesh frame**, recovered from the pre-centred model |

So for a mesh whose bounding-box centre is **not** at the origin, the two report
poses of *different frames*, and a naive comparison of translations will differ
by roughly the centre offset — without either being wrong. Any comparison must
first state which frame it is comparing.

### Consequences adopted here

* WISEPACK's provider result is tied to **NVLabs Python**, the implementation
  chosen as the numeric reference.
* WISEPACK applies **no compensation** for either convention. If a discrepancy
  is found it is reported, not silently corrected — a correction baked into a
  client is invisible the day upstream fixes the cause.
* The comparison the task describes (both implementations on a mesh whose bbox
  centre is off-origin, translations compared explicitly) remains the right
  experiment and is **not yet run** — see the status note below.

## 3. Runtime pin for this host

| | |
|---|---|
| GPU | RTX 4090 (Ada, SM 8.9), 24 GB |
| Driver | 595.84, CUDA 13.2 capable |
| Container runtime | Docker with the `nvidia` runtime registered |

The upstream `docker/dockerfile` targets `nvidia/cudagl:11.3.0-devel-ubuntu20.04`
with conda/python 3.8. That is **not** a good pin for this machine: CUDA 11.3
predates Ada (SM 8.9) and would compile the CUDA extensions without an
architecture this GPU can use.

WISEPACK therefore pins a **modern, tested combination** rather than either the
tutorial's versions or the newest available:

| Component | Pin | Why |
|---|---|---|
| Base image | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` | CUDA 12.4 supports SM 8.9 natively; 12.4 is the newest CUDA with broad wheel coverage for torch/pytorch3d |
| `TORCH_CUDA_ARCH_LIST` | `8.9` | build only for this GPU: faster builds, no silent PTX-JIT fallback |
| PyTorch | `2.4.1+cu124` | the version the FoundationPose ecosystem (pytorch3d, nvdiffrast) is most widely tested against on CUDA 12.4 |
| pytorch3d | built from source, no build isolation | no wheel exists for this combination |
| nvdiffrast | `NVlabs/nvdiffrast` pinned by commit | required by the estimator's rasteriser |
| FoundationPose | `NVlabs/FoundationPose` @ `a1b694b8` | the revision inspected above |

The upstream `c++14` → `c++17` patch in `bundlesdf/mycuda/setup.py` is still
needed with modern CUDA and is applied at build time.

## 4. Status

* Weights: **not yet obtained.** The official Google Drive folder enumerates and
  serves the two small `config.yml` files, but both `model_best.pth` return
  *"Too many users have viewed or downloaded this file recently… up to 24
  hours"*. Per the project decision, WISEPACK **waits for the official source**
  rather than substituting an unofficial mirror. The worker reports this
  precisely rather than starting without weights.
* Therefore the bolt regression (§4 of the plan) and the two-implementation
  comparison (§2 above) are **pending**, not failed, and nothing in WISEPACK
  claims either has passed.

## 5. Known issue: exit 139 on model-free representation build

**What happens.** `scripts/model_free_build.sh` runs the official
`bundlesdf/run_nerf.py::run_one_ob` to build the Neural Object Field. The job
completes — mesh, UV unwrap, baked texture, optimised poses and `model.obj` are
all written, and the completion marker is printed — and *then* the container
exits **139** (`SIGSEGV`, `free(): invalid pointer`) during interpreter
shutdown. It is a teardown crash in the OSMesa/OpenGL stack in a process that
has already finished its work.

**Scope.** REPRESENTATION BUILD ONLY. It has never been observed in the pose
estimation path: the simulated benchmark and the physical D435 run both exit
cleanly, and neither loads pyrender or an OSMesa context. It costs nothing at
run time because the representation is built once and cached.

**Current handling, and its status.** `model_free_build.sh` judges success by
*artefacts plus completion marker* rather than by exit code, and reports the
crash rather than swallowing it. This is explicitly **TEMPORARY**. It is a
workaround for a diagnosed upstream teardown crash and **must not become a
dashboard success contract** — nothing user-facing may ever conclude "it
worked" from the presence of a file. Any integration of model-free into the
dashboard has to resolve this first, not inherit it.

**Not yet diagnosed.** Deferred deliberately so it would not derail the
physical experiment. The open questions, in order:

1. Does `PyOpenGL-accelerate` participate? It is a separate wheel with its own
   C extension and is the usual suspect for a `free()` at exit.
2. Does a minimal pyrender + OSMesa script — a context, one offscreen render,
   exit — reproduce it outside FoundationPose entirely?
3. Does explicitly deleting the `OffscreenRenderer` and its context before
   interpreter shutdown prevent it?

**Constraints on the fix.** Do not patch FoundationPose to suppress the signal,
and do not change any pinned dependency version to make it go away without
reporting first: the pinned set is what the working model-free path was
established on.
