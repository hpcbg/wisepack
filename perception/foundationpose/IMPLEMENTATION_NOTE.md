# FoundationPose RGB-D perception — implementation note

Written **before** the code, as the task requires: what was inspected, what is
being adapted into WISEPACK, what stays third-party, and what this host can and
cannot validate.

---

## 1. What was inspected

### 1a. The tutorial (reference for the *pipeline*)

`references/isaac_bin_picking/` — the bin-picking tutorial that accompanies the
RGB-D → segmentation → FoundationPose → MoveIt/Panda demonstration.

| File | What it actually does |
|---|---|
| `FoundationPose_related/run_ros.py` | The whole pipeline in one ROS 2 node. RGB from `/rgb`, depth from `/depth` (`passthrough`), **intrinsics from a static `cam_K.txt`** (not from `camera_info`), YOLOv11-seg for the mask, `est.register()` on the first frame and `est.track_one()` afterwards, camera extrinsic from a **one-shot TF lookup** `World`←`Camera`, then `T_WO = T_wc @ pose` and a `PoseStamped` in `World`. |
| `FoundationPose_related/run_ros_test.py` | The same without FoundationPose — a harness for the ROS plumbing. |
| `FoundationPose_related/bolt/` | A complete FoundationPose demo dataset: `cam_K.txt`, `rgb/` (174 PNG), `depth/` (174 PNG), `masks/` (**one** PNG — the first-frame registration mask), `mesh/bolt.obj` + `.mtl` + texture. Rendered in Isaac Sim: `fx = fy = 554.2563`, `cx = 320`, `cy = 240` — the canonical 640×480 pinhole. |
| `yolo/` | `best.pt` (45 MB, YOLOv11-seg trained on bolts), the training/inference scripts, and the labelled dataset. |
| `packages_installation_and_other_commands.txt` | The tutorial's own environment: **a virtualenv, not Docker** — CUDA 12.1, torch 2.4.0+cu121, pytorch3d, nvdiffrast, CUB 1.10, `FoundationPose/build_all.sh` with `c++14`→`c++17` patched into `bundlesdf/mycuda/setup.py`. |

Answering the specific questions §3 asks:

* **RGB / depth acquisition** — two ROS topics from Isaac Sim; no camera SDK.
* **Alignment** — none performed. RGB and depth are rendered from one virtual
  camera, so they are aligned by construction. *A real sensor is not.*
* **Intrinsics** — a static text file. Fine for one fixed simulated camera;
  wrong for anything that can change resolution or lens.
* **Segmentation** — YOLOv11 **instance** segmentation at `conf=0.9`,
  `r.masks` → `(480, 640, 1)` → `×255`. A per-pixel mask, **not** a box.
* **CAD mesh** — `trimesh.load()`; `trimesh.bounds.oriented_bounds()` gives
  `to_origin` and `extents` used only for drawing the 3-D box.
* **Initialisation / registration / tracking** — `ScorePredictor`,
  `PoseRefinePredictor`, `dr.RasterizeCudaContext`, then `register()` once and
  `track_one()` thereafter; re-registration on a rising `/trigger` edge.
* **Pose output** — a 4×4 in the **camera** frame.
* **Coordinate transform** — a single static TF lookup, then one matrix multiply.
* **Visualisation** — `draw_posed_3d_box` + `draw_xyz_axis`.
* **Symmetry** — **not handled.** The node projects the object's Z axis onto the
  world plane and applies hand-tuned quadrant offsets
  (`bolt_angle + 9π/4`, `+ π/4`, `− 7π/4`) to get a gripper angle. That is
  application-specific compensation for one bolt in one cell, and it is exactly
  what WISEPACK must **not** copy — see §5 below.
* **Confidence / scoring** — the FoundationPose score is computed internally and
  **never published**. The tutorial reports no per-pose confidence at all.
* **Object-specific configuration** — hard-coded paths
  (`~/isaac_bin_picking/yolo/best.pt`, `demo_data/bolt`), a single class, and a
  single tuned angle correction.

### 1b. Official FoundationPose (authority for the *runtime*)

`https://github.com/NVlabs/FoundationPose`, `main` @
**`a1b694b83e633c2cb6115b9063d940a687759392`** (inspected in a scratch clone;
nothing was copied into WISEPACK).

* Licence: **NVIDIA Source Code License — non-commercial research use only**
  (`LICENSE` in that repository). This governs how WISEPACK may ship it: see §4.
* `run_demo.py` defines the reference input contract WISEPACK's worker mirrors:
  a directory with `cam_K.txt`, `rgb/`, `depth/`, `masks/`, and a mesh.
* Runtime: `ScorePredictor` + `PoseRefinePredictor` + `nvdiffrast` +
  `pytorch3d` + custom CUDA extensions built by `build_all.sh`.
* **Network weights are distributed only through Google Drive** — refiner
  `2023-10-28-18-33-37`, scorer `2024-01-11-20-02-45`. There is no versioned
  public URL, so they cannot be baked into an image reproducibly and must be
  fetched into a host-side cache and mounted.
* `docker/dockerfile` builds on `nvidia/cudagl:11.3.0-devel-ubuntu20.04` with
  conda/python 3.8; a prebuilt `wenbowen123/foundationpose` image also exists.

### 1c. This host

| | |
|---|---|
| GPU | NVIDIA **RTX 4090**, 24 GB |
| Driver / CUDA | **595.84**, CUDA 13.2 |
| Docker | present, **`nvidia` runtime registered** |
| Existing FoundationPose material | `/data/workspaces/isaac_ros-dev/isaac_ros_assets/isaac_ros_foundationpose/` — the **Isaac ROS** packaging's *assets only* (Mustard, Mac_and_cheese, dock, soup_can meshes, `quickstart.bag`, interface specs) and the `isaac_ros_dev-x86_64:latest` image (37 GB, 13 months old). **No NVlabs FoundationPose checkout, and no `foundationpose` image.** |
| **RGB-D camera** | **NONE ATTACHED.** `lsusb` shows one `A4Tech FHD 1080P PC Camera` (the RGB webcam the planar provider already uses) plus hubs and an LED controller. `/dev/video0` and `/dev/video1` are that one UVC device's two nodes. `librealsense2 2.56.5` **is** installed on the host, so a RealSense was evidently used here at some point — but no depth device is present now. |

**This is the single most consequential finding.** Stages B and C of the task
(§25 real RGB-D camera, §26 cylindrical object) **cannot be performed on this
host**, and nothing in this work will claim otherwise.

---

## 2. What is adapted into WISEPACK

Concepts and sequence, re-implemented behind WISEPACK's own interfaces:

* the pipeline shape RGB-D → mask → `register()` → 6-DoF pose → transform;
* the demo-directory input contract (`cam_K.txt` / `rgb` / `depth` / `masks` /
  mesh), because it is what the reference dataset already is;
* instance segmentation as the mask source, kept **behind its own seam** so a
  different segmenter is a configuration change;
* `register()`-then-`track_one()` with explicit re-registration.

Deliberately **not** adapted:

* the hand-tuned bolt-angle quadrant correction — replaced by explicit
  **symmetry metadata** and honest reporting of which rotational DoFs are
  unobservable;
* static intrinsics from a text file — replaced by intrinsics carried on the
  frame and validated;
* the one-shot TF lookup — replaced by a stored, provenance-carrying
  **extrinsic calibration** with its own revision;
* ROS in the worker — WISEPACK's worker is **HTTP only**, exactly like the
  planar service, so no second middleware appears on the host (§31).

## 3. What stays third-party

FoundationPose itself, its CUDA extensions, its weights, and the segmentation
model. **No third-party source is copied into this repository.** The image is
built from a tracked `Dockerfile` that clones a **pinned revision**, and the
weights are fetched into a git-ignored host cache and mounted read-only.

## 4. Licence consequence, stated plainly

FoundationPose is released under the **NVIDIA Source Code License for
non-commercial research use**. WISEPACK is MIT. Therefore:

* the FoundationPose worker is an **optional, separately-licensed component**;
* WISEPACK ships **no** FoundationPose code — only a build recipe and a client;
* nothing in the default WISEPACK install pulls it;
* NOTICE records the licence and the pinned revision.

## 5. Where symmetry actually bites

WISEPACK's objects are cylinders. A cylinder has a continuous rotational
symmetry about its axis, so **the rotation about that axis is not observable**
— any value fits the depth data equally well. FoundationPose will still return
*a* full orientation. Reporting it as a measurement would be inventing data.

So the object registry carries symmetry metadata, the observation records which
DoFs are ambiguous, the raw pose is kept for diagnostics, and the canonicalised
pose is what planning sees.

## 6. Honest scope for this session

| Stage | Status |
|---|---|
| WISEPACK-side architecture (method selection, RGB-D frame, 6-DoF observation, object registry, extrinsics, symmetry, provider, container lifecycle, dashboard, tests, docs) | the deliverable |
| Worker image + worker application | tracked and buildable |
| §24 Stage A — reference-dataset regression | attempted; the tutorial's `bolt` dataset is present and complete, so this is the one physical-realism check that *is* possible here |
| §25 Stage B — real RGB-D camera | **impossible on this host: no depth camera** |
| §26 Stage C — cylindrical object | **impossible for the same reason** |

Nothing will be reported as validated that was not actually run.

---

# Runtime status — built, weighted, and regression-tested

## The image

`wisepack-foundationpose:pinned`, built by `scripts/setup_foundationpose.sh`
from `perception/foundationpose/Dockerfile`. Verified inside the container:

| Check | Result |
|---|---|
| FoundationPose revision | `a1b694b83e633c2cb6115b9063d940a687759392` — **matches the pin** |
| torch | `2.4.1+cu124`, torch CUDA `12.4` |
| GPU | **NVIDIA GeForce RTX 4090**, compute capability **8.9**, 25.2 GB |
| `torch.cuda.is_available()` | **True** |
| pytorch3d / nvdiffrast | `0.7.8` / `0.3.3`, `RasterizeCudaContext()` constructs |
| Native extensions | `mycpp` and `bundlesdf.mycuda.common` both import |
| Upstream tree | clean except `bundlesdf/mycuda/setup.py` (`c++14`→`c++17`) |

The only modification to the upstream checkout is that one `sed`. Nothing else
in the FoundationPose tree is patched, and in particular nothing that would let
a failure pass silently.

### `mycpp` is required, though upstream treats it as optional

`Utils.py` wraps `import mycpp` in `try/except` and leaves it `None`. So a
container without it imports `estimater` perfectly happily — and then dies
inside `register()` at `estimater.py:120`, where `mycpp.cluster_poses` prunes the
rotation grid. The first build produced exactly that container.

Two consequences, both implemented: the Dockerfile now builds `mycpp` (a
CMake/pybind11 module, not a pip package), and `capability.py` probes both native
extensions **by name**. A capability probe that reports "runtime available" and
then crashes mid-estimate is worse than no probe, because it moves the diagnosis
from start-up into the middle of a run.

## GPU access without the NVIDIA Container Toolkit

The toolkit is **not installed on this host** and installing it needs root.
`/etc/docker/daemon.json` nonetheless declares an `nvidia` runtime pointing at a
binary that does not exist, so `docker info` lists it and `--gpus all` fails with
a CDI error that names neither problem.

The first version of `setup_foundationpose.sh` trusted that listing, passed
`--gpus all`, and the worker **could not start at all** — the precise opposite of
the rule the worker is built around. It now *probes* GPU access by running it.

When the toolkit is absent it falls back to doing the toolkit's two jobs
directly: expose `/dev/nvidia*`, and bind-mount the host's user-space driver
libraries read-only at their sonames, version-matched to the running driver
(595.84). This is **a fallback, not a recommendation** — the toolkit is the
supported mechanism and handles cases this does not — and it is *verified* by
running `torch.cuda.is_available()` against it rather than assumed to work. When
neither route works the worker still starts and still reports
`gpu_available: false` with the reason.

## Weights

Obtained from the **official** Google Drive folder named in the upstream README
at the pinned revision, once the quota cleared. No mirror was used.

| Role | Path | Bytes | SHA-256 |
|---|---|---|---|
| refiner | `2023-10-28-18-33-37/model_best.pth` | 68,220,109 | `774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60` |
| scorer | `2024-01-11-20-02-45/model_best.pth` | 190,229,389 | `81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26` |

Upstream publishes no reference checksums, so these are **observed, not
verified**: they prove a later run used the same bytes; they do not prove those
bytes are upstream's. `provenance.json` says so in as many words.

Both were loaded into upstream's own network classes with
`load_state_dict(strict=True)` — `RefineNet` (16,830,406 parameters) and
`ScoreNetMultiPair` (15,773,121). Strict, because a checkpoint that loads with
missing keys is a checkpoint that runs with random weights.

They are **not in the image**. The path the predictors hard-code,
`<code_dir>/weights`, is a symlink to the read-only `/weights` mount; the
checkpoints stay on the host under the NVIDIA licence.

## Bolt regression

`references/Robot-Mania-Bin-Picking-Tutorial/isaac_bin_picking/FoundationPose_related/bolt`
— 174 RGB + 174 depth frames, one registration mask, `cam_K.txt`, textured
`bolt.obj`. Read **in place** from the mounted reference tree; nothing copied.

Inputs, established from the data rather than assumed:

* **Mesh units: metres.** Extents `[0.0231, 0.0490, 0.0200]` — a 49 mm bolt.
  `mesh_scale_to_metres: 1.0`.
* **Depth units: millimetres**, `uint16`, no invalid pixels in frame 0, median
  800 mm. `depth_scale_mm: 1.0`.
* **Intrinsics:** fx = fy = 554.256, cx/cy = 320/240 at 640×480 — a synthetic
  Isaac Sim camera, principal point exactly centred.

`depth_scale_mm` is now **required by the API, with no default**. A `uint16`
millimetre image and a `float32` metre image are both ordinary and are
indistinguishable from the pixels; defaulting either way is a factor-of-1000
error that produces a confident, completely wrong pose.

### Result

    position (camera optical frame)  [-3.43, -28.28, 780.00] mm
    quaternion (x, y, z, w)          [-0.1227, 0.1719, -0.6259, 0.7507]
    duration                         ~11.9 s (register, 5 refine iterations)

**Repeatability: exact.** Five `register()` calls on the same frame and mask gave
position range `[0.0, 0.0, 0.0]` mm and 0.0000° rotation difference. Note what
this does and does not show: the pipeline is *deterministic*. It is not evidence
of stability under sensor noise or viewpoint change, because neither varied.

**Plausibility.** Deprojecting the mask centroid (318.9, 220.1) at the median
masked depth (777 mm) through `cam_K` gives a visible-surface point of
`[-1.6, -27.9, 777.0]` mm. The estimated mesh origin sits 3.6 mm from it and
3 mm *further* from the camera — the correct sign, since the mesh origin lies
inside the object and the deprojected point is on its surface. For a part whose
half-diagonal is 28.9 mm, that is consistent.

**Accuracy is NOT measured.** This dataset has no ground-truth pose. Repeatability
and plausibility are the only quantities available, and neither is accuracy.

The pose overlay (`GET /image/overlay`) draws the projected mesh bounding box,
the axis triad scaled to the object rather than a fixed 5 cm, and the **mask
outline** — because in a bin of four near-identical bolts a pose on the wrong one
looks exactly as convincing as a pose on the right one until the mask is drawn
beside it.

## What this does not yet establish

* No real RGB-D camera is attached, so nothing here is live-sensor validated.
* The regression object is a synthetic bolt. **The target domain is the pipe and
  tube sections**; see `REFERENCE_ASSETS.md` for their measured symmetry and the
  validation order.
* The WISEPACK provider, dashboard method selection and `PhysicalObservation`
  mapping are not yet implemented — that work starts from here.

---

# Integration into WISEPACK

## Where the boundary sits

    FoundationPose worker (container)
            |  HTTP, JSON
            v
    wisepack_core/foundationpose_client.py     transport only
            |
            v
    perception/providers/foundationpose_rgbd.py   THE ONLY schema-aware code
            |
            v
    PhysicalObservation / ObservationBatch     generic, method-neutral
            |
            v
    workflow -> packing -> Digital Twin -> approval -> DDS -> FIWARE

Nothing above the provider has heard of a mesh, a mask, a rotation grid, a
refiner iteration or the worker's HTTP schema. That is asserted as an **import
graph**, not a word search: `tests/test_foundationpose_integration.py` parses
every core module and fails if one imports the client, the provider, or names a
worker endpoint. A method NAME and a selector label are domain vocabulary the
core is entitled to hold — they name a choice an operator makes — and banning
those strings would only have banned writing the rule down.

## GPU passthrough on this host — scope and limits

**The NVIDIA Container Toolkit is not usable as a normal `docker --gpus all`
runtime here**, despite `/etc/docker/daemon.json` declaring an `nvidia` runtime:
the binary it names does not exist, so `docker info` lists a runtime that cannot
run, and `--gpus all` fails with a CDI error naming neither problem. Installing
the toolkit needs root, which this work does not have and did not use. **No host
configuration was modified.**

The worker therefore falls back to doing the toolkit's two jobs directly:
exposing `/dev/nvidia*`, and bind-mounting the host's user-space driver
libraries **read-only** at their sonames, version-matched to the running driver.

Its limits, stated rather than glossed:

* **It is not universally portable.** It assumes one visible GPU, a Debian-style
  `x86_64-linux-gnu` library path, and a driver whose libraries are named
  `lib*.so.<version>`. It does not handle MIG, multi-GPU selection, device
  enumeration or IPC, all of which the toolkit does.
* **It is verified, not assumed.** The script runs
  `torch.cuda.is_available()` inside the image against the assembled arguments
  before using them; when that fails it starts the worker WITHOUT a GPU and the
  worker reports `gpu_available: false` with the reason.
* **The toolkit is preferred whenever it works.** `--gpus all` is tried first,
  and the fallback is used only when that probe fails.

## Weights — the local integrity record

Fetched from the official Google Drive folder named in the upstream README at
the pinned revision. No mirror was used at any point.

| Role | Filename | Bytes | SHA-256 |
|---|---|---|---|
| refiner | `2023-10-28-18-33-37/model_best.pth` | 68,220,109 | `774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60` |
| scorer | `2024-01-11-20-02-45/model_best.pth` | 190,229,389 | `81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26` |

Each directory also carries upstream's own `config.yml` (708 and 778 bytes).

**These SHA-256 values are LOCAL INTEGRITY RECORDS, not verification.** Upstream
publishes no reference checksums, so they establish that a later run used the
same bytes as this one — and nothing about whether those bytes are the ones
NVIDIA published. `provenance.json` beside them states this in the same terms. If
upstream ever publishes official hashes, these become checkable and the
distinction disappears; until then, claiming "verified" would be claiming a
comparison that has no second side.

The weights stay **outside the image** and are mounted **read-only**. The path
upstream's predictors hard-code, `<code_dir>/weights`, is a symlink to that
mount.

## What is still blocked on hardware

Everything that requires a depth sensor, and only that:

* live RGB-D acquisition, and therefore `inference_ready`;
* the camera handover between the planar provider and the worker — modelled in
  `wisepack_core/camera_ownership.py`, with `handover_tested = False`, because
  it has never been performed;
* the SE(3) camera-to-work-area extrinsic, without which every FoundationPose
  pose stays in the camera frame and `workarea_pose_available` stays False. Note
  that `pose_valid` is TRUE for these: the estimate is sound where it lives, and
  the two validities are deliberately separate fields — see below;
* any CAD-to-physical-object correspondence for the pipe sections, which needs a
  controlled capture where `model_id` is known at capture time.

The runtime, the provider, the domain mapping, the serialisation, the API and
the dashboard are complete and exercised offline against the tutorial dataset.

## Two validities, kept apart

An earlier revision reported a successful camera-frame estimate as
`pose_valid=False` because it could not be transformed into the work area. That
conflated two independent questions and said the measurement was bad, which was
untrue and hid the real gap.

| Question | Field | Offline bolt regression |
|---|---|---|
| Is the estimate structurally and numerically valid in the frame it declares? | `pose_valid` | **true** |
| Where does that pose live? | `frame_id` | `camera_color_optical_frame` |
| Does a validated SE(3) camera→work-area transform exist? | `workarea_transform_valid` | **false** |
| Can the pose be placed in the work area? | `workarea_pose_available` (derived) | **false** |

`workarea_pose_available` is DERIVED — true when the pose is already in the
work-area frame, or when a validated transform into it exists, and never when
the estimate itself failed — so the two ways of being placeable cannot drift
apart. It is the flag a planner or the Isaac scene synchronizer must consult;
`pose_valid` answers a different question.

The missing extrinsic is represented as MISSING and never as identity. An
identity transform standing in for an unmeasured one places objects wherever the
camera happens to be, with total confidence. Nor is `frame_id` relabelled to
`wisepack_workarea`: the frame is where the pose actually is.
