# Reference-asset inventory

Everything here was established by **inspecting the files**, not inferred from
prose. Each claim names the file or code line it came from. Where something is
**not determinable** from the assets, it says so rather than guessing.

Nothing in this document is a WISEPACK runtime dependency: these are reference
assets under `references/`, outside the repository, and the integration copies
what it needs into a WISEPACK-owned dataset directory.

---

## 1. `references/Tubes-Capture/` — real RGB-D capture

**84 files: 42 RGB + 42 depth, perfectly paired.** Indices `005`–`047` with
`030` absent; no metadata, mask, intrinsics or pose file of any kind.

| Property | Value | How it was established |
|---|---|---|
| RGB | `1920 × 1080`, PNG **8-bit RGB** (IHDR colour type 2) | PNG IHDR bytes |
| Depth | `1920 × 1080`, PNG **16-bit greyscale** (IHDR bit depth 16, colour type 0); `cv2.IMREAD_UNCHANGED` → `uint16` | PNG IHDR + OpenCV |
| Correspondence | `rgb_NNN.png` ↔ `depth_NNN.png`, same index = same capture | `capture_from_realsense.py:61-62` writes both from one keypress |
| **Depth units** | **millimetres**, scale `0.001` m per unit | `utils.py:33` `depth = depth_frame[v,u] / 1000.0` before `rs2_deproject_pixel_to_point`; `capture_from_realsense.py:69` `depth_scale=1000.0` |
| **Invalid depth** | **`0`** | RealSense Z16 convention; `utils.py:34` skips `depth == 0` |
| Invalid fraction | 0.0 % – 4.2 % across the 42 frames | measured |
| Depth range | 430 – 1642 mm over the set; frame 031: 441 – 533 mm | measured |
| **Aligned to colour** | **Yes.** Depth is captured at 1280×720 and resampled into the 1920×1080 colour frame | `realsense_utils.py` `rs.align(rs.stream.color)`; `config.yaml` `alignment.align_depth_to_color: true`; both PNGs are 1920×1080 |
| Camera | **Intel RealSense D435**, firmware 5.16.0.1, D400 line | `open3d-pose-estimation/intermediate_results/settings_camera.json` |
| Post-processing | decimation ×2, spatial, temporal filters enabled | `config.yaml → camera.post_processing` |
| **Intrinsics** | **NOT RECORDED** | `capture_from_realsense.py:27` reads `color_intr` from the live device and never writes it out |
| **Ground-truth poses** | **NONE** | no pose/annotation file exists anywhere alongside these frames |
| **Masks** | **NONE for these frames** | only unrelated `filtered_mask_{0,1}.png` in `intermediate_results/` |

Provenance: `rgb_031.png` and `depth_031.png` are **byte-identical** (md5) to
`/data/corob/experiments/AMROPick/open3d-pose-estimation/captured_dataset/`.
`Tubes-Capture` is that dataset plus earlier indices `005`–`029`, with the
`depth_vis_*.png` previews removed.

**Scene content** (visual inspection of frames 005 and 031): a bin-picking heap
of **straight round metal tubes with open ends**, on a flat cardboard/table
surface. At least two distinct lengths are present, and possibly two diameters.
The table plane sits at ~525 mm from the camera and the pile rises ~30–90 mm
above it (measured from the depth histogram). The whole pile spans roughly
485 × 283 mm (single connected depth component, at an assumed fx ≈ 1386).

### What this dataset cannot support on its own

* **No intrinsics** → FoundationPose cannot run on it without K being supplied
  from elsewhere or recovered. The tutorial's `cam_K.txt` **does not apply**: it
  is a 640×480 Isaac Sim virtual camera (see §3).
* **No masks** → FoundationPose's `register()` has no object region.
* **No ground truth** → any result is *repeatability*, never accuracy.

## 2. `references/CAD-Models/STL-Files/` — 12 binary STL meshes

**Units are millimetres.** Established from code, not from the STL format:
`open3d-pose-estimation/convert_stl_to_ply.py:22-24` —
`scale_to_meters=True` → `mesh.scale(0.001, ...)` under the log line
*"Scaling mesh from millimeters to meters"*. Every dimension below is mm.

| File | Bounding box (mm) | Section | Geometry | Symmetry |
|---|---|---|---|---|
| `Cylinder1.stl` | 20 × 20 × 40 | 113.0 mm² | round tube **OD 20, ID 16, wall 2**, L 40 | **axial** (continuous about long axis) **+ end-for-end flip** |
| `Cylinder2.stl` | 35 × 70 × 35 | 301.3 mm² | round tube **OD 35, ID 29, wall 3**, L 70 | axial + flip |
| `Cylinder3.stl` | 35 × 190 × 35 | 301.5 mm² | round tube **OD 35, ID 29, wall 3**, L 190 | axial + flip |
| `Cylinder4.stl` | 315.5 × 25 × 25 | 202.5 mm² | round tube **OD 25, ID ≈19.2, wall ≈2.9**, L 315.5 | axial + flip |
| `Cylinder5.stl` | 315.5 × 148.0 × 25 | — | **BENT tube**, OD 25, curved in one plane (section centre traverses 5.9 → −114.8 mm while the other transverse coordinate stays 0) | **none continuous**; one mirror plane |
| `SquareTube1.stl` | 42.426 × 21 × 42.426 | 162.5 mm² | **square tube, 30 mm side** (42.426 = 30·√2 — the mesh is modelled rotated 45° about its length), L 21 | **discrete 4-fold** about the length + flip |
| `SquareTube2.stl` | 21 × 21 × 70 | 216.0 mm² | square tube **21 outer, 15 inner, wall 3**, L 70 | discrete 4-fold + flip |
| `Plate1.stl` | 300 × 50 × 2 | 97.0 mm² | thin profiled plate | 2-fold at most |
| `Plate2.stl` | 60 × 40 × 8 | 320.0 mm² | plain block (12 triangles = a box) | 3 mirror planes, 2-fold axes |
| `Plate3.stl` | 300 × 100 × 2 | 148.5 mm² | thin profiled plate | 2-fold at most |
| `Plate4.stl` | 300 × 84.95 × 2 | 145.6 mm² | thin profiled plate | 2-fold at most |
| `Plate5.stl` | 300 × 85 × 2 | 121.4 mm² | thin profiled plate | 2-fold at most |

All twelve are **watertight**, so volume — and therefore the wall thickness
derived from it — is meaningful.

`Plate1`–`Plate5` are byte-identical in size to
`open3d-pose-estimation/object_models/*.stl` and are the classes of the
`plates-segmentation` Roboflow dataset (`nc: 5`, names `Plate 1`…`Plate 5`).

### CAD ↔ captured-object mapping: **NOT ESTABLISHED**

The task asked which STL corresponds to the captured objects. It is **not
determinable from these assets**, and here is exactly why:

* the only segmentation datasets present are for **Plates**
  (`Tubes-Segmentation-2/data.yaml` → `names: ['Plate 1' … 'Plate 5']`, and a
  second with `['part 40' … 'part 44']`). **No tube class list exists.**
* `config.yaml → model_mapping` maps only `Plate 1`…`Plate 5` → `Plate*.ply`.
  **No cylinder or square-tube entry.**
* the captured frames contain **no masks and no annotations**, and the tubes in
  the pile touch, so they form a single connected depth blob — individual
  instances cannot be measured without segmentation.
* without recorded intrinsics, any dimension recovered from depth is
  proportional to an assumed `fx`.

What *can* be said: the captured objects are straight round tubes of at least
two lengths. A scanline measurement of isolated tube crossings at the measured
median depth (481 mm) gives an outer diameter of ~24–26 mm **if** `fx ≈ 1386`
(the D435 nominal at 1920×1080) — which would point at `Cylinder4`'s OD 25 —
but `Cylinder4` is 315 mm long, and no tube that long is present in a pile
spanning 485 mm overall. The measurement and the CAD lengths therefore do not
agree on a single part, and the honest conclusion is that **the captured tubes
are not represented by any of these twelve meshes**, or are represented by one
whose length differs from the captured stock.

Resolving this needs one of: the intrinsics for these frames, a labelled mask,
or a caliper measurement of the physical parts.

## 3. `references/Robot-Mania-Bin-Picking-Tutorial/` — the tutorial

`isaac_bin_picking/` (also unpacked at `references/isaac_bin_picking/`, plus the
original 84 MB zip). Fully inspected; see `IMPLEMENTATION_NOTE.md` §1a for the
pipeline reading. The asset facts:

| Item | Value |
|---|---|
| `FoundationPose_related/bolt/cam_K.txt` | `fx = fy = 554.2563`, `cx = 320`, `cy = 240` → **640 × 480**, a virtual Isaac Sim pinhole |
| `bolt/rgb/` | 174 PNG |
| `bolt/depth/` | 174 PNG |
| `bolt/masks/` | **1** PNG — the first-frame registration mask FoundationPose needs |
| `bolt/mesh/` | `bolt.obj` + `bolt.mtl` + `bolt_isaac.png` texture |
| Segmentation | **YOLOv11-seg** (`ultralytics`), custom `yolo/best.pt` (45 MB), `conf=0.9`, instance masks reshaped to `(480, 640, 1)` and scaled ×255 |
| Extrinsic | a single static ROS TF lookup `World` ← `Camera`, then `T_WO = T_wc @ pose` |
| Symmetry | **not handled** — hand-tuned quadrant offsets on a projected axis angle |
| Ground truth | none published; the tutorial compares nothing |

**This is a complete, self-consistent FoundationPose demo dataset** — intrinsics,
RGB, depth, a registration mask and a textured mesh — and it is synthetic, so
depth and colour are aligned by construction.

## 4. Consequence for the first regression dataset

Two candidate deterministic datasets exist, and they are not equivalent:

| | tutorial `bolt` | `Tubes-Capture` |
|---|---|---|
| RGB + depth | yes | yes |
| Intrinsics | **yes** (`cam_K.txt`) | **no** |
| Registration mask | **yes** (1 frame) | **no** |
| CAD mesh | **yes** (`bolt.obj`, textured) | meshes exist but **no established correspondence** |
| Aligned depth | by construction (synthetic) | yes (`rs.align`) |
| Ground-truth pose | no | no |
| Realism | synthetic | **real sensor, real noise, real clutter** |

So the **tutorial `bolt` dataset is the first deterministic regression**: it is
the only one of the two that is complete enough to run FoundationPose at all.
`Tubes-Capture` becomes the second stage and needs three things added first —
intrinsics, a mask, and a CAD correspondence — none of which may be invented.

Neither dataset has ground truth, so **both can measure repeatability and
plausibility only; neither can measure absolute pose accuracy.**

---

## Measured symmetry of the CAD models

The target domain is these pipe sections, not the bolt. Their symmetry decides
which rotational degrees of freedom WISEPACK is entitled to report as measured,
so it was **measured rather than declared by eye**:

    python3 scripts/measure_mesh_symmetry.py references/CAD-Models/STL-Files/Cylinder*.stl

Each mesh is sampled densely, rotated about its bounding-box centre, and the
residual distance back to its own surface is reported. A residual at the
sampling-noise floor means that rotation is **unobservable**.

| Mesh | Extents (mm) | Continuous | 180° | Declared symmetry |
|---|---|---|---|---|
| `Cylinder1.stl` | 20 × 20 × 40 | **z: 0.24** (noise 0.24) | x,y,z: 0.24 | `axial`, axis **z** |
| `Cylinder2.stl` | 35 × 70 × 35 | **y: 0.43** (noise 0.42) | x,y,z: 0.43 | `axial`, axis **y** |
| `Cylinder3.stl` | 35 × 190 × 35 | **y: 0.69** (noise 0.69) | x,y,z: 0.69 | `axial`, axis **y** |
| `Cylinder4.stl` | 315.5 × 25 × 25 | **x: 1.20** (noise 0.72) | x,y,z: 0.72 | `axial`, axis **x** |
| `Cylinder5.stl` | 315.5 × 148 × 25 | none (104–156 mm) | **z: 0.75** (noise 0.75) | `discrete`, fold **2**, axis **z** |

### The Cylinder5 result is the one worth reading twice

Cylinder5 is the bent section, and it is the right early 6-DoF test for exactly
the reason expected: rotating it by an arbitrary angle about any axis moves its
surface by more than 100 mm, so its orientation is genuinely constrained in a
way no straight tube's ever is.

**But it is not uniquely constrained.** A 180° rotation about z maps it onto
itself to within 0.75 mm — the sampling noise floor. It is a symmetric hairpin,
so exchanging its two legs is unobservable. Its pose is determined *up to that
flip and no further*. Declaring it `type: none` would have been the obvious
choice from looking at it, and would have caused WISEPACK to report a leg-swap
as a resolved measurement.

This is recorded as `type: discrete, fold: 2, axis: z` in
`config/perception_objects.yaml`.

### Limitation of the measurement

Only the three coordinate axes are tested. A mesh whose symmetry axis is not
aligned with x, y or z is reported as having no symmetry even when it has one —
which is why `SquareTube1.stl` (a square section modelled at 45°) shows nothing.
The tool never claims a mesh is asymmetric, only that the rotations it tried are
not symmetries of it. The five cylinders are all coordinate-aligned, so their
results stand.

## Validation order

1. **Tutorial bolt** — proves the FoundationPose runtime and the WISEPACK
   provider are correct. Nothing more; WISEPACK does not package bolts.
2. **One known WISEPACK tube, real RGB-D, CAD-matched** — verifies segmentation,
   depth, mesh scale and 6-DoF pose against a part whose identity is known
   because it was chosen, not inferred.
3. **Cylinder5 first**, because its orientation is constrained (up to the
   measured 2-fold flip above).
4. **Cylinder3 / Cylinder4 next** — position and pipe-axis direction only. Axial
   spin is marked symmetry-equivalent and is never presented as measured.
5. **Cluttered pipe scene** — instance segmentation → CAD/`model_id` association
   → FoundationPose per instance → `ObservationBatch` → planning → Digital Twin.

**No CAD-to-captured-object correspondence is claimed for `Tubes-Capture`**, and
none may be inferred from appearance: `Cylinder2` and `Cylinder3` share a 35 mm
outer diameter and differ only in length, which a single uncalibrated view does
not resolve. That correspondence needs either the physical parts in hand or a
new controlled capture in which `model_id` is known *at capture time* — along
with recorded intrinsics, aligned RGB/depth, masks, timestamps and extrinsic
calibration. That controlled set, once it exists, becomes the authoritative pipe
regression set and supersedes `Tubes-Capture` for this purpose.
