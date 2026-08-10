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

**Corrected.** An earlier revision of this file called Cylinder5 a "bent
hairpin" with a two-fold symmetry. That was wrong, and the error is instructive:
the measurement tool tested only the three COORDINATE axes, and Cylinder5 is
modelled obliquely, so its own axis was never tried. A conclusion was then drawn
that the tool's stated limitation had already invalidated.

The engineering table (`references/Cylinders.png`) is the nominal authority, and
the STL measurements now agree with it:

| Part | Nominal | Measured length / OD | Ends |
|---|---|---|---|
| Cylinder1 | D20 × L40 × T2 | 40.2 / 20.3 mm | square (0.1 mm variation) |
| Cylinder2 | D35 × L70 × T3 | 70.2 / 35.4 mm | square (0.2 mm) |
| Cylinder3 | D35 × L190 × T3 | 190.0 / 35.3 mm | square (0.0 mm) |
| Cylinder4 | D25 × L316 × T3 | 315.3 / 25.1 mm | **saddle, 5.1 mm** |
| Cylinder5 | D25 × L342 × T3 | 341.3 / 25.9 mm | **saddle, 10.5 mm** |

**All five are straight round tubes.** None is bent. The saddle (fishmouth) cuts
on C4 and C5 are the tangential joints the table's notes describe.

### What is observable, measured from the geometry

Rotations are tested about the mesh's OWN principal axis and about a transverse
axis, at the centroid, against the surface-sampling noise floor:

| Part | spin about its axis | A1/A2 end swap | Declared |
|---|---|---|---|
| Cylinder1 | 0.22 (noise 0.21) — **unobservable** | 0.22 — **unobservable** | `axial`, z |
| Cylinder2 | 0.69 (noise 0.38) — **unobservable** | 0.53 — **unobservable** | `axial`, y |
| Cylinder3 | 0.64 (noise 0.62) — **unobservable** | 0.70 — **unobservable** | `axial`, y |
| Cylinder4 | 1.43 (noise 0.64) — **unobservable** | 0.96 — **unobservable** | `axial`, x |
| Cylinder5 | 3.66 (noise 0.67) — **observable** | 0.77 about z — **unobservable** | `discrete`, fold 2, z |

### The axis is a LINE, not an arrow

For a straight tube with identical ends, a 180° rotation about a transverse axis
through the centre exchanges the two ends and leaves the geometry unchanged. So
the physically meaningful quantity is the **axis line**, and a tube pointing +z
is the same tube as one pointing −z.

`symmetry_aware_angle_deg` implements exactly that: for an `axial` symmetry it
compares axis directions with `abs(dot)`, so neither arbitrary spin nor an end
swap is ever counted as pose error.

A caution for anyone extending this: the transverse directions of a round tube
are **degenerate**. PCA returns an arbitrary pair spanning the perpendicular
plane, so "the second principal axis" is not a meaningful end-swap axis — for
Cylinder5 it reads 3.41 mm while the true swap axis (z) reads 0.77 mm.

### Cylinder5 is still the better first 6-DoF test — for the right reason

Not because it is bent; it is not. Because its 10.5 mm saddle cuts are the only
feature among these parts large enough to make spin about the tube axis
observable. For Cylinder1–4, position and axis DIRECTION are the only meaningful
quantities, and spin must be reported as symmetry-equivalent.

### Limitation of the measurement

Only the coordinate axes, the mesh's principal axis and one transverse axis are
tried. A symmetry about some other axis can still be missed. The tool never
claims a mesh is asymmetric — only that the rotations it tried are not
symmetries of it.

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

---

## Cylinder5 repeatability — measured, and what may be asserted

An earlier pair of runs reported 114.4° and 6.33° geometric orientation error
from what looked like identical inputs. The cause is now established, and it is
not estimator stochasticity.

### FoundationPose is deterministic

`estimater.py:163` calls `set_seed(0)` inside `register()`, which seeds
`numpy.random`, Python `random` and torch (`Utils.py:224-228`).

* **10 consecutive runs in one worker process: 1 distinct pose**, bit-exact.
* A **fresh container** reproduces it to `5.8e-4` in the matrix elements —
  **0.023 mm** at the reference point, **0.042°** in rotation. That residue is
  cross-process CUDA kernel selection, not sampling.

### The Isaac RGB render is NOT byte-deterministic

Regenerating the same scene changed the RGB image: **35,568 pixels differ, max
delta 16** (path-traced sampling noise). Depth and the instance mask were
**byte-identical**.

So the two runs were not given identical inputs, and the difference is entirely
attributable to RGB shading noise.

### What that noise moves, and what it does not

| render | position mm | along | transverse | axis-line ° | full rotation ° |
|---|---|---|---|---|---|
| A | 2.24 | 1.52 | 1.64 | 0.718 | 114.437 |
| B | 1.94 | 0.54 | 1.86 | 0.717 | 6.333 |

Between A and B the **full rotation differs by 108.14°** while the **tube axis
differs by 0.006°**. The variation is *purely circumferential spin*.

That is a real and expected property of this part, not a defect. Cylinder5's
saddle ends are the only cue constraining spin; from this viewpoint they are
weak enough that RGB shading noise alone flips the estimate between
near-equally-scored hypotheses roughly 108° apart.

### Regression policy

**Assertable — stable across every source of variation measured:**

* position error at the reference point (2.24 / 1.94 mm)
* tube-axis-line error (0.718° / 0.717°, agreeing to 0.006°)

**NOT assertable:**

* full geometric 6-DoF orientation, and specifically circumferential spin

This is exactly the geometric-versus-task split already in the registry.
Cylinder5's saddle geometry is real and is not erased; it is simply not
constrained well enough *from this view* to regress against. Nothing here is
averaged, and no seed is forced: forcing determinism would hide the fact that
the spin is genuinely under-determined.

A second viewpoint that shows the saddle ends more directly would likely
constrain spin better. That is worth testing before spin is ever asserted, and
the dataset must not be re-rendered merely to improve the number.
