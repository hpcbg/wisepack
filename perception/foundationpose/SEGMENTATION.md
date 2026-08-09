# Masks for FoundationPose

`register()` needs one thing from segmentation: a binary mask of the object. It
must not care how that mask was obtained — which is why segmentation is a
provider abstraction (`worker/segmentation.py`) and `mask_source` travels with
every result.

| `mask_source` | Scene | Status |
|---|---|---|
| `dataset` | the mask supplied with a reference dataset | in use — the tutorial bolt regression |
| `depth_plane_foreground` | ONE known object on a stable surface | **implemented** (Stage 1) |
| `yolo_instance` | cluttered, touching pipe sections | **not implemented** (Stage 2) |

**No silent fallback between them, ever.** An unknown or unimplemented method is
an error. A mask whose provenance nobody can state is exactly what `mask_source`
exists to prevent.

---

## Stage 1 — `depth_plane_foreground`

For the first live validation: one Cylinder5 on a tabletop.

    aligned RGB + depth
      -> deproject valid depth with the camera's OWN intrinsics
      -> fit the dominant work surface (RANSAC + least-squares refit)
      -> drop plane points
      -> keep foreground ABOVE the plane, inside the work volume and ROI
      -> project back into the aligned colour image
      -> select one connected component
      -> close small holes only
      -> binary mask

**Every threshold is a distance from the measured surface.** There is no
absolute depth cut anywhere: `depth < 0.7 m` encodes where the table happened to
be when it was written, and moving the camera would silently select the wrong
thing. The tests verify the same scene segments identically at 600 mm and
1400 mm.

**Deterministic.** Fixed RANSAC iterations and a fixed seed, because a
segmentation that wandered between runs would turn pose repeatability into a
measurement of the segmentation.

**Diagnostics** on every result: `plane_detected`, `plane_residual_mm`,
`plane_inlier_fraction`, `foreground_points`, `mask_pixels`, `components`,
`selected_component`, `mask_valid`, `mask_source`.

### It is named after its mechanism, not its authority

It is **not** a ground-truth mask. It is a measurement derived from RGB-D
geometry, with the assumption — a dominant flat work surface, one object
standing on it — stated in the name.

On the tutorial's bolt bin it refuses, and correctly: *"the foreground broke
into 5 components, which is more disconnected than the single-object assumption
allows."* That is the method telling you it is being used outside its scope,
which is the behaviour Stage 2 exists to replace.

### Mask validation, before FoundationPose sees it

`register()` given a mask of the tabletop returns a pose rather than an error —
a confident answer about the wrong pixels. So masks are rejected, with the
reason, when they are empty, negligible, nearly the whole image, backed by too
little valid depth, broken into too many components, or not standing far enough
off the fitted plane.

Two of those guards cannot fire from this method and that is worth knowing:
a mask covering most of the frame is impossible here, because RANSAC would have
fitted the *larger* surface as the plane; and the mask is derived from valid
depth, so it is depth-backed by construction. They exist for masks that did not
come from a plane fit — a YOLO mask can happily cover a region the sensor never
returned. Both are tested by tightening the limit.

**No bounding-box masks.** A rectangle from a detection box is not a
segmentation. Permitted only as an explicit debug experiment; no such path
exists in the code.

---

## Stage 2 — the cluttered multi-pipe scene

Plane removal alone is insufficient there: several touching pipes form **one
connected foreground component**, and no amount of morphology separates them
without also fusing things that should stay apart.

### What actually exists today — checked, not assumed

The tutorial ships a working Ultralytics pipeline
(`references/Robot-Mania-Bin-Picking-Tutorial/isaac_bin_picking/yolo/`):

| | |
|---|---|
| Model | `YOLO("yolo11m-seg.pt")`, fine-tuned, 200 epochs, imgsz 640 |
| Dataset | **18 train / 6 val images**, hand-annotated polygons |
| Classes | **`0: bolt`** — one class |
| Checkpoint | `best.pt` (45 MB); contains `bolt`, and **no** `cylinder`, `pipe` or `tube` |

The current official Ultralytics family is **YOLO26-seg** (`yolo26n/s/m/l/x-seg`),
pretrained on **COCO**.

Two conclusions follow, and neither is a matter of opinion:

1. **No off-the-shelf model recognises the WISEPACK pipe classes.** COCO has no
   pipe or tube category, and the tutorial's fine-tuned checkpoint is
   bolt-only.
2. **No labelled tube dataset exists** anywhere in the reference material. The
   only annotations present are the 24 bolt images above.

### Therefore: generate training data from the CAD models

Inventing labels for the existing `Tubes-Capture` images is not an option — the
CAD-to-object correspondence there was never established, and hand-labelling
would encode a guess about which pipe is which as ground truth.

The exact CAD models are the asset that makes this tractable, because a
synthetic scene knows its own masks and its own `model_id` perfectly:

    Cylinder1..Cylinder5 (exact CAD, measured symmetry)
      -> Isaac Sim scenes with domain randomisation
         (pose, clutter, lighting, materials, camera placement)
      -> EXACT instance masks + model IDs, free and correct by construction
      -> train / fine-tune instance segmentation
      -> real cluttered pipe scene
      -> one mask per object
      -> FoundationPose per instance
      -> ObservationBatch -> planning -> Digital Twin -> approval

WISEPACK already has the Isaac integration and the exact meshes, so this is
assembly rather than research.

**Not implemented here, deliberately.** This document is the path, not a
promise; building it is its own task and is not needed for the first live
Cylinder5 validation.

### Identity stays separate from pixels

Three questions, three answers, and coupling them is how a wrong CAD model gets
a confident pose:

| Question | Answered by |
|---|---|
| Which pixels belong to the object? | segmentation |
| Which CAD model is this? | model selection — **the operator, for now** |
| What is its 6-DoF pose? | FoundationPose |

For the first controlled test the operator selects `model_id = cylinder5`
explicitly. The segmentation stage is **not** asked to infer CAD identity.

Later, YOLO classes *may* supply identity if trained per-part — but
FoundationPose is not coupled to that assumption, and the `model_id` remains an
input to the estimate either way.

---

## The reference bolt regression is unchanged

It keeps its **supplied** mask (`mask_source: dataset`). It exists to test the
known reference FoundationPose inputs, and replacing its mask with a
freshly-computed one would stop it testing what it was built to test — a
regression that changes its own inputs cannot detect a change in the estimator.

---

## Live validation order

Once the RealSense enumerates (see `scripts/realsense_diagnose.sh`):

1. diagnose camera — model, serial, profiles, intrinsics, depth scale
2. capture empty tabletop, then tabletop + Cylinder5
3. verify depth alignment
4. run `depth_plane_foreground`
5. **inspect the mask visually** before trusting it
6. select the exact Cylinder5 CAD explicitly
7. run FoundationPose
8. show RGB, depth, mask and pose overlay together
9. repeat after physically moving Cylinder5
10. report **repeatability**, not absolute accuracy — there is no ground truth

Only after that: straight cylinders (Cylinder3/4), where axial spin is
unobservable and must be reported as symmetry-equivalent; then clutter, which
needs Stage 2.
