# Physical D435 -> synchronized Isaac workcell -> robot pick: evidence

Two demonstrations, both from the Intel RealSense D435 through the
**configured demo transform** into the Isaac Sim workcell and the existing
Franka Emika Panda backend:

* **Whole scene (2026-09-12).** Every workpiece on the bench — 18 of them —
  segmented, sized and classified by footprint in one acquisition, synchronized
  into Isaac and picked one by one into one container: 18 of 18 placed, 0
  failed. Section *Whole-scene run* below.
* **Single object (2026-09-11/12).** One Cylinder5 located in full 6-DoF by
  FoundationPose model-free (no CAD to the estimator), synchronized and
  picked. Three runs: placement 1 acquired **live** twice, placement 2 a
  **recorded** capture replayed. Sections *Live versus recorded* onward.

This document and README §15a describe the same runs with the same numbers.
The two cutting capabilities that follow the whole-scene chain in the
evaluator demo (`images/generated/demo/`) are documented in
`CUT_SKILL_EVIDENCE.md` (packing-driven cut of a loose pipe) and
`DISMANTLING_EVIDENCE.md` (a section cut from an installed pipe run); the demo
is assembled from the tracked stills of all three runs by
`scripts/generate_readme_gifs.py --evaluator-demo`.

**What this is and is not.** It shows that a real RGB-D observation drives the
synchronized Isaac workcell through an explicitly configured demo transform,
and that the robot's pick target is that transformed observation rather than
the generated source-row coordinates. The transform is
`config/isaac_workcell.yaml`, provenance `configured_demo`: a top-down camera
mount assumed 542 mm above the table with the work-area origin placed in front
of the robot base by choice — **500 mm** in the tracked file (revision
`demo-2026-09-12`, so that the whole bench lies inside the Panda's reach), 650 mm
when the single-object runs below were recorded (revision `demo-2026-09-11`;
their table poses are 150 mm larger in x than the same observation would give
today). **No physical calibration was performed and no
millimetre accuracy is claimed** — the numbers below are consistent with each
other, not measured against an independent truth. Object identity and ROI are
demonstrator inputs; one object is synchronized; the pick is Isaac physics,
not a physical robot.

## Whole-scene run

**What ran.** `WISEPACK_PRESET=isaac_scene_physical WISEPACK_ISAAC_HEADLESS=1
WISEPACK_ISAAC_OBJECT_MARKERS=0 ./run_wisepack_dashboard.sh isaac`, the run
reset to the camera source, then `POST /api/perception/physical/acquire` with
`perception_method: rgbd_scene_depth_plane` (no model, no ROI: the tracked
bench crop and ignore region apply), then one approval. Everything after the
approval is the existing Isaac backend.

**Live versus recorded.** Both acquisitions of 2026-09-12 were **live**. The
first (`scene-20260912-090423`, 18 observed, all synchronized) ran four
successful picks and was then derailed by the generated late-arrival dynamic
event, which injected a 1200 mm item no camera had seen; that fault is fixed
(camera runs carry no synthetic events) and pinned by
`tests/test_physical_batch_submission.py`. The second
(`scene-20260912-091535`) is the run documented below and in the README. Its
first capture attempt had 72 % valid depth and was re-taken automatically
(`capture_attempts` in the scene document); the frame used has 99 %. The
recorded capture `cylinder5-20260912-045336` of the same bench is replayed by
`tests/test_scene_perception.py` as a regression fixture and yields the same
18 objects.

**Perception.** Method `rgbd_scene_depth_plane`; work plane fitted with
residual 1.73 mm on 92 % of the depth; 20 instances measured: 18 observed, 1
ignored (configured region, lower-left non-workpiece), 1 unclassified (the desk
edge beyond the bench crop), 0 excluded by the workcell bounds or the reach
band. Identity by footprint against `config/scene_demo_classes.yaml`; the CAD
model is instantiated after classification; bolts are measured proxies. The
pose is planar (x, y, z on the plane, heading); tilt is assumed zero.

**Scene and picks.** `SCENE_READY revision 2: 18 items, container empty, source
physical_observation batch physical-scene-1`; the acknowledgement read back all
18 bodies within 7-19 mm of the commanded pose (the drop onto the table). The
packer placed all 18 in the single 380 x 220 x 150 mm bin (11.8 % used). The
Panda picked and placed 18 of 18, 0 failed; mean placement error after PhysX
settling 42.9 mm, max 182 mm (item-011, a medium tube that rolled inside the
bin). Approval 09:15:34 UTC, last placement 09:19:17 UTC.

| item | class | model | footprint L x W x H (mm) | camera-frame centre (mm) | table pose (mm) | spawn settle (mm) | `[isaac-robot]` pick (m) | outcome |
|---|---|---|---|---|---|---|---|---|
| item-001 | medium_tube | cylinder3 | 209 x 41 x 28 | (-28, -121, 510) | (472, 121, 32) | 14 | `0.472 0.121 0.432` | 14 mm, 5 deg |
| item-002 | plate | plate2 | 61 x 40 x 9 | (142, -172, 520) | (642, 172, 22) | 18 | `0.642 0.172 0.422` | 36 mm, 4 deg |
| item-003 | long_tube | cylinder5 | 357 x 32 x 19 | (-171, -14, 520) | (329, 14, 22) | 7 | `0.329 0.014 0.422` | 69 mm, 8 deg |
| item-004 | small_cylinder | cylinder1 | 46 x 22 x 13 | (44, -141, 516) | (544, 141, 26) | 16 | `0.544 0.141 0.426` | 32 mm, 8 deg |
| item-005 | long_tube | cylinder4 | 333 x 34 x 20 | (169, -3, 514) | (669, 3, 28) | 16 | `0.669 0.003 0.428` | 29 mm, 17 deg |
| item-006 | plate | plate2 | 62 x 42 x 8 | (153, -105, 521) | (653, 105, 21) | 17 | `0.653 0.105 0.421` | 9 mm, 55 deg |
| item-007 | short_tube | cylinder2 | 80 x 41 x 30 | (80, -90, 509) | (580, 90, 33) | 16 | `0.58 0.09 0.433` | 89 mm, 29 deg |
| item-008 | short_tube | cylinder2 | 80 x 43 x 28 | (-105, -78, 512) | (395, 78, 30) | 12 | `0.395 0.078 0.43` | 56 mm, 18 deg |
| item-009 | plate | plate2 | 66 x 44 x 9 | (233, -4, 521) | (733, 4, 21) | 17 | `0.733 0.004 0.421` | 49 mm, 7 deg |
| item-010 | small_cylinder | cylinder1 | 46 x 22 x 14 | (113, -1, 517) | (613, 1, 25) | 15 | `0.613 0.001 0.425` | 43 mm, 60 deg |
| item-011 | medium_tube | cylinder3 | 207 x 39 x 32 | (-17, 21, 512) | (483, -21, 30) | 12 | `0.483 -0.021 0.43` | 182 mm, 2 deg |
| item-012 | bolt | measured proxy | 40 x 15 x 10 | (217, 59, 523) | (717, -59, 19) | 16 | `0.717 -0.059 0.419` | 31 mm, 23 deg |
| item-013 | small_cylinder | cylinder1 | 49 x 24 x 18 | (66, 73, 519) | (566, -73, 23) | 13 | `0.566 -0.073 0.423` | 12 mm, 6 deg |
| item-014 | bolt | measured proxy | 46 x 8 x 4 | (179, 85, 525) | (679, -85, 17) | 14 | `0.679 -0.085 0.417` | 11 mm, 26 deg |
| item-015 | bolt | measured proxy | 56 x 12 x 5 | (18, 91, 528) | (518, -91, 14) | 12 | `0.518 -0.091 0.414` | 18 mm, 2 deg |
| item-016 | short_tube | cylinder2 | 88 x 41 x 32 | (-69, 120, 515) | (431, -120, 27) | 9 | `0.431 -0.12 0.427` | 28 mm, 3 deg |
| item-017 | bolt | measured proxy | 59 x 14 x 8 | (181, 122, 525) | (681, -122, 17) | 14 | `0.681 -0.122 0.417` | 35 mm, 3 deg |
| item-018 | bolt | measured proxy | 45 x 12 x 4 | (32, 124, 528) | (532, -124, 14) | 11 | `0.532 -0.124 0.414` | 29 mm, 28 deg |

Camera-frame centres are the instance centroid on the plane plus half the
part's thickness toward the camera; table poses are the synchronizer's
(`scene-run-execution-after-sync.json`, `isaac.scene_sync.objects[*].source_pose`);
the pick column is the pose the simulator was commanded to pick from, which
equals the table pose in metres plus the 0.40 m table height, i.e. the
transformed observation and never a generated row slot.

**Files.** `scene_sync_evidence/scene-run-d435-physical_scene.json` (the scene
document: plane, instances, classification, placeability, capture attempts),
`scene-run-execution-after-sync.json`, `scene-run-execution-after-picks.json`,
`scene-run-pick-outcomes.json` (all 18 pick and place poses and outcomes from
the Isaac log), `scene-run-pick-log.txt` (state changes with frame indices).
`images/generated/scene-sync/scene-*`: the D435 frame, the classified overlay,
the instance masks, the Isaac scene before and after synchronization and after
the run, the dashboard panels, and the 54 sequence stills the GIF
`physical-scene-to-isaac-picks.gif` is assembled from (`gif-manifest.json`).

**Limitations of this run.** Identity by size only; planar pose; one layer of
parts with no touching or stacked objects; the transform and the reach band
are the configured demo's; bolts are proxies; the placement errors are PhysX
outcomes with the temporary fixed-joint grasp, reported as measured.

## Live versus recorded, stated exactly

| run | D435 data | status |
|---|---|---|
| run 1 — placement 1, 2026-09-11 | capture `cylinder5-20260911-073311` | **LIVE** acquisition from the physical D435 |
| run 3 — placement 1, 2026-09-12 | capture `cylinder5-20260912-045336` | **LIVE** acquisition from the physical D435 of the **same physical placement** (the tube was not moved between the sessions; the two table poses agree within 0.3 mm). This is a repeat of placement 1 with the full DemoCamera frame sequence captured for the README GIF — it is **not** a second placement |
| run 2 — placement 2 | capture `cylinder5-20260810-133408` | **RECORDED** D435 capture of an earlier bench arrangement, replayed through the same worker, synchronizer and robot backend. Real sensor data, but **not** a live physical placement and never described as one |

The tube could not be physically moved during these sessions, so there is
**one live physical placement**. Placement 2 shows that the Isaac object and
the pick target follow a *different* observation; it does not show a second
live placement.

## Configured transform provenance

`config/isaac_workcell.yaml`, both links `method: configured_demo`, revision
`demo-2026-09-12` (the single-object runs used `demo-2026-09-11`, identical
except for the 650 mm offset):

| link | translation (mm) | rotation | provenance |
|---|---|---|---|
| `camera_color_optical_frame` -> `wisepack_workarea` | (0, 0, 542) | half turn about X (q = 1, 0, 0, 0) | fixed top-down mount **assumed**; height from one depth-plane fit on the bench, used as a constant, not from a calibration procedure |
| `wisepack_workarea` -> `table` | (500, 0, 0) — (650, 0, 0) in the single-object runs | identity | work-area origin placed in front of the robot **by choice**, not by measurement |
| `table` -> `world` | z + 400 mm, metres | identity | `SceneLayout.table_frame_origin_m` of the selected robot layout |

## The three runs

| | run 1: placement 1, live | run 3: placement 1 again, live | run 2: placement 2, recorded replay |
|---|---|---|---|
| operator ROI (px) | 255,70,445,719 | 255,70,445,719 | 255,70,445,719 |
| camera-frame model origin (mm) | (−161.36, 85.17, 620.76) | (−160.92, 85.30, 620.57) | (−78.35, 101.85, 594.81) |
| camera-frame body centre (mm) | (−170.58, −13.66, 520.70) | (−170.64, −13.35, 520.38) | (−162.70, 7.21, 533.21) |
| camera-frame tube axis (line) | (0.0951, 0.9954, 0.0121) | (0.0944, 0.9954, 0.0135) | (0.0872, 0.9960, −0.0187) |
| work-area pose (mm) | (−170.58, 13.66, 21.30) | (−170.64, 13.35, 21.62) | (−162.70, −7.21, 8.79) |
| table pose = **robot pick target** (mm) | **(479.42, 13.66, 21.30)** | **(479.36, 13.35, 21.62)** | **(487.30, −7.21, 8.79)** |
| Isaac world pose commanded (m) | (0.4794, 0.0137, 0.4213) | (0.4794, 0.0133, 0.4216) | (0.4873, −0.0072, 0.4088) |
| Isaac read-back after settling (m) | (0.4867, 0.0146, 0.4181), 8.0 mm from commanded | (0.4867, 0.0143, 0.4181), 8.2 mm from commanded | (0.4816, −0.0081, 0.4184), 11.2 mm from commanded |
| gripper heading from the observed axis | −84.5 deg | −84.6 deg | −85.0 deg |
| Isaac `[isaac-robot]` pick line | `pick [0.479 0.014 0.421]` | `pick [0.479 0.013 0.422]` | `pick [0.487 -0.007 0.409]` |
| outcome | ITEM_COMPLETED, settled in CNT-01, 50 mm from the planned pose, axis off 8 deg | ITEM_COMPLETED, settled in CNT-01, 47 mm from the planned pose, axis off 5 deg | ITEM_COMPLETED, settled in CNT-01, 54 mm from the planned pose, axis off 12 deg |

Runs 1 and 3 differ by 0.06, 0.31 and 0.32 mm in x, y and z: the same
placement, seen live twice. Placement 2's pick target differs from placement
1's by 8 mm in x, 21 mm in y and 12.5 mm in z, with a different tilt — the
displacement between the two captures. The Isaac source object moved by the
same amount (compare `run1-isaac-synchronized.jpg` with
`run2-isaac-synchronized.jpg`, and the read-back poses), and the arm went to
the new pose.

## The chain, numerically

`scene_sync_evidence/chain-verification.txt` recomputes every step from the
saved observation and the tracked config, and compares it with the pose the
orchestrator actually sent (`run*-execution-after-sync.json`,
`isaac.scene_sync.objects[0].source_pose`): `recomputed == sent: True` for all
three runs. The hand computation for the configured chain is

    workarea = ( x_cam,  -y_cam,  542 - z_cam )        # half turn about X, camera 542 mm up
    table    = ( x_wa + 650,  y_wa,  z_wa )             # 650 mm ahead of the base in these runs (500 mm today)
    world    = table / 1000 + (0, 0, 0.40)              # SceneLayout.table_frame_origin_m

and it reproduces the table pose above to the millimetre for every run. The
tube axis is carried with the position: the spawned body's local +Z equals the
transformed observed axis (`tests/test_scene_sync.py`).

## Proof the pick target is the observation

* `EXECUTE_ITEM.source_pose` is `isaac_transform.source_pose_for(scene_spec, …)`,
  which returns the spec's pose verbatim for a physical scene and raises for
  an unobserved item; the bridge does not import `table_pose_for_index`.
* The pose the bridge sent equals the pose the simulator spawned the body at
  (same `SceneSpec`), and the `[isaac-robot]` pick line for each run prints
  that pose, not the generated row slot (0.48, −0.32, 0.425) the placeholder
  had before synchronization (`isaac-generated-placeholder.jpg`,
  `run3-isaac-seq-01.jpg`).
* The pick targets of placements 1 and 2 differ exactly as the two
  observations differ; the two live acquisitions of placement 1 agree.

## Proof model-free still receives no CAD

Every batch's `perception_method` is `foundationpose_rgbd_model_free`; the
worker request carried the learned representation mesh only (provider branch
unchanged, `tests/test_scene_sync.py::test_the_model_free_estimator_request_is_unchanged_by_the_scene_link`).
The scene object's provenance records `estimator_geometry: learned_representation`
beside `scene_geometry.source: engineering_cad`, and the Isaac log shows the
body built from `CAD-Models/STL-Files/Cylinder5.stl` through the object
registry — CAD is used only after perception, to instantiate the Digital Twin
object.

## Captures

`images/generated/scene-sync/`:

| file | what it shows |
|---|---|
| `run1-d435-rgb.jpg`, `run1-d435-mask-overlay.jpg`, `run1-d435-pose-overlay.jpg` | the live D435 colour frame, the depth-plane mask inside the ROI, the CAD reprojected at the estimated pose (placement 1, 2026-09-11) |
| `run3-d435-rgb.jpg`, `run3-d435-pose-overlay.jpg` | the same placement, acquired live again on 2026-09-12 |
| `run2-d435-*.jpg` | the same three for the recorded capture (placement 2) |
| `isaac-generated-placeholder.jpg` | the scene BEFORE synchronization: the generated placeholder at the row slot; frame markers for the robot base, the work-area origin and the camera (blue axis pointing down) |
| `run1-isaac-synchronized.jpg`, `run2-isaac-synchronized.jpg` | the Cylinder5 CAD body at the transformed pose, before any motion |
| `run1-isaac-approach.jpg`, `run1-isaac-grasp.jpg`, `run1-isaac-carry.jpg` | the arm turning to the observed heading, closing on the tube, carrying it to the bin |
| `run2-isaac-grasp.jpg`, `run2-isaac-lift.jpg`, `run2-isaac-place.jpg` | the same for the recorded-replay placement |
| `run3-isaac-seq-01.jpg` … `run3-isaac-seq-18.jpg` | the DemoCamera sequence of run 3: placeholder, synchronized scene, approach, grasp, lift, carry, release, retreat, settled |
| `physical-to-isaac-pick.gif` | the run 3 sequence assembled with labels by `python3 scripts/generate_readme_gifs.py --scene-sync-gifs` from the frames listed in `scene_sync_evidence/gif-manifest.json`; the assembler records nothing and refuses a missing frame, so regeneration cannot substitute a generated-scenario capture |
| `scene-d435-rgb.jpg`, `scene-d435-classified.jpg`, `scene-d435-masks.jpg` | the whole-scene run's live D435 frame, the classified overlay, the instance masks |
| `scene-isaac-before.jpg`, `scene-isaac-synchronized.jpg`, `scene-isaac-final.jpg` | the Isaac workcell with the placeholder, synchronized to the 18 observed objects, and after the 18 picks |
| `scene-isaac-seq-01.jpg` … `scene-isaac-seq-54.jpg`, `physical-scene-to-isaac-picks.gif` | the whole-scene run's DemoCamera sequence and the GIF assembled from it |
| `scene-dashboard-scene.png`, `scene-dashboard-execution.png`, `scene-dashboard-twin.png` | the dashboard's scene panel, physical execution record and container Digital Twin after the run |
| `dashboard-physical-panel.png` | Scene source / Workarea transform / Isaac scene / Object, as rendered |

No two-position GIF is provided: placement 2 is a recorded replay and its
displacement from placement 1 is about two centimetres, which is not a clear
visual comparison; the two synchronized stills and the pose table above are
the comparison.

`simulators/isaac/scene_sync_evidence/`: the three observation results
(`run*-d435-physical_c5.json`), the execution status after synchronization and
after the pick for each run, the chain recomputation and the GIF manifest.

## Known limitations

* The transform is a configured assumption. The camera tilt measured by the
  depth-plane fit (about 1.5 deg) is not modelled; the residual shows up as the
  8-11 mm settle offset and the 9-13 mm difference between the transformed
  height and a tube resting on the table.
* One object, manual identity and ROI, as scoped.
* One live physical placement; the second placement is a recorded replay.
* The physical pipeline names every batch `physical-cylinder5-1`; the scene
  revision and the pose fingerprint are what distinguish two acquisitions, not
  the batch id.
* A batch adopted into a run whose plan has already completed re-plans but is
  not executed; start a new run (the dashboard's reset with the camera source)
  before the next acquisition, as was done here between the placements.
* The Isaac frames are the DemoCamera's observational capture at about 2.4
  frames per second; the pick took about eight seconds of wall-clock time, so
  intermediate frames are sparse.
