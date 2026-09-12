# Physical D435 -> synchronized Isaac workcell -> robot pick: evidence

One physical Cylinder5, carried from the Intel RealSense D435 through the
**configured demo transform** into the Isaac Sim workcell, approved, and picked
by the existing Franka Emika Panda backend from the synchronized object's
transformed pose. Three runs are documented: placement 1 acquired **live**
twice (2026-09-11 and 2026-09-12), and placement 2 as a **recorded** D435
capture replayed through the same worker. All three used
`WISEPACK_PRESET=isaac_cylinder5_physical ./run_wisepack_dashboard.sh isaac`
and perception method `foundationpose_rgbd_model_free` (the estimator received
the learned representation only; the Isaac body is the registry's Cylinder5
engineering CAD). This document and README §15a describe the same runs with the
same numbers.

**What this is and is not.** It shows that a real RGB-D observation drives the
synchronized Isaac workcell through an explicitly configured demo transform,
and that the robot's pick target is that transformed observation rather than
the generated source-row coordinates. The transform is
`config/isaac_workcell.yaml`, provenance `configured_demo`: a top-down camera
mount assumed 542 mm above the table with the work-area origin placed 650 mm
in front of the robot base. **No physical calibration was performed and no
millimetre accuracy is claimed** — the numbers below are consistent with each
other, not measured against an independent truth. Object identity and ROI are
demonstrator inputs; one object is synchronized; the pick is Isaac physics,
not a physical robot.

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
`demo-2026-09-11`:

| link | translation (mm) | rotation | provenance |
|---|---|---|---|
| `camera_color_optical_frame` -> `wisepack_workarea` | (0, 0, 542) | half turn about X (q = 1, 0, 0, 0) | fixed top-down mount **assumed**; height from one depth-plane fit on the bench, used as a constant, not from a calibration procedure |
| `wisepack_workarea` -> `table` | (650, 0, 0) | identity | work-area origin placed in front of the robot **by choice**, not by measurement |
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
    table    = ( x_wa + 650,  y_wa,  z_wa )             # work-area origin 650 mm ahead of the base
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
