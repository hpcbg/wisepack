# Physical D435 -> synchronized Isaac workcell -> robot pick: evidence

Two placements of one physical Cylinder5, each carried from the Intel RealSense
D435 through the **configured demo transform** into the Isaac Sim workcell,
approved, and picked by the existing Franka Emika Panda backend from the
synchronized object's transformed pose. Recorded on 2026-09-11 with
`WISEPACK_PRESET=isaac_cylinder5_physical ./run_wisepack_dashboard.sh isaac`,
perception method `foundationpose_rgbd_model_free` (the estimator received the
learned representation only; the Isaac body is the registry's Cylinder5 CAD).

**What this is and is not.** It shows that a real RGB-D observation drives the
synchronized Isaac workcell through an explicitly configured demo transform.
The transform is `config/isaac_workcell.yaml`, provenance `configured_demo`:
a top-down camera mount assumed 542 mm above the table with the work-area
origin placed 650 mm in front of the robot base. **No physical calibration was
performed and no millimetre accuracy is claimed** — the numbers below are
consistent with each other, not measured against an independent truth.

## The two placements

| | placement 1 | placement 2 |
|---|---|---|
| D435 data | **live** capture `cylinder5-20260911-073311` | **recorded** capture `cylinder5-20260810-133408`, replayed through the same worker |
| operator ROI (px) | 255,70,445,719 | 255,70,445,719 |
| camera-frame model origin (mm) | (−161.36, 85.17, 620.76) | (−78.35, 101.85, 594.81) |
| camera-frame body centre (mm) | (−170.58, −13.66, 520.70) | (−162.70, 7.21, 533.21) |
| camera-frame tube axis (line) | (0.0951, 0.9954, 0.0121) | (0.0872, 0.9960, −0.0187) |
| work-area pose (mm) | (−170.58, 13.66, 21.30) | (−162.70, −7.21, 8.79) |
| table pose = pick target (mm) | **(479.42, 13.66, 21.30)** | **(487.30, −7.21, 8.79)** |
| Isaac world pose (m) | (0.4794, 0.0137, 0.4213) | (0.4873, −0.0072, 0.4088) |
| Isaac read-back after settling (m) | (0.4867, 0.0146, 0.4181), 8.0 mm from commanded | (0.4816, −0.0081, 0.4184), 11.2 mm from commanded |
| gripper heading from the observed axis | −84.5 deg | −85.0 deg |
| Isaac `[isaac-robot]` pick line | `pick [0.479 0.014 0.421]` | `pick [0.487 -0.007 0.409]` |
| outcome | ITEM_COMPLETED, settled in CNT-01, 50 mm from the planned pose, axis off 8 deg | ITEM_COMPLETED, settled in CNT-01, 54 mm from the planned pose, axis off 12 deg |

The two pick targets differ by 8 mm in X, 21 mm in Y and 12.5 mm in Z, with a
different tilt — the displacement between the two captures. The Isaac source
object moved by the same amount (compare the two `*-isaac-synchronized.jpg`
frames and the read-back poses), and the arm went to the new pose. I could not
move the physical tube during this session, so the second placement is a
recorded D435 capture of the same tube at a nearby, but different, bench pose;
the displacement is small and is reported as what it is.

## The chain, numerically

`scene_sync_evidence/chain-verification.txt` recomputes every step from the
saved observation and the tracked config, and compares it with the pose the
orchestrator actually sent (`run*-execution-after-sync.json`,
`isaac.scene_sync.objects[0].source_pose`): `recomputed == sent: True` for both
placements. The hand computation for the configured chain is

    workarea = ( x_cam,  -y_cam,  542 - z_cam )        # half turn about X, camera 542 mm up
    table    = ( x_wa + 650,  y_wa,  z_wa )             # work-area origin 650 mm ahead of the base
    world    = table / 1000 + (0, 0, 0.40)              # SceneLayout.table_frame_origin_m

and it reproduces the table pose above to the millimetre for both placements.
The tube axis is carried with the position: the spawned body's local +Z equals
the transformed observed axis (`tests/test_scene_sync.py`).

## Proof the pick target is the observation

* `EXECUTE_ITEM.source_pose` is `isaac_transform.source_pose_for(scene_spec, …)`,
  which returns the spec's pose verbatim for a physical scene and raises for
  an unobserved item; the bridge does not import `table_pose_for_index`.
* The pose the bridge sent equals the pose the simulator spawned the body at
  (same `SceneSpec`), and the `[isaac-robot]` pick line for each run prints
  that pose, not the generated row slot (0.48, −0.32, 0.425) the placeholder
  had before synchronization (`isaac-generated-placeholder.jpg`).
* The two runs' pick targets differ exactly as the two observations differ.

## Proof model-free still receives no CAD

The batch's `perception_method` is `foundationpose_rgbd_model_free`; the
worker request carried the learned representation mesh only (provider branch
unchanged, `tests/test_scene_sync.py::test_the_model_free_estimator_request_is_unchanged_by_the_scene_link`).
The scene object's provenance records `estimator_geometry: learned_representation`
beside `scene_geometry.source: engineering_cad`, and the Isaac log shows the
body built from `CAD-Models/STL-Files/Cylinder5.stl` through the object
registry.

## Captures

`images/generated/scene-sync/`:

| file | what it shows |
|---|---|
| `run1-d435-rgb.jpg`, `run1-d435-mask-overlay.jpg`, `run1-d435-pose-overlay.jpg` | the live D435 colour frame, the depth-plane mask inside the ROI, the CAD reprojected at the estimated pose |
| `run2-d435-*.jpg` | the same three for the recorded capture |
| `isaac-generated-placeholder.jpg` | the scene BEFORE synchronization: the generated placeholder at the row slot; frame markers for the robot base, the work-area origin and the camera (blue axis pointing down) |
| `run1-isaac-synchronized.jpg`, `run2-isaac-synchronized.jpg` | the Cylinder5 CAD body at the transformed pose, before any motion |
| `run1-isaac-approach.jpg`, `run1-isaac-grasp.jpg`, `run1-isaac-carry.jpg` | the arm turning to the observed heading, closing on the tube, carrying it to the bin |
| `run2-isaac-grasp.jpg`, `run2-isaac-lift.jpg`, `run2-isaac-place.jpg` | the same for the second placement |
| `dashboard-physical-panel.png` | Scene source / Workarea transform / Isaac scene / Object, as rendered |

`simulators/isaac/scene_sync_evidence/`: the two observation results, the
execution status after synchronization and after the pick, and the chain
recomputation.

## Known limitations

* The transform is a configured assumption. The camera tilt measured by the
  depth-plane fit (about 1.5 deg) is not modelled; the residual shows up as the
  8-11 mm settle offset and the 9-13 mm difference between the transformed
  height and a tube resting on the table.
* One object, manual identity and ROI, as scoped.
* The physical pipeline names every batch `physical-cylinder5-1`; the scene
  revision and the pose fingerprint are what distinguish two acquisitions, not
  the batch id.
* A batch adopted into a run whose plan has already completed re-plans but is
  not executed; start a new run (the dashboard's reset with the camera source)
  before the next acquisition, as was done here between the two placements.
* The Isaac frames are the DemoCamera's observational capture; the pick took
  about eight seconds of wall-clock time, so intermediate frames are sparse.
