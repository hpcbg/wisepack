# WISEPACK — Isaac Sim physical execution backend

The optional **execution backend** that performs WISEPACK's approved placements
with a SELECTED manipulator in NVIDIA Isaac Sim 6.0.1, instead of resolving them
with the built-in simulated robot model.

```
ISAAC_SIM_ROOT/python.sh simulators/isaac/wisepack_isaac.py \
    --preset isaac_cylinders_smoke --seed 42 [--headless]
```

Normally you do not run it directly — use `./run_wisepack_dashboard.sh isaac`,
which starts the WISEPACK stack, the dashboard and this simulator together.

---

## What this is, precisely

| | |
|---|---|
| **Data source** | `sim` / `ros` / `fiware` — where the **dashboard reads state from** |
| **Execution backend** | `simulated` / `isaac` — **who moves the item** |

They are orthogonal. Isaac is *not* a fourth data source. A run can be executed
by Isaac and observed through FIWARE; it can be executed by the simulated robot
and observed over ROS.

When `execution_backend=isaac` the orchestrator **never calls**
`WorkflowEngine.step_execution()`. That is the whole single-authority guarantee:
there is no moment at which a simulated outcome and a physical outcome both
claim the same placement, because only one of the two code paths is reachable.

---

## Honest limitations

These are stated first, not buried, because each one changes how a result should
be read.

**No perception.** There is no camera, no detector and no pose estimator. Item
poses are **ground truth** — this process spawned the items, so it knows exactly
where they are. The extension point is
`wisepack_core.isaac_transform.table_pose_for_index`; replacing it with real
perception changes nothing else in the stack.

**Secure-grasp approximation.** When the gripper closes, the item is welded to
the robot's end-effector link with a temporary USD fixed joint, removed the instant the gripper
opens to release it. So the **carry is idealised**: the item cannot slip or
rotate in the fingers. A real parallel-jaw grasp of a smooth steel pipe is
exactly the case where friction modelling matters most, and this iteration does
not model it. See `grasp.py` for why a first iteration prefers a deterministic
carry to a stochastic one.

**The release and everything after it are real.** The joint is destroyed
*before* the item falls. The drop, the impact, the roll, the contacts with the
walls and with items already placed, and the final resting pose are resolved by
PhysX with no assistance. **No item is ever teleported into the container.**

**Target pose ≠ measured pose.** A released cylinder rolls. Every item reports
its measured final pose and its distance from the planned one, and that error is
never rounded away or replaced by the target. Millimetre agreement with the
optimizer is not claimed and not expected in this iteration.

**A `z` target axis is approximated.** A top-down parallel gripper cannot stand
a pipe on its end without a regrasp. If a plan asks for one, the item is placed
horizontally and the ~90° axis error is *reported*. The
`isaac_cylinders_smoke` preset restricts its items to the `x`/`y` axes
(`GeneratorConfig.permitted_axes`) so the supported scenario never hits this.

---

## The scenario

`isaac_cylinders_smoke` — four bench-scale pipe segments into one open-top bin.

Every number in it is a **hardware** constraint, not a packing choice:

| | |
|---|---|
| 4 items | one pick-and-place cycle is ~25 s of simulated time |
| ⌀ ≤ 70 mm | both supported parallel grippers open to 80 mm |
| length ≤ 250 mm | must fit the 300 mm bin horizontally with clearance |
| axes `x`,`y` | a top-down gripper cannot stand a pipe on its end |

It is **not a packing benchmark** and contributes nothing to the measured
baseline-versus-optimized result. The 40-item `mixed_pipes_dense` benchmark is
deliberately *not* the physical default: forty robotic cycles is twenty minutes
of watching an arm.

The WISEPACK stack and this scene both call
`wisepack_core.build_scenario(preset, seed)`, so object ids, dimensions and
masses match **by construction**. There is no second hard-coded object list.

---

## Module layout

Each piece is separable and only one group touches the simulator:

| file | needs Isaac? | responsibility |
|---|---|---|
| `config.py` | no | tunables, validated |
| `result.py` | no | settling + containment verdicts |
| `bridge.py` | rclpy only | ROS 2 transport |
| `scene.py` | yes | procedural table / bin / cylinders |
| `grasp.py` | yes | the temporary fixed joint |
| `robot.py` | yes | the ROBOT-NEUTRAL placement state machine |
| `wisepack_isaac.py` | yes | assembly and the main loop |

`config.py` and `result.py` are unit-tested by the ordinary suite on a machine
with no GPU. **Nothing outside `simulators/isaac/` imports `isaacsim`, `omni`,
`carb` or `pxr`** — asserted by a test, not just intended.

Structured log prefixes: `[isaac-app]`, `[isaac-scene]`, `[isaac-robot]`,
`[isaac-bridge]`, `[isaac-result]`, plus `[isaac-launch]` from the shell script.

---

## The robot sequence

```
HOME → PRE_GRASP → GRASP → ATTACH → LIFT → PRE_PLACE
     → PLACE_ORIENTATION → RELEASE → DETACH → RETREAT
     → WAIT_FOR_SETTLE → NEXT_ITEM
```

Conservative on purpose: one Cartesian servo goal per state, a convergence
tolerance, a frame budget, no blending and no re-planning. A failure is
attributable to a named state, which is worth more in a first integration than a
smoother trajectory.

**The rule the path must not break:** a held item is never moved laterally below
the container rim. Every lateral motion happens at `rim + container_clearance`;
the descent to the drop height is purely vertical at the target XY. Dragging a
cylinder through a wall would be resolved by PhysX as a large impulse or — worse
and more likely — as a quiet penetration that still ends "in" the bin.

**Controller:** `simulators/isaac/adapters/` — one generic articulation and one
kinematics implementation for every supported robot, providing damped
least-squares differential IK over the arm Jacobian plus gripper control. It is
iterative, so every state calls it once per physics frame and watches for
convergence rather than commanding a pose and assuming arrival.

MoveIt is deliberately not used: the repository has no existing integration to
reuse, and adding a planning stack plus a URDF pipeline to move a gripper
between four known poses would buy capability this sequence does not need.

**An item is not complete because the gripper opened.** `WAIT_FOR_SETTLE`
decides, and it decides from measured velocities and the measured final pose:
the body must exist, be inside the container footprint, be above the floor, not
be perched above the rim, and stay below both velocity thresholds for a stable
interval — or `settle_timeout` expires, which is *recorded* rather than silently
equivalent to a clean settle.

### Tunables

All overridable per value, in metres / seconds:

```
WISEPACK_ISAAC_PRE_GRASP_HEIGHT              0.12
WISEPACK_ISAAC_LIFT_HEIGHT                   0.25
WISEPACK_ISAAC_CONTAINER_CLEARANCE           0.15
WISEPACK_ISAAC_DROP_HEIGHT                   0.06
WISEPACK_ISAAC_RETREAT_HEIGHT                0.20
WISEPACK_ISAAC_GOAL_TOLERANCE                0.015
WISEPACK_ISAAC_SETTLE_TIMEOUT                6.0
WISEPACK_ISAAC_LINEAR_VELOCITY_THRESHOLD     0.01
WISEPACK_ISAAC_ANGULAR_VELOCITY_THRESHOLD    0.10
WISEPACK_ISAAC_SETTLE_STABLE_TIME            0.35
WISEPACK_ISAAC_GRASP_YAW_OFFSET_DEG          0.0
WISEPACK_ISAAC_PHYSICS_DEVICE                cpu
```

`MotionConfig.validate()` rejects inconsistent combinations (e.g. a
`container_clearance` below `drop_height`) before a robot moves.

---

## Scene reset — the safety path

A new WISEPACK scenario does not redraw this world; it **commands** it. The
previous run's cylinders are still in the container, and the new plan assumes
they are back at their source poses.

`RESET_SCENE` is handled in `_reset_scene()` and the order is not cosmetic:

1. **abort the sequence** — this detaches the temporary grasp joint *before* any
   body it welds to can be deleted;
2. cancel the previous run and re-key the run gate, so late feedback for it is
   rejected rather than attributed to the new run;
3. open the gripper and home the arm, and let it get there before the world
   changes underneath it;
4. **stop the timeline**, then delete every item and rebuild from the new
   `(preset, seed)`, then **play** again;
5. re-command the home pose (`stop()` restores *authored* joint values, not the
   ready pose the sequence assumes) and zero item velocities;
6. settle;
7. **prove the world is usable** — `_verify_scene_usable()`;
8. only then publish `SCENE_READY` with the scenario revision.

**Why step 4 is stop-mutate-play.** Deleting a rigid body while physics is
playing invalidates the PhysX tensor simulation view for the *whole stage*,
including the arm's articulation. Measured on a live run before this ordering
existed: the items rebuilt perfectly, `SCENE_READY` was published, and the next
joint read failed with

```
Simulation view object is invalidated and cannot be used again to call
getDofPositions
```

— a scene reporting ready with an unusable robot, which is the exact hazard the
handshake exists to remove. `_verify_scene_usable()` is the belt to that braces:
it *reads* the joints (asking a wrapper whether it is valid does not prove the
view survived), reads every item's pose, and refuses if a grasp joint survived.
Anything it rejects becomes `RESET_FAILED`, and the orchestrator holds.

Before any motion, `_pre_pick_refusal()` refuses a pick whose scenario revision
does not match the built scene, whose item does not exist or has no readable
pose, whose predecessor's grasp joint is still attached, or whose item is already
inside the destination container. Each of those is a way a de-synchronised scene
becomes uncontrolled motion.

### Bounded live validation

```bash
./scripts/run_wisepack_isaac.sh --reset-test --max-runtime 900
```

Executes one item for real, requests a new scenario, verifies the rebuilt world,
then executes the first item of the *new* run. Prints `RESET-VALIDATE` markers
and exits non-zero on failure:

```
RESET-VALIDATE PHASE1 completed=1 failed=0
RESET-VALIDATE PHASE1-IN-CONTAINER 1
RESET-VALIDATE CONTAINER-CLEARED yes
RESET-VALIDATE RESPAWNED 4/4 at-source=4/4
RESET-VALIDATE ROBOT-HOME max_joint_error=0.001 rad
RESET-VALIDATE GRASP-JOINT released
RESET-VALIDATE PHASE2 completed=1 failed=0
RESET-VALIDATE RESULT PASS
```

In-process reset is consequently **enabled** in the Isaac modes; a launcher
restart is not required.

---

## Coordinates — one conversion layer, and only one

`wisepack_core/isaac_transform.py` is the **only** place WISEPACK units become
Isaac units. Axis swaps and scale factors scattered through scene, robot and
validation code are individually plausible and collectively unfalsifiable, and
the failure they produce — a mirrored container — looks like a physics problem
rather than an arithmetic one.

| frame | units | origin |
|---|---|---|
| `world` | m, Z up | Isaac world; robot base at `(0, 0, table_top_z)` |
| `table` | mm | robot base projected onto the table top |
| `container:<id>` | mm | the container's **inner min corner** |

`container:<id>` is already exactly the frame `domain.Placement.position` uses,
so a placement needs no reinterpretation — only a min-corner→centre shift and a
unit conversion. It is covered by tests that need no GPU
(`tests/test_isaac_backend.py`), including a full world→pose round trip.

`SceneLayout.validate()` checks that the pick row and **all four** container
corners sit inside the SELECTED ROBOT's usable envelope at retreat height, because an
unreachable IK goal does *not* fail loudly in Isaac — differential IK converges
to the nearest achievable pose and the gripper closes on air, which reads as a
grasp-tuning problem and is not one.

---

## ROS 2 contract

Two `std_msgs/String` topics carrying versioned JSON — no custom messages, so
Isaac's bundled interpreter never needs to import a colcon-built package:

| topic | writer | payload |
|---|---|---|
| `/wisepack/isaac/command` | orchestrator | `IsaacCommand` |
| `/wisepack/isaac/feedback` | Isaac | `IsaacFeedback` |

Schema `wisepack-isaac/1.0`, defined once in `wisepack_core/isaac_contract.py`
and imported by **both** ends from the same file. A mismatched schema MAJOR is
refused rather than best-effort parsed — reading a v2 pose with v1 field
meanings moves a robot.

Required fields: `schema_version`, `run_id`, `item_id`, `sequence_index`,
`source_pose`, `target_pose`, `dimensions`, `command`/`state`, `timestamp`.

Feedback states: `READY`, `MOVING_TO_PICK`, `GRASPING`, `LIFTING`,
`MOVING_TO_CONTAINER`, `RELEASING`, `SETTLING`, `ITEM_COMPLETED`, `ITEM_FAILED`,
`RUN_COMPLETED`, `RUN_FAILED`. These map onto the **existing** WISEPACK workflow
stages via `wisepack_core.execution.stage_for_isaac_state` — there is no
parallel dashboard-only state machine.

**`RunGate` is used on both ends.** The command topic is `TRANSIENT_LOCAL` (Isaac
takes ~30 s to boot and must not miss `RUN_BEGIN`), so a latched command is
redelivered on every re-subscribe. Acting on the redelivery means picking an
item that is already in the container. The gate rejects foreign `run_id`s and
de-duplicates on `sequence_index`.

### Not Isaac-specific

Nothing in the contract mentions a simulator: no USD path, no PhysX setting, no
joint. A **real robot cell** answering these two topics is a drop-in replacement,
and `wisepack_orchestration/isaac_bridge.py` would not change — it contains no
simulator imports either.

---

## Process and environment

**Isaac runs on the host in its own bundled Python.** WISEPACK may run in Docker
with host networking; the two meet on the DDS wire, on a shared `ROS_DOMAIN_ID`,
and nowhere else.

Do **not** source `/opt/ros/jazzy/setup.bash` into this process. Enabling
`isaacsim.ros2.bridge` puts Isaac's internally-built ROS 2 Jazzy on the path, and
`import rclpy` then resolves to a build compiled against Isaac's Python ABI.
Sourcing the host ROS environment puts a second, ABI-incompatible `rclpy` ahead
of it and the failure is an import-time crash inside rclpy's C extension. The
launcher scrubs `PYTHONPATH`, `AMENT_PREFIX_PATH` and friends for this reason.

Discovery order for the simulator root: `ISAAC_SIM_ROOT`, then known server
locations, newest first, never silently selecting an older major version.

---

## Troubleshooting

**Isaac never reports READY.** The orchestrator holds for
`isaac_ready_timeout_s` (240 s) then enters DEGRADED with a diagnostic. Check
both ends are on the same domain:

```bash
echo $ROS_DOMAIN_ID
ros2 topic info /wisepack/isaac/feedback
```

First launch is slow — shader compilation can take several minutes and is cached
in `~/.cache/ov` afterwards.

**No display / over SSH.** `WISEPACK_ISAAC_HEADLESS=1`. GUI is the default when
`DISPLAY` is set; the launcher falls back to headless automatically and says so.

**`ImportError` inside rclpy.** The host ROS environment leaked in. Run the
launcher rather than calling `python.sh` from a shell that has sourced ROS.

**Items are grasped along their axis instead of across it.** The finger closing
direction is a property of the shipped hand asset, not of this code. Correct it
with `WISEPACK_ISAAC_GRASP_YAW_OFFSET_DEG=90` rather than editing the state
machine.

**`Could not find assets root folder`.** The robot asset is fetched from NVIDIA's asset
server at runtime; the machine needs outbound HTTPS, or a local Nucleus root set
in `~/.local/share/ov/data/Kit/**/user.config.json`.
