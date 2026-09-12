# Isaac Sim cutting skill: grip, cut, place — evidence

One Franka Emika Panda in Isaac Sim 6.0.1 carries a **combined end effector**:
its parallel-jaw gripper and a shear mounted on the same hand, independently
actuated, with a fixed, declared transform between the grasp frame and the cut
frame. The cut-aware planner picks the cut plane; the robot grips the segment
that will be retained, shears at the plane, keeps the retained segment in the
gripper, places it, leaves the remainder where it lies, registers both derived
items and re-plans — and the operator approves the packing again before any
other object moves.

The evaluator-facing demo (`images/generated/demo/wisepack-end-to-end-demo.gif`,
`.mp4`, and the shorter README cut) is assembled from the stills of this run,
of the whole-scene run in `SCENE_SYNC_EVIDENCE.md`, and of two dashboard
panels captured for this scenario (`images/generated/demo/dashboard-cut-*.png`:
the cut-aware recommendation for `isaac_cut_demo` seed 7 before and after
`approve_cut`, taken from the sim-mode dashboard whose planner output is
identical to the live run's). The storyboard is
`images/generated/demo/demo-manifest.json`; `python3
scripts/generate_readme_gifs.py --evaluator-demo` re-assembles it from tracked
stills and records nothing.

The second cutting scenario — a section cut from an INSTALLED pipe run and
packed while the remainder stays installed — is documented in
`DISMANTLING_EVIDENCE.md`; it reuses this skill unchanged.

Every number below is from the run whose files sit beside this document in
`scene_sync_evidence/cut-run-*` (2026-09-12, generated preset `isaac_cut_demo`
seed 7, live Isaac Sim execution driven through the dashboard API). README
§15b describes the same run with the same numbers.

**What this is and is not.** It is a physics-simulated robot executing a real
planner decision through the unchanged WISEPACK workflow, with the cut itself a
**discrete, authoritative scene event** — the tube body is replaced by two rigid
segment bodies where its material lay, and PhysX acts on both from then on. It
is **not** a fracture model: no steel is broken by contact forces, the blades
are visual geometry animated for the demonstration and take no part in contact,
and no tool change is modelled. The scene is generated, not observed by a
camera.

## The run, stage by stage

`cut-run-dashboard-stages.txt` is the dashboard's stage/robot-state log, one
line per change (UTC, captured frame index, workflow stage, approval state,
robot state, scenario revision, Isaac state, in-flight item, completed, failed):

```text
CUT_IN_PROGRESS  pending  idle 1 | GRASPING          - 0 0
CUT_COMPLETED    pending  idle 1 | LIFTING           - 0 0
CUT_COMPLETED    pending  idle 1 | MOVING_TO_CONTAINER - 0 0
CUT_COMPLETED    pending  idle 1 | RELEASING         - 0 0
WAIT_FOR_OPERATOR_APPROVAL pending idle 1 | ITEM_COMPLETED - 1 0   <- retained segment placed; packing approval required AGAIN
PICK_ITEM        approved picking 2 | MOVING_TO_PICK tube-short-a 1 0
...
PICK_ITEM        approved picking 2 | None tube-long-s1 2 0        <- the REMAINDER, picked from where the cut left it
...
COMPLETE         approved idle 2 | ITEM_COMPLETED - 4 0
```

1. **Cut proposal.** `compare_cut_aware`: the 420 mm tube fits the 300 x 220 x
   150 mm bin nowhere whole; no-cut leaves it unplaced. Recommended
   `cut:tube-long:150-267:max_density` — one cut, segments 150 + 267 mm, kerf
   3 mm, "net whole-process benefit 986 > 0" (`cut-run-whole-process-compare.json`).
2. **Cut approval** (`approve_cut`, operator `demo-operator`): a separate
   approval; it authorises the cut-and-place of the retained segment and no
   other pick.
3. **Grip.** `EXECUTE_CUT tube-long #-1` (sequence index -1: outside the
   placement queue). The fingers go to the retained side, 60 mm from the cut
   plane — "shear at [0.421 -0.321 0.42] (152 mm from the end), fingers at
   [0.481 -0.321 0.42] on tube-long-s2" — and close (weld offset 0.1042 m
   along the approach axis, i.e. centred on the tube).
4. **Cut.** The blades close (animated, 35 mm of travel each in 24 frames).
   At `CUT_COMPLETED` the scene deactivates the tube and spawns
   `tube-long-s1` (150 x 40 mm, 0.411 kg, at (0.345, -0.315, 0.42) m) and
   `tube-long-s2` (267 x 40 mm, 0.731 kg, at (0.557, -0.315, 0.42) m) where
   the material lay; s2 is welded to the hand in place of the tube, held
   76 mm from its centre.
5. **Retained segment placed** straight from the cut, no regrasp: settled
   inside the container, **20 mm** from the planned pose, axis off 7 deg.
   The TCP goal was offset by the 76 mm grasp offset so the segment's
   centre, not the fingertips, reached the planned pose.
6. **Remainder stays on the table.** Trace of `tube-long-s1` (metres):
   after creation (0.3451, -0.3152, 0.42), after the retained weld the same,
   lift frame 45 (0.3451, -0.3171, 0.42), carry start / at release / after
   the retained segment settled **(0.3451, -0.3171, 0.42)** — 1.9 mm from
   its cut-side pose, asleep from lift frame 30 on.
7. **Derived items registered, re-plan, approval.** The workflow registered
   both segments from the simulator's measured poses (`cut-run-state-final.json`:
   `tube-long-s1` source (345, -315, 20) mm, `tube-long-s2` (557, -315, 20)
   mm, scenario revision 2), kept the validated cut-aware plan because the
   measured lengths (150/267) match the proposal exactly, validated it again
   against the registered items, and asked for **packing approval again**.
8. **All remaining objects picked** after `approve`: `tube-short-a` 107 mm
   (rolled off the retained segment it was released beside — reported, not
   corrected), **the remainder `tube-long-s1` 11 mm**, `tube-short-b` 23 mm.
   `cut-run-execution-final.json`: 4 reported, **4 completed, 0 failed**,
   mean 40.6 mm, max 107.5 mm. Stage `COMPLETE` 63 s after `RUN_BEGIN`.

No pick was refused, no object's expected source pose changed because of
the cut (the two short tubes kept their generated row poses; the segments
carry their measured cut-side poses), and no generated row position was
substituted for a measured one: the simulator's pre-pick check compares each
item against the pose the plan carries for it.

## What had to be measured to get here

These are the three things that were wrong in the first live runs, each found
by instrumenting poses and each fixed physically, not by moving anything back:

| Symptom | Measured cause | Fix |
|---|---|---|
| Remainder rolled 6-12 cm during the lift; a later released segment popped out of the bin | The first tool authored the blades and bracket as rigid bodies with `physics:collisionEnabled = false`. Isaac 6.0.1 still resolved contacts against them: a free tube overlapping a blade by 1 mm was pushed **54 mm**, one under the bracket **49 mm** | The tool is plain USD geometry under the hand (no rigid body, collider, joint or mass), blades animated by their local translation. `tool_check.py`: 0.0 mm |
| Remainder still kicked 6 cm at lift frame 2 | The retained segment's cut face is one kerf (3 mm) from the remainder's; the first differential-IK step towards a far lift goal carried a 3.6 mm lateral transient and closed the gap | Ramped straight-up retract (`MotionConfig.cut_retract_step` = 1.5 mm per frame) until the segment has risen 60 mm, then the ordinary lift |
| Remainder still crept 4-6 cm, accelerating, while the retained segment rose beside it, with no geometric overlap | PhysX generates contacts between shapes closer than the sum of their contact offsets; with the defaults the two cut faces counted as touching and friction dragged the remainder. `tool_check.py`: moving one segment past the other rolled it **14.7 mm** with default offsets, **0.0 mm** at a 1 mm contact offset | Every item collider carries a 1 mm contact offset and 0 rest offset (`WisepackScene.ITEM_CONTACT_OFFSET_M`, sum 2 mm < kerf 3 mm) |
| Retained segment settled on the bin rim, 160 mm from plan, reported "inside" | The arm sent the TCP to the segment's target while the segment was held 76 mm off-centre | `PlacementSequence._tcp_goal_for_item`: the TCP goal is offset by the weld offset for every placement |

`cut-run-tool-check.txt` is the standalone Isaac check that pins these
(16 of 17 pass; the one expected failure is deleting a rigid prim while playing,
recorded so the cut never does it).

## Known limitations

* A generated scene: the tube is where the preset put it, not where a camera
  saw it. The physical D435 scene path (`SCENE_SYNC_EVIDENCE.md`) and this cut
  path share the planner, workflow and robot backend but were not run together.
* The cut is a discrete event on a planner-selected plane. Blade geometry
  never touches anything; kerf is accounted for by the planner, not swept.
* The remainder moved 1.9 mm during the retract (the residual contact-offset
  interaction in the first frames, before the retained face has risen clear).
  It is reported, not corrected.
* `tube-short-a` was released beside the retained segment and rolled 107 mm
  off its planned pose inside the bin. The placement error is measured and
  reported per item; nothing is nudged into place.
* Frames come from the DemoCamera at about 4.4 frames per second; the blade
  stroke (0.4 s) spans two or three of them.
