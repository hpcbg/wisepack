# Robotic dismantling in Isaac Sim: installed pipe → released section → packed — evidence

The second cutting scenario. The first (`CUT_SKILL_EVIDENCE.md`) cuts a
loose tube because it fits the container nowhere whole. This one starts
**before the waste item exists**: a steel pipe run is part of the plant,
fixed to a support frame above the bench. The robot grips the section to be
removed, cuts it from the installation with the same combined gripper +
cutter, the released section is registered as a new WISEPACK waste item with
its provenance, the plan is validated against it, the operator approves the
packing, and the released section goes into the container straight from the
cut — the fixed remainder stays installed. Everything below is from the run
whose files sit beside this document in `scene_sync_evidence/dismantling-run-*`
(2026-09-13, preset `isaac_fixed_pipe_dismantling` seed 7, live Isaac Sim
6.0.1 execution driven through the dashboard API). README §15c describes the
same run with the same numbers.

**What this is and is not.** The same grip-cut-place skill, cut-aware
workflow, Human-in-the-Loop approvals, scene synchronization and packing
architecture as the loose-pipe cut; nothing was added beside them except the
notion of an *installed component*. The cut is the same discrete,
authoritative scene event on the planner's plane — no fracture physics, the
blades are visual — and the scene is generated, not observed by a camera. The
installation is a small pipe-support rig (post, cantilever arm, clamps)
inspired by pipe test rigs; it does not reproduce any particular testbed.

## The scene: preset `isaac_fixed_pipe_dismantling`

| | |
|---|---|
| Loose parts on the bench | `tube-short-a` 160 mm, `tube-short-b` 150 mm, `tube-mid` 260 mm (OD 40 mm steel tubes, pick row as in every generated run) |
| Installed component | `pipe-run-1`: 400 mm, OD 40 mm, wall 3 mm carbon-steel pipe, status **installed**, `cut_allowed` (one cut, minimum segment 120 mm, 20 mm protected ends) |
| Where it is | table frame (100, −520, 250) mm — the table frame's origin is the robot base on the bench, so: 100 mm ahead of the base, 520 mm to the robot's left, **250 mm above the bench**, lying along X |
| Fixed end | its −X end (`fixed_end: "-z"` in the item's own axis), clamped on a support frame behind the robot's left shoulder: 40 mm aluminium-profile post on a base plate, a 260 mm cantilever arm under the pipe, two blackened pipe clamps with bolts, a cross rail on the post |
| Free end | the +X end reaching 300 mm into the workspace; the predefined dismantling request releases its outer **150 mm** (`removable_length_mm`) |
| Container | the 300 × 220 × 150 mm bench bin, one container |
| Materials | OmniPBR metals built per stage: **Steel** (grey, metallic 0.9, roughness 0.46) on every tube and segment, **Aluminium** on the frame, **Clamp** (blackened steel) on the clamps, **SteelDark** on the cutter bracket, **Blade** on the shear blades; a 512 × 512 brushed roughness map is synthesised at build time (`WisepackScene._brushed_roughness_texture`) and bound as the roughness texture with 0.3–0.7 influence. Robot, table and container keep their own looks. |

The installed pipe is a **kinematic** rigid body: PhysX never moves it,
whatever touches it. The support prims are static colliders.

## Installed versus waste, in the model

* `ItemStatus.INSTALLED` and `WasteItem.installation` (component id, fixed
  end, support id, elevation, removable length; after a cut `dismantled_from`
  and `cut_operation_id`). `Scenario.packable_items` excludes installed
  components; the baseline and optimized packers, the cut-aware comparison and
  the validator all read that set. Before the cut, `pipe-run-1` is neither
  placed nor "unplaced": it is not waste (`test_dismantling.py`).
* The pick pose of an installed component is its declared position and axis
  (`isaac_transform.installed_pose_for`), never a generated row slot — the same
  function serves the scene builder and the bridge.

## The dismantling cut through the same workflow

1. **Proposal.** `plan_cut_aware` adds one alternative per strategy for every
   installed component with a removable length: segments `[247, 150]` mm
   (kerf 3 mm), released = the free-end segment, remainder = the fixed-end
   segment re-registered as INSTALLED so the packer never sees it. The
   alternative is credited the released section entering the plan
   (`released_component_value`), not a container saving. Recommended:
   `dismantle:pipe-run-1:247-150:max_density`, "Dismantling pipe-run-1: one
   cut releases the 150 mm free section as a waste item that the plan packs;
   the 247 mm remainder stays installed. Net whole-process benefit 1477 > 0."
2. **Approval.** `approve_cut` — the separate cut approval; packing approval
   comes again after the cut. The dashboard's cut panel carries a compact
   *Dismantling* block: installed component, cut proposed, cut position (150 mm
   from the free end · 247 mm from the fixed end), released segment, fixed
   remainder, cut approved, segment registered as waste, packing target
   (CNT-01 at (75, 60, 20) mm, axis X), execution status.
3. **Dispatch.** `EXECUTE_CUT pipe-run-1` with `fixed_segment_id =
   pipe-run-1-s1`, `retained_segment_id = pipe-run-1-s2`, `installed: true`;
   the contract refuses a request that would retain the installed segment.
4. **Grip.** "shear at (0.149, −0.52, 0.65) m (248 mm from the end), fingers
   at (0.209, −0.52, 0.65) on pipe-run-1-s2, yaw −180°". The fingers close on
   the removable section; the installed pipe is **not** welded to the hand
   ("installed component: fingers closed on the removable section, no weld to
   the plant") — the plant holds it.
5. **Cut.** At `CUT_COMPLETED` the scene deactivates the pipe and spawns
   `pipe-run-1-s1` (247 mm, **kinematic**, in the clamps, at (0.0235, −0.52,
   0.65) m — its fresh cut face towards the released segment) and
   `pipe-run-1-s2` (150 mm, dynamic, at (0.225, −0.52, 0.65) m), which is
   welded to the hand in place of the pipe. Remainder trace: after creation,
   after the weld, at lift, at carry, at release, after the retained segment
   settled — **(0.0235, −0.52, 0.65) m, unchanged**.
6. **Direct transfer.** Retract straight up out of the cut, carry above the
   run (the travel height is the higher of the row clearance and the grip
   height plus approach clearance, `PlacementSequence._clear_z`), turn to the
   planned axis above the bin, descend, release: `pipe-run-1-s2` settled
   inside the container **17 mm** from the planned pose, axis off 13°. No
   drop on the bench, no regrasp.
7. **Registration.** `pipe-run-1` leaves the scenario; `pipe-run-1-s1` stays
   INSTALLED with the installation carried over (`dismantled_from`,
   `cut_operation_id` = the request id); `pipe-run-1-s2` is a derived waste
   item, generation 1, lineage `dismantling: true`, `dismantled_from:
   pipe-run-1`, `cut_operation_id`, `released_length_mm: 150`, status
   available for packing, measured cut-side pose. The validated dismantling
   plan stands (measured lengths equal the proposal), is validated again
   against the registered items, and **packing approval is required again**.
8. **The rest.** After `approve`: `tube-mid` 26 mm, `tube-short-a` 26 mm,
   `tube-short-b` 101 mm (rolled off a neighbour inside the bin; reported, not
   corrected). `dismantling-run-execution-final.json`: 4 reported, **4
   completed, 0 failed**, mean 42.6 mm; `COMPLETE` 68 s after `RUN_BEGIN`.

`dismantling-run-tool-check.txt` is the standalone Isaac check (17 of 18; the
one expected failure is deleting a prim while playing): the visual tool has
no contact effect, a split at rest stays put, and
`dismantling_fixed_remainder_stays_installed` — the fixed segment is
kinematic and moved **0.00 mm** while the released one was carried past it.

## Known limitations

* Generated scene and a predefined dismantling request (the removable length
  is declared on the component); the planner chooses the alternative and
  validates it, it does not yet search cut positions along an installation.
* One installed component, one cut, one demonstrated geometry; the support
  frame is a simple rig, not a model of a plant.
* The finger closure on a kinematic body is a contact, not a weld: the
  released segment is held by the grasp weld only from `CUT_COMPLETED` on,
  exactly as in the loose-pipe cut.
* Frames come from the DemoCamera at about 4.4 frames per second.
