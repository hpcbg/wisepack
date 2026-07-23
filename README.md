# WISEPACK

**Intelligent Robotic Sorting and Volume-Optimized Packaging of Nuclear Waste
with Human-in-the-Loop AI**

A running software demonstrator of the WISEPACK novel contribution —
**geometry-aware container packaging with Digital Twin validation and operator
approval** — built as ROS 2 Jazzy modules over DDS with a FIWARE NGSI-LD audit
trail.

| | |
|---|---|
| **Scope** | Interview demonstrator, not the nine-month TRL6 implementation. See [Limitations](#16-limitations). |
| **Runtime** | ROS 2 Jazzy on Vulcanexus / Fast DDS, Orion-LD (NGSI-LD), FastAPI dashboard |
| **Host requirements** | Docker only. No host ROS 2, no host Python packages. A no-Docker mode also works. |
| **Licence** | MIT. Reuses **TEMPO**, **HARVEST** and **HARMONY** — see [NOTICE](NOTICE) and [§17](#17-attribution). |
| **Tests** | 186, all passing, runnable without ROS |

---

## 1. The problem

Decommissioning a nuclear plant generates large volumes of metallic waste —
tubes, bent pipes, flat and curved sheets, curved panels, I-beams — that must be
packed into certified containers for transport and storage at facilities with
limited capacity. Today that packing is done either manually, exposing operators
to radiation, or by robots that retrieve objects but do not optimise container
utilisation.

The reason this is hard is easy to see and easy to under-estimate: **a pipe is
mostly air**. In the generated scenarios here, a straight tube's bounding box is
typically 60–85% empty space. Where each pipe is placed, and at what
orientation, therefore decides how many certified containers a site buys, ships
and stores — and existing bin-picking systems have no model of container
occupancy at all.

Current industrial bin-picking returns a *grasp pose*. Academic bin-packing
optimises *known geometries* with no robot. WISEPACK closes that loop.

## 2. What this demonstrator proves

Everything in this list was executed and verified on this machine.

1. **Geometry-aware packing measurably beats a competent baseline.** On the dense
   40-pipe scenario: **3 containers → 2**, utilisation **39.6% → 59.3%**, a
   **33.3%** reduction in required container capacity. Both plans pass the same
   independent validator.
2. **Every placement is validated by code that did not produce it.** The
   validator re-derives each bounding box from the item and the recorded axis and
   checks nine hard constraints. It runs in a *separate ROS 2 process* that only
   ever sees JSON that crossed DDS.
3. **No physical action happens before a human approves.** Enforced structurally
   by the behaviour tree and independently asserted in the engine. Verified: with
   the plan pending, the workflow holds at `WAIT_FOR_OPERATOR_APPROVAL` and
   progress stays at 0%.
4. **A dynamic event triggers a real re-plan.** A high-priority component
   arriving mid-execution pauses the cycle, re-plans around the items already
   physically placed, and re-requests approval.
5. **Every workflow action reaches FIWARE over DDS.** Verified end to end: 226
   action events, a gap-free sequence, and the same sequence number readable back
   from Orion-LD as an NGSI-LD attribute.
6. **The command path is bidirectional and real.** An NGSI-LD `PATCH` of
   `urn:ngsi-ld:WISEPACKSystem:main/attrs/approval` returned HTTP 204 and moved
   the workflow from `WAIT_FOR_OPERATOR_APPROVAL` to executing. An external HMI
   can drive the approval with no WISEPACK-specific API.
7. **The demo runs with nothing installed.** `./run_wisepack_demo.sh --core-only`
   needs only Python 3.

## 3. What is simulated — read this before quoting any number

This is the most important section in the README.

| Component | Status | What that means |
|---|---|---|
| Packing algorithms | **REAL** | Two real algorithms on real geometry |
| Placement validator | **REAL** | Nine hard constraints, independent process |
| Container counts, utilisation, volume reduction | **MEASURED** | Computed, not asserted |
| Optimizer computation time | **MEASURED** | Wall clock on this machine |
| ROS 2 / DDS transport | **REAL** | Vulcanexus Fast DDS |
| FIWARE audit trail | **REAL** | Orion-LD, NGSI-LD, over DDS |
| DDS→FIWARE latency | **MEASURED** | When the benchmark has been run; `not measured` otherwise |
| Perception | **SIMULATED** | No camera, no RGB-D, no detector, no pose estimation |
| Robot | **SIMULATED** | No kinematics, no MoveIt2, no physics. A failed grasp is a seeded coin flip. |
| Pick / end-to-end success rate | **SIMULATED** | Reflects the configured failure probability, nothing else |
| Dose class | **SIMULATED METADATA** | There is no radiation model anywhere in this repository |
| 5 of 6 geometry classes | **APPROXIMATED** | Conservative bounding box — over-estimates space, never under-estimates |

Every figure in the dashboard, the artefacts and the reports carries a
`measured` / `simulated` / `operator` / `target` label. Nothing is unlabelled.

### Proposal KPIs are targets, not results

The WISEPACK proposal states four acceptance KPIs. Three of them **cannot be
measured by this demonstrator at all**, and are reported as `not_applicable`
rather than scored — a green tick on a simulated grasp rate would be fabrication.

| KPI | Proposal target | This demonstrator |
|---|---|---|
| KPI1 Vision detection rate | > 85% | `not_applicable` — no perception model exists |
| KPI2 Pick success rate | > 80% | `not_applicable` — no robot exists |
| KPI3 End-to-end success rate | > 80% | `not_applicable` — derived from simulated picks |
| KPI4 Volume reduction | > 50% | **`not_met`: 33.3% measured** on the dense scenario |

**KPI4 is not met by this demonstrator, and that is an honest and useful
finding.** Against a *competent* arrival-order baseline — one that puts an item
in whichever open container accepts it — the measured reduction is 33–50%
depending on scenario, and never exceeds 50%. Three reasons, all real:

- **Integer quantisation.** Container counts are small integers, so the
  achievable reductions are 2→1 (50%), 3→2 (33%), 4→2 (50%). To clear 50% you
  need something like 3→1, which requires the baseline to be genuinely
  incompetent.
- **The optimizer is already at the floor where it matters.** On the curated
  scenario it reaches a *single* container. It cannot do better than one.
- **Five of six geometry classes are bounding-box approximations**, which
  over-states their volume and understates what a real geometric model would
  achieve.

Reaching >50% is a legitimate objective for the full project — with exact
geometry for all six classes, cutting recommendations, and a baseline calibrated
against real EDF/CEA site practice rather than a textbook shelf packer. It is not
something this demonstrator can claim.

### Results from previous HPC work are not results of this demo

The proposal cites COROB (98–99% segmentation accuracy, sub-millimetre 6D pose)
and ARISE (multimodal HRI, Digital Twin, analytics). Those are **prior results
from other projects**. Nothing in this repository reproduces or evidences them.

## 4. The novel contribution

The proposal's primary innovation is *"a new module that generates and ranks
container-filling sequences using hybrid heuristic bin-packing algorithms, with
each candidate placement validated in a Digital Twin before physical
execution"*. That module is [`wisepack_core/packing.py`](wisepack_ws/src/wisepack_core/wisepack_core/packing.py)
and its independent checker is [`wisepack_core/validator.py`](wisepack_ws/src/wisepack_core/wisepack_core/validator.py).

Three things make it more than a bin-packer:

1. **The validator is genuinely independent.** It shares no candidate-generation
   code with the optimizer, re-derives every box from first principles, and runs
   in its own ROS 2 process seeing only serialised JSON. The test suite feeds it
   hand-built broken plans to prove it rejects them.
2. **Hard constraints are never penalties.** Container bounds, collision,
   payload, segregation, orientation and support are enforced by feasibility
   filtering. No weighting can trade them away.
3. **The optimizer is not allowed to win by default.** `select_plan()` scores
   both plans and *keeps the baseline* if the optimizer loses, reporting the
   fact. A demo that always announces an improvement is not evidence of one.

## 5. Architecture

Mission–Task–Skill, in the four layers the proposal describes.

![Architecture](images/generated/topology.svg)

```
 PERCEPTION      Task generator ......... deterministic, seeded
                 Perception simulator ... SIMULATED (extension point for
                                          YOLOv12-OBB + SAM2 + FPFH/ICP)
 OPTIMIZATION    Packing optimizer ...... geometry_aware_ep_bfd, 3 strategies
 + DIGITAL TWIN  Digital Twin validator . independent process, 9 hard constraints
 HITL            py_trees orchestrator .. the approval gate
                 Operator ............... approve / reject / alternative
                 Robot simulator ........ SIMULATED (extension point for MoveIt2)
 MIDDLEWARE      ROS 2 / DDS ............ Vulcanexus Fast DDS
 ANALYTICS       Orion-LD ............... NGSI-LD audit trail
                 Dashboard .............. FastAPI + offline SVG
```

### The one architectural decision worth explaining

**There is no `wisepack_interfaces` package.** The mandatory audit path is
Orion-LD's built-in DDS bridge, and that bridge maps *only* single-member scalar
`std_msgs` — it cannot represent a custom message at all. Custom ROS types would
have forced the audit trail off the DDS path, which the brief forbids. So rich
objects travel as versioned JSON inside `std_msgs/String`, and the typed domain
model lives in `wisepack_core` as plain Python dataclasses. HARMONY reached the
same conclusion independently; its generator skips every `custom_interfaces/*`
topic as "not representable".

A second consequence: `wisepack_core` imports **no ROS at all**. That is what
makes "the same logic in both modes" true rather than aspirational — sim mode is
this code with a different transport, not a re-implementation.

## 6. Quick start

### One command, full acceptance demonstration

```bash
./run_wisepack_demo.sh
```

Builds the image, builds the workspace, runs the tests, starts Orion-LD, starts
the ROS 2 nodes, plans, requests approval, executes, injects a dynamic event,
re-plans, verifies FIWARE, and writes artefacts. Roughly 10–15 minutes on first
run (image build dominates).

```bash
./run_wisepack_demo.sh --no-fiware   # skip Orion-LD
./run_wisepack_demo.sh --core-only   # pure Python: no Docker, no ROS, no FIWARE
```

### Interactive dashboard

```bash
./run_wisepack_dashboard.sh          # live ROS 2 / DDS
./run_wisepack_dashboard.sh fiware   # live + Orion-LD, state read back over NGSI-LD
./run_wisepack_dashboard.sh sim      # presentation only — no ROS, no FIWARE, no Docker
```

Then open <http://127.0.0.1:8080>.

### Individual validations

```bash
./run_vulcanexus_wisepack.sh validate_wisepack_e2e.sh
./run_vulcanexus_wisepack.sh validate_fiware_action_log.sh
./run_vulcanexus_wisepack.sh measure_dds_fiware_latency.sh
./generate_demo_artifacts.sh
python3 -m pytest tests/ -q
```

## 7. Dashboard walkthrough

![Optimized packing](images/generated/optimized-packing.svg)

1. **Header** — scenario, run id, workflow stage, **source badge**
   (`SIMULATED` / `ROS 2 / DDS` / `FIWARE`) and FIWARE connection state. The
   badge is the honesty contract: it always names where the data came from.
2. **Digital Twin** — top and side projections per container, drawn from real
   placement geometry. Switch Optimized / Baseline / Side-by-side. Colour is the
   segregation group; solid is executed, faded-dashed is planned, red-dashed is a
   validator rejection, orange outline is a dynamic-event item, and the orange
   dashed lines are the baseline's shelf plates — the thing that explains its
   wasted height.
3. **Operator panel** — Approve, Reject & re-plan, Alternative strategy, Inject
   item, Container unavailable, Grasp failure, Pause/Resume/Step. While a plan is
   pending the page states plainly that no physical action is authorised.
4. **Scenario controls** — preset, seed, item count, length/diameter ranges,
   container spec, pick failure probability, dynamic events on/off.
5. **Baseline vs optimized** — containers, utilisation, required capacity, empty
   capacity, unplaced items, computation time, validator verdict, and which plan
   was selected *and why*.
6. **KPI cards** — each with a `measured` / `simulated` provenance chip. An
   unmeasured KPI reads **"not measured"** in muted grey, never `0`.
7. **System topology** — live node status; solid arrows telemetry, dashed
   commands.
8. **Event timeline** — newest-first action log with sequence number, stage,
   item, container, result and source; dynamic events marked.

## 8. The two algorithms

![Baseline vs optimized](images/generated/comparison-mixed_pipes_dense.svg)

### Baseline — `arrival_order_shelf`

Items in arrival order, one fixed orientation, filled as shelves: left-to-right
along a row, rows front-to-back on a level, levels resting on shelf plates, and a
new container when the current one is full.

It is called `arrival_order_shelf` and **not** "manual industry average", because
no evidence for the latter exists in this repository.

It is deliberately simple but deliberately **not a strawman**. It will put an
item in whichever already-open container accepts it — which is what an operator
with several open boxes does. An earlier version only ever looked at the *last*
container and used 22 containers on the segregated scenario; that was an unfair
comparison and was fixed. What it lacks is optimization, not competence: no
sorting, no orientation search, no reconsideration.

Its levels rest on **zero-thickness shelf plates**. That is the assumption most
favourable to it — a real plate consumes height it is not charged for here — so
any margin the optimizer shows is understated, not inflated.

### Optimized — `geometry_aware_ep_bfd`

Best-fit-decreasing over **extreme points**:

- candidate positions maintained per container, seeded at the origin and extended
  by the three face projections of each placement;
- every permitted axis-aligned orientation evaluated at every candidate;
- feasibility filtered by the same hard-constraint rules the validator applies;
- placements ranked by a fit score dominated by **contact area** with walls and
  neighbours, with height as a secondary term and a corner bias as a
  deterministic tie-break;
- **deterministic seeded multi-start** over six orderings (priority, volume,
  length, diameter, group-major, plus seeded local perturbations);
- a **container-consolidation** improvement pass that empties the least-full
  container into the others, strictly improving or reverting;
- the best-scoring complete solution wins.

> A note on the fit score, since it is the one number that changed the result
> most: an earlier version weighted height twice as heavily as contact. That is
> correct for minimising an open stack and *wrong* for a fixed bin — it made the
> optimizer lay one flat layer and spill sideways into a second container instead
> of building upward in the first.

### Objective

```
score = packing_density
      − container_count_penalty
      − unplaced_volume_penalty
      − segregation_penalty
      − retrievability_penalty
      − excessive_clearance_penalty
```

**Hard constraints are not in this expression.** Three operator-selectable
strategies (`max_density`, `retrievability`, `segregation`) differ *only* in
these weights — same constraints, same search, same validator — which is exactly
the "compare packing strategies before execution" the proposal promises.

### The nine hard constraints

| | Constraint |
|---|---|
| H1 | every placement lies fully inside the container's inner volume |
| H2 | no two placements overlap (half-open intervals: touching is legal) |
| H3 | the container's payload limit is respected |
| H4 | the item's segregation group is accepted by the container |
| H5 | the orientation is one the item permits |
| H6 | the recorded bounding box matches the item and axis |
| H7 | each item placed at most once, and known to the scenario |
| H8 | no **new** placement targets an unavailable container |
| H9 | each placement is supported from below (floor, shelf plate, or ≥70% on other items) |

H6 catches the most dangerous bug class there is here: a packer writing a
shrunken box to make a placement "fit".

## 9. Task generator and scenarios

Deterministic: the same `(preset, seed)` produces byte-identical scenario JSON on
any machine. One seeded RNG, drawn in a fixed order, no set iteration, no salted
string hashing, values rounded at creation.

Measured results, all six presets, seed 42 — reproduce with
`./generate_demo_artifacts.sh`:

| Scenario | Baseline | Baseline util. | Optimized | Optimized util. | Volume requirement reduction |
|---|---|---|---|---|---|
| `mixed_pipes_small` | 2 | 30.1% | 1 | 60.1% | **50.0%** |
| `mixed_pipes_dense` | 3 | 39.6% | 2 | 59.3% | **33.3%** |
| `segregated_materials` | 3 | 25.0% | 3 | 25.0% | **0.0%** |
| `late_arrival_replan` | 2 | 32.7% | 1 | 65.5% | **50.0%** |
| `mixed_geometries` | 2 | 28.8% | 1 | 57.7% | **50.0%** |
| `curated_volume_reduction` | 2 | 33.6% | 1 | 67.1% | **50.0%** |

**`segregated_materials` returns 0%, and that is reported rather than hidden.**
Three segregation groups need three containers whichever algorithm packs them;
one box per group is already the floor. Geometry-aware packing cannot beat it,
and a demo that quietly dropped this scenario would be misrepresenting the
method's scope.

### The curated scenario, and why it is labelled

`curated_volume_reduction` is **hand-built to expose one specific weakness** of
shelf packing, and its construction is published in full:

- 20 items alternate long-thin (1500 × 100 mm) and short-fat (400 × 300 mm);
- the shelf baseline sets each level's height from the tallest item on it, so
  every 300 mm fat item wastes 200 mm for the thin items sharing its level, and
  it cannot rotate the long items to use the 700 mm depth;
- the optimizer sorts by volume, rotates freely and nests.

It contains **no random draw at all**, so the seed is irrelevant to it. The
50.0% it reports is what the two algorithms computed on this input — it is a
curated demonstration, **not a general performance claim**. The sizing was chosen
as the largest item count for which the optimizer still reaches a *single*
container (measured across 6/8/10/12/14/16 pairs and several geometry variants);
larger counts make the ratio worse, not better.

## 10. Human-in-the-Loop workflow

```
IDLE → GENERATE_OR_LOAD_SCENARIO → SCAN_SOURCE_BIN → DETECT_ITEMS
     → GENERATE_BASELINE_PLAN → GENERATE_OPTIMIZED_PLAN
     → DIGITAL_TWIN_VALIDATE → WAIT_FOR_OPERATOR_APPROVAL
     → (PICK_ITEM → VERIFY_PICK → PLACE_ITEM → VERIFY_PLACEMENT
        → UPDATE_CONTAINER_STATE → NEXT_ITEM | REPLAN)* → COMPLETE
```

A `py_trees` behaviour tree whose behaviours are **thin adapters** — each calls
exactly one `WorkflowEngine` method and translates the result into a
`py_trees.Status`. None re-implements planning, validation or the approval rule.

**The safety invariant is enforced twice.** `ExecuteLoop` sits behind
`AwaitApproval` in a `Sequence`, so the tree structurally cannot reach a pick
before approval; and `step_execution()` independently raises `ApprovalRequired`
if the plan is not approved. `AwaitApproval` never times out — a timeout would
mean "proceed because nobody answered".

If the orchestrator stops publishing its heartbeat, consumers see a DDS
deadline-missed event and hold. Degraded means **held**, never "carry on".

## 11. Dynamic events

![Dynamic re-planning](images/generated/dynamic-replan.svg)

Nine event types: `item_inject`, `item_removed`, `item_reclassified`,
`container_unavailable`, `container_restored`, `operator_reject`,
`grasp_failure`, `segregation_rule_change`, `optimizer_timeout`.

Triggers are `stage:<STAGE>`, `placement:<n>` or `t:<seconds>` — deliberately not
wall-clock, so a demonstration is identical on a fast laptop and a loaded CI
runner. (HARVEST times its events on a simulated clock, which suits a day-long
farm simulation and not a 30-second packing cycle.)

Re-planning **freezes already-executed placements** — those items are physically
in the container and cannot be moved — and re-optimises only the remainder.

A `grasp_failure` deliberately does **not** trigger a re-plan: it is a retry at
the execution layer. Re-planning a whole container because one grasp slipped
would be both wasteful and a misleading demonstration of what re-planning is for.

## 12. ROS 2 / DDS contract

25 topics, all scalar `std_msgs` (see [§5](#5-architecture)). Full list in
[`topics.py`](wisepack_ws/src/wisepack_bringup/wisepack_bringup/topics.py).

| Profile | Topics | Why |
|---|---|---|
| `event_qos` | action events, dynamic events | RELIABLE, depth 200 — a dropped event is a hole in a regulatory record |
| `state_qos` | scenario, plans, action sequence | RELIABLE + TRANSIENT_LOCAL — a late joiner must see current state |
| `heartbeat_qos` | execution state | latched **and** Deadline/Liveliness — silence must be *detectable* |
| `command_qos` | operator approval, operator command | RELIABLE + TRANSIENT_LOCAL, **no deadline** |
| `telemetry_qos` | progress, KPI values | BEST_EFFORT — freshness beats completeness |

> **`command_qos` carries no Deadline, and that was a real bug found by running
> the live stack.** A subscription that *requests* a Deadline only matches a
> publisher that *offers* one at least as short. Orion-LD's DDS bridge and
> `ros2 topic pub` both offer an infinite deadline, so requesting 2 s made them
> incompatible — and rclpy does not raise, it silently delivers nothing forever.
> Every node reported healthy while the entire FIWARE→ROS operator path was dead.
> There is now a regression test.

## 13. FIWARE data path

**This is the only audit path. There is deliberately no direct-HTTP fallback.**

```
WISEPACK node → ROS 2 topic → DDS → Orion-LD DDS bridge → NGSI-LD entity
```

Orion-LD runs `-wip dds -mongocOnly` and reads
`context_broker_config.json`, generated from
[`bridge_config.yaml`](wisepack_ws/src/wisepack_fiware/config/bridge_config.yaml)
by an adapted HARMONY generator. **No custom bridge node runs at all.**

| Entity | Attributes |
|---|---|
| `urn:ngsi-ld:WISEPACKSystem:main` | `stage`, `readiness`, `approval`◄, `command`◄ |
| `urn:ngsi-ld:WISEPACKScenario:current` | `summary`, `config`, `detectedCount` |
| `urn:ngsi-ld:WISEPACKPackingPlan:current` | `baseline`, `optimized`, `selected`, `status` |
| `urn:ngsi-ld:WISEPACKRobot:arm-01` | `currentItem`, `currentContainer`, `progressPct` |
| `urn:ngsi-ld:WISEPACKActionStream:main` | `actionJson`, `sequence`, `dynamicEvent` |
| `urn:ngsi-ld:WISEPACKKPI:current` | 8 KPI attributes |

◄ = inbound (FIWARE → ROS). Every value is read as `<attr>.value.data`.

### Verified behaviour, and its honest limitation

Measured on this machine (`mixed_pipes_small`, seed 42):

```
containersOptimized      REAL  1
volumeReductionPct       REAL  50
utilizationBaselinePct   REAL  30.0725
utilizationOptimizedPct  REAL  60.145
sequence                 REAL  97      (matching ROS 2)
actionJson               REAL  {"sequence":97,"stage":"WAIT_FOR_OPERATOR_APPROVAL",...}
```

**The bridge is state-oriented, and this repository does not pretend otherwise.**
`WISEPACKActionStream.actionJson` holds the **latest** action, not an append-only
log; `.sequence` holds the highest number reached. The append-only record is the
timestamped `results/wisepack-actions-*.jsonl` artefact. Historical retention in
FIWARE would come from QuantumLeap subscribing to these entities — that is
documented as an extension point, not implied as present.

### Bidirectional — verified

```bash
curl -X PATCH \
  'http://localhost:1026/ngsi-ld/v1/entities/urn:ngsi-ld:WISEPACKSystem:main/attrs/approval' \
  -H 'Content-Type: application/json' \
  -d '{"type":"Property","value":{"data":"APPROVE"}}'
# HTTP 204 → stage moved WAIT_FOR_OPERATOR_APPROVAL → NEXT_ITEM
```

A plain `"value": "APPROVE"` does **not** reach DDS. The nested `value.data`
shape is required.

### Vulcanexus is required, not preferred

The FIWARE DDS Enabler only fills a value once the publisher propagates the
`std_msgs` TypeObject. Plain ROS 2 Jazzy announces the topic — so Orion-LD
creates the entity — but does not reliably propagate the type, and every
attribute stays `"uninitialized"` forever with no error anywhere. The validation
scripts detect and explain this case rather than reporting a mysterious failure.

## 14. KPI definitions

**The denominator rule.** Volume reduction compares **required container
capacity** = (containers used × capacity each):

```
volume_requirement_reduction_pct =
    100 × (baseline_required_capacity − optimized_required_capacity)
        / baseline_required_capacity
```

Total pipe **material volume must never be the denominator**. It is identical for
both algorithms — nothing an optimizer does changes how much steel exists — so
using it produces a number that cannot distinguish a good plan from a bad one.
There is a test asserting exactly this.

Other definitions worth stating: `packing_density_gain_pct` is a **relative**
gain (30%→60% is +100%, not "+30"); absolute percentage points are reported
separately. Container counts use containers *actually holding a placement*. A
rate with zero attempts is **`not measured`**, never `0` — "no attempts yet" and
"0% success" are different statements.

Full list: `items_generated`, `items_packed`, `unplaced_items`,
`containers_baseline`, `containers_optimized`,
`container_utilization_{baseline,optimized}_pct`, `packing_density_gain_pct`,
`unused_capacity_reduction_pct`, `volume_requirement_reduction_pct`,
`optimization_time_ms`, `placements_validated`, `replans`,
`simulated_pick_attempts`, `simulated_pick_success_rate_pct`,
`simulated_end_to_end_success_rate_pct`, `operator_interventions`,
`fiware_events_logged`, `dds_to_fiware_latency_ms`.

## 15. Tests and evidence

```bash
python3 -m pytest tests/ -q        # 186 tests, no ROS required
```

| File | Covers |
|---|---|
| `test_generator.py` | determinism, dimension validity, segregation classes, JSON/CSV round trip |
| `test_validator.py` | every hard constraint H1–H9, hand-built violating plans, support-area union |
| `test_optimizer.py` | all placements validate, reproducibility, multi-container, honest selection, curated result computed not constant, speed |
| `test_kpi.py` | exact known cases, zero-baseline protection, the material-volume anti-fudge test, target labelling |
| `test_workflow.py` | approval gating, rejection→re-plan, frozen placements, dynamic events, audit-trail monotonicity |
| `test_ros_fiware.py` | reserved `status` leaf, bridgeable types, YAML↔contract agreement, generated mapping, QoS regression |

Artefacts written to `results/` per run, all timestamped: `wisepack-run-*.json`,
`wisepack-actions-*.{jsonl,csv}`, `wisepack-placements-*.csv`,
`wisepack-scenario-*.{json,csv}`, `wisepack-kpis-*.{json,csv}`,
`wisepack-validation-*.md`, `wisepack-fiware-validation-*.md`,
`wisepack-dds-fiware-latency-*.{json,csv}`.

Figures in `images/generated/` are produced by running the real pipeline —
`./generate_demo_artifacts.sh` — never drawn from constants.

## 16. Limitations

Stated plainly, because a demonstrator that hides its edges is not evidence.

1. **No perception.** No camera, RGB-D, detector or 6D pose estimation.
   Extension point: `wisepack_sim/perception_sim.py::detect()`.
2. **No robot.** No kinematics, no MoveIt2, no collision-free trajectories, no
   Isaac Sim. Extension point: `RobotSimConfig` and the execution stages.
3. **No radiation model.** `dose_class` is a label used to exercise priority and
   segregation machinery.
4. **Five of six geometry classes are bounding boxes.** Conservative, so plans
   are safe but pessimistic. Only the straight tube is exact.
5. **KPI4 is not met** — 33–50% measured, target >50%. See [§3](#3-what-is-simulated--read-this-before-quoting-any-number).
6. **Axis-aligned orientations only.** No arbitrary rotation, no curved-pipe
   collision geometry.
7. **FIWARE history is state-oriented.** No QuantumLeap/CrateDB/Grafana stack
   here; the append-only record is the JSONL artefact.
8. **No voice, gesture or gaze.** The proposal's multimodal HRI is represented by
   the dashboard command path only. (HARMONY has working Vosk/MediaPipe modules
   that plug into the same NGSI-LD attributes.)
9. **Single robot, single workstation.** No fleet coordination.
10. **Latency figures include HTTP polling.** The ROS→FIWARE number is an
    upper bound on true propagation delay, not a lower one.

## 17. From here to TRL6

| Step | Work | Maps to |
|---|---|---|
| 1 | Replace `perception_sim.detect()` with YOLOv12-OBB + SAM2 + FPFH/ICP on real RGB-D | O1, KPI1, MS2 |
| 2 | Exact geometry for the five approximated classes (convex decomposition or voxel masks) instead of bounding boxes | O3, KPI4 |
| 3 | Replace the robot simulator with MoveIt2 trajectory generation and a real arm | O2, KPI2/KPI3, MS4 |
| 4 | Isaac Sim as the Digital Twin backend, keeping the current validator as the independent checker | MS4 |
| 5 | Calibrate the baseline against real EDF/CEA site practice so KPI4 is measured against reality, not a textbook packer | KPI4, MS6 |
| 6 | QuantumLeap + CrateDB + Grafana on the existing NGSI-LD entities for true historical retention | O5, MS5 |
| 7 | Re-attach HARMONY's Vosk voice and MediaPipe gesture modules to the same operator attributes | O4 |
| 8 | Cutting recommendations from advisory to an optimizer input | O3 |

**The highest-value next step is (2).** It is the only one that directly moves
the measured KPI4 number, it needs no hardware, and the bounding-box
approximation is currently the largest known source of pessimism in the result.

## 18. Attribution

Full detail in [NOTICE](NOTICE).

| From | Reused | Form |
|---|---|---|
| **TEMPO** | topic/QoS contract modules, Docker-only operation, `lib_validate.sh`, `clean_dds_shm.sh`, FastAPI dashboard structure, position-free topology, "KPI from artefact / not measured" pattern, two-theme tokens | adapted code |
| **HARMONY** | `generate_config.py`, `docker-compose.dds.yml`, `bridge_config.yaml` structure, latency measurement method, `orion_classify` rule, py_trees task pattern, and the hard-won DDS knowledge (Vulcanexus requirement, reserved `status` leaf, `value.data` nesting) | adapted code + documented findings |
| **HARVEST** | seeded task generator pattern, YAML dynamic-event model, scenario comparison, map-style rendering | **concepts only — no code.** HARVEST ships no licence file, so everything was re-implemented from scratch. |

---

**High Performance Creators Ltd** · WISEPACK · JARVIS Open Call 2 · EDF Pilot
Topic #1
