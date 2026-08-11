"""Canonical WISEPACK topic contract.

Every node above the domain core speaks only these topics. The contract is
declared here once so the nodes, the dashboard observer, the FIWARE bridge
mapping and the validation scripts cannot drift apart — TEMPO's tempo_bringup
does the same thing for the same reason.

WHY EVERYTHING IS A SCALAR std_msgs
-----------------------------------
The mandatory audit path is ROS 2 -> DDS -> Orion-LD's built-in DDS bridge ->
NGSI-LD. That bridge maps types by DDS dynamic-type discovery and only handles
single-member scalar `std_msgs` (String/Bool/Int32/Int64/Float32/Float64); it
exposes each as `<attribute>.value.data`. A `wisepack_interfaces` package with
custom messages would therefore be *unbridgeable*, and the audit trail — the one
thing the proposal calls non-negotiable — would have to leave the DDS path.

So: rich objects travel as versioned JSON inside `std_msgs/String`, and the typed
domain model lives in `wisepack_core` as plain Python dataclasses rather than in
generated ROS types. This is a deliberate, evidence-based trade, not an omission.
(HARMONY reached the same conclusion: see its `generate_config.py`, which skips
every `custom_interfaces/*` topic as "not representable" on the DDS path.)

RESERVED LEAF `status`
----------------------
Orion-LD's DDS module reserves a topic whose final segment is exactly `status`
(it collides with ROS 2 actions' `status`/`GoalStatusArray`), and such a topic
never bridges. Every status-like topic here therefore ends in `_json` or another
distinct leaf, exactly as HARMONY resolved the same problem. The FIWARE
*attribute* can still be called `status`; only the ROS leaf must differ.

STATE VS WATCHDOG — WHY THERE IS A SEPARATE HEARTBEAT TOPIC
-----------------------------------------------------------
`/wisepack/execution/state` is ORDINARY LATCHED STATE. It carries no Deadline
and no Liveliness, so anything can subscribe to it and read the current stage.

Watchdog detection lives on its own topic, `/wisepack/system/heartbeat`, which
is the only place a Deadline/Liveliness pair is declared.

This split exists because of a real, measured failure. A subscription that
REQUESTS a liveliness lease only matches a publisher that OFFERS a lease at
least as short. Orion-LD's DDS bridge publishes on every mapped topic and offers
an INFINITE lease, so a dashboard requesting a 4 s lease on the state topic got:

    New publisher discovered on topic '/wisepack/execution/state', offering
    incompatible QoS. No messages will be received from it.
    Last incompatible policy: LIVELINESS

Rendering the current stage must never depend on watchdog policy matching. The
heartbeat topic is deliberately NOT mapped into FIWARE, so no external bridge
publishes on it and the watchdog subscription only ever meets the orchestrator.

SINGLE-WRITER DISCIPLINE
------------------------
Exactly one node publishes each topic. The orchestrator owns every workflow and
execution topic; the perception simulator owns the detection topics; the twin
validator owns the plan-status topic; the dashboard and Orion-LD own only the
operator command topics. Two writers on a RELIABLE topic with a Deadline would
make deadline events ambiguous, and the degraded-state logic depends on them
being unambiguous.
"""

# --- scenario and waste -------------------------------------------------------
SCENARIO_CONFIG = "/wisepack/scenario/config"          # String: JSON scenario config
SCENARIO_STATE = "/wisepack/scenario/state"            # String: JSON scenario summary
WASTE_ITEMS = "/wisepack/waste/items"                  # String: JSON item list
WASTE_DETECTED_COUNT = "/wisepack/waste/detected_count"  # Int32

# --- physical perception (OPTIONAL — only with a real perception source) ------
# The WISEPACK-DOMAIN view of whatever perception provider is running. They carry
# `wisepack_core.perception.ObservationBatch`: generic cylindrical objects with
# a pose, a confidence and detector provenance. Nothing that subscribes to them
# learns which provider produced them, and nothing has to parse a detector's own
# schema — that translation happens once, inside the provider itself (see
# `perception/providers/`).
#
# WHERE THEY ARE PUBLISHED FROM, and why it matters. The perception service runs
# on the HOST, because the host owns the camera, the GPU and torch — and it
# speaks HTTP only. It publishes no DDS. The ORCHESTRATOR fetches batches from it
# over HTTP and publishes these two topics from INSIDE the container, on the
# Vulcanexus / Fast DDS runtime the Orion-LD DDS Enabler is validated against.
#
# That is deliberate: publishing from the host would have required a second,
# unvalidated middleware installation there, and plain host ROS 2 is NOT
# equivalent to Vulcanexus for the TypeObject propagation the Enabler needs.
#
# KEPT OUT OF `all_topics()`, for the same reason the Isaac channel is: that
# function is the contract with a publisher in EVERY run, and these two have one
# only when `WISEPACK_PERCEPTION_SOURCE=camera`. Listing them there would make
# an ordinary simulated run look broken.
#
# SINGLE WRITER: the orchestrator. One authority over the observation, on one
# middleware, exactly as with every other WISEPACK topic.
PERCEPTION_STATUS = "/wisepack/perception/status_json"   # String: JSON health/status
PERCEPTION_OBJECTS = "/wisepack/perception/objects_json"  # String: JSON ObservationBatch

# --- planning -----------------------------------------------------------------
# Plan topics carry the COMPLETE PackingPlan.to_dict() — every placement's
# position, size, axis and validation status. A dashboard cannot draw a
# container from "59.3% utilization", and publishing only summaries here is why
# the live Digital Twin rendered empty. ~27 kB per plan for a 40-item scenario.
PLAN_BASELINE = "/wisepack/plan/baseline"              # String: FULL PackingPlan JSON
PLAN_OPTIMIZED = "/wisepack/plan/optimized"            # String: FULL PackingPlan JSON
PLAN_SELECTED = "/wisepack/plan/selected"              # String: FULL PackingPlan JSON
PLAN_STATUS = "/wisepack/plan/status_json"             # String: JSON validation verdict
                                                       #   ('status' leaf is reserved)
# Compact digest of all three plans plus the selection reason (~1 kB). This is
# what the FIWARE bridge maps: the audit value of a plan in NGSI-LD is its
# container count, utilization and validity, not 40 placement coordinates, and
# pushing 27 kB into an attribute on every re-plan is not an audit trail worth
# having. The full plans stay on DDS for the dashboard.
PLAN_SUMMARY = "/wisepack/plan/summary"                # String: compact JSON digest
# Structured strategy comparison (~1-2 kB — summaries only, no geometry). Latched
# so a dashboard attaching after the operator pressed Compare still sees the
# result. Stamped with the scenario revision it was computed against.
PLAN_STRATEGY_COMPARISON = "/wisepack/plan/strategy_comparison"  # String: JSON

# --- run correlation (which run a FIWARE projection describes) ----------------
# Orion-LD holds CURRENT STATE, not a log: an attribute keeps its last value
# across scenario changes, process restarts and whole runs. Reading it back gives
# no way to tell a value written seconds ago from one written by the previous
# run — and a KPI attribute from `isaac_cylinders_smoke` rendered beside a
# `mixed_pipes_dense` Digital Twin is what that looks like on screen.
#
# One correlation topic per FIWARE entity that carries current state, each
# mapped to that entity's `runCorrelation` attribute. Payload:
# `wisepack_core.correlation.RunCorrelation`.
#
# PUBLISHED AFTER the values they stamp — see correlation.py. A reader polling
# mid-update then sees the OLD stamp with some new values and withholds the
# entity, rather than the NEW stamp with some old values and trusting them.
#
# Latched, because a projection's identity must be readable by a consumer that
# attaches long after the run started.
CORRELATION_SYSTEM = "/wisepack/correlation/system"        # String: JSON
CORRELATION_SCENARIO = "/wisepack/correlation/scenario"    # String: JSON
CORRELATION_PLAN = "/wisepack/correlation/plan"            # String: JSON
CORRELATION_KPI = "/wisepack/correlation/kpi"              # String: JSON
CORRELATION_ROBOT = "/wisepack/correlation/robot"          # String: JSON
CORRELATION_ACTIONS = "/wisepack/correlation/actions"      # String: JSON
CORRELATION_ANOMALY = "/wisepack/correlation/anomaly"      # String: JSON
CORRELATION_INVENTORY = "/wisepack/correlation/inventory"  # String: JSON
CORRELATION_CUTTING = "/wisepack/correlation/cutting"      # String: JSON

#: short entity name -> correlation topic. The dashboard and the orchestrator
#: both iterate this, so an entity cannot gain a projection without gaining a
#: stamp.
CORRELATION_TOPICS = {
    "system": CORRELATION_SYSTEM,
    "scenario": CORRELATION_SCENARIO,
    "plan": CORRELATION_PLAN,
    "kpi": CORRELATION_KPI,
    "robot": CORRELATION_ROBOT,
    "actions": CORRELATION_ACTIONS,
    "anomaly": CORRELATION_ANOMALY,
    "inventory": CORRELATION_INVENTORY,
    "cutting": CORRELATION_CUTTING,
}

# --- operator (FIWARE/dashboard -> orchestrator) ------------------------------
OPERATOR_APPROVAL = "/wisepack/operator/approval"      # String: APPROVE | REJECT
OPERATOR_COMMAND = "/wisepack/operator/command"        # String: JSON {command, args}

# --- execution ----------------------------------------------------------------
EXECUTION_STATE = "/wisepack/execution/state"          # String: workflow stage name
EXECUTION_CURRENT_ITEM = "/wisepack/execution/current_item"        # String
EXECUTION_CURRENT_CONTAINER = "/wisepack/execution/current_container"  # String
EXECUTION_PROGRESS_PCT = "/wisepack/execution/progress_pct"        # Float32
EXECUTION_BACKEND = "/wisepack/execution/backend"      # String: simulated | isaac
SYSTEM_READINESS = "/wisepack/system/readiness"        # Bool: ready to execute
SYSTEM_HEARTBEAT = "/wisepack/system/heartbeat"        # Int32: monotonic tick

# --- Isaac Sim execution backend (OPTIONAL transport) -------------------------
# These two are NOT part of `all_topics()`, and that omission is deliberate.
#
# `all_topics()` is the contract that is ALWAYS present: every entry has a
# publisher in every run, which is what lets the live QoS test assert that a
# topic with no publisher is a bug. The Isaac channel is different in kind — it
# exists only when `execution_backend:=isaac`, and its feedback topic is written
# by a process (Isaac Sim, on the host, in its own bundled Python) that is absent
# from a normal run. Listing it in the always-present contract would make the
# ordinary simulated run look broken.
#
# The FIWARE bridge does not map either topic. What belongs in the audit trail is
# the physical OUTCOME, and that already travels on /wisepack/action/event as an
# ActionEvent with actor `isaac_sim` — the same record every other actor uses.
# Mapping the raw transport as well would put the same fact in the trail twice,
# in two different shapes.
#
# Payload: a versioned JSON document defined once in
# `wisepack_core.isaac_contract` and imported by BOTH ends, so the orchestrator
# (Vulcanexus interpreter, in Docker) and the simulator (Isaac's bundled
# interpreter, on the host) cannot drift apart.
ISAAC_COMMAND = "/wisepack/isaac/command"        # String: JSON IsaacCommand
ISAAC_FEEDBACK = "/wisepack/isaac/feedback"      # String: JSON IsaacFeedback

# --- events -------------------------------------------------------------------
ACTION_EVENT = "/wisepack/action/event"                # String: JSON ActionEvent
ACTION_SEQUENCE = "/wisepack/action/sequence"          # Int32: monotonic counter
DYNAMIC_EVENT = "/wisepack/dynamic_event"              # String: JSON DynamicEvent

# --- anomaly monitoring & workflow response ----------------------------------
# A SIMULATED Topic #2-compatible anomaly stream. Latched so a dashboard
# attaching mid-run sees the current anomaly state; the orchestrator reacts
# deterministically (see hitl_orchestrator). NOT a validated detector.
# The SYSTEM'S recorded anomaly stream. Single writer: the orchestrator. Mapped
# to FIWARE. The orchestrator publishes here after ingesting an event (from the
# external seam below or an operator injection) and reacting to it.
ANOMALY_EVENT = "/wisepack/anomaly/event"              # String: JSON AnomalyEvent
# The anomaly snapshot for the dashboard panel. Single writer: the orchestrator.
ANOMALY_STATE = "/wisepack/anomaly/state"              # String: JSON anomaly snapshot
# The INGEST SEAM. A future external EDF Topic #2 detector (or the bundled
# simulator/adapter) publishes structured events here; the orchestrator
# subscribes, reacts, and republishes on ANOMALY_EVENT. Not mapped to FIWARE —
# it is a raw input, not the recorded stream.
ANOMALY_EXTERNAL = "/wisepack/anomaly/external"        # String: JSON AnomalyEvent

# --- cutting (SIMULATED cut-aware planning + external cutting skill) -----------
# Versioned JSON on scalar std_msgs, same bridge discipline as everything else.
# The physical cutting controller is an EXTERNAL FUTURE SKILL; here CUTTING_STATE
# is driven by the simulated cutting node. No leaf is the reserved `status`.
CUTTING_PROPOSAL = "/wisepack/cutting/proposal"        # String: JSON CutProposal(s)
CUTTING_APPROVAL = "/wisepack/cutting/approval"        # String: APPROVE_CUT|REJECT_CUT
CUTTING_REQUEST = "/wisepack/cutting/request"          # String: JSON cut request
CUTTING_STATE = "/wisepack/cutting/state"              # String: JSON cut skill state
CUTTING_RESULT = "/wisepack/cutting/result"            # String: JSON CutResult

# --- inventory (FIWARE-backed operational container inventory) -----------------
INVENTORY_CONTAINER_STATE = "/wisepack/inventory/container_state"    # String: JSON
INVENTORY_CONTAINER_EVENT = "/wisepack/inventory/container_event"    # String: JSON
INVENTORY_RESERVATION = "/wisepack/inventory/reservation"           # String: JSON
INVENTORY_REQUEST = "/wisepack/inventory/request"                   # String: JSON
INVENTORY_SUMMARY = "/wisepack/inventory/summary"                   # String: JSON

# --- logistics (SIMULATED container transport — no physical mobile robot) ------
LOGISTICS_CONTAINER_TASK = "/wisepack/logistics/container_task"          # String: JSON
LOGISTICS_CONTAINER_TASK_STATE = "/wisepack/logistics/container_task_state"  # String
LOGISTICS_MOBILE_ROBOT_STATE = "/wisepack/logistics/mobile_robot_state"  # String: JSON

# --- KPIs ---------------------------------------------------------------------
KPI_CONTAINERS_BASELINE = "/wisepack/kpi/containers_baseline"        # Int32
KPI_CONTAINERS_OPTIMIZED = "/wisepack/kpi/containers_optimized"      # Int32
KPI_UTILIZATION_BASELINE_PCT = "/wisepack/kpi/utilization_baseline_pct"    # Float32
KPI_UTILIZATION_OPTIMIZED_PCT = "/wisepack/kpi/utilization_optimized_pct"  # Float32
KPI_VOLUME_REDUCTION_PCT = "/wisepack/kpi/volume_reduction_pct"      # Float32
KPI_OPTIMIZATION_MS = "/wisepack/kpi/optimization_ms"                # Float32
KPI_PICK_SUCCESS_PCT = "/wisepack/kpi/pick_success_pct"              # Float32
KPI_END_TO_END_SUCCESS_PCT = "/wisepack/kpi/end_to_end_success_pct"  # Float32

# --- operator command vocabulary ---------------------------------------------
APPROVE = "APPROVE"
REJECT = "REJECT"

#: Commands accepted on OPERATOR_COMMAND, as {"command": ..., "args": {...}}.
#: This tuple is the contract the dashboard, the orchestrator and the tests all
#: check against, so a command advertised in the UI but unimplemented in the
#: orchestrator is a test failure rather than a dead button.
OPERATOR_COMMANDS = (
    "approve",                  # approve the validated plan; execution may start
    "reject",                   # reject + re-plan; returns to the approval gate
    "alternative_strategy",     # re-plan with another objective weighting
    "inject_item",              # late-arriving component -> re-plan
    "container_unavailable",    # retire a container -> re-plan
    "grasp_failure",            # force one deterministic simulated grasp failure
    "pause",                    # stop automatic execution, plan stays approved
    "resume",                   # resume automatic execution (approved plans only)
    "step",                     # execute exactly one workflow step
    "reset",                    # new run from the supplied scenario settings
    "write_artifacts",          # write run/plan/KPI/event artefacts now
    "compare_strategies",       # decision support: run + validate every strategy
    "inject_anomaly",           # SIMULATED Topic #2 anomaly (deterministic)
    "acknowledge_anomaly",      # operator acknowledges the held anomaly
    # -- object source (§6). WHERE THE NEXT RUN'S OBJECTS COME FROM. A per-run
    # selection, not an application mode: `set_object_source` edits the draft
    # and touches nothing that is running, and the next `reset` /
    # `detect_physical_objects` makes it authoritative. Rejected with a reason
    # when the requested source is not available, rather than quietly falling
    # back to the other one.
    "set_object_source",        # choose preset|camera for the NEXT run (draft)
    "detect_physical_objects",  # acquire a batch from the camera -> new run/revision
    # SUBMIT A BATCH SOMEBODY ELSE ACQUIRED. `detect_physical_objects` asks the
    # orchestrator to PULL from the perception service; this one PUSHES a batch
    # that has already been measured — the RGB-D path runs its estimator where
    # the camera and the CAD meshes are, and the orchestrator has neither.
    #
    # GENERIC ON PURPOSE. The payload is a serialised ObservationBatch and
    # nothing more: no method, model or geometry is named in the command. The
    # orchestrator adopts it exactly as it adopts a pulled one, so a future
    # detector needs no new command and this one carries no knowledge of which
    # detector produced what.
    "submit_observation_batch",  # adopt an externally measured batch -> revision
    # -- cut-aware HITL controls (brief §6) --
    "compare_cut_aware",        # generate the no-cut vs cut-aware comparison
    "select_cut_alternative",   # choose a specific cut alternative
    "limit_cuts",               # cap the cuts per plan, then re-plan
    "set_min_segment",          # raise the minimum segment length, then re-plan
    "prefer_no_cut",            # bias the planner against cutting
    "approve_cut",              # SEPARATE cut approval (not packing approval)
    "reject_cut",               # decline cutting; keep the pipe whole
    "simulate_cut",             # drive the simulated cutting skill to COMPLETED
    "simulate_cut_failure",     # drive the simulated cutting skill to FAILED
    # -- inventory + logistics controls (brief §13) --
    "init_inventory",           # initialise the simulated container inventory
    "check_containers",         # make the current plan inventory-aware
    "reserve_container",        # reserve a container for the plan
    "release_container",        # release a reservation
    "request_delivery",         # request delivery of a container to the cell
    "mark_container_unavailable",  # temporarily withdraw a container
    "restore_container",        # restore a temporarily unavailable container
    "mark_container_full",      # mark a container full
    "collect_full_containers",  # request collection + dispatch of full containers
)


def all_topics() -> dict:
    """topic -> message type. Used by the FIWARE mapping generator and the tests.

    Types are the ROS 2 names the DDS bridge understands. Keeping this mapping in
    the contract module (rather than duplicating it in bridge_config.yaml) is what
    lets a test assert the two agree.
    """
    return {
        # Which run each FIWARE projection describes — one per stamped entity.
        **{topic: "std_msgs/String" for topic in CORRELATION_TOPICS.values()},
        SCENARIO_CONFIG: "std_msgs/String",
        SCENARIO_STATE: "std_msgs/String",
        WASTE_ITEMS: "std_msgs/String",
        WASTE_DETECTED_COUNT: "std_msgs/Int32",
        PLAN_BASELINE: "std_msgs/String",
        PLAN_OPTIMIZED: "std_msgs/String",
        PLAN_SELECTED: "std_msgs/String",
        PLAN_STATUS: "std_msgs/String",
        PLAN_SUMMARY: "std_msgs/String",
        PLAN_STRATEGY_COMPARISON: "std_msgs/String",
        OPERATOR_APPROVAL: "std_msgs/String",
        OPERATOR_COMMAND: "std_msgs/String",
        EXECUTION_STATE: "std_msgs/String",
        EXECUTION_CURRENT_ITEM: "std_msgs/String",
        EXECUTION_CURRENT_CONTAINER: "std_msgs/String",
        EXECUTION_PROGRESS_PCT: "std_msgs/Float32",
        EXECUTION_BACKEND: "std_msgs/String",
        SYSTEM_READINESS: "std_msgs/Bool",
        SYSTEM_HEARTBEAT: "std_msgs/Int32",
        ACTION_EVENT: "std_msgs/String",
        ACTION_SEQUENCE: "std_msgs/Int32",
        DYNAMIC_EVENT: "std_msgs/String",
        ANOMALY_EVENT: "std_msgs/String",
        ANOMALY_STATE: "std_msgs/String",
        ANOMALY_EXTERNAL: "std_msgs/String",
        CUTTING_PROPOSAL: "std_msgs/String",
        CUTTING_APPROVAL: "std_msgs/String",
        CUTTING_REQUEST: "std_msgs/String",
        CUTTING_STATE: "std_msgs/String",
        CUTTING_RESULT: "std_msgs/String",
        INVENTORY_CONTAINER_STATE: "std_msgs/String",
        INVENTORY_CONTAINER_EVENT: "std_msgs/String",
        INVENTORY_RESERVATION: "std_msgs/String",
        INVENTORY_REQUEST: "std_msgs/String",
        INVENTORY_SUMMARY: "std_msgs/String",
        LOGISTICS_CONTAINER_TASK: "std_msgs/String",
        LOGISTICS_CONTAINER_TASK_STATE: "std_msgs/String",
        LOGISTICS_MOBILE_ROBOT_STATE: "std_msgs/String",
        KPI_CONTAINERS_BASELINE: "std_msgs/Int32",
        KPI_CONTAINERS_OPTIMIZED: "std_msgs/Int32",
        KPI_UTILIZATION_BASELINE_PCT: "std_msgs/Float32",
        KPI_UTILIZATION_OPTIMIZED_PCT: "std_msgs/Float32",
        KPI_VOLUME_REDUCTION_PCT: "std_msgs/Float32",
        KPI_OPTIMIZATION_MS: "std_msgs/Float32",
        KPI_PICK_SUCCESS_PCT: "std_msgs/Float32",
        KPI_END_TO_END_SUCCESS_PCT: "std_msgs/Float32",
    }


#: Topics Orion-LD writes INTO ROS (dashboard/HMI -> orchestrator). Everything
#: else flows ROS -> Orion-LD.
INBOUND_TOPICS = (OPERATOR_APPROVAL, OPERATOR_COMMAND, CUTTING_APPROVAL,
                  INVENTORY_REQUEST)


def perception_topics() -> dict:
    """The optional physical-perception channel: topic -> message type.

    Kept out of ``all_topics()`` on purpose (see the declarations above), and
    exposed separately so the tests can assert the same two invariants every
    WISEPACK topic must satisfy — standard scalar ``std_msgs`` only, and no
    reserved ``status`` leaf — without claiming the channel is always present.
    """
    return {
        PERCEPTION_STATUS: "std_msgs/String",
        PERCEPTION_OBJECTS: "std_msgs/String",
    }


#: Single-writer discipline across the perception channel. Stated as data so a
#: test can check it rather than trusting the comment.
PERCEPTION_WRITERS = {
    PERCEPTION_STATUS: "wisepack_orchestration",
    PERCEPTION_OBJECTS: "wisepack_orchestration",
}


def isaac_topics() -> dict:
    """The optional Isaac execution-backend channel: topic -> message type.

    Kept out of ``all_topics()`` on purpose (see the declarations above). Exposed
    as its own function so the tests can assert the same two properties that
    matter for every WISEPACK topic — standard scalar ``std_msgs`` only, and no
    reserved ``status`` leaf — without claiming the channel is always present.
    """
    return {
        ISAAC_COMMAND: "std_msgs/String",
        ISAAC_FEEDBACK: "std_msgs/String",
    }


#: Single-writer discipline across the Isaac channel: WISEPACK writes commands,
#: Isaac writes feedback, and neither ever writes the other's topic. Stated as
#: data so a test can check it rather than trusting the comment.
ISAAC_WRITERS = {
    ISAAC_COMMAND: "wisepack_orchestration",
    ISAAC_FEEDBACK: "isaac_sim",
}


def reserved_leaf_violations() -> list:
    """Topics whose final segment is the reserved `status` leaf.

    Must always be empty. A topic ending in `/status` is silently dropped by
    Orion-LD's DDS module, so it would vanish from the audit trail without any
    error anywhere — exactly the kind of failure a contract test should catch.

    The Isaac channel is included even though it is not bridged: the rule is
    cheap to keep and an unbridgeable topic name is a trap for whoever decides
    to bridge it later.
    """
    return [t for t in {**all_topics(), **isaac_topics(), **perception_topics()}
            if t.rsplit("/", 1)[-1] == "status"]
