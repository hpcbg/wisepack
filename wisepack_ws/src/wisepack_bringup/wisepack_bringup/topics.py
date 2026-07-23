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

# --- operator (FIWARE/dashboard -> orchestrator) ------------------------------
OPERATOR_APPROVAL = "/wisepack/operator/approval"      # String: APPROVE | REJECT
OPERATOR_COMMAND = "/wisepack/operator/command"        # String: JSON {command, args}

# --- execution ----------------------------------------------------------------
EXECUTION_STATE = "/wisepack/execution/state"          # String: workflow stage name
EXECUTION_CURRENT_ITEM = "/wisepack/execution/current_item"        # String
EXECUTION_CURRENT_CONTAINER = "/wisepack/execution/current_container"  # String
EXECUTION_PROGRESS_PCT = "/wisepack/execution/progress_pct"        # Float32
SYSTEM_READINESS = "/wisepack/system/readiness"        # Bool: ready to execute
SYSTEM_HEARTBEAT = "/wisepack/system/heartbeat"        # Int32: monotonic tick

# --- events -------------------------------------------------------------------
ACTION_EVENT = "/wisepack/action/event"                # String: JSON ActionEvent
ACTION_SEQUENCE = "/wisepack/action/sequence"          # Int32: monotonic counter
DYNAMIC_EVENT = "/wisepack/dynamic_event"              # String: JSON DynamicEvent

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
)


def all_topics() -> dict:
    """topic -> message type. Used by the FIWARE mapping generator and the tests.

    Types are the ROS 2 names the DDS bridge understands. Keeping this mapping in
    the contract module (rather than duplicating it in bridge_config.yaml) is what
    lets a test assert the two agree.
    """
    return {
        SCENARIO_CONFIG: "std_msgs/String",
        SCENARIO_STATE: "std_msgs/String",
        WASTE_ITEMS: "std_msgs/String",
        WASTE_DETECTED_COUNT: "std_msgs/Int32",
        PLAN_BASELINE: "std_msgs/String",
        PLAN_OPTIMIZED: "std_msgs/String",
        PLAN_SELECTED: "std_msgs/String",
        PLAN_STATUS: "std_msgs/String",
        PLAN_SUMMARY: "std_msgs/String",
        OPERATOR_APPROVAL: "std_msgs/String",
        OPERATOR_COMMAND: "std_msgs/String",
        EXECUTION_STATE: "std_msgs/String",
        EXECUTION_CURRENT_ITEM: "std_msgs/String",
        EXECUTION_CURRENT_CONTAINER: "std_msgs/String",
        EXECUTION_PROGRESS_PCT: "std_msgs/Float32",
        SYSTEM_READINESS: "std_msgs/Bool",
        SYSTEM_HEARTBEAT: "std_msgs/Int32",
        ACTION_EVENT: "std_msgs/String",
        ACTION_SEQUENCE: "std_msgs/Int32",
        DYNAMIC_EVENT: "std_msgs/String",
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
INBOUND_TOPICS = (OPERATOR_APPROVAL, OPERATOR_COMMAND)


def reserved_leaf_violations() -> list:
    """Topics whose final segment is the reserved `status` leaf.

    Must always be empty. A topic ending in `/status` is silently dropped by
    Orion-LD's DDS module, so it would vanish from the audit trail without any
    error anywhere — exactly the kind of failure a contract test should catch.
    """
    return [t for t in all_topics() if t.rsplit("/", 1)[-1] == "status"]
