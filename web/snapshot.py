"""Unified dashboard read model.

ONE SCHEMA, THREE SOURCES. Every `/api/*` endpoint renders from a
`DashboardSnapshot` produced by one of three providers, and the schema is
identical whichever provider built it. The frontend therefore contains no
per-mode branching: it draws whatever the snapshot says.

    DashboardSnapshotProvider          the shared contract
      SimSnapshotProvider              reads STATE.engine       (no ROS)
      RosSnapshotProvider              reads STATE.ros_mirror   (live topics)
      FiwareSnapshotProvider           reads Orion-LD, falls back to ROS

WHY THIS EXISTS. The previous dashboard read `STATE.engine` unconditionally.
In live modes there is no engine in this process — the orchestrator owns it in
another one — so every panel rendered empty while the header cheerfully reported
"connected". A mode is not working because its process started; it is working
when its data is on screen.

PROVENANCE IS PER PANEL, NOT PER PAGE. FiwareSnapshotProvider genuinely reads
some panels from Orion-LD and others from ROS, so each section carries its own
`source` and the header aggregates them honestly: it says `FIWARE + ROS` rather
than `FIWARE` when the plans and events came over DDS.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from wisepack_core.correlation import RunCorrelation, describe_mismatch

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

#: Seconds without a new heartbeat before the orchestrator is declared lost.
#: Mirrors wisepack_bringup.qos.HEARTBEAT_STALE_S, duplicated here so the
#: dashboard imports nothing from ROS in sim mode.
HEARTBEAT_STALE_S = 6.0

#: Panels whose provenance is tracked independently.
PANELS = ("state", "scenario", "plans", "kpis", "events", "analytics")

#: The KPI keys the dashboard tiles ask for. A provider that cannot supply one
#: returns None for it — never 0, and never a missing key, because the frontend
#: renders a missing key and a measured zero identically otherwise.
KPI_KEYS = (
    "containers_baseline", "containers_optimized",
    "container_utilization_baseline_pct", "container_utilization_optimized_pct",
    "volume_requirement_reduction_pct", "packing_density_gain_pct",
    "optimization_time_ms", "simulated_pick_success_rate_pct",
    "simulated_end_to_end_success_rate_pct", "replans",
    "operator_interventions", "action_events_published",
    "dds_to_fiware_latency_ms",
)

#: Provenance of each KPI key, so live modes label tiles the same way sim does.
KPI_SOURCES = {
    "simulated_pick_success_rate_pct": "simulated",
    "simulated_end_to_end_success_rate_pct": "simulated",
    "detection_rate_pct": "simulated",
}


def _public_robot(robot_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """The PUBLIC-SAFE description of one robot, or None.

    Public-safe means what ``RobotProfile.to_public_dict`` publishes: identity,
    capability and status. Asset URLs and prim paths never leave the process
    through this path — they are of no use to an operator choosing an arm, and
    an asset-server path in a web response is free reconnaissance.

    Returns None for an empty id (a logical run has no robot) and for an id the
    registry does not know, rather than raising: a dashboard poll must not turn
    a stale robot id into a 500 across the whole page.
    """
    if not robot_id:
        return None
    try:
        from wisepack_core.robots import load_registry           # noqa: PLC0415
        return load_registry().profiles[str(robot_id).lower()].to_public_dict()
    except Exception:                                            # noqa: BLE001
        return {"id": str(robot_id), "display_name": str(robot_id),
                "unknown": True}


def _metric(key: str, value: Optional[float], unit: str = "",
            source: Optional[str] = None, note: str = "") -> Dict[str, Any]:
    return {
        "key": key,
        "value": value,
        "unit": unit,
        "source": source or KPI_SOURCES.get(key, "measured"),
        "measured": value is not None,
        "note": note,
    }


@dataclass
class DashboardSnapshot:
    """Everything the dashboard needs, in one shape, from any source."""

    mode: str                                   # sim | ros | fiware
    # Per-panel provenance: panel -> "sim" | "ros" | "fiware"
    panel_sources: Dict[str, str] = field(default_factory=dict)

    # -- state ------------------------------------------------------------- #
    stage: str = "IDLE"
    run_id: Optional[str] = None
    cycle_id: Optional[str] = None
    finished: bool = False
    degraded_reason: str = ""
    robot_state: str = "idle"
    current_item_id: Optional[str] = None
    current_container_id: Optional[str] = None
    progress_pct: float = 0.0
    approval_state: str = "pending"
    readiness: bool = False
    auto_step: bool = False

    # -- content ----------------------------------------------------------- #
    scenario: Optional[Dict[str, Any]] = None
    baseline: Optional[Dict[str, Any]] = None
    optimized: Optional[Dict[str, Any]] = None
    selected: Optional[Dict[str, Any]] = None
    selected_plan_id: Optional[str] = None
    scenario_revision: int = 0
    strategy_comparison: Optional[Dict[str, Any]] = None
    anomaly: Optional[Dict[str, Any]] = None
    whole_process: Optional[Dict[str, Any]] = None
    selection_reason: str = ""
    plan_status: Optional[Dict[str, Any]] = None
    detected_count: int = 0
    kpis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    target_assessment: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    action_count: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)
    dynamic_events: List[Dict[str, Any]] = field(default_factory=list)

    # -- execution backend --------------------------------------------------- #
    # WHO EXECUTED, which is a different question from where this snapshot was
    # READ (that is `mode`). Both are shown, separately, and neither is inferred
    # from the other — labelling a simulated run as physical, or the reverse, is
    # the one mistake this panel exists to prevent.
    execution_backend: str = "simulated"
    execution_backend_label: str = "SIMULATED EXECUTION"
    execution_backend_detail: str = ""
    # -- CONTROL STATE: canonical ROS/DDS, never overwritten by FIWARE ------- #
    #
    # THE SPLIT THAT FIXES OPERATOR LATENCY. `stage`/`approval_state` above are
    # the DISPLAY/AUDIT view, and in FIWARE mode they are deliberately the value
    # Orion-LD echoed back — that echo is the proof the audit path works, and it
    # is what the source badge is about.
    #
    # But an operator control must not wait for that echo. The canonical
    # workflow already reached WAIT_FOR_OPERATOR_APPROVAL on ROS/DDS; making
    # "Approve" wait for Orion to repeat it back adds the whole DDS->NGSI-LD
    # bridge latency plus a dashboard poll to a safety-relevant button, for no
    # safety benefit — the authority is the orchestrator either way, and it
    # re-checks the invariant itself when the command arrives.
    #
    # So: enablement reads these fields, the badge and the audit trail read the
    # ones above, and FIWARE is never bypassed for traceability.
    control_stage: str = "IDLE"
    control_approval_state: str = "pending"
    control_plan_id: Optional[str] = None
    control_plan_valid: bool = False
    control_finished: bool = False
    #: An anomaly hold blocks EXECUTION, so it must also block the button that
    #: authorises execution. Carried in the control block for the same reason as
    #: everything else here: it is canonical ROS/DDS state, and a stale FIWARE
    #: echo must not decide whether a safety hold is in force.
    control_anomaly_hold: bool = False
    control_anomaly_ack_required: bool = False
    #: Physical-backend scene gate. None when the backend has no scene concept
    #: (the simulated backend), so it can never block anything there.
    control_scene_ready: Optional[bool] = None
    control_scene_reason: str = ""
    #: The identity every component on screen must agree on. Populated from the
    #: per-topic stamps; "" means they agree.
    control_inconsistency: str = ""
    control_identity: Dict[str, Any] = field(default_factory=dict)
    control_approval_revision: Optional[int] = None
    control_approval_plan_id: Optional[str] = None
    #: FIWARE run-correlation verdict. `fiware_sync_status` is one of
    #: synchronized | partial | stale | unknown | disconnected.
    fiware_sync_status: str = "unknown"
    fiware_sync_detail: str = ""
    fiware_run_id: Optional[str] = None
    fiware_scenario_revision: Optional[int] = None
    fiware_stage: Optional[str] = None
    #: Every field NOT applied because its entity described another run.
    #: Each entry: {entity, field, reason}. Diagnostics renders this verbatim.
    rejected_stale_fields: List[Dict[str, Any]] = field(default_factory=list)
    execution_backend_known: bool = False
    #: THE ACTIVE ROBOT — the one the RUNNING run is executing with, published
    #: by the orchestrator. Never the dashboard's draft: those are different
    #: things and the whole "Running now / Next" distinction depends on the
    #: dashboard never confusing them. None in the logical modes, which have no
    #: robot at all and must not pretend to.
    active_robot: Optional[Dict[str, Any]] = None
    active_robot_id: Optional[str] = None
    isaac: Optional[Dict[str, Any]] = None
    isaac_results: List[Dict[str, Any]] = field(default_factory=list)
    #: Backend-neutral visualization descriptor, as published by whichever
    #: backend is executing. None until one is heard from. The dashboard reads
    #: ONLY this — it never learns which simulator produced it.
    visualization: Optional[Dict[str, Any]] = None

    # -- connectivity ------------------------------------------------------- #
    fiware_connected: Optional[bool] = None
    fiware_error: str = ""
    notice: str = ""

    # -- derived ------------------------------------------------------------ #

    @property
    def plans_ready(self) -> bool:
        return bool(self.baseline and self.optimized)

    @property
    def can_approve(self) -> bool:
        """May the operator approve THIS plan revision, right now?

        Every clause is a way the button could otherwise lie:

          * the canonical workflow must actually be at the approval gate — not
            planning, not executing, not re-planning;
          * a selected plan must exist. A rendered plan is not enough: after a
            re-plan the previous geometry is still on screen for a moment;
          * that plan must still be PENDING. `approved`, `rejected` and
            `superseded` all mean this decision has already been made, and
            `superseded` is exactly the stale-revision case;
          * the plan must be VALID. Approving a plan the Digital Twin rejected
            would authorise physical action on geometry that failed validation.
        """
        return not self.approval_block_reason

    @property
    def approval_block_reason(self) -> str:
        """WHY approval is unavailable, in the operator's words. "" when it is.

        Every disabled state must be explainable. Showing "waiting for your
        decision" while every decision control is dead, with no reason given, is
        the specific failure this exists to prevent.
        """
        # FIRST, because everything below reads fields that would then be
        # describing different runs. Controls whose effect cannot be stated are
        # not offered at all.
        if self.control_inconsistency:
            return self.control_inconsistency
        if self.control_anomaly_hold:
            return ("an anomaly is holding execution — acknowledge it before "
                    "authorising anything")
        if self.control_scene_ready is False:
            # Approving authorises PHYSICAL action, and the physical scene does
            # not yet correspond to this scenario.
            return (self.control_scene_reason
                    or "the physical scene is not ready for this scenario")
        if self.control_finished:
            return "the run has finished"
        if self.control_stage != "WAIT_FOR_OPERATOR_APPROVAL":
            return (f"the workflow is at {self.control_stage}, not at the "
                    "approval gate")
        if not self.control_plan_id:
            return "there is no selected plan to approve"
        if self.control_approval_state == "approved":
            return "this plan is already approved"
        if self.control_approval_state == "superseded":
            return ("this plan was superseded by a re-plan — review the new "
                    "one")
        if self.control_approval_state != "pending":
            return f"the plan is {self.control_approval_state}"
        if not self.control_plan_valid:
            return ("the Digital Twin did not validate this plan, so it cannot "
                    "be authorised")
        return ""

    def control_state(self) -> Dict[str, Any]:
        """What the operator controls are enabled from — ROS/DDS only."""
        return {
            "stage": self.control_stage,
            "approval_state": self.control_approval_state,
            "plan_id": self.control_plan_id,
            "plan_valid": self.control_plan_valid,
            "finished": self.control_finished,
            "can_approve": self.can_approve,
            "block_reason": self.approval_block_reason,
            "anomaly_hold": self.control_anomaly_hold,
            "anomaly_ack_required": self.control_anomaly_ack_required,
            "scene_ready": self.control_scene_ready,
            "scene_reason": self.control_scene_reason,
            "consistent": not self.control_inconsistency,
            "inconsistency": self.control_inconsistency,
            "identity": dict(self.control_identity),
            "approval_revision": self.control_approval_revision,
            "approval_plan_id": self.control_approval_plan_id,
            # Same revision guard for the two other plan-scoped decisions.
            "can_reject": self.can_approve,
            "can_alternative": (not self.control_inconsistency
                                and not self.control_finished
                                and bool(self.control_plan_id)),
            "source": "ros",
        }

    def badge(self) -> Dict[str, Any]:
        """The header badge. Never claims a source the data did not come from."""
        if self.mode == "sim":
            return {
                "source": "sim", "label": "SIMULATED", "live": False,
                "detail": ("no ROS, no FIWARE — same domain logic and optimizer "
                           "as live mode; execution outcomes are simulated"),
            }
        used = {self.panel_sources.get(p, self.mode) for p in PANELS}
        if self.mode == "fiware" and "fiware" in used:
            if used - {"fiware"}:
                return {
                    "source": "fiware+ros", "label": "FIWARE + ROS", "live": True,
                    "detail": ("auditable state and KPIs read back from Orion-LD "
                               "over NGSI-LD; plan geometry and the event stream "
                               "come from ROS 2 — see the per-panel badges"),
                }
            return {"source": "fiware", "label": "FIWARE", "live": True,
                    "detail": "every panel read back from Orion-LD over NGSI-LD"}
        if self.mode == "fiware":
            # FIWARE was SELECTED but no panel could be read back from it. Say
            # exactly that. Falling through to a plain "ROS 2 / DDS" badge is
            # what let the launcher, the header and the FIWARE pill assert three
            # different things at once on the same screen.
            return {
                "source": "fiware-degraded", "label": "FIWARE DEGRADED → ROS 2 / DDS",
                "live": True,
                "detail": ("the FIWARE source was requested but Orion-LD "
                           "returned nothing readable; every panel below is "
                           "coming from ROS 2 / DDS instead. The audit trail is "
                           "NOT being read back from FIWARE in this run."),
            }
        return {
            "source": "ros", "label": "ROS 2 / DDS", "live": True,
            "detail": ("live ROS 2 topics over Fast DDS; robot outcomes remain "
                       "simulated"),
        }

    def backend_badge(self) -> Dict[str, Any]:
        """The EXECUTION badge, separate from the data-source badge.

        Never claims physics it did not perform: the label comes from the
        orchestrator's own published backend, and until that has been received
        `known` is False so the frontend can show "—" instead of asserting
        "SIMULATED" about a run it has not heard from yet.
        """
        return {
            "backend": self.execution_backend,
            "label": self.execution_backend_label,
            "detail": self.execution_backend_detail,
            "physical": self.execution_backend == "isaac",
            "known": self.execution_backend_known,
            # WHICH ARM, for the header badge and the Simulator View. Present
            # only for a physical backend: a logical run has no robot, and
            # showing one would be an outright false claim about what moved.
            "robot": self.active_robot,
            "robot_id": self.active_robot_id,
            "isaac": self.isaac,
        }

    def to_state(self) -> Dict[str, Any]:
        return {
            "badge": self.badge(),
            "execution": self.backend_badge(),
            "mode": self.mode,
            "panel_sources": dict(self.panel_sources),
            "stage": self.stage,
            "run_id": self.run_id,
            "cycle_id": self.cycle_id,
            "finished": self.finished,
            "degraded_reason": self.degraded_reason,
            "robot_state": self.robot_state,
            "current_item_id": self.current_item_id,
            "current_container_id": self.current_container_id,
            "progress_pct": round(self.progress_pct, 1),
            "approval_state": self.approval_state,
            "control": self.control_state(),
            "readiness": self.readiness,
            "auto_step": self.auto_step,
            "scenario": self.scenario,
            "detected_count": self.detected_count,
            "selected_plan_id": self.selected_plan_id,
            "scenario_revision": self.scenario_revision,
            "anomaly": self.anomaly,
            "plan_container_status": (self.whole_process or {}).get(
                "plan_container_status"),
            "selection_reason": self.selection_reason,
            "action_count": self.action_count,
            "stats": self.stats,
            "dynamic_events": self.dynamic_events,
            "notice": self.notice,
            "fiware": {"connected": self.fiware_connected,
                       "error": self.fiware_error,
                       # Reachable is NOT the same as current. Collapsing the
                       # two is what let a stale broker's values be shown as
                       # though they described the running scenario.
                       "sync_status": self.fiware_sync_status,
                       "sync_detail": self.fiware_sync_detail,
                       "run_id": self.fiware_run_id,
                       "scenario_revision": self.fiware_scenario_revision,
                       "stage": self.fiware_stage,
                       "rejected_stale_fields": list(self.rejected_stale_fields)},
        }

    def to_strategies(self) -> Dict[str, Any]:
        """The strategy comparison, or an explicit reason it is not shown.

        A comparison from a DIFFERENT scenario revision is reported ``stale`` and
        never rendered as if it were current — that is the whole point of
        stamping it with a revision.
        """
        comp = self.strategy_comparison
        if not comp or not comp.get("results"):
            status = (comp or {}).get("status", "none")
            return {"ready": False, "status": status,
                    "source": self.panel_sources.get("plans", self.mode),
                    "scenario_revision": self.scenario_revision,
                    "error": (comp or {}).get("error")}
        comp_rev = comp.get("scenario_revision")
        stale = comp_rev is not None and comp_rev != self.scenario_revision
        return {
            "ready": not stale,
            "stale": stale,
            "status": "stale" if stale else comp.get("status", "completed"),
            "source": comp.get("source", self.mode),
            "scenario_revision": self.scenario_revision,
            "comparison": comp,
        }

    def to_whole_process(self) -> Dict[str, Any]:
        """Cut-aware comparison + inventory + logistics, from any source."""
        wp = self.whole_process or {}
        return {
            "source": self.panel_sources.get("whole_process", self.mode),
            "cut": wp.get("cut"),
            "inventory": wp.get("inventory"),
            "logistics": wp.get("logistics"),
            "planning_result": wp.get("planning_result"),
            "plan_container_status": wp.get("plan_container_status", "ok"),
            "analytics": wp.get("analytics"),
        }

    def to_inventory(self) -> Dict[str, Any]:
        wp = self.whole_process or {}
        inv = wp.get("inventory") or {}
        return {
            "source": self.panel_sources.get("whole_process", self.mode),
            "ready": bool(inv.get("containers")),
            "summary": inv.get("summary", {}),
            "containers": inv.get("containers", []),
            "shortage_events": inv.get("shortage_events", []),
            "planning_result": wp.get("planning_result"),
            "analytics": (wp.get("analytics") or {}).get("inventory"),
            # PROVENANCE. Who filled these containers — the logical simulator or
            # a named arm. An inventory row outlives the run that produced it,
            # and "a robot placed this" stops being a usable record the moment
            # more than one robot is selectable.
            "execution_backend": self.execution_backend,
            "robot_id": self.active_robot_id,
            "robot_display_name": (self.active_robot or {}).get("display_name"),
        }

    def to_logistics(self) -> Dict[str, Any]:
        wp = self.whole_process or {}
        log = wp.get("logistics") or {}
        return {
            "source": self.panel_sources.get("whole_process", self.mode),
            "ready": bool(log.get("tasks") is not None),
            "facility_map": log.get("facility_map", {}),
            "tasks": log.get("tasks", []),
            "robot": log.get("robot", {}),
            "analytics": log.get("analytics"),
        }

    def to_visualization(self) -> Dict[str, Any]:
        """The visualization descriptor, ALWAYS well-formed.

        A missing descriptor becomes an explicit `unavailable` with a reason
        rather than a null the frontend has to guess about. That is what keeps
        the Simulator View from ever rendering an empty box or a permanent
        spinner: every path ends in a named state with wording attached.
        """
        from wisepack_core.visualization import (               # noqa: PLC0415
            VisualizationDescriptor, unavailable,
        )
        backend = self.execution_backend
        if self.visualization is None:
            if backend != "isaac":
                return unavailable(
                    backend,
                    "the simulated execution backend has no renderer and "
                    "offers no visual stream").to_dict()
            return unavailable(
                backend,
                "waiting for the simulator to report its visualization "
                "capability").to_dict()
        return VisualizationDescriptor.from_dict(self.visualization).to_dict()

    def to_plans(self) -> Dict[str, Any]:
        return {
            "ready": self.plans_ready,
            "source": self.panel_sources.get("plans", self.mode),
            "scenario": self.scenario,
            "baseline": self.baseline,
            "optimized": self.optimized,
            "selected": self.selected or self.optimized,
            "selected_plan_id": self.selected_plan_id,
            "selection_reason": self.selection_reason,
            "plan_status": self.plan_status,
        }

    def to_kpis(self) -> Dict[str, Any]:
        return {
            "ready": bool(self.kpis),
            "source": self.panel_sources.get("kpis", self.mode),
            "metrics": self.kpis,
            "target_assessment": self.target_assessment,
        }

    def to_events(self, limit: int = 150) -> Dict[str, Any]:
        return {
            "source": self.panel_sources.get("events", self.mode),
            "events": self.events[:limit],
            "total": self.action_count or len(self.events),
        }

    def to_analytics(self) -> Dict[str, Any]:
        """Aggregates computed from whatever events this snapshot carries.

        Deliberately derived here rather than asked of the engine: in live modes
        there is no engine in this process, and an analytics panel that only
        works in sim mode is exactly the asymmetry this module removes.
        """
        by_action: Dict[str, int] = {}
        by_stage: Dict[str, int] = {}
        duration: Dict[str, float] = {}
        sequences: List[int] = []
        for e in self.events:
            by_action[e.get("action", "?")] = by_action.get(e.get("action", "?"), 0) + 1
            stage = e.get("stage", "?")
            by_stage[stage] = by_stage.get(stage, 0) + 1
            if e.get("duration_ms"):
                duration[stage] = round(duration.get(stage, 0.0)
                                        + float(e["duration_ms"]), 3)
            if isinstance(e.get("sequence"), int):
                sequences.append(e["sequence"])
        ordered = sorted(sequences)
        gap_free = (not ordered
                    or ordered == list(range(ordered[0], ordered[0] + len(ordered))))
        return {
            "ready": bool(self.events) or bool(self.kpis),
            "source": self.panel_sources.get("analytics", self.mode),
            "by_action": by_action,
            "by_stage": by_stage,
            "duration_by_stage_ms": duration,
            "sequence_ok": gap_free,
            "total_events": self.action_count or len(self.events),
            "replans": int(self.stats.get("replans", 0) or 0),
            "replan_causes": list(self.stats.get("replan_causes", []) or []),
            "pick_attempts": int(self.stats.get("pick_attempts", 0) or 0),
            "pick_successes": int(self.stats.get("pick_successes", 0) or 0),
            # Whole-process analytics (cutting / inventory / logistics) with the
            # provenance labels the core attaches — measured / simulated / derived.
            "whole_process": (self.whole_process or {}).get("analytics"),
        }


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


class DashboardSnapshotProvider:
    """Contract: build a complete DashboardSnapshot, or explain why not."""

    mode = "sim"

    def snapshot(self) -> DashboardSnapshot:          # pragma: no cover - abstract
        raise NotImplementedError


class SimSnapshotProvider(DashboardSnapshotProvider):
    """STATE.engine is authoritative. No ROS, no FIWARE."""

    mode = "sim"

    def __init__(self, state, latency_lookup: Optional[Callable[[], Optional[float]]] = None):
        self.state = state
        self._latency = latency_lookup or (lambda: None)

    def snapshot(self) -> DashboardSnapshot:
        st = self.state
        with st.lock:
            engine = st.engine
            snap = DashboardSnapshot(
                mode="sim",
                panel_sources={p: "sim" for p in PANELS},
                auto_step=st.auto_step,
                notice=st.notice,
                events=list(reversed(st.events[-400:])),
            )
            if engine is None:
                return snap

            snap.stage = engine.stage.value
            snap.run_id = engine.run_id
            snap.cycle_id = engine.cycle_id
            snap.finished = engine.finished
            snap.degraded_reason = engine.degraded_reason
            snap.robot_state = engine.robot_state
            snap.current_item_id = engine.current_item_id
            snap.current_container_id = engine.current_container_id
            snap.progress_pct = engine.progress_pct
            snap.readiness = not engine.finished
            snap.action_count = engine.log.count
            snap.stats = engine.stats.to_dict()
            snap.dynamic_events = [e.to_dict() for e in engine.dynamic_events]
            snap.detected_count = len(engine.detected)
            snap.scenario_revision = engine.scenario_revision
            snap.anomaly = engine.anomaly_snapshot()
            snap.whole_process = engine.wp.snapshot()
            comp = engine.strategy_comparison
            if comp:
                snap.strategy_comparison = {**comp, "source": "sim"}
            if engine.scenario:
                snap.scenario = engine.scenario.to_dict()
            if engine.baseline:
                snap.baseline = engine.baseline.to_dict()
            if engine.optimized:
                snap.optimized = engine.optimized.to_dict()
            if engine.selected:
                snap.selected = engine.selected.to_dict()
                snap.selected_plan_id = engine.selected.plan_id
                snap.approval_state = engine.selected.approval_state.value
                snap.control_plan_id = engine.selected.plan_id
                snap.control_approval_state = engine.selected.approval_state.value
                snap.control_plan_valid = bool(engine.selected.is_valid)
            snap.control_stage = engine.stage.value
            snap.control_finished = engine.finished
            snap.control_anomaly_hold = bool(engine.anomaly_hold)
            snap.control_anomaly_ack_required = bool(engine.anomaly_ack_required)
            # Sim mode reads ONE object, so components cannot come from
            # different runs — but the approval invariant is still checked
            # rather than assumed, and reported the same way.
            snap.control_approval_revision = engine.approval_revision
            snap.control_approval_plan_id = engine.approval_plan_id
            snap.control_inconsistency = engine.approval_inconsistency()
            snap.control_identity = {"run_id": engine.run_id,
                                     "scenario_revision": engine.scenario_revision}
            if snap.control_inconsistency:
                snap.degraded_reason = snap.control_inconsistency
            snap.selection_reason = engine.selection_reason

            # sim mode drives the engine in THIS process, and it always uses the
            # simulated robot model — there is no Isaac here and there must be no
            # suggestion of one. Stated from the engine's own config rather than
            # hard-coded, so it stays true if sim mode ever gains a backend.
            backend = engine.config.execution_backend
            snap.execution_backend = backend.value
            snap.execution_backend_detail = backend.detail
            snap.execution_backend_known = True
            # The in-process sim mode runs the engine here, so the ACTIVE robot
            # is whatever this engine's config records — which for the logical
            # backend is nothing at all.
            snap.active_robot_id = engine.config.robot_id or None
            snap.active_robot = _public_robot(engine.config.robot_id)
            snap.execution_backend_label = backend.badge(
                (snap.active_robot or {}).get("display_name", ""))

            if engine.baseline and engine.optimized and engine.selected:
                report = engine.kpis(self._latency())
                snap.kpis = {k: m.to_dict() for k, m in report.metrics.items()}
                snap.target_assessment = report.assess_targets()
        return snap


#: Which mirror components carry an identity stamp, and what to call them when
#: they disagree. Ordered so the message names the most operator-meaningful
#: component first.
_STAMPED_COMPONENTS = (
    ("scenario", "the scenario"),
    ("selected", "the selected plan"),
    ("plan_summary", "the plan summary"),
)


def _identity_conflict(mirror: Dict[str, Any]) -> "tuple[str, Dict[str, Any]]":
    """Return (reason, identity) — reason "" when every component agrees.

    THE FAILURE THIS PREVENTS. The orchestrator's topics are latched and
    independent, so the mirror is always a mixture of whatever each publisher
    last said. After a reset or a re-plan, the scenario topic can carry the new
    run while the plan topic still carries the old one, and a consumer that just
    reads both fields renders a coherent-looking page describing two different
    runs. Observed: scenario ``mixed_pipes_dense-s42`` beside plan
    ``plan-optimized-isaac_cylinders_smoke-s42``.

    The response is to REPORT the disagreement, not to guess which half is
    current and not to hide it. Operator controls are withheld until the
    components agree again, which they do on the next full publish.
    """
    stamps = mirror.get("stamps") or {}
    seen: Dict[str, Any] = {}
    identity: Dict[str, Any] = {}
    for key, label in _STAMPED_COMPONENTS:
        stamp = stamps.get(key)
        if not isinstance(stamp, dict):
            continue
        # A component that has never been stamped is an OLDER publisher, not a
        # conflicting one; treating "unknown" as a mismatch would block every
        # control during a rolling upgrade.
        for facet in ("run_id", "scenario_revision"):
            value = stamp.get(facet)
            if value is None:
                continue
            identity.setdefault(facet, value)
            previous = seen.setdefault(facet, (value, label))
            if previous[0] != value:
                return (
                    f"{previous[1]} and {label} describe different runs "
                    f"({facet} {previous[0]!r} vs {value!r}) — the dashboard is "
                    "showing a mixture and will not offer controls until the "
                    "orchestrator republishes a consistent set",
                    identity,
                )

    # The plan the summary names and the plan actually published must be one
    # plan. They are produced by the same call but travel on separate topics.
    summary = mirror.get("plan_summary") or {}
    selected = mirror.get("selected") or {}
    summary_plan = summary.get("selected_plan_id")
    selected_plan = selected.get("plan_id")
    if summary_plan and selected_plan and summary_plan != selected_plan:
        return (f"the plan summary names {summary_plan} but the published plan "
                f"is {selected_plan} — the dashboard is showing a mixture and "
                "will not offer controls until they agree", identity)

    # An approval decision is about one plan of one revision. A stamp pointing
    # anywhere else means the decision on screen is not about what is on screen.
    approval_plan = summary.get("approval_plan_id")
    if approval_plan and selected_plan and approval_plan != selected_plan:
        return (f"the pending decision is about plan {approval_plan} but the "
                f"selected plan is {selected_plan} — a fresh decision is "
                "required", identity)
    approval_rev = summary.get("approval_revision")
    current_rev = identity.get("scenario_revision")
    if (approval_rev is not None and current_rev is not None
            and int(approval_rev) != int(current_rev)):
        return (f"the pending decision is about scenario revision "
                f"{approval_rev} but the current revision is {current_rev} — a "
                "fresh decision is required", identity)
    return "", identity


class RosSnapshotProvider(DashboardSnapshotProvider):
    """STATE.ros_mirror is authoritative — the live topic contract, mirrored."""

    mode = "ros"

    def __init__(self, state):
        self.state = state

    def _base(self) -> DashboardSnapshot:
        st = self.state
        with st.lock:
            mirror = dict(st.ros_mirror or {})
            events = list(reversed(st.events[-400:]))
            notice = st.notice
            connected = st.fiware_connected
            error = st.fiware_last_error

        snap = DashboardSnapshot(
            mode=self.mode,
            panel_sources={p: "ros" for p in PANELS},
            notice=notice,
            events=events,
            fiware_connected=connected,
            fiware_error=error,
        )
        if not mirror:
            snap.notice = (notice or
                           "waiting for the ROS 2 stack — no topics received yet")
            return snap

        snap.stage = mirror.get("stage") or "IDLE"
        snap.run_id = mirror.get("run_id")
        snap.progress_pct = float(mirror.get("progress_pct") or 0.0)
        snap.readiness = bool(mirror.get("readiness"))
        snap.current_item_id = mirror.get("current_item_id")
        snap.current_container_id = mirror.get("current_container_id")
        snap.detected_count = int(mirror.get("detected_count") or 0)
        snap.scenario = mirror.get("scenario")
        # Scenario revision: prefer the plan-summary digest (published by the
        # orchestrator), fall back to the comparison's own stamp.
        summary = mirror.get("plan_summary") or {}
        snap.scenario_revision = int(
            summary.get("scenario_revision")
            or (mirror.get("strategy_comparison") or {}).get("scenario_revision")
            or 0)
        snap.strategy_comparison = mirror.get("strategy_comparison")
        snap.anomaly = mirror.get("anomaly")
        # Whole-process, assembled from the cutting / inventory / logistics topics.
        cut = mirror.get("cut")
        inv_summary = mirror.get("inventory_summary") or {}
        inv_containers = mirror.get("inventory_containers") or []
        log_map = mirror.get("logistics_map") or {}
        log_robot = mirror.get("logistics_robot") or {}
        snap.whole_process = {
            "cut": cut,
            "inventory": {"containers": inv_containers, "summary": inv_summary,
                          "shortage_events": []},
            "logistics": {"facility_map": log_map, "robot": log_robot,
                          "tasks": (log_map.get("pending_tasks", [])
                                    + ([log_map["active_task"]]
                                       if log_map.get("active_task") else [])),
                          "analytics": (log_map.get("robot") or {})},
            "planning_result": (cut or {}).get("planning_result"),
            "plan_container_status": "ok",
            "analytics": {
                "cutting": {"provenance": "simulated_cutting_measured_packing",
                            "recommend_cut": (cut or {}).get("recommend_cut"),
                            "containers_avoided": (cut or {}).get("containers_saved")},
                "inventory": {**inv_summary, "provenance": "software_state"},
                "logistics": (log_map.get("analytics") if isinstance(
                    log_map, dict) else {}) or {},
            },
        }
        snap.baseline = mirror.get("baseline")
        snap.optimized = mirror.get("optimized")
        snap.selected = mirror.get("selected")
        snap.plan_status = mirror.get("plan_status")
        snap.action_count = int(mirror.get("action_sequence") or len(events))
        snap.finished = snap.stage in ("COMPLETE", "DEGRADED")
        snap.degraded_reason = ("orchestrator reported DEGRADED"
                                if snap.stage == "DEGRADED" else "")
        snap.robot_state = _robot_state_for(snap.stage)

        # The execution backend, as published by the orchestrator. Absent until
        # that topic arrives, and NOT defaulted to a claim: `known` stays False
        # so the header renders "—" rather than asserting "SIMULATED" about a run
        # it has not heard from. The topic is latched, so this fills in on the
        # first snapshot after the observer connects.
        backend_doc = mirror.get("execution_backend")
        if isinstance(backend_doc, dict) and backend_doc.get("backend"):
            snap.execution_backend = backend_doc["backend"]
            snap.execution_backend_label = backend_doc.get(
                "label", snap.execution_backend.upper())
            snap.execution_backend_detail = backend_doc.get("detail", "")
            snap.execution_backend_known = True
            snap.active_robot = backend_doc.get("robot")
            snap.active_robot_id = backend_doc.get("robot_id")
            snap.isaac = backend_doc.get("isaac")
            snap.visualization = backend_doc.get("visualization")
            isaac = backend_doc.get("isaac") or {}
            if "scene_ready" in isaac:
                snap.control_scene_ready = bool(isaac.get("scene_ready"))
                snap.control_scene_reason = str(isaac.get("reset_failed_reason") or "")
                if not snap.control_scene_ready and not snap.control_scene_reason:
                    # ACCURACY OVER BREVITY. "has not been rebuilt" was being
                    # shown while the operator looked at a correct Panda, four
                    # cylinders and an empty container — which reads as a broken
                    # dashboard. What is actually missing in that case is the
                    # acknowledgement correlated to THIS run, and saying so is
                    # both true and actionable.
                    if isaac.get("scene_mismatch"):
                        snap.control_scene_reason = (
                            "the simulator's scene does not match this run: "
                            + str(isaac["scene_mismatch"]))
                    elif isaac.get("reset_in_progress"):
                        snap.control_scene_reason = (
                            "the physical scene is being built and verified for "
                            "this scenario")
                    elif isaac.get("ros_bridge_ready"):
                        snap.control_scene_reason = (
                            "Isaac is ready, but the scene acknowledgement for "
                            "the current run has not been received")
                    else:
                        snap.control_scene_reason = (
                            "the simulator has not reported ready yet")
        snap.isaac_results = list(mirror.get("isaac_results") or [])

        summary = mirror.get("plan_summary") or {}
        snap.selection_reason = summary.get("selection_reason", "")
        snap.selected_plan_id = summary.get("selected_plan_id")
        if snap.selected:
            snap.selected_plan_id = snap.selected.get("plan_id", snap.selected_plan_id)
            snap.approval_state = snap.selected.get("approval_state", "pending")

        # CONTROL STATE, from the canonical ROS topics and nothing else. Set
        # here in the shared base so FiwareSnapshotProvider — which subclasses
        # this — inherits it and then overwrites only the DISPLAY fields.
        snap.control_stage = snap.stage
        snap.control_finished = snap.finished
        if snap.selected:
            snap.control_plan_id = snap.selected.get("plan_id")
            snap.control_approval_state = snap.selected.get("approval_state",
                                                            "pending")
            # `is_valid` is computed by the domain model and published with the
            # plan, so the dashboard does not re-derive validity it cannot see.
            snap.control_plan_valid = bool(snap.selected.get("is_valid", False))
        anomaly = mirror.get("anomaly") or {}
        snap.control_anomaly_hold = bool(anomaly.get("hold", False))
        snap.control_anomaly_ack_required = bool(anomaly.get("ack_required", False))

        # CROSS-COMPONENT IDENTITY. Everything above was read from independent
        # latched topics; this is where the snapshot refuses to present them as
        # one run unless they say they are one run.
        snap.control_approval_revision = summary.get("approval_revision")
        snap.control_approval_plan_id = summary.get("approval_plan_id")
        snap.control_inconsistency, snap.control_identity = _identity_conflict(mirror)
        if snap.control_inconsistency:
            snap.degraded_reason = snap.control_inconsistency

        # WATCHDOG. DDS-level liveliness is unusable in this deployment (see
        # qos.watchdog_subscribe_qos), so a stalled heartbeat counter is what
        # tells us the orchestrator is gone. Reporting DEGRADED here is the
        # honest response: WISEPACK never simulates unsafe continuation.
        beat_at = float(mirror.get("heartbeat_at") or 0.0)
        if beat_at and (time.time() - beat_at) > HEARTBEAT_STALE_S:
            stale_for = time.time() - beat_at
            snap.stage = "DEGRADED"
            snap.robot_state = "held"
            snap.finished = True
            snap.degraded_reason = (
                f"no heartbeat for {stale_for:.1f}s — the orchestrator is not "
                "responding; execution is HELD")

        snap.stats = _stats_from_events(events)
        snap.kpis = _kpis_from_mirror(mirror, snap)
        snap.target_assessment = _targets_from_kpis(snap.kpis)
        return snap

    def snapshot(self) -> DashboardSnapshot:
        return self._base()


class FiwareSnapshotProvider(RosSnapshotProvider):
    """Auditable state and KPIs from Orion-LD — but only for the CURRENT run.

    The split is stated rather than glossed. Orion-LD holds the audit-relevant
    values — stage, readiness, KPI attributes, the plan digest, the latest action
    and its sequence — because those are what crossing DDS into NGSI-LD proves.
    It does NOT hold 40 placement coordinates (see bridge_config.yaml), so the
    Digital Twin still renders from ROS, and the panel says so.

    WHAT CHANGED, AND WHY IT HAD TO
    -------------------------------
    Orion-LD holds CURRENT STATE, not a log. Every attribute keeps its last value
    until something overwrites it — across scenario changes, restarts and whole
    runs. This provider used to apply whatever it read:

        if system.get("stage"): snap.stage = system["stage"]
        merged.update(_kpis_from_fiware(kpi))

    With `mixed_pipes_dense` running (3 baseline containers, 2 optimized) and
    `isaac_cylinders_smoke` KPI attributes still sitting in the broker, that
    produced a dashboard whose Digital Twin came from one run and whose KPI cards
    came from another — and a `stage` attribute left at
    WAIT_FOR_OPERATOR_APPROVAL rewound the workflow on screen seconds after the
    operator had approved it.

    Neither symptom is detectable from the transport: an HTTP 200 says the broker
    answered, and an entity existing says something wrote it once. So every
    entity now carries a `runCorrelation` stamp, and NOTHING is applied from an
    entity whose stamp does not match the run the canonical ROS/DDS state says is
    active. Rejected values are recorded rather than dropped silently, because an
    operator looking at a "synchronizing" panel deserves to know what was held
    back and why.

    CANONICAL STATE IS NEVER OVERWRITTEN. `control_*` — what enables Approve,
    Reject, Resume and Step — is computed in `_base()` from ROS/DDS and is not
    touched here at all, for matched or unmatched entities alike. FIWARE
    confirms; it does not command.
    """

    mode = "fiware"

    def __init__(self, state, reader: Callable[[], Dict[str, Any]]):
        super().__init__(state)
        self._read = reader
        #: entity -> highest correlation sequence applied. NGSI-LD gives no
        #: ordering guarantee, and a delayed DDS sample can land after a newer
        #: one; without this, an old value would be re-applied on top of a new
        #: one and the panel would flicker backwards.
        self._seen_sequence: Dict[str, int] = {}

    # -- correlation --------------------------------------------------------- #

    def _active_run(self, snap: DashboardSnapshot) -> RunCorrelation:
        """The run the CANONICAL ROS/DDS state says is executing.

        This is the reference every FIWARE entity is compared against. It comes
        from the mirror, never from FIWARE — otherwise a stale broker would get
        to define what "current" means, which is the whole failure.
        """
        return RunCorrelation(
            run_id=snap.run_id,
            scenario_id=(snap.scenario or {}).get("scenario_id"),
            scenario_revision=snap.scenario_revision,
            plan_id=snap.control_plan_id or snap.selected_plan_id,
            plan_revision=snap.scenario_revision,
            approval_revision=snap.control_approval_revision)

    def _accept(self, name: str, entity: Dict[str, Any],
                active: RunCorrelation,
                rejected: List[Dict[str, Any]]) -> bool:
        """May this entity's values be applied to the current run's view?

        Four distinguishable answers, and they are NOT collapsed:

          * no entity            -> nothing to apply, nothing to report;
          * no stamp at all      -> UNKNOWN. An orchestrator that predates the
            correlation contract is not a conflicting one, so the values are
            applied and the entity is reported as unverified. Treating unknown
            as stale would blank every panel during a rolling upgrade;
          * stamp disagrees      -> STALE. Withheld, and every field it would
            have set is recorded;
          * sequence went back   -> OUT OF ORDER. Withheld: re-applying an older
            sample on top of a newer one is how a panel flickers backwards.
        """
        if not entity:
            return False
        correlation = RunCorrelation.from_dict(entity.get("runCorrelation"))
        if correlation is None or correlation.is_unstamped:
            return True                     # unknown, not wrong — see docstring

        seen = self._seen_sequence.get(name, 0)
        if correlation.sequence and correlation.sequence < seen:
            rejected.append({
                "entity": name, "field": "*",
                "reason": (f"out-of-order update: sequence "
                           f"{correlation.sequence} arrived after {seen}")})
            return False

        mismatches = correlation.mismatches(active)
        if mismatches:
            rejected.append({
                "entity": name, "field": "*",
                "reason": f"describes another run — {describe_mismatch(mismatches)}"})
            return False

        if correlation.sequence:
            self._seen_sequence[name] = correlation.sequence
        return True

    # -- snapshot ------------------------------------------------------------ #

    def snapshot(self) -> DashboardSnapshot:
        snap = self._base()
        snap.mode = "fiware"
        try:
            entities = self._read()
        except Exception as exc:                        # noqa: BLE001
            snap.fiware_connected = False
            snap.fiware_error = f"Orion-LD read failed: {exc}"
            snap.fiware_sync_status = "disconnected"
            snap.fiware_sync_detail = str(exc)
            return snap

        if not entities:
            snap.fiware_connected = False
            snap.fiware_sync_status = "disconnected"
            snap.fiware_sync_detail = "no entities returned by Orion-LD"
            return snap
        snap.fiware_connected = True

        active = self._active_run(snap)
        rejected: List[Dict[str, Any]] = []
        accepted: List[str] = []
        offered: List[str] = []

        def usable(name: str) -> Dict[str, Any]:
            """The entity if it describes the active run, else {}."""
            entity = entities.get(name) or {}
            if not entity:
                return {}
            offered.append(name)
            if self._accept(name, entity, active, rejected):
                accepted.append(name)
                return entity
            return {}

        system = usable("system")
        kpi = usable("kpi")
        scenario = usable("scenario")
        plan = usable("plan")
        actions = usable("actions")
        robot = usable("robot")

        # What the broker BELIEVES, recorded for diagnostics whether or not it
        # was applied. "FIWARE says X, we are running Y" is exactly the sentence
        # an operator needs, and it cannot be written without keeping both.
        raw_system = entities.get("system") or {}
        raw_correlation = RunCorrelation.from_dict(raw_system.get("runCorrelation"))
        if raw_correlation is not None:
            snap.fiware_run_id = raw_correlation.run_id
            snap.fiware_scenario_revision = raw_correlation.scenario_revision
        snap.fiware_stage = (str(raw_system["stage"])
                             if raw_system.get("stage") else None)

        # -- state: CONFIRMS the canonical stage, never rewinds it ----------- #
        #
        # The display stage is the Orion-LD echo, deliberately: it is the audit
        # proof that the workflow crossed into NGSI-LD, and the `source: FIWARE
        # + ROS` badge says so. But it is applied only when the entity describes
        # THIS run. A stale WAIT_FOR_OPERATOR_APPROVAL used to reappear on screen
        # seconds after approval; operator controls were already immune (they
        # read control_* from ROS), but the page said "waiting for approval"
        # while the arm was moving.
        if system.get("stage"):
            snap.stage = str(system["stage"])
            snap.finished = snap.stage in ("COMPLETE", "DEGRADED")
            snap.robot_state = _robot_state_for(snap.stage)
            snap.panel_sources["state"] = "fiware"
        elif snap.fiware_stage:
            snap.panel_sources["state"] = "ros (FIWARE stage is for another run)"
        if system.get("readiness") is not None:
            snap.readiness = bool(system["readiness"])
        if robot.get("progressPct") is not None:
            snap.progress_pct = float(robot["progressPct"])
        if robot.get("currentItem"):
            snap.current_item_id = str(robot["currentItem"])
        if robot.get("currentContainer"):
            snap.current_container_id = str(robot["currentContainer"])

        # -- scenario summary ------------------------------------------------ #
        if isinstance(scenario.get("summary"), dict):
            snap.panel_sources["scenario"] = "fiware"
        if scenario.get("detectedCount") is not None:
            snap.detected_count = int(scenario["detectedCount"])

        # -- plans: the DIGEST is auditable; geometry stays on ROS ----------- #
        digest = plan.get("summary")
        if isinstance(digest, dict):
            snap.selection_reason = digest.get("selection_reason",
                                               snap.selection_reason)
            snap.selected_plan_id = digest.get("selected_plan_id",
                                               snap.selected_plan_id)
            # Geometry is NOT in FIWARE by design, so this panel stays "ros"
            # unless there is no ROS geometry at all, in which case the digest
            # is all we have and the panel says so.
            if not snap.baseline:
                snap.panel_sources["plans"] = "fiware"
        if isinstance(plan.get("status"), dict):
            snap.plan_status = plan["status"]

        # -- KPIs ------------------------------------------------------------ #
        #
        # The tightest case of the whole problem: KPI attributes are bare scalars
        # (`containersBaseline: 1`) that say nothing about which scenario
        # produced them, so a stale entity is indistinguishable from a fresh one
        # by inspection. When it is withheld the ROS-derived KPIs computed in
        # _base() remain — a legitimate source for this panel — and the panel is
        # labelled accordingly rather than blanked.
        fiware_kpis = _kpis_from_fiware(kpi)
        if fiware_kpis:
            merged = dict(snap.kpis)
            merged.update(fiware_kpis)
            snap.kpis = merged
            snap.target_assessment = _targets_from_kpis(snap.kpis)
            snap.panel_sources["kpis"] = "fiware"
        elif entities.get("kpi") and "kpi" not in accepted:
            snap.panel_sources["kpis"] = "ros (FIWARE KPIs are for another run)"
            for key in _kpis_from_fiware(entities.get("kpi") or {}):
                rejected.append({"entity": "kpi", "field": key,
                                 "reason": "KPI belongs to another run"})

        # -- action trail ---------------------------------------------------- #
        if actions.get("sequence") is not None:
            snap.action_count = int(actions["sequence"])
        latest = actions.get("actionJson")
        if isinstance(latest, dict) and not snap.events:
            # No ROS event stream (dashboard attached late): show at least the
            # latest action FIWARE holds, and label the panel accordingly.
            snap.events = [latest]
            snap.panel_sources["events"] = "fiware"

        snap.rejected_stale_fields = rejected
        snap.fiware_sync_status, snap.fiware_sync_detail = _sync_status(
            offered, accepted, rejected, active)
        return snap


def _sync_status(offered: List[str], accepted: List[str],
                 rejected: List[Dict[str, Any]],
                 active: RunCorrelation) -> "tuple[str, str]":
    """Classify how well the broker's view matches the run actually executing.

    Reported as a state with its own wording rather than folded into
    "connected", because reachable-and-stale is the case that used to be
    displayed as though it were reachable-and-current.
    """
    if not offered:
        return "unknown", "no current-state entities returned by Orion-LD"
    if not rejected:
        return "synchronized", f"all {len(offered)} entities describe {active.describe()}"
    stale = sorted({r["entity"] for r in rejected})
    if not accepted:
        return "stale", (
            "FIWARE reachable — awaiting current-run synchronization; "
            f"{', '.join(stale)} still describe an earlier run")
    return "partial", (
        f"FIWARE reachable — awaiting current-run synchronization for "
        f"{', '.join(stale)}; {len(accepted)} of {len(offered)} entities are current")


# --------------------------------------------------------------------------- #
# Shared derivations
# --------------------------------------------------------------------------- #

_EXECUTING = {"PICK_ITEM", "VERIFY_PICK", "PLACE_ITEM", "VERIFY_PLACEMENT",
              "UPDATE_CONTAINER_STATE", "NEXT_ITEM"}


def _robot_state_for(stage: str) -> str:
    if stage in ("PICK_ITEM", "VERIFY_PICK"):
        return "picking"
    if stage in ("PLACE_ITEM", "VERIFY_PLACEMENT"):
        return "placing"
    if stage == "DEGRADED":
        return "held"
    return "idle"


def _stats_from_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reconstruct execution counters from the action stream.

    In live modes the engine's counters live in another process, but every one
    of them is derivable from the events that process published — which is the
    point of an audit trail.
    """
    stats = {"pick_attempts": 0, "pick_successes": 0, "replans": 0,
             "replan_causes": [], "operator_interventions": 0,
             "cycles_attempted": 0, "cycles_completed": 0}
    for e in events:
        action = e.get("action", "")
        if action == "pick_item":
            stats["pick_attempts"] += 1
        elif action == "verify_pick":
            stats["pick_successes"] += 1
        elif action == "replan_start":
            stats["replans"] += 1
            cause = (e.get("details") or {}).get("cause")
            if cause:
                stats["replan_causes"].append(cause)
        elif action in ("approve_plan", "reject_plan", "pause_execution",
                        "resume_execution", "write_artifacts"):
            if e.get("source") == "operator":
                stats["operator_interventions"] += 1
        elif action == "update_container":
            stats["cycles_completed"] += 1
            stats["cycles_attempted"] += 1
        elif action == "abandon_item":
            stats["cycles_attempted"] += 1
    return stats


def _pct(numerator: float, denominator: float) -> Optional[float]:
    return None if denominator <= 0 else 100.0 * numerator / denominator


def _kpis_from_mirror(mirror: Dict[str, Any],
                      snap: DashboardSnapshot) -> Dict[str, Dict[str, Any]]:
    """KPI tiles from the mirrored /wisepack/kpi/* topics."""
    kpi = mirror.get("kpi") or {}
    stats = snap.stats
    out: Dict[str, Dict[str, Any]] = {}

    def put(key, value, unit="", source=None, note=""):
        out[key] = _metric(key, value, unit, source, note)

    put("containers_baseline", kpi.get("containers_baseline"), "containers")
    put("containers_optimized", kpi.get("containers_optimized"), "containers")
    put("container_utilization_baseline_pct",
        kpi.get("utilization_baseline_pct"), "%")
    put("container_utilization_optimized_pct",
        kpi.get("utilization_optimized_pct"), "%")
    put("volume_requirement_reduction_pct", kpi.get("volume_reduction_pct"), "%",
        note="denominator is required container capacity, not material volume")
    put("optimization_time_ms", kpi.get("optimization_ms"), "ms")

    base = kpi.get("utilization_baseline_pct")
    opt = kpi.get("utilization_optimized_pct")
    gain = (100.0 * (opt - base) / base) if (base and opt is not None) else None
    put("packing_density_gain_pct", gain, "%",
        note="relative gain in utilization, not a difference of percentages")

    put("simulated_pick_success_rate_pct",
        kpi.get("pick_success_pct",
                _pct(stats["pick_successes"], stats["pick_attempts"])), "%",
        "simulated", "simulator failure injection, NOT a robot measurement")
    put("simulated_end_to_end_success_rate_pct",
        kpi.get("end_to_end_success_pct",
                _pct(stats["cycles_completed"], stats["cycles_attempted"])), "%",
        "simulated", "simulated packaging cycles, NOT a robot measurement")

    put("replans", float(stats["replans"]), "events")
    put("operator_interventions", float(stats["operator_interventions"]), "events")
    put("action_events_published", float(snap.action_count), "events")
    put("dds_to_fiware_latency_ms", None, "ms",
        note="run measure_dds_fiware_latency.sh to measure this")
    return out


_FIWARE_KPI_MAP = {
    "containersBaseline": ("containers_baseline", "containers"),
    "containersOptimized": ("containers_optimized", "containers"),
    "utilizationBaselinePct": ("container_utilization_baseline_pct", "%"),
    "utilizationOptimizedPct": ("container_utilization_optimized_pct", "%"),
    "volumeReductionPct": ("volume_requirement_reduction_pct", "%"),
    "optimizationMs": ("optimization_time_ms", "ms"),
    "pickSuccessPct": ("simulated_pick_success_rate_pct", "%"),
    "endToEndSuccessPct": ("simulated_end_to_end_success_rate_pct", "%"),
}


def _kpis_from_fiware(kpi: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """KPI tiles read back from the NGSI-LD KPI entity."""
    out: Dict[str, Dict[str, Any]] = {}
    for attr, (key, unit) in _FIWARE_KPI_MAP.items():
        if kpi.get(attr) is None:
            continue
        try:
            value = float(kpi[attr])
        except (TypeError, ValueError):
            continue
        out[key] = _metric(key, value, unit,
                           source=KPI_SOURCES.get(key, "measured"),
                           note="read back from Orion-LD over NGSI-LD")
    base = out.get("container_utilization_baseline_pct", {}).get("value")
    opt = out.get("container_utilization_optimized_pct", {}).get("value")
    if base and opt is not None:
        out["packing_density_gain_pct"] = _metric(
            "packing_density_gain_pct", 100.0 * (opt - base) / base, "%",
            note="derived from the Orion-LD utilization attributes")
    return out


#: Proposal targets. Mirrors wisepack_core.kpi.PROPOSAL_TARGETS so live modes
#: assess them identically; the mapping and the "not_applicable" rule are the
#: same, because a target this demonstrator cannot measure must never be scored.
_TARGETS = (
    ("KPI1", "Vision detection rate", 85.0, False, "detection_rate_pct",
     "No perception model exists in this demonstrator."),
    ("KPI2", "Pick success rate", 80.0, False, "simulated_pick_success_rate_pct",
     "No robot and no grasp planner; the figure is seeded failure injection."),
    ("KPI3", "End-to-end success rate", 80.0, False,
     "simulated_end_to_end_success_rate_pct",
     "Simulated: derived from the simulated pick outcomes."),
    ("KPI4", "Volume reduction vs baseline packing", 50.0, True,
     "volume_requirement_reduction_pct",
     "Genuinely measured: two real algorithms on real geometry."),
)


def _targets_from_kpis(kpis: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, label, threshold, measurable, metric_key, note in _TARGETS:
        metric = kpis.get(metric_key) or {}
        value = metric.get("value")
        row = {"key": key, "label": label, "target_pct": threshold,
               "measured_value": value, "measured_source": metric.get("source")}
        if not measurable:
            row["status"] = "not_applicable"
            row["explanation"] = note
        elif value is None:
            row["status"] = "not_measured"
            row["explanation"] = "no measurement recorded in this run"
        elif value > threshold:
            row["status"] = "met"
            row["explanation"] = f"measured {value:.1f}% > target {threshold:.0f}%"
        else:
            row["status"] = "not_met"
            row["explanation"] = f"measured {value:.1f}%, target is >{threshold:.0f}%"
        rows.append(row)
    return rows


def parse_attr(value: Any) -> Any:
    """Unwrap the Orion-LD DDS bridge's `{"value": {"data": ...}}` shape.

    String attributes that carry JSON are decoded, so a consumer gets an object
    rather than a string containing an object. `"uninitialized"` means the
    attribute is mapped but no DDS sample arrived yet — that is "no data", not
    an error, and it becomes None here.
    """
    if not isinstance(value, dict):
        return None
    inner = value.get("value")
    if isinstance(inner, dict) and "data" in inner:
        inner = inner["data"]
    if inner == "uninitialized" or inner is None:
        return None
    if isinstance(inner, str):
        stripped = inner.strip()
        if stripped[:1] in ("{", "["):
            try:
                return json.loads(stripped)
            except ValueError:
                return inner
    return inner


__all__ = [
    "PANELS", "KPI_KEYS", "DashboardSnapshot", "DashboardSnapshotProvider",
    "SimSnapshotProvider", "RosSnapshotProvider", "FiwareSnapshotProvider",
    "parse_attr",
]
