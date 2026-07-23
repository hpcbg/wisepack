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
    selection_reason: str = ""
    plan_status: Optional[Dict[str, Any]] = None
    detected_count: int = 0
    kpis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    target_assessment: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    action_count: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)
    dynamic_events: List[Dict[str, Any]] = field(default_factory=list)

    # -- connectivity ------------------------------------------------------- #
    fiware_connected: Optional[bool] = None
    fiware_error: str = ""
    notice: str = ""

    # -- derived ------------------------------------------------------------ #

    @property
    def plans_ready(self) -> bool:
        return bool(self.baseline and self.optimized)

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
        return {
            "source": "ros", "label": "ROS 2 / DDS", "live": True,
            "detail": ("live ROS 2 topics over Fast DDS; robot outcomes remain "
                       "simulated"),
        }

    def to_state(self) -> Dict[str, Any]:
        return {
            "badge": self.badge(),
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
            "readiness": self.readiness,
            "auto_step": self.auto_step,
            "scenario": self.scenario,
            "detected_count": self.detected_count,
            "selected_plan_id": self.selected_plan_id,
            "selection_reason": self.selection_reason,
            "action_count": self.action_count,
            "stats": self.stats,
            "dynamic_events": self.dynamic_events,
            "notice": self.notice,
            "fiware": {"connected": self.fiware_connected,
                       "error": self.fiware_error},
        }

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
            snap.selection_reason = engine.selection_reason

            if engine.baseline and engine.optimized and engine.selected:
                report = engine.kpis(self._latency())
                snap.kpis = {k: m.to_dict() for k, m in report.metrics.items()}
                snap.target_assessment = report.assess_targets()
        return snap


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
        snap.baseline = mirror.get("baseline")
        snap.optimized = mirror.get("optimized")
        snap.selected = mirror.get("selected")
        snap.plan_status = mirror.get("plan_status")
        snap.action_count = int(mirror.get("action_sequence") or len(events))
        snap.finished = snap.stage in ("COMPLETE", "DEGRADED")
        snap.degraded_reason = ("orchestrator reported DEGRADED"
                                if snap.stage == "DEGRADED" else "")
        snap.robot_state = _robot_state_for(snap.stage)

        summary = mirror.get("plan_summary") or {}
        snap.selection_reason = summary.get("selection_reason", "")
        snap.selected_plan_id = summary.get("selected_plan_id")
        if snap.selected:
            snap.selected_plan_id = snap.selected.get("plan_id", snap.selected_plan_id)
            snap.approval_state = snap.selected.get("approval_state", "pending")

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
    """Auditable state and KPIs from Orion-LD; geometry and events from ROS.

    The split is stated rather than glossed. Orion-LD holds the audit-relevant
    values — stage, readiness, KPI attributes, the plan digest, the latest action
    and its sequence — because those are what crossing DDS into NGSI-LD proves.
    It does NOT hold 40 placement coordinates (see bridge_config.yaml), so the
    Digital Twin still renders from ROS, and the panel says so.
    """

    mode = "fiware"

    def __init__(self, state, reader: Callable[[], Dict[str, Any]]):
        super().__init__(state)
        self._read = reader

    def snapshot(self) -> DashboardSnapshot:
        snap = self._base()
        snap.mode = "fiware"
        try:
            entities = self._read()
        except Exception as exc:                        # noqa: BLE001
            snap.fiware_connected = False
            snap.fiware_error = f"Orion-LD read failed: {exc}"
            return snap

        if not entities:
            snap.fiware_connected = False
            return snap
        snap.fiware_connected = True

        system = entities.get("system", {})
        kpi = entities.get("kpi", {})
        scenario = entities.get("scenario", {})
        plan = entities.get("plan", {})
        actions = entities.get("actions", {})
        robot = entities.get("robot", {})

        # -- state: authoritative from FIWARE when present ------------------ #
        if system.get("stage"):
            snap.stage = str(system["stage"])
            snap.finished = snap.stage in ("COMPLETE", "DEGRADED")
            snap.robot_state = _robot_state_for(snap.stage)
            snap.panel_sources["state"] = "fiware"
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

        # -- KPIs: authoritative from FIWARE -------------------------------- #
        fiware_kpis = _kpis_from_fiware(kpi)
        if fiware_kpis:
            merged = dict(snap.kpis)
            merged.update(fiware_kpis)
            snap.kpis = merged
            snap.target_assessment = _targets_from_kpis(snap.kpis)
            snap.panel_sources["kpis"] = "fiware"

        # -- action trail ---------------------------------------------------- #
        if actions.get("sequence") is not None:
            snap.action_count = int(actions["sequence"])
        latest = actions.get("actionJson")
        if isinstance(latest, dict) and not snap.events:
            # No ROS event stream (dashboard attached late): show at least the
            # latest action FIWARE holds, and label the panel accordingly.
            snap.events = [latest]
            snap.panel_sources["events"] = "fiware"
        return snap


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
