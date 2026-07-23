"""WISEPACK dashboard backend.

Structure follows TEMPO's web/app.py — FastAPI, an in-memory cache, REST plus an
optional WebSocket, and KPIs read from the newest results/ artefact rather than
invented live. What is different is the source model and the fact that this
dashboard *commands* as well as observes: WISEPACK's whole point is a human in
the loop, so approval and rejection have to travel from here to the workflow.

THREE SOURCES, selected with --source. The badge in the header always names the
one in use, and no figure is ever shown without it.

  sim     Self-contained. Drives wisepack_core.WorkflowEngine in a background
          task. NO ROS, NO FIWARE, NO Docker. Every number is produced by the
          same domain logic the live stack runs — this is not a separate
          animation — but execution outcomes are simulated and labelled so.

  ros     Read-mostly observer of the canonical WISEPACK topics, plus a command
          publisher for the two operator topics. Packing figures here are the
          same measured values; robot outcomes are still simulated, because the
          robot is still a simulator.

  fiware  Authoritative state read back from Orion-LD over NGSI-LD. This is the
          mode that proves the audit path end to end: if a value renders here it
          survived ROS -> DDS -> Orion-LD. High-frequency animation data still
          come from ROS, and the badge says "fiware+ros" when that is the case.

Run:  python3 app.py --source sim --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(REPO, "results")

# wisepack_core is imported from the workspace source tree, so the dashboard
# runs with nothing installed and nothing built. This is the same import the
# ROS nodes do after a colcon build; only the path discovery differs.
for _pkg in ("wisepack_core", "wisepack_fiware", "wisepack_bringup"):
    _path = os.path.join(REPO, "wisepack_ws", "src", _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from wisepack_core.artifacts import (                              # noqa: E402
    latest_artifact, latest_latency_p50_ms, write_run_artifacts,
    write_validation_report,
)
from wisepack_core.domain import Strategy                          # noqa: E402
from wisepack_core.events import (                                 # noqa: E402
    DynamicEvent, DynamicEventType, Stage,
)
from wisepack_core.generator import CONTAINER_SPECS, PRESETS       # noqa: E402
from wisepack_core.kpi import compare_strategies                   # noqa: E402
from wisepack_core.packing import OptimizerConfig                  # noqa: E402
from wisepack_core.workflow import (                               # noqa: E402
    ApprovalRequired, RobotSimConfig, WorkflowConfig, WorkflowEngine,
)

SOURCE = os.environ.get("WISEPACK_SOURCE", "sim")
ORION = os.environ.get("ORION", "http://localhost:1026").rstrip("/")

#: Seconds between execution steps in sim mode. Slow enough to watch, fast
#: enough that a 40-item scenario finishes inside a demo slot.
STEP_PERIOD_S = float(os.environ.get("WISEPACK_STEP_PERIOD_S", "0.7"))


# --------------------------------------------------------------------------- #
# Shared state
# --------------------------------------------------------------------------- #


class DemoState:
    """The dashboard's view of one run, guarded by one lock.

    A single lock rather than per-field locking: the workflow engine mutates
    several related fields per step and a reader that saw half an update would
    render a container holding an item the plan no longer contains.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.engine: Optional[WorkflowEngine] = None
        self.source = SOURCE
        self.events: List[Dict[str, Any]] = []
        self.running = False
        self.auto_step = False
        #: Populated by ros_observer in live mode: the latest value seen on each
        #: canonical topic. None in sim mode, where there is no ROS.
        self.ros_mirror: Optional[Dict[str, Any]] = None
        self.fiware_connected: Optional[bool] = None
        self.fiware_last_error = ""
        self.notice = ""
        self.settings: Dict[str, Any] = {
            "preset": "mixed_pipes_dense",
            "seed": 42,
            "strategy": Strategy.MAX_DENSITY.value,
            "item_count": None,
            "length_range_mm": None,
            "diameter_range_mm": None,
            "container_spec": None,
            "dynamic_events_enabled": True,
            "pick_failure_probability": 0.08,
        }

    # -- event capture ---------------------------------------------------- #

    def sink(self, event) -> None:
        with self.lock:
            self.events.append(event.to_dict())
            if len(self.events) > 4000:
                del self.events[:len(self.events) - 4000]

    def recent(self, limit: int = 120) -> List[Dict[str, Any]]:
        with self.lock:
            return list(reversed(self.events[-limit:]))


STATE = DemoState()


# --------------------------------------------------------------------------- #
# Scenario construction
# --------------------------------------------------------------------------- #

#: The dynamic-event script used by the interactive demo. Deterministic and
#: declared here rather than randomly generated, so a presenter knows exactly
#: what will happen and when.
DEMO_EVENTS = [
    DynamicEvent(
        event_type=DynamicEventType.ITEM_INJECT,
        trigger="placement:4",
        label="High-priority ILW component arrives late",
        payload={"item": {"length_mm": 1200, "outer_diameter_mm": 220,
                          "inner_diameter_mm": 186, "material": "stainless_316L",
                          "priority": 9, "dose_class": "ILW"}}),
]


def _generator_overrides(settings: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if settings.get("item_count"):
        overrides["item_count"] = int(settings["item_count"])
    if settings.get("length_range_mm"):
        overrides["length_range_mm"] = tuple(settings["length_range_mm"])
    if settings.get("diameter_range_mm"):
        overrides["diameter_range_mm"] = tuple(settings["diameter_range_mm"])
    if settings.get("container_spec"):
        overrides["container_spec"] = settings["container_spec"]
    return overrides


def build_engine(settings: Dict[str, Any]) -> WorkflowEngine:
    preset = settings.get("preset", "mixed_pipes_dense")
    seed = int(settings.get("seed", 42))
    # The curated dataset is hand-built; generator overrides do not apply to it
    # and silently accepting them would misrepresent what was run.
    overrides = ({} if preset == "curated_volume_reduction"
                 else _generator_overrides(settings))
    config = WorkflowConfig(
        preset=preset,
        seed=seed,
        strategy=Strategy(settings.get("strategy", "max_density")),
        optimizer=OptimizerConfig(seed=seed, restarts=6, time_budget_ms=4000.0),
        robot=RobotSimConfig(
            pick_failure_probability=float(
                settings.get("pick_failure_probability", 0.08)),
            seed=seed),
        dynamic_events=([DynamicEvent.from_dict(e.to_dict()) for e in DEMO_EVENTS]
                        if settings.get("dynamic_events_enabled", True) else []),
        generator_overrides=overrides,
        auto_approve=False)
    engine = WorkflowEngine(config)
    engine.log.add_sink(STATE.sink)
    return engine


def start_run(settings: Dict[str, Any]) -> WorkflowEngine:
    """Plan a fresh run and stop at the approval gate. Never auto-executes."""
    engine = build_engine(settings)
    engine.generate_or_load_scenario()
    engine.scan_and_detect()
    engine.generate_plans()
    engine.digital_twin_validate()
    engine.request_approval()
    return engine


# --------------------------------------------------------------------------- #
# Background driver (sim mode)
# --------------------------------------------------------------------------- #


async def sim_driver() -> None:
    """Advance execution one placement at a time once the operator approves.

    It never approves on its own. If the plan is unapproved this loop simply
    idles — which is the visible, demonstrable form of the safety invariant.
    """
    with STATE.lock:
        STATE.engine = start_run(STATE.settings)
        STATE.events.clear()
        for event in STATE.engine.log.events():
            STATE.events.append(event.to_dict())
    while True:
        await asyncio.sleep(STEP_PERIOD_S)
        with STATE.lock:
            engine = STATE.engine
            if engine is None or not STATE.auto_step or engine.finished:
                continue
            try:
                engine.step_execution()
            except ApprovalRequired:
                # Expected while awaiting a decision — not an error.
                STATE.auto_step = False
            except Exception as exc:                    # noqa: BLE001
                STATE.notice = f"execution error: {exc}"
                STATE.auto_step = False
            if engine.finished:
                STATE.auto_step = False
                _write_artifacts_locked()


def _write_artifacts_locked() -> None:
    """Persist evidence at the end of a run. Caller holds the lock."""
    engine = STATE.engine
    if engine is None or engine.selected is None:
        return
    try:
        kpis = engine.kpis(latest_latency_p50_ms(RESULTS_DIR))
        artifacts = write_run_artifacts(
            engine.scenario, engine.baseline, engine.optimized, engine.selected,
            kpis, engine.log, RESULTS_DIR)
        write_validation_report(
            engine.scenario, engine.baseline, engine.optimized, engine.selected,
            kpis, engine.log, artifacts, RESULTS_DIR)
        STATE.notice = f"artefacts written: results/wisepack-run-{artifacts.stamp}.json"
    except Exception as exc:                            # noqa: BLE001
        STATE.notice = f"artefact write failed: {exc}"


# --------------------------------------------------------------------------- #
# FIWARE reader
# --------------------------------------------------------------------------- #


def _orion_get(path: str, timeout: float = 3.0):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(f"{ORION}{path}",
                                 headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception:                                   # noqa: BLE001
        return None, None


def fiware_health() -> Dict[str, Any]:
    """Is an Orion-LD (not a plain NGSI-v2 Orion) reachable?"""
    status, body = _orion_get("/version")
    if status is None:
        return {"connected": False, "reason": f"no response from {ORION}"}
    text = json.dumps(body or {}).lower()
    if "orionld" not in text and "orion" in text:
        return {"connected": False,
                "reason": "broker on :1026 is NGSI-v2 Orion, not Orion-LD"}
    return {"connected": True, "broker": ORION,
            "version": (body or {}).get("orionld version")
                       or (body or {}).get("version", "unknown")}


def fiware_entity(entity_id: str) -> Optional[Dict[str, Any]]:
    status, body = _orion_get(
        f"/ngsi-ld/v1/entities/{entity_id}?local=true")
    return body if status == 200 else None


def _attr(entity: Optional[Dict[str, Any]], name: str) -> Any:
    """Read ``<attr>.value.data`` — the shape the Orion-LD DDS bridge produces.

    Documented in HARMONY's dds_native_contract.md: the bridge wraps every
    std_msgs payload under ``value.data``, for String and Int32 alike. A plain
    ``value`` is also accepted so a hand-PATCHed entity still renders.
    """
    if not entity:
        return None
    attr = entity.get(name)
    if not isinstance(attr, dict):
        return None
    value = attr.get("value")
    if isinstance(value, dict) and "data" in value:
        return value["data"]
    if value == "uninitialized":
        return None
    return value


# --------------------------------------------------------------------------- #
# Topology
# --------------------------------------------------------------------------- #

# Position-free graph, exactly as TEMPO does it: a node declares what it IS and
# which layer it sits in, and the frontend derives coordinates. Adding a node
# needs no layout change here and no renderer change there.
TOPOLOGY = {
    "layers": ["perception", "planning", "supervision", "bus", "cloud"],
    "nodes": [
        {"id": "generator", "label": "Task generator", "kind": "Scenario",
         "role": "source", "layer": 0},
        {"id": "perception", "label": "Perception simulator", "kind": "Simulated",
         "role": "sim", "layer": 0},
        {"id": "optimizer", "label": "Packing optimizer", "kind": "Intelligence",
         "role": "brain", "layer": 1},
        {"id": "twin", "label": "Digital Twin validator", "kind": "Validation",
         "role": "check", "layer": 1},
        {"id": "orchestrator", "label": "HitL orchestrator", "kind": "py_trees",
         "role": "control", "layer": 2},
        {"id": "operator", "label": "Operator", "kind": "Human",
         "role": "human", "layer": 2},
        {"id": "robot", "label": "Robot simulator", "kind": "Simulated",
         "role": "sim", "layer": 2},
        {"id": "dds", "label": "ROS 2 / DDS", "kind": "Middleware",
         "role": "bus", "layer": 3},
        {"id": "orion", "label": "Orion-LD (NGSI-LD)", "kind": "Context broker",
         "role": "cloud", "layer": 4},
        {"id": "dashboard", "label": "Dashboard / analytics", "kind": "UI",
         "role": "cloud", "layer": 4},
    ],
    # c: "telemetry" renders solid, "control" renders dashed.
    "edges": [
        {"s": "generator", "t": "optimizer", "c": "telemetry"},
        {"s": "perception", "t": "optimizer", "c": "telemetry"},
        {"s": "optimizer", "t": "twin", "c": "telemetry"},
        {"s": "twin", "t": "orchestrator", "c": "telemetry"},
        {"s": "orchestrator", "t": "operator", "c": "telemetry"},
        {"s": "operator", "t": "orchestrator", "c": "control"},
        {"s": "orchestrator", "t": "robot", "c": "control"},
        {"s": "robot", "t": "orchestrator", "c": "telemetry"},
        {"s": "generator", "t": "dds", "c": "telemetry"},
        {"s": "orchestrator", "t": "dds", "c": "telemetry"},
        {"s": "robot", "t": "dds", "c": "telemetry"},
        {"s": "optimizer", "t": "dds", "c": "telemetry"},
        {"s": "dds", "t": "orion", "c": "telemetry"},
        {"s": "orion", "t": "dds", "c": "control"},
        {"s": "orion", "t": "dashboard", "c": "telemetry"},
        {"s": "dashboard", "t": "orion", "c": "control"},
    ],
}


def topology_status() -> Dict[str, str]:
    """Colour each node from the live workflow state."""
    with STATE.lock:
        engine = STATE.engine
        stage = engine.stage if engine else Stage.IDLE
        finished = engine.finished if engine else False
        degraded = stage is Stage.DEGRADED
        approved = (engine.selected.approval_state.value == "approved"
                    if engine and engine.selected else False)
        fiware_ok = STATE.fiware_connected

    def s(active: bool, warn: bool = False) -> str:
        if degraded:
            return "fault"
        if warn:
            return "standby"
        return "active" if active else "idle"

    planning = stage in (Stage.GENERATE_BASELINE_PLAN,
                         Stage.GENERATE_OPTIMIZED_PLAN, Stage.REPLAN)
    executing = stage in (Stage.PICK_ITEM, Stage.VERIFY_PICK, Stage.PLACE_ITEM,
                          Stage.VERIFY_PLACEMENT, Stage.UPDATE_CONTAINER_STATE,
                          Stage.NEXT_ITEM)
    return {
        "generator": s(stage is Stage.GENERATE_OR_LOAD_SCENARIO or finished),
        "perception": s(stage in (Stage.SCAN_SOURCE_BIN, Stage.DETECT_ITEMS)),
        "optimizer": s(planning),
        "twin": s(stage is Stage.DIGITAL_TWIN_VALIDATE),
        "orchestrator": s(True, warn=degraded),
        "operator": s(stage is Stage.WAIT_FOR_OPERATOR_APPROVAL,
                      warn=stage is Stage.WAIT_FOR_OPERATOR_APPROVAL),
        "robot": s(executing and approved),
        "dds": s(SOURCE in ("ros", "fiware")),
        "orion": ("active" if fiware_ok else
                  ("idle" if fiware_ok is None else "fault")),
        "dashboard": "active",
    }


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = []
    if SOURCE == "sim":
        tasks.append(asyncio.create_task(sim_driver()))
    else:
        from ros_observer import start_ros_observer                # noqa: PLC0415
        tasks.append(asyncio.create_task(start_ros_observer(STATE)))
    if SOURCE in ("ros", "fiware"):
        tasks.append(asyncio.create_task(_fiware_poller()))
    yield
    for task in tasks:
        task.cancel()


async def _fiware_poller() -> None:
    while True:
        health = await asyncio.get_event_loop().run_in_executor(None, fiware_health)
        with STATE.lock:
            STATE.fiware_connected = health["connected"]
            STATE.fiware_last_error = health.get("reason", "")
        await asyncio.sleep(5.0)


app = FastAPI(title="WISEPACK dashboard", lifespan=lifespan)


def _source_badge() -> Dict[str, Any]:
    """The badge is the honesty contract: it always names where data came from."""
    with STATE.lock:
        connected = STATE.fiware_connected
    if SOURCE == "sim":
        return {"source": "sim", "label": "SIMULATED",
                "detail": "no ROS, no FIWARE — same domain logic and optimizer "
                          "as live mode; execution outcomes are simulated",
                "live": False}
    if SOURCE == "fiware":
        return {"source": "fiware+ros" if connected else "ros",
                "label": "FIWARE" if connected else "ROS (FIWARE unreachable)",
                "detail": "state read back from Orion-LD over NGSI-LD; "
                          "high-frequency animation still comes from ROS",
                "live": True}
    return {"source": "ros", "label": "ROS 2 / DDS",
            "detail": "live ROS 2 topics over Fast DDS; robot outcomes remain "
                      "simulated",
            "live": True}


@app.get("/api/state")
def api_state():
    """Complete initial state. The page renders fully from this one call."""
    with STATE.lock:
        engine = STATE.engine
        snapshot = engine.snapshot() if engine else {}
        auto = STATE.auto_step
        notice = STATE.notice
        settings = dict(STATE.settings)
    return {
        "badge": _source_badge(),
        "stage": snapshot.get("stage", Stage.IDLE.value),
        "auto_step": auto,
        "notice": notice,
        "settings": settings,
        "presets": sorted(PRESETS),
        "container_specs": {k: v["description"] for k, v in CONTAINER_SPECS.items()},
        "strategies": [s.value for s in Strategy],
        "fiware": {"connected": STATE.fiware_connected,
                   "error": STATE.fiware_last_error, "broker": ORION},
        "topology_status": topology_status(),
        "ros_mirror": STATE.ros_mirror,
        **snapshot,
        "ts": time.time(),
    }


@app.get("/api/plans")
def api_plans():
    """Geometry for the Digital Twin view and the baseline/optimized comparison."""
    with STATE.lock:
        engine = STATE.engine
        if engine is None or engine.baseline is None:
            return JSONResponse({"ready": False}, status_code=200)
        return {
            "ready": True,
            "scenario": engine.scenario.to_dict(),
            # baseline/optimized are the comparison pair over the full batch;
            # `selected` is the execution plan and is what the twin renders, so
            # already-executed placements show as executed.
            "baseline": engine.baseline.to_dict(),
            "optimized": engine.optimized.to_dict(),
            "selected": engine.selected.to_dict() if engine.selected else None,
            "selected_plan_id": engine.selected.plan_id if engine.selected else None,
            "selection_reason": engine.selection_reason,
        }


@app.get("/api/kpis")
def api_kpis():
    with STATE.lock:
        engine = STATE.engine
        if engine is None or engine.selected is None:
            return {"ready": False}
        report = engine.kpis(latest_latency_p50_ms(RESULTS_DIR))
        return {"ready": True, **report.to_dict()}


@app.get("/api/strategies")
def api_strategies():
    """Run all three operator-selectable strategies and compare them."""
    with STATE.lock:
        engine = STATE.engine
        if engine is None or engine.scenario is None:
            return {"ready": False}
        plans = engine.compare_strategies()
    return {"ready": True, "rows": compare_strategies(plans),
            "plans": {k: v.to_dict() for k, v in plans.items()}}


@app.get("/api/events")
def api_events(limit: int = 150):
    return {"events": STATE.recent(limit), "total": len(STATE.events)}


@app.get("/api/analytics")
def api_analytics():
    """Aggregations for the analytics panel."""
    with STATE.lock:
        engine = STATE.engine
        if engine is None:
            return {"ready": False}
        log = engine.log
        return {
            "ready": True,
            "by_action": log.by_action(),
            "by_stage": log.by_stage(),
            "duration_by_stage_ms": log.duration_by_stage_ms(),
            "sequence_ok": log.sequence_is_monotonic()[0],
            "total_events": log.count,
            "replans": engine.stats.replans,
            "replan_causes": list(engine.stats.replan_causes),
            "pick_attempts": engine.stats.pick_attempts,
            "pick_successes": engine.stats.pick_successes,
            "latency": latest_artifact("dds-fiware-latency", RESULTS_DIR),
        }


@app.get("/api/topology")
def api_topology():
    return {**TOPOLOGY, "status": topology_status()}


@app.get("/api/fiware")
def api_fiware():
    """Live NGSI-LD read-back — the proof that the audit path works."""
    if SOURCE == "sim":
        raise HTTPException(
            status_code=409,
            detail="sim mode has no FIWARE. Start with --source ros or fiware.")
    from wisepack_fiware.entities import ENTITY_IDS                # noqa: PLC0415
    out = {}
    for name, entity_id in ENTITY_IDS.items():
        entity = fiware_entity(entity_id)
        out[name] = {
            "entity_id": entity_id,
            "present": entity is not None,
            "attributes": {k: _attr(entity, k) for k in entity
                           if k not in ("id", "type", "@context")}
            if entity else {},
        }
    return {"broker": ORION, "health": fiware_health(), "entities": out}


# --------------------------------------------------------------------------- #
# Operator commands
# --------------------------------------------------------------------------- #


@app.post("/api/command")
async def api_command(payload: Dict[str, Any]):
    """Operator actions.

    In live (ros/fiware) mode these do NOT mutate Python state directly: they are
    published on the documented operator command path
    (/wisepack/operator/approval, /wisepack/operator/command), which reaches the
    orchestrator over DDS exactly as an NGSI-LD PATCH from an external HMI would.
    Mutating the engine in-process would demo a control path that does not exist.
    """
    command = str(payload.get("command", "")).strip()
    args = payload.get("args", {}) or {}

    if SOURCE != "sim":
        from ros_observer import publish_operator_command           # noqa: PLC0415
        ok, detail = publish_operator_command(command, args)
        return {"ok": ok, "command": command, "path": "ros2 -> dds -> orchestrator",
                "detail": detail}

    with STATE.lock:
        engine = STATE.engine
        if engine is None:
            raise HTTPException(status_code=409, detail="no run in progress")
        try:
            return _apply_command_locked(engine, command, args)
        except (ApprovalRequired, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


def _apply_command_locked(engine: WorkflowEngine, command: str,
                          args: Dict[str, Any]) -> Dict[str, Any]:
    if command == "approve":
        engine.approve(operator=str(args.get("operator", "dashboard operator")))
        STATE.auto_step = True
        return {"ok": True, "stage": engine.stage.value}

    if command == "reject":
        engine.reject(reason=str(args.get("reason", "rejected from dashboard")))
        STATE.auto_step = False
        return {"ok": True, "stage": engine.stage.value,
                "replans": engine.stats.replans}

    if command == "alternative_strategy":
        strategy = Strategy(args.get("strategy", "retrievability"))
        engine.config.strategy = strategy
        engine.stats.operator_interventions += 1
        engine.generate_plans(strategy)
        engine.digital_twin_validate()
        engine.request_approval()
        STATE.auto_step = False
        return {"ok": True, "strategy": strategy.value,
                "containers": engine.optimized.containers_required}

    if command == "inject_item":
        engine.apply_dynamic_event(DynamicEvent(
            event_type=DynamicEventType.ITEM_INJECT,
            trigger=f"placement:{engine.cursor.index}",
            label=str(args.get("label", "Operator-injected component")),
            payload={"item": args.get("item", {
                "length_mm": 1100, "outer_diameter_mm": 200,
                "inner_diameter_mm": 170, "priority": 9, "dose_class": "ILW"})}))
        STATE.auto_step = False
        return {"ok": True, "replans": engine.stats.replans}

    if command == "container_unavailable":
        container_id = args.get("container_id") or (
            engine.selected.containers_used[0].container_id
            if engine.selected and engine.selected.containers_used else None)
        if not container_id:
            raise ValueError("no container to mark unavailable")
        engine.apply_dynamic_event(DynamicEvent(
            event_type=DynamicEventType.CONTAINER_UNAVAILABLE,
            trigger=f"placement:{engine.cursor.index}",
            label=f"{container_id} taken out of service",
            payload={"container_id": container_id}))
        STATE.auto_step = False
        return {"ok": True, "container_id": container_id}

    if command == "grasp_failure":
        engine.apply_dynamic_event(DynamicEvent(
            event_type=DynamicEventType.GRASP_FAILURE,
            trigger=f"placement:{engine.cursor.index}",
            label="Operator-injected grasp failure"))
        return {"ok": True}

    if command == "pause":
        STATE.auto_step = False
        return {"ok": True, "auto_step": False}

    if command == "resume":
        if engine.selected and engine.selected.approval_state.value == "approved":
            STATE.auto_step = True
            return {"ok": True, "auto_step": True}
        raise ApprovalRequired("cannot resume: the current plan is not approved")

    if command == "step":
        engine.step_execution()
        return {"ok": True, "progress_pct": round(engine.progress_pct, 1)}

    if command == "reset":
        STATE.settings.update({k: v for k, v in args.items()
                               if k in STATE.settings})
        STATE.events.clear()
        STATE.notice = ""
        STATE.engine = start_run(STATE.settings)
        STATE.auto_step = False
        for event in STATE.engine.log.events():
            STATE.events.append(event.to_dict())
        return {"ok": True, "scenario_id": STATE.engine.scenario.scenario_id}

    if command == "write_artifacts":
        _write_artifacts_locked()
        return {"ok": True, "notice": STATE.notice}

    raise ValueError(f"unknown command {command!r}")


# --------------------------------------------------------------------------- #
# WebSocket (enhancement only — the page works on polling alone)
# --------------------------------------------------------------------------- #


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    try:
        while True:
            await sock.send_json({
                "type": "tick",
                "state": api_state(),
                "events": STATE.recent(12),
            })
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:                                   # noqa: BLE001
        # A failed socket must degrade to polling, never take the server down.
        pass


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as fh:
        return fh.read()


@app.get("/healthz")
def healthz():
    with STATE.lock:
        return {"ok": True, "source": SOURCE,
                "stage": STATE.engine.stage.value if STATE.engine else "idle"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["sim", "ros", "fiware"],
                        default=os.environ.get("WISEPACK_SOURCE", "sim"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("WISEPACK_DASH_PORT", "8080")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--preset", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    os.environ["WISEPACK_SOURCE"] = args.source
    SOURCE = args.source
    STATE.source = args.source
    if args.preset:
        STATE.settings["preset"] = args.preset
    if args.seed is not None:
        STATE.settings["seed"] = args.seed

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
