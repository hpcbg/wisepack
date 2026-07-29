"""Diagnostics aggregation — read-only, allowlisted, secret-free.

This backs the /diagnostics page. It is for local engineering, interview
transparency and debugging, and it is built to expose NOTHING sensitive:

  * no environment-variable dumps, credentials, tokens, keys or file contents;
  * no Docker socket, no arbitrary `ros2` commands, no shell execution;
  * container facts come only from a host-generated JSON file
    (scripts/collect_runtime_status.sh) restricted to WISEPACK container names;
  * topics and components are FIXED ALLOWLISTS derived from the contract.

Everything it reports is either the canonical topic contract, the live snapshot
already computed for the dashboard, the parsed bridge configuration, or the
allowlisted runtime-status file. It introduces no new source of truth.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO, "results")
BRIDGE_YAML = os.path.join(REPO, "wisepack_ws", "src", "wisepack_fiware",
                           "config", "bridge_config.yaml")

_START_MONOTONIC = time.monotonic()

# --------------------------------------------------------------------------- #
# Allowlisted component roster
# --------------------------------------------------------------------------- #

#: (key, label, kind, role). kind is measured / simulated / external.
COMPONENTS = [
    ("task_generator", "Task generator", "measured", "seeded scenario source"),
    ("perception_sim", "Perception simulator", "simulated", "detections from ground truth"),
    ("optimizer", "Packing optimizer", "measured", "geometry-aware bin packing"),
    ("twin_validator", "Digital Twin validator", "measured", "independent placement validation"),
    ("orchestrator", "HitL orchestrator", "measured", "py_trees workflow authority"),
    ("robot_sim", "Robot simulator", "simulated", "seeded pick/place outcomes"),
    ("anomaly_sim", "Anomaly simulator / adapter", "simulated",
     "Structured anomaly ingestion, workflow response and FIWARE analytics integration"),
    ("dashboard", "Dashboard", "measured", "read model + operator commands"),
    ("orion", "Orion-LD", "external", "NGSI-LD context broker"),
    ("mongo", "Mongo-DDS", "external", "Orion-LD datastore"),
]

#: The simulated / unavailable / future interfaces table (Section 6.6). This is
#: the honesty surface: deliberate simulation must never read as a failure.
INTERFACES = [
    ("RGB-D camera frames", "future interface", "No physical camera in the demo"),
    ("Object detections", "simulated source", "Generated from scenario ground truth"),
    ("6D pose estimates", "simulated source", "Not produced by a real CV backend"),
    ("Robot joint states", "simulated source", "No physical robot"),
    ("MoveIt2 trajectory", "future interface", "Not implemented in the interview demo"),
    ("Cutting anomaly detector", "simulated adapter", "Architecture demonstration only"),
    ("Packing optimizer", "measured software", "Real algorithm"),
    ("Digital Twin validator", "measured software", "Real independent validator"),
    ("FIWARE event mapping", "live", "Real DDS-to-Orion-LD path"),
]


# --------------------------------------------------------------------------- #
# Topic diagnostics — fixed allowlist, statuses from the live mirror
# --------------------------------------------------------------------------- #

def _topic_statuses(mode: str, mirror: Optional[Dict[str, Any]],
                    fiware_mapped: Dict[str, str]) -> List[Dict[str, Any]]:
    """Classify each canonical topic. No arbitrary `ros2` calls — uses the
    contract and the observer mirror the dashboard already maintains."""
    from wisepack_bringup import topics as T                       # local import

    contract = T.all_topics()
    # Which mirror key (if any) reflects a topic having produced data.
    seen: Dict[str, bool] = {}
    if mirror:
        seen = {
            T.EXECUTION_STATE: bool(mirror.get("stage") and mirror["stage"] != "IDLE"),
            T.PLAN_BASELINE: mirror.get("baseline") is not None,
            T.PLAN_OPTIMIZED: mirror.get("optimized") is not None,
            T.PLAN_SELECTED: mirror.get("selected") is not None,
            T.PLAN_SUMMARY: mirror.get("plan_summary") is not None,
            T.PLAN_STRATEGY_COMPARISON: mirror.get("strategy_comparison") is not None,
            T.SCENARIO_STATE: mirror.get("scenario") is not None,
            T.WASTE_ITEMS: bool(mirror.get("items")),
            T.ACTION_SEQUENCE: bool(mirror.get("action_sequence")),
            T.SYSTEM_HEARTBEAT: bool(mirror.get("heartbeat")),
            T.ANOMALY_STATE: mirror.get("anomaly") is not None,
            T.CUTTING_PROPOSAL: mirror.get("cut") is not None,
            T.INVENTORY_SUMMARY: mirror.get("inventory_summary") is not None,
            T.INVENTORY_CONTAINER_STATE: bool(mirror.get("inventory_containers")),
            T.LOGISTICS_CONTAINER_TASK: mirror.get("logistics_map") is not None,
            T.LOGISTICS_MOBILE_ROBOT_STATE: mirror.get("logistics_robot") is not None,
        }

    # Topics whose source is a simulator, and future interfaces.
    simulated = {T.WASTE_DETECTED_COUNT, T.ANOMALY_EVENT, T.ANOMALY_STATE,
                 T.ANOMALY_EXTERNAL, T.CUTTING_PROPOSAL, T.CUTTING_STATE,
                 T.CUTTING_RESULT, T.CUTTING_REQUEST, T.LOGISTICS_CONTAINER_TASK,
                 T.LOGISTICS_CONTAINER_TASK_STATE, T.LOGISTICS_MOBILE_ROBOT_STATE}
    rows = []
    for topic, ros_type in sorted(contract.items()):
        mapped = fiware_mapped.get(topic)
        if mode == "sim":
            status = "NOT EXPECTED IN THIS MODE"       # sim has no ROS graph
        elif topic in seen:
            status = "ACTIVE" if seen[topic] else "WAITING"
        elif topic in simulated:
            status = "SIMULATED SOURCE"
        else:
            status = "WAITING"
        rows.append({
            "topic": topic,
            "type": ros_type,
            "fiware": mapped or ("UNMAPPED TO FIWARE"
                                 if topic not in T.INBOUND_TOPICS else "inbound"),
            "expected_source": ("orchestrator" if topic.startswith("/wisepack/plan")
                                or topic.startswith("/wisepack/execution")
                                or topic.startswith("/wisepack/kpi")
                                else "perception/anomaly/operator"),
            "status": status,
        })
    return rows


def _fiware_mappings() -> List[Dict[str, Any]]:
    """Parse the generated bridge configuration into a mapping table."""
    try:
        import yaml
        with open(BRIDGE_YAML, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except Exception:                                   # noqa: BLE001
        return []
    rows = []
    for direction, block in (("ros->fiware", "ros_to_fiware"),
                             ("fiware->ros", "fiware_to_ros")):
        for m in cfg.get(block, []) or []:
            etype = m.get("fiware_entity_type") or m["fiware_entity"].split(":", 1)[0]
            eid = m["fiware_entity"]
            urn = (f"urn:ngsi-ld:{eid}" if eid.split(":", 1)[0] == etype
                   else f"urn:ngsi-ld:{etype}:{eid}")
            rows.append({
                "ros_topic": m["ros_topic"],
                "ros_type": m.get("ros_msg_type", ""),
                "direction": direction,
                "entity": urn,
                "attribute": m["fiware_attribute"],
                "status": "mapped",
            })
    return rows


def _mapped_topics() -> Dict[str, str]:
    rows = _fiware_mappings()
    out = {}
    for r in rows:
        out[r["ros_topic"]] = f"{r['entity'].split(':')[-2]}.{r['attribute']}"
    return out


# --------------------------------------------------------------------------- #
# Runtime status file (host-generated, allowlisted)
# --------------------------------------------------------------------------- #

#: The two startup-status files, written by the two launchers that own
#: processes. Read-only here; see scripts/startup_status.py for the schema.
#:
#: The WRITER tells the reader where it wrote. `results/` is frequently owned by
#: root — the container writes into it — and the host launcher then falls back
#: to a temporary file. Hard-coding the results path lost the entire host half
#: of this table whenever that happened, silently.
STARTUP_FILES = (("host", "startup-host.json"), ("stack", "startup-stack.json"))
STARTUP_ENV = {"host": "WISEPACK_HOST_STATUS", "stack": "WISEPACK_STARTUP_STATUS"}

#: Every process the stack is expected to own, and which scope reports it. A
#: name that never reports is shown as "expected, not reported" rather than
#: omitted — a process that died before its first write is exactly the case
#: this table exists for, and an omitted row looks like a process nobody wanted.
EXPECTED_PROCESSES = [
    ("dashboard", "stack", "the FastAPI application serving this page"),
    ("ros-launch", "stack", "ros2 launch wisepack_bringup demo.launch.py"),
    ("orchestrator", "stack", "HitL orchestrator — owns the workflow engine"),
    ("perception-sim", "stack", "perception simulator"),
    ("twin-validator", "stack", "Digital Twin validator"),
    ("anomaly-simulator", "stack", "SIMULATED anomaly source"),
    ("isaac-sim", "host", "Isaac Sim process group on the host"),
    ("isaac-watcher", "host", "launcher-side Isaac readiness watcher"),
    ("wisepack-container", "host", "the container this stack runs in"),
]


def _startup_status() -> Dict[str, Any]:
    """Merge the host and container startup-status files. Never raises.

    Two writers, two files, merged for reading — so neither launcher has to
    coordinate with the other and a missing file is simply a scope that has not
    reported. `docker ps` is not consulted: the question this answers is
    "did the processes inside actually start", which a container's state cannot.
    """
    out: Dict[str, Any] = {"scopes": {}, "processes": [], "degraded": False,
                           "degraded_reason": "", "robot": {}}
    reported: Dict[str, Dict[str, Any]] = {}
    for scope, name in STARTUP_FILES:
        path = os.environ.get(STARTUP_ENV.get(scope, ""), "") \
            or os.path.join(RESULTS_DIR, name)
        if not os.path.isfile(path):
            out["scopes"][scope] = "not reported"
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as exc:
            # The TYPE, never the message: an exception from a file operation
            # carries the absolute path, and this payload is served over HTTP.
            # Diagnostics reports state, not this host's filesystem.
            out["scopes"][scope] = f"unreadable ({type(exc).__name__})"
            continue
        out["scopes"][scope] = (f"reported {doc.get('generated_at', '?')} "
                                f"(age {round(time.time() - os.path.getmtime(path), 1)}s)")
        if doc.get("degraded"):
            out["degraded"] = True
            reason = doc.get("degraded_reason") or f"{scope} reported DEGRADED"
            out["degraded_reason"] = "; ".join(
                filter(None, [out["degraded_reason"], reason]))
        if doc.get("robot"):
            # EMPTY VALUES DO NOT OVERWRITE. Both scopes report a robot block
            # and the container's is deliberately sparse — it knows the id it
            # was given, not the registry it came from. A plain update() let
            # those blanks erase the launcher's richer answer, which is how
            # "configured default" rendered as an em dash on a healthy run.
            out["robot"].update({k: v for k, v in doc["robot"].items() if v})
        for entry in doc.get("processes", []) or []:
            if entry.get("name"):
                reported[entry["name"]] = {**entry, "scope": scope}

    for name, scope, role in EXPECTED_PROCESSES:
        entry = reported.pop(name, None)
        if entry is None:
            out["processes"].append({
                "process": name, "scope": scope, "role": role,
                "pid": "—", "expected": "yes", "running": "not reported",
                "exit_code": "—", "last_heartbeat": "—", "last_error": "—"})
            continue
        running = entry.get("running")
        out["processes"].append({
            "process": name, "scope": entry.get("scope", scope), "role": role,
            "pid": entry.get("pid") if entry.get("pid") is not None else "—",
            "expected": "yes" if entry.get("expected", True) else "no",
            "running": ("unknown" if running is None
                        else ("yes" if running else "NO")),
            "exit_code": (entry.get("exit_code")
                          if entry.get("exit_code") is not None else "—"),
            "last_heartbeat": entry.get("last_heartbeat") or "—",
            "last_error": entry.get("last_error") or "—",
        })
    # Anything reported but not on the roster still gets shown; a process the
    # allowlist has not caught up with is information, not noise.
    for name, entry in reported.items():
        out["processes"].append({
            "process": name, "scope": entry.get("scope", "?"), "role": "—",
            "pid": entry.get("pid", "—"), "expected": "unlisted",
            "running": "yes" if entry.get("running") else "NO",
            "exit_code": entry.get("exit_code", "—"),
            "last_heartbeat": entry.get("last_heartbeat") or "—",
            "last_error": entry.get("last_error") or "—"})
    return out


def _startup_blocker(snap, startup: Dict[str, Any], isaac: Dict[str, Any]) -> str:
    """WHY there is no run yet, in one sentence. "" once one exists.

    IDLE on its own is not a diagnosis. It is the state a stack sits in when it
    is starting normally, when its ROS launch died on the first line, and when
    it is waiting for an operator — and an operator cannot tell those apart from
    the word alone.
    """
    if snap.run_id:
        return ""
    if startup.get("degraded"):
        return (startup.get("degraded_reason")
                or "a launcher reported the stack DEGRADED")
    dead = [p["process"] for p in startup.get("processes", [])
            if p.get("running") == "NO" and p.get("expected") == "yes"]
    if dead:
        return (f"{', '.join(dead)} is not running — the stack cannot create a "
                "run without it")
    missing = [p["process"] for p in startup.get("processes", [])
               if p.get("running") == "not reported"
               and p["process"] in ("ros-launch", "orchestrator")]
    if missing:
        return (f"{', '.join(missing)} has not reported — the ROS stack may not "
                "have started")
    if isaac and not isaac.get("simulator_ready"):
        return ("waiting for Isaac Sim to report SIMULATOR_READY; the run is "
                "created independently and approval stays disabled until the "
                "scene is acknowledged for it")
    return "no run has been created yet"


def _runtime_status() -> Optional[Dict[str, Any]]:
    path = os.path.join(RESULTS_DIR, "runtime-status.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    doc["_age_s"] = round(time.time() - os.path.getmtime(path), 1)
    return doc


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #

def build(snap, mode: str, mirror: Optional[Dict[str, Any]],
          latency_artifact: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble the diagnostics report from allowlisted, read-only sources."""
    mapped = _mapped_topics()
    analytics = snap.to_analytics()
    scenario = snap.scenario or {}

    # -- runtime overview (6.1) ------------------------------------------- #
    heartbeat_age = None
    if mirror and mirror.get("heartbeat_at"):
        heartbeat_age = round(time.time() - mirror["heartbeat_at"], 1)
    overview = {
        "mode": mode,
        "app_uptime_s": round(time.monotonic() - _START_MONOTONIC, 1),
        "run_id": snap.run_id,
        "scenario_id": scenario.get("scenario_id"),
        "scenario_revision": snap.scenario_revision,
        "stage": snap.stage,
        "current_item": snap.current_item_id,
        "current_container": snap.current_container_id,
        "selected_plan_id": snap.selected_plan_id,
        "approval_state": snap.approval_state,
        "heartbeat_age_s": heartbeat_age,
        "action_sequence": snap.action_count,
        "sequence_gap_free": analytics.get("sequence_ok"),
        "panel_sources": snap.panel_sources,
        "anomaly_hold": (snap.anomaly or {}).get("hold") if snap.anomaly else False,
        # CANONICAL control state, beside the display values above. In FIWARE
        # mode `stage` and `approval_state` are the Orion-LD echo — the audit
        # proof, and deliberately lagged — while these are what the controls are
        # actually enabled from. Reading one row of this page and seeing
        # "WAIT_FOR_OPERATOR_APPROVAL / approved" was the report that started
        # this; showing both makes a lag distinguishable from a contradiction.
        "control_stage": snap.control_stage,
        "control_approval_state": snap.control_approval_state,
        "control_plan_id": snap.control_plan_id,
        "approval_revision": snap.control_approval_revision,
        "approval_plan_id": snap.control_approval_plan_id,
        "state_consistent": not snap.control_inconsistency,
        "inconsistency": snap.control_inconsistency or "—",
    }

    # -- ROS/FIWARE run correlation ---------------------------------------- #
    #
    # Reachable is not current. Orion-LD keeps every attribute's last value
    # across scenario changes and whole runs, so "connected" says nothing about
    # whether what it returned describes the run on screen. These rows are what
    # make a bridge lag distinguishable from a stale projection — and they were
    # the only way to see, rather than infer, that KPI cards reading 1/1 came
    # from a previous isaac_cylinders_smoke run.
    correlation = {
        "canonical_run_id": snap.run_id,
        "fiware_run_id": snap.fiware_run_id,
        "canonical_scenario_revision": snap.scenario_revision,
        "fiware_scenario_revision": snap.fiware_scenario_revision,
        "canonical_stage": snap.control_stage,
        "fiware_stage": snap.fiware_stage,
        "fiware_sync_status": snap.fiware_sync_status,
        "fiware_sync_detail": snap.fiware_sync_detail or "—",
        "rejected_stale_fields": (
            ", ".join(f"{r['entity']}.{r['field']}"
                      for r in snap.rejected_stale_fields) or "none"),
    }
    overview.update(correlation)

    # -- physical scene readiness ------------------------------------------ #
    #
    # TWO LEVELS, reported separately and never collapsed. "Isaac is up" and
    # "the world in front of the robot is the one this run planned against" are
    # different claims; only the second authorises a pick, and only showing both
    # makes the difference visible when a correct-looking scene is nonetheless
    # unacknowledged.
    # -- startup: what actually started, and what died --------------------- #
    startup = _startup_status()
    isaac = snap.isaac or {}
    blocker = _startup_blocker(snap, startup, isaac)
    overview["startup_scopes"] = ", ".join(
        f"{k}: {v}" for k, v in sorted(startup["scopes"].items())) or "—"
    overview["startup_degraded"] = "YES" if startup["degraded"] else "no"
    if startup["degraded_reason"]:
        overview["startup_degraded_reason"] = startup["degraded_reason"]
    if blocker:
        # SHOWN INSTEAD OF BARE IDLE. An operator reading "IDLE" cannot tell a
        # stack that is starting from one whose launch process exited.
        overview["no_active_run_because"] = blocker
    overview.update(_robot_startup_rows(startup, isaac, snap))

    if isaac:
        overview.update({
            "simulator_process": ("ready" if isaac.get("simulator_ready")
                                  else "not ready"),
            "ros_bridge": ("ready" if isaac.get("ros_bridge_ready")
                           else "not ready"),
            "requested_scene_revision": isaac.get("required_revision"),
            "acknowledged_scene_revision": isaac.get("scene_revision"),
            "expected_object_count": isaac.get("expected_object_count"),
            "actual_object_count": isaac.get("actual_object_count"),
            # Truncated for readability; a digest is compared, not read.
            "expected_fingerprint": str(isaac.get("requested_fingerprint") or "")[:16] or "—",
            "acknowledged_fingerprint": str(
                isaac.get("acknowledged_fingerprint") or "")[:16] or "—",
            "scene_status": isaac.get("scene_status", "unknown"),
            "scene_mismatch": isaac.get("scene_mismatch") or "—",
            "simulator_version": isaac.get("simulator_version") or "—",
        })
        overview.update(_robot_rows(snap, isaac))

    # -- component status (6.2) — allowlisted ------------------------------ #
    rt = _runtime_status()
    rt_by_role = {}
    if rt:
        for c in rt.get("containers", []):
            rt_by_role[c.get("known_role", "")] = c
    live_stage = snap.stage
    components = []
    for key, label, kind, role in COMPONENTS:
        health = "unknown"
        if key in ("orion", "mongo"):
            cont = rt_by_role.get(key)
            health = (cont or {}).get("health") or (cont or {}).get("state") or (
                "healthy" if (mode == "fiware" and snap.fiware_connected) else "unknown")
        elif mode == "sim":
            health = "n/a (no ROS in sim)" if key not in (
                "optimizer", "twin_validator", "task_generator", "dashboard",
                "robot_sim", "perception_sim") else "in-process"
        else:
            health = "active" if live_stage not in ("IDLE", "") else "starting"
        components.append({
            "key": key, "label": label, "kind": kind, "role": role,
            "health": health,
        })

    # -- timing (6.4) — labelled ------------------------------------------- #
    durations = analytics.get("duration_by_stage_ms", {})
    latency = None
    if latency_artifact:
        summ = latency_artifact.get("summary", {})
        r2f = summ.get("ros_to_fiware", {})
        latency = {
            "ros_to_fiware_p50_ms": r2f.get("median"),
            "ros_to_fiware_p95_ms": r2f.get("p95"),
            "ros_to_fiware_max_ms": r2f.get("max"),
            "samples": r2f.get("samples"),
            "source": "measured",
        }
    timing = {
        "duration_by_stage_ms": durations,
        "optimization_time_ms": (snap.kpis.get("optimization_time_ms", {}) or {}).get("value"),
        "anomaly_to_hold_latency_ms": (snap.anomaly or {}).get(
            "anomaly_to_hold_latency_ms") if snap.anomaly else None,
        "dds_to_fiware_latency": latency,
        "labels": "durations are measured/simulated per action source; "
                  "latency is measured when the benchmark has run, else unavailable",
    }

    # -- cutting / inventory / logistics status (brief §19) ---------------- #
    wp = snap.to_whole_process()
    cut = wp.get("cut") or {}
    rec = cut.get("no_cut") and next(
        (a for a in ([cut.get("no_cut")] + cut.get("alternatives", []))
         if a.get("label") == cut.get("recommended_label")), None) or {}
    cutting_status = {
        "cut_aware_mode": bool(cut),
        "eligible_items": len(cut.get("pipes_considered", [])),
        "candidate_count": cut.get("candidates_evaluated", 0),
        "recommend_cut": cut.get("recommend_cut"),
        "selected_proposal": cut.get("selected_label"),
        "cut_skill_state": cut.get("cut_skill_state"),
        "cut_approval": cut.get("cut_approval_state"),
        "plan_revision": cut.get("plan_revision"),
        "approval_revision": cut.get("approval_revision"),
        "cut_validation": (rec.get("proposals", [{}])[0].get("validator_result", {})
                           .get("valid") if rec.get("proposals") else None),
        "lineage_consistent": all(
            (r.get("validation", {}).get("valid", True))
            for r in cut.get("cut_results", [])) if cut.get("cut_results") else None,
        "conservation_ok": (cut.get("latest_cut_result", {}) or {}).get(
            "validation", {}).get("valid") if cut.get("latest_cut_result") else None,
    }
    inv = wp.get("inventory") or {}
    inv_summary = inv.get("summary", {})
    inventory_status = {
        "container_entity_count": len(inv.get("containers", [])),
        "available": inv_summary.get("available"),
        "reserved": inv_summary.get("reserved"),
        "available_capacity_mm3": inv_summary.get("compatible_capacity_available_mm3"),
        "forecast_shortage": inv_summary.get("forecast_shortage"),
        "shortage_events": inv_summary.get("shortage_events"),
        "invalid_transitions": sum(
            1 for c in inv.get("containers", [])
            for h in c.get("history", []) if h.get("rejected")),
        "revision": inv_summary.get("revision"),
        "fiware_synchronised": (mode != "sim"),
        "source": inv_summary.get("source"),
    }
    log = wp.get("logistics") or {}
    log_an = log.get("analytics", {}) or {}
    robot = log.get("robot", {}) or {}
    logistics_status = {
        "pending_tasks": len((log.get("facility_map", {}) or {}).get(
            "pending_tasks", [])),
        "active_task": (log.get("facility_map", {}) or {}).get("active_task"),
        "simulated_robot_state": robot.get("status"),
        "robot_location": robot.get("location"),
        "failed_tasks": log_an.get("failed_tasks"),
        "avg_task_duration_ticks": log_an.get("avg_task_duration_ticks"),
        "ticks_elapsed": log_an.get("ticks_elapsed"),
        "label": "SIMULATED — no physical mobile robot",
    }

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overview": overview,
        "components": components,
        "cutting_status": cutting_status,
        "inventory_status": inventory_status,
        "logistics_status": logistics_status,
        "startup_processes": startup["processes"],
        "topics": _topic_statuses(mode, mirror, mapped),
        "interfaces": [
            {"interface": i, "state": s, "meaning": m} for i, s, m in INTERFACES],
        "fiware_mappings": _fiware_mappings(),
        "timing": timing,
        "runtime_status": rt,
        "security_note": (
            "Read-only. No environment, credentials, tokens, keys, Docker socket "
            "or shell access are exposed. Container facts come only from an "
            "allowlisted host-generated file."),
    }


def _robot_startup_rows(startup: Dict[str, Any], isaac: Dict[str, Any],
                        snap) -> Dict[str, Any]:
    """Robot resolution, end to end, so a mismatch names itself.

    The value is followed from where it was DECIDED (the launcher, against the
    registry) to where it is USED (the host simulator, the orchestrator, the
    scene request, the acknowledgement). A disagreement anywhere along that
    chain is the difference between an arm that moves and an approval gate that
    never opens, and it is invisible if only the end of the chain is reported.
    """
    robot = startup.get("robot") or {}
    isaac_robot = dict(isaac.get("robot_status") or {})
    ack = isaac.get("acknowledged_scene") or {}
    switch = dict(isaac.get("robot_switch") or {})
    rows = {}
    if switch:
        # FOUR ANSWERS, never collapsed into one. They agree except while a
        # switch is in flight or has failed, and that is precisely when a single
        # number would tell an operator the new arm is running when it is not.
        rows.update({
            "robot_switch_available": ("yes" if switch.get("available")
                                       else switch.get("unavailable_reason")
                                       or "no"),
            "robot_switch_phase": switch.get("phase_label")
            or switch.get("phase") or "—",
            "robot_switch_in_flight": "YES" if switch.get("in_flight") else "no",
            "robot_switch_requested": switch.get("requested_robot_id") or "—",
            "robot_switch_previous": switch.get("previous_robot_id") or "—",
            "robot_host_reported": switch.get("host_robot_id") or "—",
            "robot_host_generation": switch.get("host_generation") or "—",
            "robot_expected_generation": switch.get("expected_generation") or "—",
            "robot_acknowledged_generation": (
                isaac.get("acknowledged_simulator_generation") or "—"),
            "robot_switch_failed": (switch.get("failed_reason") or "no"
                                    if switch.get("failed") else "no"),
        })
    rows.update({
        "robot_registry_loaded": ("yes" if robot.get("registry_loaded")
                                  else "not reported"),
        # RESOLVED, not WHERE. The launcher already printed the path on the
        # terminal it was run from; publishing it here would put this host's
        # filesystem layout in an HTTP response for no operator benefit.
        "robot_registry_resolved": (os.path.basename(robot["registry_path"])
                                    if robot.get("registry_path") else "no"),
        "robot_configured_default": robot.get("registry_default") or "—",
        # AT STARTUP — not "now". A robot switch replaces the simulator, so this
        # is the arm the launcher resolved when the stack came up and
        # `robot_host_reported` is the one running. They differ legitimately
        # after any switch, and labelling this one "effective" without saying
        # when made a completed switch look like a disagreement.
        "robot_effective_at_startup": robot.get("effective")
        or "— (logical run: none)",
        "robot_selection_source": robot.get("source") or "—",
        "robot_startup_profile_revision": robot.get("profile_revision") or "—",
        "robot_host_id": isaac_robot.get("robot_id") or "—",
        "robot_orchestrator_id": isaac.get("robot_id") or "—",
        "robot_requested_scene_id": isaac.get("robot_id") or "—",
        "robot_acknowledged_scene_id": (ack.get("robot_id")
                                        or isaac.get("acknowledged_robot") or "—"),
    })
    return rows


def _robot_rows(snap, isaac: Dict[str, Any]) -> Dict[str, Any]:
    """Everything Diagnostics reports about the SELECTED robot.

    EXPECTED VERSUS DISCOVERED, side by side, because that is the pairing that
    makes a wrong robot configuration visible. Isaac does not fail loudly when a
    profile does not match an asset: a joint name that is not in the
    articulation resolves to an empty index list and the command silently does
    nothing. A row that showed only the CONFIGURED joints would render
    identically whether or not the robot has them.

    Everything here is what the simulator reported. The orchestrator forwards it
    verbatim — it does not know what an articulation is and must not learn.
    """
    robot = dict(isaac.get("robot_status") or {})
    selected = (isaac.get("robot_id") or snap.active_robot_id or "") or "—"
    expected = robot.get("expected_arm_joints") or []
    discovered = robot.get("discovered_arm_joints") or []

    def _yn(value, unknown="unknown"):
        return unknown if value is None else ("yes" if value else "NO")

    rows: Dict[str, Any] = {
        "configured_robots": ", ".join(robot.get("configured_robots") or []) or "—",
        "selected_robot": selected,
        # SELECTED and ACTIVE are different claims. "Selected" is what this run
        # asked for; "active" is what actually stood up and validated. They
        # differ exactly when something is wrong, which is when it matters.
        "active_robot": robot.get("active_robot") or "—",
        "robot_profile_revision": robot.get("robot_profile_revision") or "—",
        "robot_registry_revision": robot.get("registry_revision") or "—",
        "robot_asset_resolved": robot.get("asset_resolved") or "—",
        "robot_articulation_valid": _yn(robot.get("articulation_valid")),
        "robot_expected_arm_joints": ", ".join(expected) or "—",
        "robot_discovered_arm_joints": ", ".join(discovered) or "—",
        "robot_end_effector_resolved": robot.get("end_effector_resolved") or "—",
        "robot_gripper_ready": _yn(robot.get("gripper_ready")),
        "robot_home_verified": _yn(robot.get("home_verified")),
        "robot_kinematics": robot.get("kinematics") or "—",
        "robot_kinematics_ready": _yn(robot.get("kinematics_ready")),
        # Stated, never inferred. A differential IK controller is not a motion
        # planner, and a dashboard that leaves this to inference invites the
        # opposite conclusion.
        "robot_motion_planning": _yn(robot.get("motion_planning"), "no"),
        "robot_scene_ready": _yn(isaac.get("scene_ready")),
        "acknowledged_robot": isaac.get("acknowledged_robot") or "—",
        "last_robot_error": (isaac.get("robot_model_error")
                             or robot.get("last_robot_error") or "none"),
    }
    if expected and discovered and expected != discovered:
        rows["robot_joint_mismatch"] = (
            f"configured {expected} but the articulation reports {discovered}")
    return rows


__all__ = ["build", "COMPONENTS", "INTERFACES"]
