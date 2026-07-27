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
    }

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


__all__ = ["build", "COMPONENTS", "INTERFACES"]
