"""EDF Topic #2 anomaly integration — deterministic reaction and honesty labels.

This is an ARCHITECTURE DEMONSTRATION and the tests enforce that framing: every
event is simulated, the official Topic #2 detection KPI is never marked achieved,
and the workflow reactions (record / pause / hold) are deterministic and local.
"""

from __future__ import annotations

import json

import pytest

from wisepack_core.anomaly import (
    AnomalyClass, AnomalyEvent, Reaction, Severity, SIMULATED_LABEL,
    default_severity, reaction_for,
)
from wisepack_core.domain import ApprovalState, Source
from wisepack_core.packing import OptimizerConfig
from wisepack_core.workflow import (
    AnomalyHold, WorkflowConfig, WorkflowEngine,
)


def planned(**kw) -> WorkflowEngine:
    kw.setdefault("preset", "mixed_pipes_dense")
    kw.setdefault("seed", 42)
    kw.setdefault("optimizer", OptimizerConfig(seed=42, restarts=3))
    engine = WorkflowEngine(WorkflowConfig(**kw))
    engine.generate_or_load_scenario()
    engine.scan_and_detect()
    engine.generate_plans()
    engine.digital_twin_validate()
    engine.request_approval()
    return engine


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #

def test_severity_reaction_mapping():
    assert reaction_for(Severity.INFO) is Reaction.CONTINUE
    assert reaction_for(Severity.WARNING) is Reaction.PAUSE
    assert reaction_for(Severity.CRITICAL) is Reaction.HOLD


def test_every_event_is_labelled_simulated():
    for cls in AnomalyClass:
        e = AnomalyEvent.simulate(cls.value)
        d = e.to_dict()
        assert d["label"] == SIMULATED_LABEL
        assert d["source"] == Source.SIMULATED.value
        assert d["source_module"] == "simulated_anomaly_detector"


def test_operation_ok_is_info_and_ok_status():
    e = AnomalyEvent.simulate("operation_ok")
    assert e.severity is Severity.INFO
    assert e.status == "OK"
    assert e.is_ok


def test_all_required_classes_exist():
    for name in ("operation_ok", "shear_position_too_high",
                 "shear_position_too_low", "shear_closed_before_contact",
                 "camera_view_lost", "tool_pose_deviation"):
        assert AnomalyClass(name)


def test_event_round_trips_through_json():
    e = AnomalyEvent.simulate("shear_position_too_high", sequence=3)
    back = AnomalyEvent.from_dict(json.loads(json.dumps(e.to_dict())))
    assert back.anomaly_class is e.anomaly_class
    assert back.severity is e.severity
    assert back.source is Source.SIMULATED


# --------------------------------------------------------------------------- #
# The workflow reaction — deterministic and local
# --------------------------------------------------------------------------- #

def test_info_anomaly_records_and_continues():
    engine = planned()
    engine.approve()
    engine.step_execution()
    before = sum(1 for p in engine.selected.placements if p.executed)
    engine.apply_anomaly(AnomalyEvent.simulate("operation_ok"))
    assert not engine.anomaly_hold
    assert not engine.anomaly_ack_required
    engine.step_execution()          # execution continues
    assert sum(1 for p in engine.selected.placements if p.executed) >= before


def test_warning_anomaly_pauses_and_requires_acknowledgement():
    engine = planned()
    engine.approve()
    engine.step_execution()
    engine.apply_anomaly(AnomalyEvent.simulate("camera_view_lost"))
    assert engine.anomaly_hold
    assert engine.anomaly_ack_required
    with pytest.raises(AnomalyHold):
        engine.step_execution()
    # Acknowledge clears the hold; the plan is still approved, so it resumes.
    engine.acknowledge_anomaly("op")
    assert not engine.anomaly_hold
    assert engine.selected.approval_state is ApprovalState.APPROVED
    engine.step_execution()          # no exception


def test_critical_anomaly_revokes_authorisation():
    engine = planned()
    engine.approve()
    engine.step_execution()
    executed = {p.item_id for p in engine.selected.placements if p.executed}
    engine.apply_anomaly(AnomalyEvent.simulate("shear_position_too_high"))
    assert engine.anomaly_hold
    # Authorisation is revoked and completed placements are preserved.
    assert engine.selected.approval_state is not ApprovalState.APPROVED
    still = {p.item_id for p in engine.selected.placements if p.executed}
    assert executed <= still
    # Acknowledgement alone is NOT authorisation: a re-approval is required.
    engine.acknowledge_anomaly("op")
    from wisepack_core.workflow import ApprovalRequired
    with pytest.raises(ApprovalRequired):
        engine.step_execution()


def test_acknowledge_without_pending_anomaly_is_refused():
    engine = planned()
    from wisepack_core.workflow import WorkflowError
    with pytest.raises(WorkflowError):
        engine.acknowledge_anomaly("op")


def test_anomaly_events_appear_in_the_audit_trail_and_are_monotonic():
    engine = planned()
    engine.approve()
    engine.apply_anomaly(AnomalyEvent.simulate("camera_view_lost"))
    engine.acknowledge_anomaly("op")
    engine.apply_anomaly(AnomalyEvent.simulate("shear_position_too_low"))
    actions = [e.action for e in engine.log.events()]
    assert any(a.startswith("anomaly:") for a in actions)
    assert "anomaly_acknowledged" in actions
    ok, note = engine.log.sequence_is_monotonic()
    assert ok, note


def test_anomaly_snapshot_aggregates_history():
    engine = planned()
    engine.approve()
    engine.apply_anomaly(AnomalyEvent.simulate("camera_view_lost"))
    engine.acknowledge_anomaly("op")
    engine.apply_anomaly(AnomalyEvent.simulate("shear_position_too_high"))
    snap = engine.anomaly_snapshot()
    assert snap["count"] == 2
    assert snap["pauses"] == 1
    assert snap["holds"] == 1
    assert snap["nok_count"] == 2
    assert "not a validated anomaly detector" in snap["label"]


def test_operator_source_is_labelled_for_acknowledgement():
    engine = planned()
    engine.approve()
    engine.apply_anomaly(AnomalyEvent.simulate("camera_view_lost"))
    engine.acknowledge_anomaly("inspector-1")
    ack = next(e for e in engine.log.events() if e.action == "anomaly_acknowledged")
    assert ack.source is Source.OPERATOR


# --------------------------------------------------------------------------- #
# Honesty: the official Topic #2 KPI is never claimed
# --------------------------------------------------------------------------- #

def test_anomaly_events_are_never_measured_source():
    """A simulated anomaly must never masquerade as a measured detection."""
    engine = planned()
    engine.approve()
    engine.apply_anomaly(AnomalyEvent.simulate("shear_position_too_high"))
    anomaly_events = [e for e in engine.log.events()
                      if e.action.startswith("anomaly:")]
    assert anomaly_events
    for e in anomaly_events:
        assert e.source is Source.SIMULATED


def test_kpis_do_not_claim_a_topic2_detection_kpi():
    """The KPI report must not invent an anomaly-detection KPI as achieved."""
    engine = planned()
    engine.approve()
    engine.apply_anomaly(AnomalyEvent.simulate("camera_view_lost"))
    report = engine.kpis()
    # No metric key implies a validated Topic #2 detector.
    for key in report.metrics:
        assert "detection_accuracy" not in key
        assert "topic2" not in key.lower()


# --------------------------------------------------------------------------- #
# ROS / FIWARE contract
# --------------------------------------------------------------------------- #

def _topics():
    import importlib.util, os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo, "wisepack_ws", "src", "wisepack_bringup",
                        "wisepack_bringup", "topics.py")
    spec = importlib.util.spec_from_file_location("wp_topics_anom", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_anomaly_topics_and_commands_in_contract():
    T = _topics()
    assert T.ANOMALY_EVENT in T.all_topics()
    assert T.ANOMALY_STATE in T.all_topics()
    assert T.ANOMALY_EXTERNAL in T.all_topics()
    assert "inject_anomaly" in T.OPERATOR_COMMANDS
    assert "acknowledge_anomaly" in T.OPERATOR_COMMANDS


def test_anomaly_event_is_mapped_to_fiware():
    yaml = pytest.importorskip("yaml")
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo, "wisepack_ws", "src", "wisepack_fiware",
                           "config", "bridge_config.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    mapped = {m["ros_topic"] for m in cfg["ros_to_fiware"]}
    assert "/wisepack/anomaly/event" in mapped
    # The raw ingest seam is NOT mapped (it is input, not the recorded stream).
    assert "/wisepack/anomaly/external" not in mapped
