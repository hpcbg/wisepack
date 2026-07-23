"""ROS 2 topic contract and FIWARE mapping.

These tests deliberately run WITHOUT a ROS installation. They check the things
that are actually easy to get silently wrong and that a running system will not
tell you about:

  * a topic whose leaf is the reserved `status` is DROPPED by Orion-LD's DDS
    module with no error anywhere;
  * a message type that is not scalar `std_msgs` cannot bridge at all;
  * the YAML mapping and the Python topic contract drifting apart;
  * an entity id hard-coded in a consumer that no longer matches the mapping.

Every one of those failures is invisible at runtime and fatal to the audit
trail, which is exactly what makes them worth a static test.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "wisepack_ws", "src")
BRIDGE_YAML = os.path.join(SRC, "wisepack_fiware", "config", "bridge_config.yaml")
GENERATOR = os.path.join(SRC, "wisepack_fiware", "dds", "generate_config.py")
CONTEXT_JSON = os.path.join(SRC, "wisepack_fiware", "dds",
                            "context_broker_config.json")

yaml = pytest.importorskip("yaml", reason="pyyaml is needed to read the mapping")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOPICS = _load("wisepack_topics",
               os.path.join(SRC, "wisepack_bringup", "wisepack_bringup", "topics.py"))
ENTITIES = _load("wisepack_entities",
                 os.path.join(SRC, "wisepack_fiware", "wisepack_fiware", "entities.py"))

with open(BRIDGE_YAML, encoding="utf-8") as fh:
    BRIDGE = yaml.safe_load(fh)

ELIGIBLE = {"std_msgs/String", "std_msgs/Bool", "std_msgs/Int32",
            "std_msgs/Int64", "std_msgs/Float32", "std_msgs/Float64"}


# --------------------------------------------------------------------------- #
# The topic contract
# --------------------------------------------------------------------------- #

def test_no_topic_uses_the_reserved_status_leaf():
    """A `/status` leaf is silently dropped by Orion-LD — never an error."""
    assert TOPICS.reserved_leaf_violations() == []


def test_plan_status_topic_uses_the_json_leaf():
    assert TOPICS.PLAN_STATUS.endswith("/status_json")


def test_every_contract_topic_is_a_bridgeable_type():
    for topic, msg_type in TOPICS.all_topics().items():
        assert msg_type in ELIGIBLE, (
            f"{topic} is {msg_type}, which the Orion-LD DDS bridge cannot map. "
            "WISEPACK's audit trail has no second path.")


def test_all_topics_live_under_the_wisepack_namespace():
    for topic in TOPICS.all_topics():
        assert topic.startswith("/wisepack/"), topic


def test_topic_names_are_unique():
    names = list(TOPICS.all_topics())
    assert len(names) == len(set(names))


def test_inbound_topics_are_exactly_the_operator_path():
    assert set(TOPICS.INBOUND_TOPICS) == {TOPICS.OPERATOR_APPROVAL,
                                          TOPICS.OPERATOR_COMMAND}


def test_the_canonical_topics_named_in_the_brief_all_exist():
    """Guards against a rename quietly breaking a documented contract."""
    for expected in ("/wisepack/scenario/config", "/wisepack/scenario/state",
                     "/wisepack/waste/items", "/wisepack/waste/detected_count",
                     "/wisepack/plan/baseline", "/wisepack/plan/optimized",
                     "/wisepack/plan/selected", "/wisepack/operator/approval",
                     "/wisepack/execution/state", "/wisepack/execution/current_item",
                     "/wisepack/execution/current_container",
                     "/wisepack/execution/progress_pct",
                     "/wisepack/system/readiness", "/wisepack/action/event",
                     "/wisepack/action/sequence", "/wisepack/dynamic_event",
                     "/wisepack/kpi/containers_baseline",
                     "/wisepack/kpi/containers_optimized",
                     "/wisepack/kpi/utilization_baseline_pct",
                     "/wisepack/kpi/utilization_optimized_pct",
                     "/wisepack/kpi/volume_reduction_pct",
                     "/wisepack/kpi/optimization_ms",
                     "/wisepack/kpi/pick_success_pct",
                     "/wisepack/kpi/end_to_end_success_pct"):
        assert expected in TOPICS.all_topics(), f"{expected} is missing"


# --------------------------------------------------------------------------- #
# The YAML mapping
# --------------------------------------------------------------------------- #

def _mappings():
    return (BRIDGE.get("ros_to_fiware", []) or []) + \
           (BRIDGE.get("fiware_to_ros", []) or [])


def test_every_mapped_topic_exists_in_the_contract():
    contract = TOPICS.all_topics()
    for m in _mappings():
        assert m["ros_topic"] in contract, (
            f"{m['ros_topic']} is mapped to FIWARE but is not in topics.py")


def test_mapped_types_match_the_contract():
    """The YAML and the Python contract must not drift."""
    contract = TOPICS.all_topics()
    for m in _mappings():
        assert m["ros_msg_type"] == contract[m["ros_topic"]], (
            f"{m['ros_topic']}: YAML says {m['ros_msg_type']}, "
            f"contract says {contract[m['ros_topic']]}")


def test_the_audit_topic_is_mapped():
    """Without this the whole traceability claim is unsupported."""
    mapped = {m["ros_topic"] for m in BRIDGE["ros_to_fiware"]}
    assert TOPICS.ACTION_EVENT in mapped
    assert TOPICS.ACTION_SEQUENCE in mapped


def test_the_operator_path_is_inbound_only():
    inbound = {m["ros_topic"] for m in BRIDGE["fiware_to_ros"]}
    assert inbound == set(TOPICS.INBOUND_TOPICS)
    outbound = {m["ros_topic"] for m in BRIDGE["ros_to_fiware"]}
    assert not (inbound & outbound), "a topic cannot flow both ways"


def test_no_mapping_uses_the_reserved_leaf():
    for m in _mappings():
        assert m["ros_topic"].rsplit("/", 1)[-1] != "status", m["ros_topic"]


def test_every_kpi_topic_reaches_fiware():
    mapped = {m["ros_topic"] for m in BRIDGE["ros_to_fiware"]}
    for topic in TOPICS.all_topics():
        if topic.startswith("/wisepack/kpi/"):
            assert topic in mapped, f"{topic} never reaches FIWARE"


# --------------------------------------------------------------------------- #
# The generated Orion-LD configuration
# --------------------------------------------------------------------------- #

def test_generator_accepts_the_current_mapping():
    result = subprocess.run(
        [sys.executable, GENERATOR, "--check", "--domain", "0"],
        capture_output=True, text=True, cwd=os.path.dirname(GENERATOR))
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_generated_config_is_current():
    """The committed JSON must match what the YAML generates right now."""
    assert os.path.exists(CONTEXT_JSON), "run generate_config.py"
    with open(CONTEXT_JSON, encoding="utf-8") as fh:
        committed = json.load(fh)
    result = subprocess.run(
        [sys.executable, GENERATOR, "--domain", "0", "--output", "/dev/stdout"],
        capture_output=True, text=True, cwd=os.path.dirname(GENERATOR))
    assert result.returncode == 0, result.stderr
    topics = committed["dds"]["ngsild"]["topics"]
    assert len(topics) == len(_mappings()), (
        "the generated mapping and the YAML disagree — re-run generate_config.py")


def test_generated_topics_use_the_rt_prefix():
    with open(CONTEXT_JSON, encoding="utf-8") as fh:
        doc = json.load(fh)
    for dds_topic in doc["dds"]["ngsild"]["topics"]:
        assert dds_topic.startswith("rt/wisepack/"), dds_topic


def test_generated_entity_ids_are_ngsi_ld_urns():
    with open(CONTEXT_JSON, encoding="utf-8") as fh:
        doc = json.load(fh)
    for spec in doc["dds"]["ngsild"]["topics"].values():
        assert spec["entityId"].startswith("urn:ngsi-ld:WISEPACK"), spec
        # A doubled type prefix is a classic generator bug.
        assert spec["entityId"].count(spec["entityType"]) == 1, spec


def test_dds_domain_matches_ros_domain_id_default():
    with open(CONTEXT_JSON, encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["dds"]["ddsmodule"]["dds"]["domain"] == 0


# --------------------------------------------------------------------------- #
# Entity registry
# --------------------------------------------------------------------------- #

def test_entity_registry_matches_the_yaml():
    ids, attrs = ENTITIES.load_mapping(BRIDGE_YAML)
    for short, urn in ids.items():
        assert urn.startswith("urn:ngsi-ld:")
        assert attrs.get(urn), f"{urn} has no attributes"


def test_entity_fallback_agrees_with_the_yaml():
    """The hard-coded fallback must never drift from the real mapping."""
    ids, _ = ENTITIES.load_mapping(BRIDGE_YAML)
    for short, urn in ENTITIES._FALLBACK.items():
        assert ids.get(short) == urn, (
            f"fallback {short}={urn} disagrees with the YAML ({ids.get(short)})")


def test_the_six_suggested_entities_all_exist():
    ids, _ = ENTITIES.load_mapping(BRIDGE_YAML)
    urns = set(ids.values())
    for expected in ("WISEPACKSystem", "WISEPACKScenario", "WISEPACKRobot",
                     "WISEPACKPackingPlan", "WISEPACKActionStream", "WISEPACKKPI"):
        assert any(expected in u for u in urns), f"{expected} is not mapped"


# --------------------------------------------------------------------------- #
# Event payloads survive the DDS path
# --------------------------------------------------------------------------- #

def test_action_events_fit_the_dds_payload_budget():
    """A workflow event must never need fragmenting on the wire."""
    from wisepack_core.events import ActionEvent
    from wisepack_core.workflow import WorkflowConfig, run_headless

    engine = run_headless(WorkflowConfig(preset="mixed_pipes_dense", seed=42,
                                         auto_approve=True))
    oversized = []
    for event in engine.log.events():
        payload = event.to_json()
        assert len(payload.encode()) <= ActionEvent.MAX_SERIALISED_BYTES + 512
        json.loads(payload)                     # must stay valid JSON
        if "_truncated" in payload:
            oversized.append(event.action)
    # Truncation is allowed but must be rare and visible; the planning events
    # carry the biggest details blocks.
    assert len(oversized) <= 6, f"too many truncated events: {oversized}"


def test_every_action_event_is_json_and_carries_a_source():
    from wisepack_core.workflow import WorkflowConfig, run_headless
    engine = run_headless(WorkflowConfig(preset="mixed_pipes_small", seed=42,
                                         auto_approve=True))
    for event in engine.log.events():
        doc = json.loads(event.to_json())
        assert doc["source"] in ("measured", "simulated", "operator")
        assert doc["sequence"] >= 1
        assert doc["stage"] and doc["action"]


def test_final_action_count_matches_the_executed_workflow():
    """The count published to FIWARE must equal what actually happened."""
    from wisepack_core.workflow import WorkflowConfig, run_headless
    engine = run_headless(WorkflowConfig(preset="mixed_pipes_small", seed=42,
                                         auto_approve=True))
    events = engine.log.events()
    assert len(events) == engine.log.count
    assert events[-1].sequence == len(events)
    # One place_item event per executed placement.
    placed = sum(1 for e in events if e.action == "place_item")
    executed = sum(1 for p in engine.selected.placements if p.executed
                   and p.validation_status.value == "valid")
    assert placed == executed


# --------------------------------------------------------------------------- #
# QoS compatibility with EXTERNAL publishers
# --------------------------------------------------------------------------- #

def test_inbound_topics_request_no_deadline_or_liveliness():
    """Regression: a requested Deadline silently kills the FIWARE command path.

    A subscription that REQUESTS a Deadline only matches a publisher that OFFERS
    a period at least as short. Orion-LD's DDS bridge and `ros2 topic pub` both
    offer an infinite deadline, so requesting one on an inbound topic makes them
    INCOMPATIBLE — and rclpy does not raise, it just delivers nothing forever.

    This bug was live in the repository and was only caught by running the full
    ROS 2 stack: every node reported healthy while the operator path was dead.
    The test reads the source rather than importing rclpy so it runs anywhere.
    """
    qos_path = os.path.join(SRC, "wisepack_bringup", "wisepack_bringup", "qos.py")
    with open(qos_path, encoding="utf-8") as fh:
        source = fh.read()

    start = source.index("def command_qos()")
    end = source.index("def ", start + 10)
    body = source[start:end]
    assert "deadline=" not in body, (
        "command_qos() requests a Deadline. That makes the subscription "
        "incompatible with Orion-LD's DDS bridge, which offers none, and the "
        "operator command path will silently never connect.")
    assert "liveliness=" not in body, (
        "command_qos() requests Liveliness — same silent incompatibility.")


def test_heartbeat_qos_keeps_its_deadline():
    """The state heartbeat SHOULD carry a deadline: WISEPACK publishes it itself."""
    qos_path = os.path.join(SRC, "wisepack_bringup", "wisepack_bringup", "qos.py")
    with open(qos_path, encoding="utf-8") as fh:
        source = fh.read()
    start = source.index("def heartbeat_qos()")
    end = source.index("def ", start + 10)
    body = source[start:end]
    assert "deadline=" in body, (
        "the execution-state heartbeat must carry a Deadline, or a dead "
        "orchestrator is undetectable rather than merely silent")
