"""QoS contract tests.

Two layers, because the interesting failures live in different places:

  * STATIC — assertions about `qos_for()` and the profiles it returns. These run
    anywhere, with no ROS installed.
  * LIVE — assertions about the ACTUAL running DDS graph, parsed from
    `ros2 topic info -v`. These are opt-in and need the stack up:

        WISEPACK_QOS_LIVE=1 pytest tests/test_qos_contract.py

Why the live layer matters. Every QoS failure in this repository was invisible
to a static check:

  1. `command_qos()` requested a 2 s Deadline. Orion-LD's DDS bridge offers an
     infinite one, so the subscription never matched and the entire FIWARE ->
     ROS operator path was silently dead while every node reported healthy.
  2. `/wisepack/execution/state` requested a 4 s liveliness lease, which is
     incompatible with the same generic publishers. Measured symptom:
     "Last incompatible policy: LIVELINESS", and a blank live dashboard.
  3. KPI topics were BEST_EFFORT + VOLATILE, so a dashboard attaching after the
     orchestrator had planned received nothing and rendered every tile as
     "not measured".

rclpy raises for none of these. It just delivers nothing, forever.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "wisepack_ws", "src")
QOS_PY = os.path.join(SRC, "wisepack_bringup", "wisepack_bringup", "qos.py")
TOPICS_PY = os.path.join(SRC, "wisepack_bringup", "wisepack_bringup", "topics.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOPICS = _load("wp_topics", TOPICS_PY)

with open(QOS_PY, encoding="utf-8") as fh:
    QOS_SOURCE = fh.read()


def _profile_body(name: str) -> str:
    start = QOS_SOURCE.index(f"def {name}(")
    nxt = QOS_SOURCE.find("\ndef ", start + 1)
    return QOS_SOURCE[start:nxt if nxt != -1 else len(QOS_SOURCE)]


# --------------------------------------------------------------------------- #
# Static
# --------------------------------------------------------------------------- #

def test_only_the_heartbeat_profile_declares_liveliness():
    """Any OTHER profile requesting liveliness re-breaks the live dashboard."""
    for name in ("state_qos", "command_qos", "telemetry_qos", "event_qos",
                 "watchdog_subscribe_qos"):
        body = _profile_body(name)
        assert "liveliness=" not in body, (
            f"{name}() requests Liveliness. Generic publishers — Orion-LD's DDS "
            "bridge included — offer an infinite lease, so this subscription "
            "will silently receive nothing.")
        assert "deadline=" not in body, f"{name}() requests a Deadline"


def test_heartbeat_offering_profile_keeps_its_watchdog_policy():
    body = _profile_body("heartbeat_qos")
    assert "deadline=" in body and "liveliness=" in body, (
        "the orchestrator must still OFFER the watchdog policy on the heartbeat")


def test_execution_state_is_plain_latched_state():
    """Regression: the reported LIVELINESS incompatibility was on this topic."""
    assert "if topic == T.EXECUTION_STATE" not in QOS_SOURCE, (
        "execution/state must fall through to state_qos, not get a special "
        "watchdog profile — that is what broke the live dashboard")


def test_a_dedicated_heartbeat_topic_exists():
    assert hasattr(TOPICS, "SYSTEM_HEARTBEAT")
    assert TOPICS.SYSTEM_HEARTBEAT in TOPICS.all_topics()


def test_heartbeat_is_not_bridged_to_fiware():
    """Keeping it off the bridge keeps generic publishers off the topic."""
    yaml = pytest.importorskip("yaml")
    with open(os.path.join(SRC, "wisepack_fiware", "config",
                           "bridge_config.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    mapped = {m["ros_topic"] for block in ("ros_to_fiware", "fiware_to_ros")
              for m in (cfg.get(block) or [])}
    assert TOPICS.SYSTEM_HEARTBEAT not in mapped


def test_kpi_topics_are_latched_not_best_effort():
    """A dashboard attaching mid-run must see the current KPI values."""
    assert 'topic.startswith("/wisepack/kpi/")' in QOS_SOURCE
    idx = QOS_SOURCE.index('topic.startswith("/wisepack/kpi/")')
    following = QOS_SOURCE[idx:idx + 400]
    assert "state_qos()" in following, (
        "KPI topics must be latched state; BEST_EFFORT+VOLATILE gave the live "
        "dashboard a full set of 'not measured' tiles")


def test_action_events_are_transient_local():
    """Late attachment is the NORMAL case for the dashboard."""
    body = _profile_body("event_qos")
    assert "TRANSIENT_LOCAL" in body, (
        "the audit stream must be replayable to a late joiner, or the timeline "
        "renders empty against a healthy stack")
    assert "RELIABLE" in body


def test_progress_stays_best_effort():
    """The one genuinely high-rate topic keeps the cheap profile."""
    assert "if topic == T.EXECUTION_PROGRESS_PCT" in QOS_SOURCE


# --------------------------------------------------------------------------- #
# Live graph
# --------------------------------------------------------------------------- #

LIVE = os.environ.get("WISEPACK_QOS_LIVE") == "1"
pytestmark_live = pytest.mark.skipif(
    not LIVE, reason="set WISEPACK_QOS_LIVE=1 with the ROS stack running")

_ENDPOINT = re.compile(
    r"Node name: (?P<node>\S+).*?"
    r"Endpoint type: (?P<kind>PUBLISHER|SUBSCRIPTION).*?"
    r"Reliability: (?P<rel>\S+).*?"
    r"Durability: (?P<dur>\S+).*?"
    r"Deadline: (?P<deadline>[^\n]+).*?"
    r"Liveliness: (?P<liveliness>\S+).*?"
    r"Liveliness lease duration: (?P<lease>[^\n]+)",
    re.S)


def topic_endpoints(topic: str):
    """Parse `ros2 topic info -v` into endpoint dicts."""
    out = subprocess.run(["ros2", "topic", "info", "-v", topic],
                         capture_output=True, text=True, timeout=30).stdout
    return [m.groupdict() for m in _ENDPOINT.finditer(out)]


@pytestmark_live
@pytest.mark.parametrize("topic", [
    "/wisepack/execution/state", "/wisepack/plan/selected",
    "/wisepack/action/event", "/wisepack/kpi/containers_optimized",
    "/wisepack/operator/approval",
])
def test_live_no_subscription_requests_liveliness_or_deadline(topic):
    """The real graph, not the Python objects.

    A subscription requesting either policy cannot match a generic publisher,
    and that is exactly how the live dashboard went blank.
    """
    for ep in topic_endpoints(topic):
        if ep["kind"] != "SUBSCRIPTION":
            continue
        assert ep["lease"].strip() == "Infinite", (
            f"{topic}: {ep['node']} requests a {ep['lease'].strip()} liveliness "
            "lease — incompatible with any publisher offering Infinite")
        assert ep["deadline"].strip() in ("Infinite", "0 nanoseconds"), (
            f"{topic}: {ep['node']} requests deadline {ep['deadline'].strip()}")


@pytestmark_live
@pytest.mark.parametrize("topic", [
    "/wisepack/execution/state", "/wisepack/plan/baseline",
    "/wisepack/plan/optimized", "/wisepack/plan/selected",
    "/wisepack/kpi/containers_optimized", "/wisepack/action/event",
])
def test_live_state_topics_are_latched(topic):
    """A late joiner must receive the current value."""
    pubs = [e for e in topic_endpoints(topic) if e["kind"] == "PUBLISHER"
            and "BARE_DDS" not in e["node"]]
    assert pubs, f"{topic} has no WISEPACK publisher"
    for ep in pubs:
        assert ep["dur"] == "TRANSIENT_LOCAL", (
            f"{topic}: publisher offers {ep['dur']}; a dashboard attaching after "
            "the orchestrator would receive nothing")


@pytestmark_live
def test_live_heartbeat_publisher_offers_the_watchdog_policy():
    pubs = [e for e in topic_endpoints("/wisepack/system/heartbeat")
            if e["kind"] == "PUBLISHER" and "BARE_DDS" not in e["node"]]
    assert pubs, "no heartbeat publisher"
    assert any(p["lease"].strip() != "Infinite" for p in pubs), (
        "the orchestrator must OFFER a finite liveliness lease on the heartbeat")


@pytestmark_live
def test_live_every_contract_topic_has_a_publisher():
    listed = subprocess.run(["ros2", "topic", "list", "--no-daemon"],
                            capture_output=True, text=True, timeout=60).stdout
    present = set(listed.split())
    missing = [t for t in TOPICS.all_topics() if t not in present]
    assert not missing, f"topics with no publisher: {missing}"
