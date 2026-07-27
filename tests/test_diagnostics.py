"""Diagnostics must be transparent AND secret-free.

The page exists for interview transparency, so the tests enforce the two
properties that make that safe: it exposes no secrets / no Docker socket, and it
classifies simulated and future interfaces correctly rather than as failures.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(REPO, "web")
sys.path.insert(0, WEB)
for _pkg in ("wisepack_core", "wisepack_fiware", "wisepack_bringup"):
    sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", _pkg))

import diagnostics                                              # noqa: E402
from snapshot import SimSnapshotProvider                       # noqa: E402
from wisepack_core.workflow import WorkflowConfig, WorkflowEngine  # noqa: E402
from wisepack_core.packing import OptimizerConfig              # noqa: E402


class _State:
    """Minimal STATE stand-in for SimSnapshotProvider."""
    def __init__(self, engine):
        import threading
        self.lock = threading.RLock()
        self.engine = engine
        self.events = [e.to_dict() for e in engine.log.events()]
        self.auto_step = False
        self.notice = ""


def _report():
    engine = WorkflowEngine(WorkflowConfig(
        preset="mixed_pipes_dense", seed=42,
        optimizer=OptimizerConfig(seed=42, restarts=3)))
    engine.generate_or_load_scenario()
    engine.scan_and_detect()
    engine.generate_plans()
    engine.digital_twin_validate()
    engine.request_approval()
    snap = SimSnapshotProvider(_State(engine)).snapshot()
    return diagnostics.build(snap, "sim", None, None)


# --------------------------------------------------------------------------- #
# Security
# --------------------------------------------------------------------------- #

def test_no_secret_or_environment_leak():
    report = _report()
    blob = json.dumps(report).lower()
    # None of the actual environment VALUES may appear.
    for key in ("PATH", "HOME", "USER", "PWD", "VIRTUAL_ENV", "LS_COLORS"):
        val = os.environ.get(key, "")
        if val and len(val) > 8:
            assert val.lower() not in blob, f"environment value for {key} leaked"


def test_no_docker_socket_dependency():
    report = _report()
    assert "/var/run/docker.sock" not in json.dumps(report)
    # The diagnostics module must not import docker or subprocess for its data.
    with open(os.path.join(WEB, "diagnostics.py"), encoding="utf-8") as fh:
        src = fh.read()
    assert "docker.sock" not in src
    assert "subprocess" not in src, "diagnostics must not shell out"


def test_only_allowlisted_containers_appear():
    report = _report()
    roles = {c["key"] for c in report["components"]}
    # Container components are limited to the two external services.
    assert "orion" in roles and "mongo" in roles
    # collect_runtime_status.sh must restrict to the three WISEPACK names.
    with open(os.path.join(REPO, "scripts", "collect_runtime_status.sh"),
              encoding="utf-8") as fh:
        sh = fh.read()
    assert "wisepack-dashboard" in sh and "wisepack-orion-ld" in sh
    assert "wisepack-mongo-dds" in sh
    # It must not inspect arbitrary containers (`docker ps -a` over everything).
    assert "docker ps -a" not in sh


# --------------------------------------------------------------------------- #
# Correct classification
# --------------------------------------------------------------------------- #

def test_simulated_and_future_interfaces_are_labelled_not_failed():
    report = _report()
    by_iface = {i["interface"]: i["state"] for i in report["interfaces"]}
    assert by_iface["RGB-D camera frames"] == "future interface"
    assert by_iface["Object detections"] == "simulated source"
    assert by_iface["Packing optimizer"] == "measured software"
    assert by_iface["FIWARE event mapping"] == "live"
    # No interface is reported with an ERROR/FAILED state.
    for i in report["interfaces"]:
        assert "error" not in i["state"].lower()
        assert "fail" not in i["state"].lower()


def test_sim_mode_topics_are_not_reported_as_errors():
    """In sim mode there is no ROS graph; that is not a failure."""
    report = _report()
    for t in report["topics"]:
        assert t["status"] != "ERROR"
    assert any(t["status"] == "NOT EXPECTED IN THIS MODE" for t in report["topics"])


def test_fiware_mappings_match_the_generated_configuration():
    """The mapping table must be derived from bridge_config, not invented."""
    yaml = pytest.importorskip("yaml")
    report = _report()
    with open(os.path.join(REPO, "wisepack_ws", "src", "wisepack_fiware",
                           "config", "bridge_config.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    expected = len(cfg.get("ros_to_fiware", []) or []) + \
        len(cfg.get("fiware_to_ros", []) or [])
    assert len(report["fiware_mappings"]) == expected


def test_report_has_all_required_sections():
    report = _report()
    for section in ("overview", "components", "topics", "interfaces",
                    "fiware_mappings", "timing", "security_note"):
        assert section in report, f"diagnostics missing section {section}"
    # Overview must include the transparency-relevant fields.
    for field in ("mode", "scenario_revision", "stage", "action_sequence",
                  "sequence_gap_free", "panel_sources"):
        assert field in report["overview"]


# --------------------------------------------------------------------------- #
# Support bundle
# --------------------------------------------------------------------------- #

def test_support_bundle_contains_only_allowlisted_files(tmp_path):
    env = dict(os.environ, WISEPACK_RESULTS_DIR=str(tmp_path))
    r = subprocess.run([os.path.join(REPO, "collect_wisepack_diagnostics.sh")],
                       capture_output=True, text=True, cwd=REPO, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    bundles = list(tmp_path.glob("diagnostics-*"))
    assert bundles, "no bundle written"
    files = {p.name for p in bundles[0].iterdir()}
    # An allowlist: every file must match an expected pattern.
    import re
    allowed = [r"runtime-status\.json", r"topic-qos-summary\.json",
               r"fiware-summary\.json", r"context_broker_config\.json",
               r"manifest\.md", r"wisepack-run-.*\.json", r"wisepack-kpis-.*\.json",
               r"wisepack-validation-.*\.md", r"wisepack-fiware-validation-.*\.md",
               r"wisepack-dds-fiware-latency-.*\.json"]
    for f in files:
        assert any(re.fullmatch(p, f) for p in allowed), \
            f"unexpected file in support bundle: {f}"
    # And no secret-looking content.
    for p in bundles[0].iterdir():
        text = p.read_text(errors="ignore").lower()
        for bad in ("begin private key", "aws_secret", "password="):
            assert bad not in text, f"{p.name} contains a secret-looking string"
