"""Operator-control responsiveness and the draft/active scenario split.

Two manual-validation defects are pinned here.

1. APPROVE WAS DISABLED FOR TOO LONG IN FIWARE MODE. `FiwareSnapshotProvider`
   overwrites `stage` with the value Orion-LD echoed back — correct for the audit
   badge, wrong for a button — so enablement waited for the DDS -> NGSI-LD
   bridge plus a dashboard poll. Control state now comes from the canonical
   ROS/DDS topics and the audit view is untouched.

2. THE PRESET DROPDOWN WAS UNUSABLE IN EVERY LIVE MODE. Each launcher starts a
   run automatically, the workflow reaches WAIT_FOR_OPERATOR_APPROVAL within
   seconds, and the controls were locked in exactly that state — so the operator
   never got a chance to choose. The controls are now a DRAFT for the next run.

Neither test needs ROS, FIWARE, Isaac, a GPU or a browser.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import threading

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO, "web", "index.html")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _snapshot_module():
    name = "wp_snapshot_controls"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO, "web", "snapshot.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeState:
    def __init__(self, mirror=None):
        self.lock = threading.RLock()
        self.engine = None
        self.events = []
        self.notice = ""
        self.auto_step = False
        self.ros_mirror = mirror
        self.fiware_connected = None
        self.fiware_last_error = ""
        self.settings_touched = False


def _plan(plan_id="plan-1", approval="pending", valid=True):
    return {"plan_id": plan_id, "approval_state": approval, "is_valid": valid,
            "containers": [], "placements": []}


def _mirror(stage="WAIT_FOR_OPERATOR_APPROVAL", plan=None):
    return {"stage": stage, "selected": plan if plan is not None else _plan(),
            "heartbeat_at": 0.0}


# --------------------------------------------------------------------------- #
# 1. Approve enablement comes from canonical ROS state
# --------------------------------------------------------------------------- #

def test_approve_is_enabled_from_ros_state_in_plain_ros_mode():
    snapshot = _snapshot_module()
    snap = snapshot.RosSnapshotProvider(_FakeState(_mirror())).snapshot()
    assert snap.can_approve is True
    assert snap.control_state()["can_approve"] is True


def test_stale_fiware_state_does_not_keep_a_valid_approval_disabled():
    """THE DEFECT. Orion-LD still echoes an older stage; the canonical workflow
    is already at the gate. The button must not wait for the echo."""
    snapshot = _snapshot_module()

    def reader():
        # Orion is a step behind — it still reports the planning stage.
        return {"system": {"stage": "DIGITAL_TWIN_VALIDATE"},
                "kpi": {}, "scenario": {}, "plan": {}, "actions": {}, "robot": {}}

    snap = snapshot.FiwareSnapshotProvider(_FakeState(_mirror()), reader).snapshot()
    # The AUDIT view still shows what FIWARE echoed — that is the point of it.
    assert snap.stage == "DIGITAL_TWIN_VALIDATE"
    assert snap.panel_sources["state"] == "fiware"
    # ...and the CONTROL view is the canonical one, so Approve is live.
    assert snap.control_stage == "WAIT_FOR_OPERATOR_APPROVAL"
    assert snap.can_approve is True


def test_the_fiware_plus_ros_badge_is_unchanged():
    """The combined badge is intentional and must not be collateral damage."""
    snapshot = _snapshot_module()

    def reader():
        return {"system": {"stage": "WAIT_FOR_OPERATOR_APPROVAL"},
                "kpi": {}, "scenario": {}, "plan": {}, "actions": {}, "robot": {}}

    snap = snapshot.FiwareSnapshotProvider(_FakeState(_mirror()), reader).snapshot()
    badge = snap.badge()
    assert badge["source"] == "fiware+ros"
    assert badge["label"] == "FIWARE + ROS"


@pytest.mark.parametrize("stage", [
    "GENERATE_OPTIMIZED_PLAN", "DIGITAL_TWIN_VALIDATE", "PICK_ITEM",
    "PLACE_ITEM", "REPLAN", "COMPLETE", "DEGRADED",
])
def test_approve_is_disabled_away_from_the_gate(stage):
    snapshot = _snapshot_module()
    snap = snapshot.RosSnapshotProvider(_FakeState(_mirror(stage=stage))).snapshot()
    assert snap.can_approve is False, f"{stage} must not enable Approve"


def test_a_rendered_plan_alone_never_enables_approval():
    """After a re-plan the previous geometry is still on screen for a moment."""
    snapshot = _snapshot_module()
    snap = snapshot.RosSnapshotProvider(
        _FakeState(_mirror(stage="REPLAN"))).snapshot()
    assert snap.selected is not None          # a plan IS rendered
    assert snap.can_approve is False


def test_a_missing_plan_never_enables_approval():
    snapshot = _snapshot_module()
    mirror = _mirror(); mirror["selected"] = None
    snap = snapshot.RosSnapshotProvider(_FakeState(mirror)).snapshot()
    assert snap.can_approve is False


@pytest.mark.parametrize("approval", ["approved", "rejected", "superseded"])
def test_an_already_decided_or_superseded_plan_never_enables_approval(approval):
    """`superseded` IS the stale-revision case."""
    snapshot = _snapshot_module()
    snap = snapshot.RosSnapshotProvider(
        _FakeState(_mirror(plan=_plan(approval=approval)))).snapshot()
    assert snap.can_approve is False


def test_an_invalid_plan_never_enables_approval():
    snapshot = _snapshot_module()
    snap = snapshot.RosSnapshotProvider(
        _FakeState(_mirror(plan=_plan(valid=False)))).snapshot()
    assert snap.can_approve is False


def test_reject_and_alternative_use_the_same_revision_guard():
    snapshot = _snapshot_module()
    snap = snapshot.RosSnapshotProvider(_FakeState(_mirror())).snapshot()
    ctl = snap.control_state()
    assert ctl["can_reject"] == ctl["can_approve"] is True
    assert ctl["can_alternative"] is True
    assert ctl["source"] == "ros", "controls must never be enabled from FIWARE"


def test_the_control_block_is_published_to_the_frontend():
    snapshot = _snapshot_module()
    state = snapshot.RosSnapshotProvider(_FakeState(_mirror())).snapshot().to_state()
    assert "control" in state
    assert state["control"]["can_approve"] is True


# --------------------------------------------------------------------------- #
# Frontend: enablement, double submission, progress, timeout
# --------------------------------------------------------------------------- #

def test_the_frontend_enables_approval_from_the_control_block_only():
    html = _read(INDEX)
    assert '"#c-approve": ctl.can_approve === true' in html
    assert '"#c-reject": ctl.can_reject === true' in html
    # The old, latency-bound condition must be gone.
    assert '"#c-approve": awaiting' not in html


def test_double_submission_is_prevented():
    html = _read(INDEX)
    assert "APPROVAL_IN_FLIGHT" in html
    assert "if (APPROVAL_IN_FLIGHT) return;" in html
    # A poll landing mid-submission must not re-enable the button.
    assert "&& !APPROVAL_IN_FLIGHT" in html


def test_a_sending_state_is_shown_after_click():
    html = _read(INDEX)
    assert 'decide("approve", "Sending approval…")' in html
    assert 'decide("reject", "Sending rejection…")' in html


def test_an_unacknowledged_decision_restores_the_button_with_an_error():
    html = _read(INDEX)
    assert "APPROVAL_ACK_TIMEOUT_MS" in html
    assert "no acknowledgement within" in html
    # `finally` guarantees restoration whatever happened.
    assert "APPROVAL_IN_FLIGHT = false;" in html


def test_the_decision_pins_the_plan_revision():
    html = _read(INDEX)
    assert "const planId = before.plan_id || null;" in html
    assert "c.plan_id !== planId" in html, \
        "acknowledgement must not be satisfied by a different plan"


def test_the_approval_still_travels_the_documented_command_path():
    html = _read(INDEX)
    block = html[html.index("async function decide("):html.index("async function command(")]
    assert '"/api/command"' in block
    assert '"command": cmd' in block or "command: cmd" in block


def test_the_server_refuses_a_decision_on_a_superseded_plan():
    app = _read(os.path.join(REPO, "web", "app.py"))
    assert 'command in ("approve", "reject") and args.get("plan_id")' in app
    assert "status_code=409" in app
    assert "The plan changed while you were deciding" in app
    # The revision token is a dashboard concept and must not leak onto the
    # orchestrator's command vocabulary.
    assert 'if k != "plan_id"' in app


# --------------------------------------------------------------------------- #
# 2. Draft vs active scenario
# --------------------------------------------------------------------------- #

def test_the_controls_are_no_longer_locked():
    """Every launcher auto-starts a run, so locking at the gate locked forever."""
    app = _read(os.path.join(REPO, "web", "app.py"))
    assert '"settings_locked": False' in app
    html = _read(INDEX)
    assert "node.disabled = false;" in html
    assert "node.disabled = locked" not in html


def test_the_draft_is_not_overwritten_once_the_operator_touches_it():
    app = _read(os.path.join(REPO, "web", "app.py"))
    assert "if not STATE.settings_touched:" in app, \
        "the active run must stop seeding the draft once the operator chooses"


def test_the_draft_endpoint_never_mutates_the_running_scenario():
    app = _read(os.path.join(REPO, "web", "app.py"))
    block = app[app.index('@app.post("/api/draft")'):app.index('@app.get("/api/visualization")')]
    assert "STATE.settings_touched = True" in block
    # It only writes the draft dict — no engine call, no ROS publish.
    for forbidden in ("start_run", "publish_operator_command", "engine."):
        assert forbidden not in block, f"/api/draft must not call {forbidden}"


def test_the_active_scenario_is_reported_separately_from_the_draft():
    app = _read(os.path.join(REPO, "web", "app.py"))
    assert '"active_scenario"' in app
    assert '"run_active"' in app
    html = _read(INDEX)
    assert "s.active_scenario" in html
    assert "Running now:" in html and "Next:" in html, \
        "the operator must be able to tell the two apart"


def test_generate_and_plan_is_the_only_path_from_draft_to_active():
    html = _read(INDEX)
    assert '$("#c-reset").onclick' in html
    assert 'command("reset", readSettings())' in html
    # A dropdown change must not start anything.
    changed = html[html.index("async function onDraftChanged()"):
                   html.index("function syncScenarioControls(")]
    for forbidden in ('command("reset"', "readSettings()"):
        assert forbidden not in changed, "changing a control must not start a run"


def test_an_active_run_requires_explicit_confirmation_before_being_discarded():
    html = _read(INDEX)
    assert "window.confirm" in html
    assert "A PHYSICAL Isaac Sim run is in progress" in html, \
        "a physical run deserves its own, stronger wording"
    assert "Reset it and generate a new plan?" in html


def test_the_generate_button_renames_itself_when_a_run_is_active():
    html = _read(INDEX)
    # ONE label drives both the button and the help text, so they cannot drift.
    assert 'const genLabel = s.run_active ? "Reset run & generate" : "Generate & plan"' in html
    assert "gen.textContent = genLabel" in html
    assert 'press “${genLabel}”' in html, "the help must quote the actual button"


def test_the_draft_survives_page_switches():
    html = _read(INDEX)
    assert "localStorage" in html and "wisepack-draft" in html
    assert "loadDraft()" in html and "saveDraft(" in html


def test_the_draft_defaults_to_the_isaac_preset_for_the_isaac_backend():
    app = _read(os.path.join(REPO, "web", "app.py"))
    block = app[app.index('"default_preset"'):app.index('"default_preset"') + 260]
    assert "isaac_cylinders_smoke" in block
    assert 'snap.execution_backend == "isaac"' in block
    assert "mixed_pipes_dense" in block, "other modes keep their normal preset"


def test_a_user_selected_preset_is_not_overwritten_by_the_isaac_default():
    html = _read(INDEX)
    block = html[html.index("function syncScenarioControls("):
                 html.index("function fillSelects(")]
    assert "if (!DRAFT_TOUCHED) {" in block, \
        "the seeding branch must be skipped once the operator has chosen"
    assert "s.default_preset" in block


def test_the_operator_may_still_choose_another_preset_under_isaac():
    """The Isaac default is a DEFAULT, not a restriction."""
    html = _read(INDEX)
    block = html[html.index("function syncScenarioControls("):
                 html.index("function fillSelects(")]
    # Nothing removes or disables options for the Isaac backend.
    assert "removeChild" not in block
    assert "disabled = true" not in block


def test_state_polling_cannot_revert_a_selection():
    """The regression that made the dropdown feel broken even when unlocked."""
    html = _read(INDEX)
    assert "let DRAFT_TOUCHED = false;" in html
    assert "DRAFT_TOUCHED = true;" in html


def test_the_active_scenario_still_drives_the_header():
    app = _read(os.path.join(REPO, "web", "app.py"))
    assert 'active_preset = scenario.get("preset")' in app
    html = _read(INDEX)
    # The header scenario badge reads the ACTIVE scenario id, not the draft.
    assert "s.scenario && s.scenario.scenario_id" in html


# --------------------------------------------------------------------------- #
# Nothing regressed structurally
# --------------------------------------------------------------------------- #

def test_the_frontend_javascript_parses():
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:                                     # pragma: no cover
        pytest.skip("node is not installed")
    html = _read(INDEX)
    body = "\n".join(re.findall(r'<script>\n"use strict";(.*?)</script>', html, re.S))
    result = subprocess.run([node, "--check", "-"], input=body,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
