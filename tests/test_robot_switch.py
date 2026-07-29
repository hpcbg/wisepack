"""Switching robots from the web application.

WHAT WENT WRONG
---------------
Selecting the other arm and pressing "Reset run & generate" reset the workcell
objects and left the previous robot standing on the stage.

The robot is chosen when the Isaac PROCESS starts: the adapter is built from the
profile and the USD model is referenced into the stage, and neither can be
changed afterwards. A cross-robot reset re-bound only the ORCHESTRATOR's view of
the robot and then published RESET_SCENE — to the still-running old simulator,
which rebuilt its own workcell and acknowledged it with its own robot id. The
gate correctly refused that acknowledgement, so nothing unsafe happened, but the
operator watched the objects reset around the wrong arm.

So a robot change is now a different operation from a scene reset: a bounded
host-side restart, driven by a supervisor that owns the Isaac process group,
through an allowlisted request the container may write and nothing else.

NO ISAAC, NO DOCKER, NO GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _pkg in ("wisepack_core",):
    _path = os.path.join(REPO, "wisepack_ws", "src", _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)
sys.path.insert(0, os.path.join(REPO, "web"))

from wisepack_core.isaac_contract import (
    IsaacCommand, IsaacCommandType, IsaacFeedback, IsaacState,
    SceneAcknowledgement,
)
from wisepack_core.robot_switch import (
    ALLOWED_OPS, CONTROL_DIR_ENV, IN_FLIGHT_PHASES, PHASE_FAILED, PHASE_READY,
    PHASE_STARTING, PHASE_STOPPING, RobotSwitchClient, SupervisorStatus,
    SwitchRequest, describe_phase,
)

DASHBOARD = os.path.join(REPO, "run_wisepack_dashboard.sh")
SUPERVISOR = os.path.join(REPO, "scripts", "isaac_supervisor.py")
BRIDGE = os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                      "wisepack_orchestration", "isaac_bridge.py")
ORCH = os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                    "wisepack_orchestration", "hitl_orchestrator.py")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _code(path: str) -> str:
    """Shell CODE with comment lines removed — prose quotes the old behaviour."""
    return "\n".join(line for line in _read(path).splitlines()
                     if not line.lstrip().startswith("#"))


def _py_code(path: str) -> str:
    import ast
    tree = ast.parse(_read(path))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


# --------------------------------------------------------------------------- #
# 1. Same robot versus different robot
# --------------------------------------------------------------------------- #


def test_the_reset_path_distinguishes_the_two_operations():
    src = _read(ORCH)
    reset = src[src.index("def _reset_run("):src.index("def _write_artifacts(")]
    assert "robot_switch = (self.isaac is not None" in reset
    assert "robot_id != self.isaac.robot_id" in reset
    # Same robot -> the existing in-process scene reset, unchanged.
    assert "self.isaac.request_scene_reset(" in reset
    # Different robot -> a host restart.
    assert "self.isaac.request_robot_switch(" in reset


def test_a_same_robot_reset_never_requests_a_host_restart():
    src = _py_code(ORCH)
    reset = src[src.index("def _reset_run("):src.index("def _write_artifacts(")]
    switch_at = reset.index("request_robot_switch")
    branch = reset[:switch_at]
    assert "if robot_switch:" in branch, \
        "the restart must sit inside the cross-robot branch"
    # ...and the scene reset sits in the else.
    assert "elif self.isaac is not None:" in reset


def test_the_scene_request_is_not_sent_during_a_cross_robot_reset():
    """THE bug. The old simulator would rebuild its own workcell and ack it."""
    src = _py_code(ORCH)
    reset = src[src.index("def _reset_run("):src.index("def _write_artifacts(")]
    switch = reset.index("request_robot_switch")
    scene = reset.index("request_scene_reset")
    assert switch < scene, "the switch branch comes first"
    # They are mutually exclusive branches of one if/elif, so a cross-robot
    # reset cannot reach request_scene_reset at all.
    assert reset.count("request_scene_reset") == 1


def test_the_bridge_refuses_to_sync_a_scene_while_switching():
    src = _read(BRIDGE)
    body = src[src.index("def _sync_scene_if_needed("):src.index("def _request_scene(")]
    assert "self.switch_in_flight or self.switch_failed_reason" in body
    assert "return" in body


# --------------------------------------------------------------------------- #
# 2. The supervisor and its narrow control surface
# --------------------------------------------------------------------------- #


def test_only_switch_robot_is_allowed():
    assert ALLOWED_OPS == ("switch_robot",)
    src = _read(SUPERVISOR)
    assert "if op not in ALLOWED_OPS:" in src
    assert "is not allowed" in src


def test_the_container_gets_no_docker_socket_or_host_shell():
    """The whole reason this is a file protocol and not a shell call."""
    text = _code(DASHBOARD)
    docker = text[text.index("DOCKER_RUN=(docker run"):text.index("bash -lc '")]
    for forbidden in ("/var/run/docker.sock", "docker.sock", "--pid=host",
                      "--privileged=true", "/proc:", "-v /:/"):
        assert forbidden not in docker, f"the container must not get {forbidden}"
    # Exactly one read-write mount beyond the repository, and it holds only
    # control documents.
    assert 'CONTROL_MOUNT=(-v "$ISAAC_CONTROL_DIR:$ISAAC_CONTROL_DIR:rw")' in text


def test_the_request_schema_carries_only_identifiers():
    """Nothing in a request is executed. No command, path, signal or pid."""
    request = SwitchRequest(requested_robot_id="panda",
                            requested_profile_revision="abc123",
                            run_id="run-1", scenario_revision=3)
    doc = request.to_dict()
    assert set(doc) == {"op", "request_id", "requested_robot_id",
                        "requested_profile_revision", "run_id",
                        "scenario_revision", "timestamp"}
    assert doc["op"] == "switch_robot"
    blob = json.dumps(doc)
    for shellish in ("/", "$", ";", "|", "&", "`"):
        assert shellish not in doc["requested_robot_id"]
    assert "python" not in blob and "bash" not in blob


def test_the_supervisor_validates_the_robot_before_stopping_anything():
    """A switch to a robot that does not exist must cost nothing."""
    src = _read(SUPERVISOR)
    handle = src[src.index("    def handle(self"):src.index("    def switch(self")]
    validate = handle.index("load_registry")
    switch = handle.index("self.switch(")
    assert validate < switch, "validate first, stop second"
    assert "is not a usable robot" in handle
    assert "profile revision" in handle


def test_the_supervisor_stops_the_old_robot_before_starting_the_new_one():
    src = _read(SUPERVISOR)
    body = src[src.index("    def switch(self"):src.index("    def run(self")]
    stop = body.index("self.stop_isaac()")
    start = body.index("self.start_isaac(robot_id)")
    assert stop < start
    # ...and refuses to start a replacement if the old one is still there.
    assert "did not fully stop" in body
    assert "on the same GPU and DDS domain" in body
    # The refusal RETURNS: no replacement is started after a failed stop.
    failed_at = body.index("PHASE_FAILED")
    assert body.index("return", failed_at) < body.index("self.start_isaac(robot_id)")


def test_the_supervisor_waits_on_group_membership_not_the_leader():
    """Kit spawns children that outlive their parent during shutdown."""
    src = _read(SUPERVISOR)
    stop = src[src.index("    def stop_isaac(self"):src.index("    def note_ready(")]
    assert "self.group_size()" in stop
    assert "SIGKILL" in stop, "bounded escalation"
    assert "remain in group" in stop, "and it verifies"


def test_a_defunct_child_does_not_count_as_a_live_group_member():
    """A zombie keeps its process-group id and made every switch time out.

    Measured: STOPPING_OLD_ROBOT sat with one `<defunct>` python.sh in the
    group, `group_size()` reported 1 forever, and the transaction would have
    ended in "the previous simulator did not fully stop".
    """
    src = _read(SUPERVISOR)
    body = src[src.index("    def group_size(self"):src.index("    def start_isaac(")]
    assert "pgid=,pid=,stat=" in body, "the process state has to be read"
    assert 'parts[2].startswith("Z")' in body
    assert "self._reap()" in body
    # ...and the direct child is reaped before the final verdict.
    stop = src[src.index("    def stop_isaac(self"):src.index("    def note_ready(")]
    assert "proc.wait(timeout=5)" in stop


def test_every_generation_gets_its_own_number():
    src = _read(SUPERVISOR)
    start = src[src.index("    def start_isaac(self"):src.index("    def stop_isaac(")]
    assert "self.generation += 1" in start
    assert 'env["WISEPACK_ISAAC_GENERATION"] = str(self.generation)' in start
    assert 'env["WISEPACK_ISAAC_ROBOT"] = robot_id' in start


def test_the_supervisor_never_restarts_by_itself():
    src = _py_code(SUPERVISOR)
    loop = src[src.index("def run(self)"):]
    assert loop.count("self.start_isaac(") == 1, \
        "only the initial start; a death is reported, not retried"


# --------------------------------------------------------------------------- #
# 3. The client
# --------------------------------------------------------------------------- #


def test_a_request_is_written_atomically_and_read_once(tmp_path):
    client = RobotSwitchClient(str(tmp_path))
    assert client.available
    rid = client.request_switch(SwitchRequest(requested_robot_id="panda",
                                              run_id="run-1"))
    files = sorted(p.name for p in tmp_path.iterdir())
    assert len(files) == 1 and files[0].startswith("request-")
    assert not files[0].startswith("."), "no temporary file left behind"
    doc = json.loads((tmp_path / files[0]).read_text())
    assert doc["request_id"] == rid and doc["op"] == "switch_robot"


def test_a_request_is_readable_by_the_host_user(tmp_path):
    """The writer is root in a container; the reader is the invoking user.

    mkstemp creates 0600, and the supervisor then rejected every request with
    "Permission denied" — a switch that silently never happened. The document
    holds four identifiers and a timestamp; there is nothing in it to protect.
    """
    client = RobotSwitchClient(str(tmp_path))
    client.request_switch(SwitchRequest(requested_robot_id="panda"))
    written = next(p for p in tmp_path.iterdir() if p.name.startswith("request-"))
    assert oct(written.stat().st_mode)[-3:] == "644"


def test_the_status_file_is_readable_by_the_container():
    """The mirror image: the supervisor writes it, the container reads it."""
    src = _read(SUPERVISOR)
    assert "os.chmod(self.status_path, 0o644)" in src


def test_without_a_control_directory_switching_is_refused_not_faked():
    client = RobotSwitchClient("")
    assert not client.available
    reason = client.unavailable_reason()
    assert "restart the launcher" in reason
    with pytest.raises(RuntimeError):
        client.request_switch(SwitchRequest(requested_robot_id="panda"))


def test_the_status_reader_never_raises_on_a_missing_or_broken_file(tmp_path):
    client = RobotSwitchClient(str(tmp_path))
    assert client.status().present is False
    (tmp_path / "supervisor-status.json").write_text("{not json")
    assert client.status().present is False


def test_the_phase_vocabulary_is_shared_by_both_ends():
    """Two copies of a phase list agree only until one of them is edited."""
    src = _read(SUPERVISOR)
    assert "from wisepack_core.robot_switch import (" in src
    for phase in ("PHASE_STOPPING", "PHASE_STARTING", "PHASE_READY",
                  "PHASE_FAILED"):
        assert phase in src
    assert 'PHASE_STOPPING = "STOPPING_OLD_ROBOT"' not in src, \
        "the supervisor must not restate the vocabulary"


def test_the_phase_labels_name_the_robots():
    assert describe_phase(PHASE_STOPPING, previous="xArm 7") == \
        "Stopping the xArm 7 simulator"
    assert describe_phase(PHASE_STARTING, requested="Panda") == \
        "Starting the Panda simulator"


# --------------------------------------------------------------------------- #
# 4. Stale data
# --------------------------------------------------------------------------- #


def test_the_generation_survives_a_round_trip():
    command = IsaacCommand(command=IsaacCommandType.RUN_BEGIN, run_id="r1",
                           robot_id="panda", simulator_generation=3)
    assert IsaacCommand.from_json(command.to_json()).simulator_generation == 3
    feedback = IsaacFeedback(state=IsaacState.READY, run_id="r1",
                             robot_id="panda", simulator_generation=3)
    assert IsaacFeedback.from_json(feedback.to_json()).simulator_generation == 3


def test_a_scene_from_an_earlier_generation_is_rejected():
    """A -> B -> A returns to the same robot id. Only the generation differs."""
    ack = SceneAcknowledgement(run_id="r1", robot_id="panda",
                               simulator_generation=1,
                               robot_home_verified=True,
                               container_empty_verified=True)
    reasons = ack.mismatches(run_id="r1", scenario_id="", revision=0, preset="",
                             seed=0, fingerprint="", object_count=0,
                             robot_id="panda", simulator_generation=3)
    assert any("generation 1" in r and "generation 3" in r for r in reasons)


def test_a_matching_generation_produces_no_reason():
    ack = SceneAcknowledgement(run_id="r1", robot_id="panda",
                               simulator_generation=3,
                               robot_home_verified=True,
                               container_empty_verified=True)
    assert ack.mismatches(run_id="r1", scenario_id="", revision=0, preset="",
                          seed=0, fingerprint="", object_count=0,
                          robot_id="panda", simulator_generation=3) == []


def test_an_unstamped_generation_is_tolerated():
    """A standalone simulator started outside a supervisor sends 0."""
    ack = SceneAcknowledgement(run_id="r1", robot_id="panda",
                               robot_home_verified=True,
                               container_empty_verified=True)
    assert ack.mismatches(run_id="r1", scenario_id="", revision=0, preset="",
                          seed=0, fingerprint="", object_count=0,
                          robot_id="panda", simulator_generation=3) == []


def test_the_bridge_drops_feedback_from_an_older_generation():
    src = _read(BRIDGE)
    guard = src[src.index("STALE FEEDBACK FROM AN EARLIER SIMULATOR"):
                src.index("try:\n            self._apply(engine, feedback)")]
    assert "feedback.simulator_generation < self.expected_generation" in guard
    assert "return" in guard


def test_the_bridge_drops_execution_feedback_while_switching():
    src = _read(BRIDGE)
    guard = src[src.index("A switch is in flight: NOTHING from the outgoing"):
                src.index("try:\n            self._apply(engine, feedback)")]
    assert "self.switch_in_flight" in guard
    # ...except the lifecycle states that tell it the new simulator is up.
    assert "IsaacState.READY" in guard
    assert "IsaacState.ROBOT_MODEL_INVALID" in guard


def test_the_simulator_refuses_a_command_for_another_generation():
    src = _read(os.path.join(REPO, "simulators", "isaac", "wisepack_isaac.py"))
    body = src[src.index("def _pre_pick_refusal("):src.index("def prepare_smoke_run(")]
    assert "command.simulator_generation != SIMULATOR_GENERATION" in body
    match = src[src.index("def _scene_matches("):src.index("def _sync_scene(")]
    assert "command.simulator_generation != SIMULATOR_GENERATION" in match


# --------------------------------------------------------------------------- #
# 5. Approval stays shut
# --------------------------------------------------------------------------- #


def test_no_scene_is_ready_while_a_switch_is_in_flight_or_failed():
    src = _read(BRIDGE)
    ready = src[src.index("    def scene_ready("):src.index("    def scene_block_reason(")]
    assert "not self.switch_in_flight" in ready
    assert "not self.switch_failed_reason" in ready


def test_the_block_reason_names_the_switch_phase():
    src = _read(BRIDGE)
    reason = src[src.index("    def scene_block_reason("):
                 src.index("    def switch_in_flight") if "def switch_in_flight" in src
                 else src.index("    def request_scene_reset(")]
    assert "self.switch_failed_reason" in reason
    assert "describe_phase(" in reason


def test_a_failed_switch_degrades_and_keeps_the_requested_robot():
    src = _read(BRIDGE)
    fail = src[src.index("    def _fail_switch("):src.index("    def rebind_robot(")]
    assert "enter_degraded(" in fail
    assert "approval is disabled" in fail
    assert "requested robot is kept" in fail
    # No fallback to the previous robot anywhere in it.
    code = _py_code(BRIDGE)
    body = code[code.index("def _fail_switch("):code.index("def rebind_robot(")]
    assert "switch_previous_robot" not in body.replace(
        "'previous_robot_id': self.switch_previous_robot", ""), \
        "a failed switch must not restore the previous robot"


def test_a_switch_is_bounded():
    src = _read(BRIDGE)
    assert "self.switch_timeout_s" in src
    advance = src[src.index("    def _advance_switch("):src.index("    def _fail_switch(")]
    assert "self._switch_started_at > self.switch_timeout_s" in advance
    supervisor = _read(SUPERVISOR)
    assert "STOP_TIMEOUT_S" in supervisor and "START_TIMEOUT_S" in supervisor


def test_the_switch_is_driven_from_the_node_tick_not_the_execution_loop():
    """The loop only runs after approval, and the switch is holding it shut."""
    orch = _read(ORCH)
    tick = orch[orch.index("    def _tick(self)"):orch.index("    def _publish_event(")]
    assert "self.isaac.poll_switch(self.engine)" in tick
    bridge = _read(BRIDGE)
    poll = bridge[bridge.index("    def poll_switch("):bridge.index("    def _advance_switch(")]
    assert "self._sync_scene_if_needed(engine)" in poll, \
        "the scene is requested once the new simulator is up"


# --------------------------------------------------------------------------- #
# 6. The dashboard
# --------------------------------------------------------------------------- #


def test_four_robot_values_are_reported_separately():
    app = _read(os.path.join(REPO, "web", "app.py"))
    for field in ('"active_robot_id"', '"robot_switch"',
                  '"acknowledged_scene_robot_id"'):
        assert field in app
    bridge = _read(BRIDGE)
    status = bridge[bridge.index("    def switch_status("):
                    bridge.index("    def request_robot_switch(")]
    for field in ('"requested_robot_id"', '"host_robot_id"',
                  '"previous_robot_id"', '"host_generation"'):
        assert field in status


def test_the_header_follows_the_host_not_the_selection():
    """During a switch the selected arm is not the one on the stage."""
    snap = _read(os.path.join(REPO, "web", "snapshot.py"))
    block = snap[snap.index("THE HEADER FOLLOWS THE HOST"):
                 snap.index("snap.isaac_results")]
    assert "host_robot = switch.get(\"host_robot_id\")" in block
    assert "snap.active_robot_id = host_robot or selected_id" in block


def test_the_confirmation_states_every_consequence():
    html = _read(os.path.join(REPO, "web", "index.html"))
    block = html[html.index("STATE.robot_change_requires_reset"):
                 html.index("if (!window.confirm(what)) return;")]
    for claim in ("CANCELLED", "RESTARTED", "WebRTC", "DISABLED"):
        assert claim in block, f"the dialog must state {claim}"


def test_the_scenario_panel_shows_switch_progress():
    html = _read(os.path.join(REPO, "web", "index.html"))
    assert "const sw = s.robot_switch || {};" in html
    assert "Switching from" in html
    assert "phase_label" in html
    assert "approval is disabled until the new scene" in html


def test_the_simulator_view_refreshes_its_stream_descriptor():
    app = _read(os.path.join(REPO, "web", "app.py"))
    assert '"stream_reconnect_required"' in app
    assert '"simulator_generation"' in app
    assert "NEW stream" in app
    html = _read(os.path.join(REPO, "web", "simulator.html"))
    assert "simulator generation" in html
    assert "d.stream_reconnect_required" in html


def test_diagnostics_reports_the_switch():
    diag = _read(os.path.join(REPO, "web", "diagnostics.py"))
    for row in ("robot_switch_phase", "robot_switch_requested",
                "robot_host_reported", "robot_host_generation",
                "robot_expected_generation", "robot_acknowledged_generation",
                "robot_switch_failed"):
        assert f'"{row}"' in diag


# --------------------------------------------------------------------------- #
# 7. The supervisor, exercised
# --------------------------------------------------------------------------- #


def _fake_status(tmp_path, **fields):
    doc = {"robot_id": "xarm7", "requested_robot_id": "", "phase": "SIMULATOR_READY",
           "simulator_generation": 1, "isaac_running": True,
           "simulator_ready": True, "request_id": "", "last_error": "",
           "generated_at": "now"}
    doc.update(fields)
    (tmp_path / "supervisor-status.json").write_text(json.dumps(doc))
    return doc


def test_the_status_model_separates_running_from_requested(tmp_path):
    _fake_status(tmp_path, robot_id="xarm7", requested_robot_id="panda",
                 phase="STOPPING_OLD_ROBOT", simulator_ready=False)
    status = RobotSwitchClient(str(tmp_path)).status()
    assert status.robot_id == "xarm7", "what is RUNNING"
    assert status.requested_robot_id == "panda", "what was ASKED FOR"
    assert status.switch_in_flight and not status.failed
    assert status.to_dict()["phase_label"].startswith("Stopping")


def test_a_failed_phase_is_not_in_flight(tmp_path):
    _fake_status(tmp_path, phase=PHASE_FAILED, simulator_ready=False,
                 last_error="the panda simulator exited during startup")
    status = RobotSwitchClient(str(tmp_path)).status()
    assert status.failed and not status.switch_in_flight
    assert "exited" in status.last_error


def test_the_in_flight_phases_are_exactly_the_transitional_ones():
    assert PHASE_READY not in IN_FLIGHT_PHASES
    assert PHASE_FAILED not in IN_FLIGHT_PHASES
    assert PHASE_STOPPING in IN_FLIGHT_PHASES
    assert PHASE_STARTING in IN_FLIGHT_PHASES


def test_the_supervisor_refuses_an_operation_outside_the_allowlist(tmp_path):
    """End to end through the real script, with no Isaac."""
    control = tmp_path / "control"
    control.mkdir()
    (control / "request-x.json").write_text(json.dumps({
        "op": "run_shell", "request_id": "evil", "command": "rm -rf /"}))
    # Import the supervisor and drive one request, with the process control
    # stubbed out — the point is the refusal, not a restart.
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import importlib
    mod = importlib.import_module("isaac_supervisor")
    sup = mod.Supervisor(str(control), "xarm7", str(tmp_path / "logs"))
    started = []
    sup.start_isaac = lambda robot: started.append(robot) or True
    sup.stop_isaac = lambda *a, **k: True
    for path, doc in sup._requests():
        sup.handle(path, doc)
    assert started == [], "an unlisted operation must not start anything"
    assert "not allowed" in sup.last_error
    assert (control / "request-x.json.done").exists(), \
        "the request is kept for inspection, not deleted"


def test_the_supervisor_refuses_an_unknown_robot(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    (control / "request-y.json").write_text(json.dumps({
        "op": "switch_robot", "request_id": "r2", "requested_robot_id": "ur10e"}))
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import importlib
    mod = importlib.import_module("isaac_supervisor")
    sup = mod.Supervisor(str(control), "xarm7", str(tmp_path / "logs"))
    stopped = []
    sup.start_isaac = lambda robot: True
    sup.stop_isaac = lambda *a, **k: stopped.append(1) or True
    for path, doc in sup._requests():
        sup.handle(path, doc)
    assert stopped == [], "nothing may be stopped for a robot that does not exist"
    assert sup.phase == PHASE_FAILED
    assert "not a usable robot" in sup.last_error
