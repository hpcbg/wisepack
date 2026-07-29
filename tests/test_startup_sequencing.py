"""The Isaac-mode startup regression, and the guarantees that replace it.

WHAT WENT WRONG, so the tests below read as more than string matching:

  1. The launcher BLOCKED on Isaac reporting READY before `docker run` was
     reached, so port 8080 stayed closed for as long as Isaac took to compile
     shaders. An operator watching a dead port cannot tell a slow simulator
     from a broken launcher.

  2. `robot:="${WISEPACK_ISAAC_ROBOT:-}"` expanded to the literal `robot:=`
     whenever the variable was unset — and it was ALWAYS unset, because the
     launcher never passed it through `docker run`. `ros2 launch` rejects that
     as a malformed argument and exits on its first line, in EVERY
     container-backed mode. The wrapper then waited out a fixed timeout for a
     topic that was never coming, announced "WISEPACK stack up" and started the
     dashboard against nothing: container Up, stage IDLE, no run, topics
     WAITING, and not one message naming the cause.

  3. The launcher printed `robot : <configured default>` — a placeholder, not a
     diagnostic, and the same unresolved value that became `robot:=`.

NO ISAAC, NO DOCKER, NO GPU. These are contract tests over the launcher script,
the resolver, the status writer and the dashboard payload.
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

DASHBOARD = os.path.join(REPO, "run_wisepack_dashboard.sh")
RESOLVER = os.path.join(REPO, "scripts", "resolve_robot.py")
STATUS = os.path.join(REPO, "scripts", "startup_status.py")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _code(path: str) -> str:
    """The shell script's CODE, with comment lines removed.

    The comments explain the regression and necessarily quote the strings that
    caused it — `robot:=`, `<configured default>`. A property about what the
    script DOES must not be satisfied, or broken, by prose describing what it
    used to do.
    """
    return "\n".join(line for line in _read(path).splitlines()
                      if not line.lstrip().startswith("#"))


def _run(argv, env=None):
    merged = dict(os.environ)
    merged.pop("WISEPACK_ISAAC_ROBOT", None)
    merged.update(env or {})
    return subprocess.run([sys.executable] + argv, capture_output=True,
                          text=True, env=merged, timeout=60)


# --------------------------------------------------------------------------- #
# 1. Dashboard availability is not blocked by Isaac startup
# --------------------------------------------------------------------------- #


def test_nothing_blocks_between_starting_isaac_and_starting_the_stack():
    text = _read(DASHBOARD)
    start = text.index("ISAAC_WATCHER_PID=$!")
    docker = text.index("DOCKER_RUN=(docker run")
    assert start < docker, "Isaac is started before the stack, and not waited on"
    between = text[start:docker]
    for blocker in ("ISAAC_READY=1", "for _ in $(seq 1 \"$ISAAC_READY_TIMEOUT\")",
                    "waiting up to"):
        assert blocker not in between, \
            f"{blocker!r} would delay the dashboard behind Isaac again"


def test_the_readiness_wait_became_a_background_watcher():
    text = _read(DASHBOARD)
    assert "ISAAC_WATCHER_PID=$!" in text
    assert "not blocking on Isaac" in text
    # ...and the launcher stops the watcher it owns, and only that one.
    cleanup = text[text.index("isaac_cleanup() {"):
                   text.index("THE LAUNCHER NO LONGER STARTS ISAAC")]
    assert 'kill -TERM "$ISAAC_WATCHER_PID"' in cleanup
    assert 'kill -TERM "$ISAAC_SUPERVISOR_PID"' in cleanup


def test_the_isaac_readiness_gate_upstream_is_untouched():
    """Starting the UI early must not weaken authorisation.

    Approval still requires an active run, a SCENE_READY correlated to it, a
    matching robot and profile revision, a matching fingerprint and a verified
    home pose — none of which the launcher can grant.
    """
    bridge = _read(os.path.join(
        REPO, "wisepack_ws", "src", "wisepack_orchestration",
        "wisepack_orchestration", "isaac_bridge.py"))
    ready = bridge[bridge.index("def scene_ready("):
                   bridge.index("def scene_block_reason(")]
    for gate in ("reset_in_progress", "reset_failed_reason", "scene_mismatch",
                 "robot_model_error", "scene_revision == self.required_revision"):
        assert gate in ready
    ack = _read(os.path.join(
        REPO, "wisepack_ws", "src", "wisepack_core", "wisepack_core",
        "isaac_contract.py"))
    mism = ack[ack.index("    def mismatches("):ack.index("class IsaacFeedback")]
    for facet in ("robot_id", "robot_profile_revision", "scene_fingerprint",
                  "robot_home_verified", "container_empty_verified"):
        assert facet in mism


# --------------------------------------------------------------------------- #
# 2. The malformed launch argument
# --------------------------------------------------------------------------- #


def test_an_empty_robot_never_reaches_ros2_launch():
    """THE regression. `robot:=` killed the ROS stack in every container mode."""
    text = _code(DASHBOARD)
    assert 'robot:="${WISEPACK_ISAAC_ROBOT:-}"' not in text, \
        "an unset variable must not expand into a launch argument"
    # The argument is built in an array and appended only when non-empty.
    assert "LAUNCH_ARGS=(preset:=" in text
    assert 'if [ -n "${WISEPACK_ISAAC_ROBOT:-}" ]; then' in text
    assert 'LAUNCH_ARGS+=(robot:="${WISEPACK_ISAAC_ROBOT}")' in text
    assert 'ros2 launch wisepack_bringup demo.launch.py "${LAUNCH_ARGS[@]}"' in text


def test_the_resolved_robot_is_passed_into_the_container():
    """It never was, which is why an explicit override had no effect either."""
    text = _read(DASHBOARD)
    start = text.index("DOCKER_RUN=(docker run")
    docker = text[start:text.index("bash -lc '", start)]
    assert '-e "WISEPACK_ISAAC_ROBOT=$ROBOT_ID"' in docker
    assert '-e "WISEPACK_STARTUP_STATUS=$STACK_STATUS"' in docker


def test_the_launch_file_still_declares_an_optional_robot():
    """Omitting the argument must be legal, so the orchestrator can resolve it."""
    launch = _read(os.path.join(REPO, "wisepack_ws", "src", "wisepack_bringup",
                                "launch", "demo.launch.py"))
    assert 'DeclareLaunchArgument(\n            "robot", default_value=""' in launch


# --------------------------------------------------------------------------- #
# 3. Essential child failure is surfaced
# --------------------------------------------------------------------------- #


def test_the_container_checks_the_launch_process_not_only_the_topic():
    text = _read(DASHBOARD)
    block = text[text.index("STACK_UP=0"):text.index("if [ \"$STACK_UP\" -eq 1 ]")]
    assert 'kill -0 "$LAUNCH_PID"' in block, "liveness, every iteration"
    assert "ros2 launch exited with status" in block
    assert "tail -25 /tmp/wisepack_stack.log" in block, "and the log tail"


def test_a_failed_stack_is_never_announced_as_up():
    text = _read(DASHBOARD)
    up = text.index('echo "[container] WISEPACK stack up')
    guard = text.rindex('if [ "$STACK_UP" -eq 1 ]; then', 0, up)
    assert guard < up, '"stack up" must be inside the success branch'
    assert "WISEPACK stack is NOT running" in text
    assert "Execution is DEGRADED" in text


def test_a_failed_stack_still_serves_the_dashboard():
    """Diagnostics is where the operator reads WHY. It must stay reachable."""
    text = _read(DASHBOARD)
    tail = text[text.index('echo "[container] WISEPACK stack is NOT running'):]
    assert "exec python3 web/app.py" in tail


def test_nothing_restarts_a_failed_process_indefinitely():
    text = _read(DASHBOARD)
    code = "\n".join(line for line in text.splitlines()
                     if not line.strip().startswith("#"))
    for reckless in ("until ros2 launch", "while ros2 launch", "--restart",
                     "restart=always"):
        assert reckless not in code


def test_a_later_launch_death_is_also_surfaced():
    """A heartbeat, so a stale "running" cannot outlive the process."""
    text = _read(DASHBOARD)
    assert "HEARTBEAT_PID=$!" in text
    beat = text[text.index("# A background heartbeat"):text.index("HEARTBEAT_PID=$!")]
    assert 'while kill -0 "$LAUNCH_PID"' in beat
    assert "exited during the run" in beat


# --------------------------------------------------------------------------- #
# 4. Robot resolution
# --------------------------------------------------------------------------- #


def test_the_launcher_prints_a_concrete_robot_and_its_source():
    text = _code(DASHBOARD)
    assert "<configured default>" not in text, "a placeholder is not a diagnostic"
    assert 'echo "[isaac-launch] robot        : $ROBOT_ID"' in text
    assert 'echo "[isaac-launch] robot source : $ROBOT_SOURCE"' in text


def test_the_launcher_refuses_an_unresolved_robot_before_starting_anything():
    text = _read(DASHBOARD)
    guard = text[text.index('case "$ROBOT_ID" in'):
                 text.index('export WISEPACK_ISAAC_ROBOT="$ROBOT_ID"')]
    assert '*"<"*' in guard and '*">"*' in guard and '*" "*' in guard
    assert "exit 5" in guard
    # ...and the resolution happens BEFORE Isaac or the container is started.
    assert text.index("resolve_robot.py") < text.index("starting the Isaac supervisor")
    assert text.index("resolve_robot.py") < text.index("DOCKER_RUN=(docker run")


def test_the_resolver_prints_a_concrete_id_and_source():
    result = _run([RESOLVER])
    assert result.returncode == 0, result.stderr
    fields = result.stdout.strip().split("\t")
    assert len(fields) == 5
    robot_id, source, revision, path, default = fields
    assert robot_id and "<" not in robot_id and " " not in robot_id
    assert source in ("command line", "environment", "scenario",
                      "registry default")
    assert revision and os.path.isfile(path) and default


def test_the_resolver_reports_the_environment_override_and_its_source():
    result = _run([RESOLVER], env={"WISEPACK_ISAAC_ROBOT": "xarm7"})
    assert result.returncode == 0, result.stderr
    robot_id, source = result.stdout.strip().split("\t")[:2]
    assert robot_id == "xarm7"
    assert source == "environment"


def test_the_resolver_reports_the_registry_default_by_name():
    result = _run([RESOLVER])
    robot_id, source, _, _, default = result.stdout.strip().split("\t")
    assert source == "registry default"
    assert robot_id == default, "the effective default must be concrete"


@pytest.mark.parametrize("bad", ["<configured default>", "default", "none",
                                 "unresolved", "some robot"])
def test_the_resolver_rejects_a_placeholder(bad):
    result = _run([RESOLVER], env={"WISEPACK_ISAAC_ROBOT": bad})
    assert result.returncode == 5, result.stdout
    assert "ERROR" in result.stderr
    assert "<" not in result.stdout


def test_the_resolver_rejects_an_unknown_robot():
    result = _run([RESOLVER], env={"WISEPACK_ISAAC_ROBOT": "ur10e"})
    assert result.returncode == 5
    assert "unknown robot" in result.stderr


def test_the_resolver_rejects_an_incompatible_robot_and_preset():
    """Refused before Isaac spends a minute loading an asset it may not use."""
    result = _run([RESOLVER, "--preset", "mixed_pipes_dense"])
    assert result.returncode == 5
    assert "not among them" in result.stderr or "configured for" in result.stderr


def test_the_orchestrator_and_the_host_get_the_same_value():
    text = _read(DASHBOARD)
    # One assignment, exported once, consumed by both.
    assert text.count('export WISEPACK_ISAAC_ROBOT="$ROBOT_ID"') == 1
    # The supervisor is given the resolved id and passes it to every Isaac
    # generation it starts (see scripts/isaac_supervisor.py).
    supervisor = text[text.index("setsid env ROS_DOMAIN_ID"):
                      text.index("ISAAC_SUPERVISOR_PID=$!")]
    assert '--robot "$ROBOT_ID"' in supervisor
    assert '-e "WISEPACK_ISAAC_ROBOT=$ROBOT_ID"' in text


def test_a_host_and_orchestrator_robot_mismatch_is_rejected():
    """Even with one source, a stale simulator on the domain must not be used."""
    from wisepack_core.isaac_contract import SceneAcknowledgement
    ack = SceneAcknowledgement(run_id="r1", robot_id="panda",
                               robot_home_verified=True,
                               container_empty_verified=True)
    reasons = ack.mismatches(run_id="r1", scenario_id="", revision=0, preset="",
                             seed=0, fingerprint="", object_count=0,
                             robot_id="xarm7")
    assert any("robot panda" in r for r in reasons)

    app = _read(os.path.join(REPO, "simulators", "isaac", "wisepack_isaac.py"))
    refusal = app[app.index("def _pre_pick_refusal("):app.index("def prepare_smoke_run(")]
    assert "command.robot_id != self.profile.robot_id" in refusal


# --------------------------------------------------------------------------- #
# 5. The startup status file
# --------------------------------------------------------------------------- #


def test_the_status_writer_records_processes_and_degradation(tmp_path):
    out = str(tmp_path / "startup-host.json")
    assert _run([STATUS, "init", "--out", out, "--scope", "host",
                 "--mode", "isaac", "--robot", "panda",
                 "--robot-source", "environment"]).returncode == 0
    doc = json.loads(_read(out))
    assert doc["scope"] == "host"
    assert doc["robot"]["effective"] == "panda"
    assert doc["robot"]["source"] == "environment"
    assert doc["degraded"] is False
    assert {p["name"] for p in doc["processes"]} >= {"isaac-sim", "isaac-watcher"}

    _run([STATUS, "proc", "--out", out, "--name", "isaac-sim", "--pid", "42",
          "--running", "1"])
    _run([STATUS, "proc", "--out", out, "--name", "isaac-sim", "--running", "0",
          "--exit-code", "5", "--error", "ROBOT_MODEL_INVALID"])
    _run([STATUS, "degrade", "--out", out, "--reason", "Isaac Sim exited"])
    doc = json.loads(_read(out))
    proc = next(p for p in doc["processes"] if p["name"] == "isaac-sim")
    assert proc["running"] is False and proc["exit_code"] == 5
    assert "ROBOT_MODEL_INVALID" in proc["last_error"]
    assert proc["last_heartbeat"]
    assert doc["degraded"] is True and "exited" in doc["degraded_reason"]


def test_concurrent_writers_do_not_lose_each_others_updates(tmp_path):
    """Two writers per scope. Without a lock one silently discards the other.

    Observed: the container recorded the dashboard as running while its
    heartbeat was already ticking, and the row came back "unknown".
    """
    import threading
    out = str(tmp_path / "startup-stack.json")
    _run([STATUS, "init", "--out", out, "--scope", "stack", "--mode", "isaac"])

    names = [f"proc-{n}" for n in range(12)]
    threads = [threading.Thread(target=_run, args=(
        [STATUS, "proc", "--out", out, "--name", n, "--running", "1"],))
        for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    doc = json.loads(_read(out))
    recorded = {p["name"] for p in doc["processes"]}
    assert set(names) <= recorded, f"lost: {sorted(set(names) - recorded)}"


def test_the_status_file_is_replaced_atomically():
    """A reader polling it must never see half a document."""
    src = _read(STATUS)
    assert "os.replace(tmp, path)" in src
    assert "mkstemp" in src
    # ...which is also why the CONTAINER is given the directory, not the file:
    # an atomic replace changes the inode and a file bind-mount would freeze.
    dashboard = _read(DASHBOARD)
    assert 'HOST_STATUS_MOUNT=(-v "$HOST_STATUS_OWNED_DIR:$HOST_STATUS_OWNED_DIR:ro")' \
        in dashboard
    assert 'rm -rf -- "$HOST_STATUS_OWNED_DIR"' in dashboard, \
        "the launcher removes only the directory it created"


def test_a_status_write_failure_never_takes_the_launcher_down(tmp_path):
    """Reporting is best-effort. The run is still valid without it."""
    unwritable = str(tmp_path / "nope" / "x" / "startup.json")
    os.makedirs(os.path.dirname(os.path.dirname(unwritable)))
    os.chmod(os.path.dirname(os.path.dirname(unwritable)), 0o500)
    try:
        result = _run([STATUS, "init", "--out", unwritable, "--scope", "host"])
        assert result.returncode == 0
    finally:
        os.chmod(os.path.dirname(os.path.dirname(unwritable)), 0o700)


def test_the_launcher_writes_the_status_before_starting_anything():
    text = _read(DASHBOARD)
    init = text.index("scripts/startup_status.py")
    assert init < text.index("starting the Isaac supervisor")
    assert init < text.index("DOCKER_RUN=(docker run")
    # A previous run's stack status must not be read as this run's.
    assert 'rm -f "$STACK_STATUS"' in text


# --------------------------------------------------------------------------- #
# 6. Diagnostics
# --------------------------------------------------------------------------- #


def test_diagnostics_merges_both_startup_scopes():
    import diagnostics
    assert [s for s, _ in diagnostics.STARTUP_FILES] == ["host", "stack"]
    names = [n for n, _, _ in diagnostics.EXPECTED_PROCESSES]
    for required in ("dashboard", "ros-launch", "orchestrator", "perception-sim",
                     "twin-validator", "anomaly-simulator", "isaac-sim"):
        assert required in names


def test_a_process_that_never_reported_is_shown_rather_than_omitted(monkeypatch,
                                                                    tmp_path):
    import diagnostics
    monkeypatch.setattr(diagnostics, "RESULTS_DIR", str(tmp_path))
    startup = diagnostics._startup_status()
    by_name = {p["process"]: p for p in startup["processes"]}
    assert by_name["ros-launch"]["running"] == "not reported"
    assert by_name["ros-launch"]["expected"] == "yes"


def test_a_dead_process_reads_differently_from_one_that_never_reported(
        monkeypatch, tmp_path):
    import diagnostics
    monkeypatch.setattr(diagnostics, "RESULTS_DIR", str(tmp_path))
    out = str(tmp_path / "startup-stack.json")
    _run([STATUS, "init", "--out", out, "--scope", "stack", "--mode", "isaac"])
    _run([STATUS, "proc", "--out", out, "--name", "ros-launch", "--running", "0",
          "--exit-code", "1", "--error", "malformed launch argument"])
    startup = diagnostics._startup_status()
    row = next(p for p in startup["processes"] if p["process"] == "ros-launch")
    assert row["running"] == "NO"
    assert row["exit_code"] == 1
    assert "malformed" in row["last_error"]


def test_the_blocker_is_named_instead_of_bare_idle(monkeypatch, tmp_path):
    import diagnostics

    class _Snap:
        run_id = ""

    monkeypatch.setattr(diagnostics, "RESULTS_DIR", str(tmp_path))
    out = str(tmp_path / "startup-stack.json")
    _run([STATUS, "init", "--out", out, "--scope", "stack", "--mode", "isaac"])
    _run([STATUS, "proc", "--out", out, "--name", "ros-launch", "--running", "0",
          "--exit-code", "1"])
    _run([STATUS, "degrade", "--out", out, "--reason",
          "ros2 launch exited with status 1"])
    startup = diagnostics._startup_status()
    blocker = diagnostics._startup_blocker(_Snap(), startup, {})
    assert "ros2 launch exited" in blocker

    # ...and once a run exists there is no blocker to report.
    class _Running:
        run_id = "run-1"
    assert diagnostics._startup_blocker(_Running(), startup, {}) == ""


def test_diagnostics_reports_the_robot_resolution_chain(monkeypatch, tmp_path):
    import diagnostics
    monkeypatch.setattr(diagnostics, "RESULTS_DIR", str(tmp_path))
    out = str(tmp_path / "startup-host.json")
    _run([STATUS, "init", "--out", out, "--scope", "host", "--mode", "isaac",
          "--robot", "xarm7", "--robot-source", "environment",
          "--robot-revision", "abc123", "--registry-path", "/x/isaac_robots.yaml",
          "--registry-default", "panda"])
    rows = diagnostics._robot_startup_rows(
        diagnostics._startup_status(),
        {"robot_id": "xarm7", "robot_status": {"robot_id": "xarm7"},
         "acknowledged_scene": {"robot_id": "xarm7"}}, None)
    assert rows["robot_registry_loaded"] == "yes"
    # The basename only — this payload is served over HTTP and must not carry
    # the host's filesystem layout.
    assert rows["robot_registry_resolved"] == "isaac_robots.yaml"
    assert "/" not in rows["robot_registry_resolved"]
    assert rows["robot_configured_default"] == "panda"
    assert rows["robot_effective_at_startup"] == "xarm7"
    assert rows["robot_selection_source"] == "environment"
    assert rows["robot_startup_profile_revision"] == "abc123"
    assert rows["robot_host_id"] == "xarm7"
    assert rows["robot_orchestrator_id"] == "xarm7"
    assert rows["robot_acknowledged_scene_id"] == "xarm7"


def test_the_diagnostics_page_renders_the_startup_table():
    html = _read(os.path.join(REPO, "web", "diagnostics.html"))
    assert 'id="startup"' in html
    for column in ("Process", "PID", "Expected", "Running", "Exit",
                   "Last heartbeat", "Last error"):
        assert column in html


# --------------------------------------------------------------------------- #
# 7. Logical modes are unchanged and honest
# --------------------------------------------------------------------------- #


def test_sim_mode_never_starts_the_ros_stack_and_is_untouched():
    """`sim` was the one mode the regression could not reach. Keep it that way."""
    text = _read(DASHBOARD)
    sim = text[text.index('if [ "$MODE" = "sim" ]; then'):
               text.index("# ---- live modes")] if "# ---- live modes" in text \
        else text[text.index('if [ "$MODE" = "sim" ]; then'):
                  text.index("SOURCE=\"ros\"")]
    assert "ros2 launch" not in sim
    assert "LAUNCH_ARGS" not in sim


def test_the_robot_is_resolved_only_for_a_physical_backend():
    text = _code(DASHBOARD)
    block = text[text.index('ROBOT_ID=""'):text.index("scripts/startup_status.py")]
    assert 'if [ "$EXECUTION_BACKEND" = "isaac" ]; then' in block
    # A logical run resolves nothing and therefore claims nothing: the
    # placeholders are cleared before the resolver is ever reached.
    assert text.index('ROBOT_ID=""') < text.index("resolve_robot.py")


def test_the_logical_modes_show_a_fixed_execution_source():
    html = _read(os.path.join(REPO, "web", "index.html"))
    assert 'id="s-robot-fixed"' in html
    assert "Logical workflow simulator" in html
    branch = html[html.index("if (s.robot_selector) {"):
                  html.index("if (s.robot_selector && rnote)")]
    assert 'rb.style.display = "none"' in branch, "hidden, not merely disabled"
    assert 'rfixed.style.display = ""' in branch
    assert 'rlabel.textContent = "Execution source"' in branch


def test_the_selector_is_shown_only_in_isaac_modes():
    app = _read(os.path.join(REPO, "web", "app.py"))
    assert '"robot_selector": physical,' in app
    html = _read(os.path.join(REPO, "web", "index.html"))
    assert "if (s.robot_selector) {" in html
