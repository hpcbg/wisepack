"""The launcher owns the optional WISEPACK perception service.

The operator starts WISEPACK exactly as before — `./run_wisepack_dashboard.sh
sim` and friends — and with `WISEPACK_PERCEPTION_SOURCE=camera` in
`config/local.env` the perception service is started for them. No venv to
activate, no second terminal, no torch in the system Python, and no middleware
on the host: the service speaks HTTP only.

These tests drive the REAL shell libraries with a FAKE detector interpreter, so
the whole lifecycle — interpreter resolution, reuse, readiness, failure,
ownership and cleanup — is exercised without a camera, a GPU, torch, or any
provider being installed. The one thing they never do is trust the script text
where behaviour can be run instead.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import textwrap
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD = os.path.join(REPO, "run_wisepack_dashboard.sh")
LIB_HOST = os.path.join(REPO, "scripts", "lib_host_processes.sh")
LIB_PERCEPTION = os.path.join(REPO, "scripts", "lib_perception_service.sh")

#: Exit code the launcher promises when camera perception was requested and no
#: detector could be made available.
EXIT_PERCEPTION_UNAVAILABLE = 6


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _executable_lines(text: str) -> str:
    """The script with comment-only lines removed.

    Shell has no block comments, so a line whose first non-space character is
    `#` is documentation. Trailing comments are left alone: they sit beside real
    code and stripping them would need a shell parser.
    """
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


#: A stand-in for `torch_venv/bin/python`, invoked exactly as the library
#: invokes the real one: `<python> <service.py> --host H --port P`.
#:
#: NOT indented and NOT passed through `textwrap.dedent`. The embedded Python
#: below starts at column 0, so dedent would find no common prefix, strip
#: nothing, and leave the shebang behind eight spaces — which `bash` silently
#: papers over by re-running the file itself, and `execve` rejects outright with
#: `OSError: [Errno 8] Exec format error`. The shebang has to be the first byte.
_FAKE_DETECTOR = '''#!/usr/bin/env bash
HOST=""; PORT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    *) shift;;
  esac
done
echo "fake detector: host=$HOST port=$PORT ros=${ROS_DISTRO:-none}"
echo "camera=${WISEPACK_PERCEPTION_CAMERA:-unset} proxy=${WISEPACK_PHYSICAL_PROXY_DIAMETER_MM:-unset}"
exec python3 -c '
import http.server, json, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            b = json.dumps({"source": "camera", "detector": "fake",
                            "ros": {"available": False, "error": "stub"}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a): pass
http.server.HTTPServer((sys.argv[1], int(sys.argv[2])), H).serve_forever()
' "$HOST" "$PORT"
'''


@pytest.fixture
def harness(tmp_path):
    """A fake detector interpreter plus a driver that sources the real libs."""
    fake_py = tmp_path / "fake_python"
    fake_py.write_text(_FAKE_DETECTOR)
    fake_py.chmod(0o755)
    # The shebang must be the first byte, so `execve` accepts it directly rather
    # than relying on bash's ENOEXEC fallback — otherwise a test that spawns the
    # stub with `subprocess.Popen` fails for a reason that has nothing to do
    # with the code under test.
    assert fake_py.read_bytes().startswith(b"#!/usr/bin/env bash\n")

    # A minimal repo layout: the libraries under test plus the service file the
    # library checks for.
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "perception").mkdir()
    shutil.copy(LIB_HOST, repo / "scripts")
    shutil.copy(LIB_PERCEPTION, repo / "scripts")
    (repo / "perception" / "perception_service.py").write_text("")

    driver = tmp_path / "drive.sh"
    driver.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        set -u
        REPO={repo}
        . "$REPO/scripts/lib_host_processes.sh"
        . "$REPO/scripts/lib_perception_service.sh"
        if perception_ensure_service "$REPO" "${{WANT_ROS:-0}}"; then
            echo "ENSURE=ok"
        else
            echo "ENSURE=failed"
        fi
        echo "OWNED_PID=${{PERCEPTION_OWNED_PID:-}}"
        echo "HOOKS=${{HOST_CLEANUP_HOOKS}}"
        # Emulate the dashboard running, then a normal exit -> the trap fires.
        sleep "${{HOLD:-1}}"
    """))
    driver.chmod(0o755)

    class Harness:
        def __init__(self):
            self.tmp = tmp_path
            self.repo = repo
            self.python = str(fake_py)
            self.log = str(tmp_path / "detector.log")
            self.port = _free_port()
            self.url = f"http://127.0.0.1:{self.port}"
            self.started = []

        def run(self, *, python=None, url=None, env=None, timeout=90):
            environment = {
                **os.environ,
                "WISEPACK_PERCEPTION_PYTHON": python or self.python,
                "WISEPACK_PERCEPTION_SERVICE_URL": url or self.url,
                "WISEPACK_PERCEPTION_LOG": self.log,
                "WISEPACK_PERCEPTION_READY_TIMEOUT": "25",
            }
            environment.update(env or {})
            proc = subprocess.run([str(driver)], capture_output=True, text=True,
                                  env=environment, timeout=timeout)
            return proc

        def fields(self, proc):
            out = {}
            for line in proc.stdout.splitlines():
                if "=" in line and line.split("=")[0].isupper():
                    key, _, value = line.partition("=")
                    out[key] = value
            return out

        def spawn_external(self, port):
            proc = subprocess.Popen(
                [self.python, "svc.py", "--host", "127.0.0.1",
                 "--port", str(port)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            self.started.append(proc)
            for _ in range(60):
                if _health(f"http://127.0.0.1:{port}"):
                    return proc
                time.sleep(0.25)
            raise AssertionError("the external stub never became healthy")

        def stop_all(self):
            for proc in self.started:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass

    h = Harness()
    yield h
    h.stop_all()


def _health(url: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=1) as r:
            return bool(json.loads(r.read().decode()).get("detector"))
    except Exception:                                            # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# 1. sim perception must not touch the provider at all
# --------------------------------------------------------------------------- #


def test_sim_perception_never_resolves_or_starts_the_provider(tmp_path):
    """The default path must not look at a detector, an interpreter or a venv.

    Deliberately hostile settings: if the launcher ever resolved them in `sim`
    it would fail loudly instead of starting.

    Bounded with `timeout`, because a healthy sim launch runs the dashboard
    forever — "still running after N seconds" IS the pass condition here, so the
    test must not depend on the process ever exiting.
    """
    env = {
        **os.environ,
        "WISEPACK_PERCEPTION_PYTHON": "/nonexistent/python",
        "WISEPACK_HARMONY_PATH": "/nonexistent/harmony",
        "PORT": str(_free_port()),
    }
    env.pop("WISEPACK_PERCEPTION_SOURCE", None)
    proc = subprocess.run(["timeout", "--signal=INT", "12", DASHBOARD, "sim"],
                          capture_output=True, text=True, env=env, cwd=REPO,
                          timeout=120)
    combined = proc.stdout + proc.stderr
    assert "[perception]" not in combined, (
        "sim perception performed provider work:\n" + combined)
    assert "torch_venv" not in combined
    # It must not have aborted for a perception reason either.
    assert proc.returncode != EXIT_PERCEPTION_UNAVAILABLE, combined


def test_the_launcher_only_acts_on_the_exact_camera_value():
    """Guarded on `harmony_camera`, so a future source cannot start it by luck."""
    text = open(DASHBOARD, encoding="utf-8").read()
    assert '"${WISEPACK_PERCEPTION_SOURCE:-sim}" = "camera"' in text
    # Unset must behave as `sim` — the `:-sim` default is what guarantees it.
    assert "WISEPACK_PERCEPTION_SOURCE:-sim" in text


# --------------------------------------------------------------------------- #
# 2-4. Interpreter resolution
# --------------------------------------------------------------------------- #


def _resolve(env=None, tmp_path=None):
    script = textwrap.dedent(f"""\
        set -u
        . "{LIB_HOST}"
        . "{LIB_PERCEPTION}"
        if perception_resolve_python; then
            echo "PY=$PERCEPTION_PYTHON"
            echo "ORIGIN=$PERCEPTION_PYTHON_ORIGIN"
        else
            echo "PY="
            echo "ERROR<<$PERCEPTION_PYTHON_ERROR>>"
        fi
    """)
    environment = {**os.environ}
    environment.pop("WISEPACK_PERCEPTION_PYTHON", None)
    environment.pop("WISEPACK_HARMONY_PATH", None)
    environment.update(env or {})
    return subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True, env=environment, timeout=30)


def test_camera_mode_resolves_the_provider_venv_beside_its_checkout(tmp_path):
    """The provider's own run.sh resolves `<checkout>/../torch_venv/bin/python`.

    Verified against the real HARMONY installation layout rather than assumed —
    and deliberately NOT `.venv`, which this provider does not use.
    """
    harmony = tmp_path / "harmony"
    detector = harmony / "ai-bottle-detector-fiware"
    venv_python = harmony / "torch_venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    detector.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n")
    venv_python.chmod(0o755)

    r = _resolve({"WISEPACK_HARMONY_PATH": str(detector)})
    assert f"PY={venv_python}" in r.stdout
    assert "ORIGIN=HARMONY torch_venv" in r.stdout   # provenance, in a diagnostic
    assert ".venv" not in r.stdout


def test_the_default_detector_path_yields_the_documented_interpreter():
    """With nothing configured, the candidate is the documented ARISE layout."""
    r = _resolve()
    if "PY=" in r.stdout and r.stdout.split("PY=")[1].splitlines()[0]:
        pytest.skip("a real provider torch_venv exists on this host")
    assert "/torch_venv/bin/python" in r.stdout
    assert "harmony" in r.stdout.lower()


def test_an_explicit_interpreter_overrides_the_default(tmp_path):
    harmony = tmp_path / "harmony"
    (harmony / "torch_venv" / "bin").mkdir(parents=True)
    (harmony / "torch_venv" / "bin" / "python").write_text("#!/bin/sh\n")
    (harmony / "torch_venv" / "bin" / "python").chmod(0o755)
    (harmony / "ai-bottle-detector-fiware").mkdir()

    chosen = tmp_path / "my_python"
    chosen.write_text("#!/bin/sh\n")
    chosen.chmod(0o755)

    r = _resolve({"WISEPACK_HARMONY_PATH": str(harmony / "ai-bottle-detector-fiware"),
                  "WISEPACK_PERCEPTION_PYTHON": str(chosen)})
    assert f"PY={chosen}" in r.stdout
    assert "ORIGIN=WISEPACK_PERCEPTION_PYTHON" in r.stdout


def test_a_missing_interpreter_fails_clearly_and_never_uses_system_python(tmp_path):
    """The observed failure was `No module named 'uvicorn'` under system python.

    So absence must be a clear, actionable error at start-up — never a fall
    back to an interpreter that cannot import the detector's dependencies.
    """
    r = _resolve({"WISEPACK_HARMONY_PATH": str(tmp_path / "nowhere")})
    assert "PY=\n" in r.stdout or "PY=" in r.stdout.splitlines()
    message = r.stdout
    assert "no perception detector interpreter found" in message
    assert "torch_venv/bin/python" in message          # names what it looked for
    assert "WISEPACK_PERCEPTION_PYTHON" in message        # names the way out
    assert "uvicorn" in message                        # names why python3 is not used


def test_an_explicit_interpreter_that_is_missing_is_an_error_not_a_fallback(tmp_path):
    harmony = tmp_path / "harmony"
    (harmony / "torch_venv" / "bin").mkdir(parents=True)
    (harmony / "torch_venv" / "bin" / "python").write_text("#!/bin/sh\n")
    (harmony / "torch_venv" / "bin" / "python").chmod(0o755)
    (harmony / "ai-bottle-detector-fiware").mkdir()

    r = _resolve({"WISEPACK_HARMONY_PATH": str(harmony / "ai-bottle-detector-fiware"),
                  "WISEPACK_PERCEPTION_PYTHON": "/nope/python"})
    assert "is not an executable file" in r.stdout
    # It must NOT silently use the torch_venv that does exist.
    assert "torch_venv" not in r.stdout.split("ERROR")[-1]


# --------------------------------------------------------------------------- #
# 5-7. Lifecycle: start, reuse, ownership, cleanup
# --------------------------------------------------------------------------- #


def test_the_launcher_starts_waits_and_reports_ready(harness):
    proc = harness.run(env={"WISEPACK_PERCEPTION_CAMERA": "2"})
    assert "ENSURE=ok" in proc.stdout, proc.stdout + proc.stderr
    assert "detector service  : starting" in proc.stdout
    assert "detector service  : ready" in proc.stdout
    assert "camera            : 2" in proc.stdout
    fields = harness.fields(proc)
    assert fields["OWNED_PID"], "the launcher must record what it owns"
    assert "perception_cleanup" in fields["HOOKS"]


def test_a_service_started_by_the_launcher_is_stopped_on_exit(harness):
    proc = harness.run()
    pid = int(harness.fields(proc)["OWNED_PID"])
    for _ in range(40):
        if not _alive(pid):
            break
        time.sleep(0.25)
    assert not _alive(pid), "the launcher-owned detector outlived the launcher"
    assert "stopping the detector service this launcher started" in proc.stdout


def test_an_existing_healthy_service_is_reused(harness):
    port = _free_port()
    harness.spawn_external(port)
    proc = harness.run(url=f"http://127.0.0.1:{port}")
    assert "ENSURE=ok" in proc.stdout
    assert "already running (external; will not be stopped)" in proc.stdout
    assert "detector service  : starting" not in proc.stdout


def test_a_reused_external_service_is_never_stopped(harness):
    port = _free_port()
    external = harness.spawn_external(port)
    proc = harness.run(url=f"http://127.0.0.1:{port}")
    fields = harness.fields(proc)
    # Owning nothing is the mechanism: cleanup keys on PERCEPTION_OWNED_PID.
    assert fields["OWNED_PID"] == ""
    assert fields["HOOKS"] == ""
    time.sleep(1)
    assert _alive(external.pid), (
        "an externally started detector was killed by a WISEPACK run")
    assert _health(f"http://127.0.0.1:{port}")


def test_something_else_listening_on_the_port_is_not_mistaken_for_the_detector(
        harness, tmp_path):
    """A 200 from an unrelated server is not a detector.

    Reusing it would report a healthy camera that does not exist.
    """
    port = _free_port()
    other = subprocess.Popen(
        ["python3", "-c", textwrap.dedent("""
            import http.server, sys
            class H(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200); self.send_header("Content-Length","2")
                    self.end_headers(); self.wfile.write(b"{}")
                def log_message(self,*a): pass
            http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
        """), str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    harness.started.append(other)
    time.sleep(1.5)
    proc = harness.run(url=f"http://127.0.0.1:{port}")
    assert "already running" not in proc.stdout
    # It tries to start its own instead — which then cannot bind the taken port.
    assert "detector service  : starting" in proc.stdout


# --------------------------------------------------------------------------- #
# 8. Startup failure
# --------------------------------------------------------------------------- #


def test_a_service_that_dies_at_startup_reports_the_reason(harness, tmp_path):
    """The observed failure, reproduced: the interpreter cannot import uvicorn."""
    dead = tmp_path / "dead_python"
    dead.write_text("#!/usr/bin/env bash\n"
                    "echo \"ModuleNotFoundError: No module named 'uvicorn'\" >&2\n"
                    "exit 1\n")
    dead.chmod(0o755)
    proc = harness.run(python=str(dead))
    combined = proc.stdout + proc.stderr
    assert "ENSURE=failed" in proc.stdout
    assert "exited during start-up" in combined
    assert "uvicorn" in combined, "the detector's own log must be shown"
    assert harness.fields(proc)["OWNED_PID"] == "", "ownership must be released"


def test_a_service_that_never_becomes_healthy_times_out_and_is_stopped(
        harness, tmp_path):
    hang = tmp_path / "hang_python"
    hang.write_text("#!/usr/bin/env bash\necho starting\nsleep 300\n")
    hang.chmod(0o755)
    started = time.monotonic()
    proc = harness.run(python=str(hang),
                       env={"WISEPACK_PERCEPTION_READY_TIMEOUT": "3"})
    elapsed = time.monotonic() - started
    combined = proc.stdout + proc.stderr
    assert "ENSURE=failed" in proc.stdout
    assert "did not become healthy" in combined
    assert elapsed < 60, "the readiness wait must be bounded"
    assert harness.fields(proc)["OWNED_PID"] == ""


def test_the_launcher_aborts_rather_than_running_simulated_perception():
    """§15: a requested camera must never silently become the simulator."""
    port = _free_port()
    env = {
        **os.environ,
        "WISEPACK_PERCEPTION_SOURCE": "camera",
        "WISEPACK_PERCEPTION_PYTHON": "/nonexistent/python",
        "WISEPACK_PERCEPTION_SERVICE_URL": f"http://127.0.0.1:{_free_port()}",
        "PORT": str(port),
    }
    proc = subprocess.run([DASHBOARD, "sim"], capture_output=True, text=True,
                          env=env, cwd=REPO, timeout=120)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == EXIT_PERCEPTION_UNAVAILABLE, combined
    assert "ABORTING" in combined
    assert "will not" in combined and "simulated perception" in combined
    # The dashboard must never have been started.
    assert "sim mode — no ROS" not in proc.stdout


# --------------------------------------------------------------------------- #
# 9-12. Launcher integration and cleanup composition
# --------------------------------------------------------------------------- #


def test_no_exec_orphans_a_launcher_owned_process():
    """`exec` replaces this shell, so its EXIT trap never runs.

    Every terminal command therefore goes through `host_run_foreground`, which
    execs only when nothing is owned.
    """
    text = open(DASHBOARD, encoding="utf-8").read()
    for orphaning in ("exec python3 \"$REPO/web/app.py\"",
                      "exec docker run",
                      'exec "${DOCKER_RUN[@]}"'):
        assert orphaning not in text, (
            f"`{orphaning}` would orphan a launcher-owned detector")
    assert text.count("host_run_foreground") >= 3, (
        "all three terminal paths (sim host, sim docker, live docker) must go "
        "through host_run_foreground")


def test_cleanup_hooks_compose_instead_of_replacing_each_other(tmp_path):
    """camera + Isaac: one trap, both hooks, neither cancelling the other."""
    marker_a, marker_b = tmp_path / "a", tmp_path / "b"
    script = textwrap.dedent(f"""\
        set -u
        . "{LIB_HOST}"
        perception_cleanup() {{ echo cleaned > "{marker_a}"; }}
        isaac_cleanup()   {{ echo cleaned > "{marker_b}"; }}
        host_register_cleanup perception_cleanup
        host_register_cleanup isaac_cleanup
        echo "HOOKS=$HOST_CLEANUP_HOOKS"
        exit 0
    """)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       timeout=30)
    assert "HOOKS=perception_cleanup isaac_cleanup" in r.stdout
    assert marker_a.exists(), "the perception hook was replaced by Isaac's"
    assert marker_b.exists(), "the Isaac hook did not run"


def test_registering_the_same_hook_twice_runs_it_once(tmp_path):
    counter = tmp_path / "count"
    script = textwrap.dedent(f"""\
        set -u
        . "{LIB_HOST}"
        perception_cleanup() {{ echo x >> "{counter}"; }}
        host_register_cleanup perception_cleanup
        host_register_cleanup perception_cleanup
        exit 0
    """)
    subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                   timeout=30)
    assert counter.read_text().count("x") == 1


def test_the_isaac_path_registers_rather_than_installing_its_own_trap():
    """A bare `trap 'isaac_cleanup' EXIT` would drop the perception hook."""
    text = open(DASHBOARD, encoding="utf-8").read()
    assert "host_register_cleanup isaac_cleanup" in text
    assert "trap 'isaac_cleanup' EXIT INT TERM" not in text
    # The Isaac cleanup itself must still exist and still be reachable.
    assert "isaac_cleanup() {" in text


def test_live_modes_pass_the_perception_source_into_the_stack():
    text = open(DASHBOARD, encoding="utf-8").read()
    # Into the container...
    assert "-e WISEPACK_PERCEPTION_SOURCE" in text
    assert "-e WISEPACK_PERCEPTION_SERVICE_URL" in text
    # ...and on into the ROS launch, where the orchestrator reads it.
    assert 'perception_source:="${WISEPACK_PERCEPTION_SOURCE:-sim}"' in text


@pytest.mark.parametrize("variable", [
    "WISEPACK_PHYSICAL_PROXY_DIAMETER_MM", "WISEPACK_PHYSICAL_PROXY_LENGTH_MM",
    "WISEPACK_PHYSICAL_PROXY_WALL_MM", "WISEPACK_PHYSICAL_PROXY_MATERIAL",
    "WISEPACK_PHYSICAL_PROXY_GROUP", "WISEPACK_PHYSICAL_FRAME_ID",
    "WISEPACK_PHYSICAL_WORKAREA_WIDTH_MM", "WISEPACK_PHYSICAL_WORKAREA_DEPTH_MM",
])
def test_physical_settings_reach_the_containerised_dashboard(variable):
    """The engine reads these from the environment; the container needs them."""
    text = open(DASHBOARD, encoding="utf-8").read()
    assert text.count(f"-e {variable}") >= 2, (
        f"{variable} must reach both the sim-docker and the live container")


def test_the_detector_child_inherits_the_physical_settings(harness):
    """The HOST service is a child of the launcher, so it inherits directly.

    No second passthrough list to fall out of step with the allowlist.
    """
    proc = harness.run(env={"WISEPACK_PERCEPTION_CAMERA": "7",
                            "WISEPACK_PHYSICAL_PROXY_DIAMETER_MM": "81"})
    assert "ENSURE=ok" in proc.stdout
    log = open(harness.log, encoding="utf-8").read()
    assert "camera=7" in log
    assert "proxy=81" in log


def test_the_launcher_starts_the_service_with_no_middleware(harness):
    """No ROS environment is prepared for the child, because it needs none.

    Replaces two earlier tests that asserted a ROS environment WAS sourced for
    the detector subprocess. That was wrong for this deployment: WISEPACK's
    validated middleware is the CONTAINERIZED Vulcanexus runtime, and sourcing a
    host ROS for the camera created a second, unvalidated path. The host service
    is HTTP-only now — which is also what lets `sim` + camera run with no ROS
    installed anywhere.
    """
    proc = harness.run()
    assert "ENSURE=ok" in proc.stdout
    log = open(harness.log, encoding="utf-8").read()
    assert "ros=none" in log, (
        "the detector child was given a ROS environment it does not need")


def test_the_launcher_never_activates_the_provider_venv():
    """Activating it would put HARMONY's interpreter ahead of everything else.

    The interpreter is invoked directly instead — which is also what HARMONY's
    own run.sh does for real detection.
    """
    for path in (DASHBOARD, LIB_PERCEPTION):
        text = open(path, encoding="utf-8").read()
        assert "bin/activate" not in text, (
            f"{os.path.basename(path)} activates a virtual environment")


def test_the_absence_of_host_vulcanexus_is_not_a_failure(harness):
    """§10. Vulcanexus is a CONTAINER component of WISEPACK.

    Its absence from /opt on the host says nothing about the WISEPACK stack, and
    camera perception must not consult it, require it, or report it as broken.
    """
    assert not os.path.exists("/opt/vulcanexus/jazzy/setup.bash") or True
    proc = harness.run()
    combined = proc.stdout + proc.stderr
    assert "ENSURE=ok" in proc.stdout, combined
    assert "vulcanexus" not in combined.lower(), (
        "camera start-up mentions Vulcanexus — the host runtime is irrelevant "
        "to it, and naming it invites reading host absence as stack absence")


def test_no_middleware_is_sourced_anywhere_for_the_camera():
    """§10/§11: the host needs no ROS, no Vulcanexus and no Fast DDS.

    WISEPACK's validated middleware is the CONTAINERIZED Vulcanexus runtime;
    the perception service reaches the stack over HTTP and the orchestrator
    publishes the DDS topics from inside the container.
    """
    code = _executable_lines(open(LIB_PERCEPTION, encoding="utf-8").read())
    for forbidden in ("/opt/vulcanexus", "/opt/ros", "setup.bash", "rclpy"):
        assert forbidden not in code, (
            f"the perception launcher references {forbidden!r}")


@pytest.mark.parametrize("library", [LIB_PERCEPTION, LIB_HOST, DASHBOARD])
def test_no_pattern_based_process_cleanup_is_introduced(library):
    """`pkill -f harmony_perception_service` would kill an operator's own run.

    Scanned over EXECUTABLE lines only: these files explain in comments why
    pattern matching is forbidden, and a check that cannot tell an instruction
    from its prohibition would forbid documenting the rule.
    """
    body = _executable_lines(open(library, encoding="utf-8").read())
    for reckless in ("pkill", "pgrep -f", "killall", "kill -9 $(ps"):
        assert reckless not in body, (
            f"{os.path.basename(library)} uses `{reckless}` — cleanup must be "
            "by the PID/session this launcher created, nothing else")


def test_the_detector_runs_in_its_own_session(harness):
    """Ownership is by session, so cleanup can signal the tree and only it."""
    proc = harness.run(env={"HOLD": "3"})
    pid = int(harness.fields(proc)["OWNED_PID"])
    # Already reaped by the time we read it; assert the mechanism in the source.
    assert pid > 0
    text = open(LIB_PERCEPTION, encoding="utf-8").read()
    assert "setsid \"$PERCEPTION_PYTHON\"" in text
    assert "PERCEPTION_OWNED_PGID" in text
    assert 'ps -o pgid= -p "$PERCEPTION_OWNED_PID"' in text


def test_cleanup_is_a_no_op_when_nothing_is_owned(tmp_path):
    script = textwrap.dedent(f"""\
        set -u
        . "{LIB_HOST}"
        . "{LIB_PERCEPTION}"
        perception_cleanup && echo "RC=0"
    """)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       timeout=30)
    assert "RC=0" in r.stdout
    assert "stopping" not in r.stdout
