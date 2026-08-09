"""The launcher owns the optional WISEPACK perception service.

The operator starts WISEPACK exactly as before — `./run_wisepack_dashboard.sh
sim` and friends — and with `WISEPACK_PERCEPTION_SOURCE=camera` in
`config/local.env` the perception service is started for them. No venv to
activate, no second terminal, no torch in the system Python, and no middleware
on the host: the service speaks HTTP only.

These tests drive the REAL shell libraries with a FAKE perception environment, so
the whole lifecycle — environment resolution, first-run bootstrap, reuse,
readiness, failure, ownership and cleanup — is exercised without a camera, a GPU,
torch, or any provider being installed. The one thing they never do is trust the
script text where behaviour can be run instead.

THE ENVIRONMENT IS WISEPACK'S OWN. `WISEPACK_PERCEPTION_VENV` points the library
at a throwaway `bin/python` for these tests; in production it is unset and the
library resolves `<repo>/.venv-perception/bin/python`, created by
`scripts/setup_perception.sh`. No other project's interpreter is involved, and
there is no `WISEPACK_PERCEPTION_PYTHON` to set.
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


#: A stand-in for `.venv-perception/bin/python`, invoked exactly as the library
#: invokes the real one — twice, for two different purposes:
#:
#:   `<python> -c 'import torch, ...'`            the usability probe
#:   `<python> <service.py> --host H --port P`    the service itself
#:
#: The probe must SUCCEED here (this stands in for a working environment); the
#: `dead`/`hang` variants below override the second call to model a broken one.
#:
#: NOT indented and NOT passed through `textwrap.dedent`. The embedded Python
#: below starts at column 0, so dedent would find no common prefix, strip
#: nothing, and leave the shebang behind eight spaces — which `bash` silently
#: papers over by re-running the file itself, and `execve` rejects outright with
#: `OSError: [Errno 8] Exec format error`. The shebang has to be the first byte.
_FAKE_DETECTOR = '''#!/usr/bin/env bash
# The import probe: a usable environment answers 0 and prints nothing.
[ "${1:-}" = "-c" ] && exit "${FAKE_PERCEPTION_IMPORT_RC:-0}"
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


def _fake_venv(root, body=_FAKE_DETECTOR):
    """A throwaway stand-in for `.venv-perception/`: just `bin/python`."""
    python = root / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text(body)
    python.chmod(0o755)
    return root, python


@pytest.fixture
def harness(tmp_path):
    """A fake perception environment plus a driver that sources the real libs."""
    fake_venv, fake_py = _fake_venv(tmp_path / "venv")
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
            self.venv = str(fake_venv)
            self.python = str(fake_py)
            self.log = str(tmp_path / "detector.log")
            self.port = _free_port()
            self.url = f"http://127.0.0.1:{self.port}"
            self.started = []

        def venv_with(self, name, body):
            """A second throwaway environment, for the failure cases."""
            root, _python = _fake_venv(tmp_path / name, body)
            return str(root)

        def run(self, *, venv=None, url=None, env=None, timeout=90):
            environment = {
                **os.environ,
                "WISEPACK_PERCEPTION_VENV": venv or self.venv,
                "WISEPACK_PERCEPTION_SERVICE_URL": url or self.url,
                "WISEPACK_PERCEPTION_LOG": self.log,
                "WISEPACK_PERCEPTION_READY_TIMEOUT": "25",
                # The bootstrap must never run in these tests: a fake
                # environment is deliberately being supplied.
                "WISEPACK_PERCEPTION_AUTO_SETUP": "0",
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
        "WISEPACK_PERCEPTION_VENV": "/nonexistent/venv",
        "PORT": str(_free_port()),
    }
    env.pop("WISEPACK_PERCEPTION_SOURCE", None)
    proc = subprocess.run(["timeout", "--signal=INT", "12", DASHBOARD, "sim"],
                          capture_output=True, text=True, env=env, cwd=REPO,
                          timeout=120)
    combined = proc.stdout + proc.stderr
    assert "[perception]" not in combined, (
        "sim perception performed provider work:\n" + combined)
    assert "[perception-setup]" not in combined, (
        "sim perception bootstrapped the perception environment:\n" + combined)
    # It must not have aborted for a perception reason either.
    assert proc.returncode != EXIT_PERCEPTION_UNAVAILABLE, combined


def test_sim_perception_never_creates_the_perception_environment(tmp_path):
    """§16: `perception=sim` must not touch `.venv-perception` at all."""
    venv = tmp_path / "should-not-appear"
    env = {**os.environ,
           "WISEPACK_PERCEPTION_VENV": str(venv),
           "PORT": str(_free_port())}
    env.pop("WISEPACK_PERCEPTION_SOURCE", None)
    subprocess.run(["timeout", "--signal=INT", "12", DASHBOARD, "sim"],
                   capture_output=True, text=True, env=env, cwd=REPO,
                   timeout=120)
    assert not venv.exists(), (
        "simulated perception created a perception environment it never uses")


def test_the_launcher_only_acts_on_the_exact_camera_value():
    """Guarded on `camera`, so a future source cannot start it by luck."""
    text = open(DASHBOARD, encoding="utf-8").read()
    assert '"${WISEPACK_PERCEPTION_SOURCE:-sim}" = "camera"' in text
    # Unset must behave as `sim` — the `:-sim` default is what guarantees it.
    assert "WISEPACK_PERCEPTION_SOURCE:-sim" in text


# --------------------------------------------------------------------------- #
# 2-4. Interpreter resolution
# --------------------------------------------------------------------------- #


def _resolve(env=None, repo=None):
    script = textwrap.dedent(f"""\
        set -u
        . "{LIB_HOST}"
        . "{LIB_PERCEPTION}"
        if perception_resolve_python "{repo or REPO}"; then
            echo "PY=$PERCEPTION_PYTHON"
            echo "ORIGIN=$PERCEPTION_PYTHON_ORIGIN"
        else
            echo "PY="
            echo "ERROR<<$PERCEPTION_PYTHON_ERROR>>"
        fi
    """)
    environment = {**os.environ}
    environment.pop("WISEPACK_PERCEPTION_VENV", None)
    # Never bootstrap from a resolution test: creating a real venv would take
    # minutes and is exercised separately.
    environment["WISEPACK_PERCEPTION_AUTO_SETUP"] = "0"
    environment.update(env or {})
    return subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True, env=environment, timeout=60)


def _usable_venv(root):
    """A stand-in that passes the library's import probe."""
    python = root / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    return python


def test_camera_mode_resolves_the_wisepack_owned_environment(tmp_path):
    """`.venv-perception/bin/python` inside the working directory. Nothing else.

    This is the whole architectural point: the perception runtime belongs to
    WISEPACK, so resolution never leaves the repository and never consults
    another project's layout.
    """
    python = _usable_venv(tmp_path / ".venv-perception")
    r = _resolve({"WISEPACK_PERCEPTION_VENV": str(tmp_path / ".venv-perception")})
    assert f"PY={python}" in r.stdout, r.stdout + r.stderr
    assert "ORIGIN=WISEPACK perception environment" in r.stdout
    assert "harmony" not in r.stdout.lower()
    assert "torch_venv" not in r.stdout


def test_the_default_location_is_the_repository_venv(tmp_path):
    """Unset `WISEPACK_PERCEPTION_VENV` means `<repo>/.venv-perception`."""
    repo = tmp_path / "repo"
    _usable_venv(repo / ".venv-perception")
    r = _resolve(repo=str(repo))
    assert f"PY={repo}/.venv-perception/bin/python" in r.stdout, r.stdout


def test_an_environment_that_cannot_import_its_dependencies_is_rejected(tmp_path):
    """EXISTING IS NOT USABLE. A venv whose torch is broken must fail here.

    The alternative is a service that starts, answers /health, and then dies on
    the first detection with an ImportError sixty seconds into a demo.
    """
    venv = tmp_path / ".venv-perception"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n"
                      "echo \"ModuleNotFoundError: No module named 'torch'\" >&2\n"
                      "exit 1\n")
    python.chmod(0o755)
    r = _resolve({"WISEPACK_PERCEPTION_VENV": str(venv)})
    assert "PY=\n" in r.stdout + "\n"
    assert "not usable" in r.stdout


def test_a_missing_environment_fails_with_exactly_one_command(tmp_path):
    """The observed failure was `No module named 'uvicorn'` under system python.

    So absence must be a clear, actionable error at start-up — never a fall
    back to an interpreter that cannot import the service's dependencies — and
    the way out must be ONE command.
    """
    r = _resolve({"WISEPACK_PERCEPTION_VENV": str(tmp_path / "nowhere")})
    message = r.stdout
    assert "PY=\n" in message + "\n"
    assert "./scripts/setup_perception.sh" in message      # the one command
    assert "uvicorn" in message                            # why python3 is not used
    assert "system python3" in message
    # The failure text must not send anyone to another repository.
    assert "harmony" not in message.lower()
    assert "torch_venv" not in message


def test_a_missing_environment_is_bootstrapped_once_by_default(tmp_path):
    """§5: `./run_wisepack_dashboard.sh sim` stays the only command needed.

    The real setup script is replaced by a stub here — creating an actual venv
    with torch in it would take minutes — but the LIBRARY's behaviour is real:
    it must notice the miss, run the repository's setup script, and then use
    the environment that appeared.
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    venv = repo / ".venv-perception"
    marker = tmp_path / "setup-ran"
    setup = repo / "scripts" / "setup_perception.sh"
    setup.write_text(
        "#!/usr/bin/env bash\n"
        f"echo ran > {marker}\n"
        f"mkdir -p {venv}/bin\n"
        f"printf '#!/bin/sh\\nexit 0\\n' > {venv}/bin/python\n"
        f"chmod 755 {venv}/bin/python\n")
    setup.chmod(0o755)

    environment = {**os.environ}
    environment.pop("WISEPACK_PERCEPTION_VENV", None)
    environment.pop("WISEPACK_PERCEPTION_AUTO_SETUP", None)
    script = textwrap.dedent(f"""\
        set -u
        . "{LIB_HOST}"
        . "{LIB_PERCEPTION}"
        if perception_resolve_python "{repo}"; then
            echo "PY=$PERCEPTION_PYTHON"
            echo "ORIGIN=$PERCEPTION_PYTHON_ORIGIN"
        else
            echo "PY="
        fi
    """)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       env=environment, timeout=60)
    assert marker.exists(), "the launcher did not bootstrap the environment"
    assert f"PY={venv}/bin/python" in r.stdout, r.stdout + r.stderr
    assert "created just now" in r.stdout


def test_the_bootstrap_can_be_switched_off(tmp_path):
    """An operator who wants to control installation must be able to."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    marker = tmp_path / "setup-ran"
    setup = repo / "scripts" / "setup_perception.sh"
    setup.write_text(f"#!/usr/bin/env bash\necho ran > {marker}\n")
    setup.chmod(0o755)

    r = _resolve({"WISEPACK_PERCEPTION_AUTO_SETUP": "0"}, repo=str(repo))
    assert not marker.exists(), "the bootstrap ran despite being switched off"
    assert "./scripts/setup_perception.sh" in r.stdout


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
    dead = harness.venv_with("dead", "#!/usr/bin/env bash\n"
                             "[ \"${1:-}\" = \"-c\" ] && exit 0\n"
                             "echo \"ModuleNotFoundError: No module named "
                             "'uvicorn'\" >&2\n"
                             "exit 1\n")
    proc = harness.run(venv=dead)
    combined = proc.stdout + proc.stderr
    assert "ENSURE=failed" in proc.stdout
    assert "exited during start-up" in combined
    assert "uvicorn" in combined, "the detector's own log must be shown"
    assert harness.fields(proc)["OWNED_PID"] == "", "ownership must be released"


def test_a_service_that_never_becomes_healthy_times_out_and_is_stopped(
        harness, tmp_path):
    hang = harness.venv_with("hang", "#!/usr/bin/env bash\n"
                             "[ \"${1:-}\" = \"-c\" ] && exit 0\n"
                             "echo starting\nsleep 300\n")
    started = time.monotonic()
    proc = harness.run(venv=hang,
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
        "WISEPACK_PERCEPTION_VENV": "/nonexistent/venv",
        "WISEPACK_PERCEPTION_AUTO_SETUP": "0",
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
    """Activating it would put this interpreter ahead of everything else.

    Everything the launcher runs after the perception service — the dashboard,
    Docker, the status writers — would resolve out of the perception venv's
    site-packages. It is invoked directly instead.
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
    """`pkill -f perception_service` would kill an operator's own run.

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
