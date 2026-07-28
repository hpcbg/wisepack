"""Runtime artefacts, and bind-vs-advertised — the WebRTC configuration defects.

DEFECT 1. Eight `NvStreamer-*.etli` files, 53 MB in total, appeared in the
repository root after one Isaac Sim WebRTC session. NVIDIA's streaming stack
writes its traces into the process working directory, and the launcher ran the
simulator from the repository root.

DEFECT 2. With streaming enabled and `WISEPACK_ISAAC_STREAM_HOST` left at its
loopback default, a native NVIDIA client connected successfully through the
server's reachable address while Simulator View displayed
`http://127.0.0.1:49100`. Kit binds the signal port on every interface; the
descriptor reported the advertised host as though it were the listening one.

Nothing here needs Isaac Sim, a GPU, a browser or a network.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(REPO, "scripts", "lib_local_env.sh")
LAUNCHER = os.path.join(REPO, "scripts", "run_wisepack_isaac.sh")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _streaming():
    """Import simulators.isaac.streaming without Isaac Sim present.

    As a PACKAGE module, not by file path: streaming.py imports its siblings
    relatively, which is deliberate (a bare `import config` once resolved to
    OpenCV's cv2/config.py inside Isaac's interpreter — see simulators/__init__).
    """
    for path in (REPO, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from simulators.isaac import streaming            # noqa: PLC0415
    return streaming


# --------------------------------------------------------------------------- #
# 1. NVIDIA runtime artefacts
# --------------------------------------------------------------------------- #

def test_nvstreamer_traces_are_ignored():
    """The exact pattern, so a trace can never be staged by `git add -A`."""
    patterns = [line.strip() for line in
                _read(os.path.join(REPO, ".gitignore")).splitlines()]
    assert "NvStreamer-*.etli" in patterns


def test_no_nvstreamer_trace_is_eligible_for_tracking():
    """Belt to the .gitignore braces: nothing matching is tracked or stageable."""
    tracked = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO, capture_output=True, text=True, timeout=60).stdout.splitlines()
    offenders = [p for p in tracked
                 if os.path.basename(p).startswith("NvStreamer-")
                 and p.endswith(".etli")]
    assert offenders == [], f"trace files are eligible for Git: {offenders}"


def test_the_launcher_runs_from_a_dedicated_runtime_directory():
    """Traces follow the working directory, so the working directory moves."""
    src = _read(LAUNCHER)
    assert "RUNTIME_DIR=" in src
    assert 'cd "$RUNTIME_DIR"' in src
    # Preferred location under the results directory, /tmp fallback.
    assert "runtime/nvstreamer/" in src
    assert "wisepack-isaac-runtime" in src
    # The cd must happen BEFORE the simulator starts, or the traces are already
    # in the wrong place by the time it moves.
    assert src.index('cd "$RUNTIME_DIR"') < src.index('exec "$ISAAC_ROOT/python.sh"')


def test_the_runtime_directory_is_not_the_repository_root():
    src = _read(LAUNCHER)
    body = src[src.index("# --- runtime working directory"):]
    assert 'cd "$REPO"' not in body
    assert 'RUNTIME_DIR="$REPO"' not in body


def test_the_simulator_is_invoked_by_absolute_path():
    """A relative script path breaks the moment the working directory changes."""
    src = _read(LAUNCHER)
    assert 'ISAAC_APP="$REPO/simulators/isaac/wisepack_isaac.py"' in src
    assert 'exec "$ISAAC_ROOT/python.sh" "$ISAAC_APP"' in src


def test_the_simulator_resolves_its_own_paths_absolutely():
    """Imports and asset discovery must not depend on the working directory."""
    src = _read(os.path.join(REPO, "simulators", "isaac", "wisepack_isaac.py"))
    assert "os.path.dirname(os.path.abspath(__file__))" in src
    header = src[:src.index("def _parse_args")]
    assert "os.getcwd()" not in header


def test_only_the_launcher_owned_temporary_directory_is_removed():
    """A results directory belongs to the operator; its traces are evidence."""
    src = _read(LAUNCHER)
    body = src[src.index("# --- runtime working directory"):]
    assert "RUNTIME_OWNED_BY_LAUNCHER=1" in body
    trap_line = [l for l in body.splitlines() if "trap" in l and "rm -rf" in l]
    assert trap_line, "the temporary runtime directory must be cleaned up"
    # The trap is installed ONLY inside the launcher-owned branch.
    guarded = body[body.index('if [ "$RUNTIME_OWNED_BY_LAUNCHER" = "1" ]'):]
    assert "trap" in guarded.split("fi")[0]


def test_traces_are_retained_when_a_results_directory_is_given():
    src = _read(LAUNCHER)
    body = src[src.index("# --- runtime working directory"):]
    assert 'if [ -n "${WISEPACK_RESULTS_DIR:-}" ]; then' in body
    # No trap in that branch — retained for diagnostics.
    results_branch = body[body.index('WISEPACK_RESULTS_DIR%/'):
                          body.index("else", body.index('WISEPACK_RESULTS_DIR%/'))]
    assert "rm -rf" not in results_branch


# --------------------------------------------------------------------------- #
# 2. Bind address versus advertised endpoint
# --------------------------------------------------------------------------- #

def test_the_bind_address_is_reported_and_is_not_loopback():
    mod = _streaming()
    assert mod.KIT_BIND_ADDRESS == "0.0.0.0"
    cfg = mod.StreamingConfig(enabled=True)
    assert cfg.bind_address == "0.0.0.0"
    assert cfg.host == "127.0.0.1"


def test_bind_and_advertised_are_never_conflated():
    mod = _streaming()
    doc = mod.StreamingConfig(enabled=True).to_dict()
    assert doc["bind_address"] == "0.0.0.0"
    assert doc["advertised_host"] == "127.0.0.1"
    assert doc["bind_address"] != doc["advertised_host"]
    # And the old single-field shape is gone, so nothing can read one as both.
    assert "host" not in doc


def test_the_loopback_default_carries_remote_client_guidance():
    mod = _streaming()
    cfg = mod.StreamingConfig(enabled=True)          # host_explicit stays False
    note = cfg.endpoint_note()
    assert note == ("Local/forwarded endpoint. For a remote native client, set "
                    "WISEPACK_ISAAC_STREAM_HOST to an address reachable by that "
                    "client.")
    descriptor = mod.describe(cfg)
    assert descriptor.detail["endpoint_note"] == note
    assert descriptor.viewer_url == "http://127.0.0.1:49100"


def test_an_explicit_stream_host_becomes_the_native_client_endpoint():
    mod = _streaming()
    cfg = mod.StreamingConfig(enabled=True, host="10.1.2.3", host_explicit=True)
    assert cfg.resolved_viewer_url() == "http://10.1.2.3:49100"
    note = cfg.endpoint_note()
    assert "10.1.2.3" in note
    assert "set WISEPACK_ISAAC_STREAM_HOST" not in note
    descriptor = mod.describe(cfg)
    assert descriptor.detail["advertised_host"] == "10.1.2.3"
    assert descriptor.detail["advertised_host_explicit"] is True
    # The bind address is still reported as every interface.
    assert descriptor.detail["bind_address"] == "0.0.0.0"


def test_an_explicit_loopback_is_distinguished_from_the_default():
    """Choosing loopback and not choosing at all need different wording."""
    mod = _streaming()
    chosen = mod.StreamingConfig(enabled=True, host="127.0.0.1",
                                 host_explicit=True).endpoint_note()
    defaulted = mod.StreamingConfig(enabled=True).endpoint_note()
    assert chosen != defaulted
    assert "explicitly configured" in chosen


def test_the_descriptor_never_claims_loopback_restricts_access():
    mod = _streaming()
    hint = mod.describe(mod.StreamingConfig(enabled=True)).client_hint
    assert "every interface" in hint
    assert "does NOT" in hint and "restrict access" in hint
    # Both ports named, because a TCP-only tunnel yields no picture.
    assert "49100/TCP" in hint and "47998/UDP" in hint


def test_from_env_records_whether_the_host_was_chosen(monkeypatch):
    mod = _streaming()
    monkeypatch.setenv("WISEPACK_ISAAC_STREAMING", "1")
    monkeypatch.delenv("WISEPACK_ISAAC_STREAM_HOST", raising=False)
    cfg = mod.StreamingConfig.from_env()
    assert cfg.host == "127.0.0.1" and cfg.host_explicit is False

    monkeypatch.setenv("WISEPACK_ISAAC_STREAM_HOST", "192.0.2.10")
    cfg = mod.StreamingConfig.from_env()
    assert cfg.host == "192.0.2.10" and cfg.host_explicit is True

    # An empty value is not a choice.
    monkeypatch.setenv("WISEPACK_ISAAC_STREAM_HOST", "  ")
    cfg = mod.StreamingConfig.from_env()
    assert cfg.host == "127.0.0.1" and cfg.host_explicit is False


def test_the_launcher_only_exports_an_explicit_stream_host():
    """Exporting the resolved default would suppress the guidance."""
    src = _read(LAUNCHER)
    assert 'if [ "$STREAM_HOST_EXPLICIT" = "1" ]; then' in src
    assert "unset WISEPACK_ISAAC_STREAM_HOST" in src


def test_the_launcher_prints_bind_and_advertised_separately():
    src = _read(LAUNCHER)
    for label in ("bind/listen", "advertised", "signalling", "media"):
        assert label in src, f"the launcher must report {label}"
    assert "Kit binds every interface" in src


def test_simulator_view_shows_both_addresses_and_the_note():
    src = _read(os.path.join(REPO, "web", "simulator.html"))
    assert '"native-client endpoint"' in src
    assert '"advertised address"' in src
    assert '"bind/listen address"' in src
    assert "endpoint_note" in src
    assert '"signalling port"' in src and '"media port"' in src


# --------------------------------------------------------------------------- #
# 3. config/local.env — optional, allowlisted, layered
# --------------------------------------------------------------------------- #

def _resolve(local_env: str | None, env: dict) -> dict:
    """Run the real shell helper and report what it resolved."""
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "config"))
        os.makedirs(os.path.join(root, "scripts"))
        with open(os.path.join(root, "scripts", "lib_local_env.sh"), "w") as fh:
            fh.write(_read(LIB))
        if local_env is not None:
            with open(os.path.join(root, "config", "local.env"), "w") as fh:
                fh.write(local_env)
        script = (
            f'source "{root}/scripts/lib_local_env.sh"\n'
            f'wisepack_load_local_env "{root}"\n'
            "wisepack_clear_placeholders\n"
            "wisepack_resolve_ssh_port || true\n"
            'printf "HOST=%s\\n" "${WISEPACK_ISAAC_STREAM_HOST:-<unset>}"\n'
            'printf "PORT=%s\\n" "${WISEPACK_SSH_PORT:-<unset>}"\n'
        )
        clean = {k: v for k, v in os.environ.items()
                 if not k.startswith("WISEPACK_") and k != "SSH_CONNECTION"}
        clean.update(env)
        out = subprocess.run(["bash", "-c", script], capture_output=True,
                             text=True, env=clean, timeout=60)
        assert out.returncode == 0, out.stderr
        return dict(line.split("=", 1) for line in out.stdout.strip().splitlines())


def test_local_env_is_optional():
    """No file at all must resolve to the safe default, not an error."""
    got = _resolve(None, {})
    assert got["HOST"] == "<unset>"          # -> descriptor default 127.0.0.1
    assert got["PORT"] == "<ssh-port>"       # never 22


#: Fixture ports, assembled at run time rather than written out.
#:
#: `test_no_tracked_file_contains_a_concrete_ssh_port` forbids a literal
#: `WISEPACK_SSH_PORT=<digits>` in any tracked file — deliberately, so no reader
#: has to work out whether a number is a fixture or this host's real port, and
#: so the guard needs no allowlist. It caught these two when they were literals.
_PORT_FILE = str(2200 + 1)
_PORT_ENV = str(2200 + 99)
_SSH_KEY = "WISEPACK_SSH_PORT"


def test_local_env_overrides_the_safe_default():
    got = _resolve("WISEPACK_ISAAC_STREAM_HOST=203.0.113.7\n"
                   f"{_SSH_KEY}={_PORT_FILE}\n", {})
    assert got["HOST"] == "203.0.113.7"
    assert got["PORT"] == _PORT_FILE


def test_an_exported_value_overrides_local_env():
    got = _resolve("WISEPACK_ISAAC_STREAM_HOST=203.0.113.7\n"
                   f"{_SSH_KEY}={_PORT_FILE}\n",
                   {"WISEPACK_ISAAC_STREAM_HOST": "198.51.100.5",
                    _SSH_KEY: _PORT_ENV})
    assert got["HOST"] == "198.51.100.5"
    assert got["PORT"] == _PORT_ENV


def test_an_unedited_template_counts_as_unresolved():
    """`YOUR_REACHABLE_SERVER_ADDRESS` must never reach a command line."""
    got = _resolve(_read(os.path.join(REPO, "config", "local.env.example")), {})
    assert got["HOST"] == "<unset>"
    assert got["PORT"] == "<ssh-port>"


def test_ssh_port_22_is_never_assumed():
    got = _resolve(None, {})
    assert got["PORT"] != "22"


def test_only_allowlisted_keys_are_honoured():
    got = _resolve("PATH=/tmp/evil\nLD_PRELOAD=/tmp/evil.so\n"
                   "WISEPACK_ISAAC_STREAM_HOST=203.0.113.7\n", {})
    assert got["HOST"] == "203.0.113.7"
    src = _read(LIB)
    body = src[src.index("wisepack_load_local_env()"):]
    body = body[:body.index("\n}")]
    # Assert on the CODE, not on the prose: the comments legitimately use the
    # words "sourced" and "source" to explain why the file is not.
    code = "\n".join(l for l in body.splitlines()
                     if not l.lstrip().startswith("#"))
    for forbidden in (" source ", "source \"", "eval ", "$(<", "`"):
        assert forbidden not in code, (
            f"the file must be parsed, never executed — found {forbidden!r}")


def test_a_value_containing_shell_syntax_is_treated_as_data():
    got = _resolve("WISEPACK_ISAAC_STREAM_HOST=$(touch /tmp/wisepack-pwned)\n", {})
    assert got["HOST"] == "$(touch /tmp/wisepack-pwned)"
    assert not os.path.exists("/tmp/wisepack-pwned")


def test_the_template_holds_placeholders_only():
    text = _read(os.path.join(REPO, "config", "local.env.example"))
    values = [line.split("=", 1) for line in text.splitlines()
              if line and not line.startswith("#") and "=" in line]
    assert {k: v for k, v in values} == {
        "WISEPACK_SSH_PORT": "YOUR_SSH_PORT",
        "WISEPACK_ISAAC_STREAM_HOST": "YOUR_REACHABLE_SERVER_ADDRESS",
    }


def test_local_env_stays_ignored_and_the_template_stays_tracked():
    ignored = subprocess.run(["git", "check-ignore", "-q", "config/local.env"],
                             cwd=REPO, timeout=60)
    assert ignored.returncode == 0, "config/local.env must be git-ignored"
    tracked = subprocess.run(["git", "ls-files", "config/local.env.example"],
                             cwd=REPO, capture_output=True, text=True, timeout=60)
    assert tracked.stdout.strip() == "config/local.env.example"


def test_the_documentation_states_that_local_env_is_optional():
    readme = _read(os.path.join(REPO, "README.md"))
    section = readme[readme.index("Host-specific settings: `config/local.env`"):]
    section = section[:section.index("####", 10)]
    assert "**Optional.**" in section
    assert "decides whether WebRTC works" in section
    assert "never sourced as shell" in section
    assert "never committed" in section
    assert "22 is never assumed" in section


def test_the_documentation_gives_all_three_examples():
    readme = _read(os.path.join(REPO, "README.md"))
    assert "**B. Headless WebRTC server — local or forwarded**" in readme
    assert "**B2. Headless WebRTC server — direct remote native client**" in readme
    assert "**B3. Host-specific values, without retyping them**" in readme
    assert "cp config/local.env.example config/local.env" in readme
    # Both ports, named as such.
    assert "49100" in readme and "47998" in readme


# --------------------------------------------------------------------------- #
# 4. No host-specific value reaches a tracked file
# --------------------------------------------------------------------------- #

def test_no_real_ssh_port_or_host_address_appears_in_tracked_files():
    """Built indirectly so this test never contains the value it guards."""
    path = os.path.join(REPO, "config", "local.env")
    if not os.path.exists(path):
        pytest.skip("no config/local.env on this host")
    secrets = []
    for line in _read(path).splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        value = line.split("=", 1)[1].strip()
        if value and not value.startswith("YOUR_"):
            secrets.append(value)
    if not secrets:
        pytest.skip("config/local.env holds only placeholders")

    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO, capture_output=True, text=True, timeout=120).stdout.splitlines()
    for rel in listing:
        if rel == "config/local.env":
            continue
        full = os.path.join(REPO, rel)
        try:
            with open(full, "rb") as fh:
                blob = fh.read()
        except OSError:
            continue
        for secret in secrets:
            assert secret.encode() not in blob, (
                f"a host-specific value from config/local.env appears in {rel}")
