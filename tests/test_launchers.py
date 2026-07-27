"""Regression tests for the launcher/wrapper scripts.

Every test here pins a failure that was REPRODUCED on this machine, not a
hypothetical one. They run without Docker and without ROS: they exercise the
host-side argument handling and assert on the script text for the parts that
only execute inside the container.

The headline bug:

    ./run_vulcanexus_wisepack.sh validate_wisepack_e2e.sh
    _: line 11: /data/jarvis/wisepack/wisepack/: Is a directory
    _: line 11: exec: /data/jarvis/wisepack/wisepack/: cannot execute: Is a directory

`TARGET` was a HOST variable referenced inside a single-quoted `bash -lc` body,
so it was empty in the container and `exec ./"$TARGET"` became `exec ./` — the
repository directory.
"""

from __future__ import annotations

import os
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRAPPER = os.path.join(REPO, "run_vulcanexus_wisepack.sh")
DASHBOARD = os.path.join(REPO, "run_wisepack_dashboard.sh")

# Exit codes the wrapper promises, so a caller can tell a broken invocation from
# a failed validation.
EXIT_USAGE, EXIT_MISSING, EXIT_NOT_EXEC = 2, 3, 4


def run_wrapper(*args, timeout=60):
    return subprocess.run([WRAPPER, *args], capture_output=True, text=True,
                          cwd=REPO, timeout=timeout)


# --------------------------------------------------------------------------- #
# Argument handling — these never reach Docker
# --------------------------------------------------------------------------- #

def test_no_target_is_a_usage_error():
    r = run_wrapper()
    assert r.returncode == EXIT_USAGE
    assert "no validation script named" in r.stderr


def test_absolute_path_is_rejected():
    r = run_wrapper("/etc/passwd")
    assert r.returncode == EXIT_USAGE
    assert "absolute path" in r.stderr


def test_parent_traversal_is_rejected():
    r = run_wrapper("../escape.sh")
    assert r.returncode == EXIT_USAGE
    assert ".." in r.stderr


def test_directory_target_is_rejected():
    """THE regression: `./` used to be executed and reported 'Is a directory'."""
    r = run_wrapper("web")
    assert r.returncode == EXIT_USAGE
    assert "is a directory" in r.stderr.lower()
    assert "cannot execute" not in r.stderr


def test_missing_file_is_distinguished_from_other_failures():
    r = run_wrapper("definitely_not_here.sh")
    assert r.returncode == EXIT_MISSING
    assert "not found" in r.stderr


def test_non_executable_file_is_distinguished():
    r = run_wrapper("README.md")
    assert r.returncode == EXIT_NOT_EXEC
    assert "not executable" in r.stderr
    assert "chmod +x" in r.stderr


# --------------------------------------------------------------------------- #
# The script body — asserted on text, because it only runs in the container
# --------------------------------------------------------------------------- #

def _wrapper_text() -> str:
    with open(WRAPPER, encoding="utf-8") as fh:
        return fh.read()


def test_target_is_passed_as_a_positional_argument():
    """The fix. `$TARGET` must be expanded by the HOST shell, into `_ "$TARGET"`.

    If this reverts, the container executes the repository directory again.
    """
    text = _wrapper_text()
    assert '_ "$TARGET" "$@"' in text, (
        "TARGET must be passed positionally into the container; referencing it "
        "inside the single-quoted bash -lc body leaves it empty there")
    # Comments are stripped first: the wrapper deliberately QUOTES the broken
    # form in a comment to explain the bug, and that documentation must not
    # trip its own regression test.
    code = "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))
    assert 'exec ./"$TARGET"' not in code, (
        "this is the exact broken form: TARGET is a host variable and is unset "
        "inside the container")


def test_container_body_validates_its_target_before_exec():
    text = _wrapper_text()
    body = text[text.index("bash -lc '"):]
    assert 'target="$1"' in body
    assert '! -f "./$target"' in body
    assert '! -x "./$target"' in body
    assert 'exec "./$target" "$@"' in body


def test_container_body_prints_the_resolved_target():
    text = _wrapper_text()
    assert "[host] resolved target" in text
    assert '[container] running ./$target' in text


def test_set_u_comes_after_sourcing_the_ros_environment():
    """Vulcanexus setup.bash dereferences COLCON_TRACE while it is unset.

    `set -u` before the source aborts the container before ROS is even
    available — reproduced while fixing the wrapper.
    """
    text = _wrapper_text()
    body = text[text.index("bash -lc '"):]
    source_at = body.index("source /opt/vulcanexus/jazzy/setup.bash")
    setu_at = body.index("set -u", source_at)
    assert source_at < setu_at, "set -u must not precede the ROS environment"


# --------------------------------------------------------------------------- #
# Always build current source
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("script", [WRAPPER, DASHBOARD])
def test_launchers_build_by_default(script):
    """A launcher must never silently validate a STALE install/."""
    with open(script, encoding="utf-8") as fh:
        text = fh.read()
    assert "colcon build --symlink-install" in text
    assert "WISEPACK_SKIP_BUILD" in text, (
        f"{os.path.basename(script)} must offer an explicit build opt-out "
        "rather than skipping the build implicitly when install/ exists")


@pytest.mark.parametrize("script", [WRAPPER, DASHBOARD])
def test_launchers_do_not_skip_the_build_merely_because_install_exists(script):
    with open(script, encoding="utf-8") as fh:
        text = fh.read()
    # The stale-workspace trap is building ONLY when install/ is absent. A
    # presence check inside the explicit WISEPACK_SKIP_BUILD branch is fine —
    # that one is guarding an opt-out, not silently skipping.
    build_at = text.index("colcon build --symlink-install")
    preceding = text[max(0, build_at - 600):build_at]
    assert "WISEPACK_SKIP_BUILD" in preceding, (
        "the build must be guarded by the explicit opt-out, not by whether "
        "install/setup.bash happens to exist")


# --------------------------------------------------------------------------- #
# Cleanup must be targeted
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("script", [
    WRAPPER, DASHBOARD,
    os.path.join(REPO, "validate_wisepack_e2e.sh"),
    os.path.join(REPO, "validate_fiware_action_log.sh"),
])
def test_cleanup_does_not_kill_unrelated_ros_processes(script):
    """`pkill -f ros2` would kill another project's stack on a shared machine."""
    with open(script, encoding="utf-8") as fh:
        text = fh.read()
    for reckless in ("pkill -f ros2", "pkill -f python3", "pkill ros2",
                     "pkill -9 -f ros"):
        assert reckless not in text, (
            f"{os.path.basename(script)} uses `{reckless}`, which would kill "
            "unrelated ROS processes on a shared machine")


def test_shell_scripts_are_syntactically_valid():
    for name in sorted(os.listdir(REPO)):
        if not name.endswith(".sh"):
            continue
        r = subprocess.run(["bash", "-n", os.path.join(REPO, name)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{name}: {r.stderr}"
    scripts = os.path.join(REPO, "scripts")
    for name in sorted(os.listdir(scripts)):
        if not name.endswith(".sh"):
            continue
        r = subprocess.run(["bash", "-n", os.path.join(scripts, name)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"scripts/{name}: {r.stderr}"


def test_every_executable_script_referenced_by_the_readme_exists():
    """A README command that does not resolve is a failed demo, not a typo."""
    with open(os.path.join(REPO, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    for script in ("run_wisepack_demo.sh", "run_wisepack_dashboard.sh",
                   "run_vulcanexus_wisepack.sh", "validate_wisepack_e2e.sh",
                   "validate_fiware_action_log.sh",
                   "measure_dds_fiware_latency.sh", "generate_demo_artifacts.sh"):
        if script in readme:
            path = os.path.join(REPO, script)
            assert os.path.isfile(path), f"README references missing {script}"
            assert os.access(path, os.X_OK), f"{script} is not executable"


# --------------------------------------------------------------------------- #
# Frontend regressions
# --------------------------------------------------------------------------- #

def test_frontend_never_assigns_a_property_to_an_append_result():
    """Regression: `Element.append()` returns undefined.

        c.append(new Option(k, k)).title = v
        -> Cannot set properties of undefined (setting 'title')

    That single line blanked the entire dashboard on every refresh.
    """
    with open(os.path.join(REPO, "web", "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    import re
    # Strip `//` comment lines: the fix DOCUMENTS the broken form so the next
    # reader understands the bug, and that documentation must not fail its own
    # regression test.
    code = "\n".join(line for line in html.splitlines()
                     if not line.lstrip().startswith("//"))
    bad = re.findall(r"\.append\([^;\n]*\)\s*\.\w+\s*=", code)
    assert not bad, f"assigning to the result of .append(): {bad}"


def test_frontend_isolates_panel_failures():
    """One panel failing must not stop the others updating."""
    with open(os.path.join(REPO, "web", "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    assert "async function panel(" in html, (
        "panels must refresh independently; a single try/catch around the whole "
        "sequence is what produced 'refresh failed' and a blank page")
    assert "PANEL_ERRORS" in html


def test_dashboard_and_orchestrator_advertise_the_same_commands():
    """No dead buttons: the UI list, the contract and the orchestrator agree."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "wp_topics", os.path.join(REPO, "wisepack_ws", "src", "wisepack_bringup",
                                  "wisepack_bringup", "topics.py"))
    topics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(topics)

    orch = os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                        "wisepack_orchestration", "hitl_orchestrator.py")
    with open(orch, encoding="utf-8") as fh:
        source = fh.read()
    handler = source[source.index("def _apply_command("):]
    for cmd in topics.OPERATOR_COMMANDS:
        # Implemented either as an explicit `== "cmd"` branch or inside a grouped
        # `command in ("cmd", ...)` dispatch — both leave the command as a quoted
        # literal in the handler, so a dead button is still caught.
        assert (f'== "{cmd}"' in handler) or (f'"{cmd}"' in handler), (
            f"the orchestrator does not implement the advertised command "
            f"{cmd!r} — it would be a dead button in live mode")


def test_full_plans_are_published_not_summaries():
    """The live Digital Twin needs geometry, not utilization percentages."""
    orch = os.path.join(REPO, "wisepack_ws", "src", "wisepack_orchestration",
                        "wisepack_orchestration", "hitl_orchestrator.py")
    with open(orch, encoding="utf-8") as fh:
        source = fh.read()
    body = source[source.index("def publish_plans("):]
    body = body[:body.index("\n    def ")]
    for plan in ("baseline", "optimized", "selected"):
        assert f"engine.{plan}.to_dict()" in body, (
            f"{plan} must be published in full; a summary cannot be drawn")


# --------------------------------------------------------------------------- #
# Clean shutdown
# --------------------------------------------------------------------------- #

def test_ros_observer_catches_external_shutdown():
    """Ctrl+C must not print an ExternalShutdownException traceback.

    Verified live (no traceback, Orion-LD/Mongo persist); this pins the source so
    it cannot regress. The observer's spin thread must catch the expected
    shutdown exception and still report unexpected ones.
    """
    with open(os.path.join(REPO, "web", "ros_observer.py"), encoding="utf-8") as fh:
        src = fh.read()
    assert "ExternalShutdownException" in src, (
        "the observer spin thread must catch ExternalShutdownException so Ctrl+C "
        "does not print a traceback")
    spin = src[src.index("def spin("):]
    spin = spin[:spin.index("_thread = threading.Thread")]
    assert "except (ExternalShutdownException, KeyboardInterrupt)" in spin
    # Unexpected exceptions must still surface.
    assert "unexpected error" in spin


def test_fiware_launch_does_not_tear_down_orion_or_mongo():
    """Ctrl+C must leave Orion-LD and Mongo-DDS running (no compose down)."""
    with open(os.path.join(REPO, "run_wisepack_dashboard.sh"), encoding="utf-8") as fh:
        sh = fh.read()
    assert "docker compose -f docker-compose.dds.yml down" not in sh, (
        "the dashboard launcher must never tear down the Orion-LD/Mongo stack")
    # The cleanup trap must only kill the dashboard's own launch process group,
    # not a broad pattern. Ignore comment lines (the script documents the
    # rejected `pkill -f wisepack_` to explain why it is not used).
    code = "\n".join(l for l in sh.splitlines() if not l.lstrip().startswith("#"))
    assert "pkill -f wisepack_" not in code
