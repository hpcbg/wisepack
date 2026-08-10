"""Artefacts must land somewhere the launching user can actually write.

WHAT WENT WRONG. The dashboard hard-coded `<repo>/results` and so ignored the
`WISEPACK_RESULTS_DIR` override the rest of the project honours. On a shared
checkout that directory can belong to another user, and every run then finished
by reporting a permission error it could do nothing about — beside a green
COMPLETE badge.

Nothing here changes permissions. It chooses a destination, and it reports the
one it chose.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))

from test_cad_scenarios import _code_only                          # noqa: E402
from wisepack_core.artifacts import (                              # noqa: E402
    FALLBACK_RESULTS_DIR, RESULTS_DIR_ENV, resolve_results_dir)


def test_an_explicit_directory_is_honoured_verbatim(tmp_path):
    """An operator who names a directory gets that directory — including when
    it is unwritable, because silently redirecting their configuration behind
    their back is worse than failing where they can see it."""
    wanted = tmp_path / "somewhere"
    assert resolve_results_dir(str(wanted)) == str(wanted)


def test_the_environment_override_is_honoured(monkeypatch, tmp_path):
    monkeypatch.setenv(RESULTS_DIR_ENV, str(tmp_path / "env"))
    assert resolve_results_dir() == str(tmp_path / "env")


def test_a_relative_override_is_resolved_to_an_absolute_path(monkeypatch):
    """A relative default landed wherever the process happened to be started
    from, which for the dashboard is web/."""
    monkeypatch.setenv(RESULTS_DIR_ENV, "results")
    assert os.path.isabs(resolve_results_dir())


def test_a_writable_repo_results_directory_is_preferred(monkeypatch, tmp_path):
    monkeypatch.delenv(RESULTS_DIR_ENV, raising=False)
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    assert resolve_results_dir(repo_root=str(root)) == str(root / "results")


def test_an_unwritable_repo_directory_falls_back_to_a_user_location(
        monkeypatch, tmp_path):
    """THE CASE ON THIS MACHINE: results/ belongs to another user. The
    alternative would be requiring `sudo chown` to run a dashboard."""
    monkeypatch.delenv(RESULTS_DIR_ENV, raising=False)
    root = tmp_path / "repo"
    results = root / "results"
    results.mkdir(parents=True)
    results.chmod(0o500)                       # readable, not writable
    try:
        resolved = resolve_results_dir(repo_root=str(root))
        assert resolved == FALLBACK_RESULTS_DIR
        assert resolved != str(results)
    finally:
        results.chmod(0o700)                   # so tmp cleanup can remove it


def test_writability_is_tested_by_writing(tmp_path):
    """`os.access(W_OK)` answers about permission BITS and gets group- and
    ACL-owned directories wrong; the probe must match what the write will do.

    Asserted by behaviour, not by grepping for `os.access` — the tokeniser
    splits that into `os . access`, so the string check could never have
    matched and would have passed no matter what the function did.
    """
    from wisepack_core.artifacts import _writable

    good = tmp_path / "good"
    assert _writable(str(good)) is True
    assert good.is_dir(), "a missing directory is created, not merely reported"
    assert list(good.iterdir()) == [], "the probe file must not be left behind"

    bad = tmp_path / "bad"
    bad.mkdir()
    bad.chmod(0o500)
    try:
        assert _writable(str(bad)) is False
    finally:
        bad.chmod(0o700)


def test_writability_never_raises_on_a_hopeless_path():
    """The resolver runs at import time. A probe that raises there takes the
    dashboard down at start-up instead of falling back."""
    from wisepack_core.artifacts import _writable
    assert _writable("/proc/wisepack-cannot-exist/nested") is False


def test_the_resolver_changes_no_permissions():
    """§: no chmod, no chown, no sudo."""
    import inspect
    from wisepack_core import artifacts
    source = _code_only(inspect.getsource(artifacts))
    for forbidden in ("os.chmod", "os.chown", "sudo", "subprocess"):
        assert forbidden not in source, forbidden


def test_artifacts_can_actually_be_written_to_the_resolved_default(tmp_path):
    """End to end: a real run's artefacts land on disk and the path is real."""
    from wisepack_core.artifacts import write_run_artifacts
    from wisepack_core.generator import build_scenario
    from wisepack_core.kpi import KPIReport
    from wisepack_core.events import ActionLog
    from wisepack_core.packing import OptimizerConfig, pack_optimized

    scenario = build_scenario("cad_cylinder5_single")
    plan = pack_optimized(scenario, config=OptimizerConfig(restarts=4, seed=1))
    destination = str(tmp_path / "out")
    artifacts = write_run_artifacts(
        scenario, plan, plan, plan,
        KPIReport(scenario_id=scenario.scenario_id, run_id="test-run"),
        ActionLog(), destination)
    written = os.path.join(destination,
                           f"wisepack-run-{artifacts.stamp}.json")
    assert os.path.isfile(written), written
    assert os.path.getsize(written) > 0


def test_the_dashboard_does_not_hard_code_the_results_directory():
    """The bypass is what made the override useless."""
    source = open(os.path.join(REPO, "web", "app.py"), encoding="utf-8").read()
    assert 'os.path.join(REPO, "results")' not in source
    assert "resolve_results_dir(" in source


def test_a_failed_write_is_not_reported_as_success():
    """§: a COMPLETE run must not be shown beside a write failure. The command
    raises, and the run carries the outcome rather than a transient notice."""
    source = open(os.path.join(REPO, "web", "app.py"), encoding="utf-8").read()
    start = source.index('if command == "write_artifacts"')
    body = source[start:start + 800]
    assert 'if not outcome.get("ok")' in body
    assert "HTTPException" in body
    # And the state carries where it went, or why it did not.
    assert '"artifacts": (STATE.artifacts' in source
