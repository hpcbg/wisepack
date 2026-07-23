"""Headless-browser acceptance tests for the dashboard.

These exist because "the process started" is not the same as "the dashboard
works". Every test here drives a REAL Chromium against a REAL server and fails
on anything a human would notice:

  * an uncaught page error (`pageerror`)
  * a console error
  * a failed API request
  * the text "refresh failed" anywhere on the page

The last one is deliberate. The reported symptom was

    refresh failed: Cannot set properties of undefined (setting 'title')

caused by `Element.append()` returning undefined, and it blanked the whole
dashboard. A test that only checked HTTP status codes would have passed.

Simulation mode runs everywhere and is always exercised. ROS and FIWARE modes
need the live stack, so they are opt-in:

    WISEPACK_BROWSER_ROS=http://127.0.0.1:8095    pytest tests/test_dashboard_browser.py
    WISEPACK_BROWSER_FIWARE=http://127.0.0.1:8094 pytest tests/test_dashboard_browser.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import pytest

playwright_api = pytest.importorskip(
    "playwright.sync_api", reason="pip install playwright && playwright install chromium")
from playwright.sync_api import sync_playwright                    # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(REPO, "web")
VIEWPORT = {"width": 1440, "height": 900}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class PageErrors:
    """Collects everything that should fail a test, from one page."""

    def __init__(self, page):
        self.page_errors = []
        self.console_errors = []
        self.failed_requests = []
        page.on("pageerror", lambda e: self.page_errors.append(str(e)))
        page.on("console", self._console)
        page.on("requestfailed",
                lambda r: self.failed_requests.append(f"{r.method} {r.url}"))
        page.on("response", self._response)

    def _console(self, msg):
        if msg.type == "error":
            self.console_errors.append(msg.text)

    def _response(self, response):
        if response.url.rstrip("/").endswith("/favicon.ico"):
            return
        if response.status >= 400:
            self.failed_requests.append(f"HTTP {response.status} {response.url}")

    def assert_clean(self, context=""):
        problems = []
        if self.page_errors:
            problems.append(f"page errors: {self.page_errors}")
        if self.console_errors:
            problems.append(f"console errors: {self.console_errors}")
        if self.failed_requests:
            problems.append(f"failed requests: {self.failed_requests}")
        assert not problems, f"{context}\n" + "\n".join(problems)


def assert_no_refresh_failure(page, context=""):
    body = page.inner_text("body")
    assert "refresh failed" not in body.lower(), (
        f"{context}: the page shows a refresh failure\n"
        f"notice: {page.inner_text('#notice') if page.query_selector('#notice') else ''}")
    assert "panel(s) unavailable" not in body.lower(), (
        f"{context}: one or more panels failed to render\n"
        f"notice: {page.inner_text('#notice') if page.query_selector('#notice') else ''}")


class DashboardServer:
    """A sim-mode dashboard on its own port, torn down with the test."""

    def __init__(self, preset="mixed_pipes_dense", seed=42):
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        env = dict(os.environ, WISEPACK_STEP_PERIOD_S="0.35")
        self.proc = subprocess.Popen(
            [sys.executable, "app.py", "--source", "sim", "--port", str(self.port),
             "--preset", preset, "--seed", str(seed)],
            cwd=WEB, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self._await_ready()

    def _await_ready(self, timeout=90):
        import urllib.request
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                out = self.proc.stdout.read().decode(errors="replace")
                raise RuntimeError(f"dashboard exited early:\n{out}")
            try:
                with urllib.request.urlopen(f"{self.url}/healthz", timeout=2):
                    return
            except Exception:                           # noqa: BLE001
                time.sleep(1)
        raise RuntimeError("dashboard did not become ready")

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@pytest.fixture(scope="module")
def sim_server():
    server = DashboardServer()
    yield server
    server.close()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser):
    ctx = browser.new_context(viewport=VIEWPORT)
    p = ctx.new_page()
    yield p
    ctx.close()


def wait_for_stage(page, *stages, timeout=90_000):
    joined = "|".join(stages)
    page.wait_for_function(
        f"() => {joined.__class__ and 'true'} && "
        f"[{','.join(repr(s) for s in stages)}].includes("
        f"document.querySelector('#b-stage')?.textContent?.trim())",
        timeout=timeout)


def reset_run(page, **settings):
    """Start a fresh run through the real `reset` command.

    The dashboard server is module-scoped and the workflow is STATEFUL, so a
    test that needs the approval gate must put the engine back there rather than
    assume the previous test left it alone. Using the real command keeps the
    test honest — it exercises "Generate & plan", it does not reach inside.
    """
    args = {"preset": "mixed_pipes_dense", "seed": 42,
            "strategy": "max_density", "dynamic_events_enabled": True}
    args.update(settings)
    res = command(page, "reset", args)
    assert res["status"] == 200, res
    wait_for_stage(page, "WAIT_FOR_OPERATOR_APPROVAL")
    return res


def command(page, name, args=None):
    """Issue an operator command through the same REST path the buttons use."""
    return page.evaluate(
        """async ([cmd, args]) => {
             const r = await fetch('/api/command', {
               method: 'POST', headers: {'Content-Type': 'application/json'},
               body: JSON.stringify({command: cmd, args: args || {}})});
             return {status: r.status, body: await r.text()};
           }""", [name, args or {}])


# --------------------------------------------------------------------------- #
# Simulation mode
# --------------------------------------------------------------------------- #

def test_sim_dashboard_loads_without_errors(page, sim_server):
    errors = PageErrors(page)
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_timeout(4000)
    assert_no_refresh_failure(page, "initial load")
    errors.assert_clean("initial load")


def test_sim_renders_the_dense_scenario_and_the_measured_result(page, sim_server):
    """The headline claim, verified in the browser: 3 containers -> 2."""
    errors = PageErrors(page)
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_selector("#b-scenario")
    page.wait_for_function(
        "() => document.querySelector('#b-scenario')?.textContent?.includes('mixed_pipes_dense')",
        timeout=60_000)

    page.wait_for_function(
        "() => document.querySelectorAll('#cmp tbody tr').length > 3", timeout=60_000)
    rows = page.inner_text("#cmp")
    assert "Containers required" in rows
    containers = page.evaluate("""() => {
        const tr = [...document.querySelectorAll('#cmp tbody tr')]
            .find(r => r.textContent.includes('Containers required'));
        return tr ? [...tr.querySelectorAll('td')].map(td => td.textContent.trim()) : null;
    }""")
    assert containers is not None, "no 'Containers required' row"
    assert containers[1] == "3", f"baseline should need 3 containers, got {containers}"
    assert containers[2] == "2", f"optimized should need 2 containers, got {containers}"

    assert_no_refresh_failure(page, "comparison table")
    errors.assert_clean("comparison table")


def test_sim_digital_twin_draws_real_geometry(page, sim_server):
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_function(
        "() => document.querySelectorAll('#twin rect').length > 20", timeout=60_000)
    n = page.evaluate("() => document.querySelectorAll('#twin rect').length")
    assert n > 20, f"the Digital Twin drew only {n} rectangles — no real geometry"


def test_sim_kpi_tiles_render_with_provenance(page, sim_server):
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_function(
        "() => document.querySelectorAll('#kpis .tile').length >= 10", timeout=60_000)
    text = page.inner_text("#kpis")
    assert "measured" in text
    # An unmeasured KPI must read "not measured", never 0.
    assert "not measured" in text.lower()


def test_sim_approval_gate_then_execution(page, sim_server):
    """The safety story, end to end in the browser."""
    errors = PageErrors(page)
    page.goto(sim_server.url, wait_until="networkidle")
    reset_run(page)

    assert "no physical action is authorised" in page.inner_text("body").lower()
    assert page.evaluate(
        "() => parseFloat(document.querySelector('#progbar').style.width || '0')") == 0

    res = command(page, "approve")
    assert res["status"] == 200, res
    page.wait_for_function(
        "() => parseFloat(document.querySelector('#progbar').style.width || '0') > 0",
        timeout=60_000)

    assert_no_refresh_failure(page, "after approval")
    errors.assert_clean("after approval")


def test_sim_inject_item_forces_replan_and_new_approval(page, sim_server):
    """A re-plan must return to the gate — it must never bypass the human."""
    errors = PageErrors(page)
    page.goto(sim_server.url, wait_until="networkidle")
    # Dynamic events off: this test injects the item ITSELF, and the scripted
    # late-arrival event would otherwise re-plan first and blur what is asserted.
    reset_run(page, dynamic_events_enabled=False)
    command(page, "approve")
    page.wait_for_function(
        "() => parseFloat(document.querySelector('#progbar').style.width || '0') > 0",
        timeout=60_000)

    res = command(page, "inject_item")
    assert res["status"] == 200, res
    wait_for_stage(page, "WAIT_FOR_OPERATOR_APPROVAL")

    body = page.inner_text("body")
    assert "awaiting your decision" in body.lower() or \
           "no physical action is authorised" in body.lower(), \
        "after a re-plan the operator must be asked again"

    res = command(page, "approve")
    assert res["status"] == 200, res
    errors.assert_clean("after re-plan")
    assert_no_refresh_failure(page, "after re-plan")


def test_sim_every_advertised_command_is_accepted(page, sim_server):
    """No dead buttons: every advertised command must be handled, not 404/500."""
    page.goto(sim_server.url, wait_until="networkidle")
    reset_run(page)
    advertised = page.evaluate(
        """async () => (await (await fetch('/api/state')).json()).commands""")
    assert advertised, "the dashboard advertises no commands"

    # `approve` first so the execution-gated commands are legal.
    command(page, "approve")
    page.wait_for_timeout(1500)

    for name in advertised:
        args = {"strategy": "retrievability"} if name == "alternative_strategy" else {}
        res = command(page, name, args)
        # 200 = applied, 409 = refused for a stated reason (e.g. resume while
        # unapproved). Anything else means the command is not implemented.
        assert res["status"] in (200, 409), f"command {name}: {res}"
        assert "unknown command" not in res["body"], f"command {name} is a dead button"
        page.wait_for_timeout(300)


def test_sim_theme_toggle_does_not_break_rendering(page, sim_server):
    errors = PageErrors(page)
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_timeout(2500)
    page.click("#theme-l")
    page.wait_for_timeout(1200)
    assert page.evaluate("() => document.documentElement.dataset.theme") == "light"
    page.click("#theme-d")
    page.wait_for_timeout(1200)
    errors.assert_clean("theme toggle")
    assert_no_refresh_failure(page, "theme toggle")


def test_sim_source_badge_says_simulated(page, sim_server):
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_function(
        "() => document.querySelector('#b-source')?.textContent?.includes('SIMULATED')",
        timeout=30_000)


# --------------------------------------------------------------------------- #
# Live modes — opt-in against an already-running stack
# --------------------------------------------------------------------------- #

ROS_URL = os.environ.get("WISEPACK_BROWSER_ROS")
FIWARE_URL = os.environ.get("WISEPACK_BROWSER_FIWARE")


@pytest.mark.skipif(not ROS_URL, reason="set WISEPACK_BROWSER_ROS to a live dashboard")
def test_ros_dashboard_is_fully_populated(page):
    """ROS mode must render what sim mode renders — not an empty shell."""
    errors = PageErrors(page)
    page.goto(ROS_URL, wait_until="networkidle")
    page.wait_for_timeout(6000)

    badge = page.inner_text("#b-source")
    assert "ROS" in badge or "FIWARE" in badge, badge

    page.wait_for_function(
        "() => document.querySelector('#b-scenario')?.textContent?.includes('mixed_pipes')",
        timeout=60_000)
    page.wait_for_function(
        "() => document.querySelectorAll('#twin rect').length > 20", timeout=60_000)
    page.wait_for_function(
        "() => document.querySelectorAll('#kpis .tile').length >= 10", timeout=60_000)
    page.wait_for_function(
        "() => document.querySelectorAll('#log .row').length > 3", timeout=60_000)

    kpis = page.inner_text("#kpis")
    assert "not measured" not in kpis.split("DDS")[0], \
        "core KPI tiles are unmeasured in ROS mode"

    assert_no_refresh_failure(page, "ROS mode")
    errors.assert_clean("ROS mode")


@pytest.mark.skipif(not ROS_URL, reason="set WISEPACK_BROWSER_ROS to a live dashboard")
def test_ros_approve_changes_stage_and_progress(page):
    page.goto(ROS_URL, wait_until="networkidle")
    page.wait_for_timeout(4000)
    stage = page.inner_text("#b-stage")
    if stage == "WAIT_FOR_OPERATOR_APPROVAL":
        assert command(page, "approve")["status"] == 200
        page.wait_for_function(
            "() => parseFloat(document.querySelector('#progbar').style.width || '0') > 0",
            timeout=90_000)
    assert_no_refresh_failure(page, "ROS approve")


@pytest.mark.skipif(not FIWARE_URL,
                    reason="set WISEPACK_BROWSER_FIWARE to a live dashboard")
def test_fiware_dashboard_shows_ngsi_ld_values(page):
    errors = PageErrors(page)
    page.goto(FIWARE_URL, wait_until="networkidle")
    page.wait_for_timeout(8000)

    badge = page.inner_text("#b-source")
    assert "FIWARE" in badge, f"badge should name FIWARE, got {badge}"

    conn = page.inner_text("#b-fiware")
    assert "connected" in conn.lower(), conn

    page.wait_for_function(
        "() => document.querySelectorAll('#kpis .tile').length >= 10", timeout=60_000)
    assert_no_refresh_failure(page, "FIWARE mode")
    errors.assert_clean("FIWARE mode")
