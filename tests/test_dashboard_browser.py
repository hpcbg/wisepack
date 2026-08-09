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

    def __init__(self, preset="mixed_pipes_dense", seed=42,
                 perception_url=None):
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        # SELF-CONTAINED BY DEFAULT. Without this the dashboard inherits
        # WISEPACK_PERCEPTION_SERVICE_URL from the developer's shell and finds
        # whatever detector happens to be running on that machine — so a test
        # about "no camera is available" passed or failed depending on whether
        # someone had a service up. Pointed at a port nothing listens on, the
        # answer is deterministic.
        env = dict(os.environ, WISEPACK_STEP_PERIOD_S="0.35",
                   WISEPACK_PERCEPTION_SERVICE_URL=(
                       perception_url or f"http://127.0.0.1:{_free_port()}"))
        env.pop("WISEPACK_PERCEPTION_SOURCE", None)
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


def test_sim_compare_strategies_renders_all_rows(page, sim_server):
    """The confirmed defect: Compare strategies must produce a visible table.

    Clicks the real button, waits for the asynchronous comparison, and asserts
    the table shows one row per strategy with the required columns — without a
    page reload. A HTTP 200 is not enough; the rows must be on screen.
    """
    errors = PageErrors(page)
    page.goto(sim_server.url, wait_until="networkidle")
    reset_run(page)

    before_rev = page.evaluate(
        "async () => (await (await fetch('/api/state')).json()).scenario_revision")
    before_plan = page.evaluate(
        "async () => (await (await fetch('/api/state')).json()).selected_plan_id")

    page.click("#c-strategies")
    # Rows appear asynchronously once the comparison is published and polled.
    page.wait_for_function(
        "() => document.querySelectorAll('#strategies tbody tr').length >= 4",
        timeout=30_000)                                 # header + >=3 strategies

    body = page.inner_text("#strategies")
    lower = body.lower()                    # headers are CSS-uppercased
    for strat in ("max_density", "retrievability", "segregation"):
        assert strat in body, f"strategy {strat} missing from the comparison table"
    for col in ("cont.", "util %", "score", "valid"):
        assert col in lower, f"column {col} missing"

    # Comparing must not have changed the active plan or the scenario revision.
    after = page.evaluate(
        "async () => await (await fetch('/api/state')).json()")
    assert after["selected_plan_id"] == before_plan, "compare changed the plan"
    assert after["scenario_revision"] == before_rev, "compare bumped the revision"
    assert after["approval_state"] != "approved", "compare approved a plan"

    assert_no_refresh_failure(page, "after compare strategies")
    errors.assert_clean("after compare strategies")


def test_sim_stale_comparison_clears_on_new_scenario(page, sim_server):
    """A comparison from a superseded batch must never be rendered."""
    page.goto(sim_server.url, wait_until="networkidle")
    reset_run(page)
    page.click("#c-strategies")
    page.wait_for_function(
        "() => document.querySelectorAll('#strategies tbody tr').length >= 4",
        timeout=30_000)

    # Inject an item -> new revision -> the old comparison is stale.
    command(page, "approve")
    page.wait_for_timeout(800)
    command(page, "inject_item")
    page.wait_for_function(
        "() => document.querySelector('#b-stage')?.textContent?.trim() === "
        "'WAIT_FOR_OPERATOR_APPROVAL'", timeout=30_000)
    # The stale table must clear (the periodic refresh drops a stale comparison).
    page.wait_for_function(
        "() => document.querySelectorAll('#strategies tbody tr').length === 0",
        timeout=20_000)


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

    # HOW MANY OBJECTS THERE ACTUALLY ARE, asked rather than assumed.
    #
    # This used to hard-code "> 20 rects", which silently encoded "the attached
    # stack is running the 40-item dense scenario with SIMULATED perception".
    # Against a stack running WISEPACK_PERCEPTION_SOURCE=camera the scenario is
    # whatever is physically on the table — two proxy cylinders in the recorded
    # demonstration — and the test failed on a Digital Twin that was rendering
    # the truth. The threshold now follows the scenario: unchanged (> 20) for
    # the dense batch, and "every detected object is drawn" for a small one.
    items = page.evaluate(
        "async () => (await (await fetch('/api/state')).json())"
        "?.scenario?.totals?.items || 0")
    assert items > 0, "the attached stack reports no items at all"
    floor = min(items - 1, 20)
    page.wait_for_function(
        f"() => document.querySelectorAll('#twin rect').length > {floor}",
        timeout=60_000)
    page.wait_for_function(
        "() => document.querySelectorAll('#kpis .tile').length >= 10", timeout=60_000)
    page.wait_for_function(
        "() => document.querySelectorAll('#log .row').length > 3", timeout=60_000)

    kpis = page.inner_text("#kpis")
    core = kpis.split("DDS")[0]
    # The vision detection RATE is deliberately `not measured` whenever a real
    # detector is running — confidence is not a rate, and only a labelled
    # ground-truth trial produces one (§14). Excluding that one tile keeps the
    # check meaningful instead of turning an honest label into a failure.
    core = "\n".join(line for line in core.splitlines()
                      if "detection rate" not in line.lower())
    assert "not measured" not in core, \
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


def test_sim_diagnostics_page_loads_without_errors(page, sim_server):
    """The /diagnostics page must render every section with no page error."""
    errors = PageErrors(page)
    page.goto(sim_server.url + "/diagnostics", wait_until="networkidle")
    page.wait_for_function(
        "() => document.querySelectorAll('#components tbody tr').length > 3",
        timeout=30_000)
    page.wait_for_function(
        "() => document.querySelectorAll('#topics tbody tr').length > 10",
        timeout=30_000)
    page.wait_for_function(
        "() => document.querySelectorAll('#fiware tbody tr').length > 10",
        timeout=30_000)
    body = page.inner_text("body").lower()
    assert "simulated, unavailable and future interfaces" in body
    assert "not a system failure" in body
    # No secret-looking content on the page.
    for bad in ("BEGIN PRIVATE KEY", "aws_secret", "password="):
        assert bad not in body
    errors.assert_clean("diagnostics page")


def test_sim_diagnostics_link_is_in_the_header(page, sim_server):
    page.goto(sim_server.url, wait_until="networkidle")
    link = page.query_selector('a[href="/diagnostics"]')
    assert link is not None, "the dashboard header must link to /diagnostics"


def test_sim_anomaly_panel_present_and_labelled(page, sim_server):
    errors = PageErrors(page)
    page.goto(sim_server.url, wait_until="networkidle")
    reset_run(page)
    page.wait_for_timeout(1500)
    # textContent, not inner_text: the title has a literal '&' that inner_text
    # does not preserve verbatim.
    body = page.evaluate("() => document.body.textContent || ''").lower()
    # Panel title: Anomaly Monitoring & Workflow Response
    assert "anomaly monitoring & workflow response" in body
    assert "edf topic #2 integration demo" not in body      # old label removed
    assert "not a validated anomaly detector" in body
    # Inject a critical anomaly and confirm it holds execution visibly.
    command(page, "approve")
    page.wait_for_timeout(1000)
    page.select_option("#a-class", "shear_position_too_high")
    command(page, "inject_anomaly", {"anomaly_class": "shear_position_too_high"})
    page.wait_for_function(
        "() => (document.querySelector('#anomaly-state')?.textContent||'').includes('HELD')",
        timeout=20_000)
    errors.assert_clean("anomaly panel")


# --------------------------------------------------------------------------- #
# Cut-aware whole-process UI (brief §6, §23)
# --------------------------------------------------------------------------- #

def test_sim_cut_aware_comparison_renders_and_approves(page, sim_server):
    """Compare no-cut vs cut-aware, approve cutting, simulate: containers drop."""
    errors = PageErrors(page)
    page.goto(sim_server.url, wait_until="networkidle")
    reset_run(page, preset="cut_avoids_extra_container", seed=7)
    page.wait_for_timeout(800)
    # Generate the whole-process comparison through the real button path.
    command(page, "compare_cut_aware")
    page.wait_for_function(
        "() => (document.querySelector('#cut-state')?.textContent||'')"
        ".includes('CUT RECOMMENDED')", timeout=20_000)
    body = page.inner_text("#cut-state")
    assert "saved 1" in body.lower() or "→ 1" in body
    # Cut approval is SEPARATE from packing approval.
    command(page, "approve_cut")
    page.wait_for_timeout(500)
    res = command(page, "simulate_cut")
    assert res["status"] == 200, res
    # After the cut the plan re-plans to fewer containers and awaits packing approval.
    wait_for_stage(page, "WAIT_FOR_OPERATOR_APPROVAL")
    errors.assert_clean("cut-aware comparison")


def test_sim_cut_not_worthwhile_shows_no_cut(page, sim_server):
    errors = PageErrors(page)
    page.goto(sim_server.url, wait_until="networkidle")
    reset_run(page, preset="cut_not_worthwhile", seed=7)
    command(page, "compare_cut_aware")
    page.wait_for_function(
        "() => (document.querySelector('#cut-state')?.textContent||'').includes('NO CUT')",
        timeout=20_000)
    errors.assert_clean("no-cut recommendation")


def test_sim_timeline_filters_switch(page, sim_server):
    errors = PageErrors(page)
    page.goto(sim_server.url, wait_until="networkidle")
    reset_run(page, preset="cut_avoids_extra_container", seed=7)
    command(page, "compare_cut_aware")
    command(page, "approve_cut")
    command(page, "simulate_cut")
    page.wait_for_timeout(1200)
    # Switch the timeline to the cutting category.
    page.click("#log-filters button[data-f='cutting']")
    page.wait_for_timeout(400)
    assert page.get_attribute("#log-filters button[data-f='cutting']", "aria-pressed") == "true"
    errors.assert_clean("timeline filters")


# --------------------------------------------------------------------------- #
# Inventory + logistics page (brief §12, §16)
# --------------------------------------------------------------------------- #

def test_inventory_page_opens_and_initialises(page, sim_server):
    errors = PageErrors(page)
    page.goto(sim_server.url + "/inventory", wait_until="networkidle")
    page.wait_for_timeout(600)
    body = page.inner_text("body").lower()
    assert "container inventory" in body
    assert "no physical mobile robot" in body           # honesty label
    # Initialise the simulated inventory and confirm the table fills.
    page.click("#op-init")
    page.wait_for_function(
        "() => document.querySelectorAll('#rows tr').length >= 1", timeout=20_000)
    errors.assert_clean("inventory page")


def test_logistics_route_shows_facility_map(page, sim_server):
    errors = PageErrors(page)
    page.goto(sim_server.url + "/logistics", wait_until="networkidle")
    page.wait_for_timeout(600)
    assert page.query_selector("#map") is not None
    assert "simulated container-logistics" in page.inner_text("body").lower()
    errors.assert_clean("logistics route")


def test_diagnostics_has_cut_inventory_logistics_status(page, sim_server):
    errors = PageErrors(page)
    page.goto(sim_server.url + "/diagnostics", wait_until="networkidle")
    page.wait_for_timeout(1500)
    body = page.inner_text("body").lower()
    assert "cutting status" in body
    assert "inventory status" in body
    assert "logistics status" in body
    errors.assert_clean("diagnostics whole-process")


# --------------------------------------------------------------------------- #
# Object source: both workflows in ONE dashboard session
# --------------------------------------------------------------------------- #


#: A 1x1 JPEG. The smallest thing that decodes in a browser, so the panel's
#: <img> resolves instead of reporting a failed request.
_PIXEL_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300ffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffc2000b080001000101011100ffc4"
    "001400010000000000000000000000000000000affda0008010100013f10")


class StubPerceptionService:
    """A perception service that answers /health and returns one batch.

    A REAL HTTP SERVER, on a real port, because the whole property under test is
    "the dashboard notices a service that appears and offers the camera without
    being restarted" — and a monkeypatched client would not exercise that at
    all. No torch, no camera, no model: the two endpoints the dashboard uses.
    """

    OBJECTS = [
        {"observation_id": "physical-cylinder-001", "object_type": "cylindrical_proxy",
         "source": "camera", "frame_id": "wisepack_workarea",
         "pose": {"x_mm": 184.5, "y_mm": 54.1, "z_mm": 0.0, "yaw_deg": 51.7},
         "confidence": 0.99, "calibration_status": "valid",
         "geometry": {"diameter_mm": 65, "length_mm": 215,
                      "source": "configured_proxy"}},
        {"observation_id": "physical-cylinder-002", "object_type": "cylindrical_proxy",
         "source": "camera", "frame_id": "wisepack_workarea",
         "pose": {"x_mm": 77.5, "y_mm": 53.9, "z_mm": 0.0, "yaw_deg": -21.1},
         "confidence": 0.99, "calibration_status": "valid",
         "geometry": {"diameter_mm": 65, "length_mm": 215,
                      "source": "configured_proxy"}},
    ]

    def __init__(self):
        import http.server
        import json as _json
        import threading as _threading

        batch = {
            "batch_id": "batch-001", "source": "camera", "status": "ok",
            "count": len(self.OBJECTS), "frame_id": "wisepack_workarea",
            "captured_at": "2026-08-09T10:00:00.000Z",
            "requested_at": "2026-08-09T09:59:59.000Z",
            "detector": "fasterrcnn_resnet50_fpn/bottle",
            "model_id": "/stub/best_model.pth",
            "calibration_status": "valid", "calibration_revision": "stub0001",
            "error": "", "mean_confidence": 0.99,
            "observations": self.OBJECTS, "detector_status": {},
        }
        health = {
            "source": "camera", "service_reachable": True,
            "camera_available": True, "model_available": True,
            "model_loaded": True, "calibration_status": "valid",
            "calibration_valid": True, "detector": "fasterrcnn_resnet50_fpn/bottle",
            "detector_display_name": "Faster R-CNN", "provider": "fasterrcnn_bottle",
            "last_error": "", "detected_objects": len(self.OBJECTS),
        }

        class Handler(http.server.BaseHTTPRequestHandler):
            def _send(self, payload):
                body = _json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    self._send(health)
                elif self.path == "/api/v1/camera/last-detection":
                    self._send(batch)
                elif self.path.startswith("/api/v1/detection/image/"):
                    # The panel shows the annotated frame, so the stub has to
                    # return SOMETHING decodable: a 503 here would surface as a
                    # failed request and fail the test for a reason that has
                    # nothing to do with the workflow under examination.
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(_PIXEL_JPEG)))
                    self.end_headers()
                    self.wfile.write(_PIXEL_JPEG)
                else:
                    self.send_response(404); self.end_headers()

            def do_POST(self):
                if self.path == "/api/v1/detect":
                    self._send(batch)
                else:
                    self.send_response(404); self.end_headers()

            def log_message(self, *a):
                pass

        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = http.server.HTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = _threading.Thread(target=self.server.serve_forever,
                                        daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture(scope="module")
def camera_dashboard():
    """A dashboard whose perception service exists, started on PRESETS.

    `WISEPACK_PERCEPTION_SOURCE` is deliberately NOT set: the session opens on
    the ordinary preset workflow, exactly as `./run_wisepack_dashboard.sh` does,
    and the camera is merely available.
    """
    service = StubPerceptionService()
    port = _free_port()
    env = dict(os.environ, WISEPACK_STEP_PERIOD_S="0.35",
               WISEPACK_PERCEPTION_SERVICE_URL=service.url)
    env.pop("WISEPACK_PERCEPTION_SOURCE", None)
    proc = subprocess.Popen(
        [sys.executable, "app.py", "--source", "sim", "--port", str(port),
         "--preset", "mixed_pipes_dense", "--seed", "42"],
        cwd=WEB, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    import urllib.request
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("dashboard exited early:\n"
                               + proc.stdout.read().decode(errors="replace"))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz",
                                        timeout=2):
                break
        except Exception:                                # noqa: BLE001
            time.sleep(1)
    else:
        raise RuntimeError("dashboard did not become ready")

    class Harness:
        pass

    harness = Harness()
    harness.url = f"http://127.0.0.1:{port}"
    harness.service = service
    yield harness
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    service.close()


def _twin_rects(page) -> int:
    return page.evaluate("() => document.querySelectorAll('#twin rect').length")


def _object_source(page):
    return page.evaluate(
        "async () => (await (await fetch('/api/state')).json()).object_source")


def test_object_source_switches_within_one_dashboard_session(page,
                                                             camera_dashboard):
    """THE HEADLINE: preset -> camera -> preset, one page, no restart.

    Every step is driven through the real UI — the selector and the acquisition
    button — not through the API, because the thing that was broken was the
    workflow an operator can actually perform.
    """
    errors = PageErrors(page)
    # STARTING A NEW RUN ASKS FIRST when one is active — including when the
    # acquisition also switches the object source. Headless Chromium dismisses
    # dialogs by default, which would silently turn every click below into a
    # no-op, so the operator's "yes" is given explicitly.
    page.on("dialog", lambda dialog: dialog.accept())
    page.goto(camera_dashboard.url, wait_until="networkidle")
    page.wait_for_timeout(2500)

    # -- 2-5. the preset workflow, unchanged --------------------------------
    assert page.query_selector("#s-objsrc") is not None, \
        "the Object source selector must exist in the Scenario panel"
    source = _object_source(page)
    assert source["current"] == "sim" and source["selected"] == "sim"
    assert source["camera_available"] is True, \
        "an available camera must be offered even on a preset session"
    page.wait_for_function(
        "() => document.querySelectorAll('#twin rect').length > 20",
        timeout=60_000)
    preset_rects = _twin_rects(page)
    assert "generate" in page.inner_text("#c-reset").strip().lower()
    wait_for_stage(page, "WAIT_FOR_OPERATOR_APPROVAL")
    assert page.eval_on_selector("#c-approve", "b => !b.disabled")

    # -- 6. switch the NEXT run to the camera -------------------------------
    page.select_option("#s-objsrc", "camera")
    page.wait_for_function(
        "async () => ((await (await fetch('/api/state')).json())"
        ".object_source || {}).selected === 'camera'", timeout=20_000)
    # The RUNNING run is untouched: same twin, still awaiting the same decision.
    assert _twin_rects(page) == preset_rects, \
        "selecting a source changed the running run"
    assert _object_source(page)["current"] == "sim"
    page.wait_for_function(
        "() => document.querySelector('#c-reset')?.textContent?.includes('detect')"
        " || document.querySelector('#c-reset')?.textContent?.includes('Detect')",
        timeout=20_000)

    # -- 7-9. acquire from the camera ---------------------------------------
    page.click("#c-reset")
    page.wait_for_function(
        "async () => ((await (await fetch('/api/state')).json())"
        ".object_source || {}).current === 'camera'", timeout=60_000)
    page.wait_for_timeout(2500)
    camera_rects = _twin_rects(page)
    assert 0 < camera_rects < preset_rects, (
        "the Digital Twin must now show the two detected objects, not the "
        f"generated batch ({camera_rects} vs {preset_rects})")
    items = page.evaluate(
        "async () => ((await (await fetch('/api/state')).json())"
        ".scenario || {}).totals?.items")
    assert items == 2, f"expected the two detected objects, got {items}"
    wait_for_stage(page, "WAIT_FOR_OPERATOR_APPROVAL")
    assert page.eval_on_selector("#c-approve", "b => !b.disabled"), \
        "a fresh decision must be available for the camera batch"

    # -- 10-12. back to a preset, still the same page -----------------------
    page.select_option("#s-objsrc", "sim")
    page.wait_for_function(
        "async () => ((await (await fetch('/api/state')).json())"
        ".object_source || {}).selected === 'sim'", timeout=20_000)
    page.click("#c-reset")
    page.wait_for_function(
        "async () => ((await (await fetch('/api/state')).json())"
        ".object_source || {}).current === 'sim'", timeout=60_000)
    page.wait_for_function(
        "() => document.querySelectorAll('#twin rect').length > 20",
        timeout=60_000)
    assert _twin_rects(page) > camera_rects
    wait_for_stage(page, "WAIT_FOR_OPERATOR_APPROVAL")

    # -- 14. nothing restarted, nothing broke -------------------------------
    assert_no_refresh_failure(page, "object-source switching")
    errors.assert_clean("object-source switching")


def test_the_perception_panel_stays_visible_during_a_preset_run(
        page, camera_dashboard):
    """The camera is a capability; the panel reports it whatever is running."""
    page.goto(camera_dashboard.url, wait_until="networkidle")
    page.wait_for_timeout(2500)
    panel = page.query_selector("#perceppanel")
    assert panel is not None and panel.is_visible(), (
        "the Physical Perception panel must stay visible while a camera exists")
    text = page.inner_text("#perceppanel")
    assert "Current run source" in text
    assert "Camera" in text


def test_an_absent_camera_leaves_the_preset_workflow_working(page, sim_server):
    """No perception service at all: an ordinary WISEPACK, plus a clear reason.

    `sim_server` has no `WISEPACK_PERCEPTION_SERVICE_URL`, so nothing answers.
    """
    errors = PageErrors(page)
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_timeout(1500)
    # The dashboard server is module-scoped and the workflow is STATEFUL, so put
    # the engine back at the gate on a known scenario rather than inheriting
    # whatever the previous test left behind.
    reset_run(page)
    page.wait_for_timeout(1500)

    source = _object_source(page)
    assert source["camera_available"] is False
    assert source["camera_unavailable_reason"], \
        "an unavailable camera must say why"
    # OFFERED BUT DISABLED, never silently missing: an operator has to be able
    # to see that the capability exists and what is wrong with it.
    disabled = page.evaluate(
        "() => [...document.querySelectorAll('#s-objsrc option')]"
        ".map(o => [o.value, o.disabled])")
    assert ["camera", True] in [list(x) for x in disabled], disabled

    # ... and the preset workflow is completely unaffected.
    page.wait_for_function(
        "() => document.querySelectorAll('#twin rect').length > 20",
        timeout=60_000)
    wait_for_stage(page, "WAIT_FOR_OPERATOR_APPROVAL")
    # The acquisition button names a GENERATION, never a detection — a run is
    # active by now, so the label carries the reset wording too.
    label = page.inner_text("#c-reset").strip()
    assert "generate" in label.lower(), label
    assert "detect" not in label.lower(), label
    assert_no_refresh_failure(page, "absent camera")
    errors.assert_clean("absent camera")


def test_asking_to_detect_without_a_camera_is_refused_not_simulated(
        page, sim_server):
    """NO SILENT FALLBACK, asserted through the API the button uses."""
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_timeout(1500)
    reset_run(page)
    page.wait_for_timeout(1000)
    before = page.evaluate(
        "async () => ((await (await fetch('/api/state')).json())"
        ".scenario || {}).totals?.items")
    result = page.evaluate(
        """async () => {
             const r = await fetch('/api/command', {
               method: 'POST', headers: {'Content-Type': 'application/json'},
               body: JSON.stringify({command: 'detect_physical_objects', args: {}})});
             return {status: r.status, body: await r.text()};
           }""")
    assert result["status"] == 409, result
    assert "not available" in result["body"]
    after = page.evaluate(
        "async () => ((await (await fetch('/api/state')).json())"
        ".scenario || {}).totals?.items")
    assert after == before, "a refused detection changed the run"


# --------------------------------------------------------------------------- #
# Perception method selector and FoundationPose status
# --------------------------------------------------------------------------- #
#
# These render against whatever the deployment actually has. On a machine with
# no FoundationPose worker the method shows as unavailable WITH ITS REASON, and
# that is a state these tests accept and check — a test that required a GPU
# container would not be a dashboard test.


def test_the_perception_method_selector_appears_only_with_a_camera_selected(page, sim_server):
    """"How is the frame read" has no answer when there is no frame. With a
    preset selected the row is hidden, not merely disabled."""
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_timeout(3000)
    field = page.query_selector("#s-method-field")
    assert field is not None, "the perception-method field is missing"
    # The default scenario is a preset, so the row starts hidden.
    assert not field.is_visible(), (
        "the method selector is shown while the objects come from a preset")


def test_the_method_selector_offers_both_methods_naming_the_capability(page, sim_server):
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_timeout(3000)
    values = page.eval_on_selector_all(
        "#s-method option", "els => els.map(e => e.value)")
    assert "planar_fasterrcnn" in values
    assert "foundationpose_rgbd" in values
    labels = page.eval_on_selector_all(
        "#s-method option", "els => els.map(e => e.textContent)")
    joined = " ".join(labels)
    # NAMED FOR WHAT THEY DO, not for the module that implements them.
    assert "Planar RGB" in joined
    assert "6-DoF" in joined


def test_an_unavailable_method_is_disabled_with_a_reason_not_hidden(page, sim_server):
    """A missing option tells an operator nothing; a disabled one with its
    reason tells them what to fix."""
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_timeout(3000)
    options = page.eval_on_selector_all(
        "#s-method option",
        "els => els.map(e => ({value: e.value, disabled: e.disabled, title: e.title}))")
    by_value = {o["value"]: o for o in options}
    assert "foundationpose_rgbd" in by_value, (
        "an unavailable method must still be offered")
    entry = by_value["foundationpose_rgbd"]
    if entry["disabled"]:
        assert entry["title"], "a disabled method must carry its reason"


def test_the_planar_method_is_the_default_selection(page, sim_server):
    """FoundationPose is never the default: it needs a depth camera, a GPU,
    weights and a CAD model."""
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_timeout(3000)
    assert page.eval_on_selector("#s-method", "e => e.value") == "planar_fasterrcnn"


def test_the_foundationpose_block_reports_every_prerequisite_separately(page, sim_server):
    """One collapsed "unavailable" is what produces a dashboard nobody can act
    on. Each link in the chain gets its own statement."""
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_timeout(3000)
    block = page.query_selector("#fp-block")
    assert block is not None, "the FoundationPose block is missing"
    if not block.is_visible():
        pytest.skip("this build does not offer the FoundationPose method")
    status = page.inner_text("#fp-status")
    for label in ("Worker:", "GPU:", "FoundationPose runtime:",
                  "Scorer weights:", "Refiner weights:", "RGB-D camera:",
                  "Live inference:"):
        assert label in status, f"the FoundationPose status omits {label!r}"


def test_a_ready_runtime_without_a_camera_says_exactly_that(page, sim_server):
    """The state this deployment is actually in, and the wording that must not
    collapse it: runtime READY, camera unavailable, live inference unavailable."""
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_timeout(3000)
    block = page.query_selector("#fp-block")
    if block is None or not block.is_visible():
        pytest.skip("this build does not offer the FoundationPose method")
    status = page.inner_text("#fp-status")
    badge = page.inner_text("#fp-badge")
    if "RGB-D camera: unavailable" in status:
        assert "Live inference: unavailable" in status, (
            "a missing depth camera must make live inference unavailable")
        assert "INFERENCE READY" not in badge, (
            "the badge claims inference is ready with no RGB-D camera")


def test_the_offline_regression_control_is_labelled_as_offline(page, sim_server):
    """§16: it must read as a reference regression, never as a live camera
    acquisition. A control labelled "detect" would put a saved pose into a run
    as though a camera produced it."""
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_timeout(3000)
    button = page.query_selector("#c-fp-reference")
    if button is None or not button.is_visible():
        pytest.skip("this build does not offer the FoundationPose method")
    label = button.inner_text().lower()
    assert "reference" in label or "offline" in label
    assert "detect" not in label


def test_the_dashboard_renders_with_the_method_selector_present(page, sim_server):
    """The whole point of an additive change: nothing else breaks."""
    page.goto(sim_server.url, wait_until="networkidle")
    page.wait_for_timeout(3000)
    assert_no_refresh_failure(page, "perception method selector")
