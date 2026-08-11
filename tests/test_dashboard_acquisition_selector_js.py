"""The Acquisition selector, driven in a real DOM against a real server.

WHY THIS EXISTS. The selector rendered BLANK on an ordinary startup, and every
source-level check passed while it did. `#s-acq` was the one VISIBLE control
whose entire contents came from the backend — the preset list, the strategy list
and the container list are static or always present, and the object-source and
perception-method selectors are hidden under a preset — so a state document that
did not carry `acquisition_choice` left it as an empty box. `panel()` catches a
failed refresh per panel, so the rest of the page rendered normally and only the
dropdown was blank.

A check on the JavaScript SOURCE cannot see that: the code that populates the
selector is correct, and the defect is what the control shows when it does not
run. So this file loads the real page, from the real server, into a real DOM,
and looks at what an operator would see.

    initial load          -> the selector VISIBLY shows "Preset scenario"
    >= 2 polling cycles   -> it still does, and a user's edit is not reverted
    degraded payload      -> it still shows a valid, selectable option
    every choice          -> selecting it leaves a visible, valid selection

WHAT IS ASSERTED IS `selectedIndex` AND THE VISIBLE TEXT, not just `.value`.
`select.value = "x"` with no matching option sets `selectedIndex` to -1 and
displays nothing while `.value` reads back as "" — so a test that only read
`.value` would have called the blank control correct.

Needs Node and jsdom. Skipped, not failed, where they are absent:

    npm install jsdom && NODE_PATH=$PWD/node_modules pytest tests/test_dashboard_acquisition_selector_js.py
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(REPO, "web")
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))

from wisepack_core.acquisition import (                            # noqa: E402
    ACQUISITION_ISAAC, ACQUISITION_PLANAR, ACQUISITION_REALSENSE)

NODE = shutil.which("node") or shutil.which("nodejs")
if NODE is None:                                                # pragma: no cover
    pytest.skip("node is not installed", allow_module_level=True)

if subprocess.run([NODE, "-e", "require('jsdom')"],
                  cwd=REPO, capture_output=True).returncode != 0:  # pragma: no cover
    pytest.skip("jsdom is not resolvable — `npm install jsdom` and set NODE_PATH",
                allow_module_level=True)


#: The driver. Loads the page, lets `init()` and the 1 s state poll run, and
#: reports what the selector SHOWS at each sample. Everything it does is what a
#: browser does; nothing about the selector is reimplemented here.
DRIVER = r"""
const { JSDOM } = require("jsdom");
// PASSED IN THE ENVIRONMENT, not in argv: `node -e` does not put a script path
// in `process.argv`, so positional indices differ from a normal invocation and
// are an easy thing to get quietly wrong.
const BASE = process.env.WISEPACK_JS_BASE;
const opts = JSON.parse(process.env.WISEPACK_JS_OPTS || "{}");

function sample(w, tag) {
  const s = w.document.querySelector("#s-acq");
  if (!s) return { tag, present: false };
  const shown = s.selectedOptions && s.selectedOptions[0];
  const note = w.document.querySelector("#s-acq-note");
  return {
    tag, present: true,
    value: s.value,
    selectedIndex: s.selectedIndex,
    shownText: shown ? shown.textContent : null,
    optionValues: [...s.options].map(o => o.value),
    enabledValues: [...s.options].filter(o => !o.disabled).map(o => o.value),
    disabledValues: [...s.options].filter(o => o.disabled).map(o => o.value),
    labels: Object.fromEntries([...s.options].map(o => [o.value, o.textContent])),
    titles: Object.fromEntries([...s.options].map(o => [o.value, o.title])),
    note: note ? note.textContent : "",
    method: methodSample(w),
    button: (w.document.querySelector("#c-reset") || {}).textContent || "",
  };
}

// The perception-method control beside it. WHAT IT OFFERS is the point: a source
// must not present a method that cannot read it.
function methodSample(w) {
  const m = w.document.querySelector("#s-method");
  if (!m) return { present: false };
  return {
    present: true, value: m.value, disabled: m.disabled,
    selectedIndex: m.selectedIndex,
    optionValues: [...m.options].map(o => o.value),
    enabledValues: [...m.options].filter(o => !o.disabled).map(o => o.value),
    shownText: m.selectedOptions[0] ? m.selectedOptions[0].textContent : null,
  };
}

//: "before" until the driver flips it, so a capability that appears or
//: disappears while the page is open is exercised WITHOUT a reload — which is
//: the whole point of asking the backend on every poll.
let PHASE = "before";

(async () => {
  const html = await (await fetch(BASE + "/")).text();
  const problems = [];
  //: Every operator command and acquisition endpoint the page POSTs, in order.
  //: Which one a control sends is the behaviour under test — a button that
  //: calls the planar detector for an RGB-D source looks identical from the
  //: outside until you watch the wire.
  const posted = [];
  const dom = new JSDOM(html, {
    url: BASE + "/", runScripts: "dangerously", resources: "usable",
    pretendToBeVisual: true,
    beforeParse(w) {
      w.fetch = async (u, o) => {
        const url = String(u).startsWith("http") ? String(u) : BASE + u;
        if (o && String(o.method).toUpperCase() === "POST") {
          let name = url.replace(BASE, "");
          try { const body = JSON.parse(o.body); if (body.command) name = body.command; }
          catch (e) { /* not a command envelope */ }
          posted.push(name);
        }
        // A STUBBED COMMAND PATH, only when a case asks for it: the rule under
        // test is the FRONTEND's (a poll must not revert an edit), and on a
        // deployment with one available acquisition the backend would rightly
        // refuse the others. The selector logic is the real one either way.
        if (opts.okCommands && url.includes("/api/command")) {
          return new Response(JSON.stringify({ ok: true }),
                              { status: 200, headers: { "content-type": "application/json" } });
        }
        const res = await fetch(url, o);
        const rewriting = opts.state || opts.lateEnable || opts.lateDisable;
        if (!url.includes("/api/state") || !rewriting) return res;
        const doc = await res.json();
        if (opts.state === "strip") { delete doc.acquisition_choice; }
        else if (opts.state === "empty") {
          doc.acquisition_choice = { options: [], selected: "", current: "" };
        } else if (doc.acquisition_choice) {
          const ac = doc.acquisition_choice;
          const only = (allowed) => {
            for (const o of ac.options) {
              const ok = o.value === "preset" || allowed.includes(o.value);
              o.available = ok;
              o.reason = ok ? "" : (o.reason || `${o.value} is not connected`);
            }
            ac.available = ac.options.filter(o => o.available).map(o => o.value);
          };
          if (opts.state === "all-available") only(ac.options.map(o => o.value));
          if (opts.state === "none-available") only([]);
          // ONE capability appears mid-session; the other two stay as they were,
          // which makes this a test of INDEPENDENCE as well as of the refresh.
          if (opts.lateEnable) only(PHASE === "after" ? [opts.lateEnable] : []);
          if (opts.lateDisable && PHASE === "after") only([]);
        }
        return new Response(JSON.stringify(doc),
                            { status: 200, headers: { "content-type": "application/json" } });
      };
      w.WebSocket = function () { this.close = () => {}; };
      w.confirm = () => true;
      w.localStorage.clear();
      if (opts.storedDraft) {
        w.localStorage.setItem("wisepack-draft", JSON.stringify(opts.storedDraft));
      }
    },
  });
  const w = dom.window;
  w.addEventListener("error", e => problems.push(
    "pageerror: " + ((e.error && e.error.message) || e.message)));
  const realError = w.console.error.bind(w.console);
  w.console.error = (...a) => { problems.push("console.error: " + a.join(" ")); realError(...a); };

  const samples = [];
  // init() plus the first state fetch.
  await new Promise(r => setTimeout(r, 2000));
  samples.push(sample(w, "load"));

  if (opts.selectValue) {
    const s = w.document.querySelector("#s-acq");
    s.value = opts.selectValue;
    s.dispatchEvent(new w.Event("change", { bubbles: true }));
    // PRESSED IMMEDIATELY when a case asks for it: the reported failure was a
    // button that used the PREVIOUS source because no poll had landed yet.
    if (opts.clickImmediately) w.document.querySelector("#c-reset").click();
    await new Promise(r => setTimeout(r, opts.clickImmediately ? 3500 : 400));
    samples.push(sample(w, "after-edit"));
  }
  if (opts.clickAfterSettle) {
    posted.length = 0;
    w.document.querySelector("#c-reset").click();
    await new Promise(r => setTimeout(r, 3500));
    samples.push(sample(w, "after-click"));
  }

  // AT LEAST TWO MORE POLLING CYCLES. The poll runs every second, and reverting
  // a selection one second after it was made is the exact failure this guards.
  await new Promise(r => setTimeout(r, 1400));
  samples.push(sample(w, "poll-1"));

  if (opts.lateEnable || opts.lateDisable) {
    // NO RELOAD. The page stays exactly where it is; only the capability the
    // backend reports changes, and the next poll must pick it up.
    PHASE = "after";
    await new Promise(r => setTimeout(r, 1600));
    samples.push(sample(w, "after-capability-change"));
  }

  await new Promise(r => setTimeout(r, 1400));
  samples.push(sample(w, "poll-2"));

  console.log(JSON.stringify({ samples, problems, posted }));
  process.exit(0);
})().catch(e => { console.log(JSON.stringify({ fatal: String(e && e.stack || e) })); process.exit(0); });
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakePerceptionService:
    """A perception service that answers /health and nothing else.

    WHY A FAKE RATHER THAN A STUB IN THE BROWSER. The rule under test is that a
    poll does not revert an edit, and an edit is only kept when the BACKEND
    accepts it — a selector that held a value the server refused would be the
    very bug the snap-back exists to prevent. So the deployment is made to
    genuinely offer a second acquisition, and the command path is the real one.

    It never detects anything. Availability is all these tests need, and a fake
    that produced observations would be a second detector.
    """

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading

        document = {
            "source": "camera", "service_reachable": True,
            "camera_available": True, "model_available": True,
            "model_loaded": True, "provider": "fasterrcnn_bottle",
            "calibration_status": "valid", "calibration_valid": True,
        }

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):                                    # noqa: N802
                body = json.dumps(document).encode()
                self.send_response(200 if self.path.startswith("/health") else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):                       # noqa: D102
                pass

        self.server = HTTPServer(("127.0.0.1", _free_port()), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class _Server:
    """A sim-mode dashboard on its own port, self-contained.

    Pointed at a perception service that does not exist and with the simulated
    RGB-D backend disabled, so "which acquisitions are available" is decided by
    this fixture rather than by whatever happens to be running on the machine.
    """

    def __init__(self, perception_url: str = "") -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        env = dict(os.environ, WISEPACK_STEP_PERIOD_S="0.35",
                   WISEPACK_DISABLE_SIMULATED_RGBD="1",
                   WISEPACK_PERCEPTION_SERVICE_URL=(
                       perception_url or f"http://127.0.0.1:{_free_port()}"))
        env.pop("WISEPACK_PERCEPTION_SOURCE", None)
        self.proc = subprocess.Popen(
            [sys.executable, "app.py", "--source", "sim", "--port", str(self.port),
             "--preset", "mixed_pipes_dense", "--seed", "42"],
            cwd=WEB, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self._await_ready()

    def _await_ready(self, timeout: int = 90) -> None:
        import urllib.request
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                out = self.proc.stdout.read().decode(errors="replace")
                raise RuntimeError(f"dashboard exited early:\n{out}")
            try:
                with urllib.request.urlopen(f"{self.url}/healthz", timeout=2):
                    return
            except Exception:                                    # noqa: BLE001
                time.sleep(1)
        raise RuntimeError("dashboard did not become ready")

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:                        # pragma: no cover
            self.proc.kill()


@pytest.fixture(scope="module")
def server():
    if not shutil.which(sys.executable):                         # pragma: no cover
        pytest.skip("no interpreter to run the dashboard with")
    try:
        import fastapi                                           # noqa: F401
    except ImportError:                                          # pragma: no cover
        pytest.skip("web/app.py needs FastAPI")
    instance = _Server()
    yield instance
    instance.close()


@pytest.fixture(scope="module")
def server_with_camera():
    """A deployment where TWO acquisitions are genuinely available."""
    try:
        import fastapi                                           # noqa: F401
    except ImportError:                                          # pragma: no cover
        pytest.skip("web/app.py needs FastAPI")
    fake = _FakePerceptionService()
    instance = _Server(perception_url=fake.url)
    yield instance
    instance.close()
    fake.close()


def _drive(server, **options) -> dict:
    """Load the page in a real DOM and report what the selector showed."""
    completed = subprocess.run(
        [NODE, "-e", DRIVER], cwd=REPO, capture_output=True, text=True,
        timeout=180,
        env=dict(os.environ, WISEPACK_JS_BASE=server.url,
                 WISEPACK_JS_OPTS=json.dumps(options)))
    line = next((l for l in reversed(completed.stdout.splitlines())
                 if l.startswith("{")), "")
    assert line, (f"the driver produced no result\n"
                  f"stdout:\n{completed.stdout[-2000:]}\n"
                  f"stderr:\n{completed.stderr[-2000:]}")
    document = json.loads(line)
    assert "fatal" not in document, document["fatal"]
    return document


def _by_tag(document: dict, tag: str) -> dict:
    for entry in document["samples"]:
        if entry["tag"] == tag:
            return entry
    raise AssertionError(f"no sample {tag!r} in {document['samples']}")


# --------------------------------------------------------------------------- #
# 1. It shows "Preset scenario" on an ordinary startup
# --------------------------------------------------------------------------- #


def test_the_selector_shows_preset_scenario_on_load(server):
    """THE REGRESSION. It rendered blank; it must read "Preset scenario"."""
    document = _drive(server)
    load = _by_tag(document, "load")
    assert load["present"], "the selector is not in the page at all"
    assert load["shownText"] == "Preset scenario", (
        f"the selector shows {load['shownText']!r} on startup; an operator "
        f"sees {'a blank box' if load['selectedIndex'] < 0 else 'the wrong option'}")
    assert load["value"] == "preset"
    assert load["selectedIndex"] >= 0, (
        "selectedIndex is -1 — the control displays nothing, which is what a "
        "value with no matching option does")
    assert not document["problems"], document["problems"]


def test_the_four_choices_are_offered(server):
    document = _drive(server)
    load = _by_tag(document, "load")
    assert load["optionValues"] == [
        "preset", "planar_webcam", "realsense_d435", "isaac_simulated"]


def test_the_selection_survives_two_polling_cycles(server):
    """A poll one second later must not blank or change what is shown."""
    document = _drive(server)
    for tag in ("load", "poll-1", "poll-2"):
        entry = _by_tag(document, tag)
        assert entry["shownText"] == "Preset scenario", (
            f"at {tag} the selector shows {entry['shownText']!r}")
        assert entry["selectedIndex"] >= 0, f"blank at {tag}"


# --------------------------------------------------------------------------- #
# 2. A degraded state document must not blank the control
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("state", ["strip", "empty"])
def test_a_state_without_acquisition_choice_still_shows_a_valid_option(server, state):
    """THE ROOT CAUSE. `#s-acq` was the only visible selector whose entire
    contents came from the backend, so a state document without
    `acquisition_choice` left an empty box — and `panel()` swallowed the failure,
    so the rest of the page looked fine."""
    document = _drive(server, state=state)
    for tag in ("load", "poll-1", "poll-2"):
        entry = _by_tag(document, tag)
        assert entry["selectedIndex"] >= 0, (
            f"the selector is blank at {tag} with a {state} state document")
        assert entry["shownText"] == "Preset scenario"
        assert entry["value"] in entry["optionValues"], (
            "a value was assigned that is not one of the options — the control "
            "would display nothing")


# --------------------------------------------------------------------------- #
# 3. Draft ownership: a poll must not revert an edit
# --------------------------------------------------------------------------- #


def test_a_user_edit_is_not_reverted_by_polling(server_with_camera):
    """NOTHING IS STUBBED. The deployment genuinely offers two acquisitions, so
    the real `set_acquisition` accepts the switch and the poll that follows is
    the real one. An edit reverted a second later is the failure this guards."""
    document = _drive(server_with_camera, selectValue="planar_webcam")
    edited = _by_tag(document, "after-edit")
    assert edited["value"] == "planar_webcam", (
        f"the edit did not take: {edited['shownText']!r}")
    for tag in ("poll-1", "poll-2"):
        entry = _by_tag(document, tag)
        assert entry["value"] == "planar_webcam", (
            f"a poll reverted the operator's selection at {tag} — it now shows "
            f"{entry['shownText']!r}")
        assert entry["selectedIndex"] >= 0


@pytest.mark.parametrize("choice", ["preset", "planar_webcam"])
def test_an_available_choice_leaves_a_visible_valid_selection(
        server_with_camera, choice):
    """Both acquisitions this deployment can actually do, end to end."""
    document = _drive(server_with_camera, selectValue=choice)
    for tag in ("after-edit", "poll-1", "poll-2"):
        entry = _by_tag(document, tag)
        assert entry["selectedIndex"] >= 0, f"{choice} left the control blank at {tag}"
        assert entry["value"] == choice
        assert entry["value"] in entry["enabledValues"], (
            "the shown option is disabled — it cannot be what the next run uses")


@pytest.mark.parametrize("choice", ["realsense_d435", "isaac_simulated"])
def test_an_unavailable_choice_leaves_a_visible_valid_selection(
        server_with_camera, choice):
    """§5 and §6 together. Neither RGB-D device exists in this deployment, so
    the backend refuses and the control snaps back — to a REAL, selectable
    option, never to a blank box and never to the refused value."""
    document = _drive(server_with_camera, selectValue=choice)
    for tag in ("after-edit", "poll-1", "poll-2"):
        entry = _by_tag(document, tag)
        assert entry["selectedIndex"] >= 0, f"{choice} left the control blank at {tag}"
        assert entry["value"] in entry["enabledValues"], (
            f"after refusing {choice} the control shows {entry['value']!r}, "
            f"which is not selectable; enabled are {entry['enabledValues']}")


# --------------------------------------------------------------------------- #
# 4. A stored draft is restored, and never as an unknown value
# --------------------------------------------------------------------------- #


def test_a_stored_draft_from_before_this_feature_still_renders(server):
    """A returning browser holds a draft with no `acquisition` key at all."""
    document = _drive(server, storedDraft={
        "preset": "mixed_pipes_dense", "seed": 42, "strategy": "max_density",
        "object_source": "sim", "perception_method": "planar_fasterrcnn"})
    load = _by_tag(document, "load")
    assert load["shownText"] == "Preset scenario"
    assert load["selectedIndex"] >= 0


def test_a_stored_draft_naming_an_unavailable_device_is_not_assigned(server):
    """§6: no silent fall back to an unknown or unselectable value.

    The stored device is not available in this deployment, so it must NOT be put
    on the control — and the control must still show something valid rather than
    going blank.
    """
    document = _drive(server, storedDraft={
        "preset": "mixed_pipes_dense", "object_source": "camera",
        "acquisition": "realsense_d435"})
    load = _by_tag(document, "load")
    assert load["selectedIndex"] >= 0, "an unavailable stored device blanked it"
    assert load["value"] in load["enabledValues"], (
        f"{load['value']!r} is not selectable in this deployment; enabled "
        f"options are {load['enabledValues']}")


def test_a_stored_draft_naming_an_unknown_device_is_ignored(server):
    """A value that is not one of the four must never reach the control."""
    document = _drive(server, storedDraft={
        "preset": "mixed_pipes_dense", "acquisition": "lidar_from_the_future"})
    load = _by_tag(document, "load")
    assert load["value"] != "lidar_from_the_future"
    assert load["selectedIndex"] >= 0
    assert load["value"] in load["optionValues"]


# --------------------------------------------------------------------------- #
# 5. All four modes are ALWAYS represented — disabled, never removed
# --------------------------------------------------------------------------- #

#: What WISEPACK supports. A build that supports four ways of acquiring objects
#: must show four, whatever is plugged in: an operator who cannot see "Physical
#: RGB-D camera" cannot tell a build without the capability from a deployment
#: whose camera is unplugged.
ALL_MODES = ["preset", "planar_webcam", "realsense_d435", "isaac_simulated"]


@pytest.mark.parametrize("state", [None, "strip", "empty", "none-available",
                                   "all-available"])
def test_all_four_modes_are_always_visible(server, state):
    """THE REGRESSION. Whatever the backend says — including saying nothing —
    every supported acquisition mode is in the list."""
    document = _drive(server, **({"state": state} if state else {}))
    for tag in ("load", "poll-1", "poll-2"):
        entry = _by_tag(document, tag)
        assert entry["optionValues"] == ALL_MODES, (
            f"at {tag} the selector offers {entry['optionValues']} — a supported "
            "mode was removed rather than disabled")


@pytest.mark.parametrize("state", [None, "strip", "empty", "none-available"])
def test_preset_is_always_enabled(server, state):
    """It needs no service, no camera, no worker and no GPU."""
    document = _drive(server, **({"state": state} if state else {}))
    for tag in ("load", "poll-1", "poll-2"):
        entry = _by_tag(document, tag)
        assert "preset" in entry["enabledValues"], (
            f"at {tag} the preset was disabled; it depends on nothing")


def test_unavailable_modes_are_disabled_and_say_why(server):
    """Visible, not selectable, and carrying a concise reason."""
    document = _drive(server, state="none-available")
    entry = _by_tag(document, "poll-1")
    for mode in ALL_MODES:
        if mode == "preset":
            continue
        assert mode in entry["disabledValues"], (
            f"{mode} is not disabled on a deployment that cannot run it")
        assert "unavailable" in entry["labels"][mode].lower(), (
            f"{mode} reads {entry['labels'][mode]!r} — an operator cannot tell "
            "it apart from an available one")
        assert entry["titles"][mode], f"{mode} carries no reason in its tooltip"
        assert mode.split("_")[0] in entry["note"] or entry["titles"][mode], (
            "the reason must be reachable from the option or the adjacent note")


def test_the_three_capabilities_are_independent(server):
    """A missing webcam must not disable the simulated path, and a missing D435
    must not disable either of the others.

    The deployment behind this fixture has no perception service and no
    simulated RGB-D, so those two are unavailable INDIVIDUALLY — the assertion
    is that each option's state is its own, not a shared "camera" flag.
    """
    document = _drive(server)
    entry = _by_tag(document, "poll-1")
    states = {m: (m in entry["enabledValues"]) for m in ALL_MODES}
    assert states["preset"] is True
    assert states["planar_webcam"] is False, (
        "this fixture points at a perception service that does not exist")
    assert states["isaac_simulated"] is False, (
        "this fixture sets WISEPACK_DISABLE_SIMULATED_RGBD=1")
    # The RealSense answer belongs to the FoundationPose worker alone and is
    # whatever this machine reports — what must NOT happen is it tracking either
    # of the two above, which are both unavailable for unrelated reasons.
    assert entry["titles"]["realsense_d435"] != entry["titles"]["planar_webcam"], (
        "the D435 and the webcam are reporting the same reason — they are not "
        "answering for themselves")


@pytest.mark.parametrize("mode", ["planar_webcam", "realsense_d435",
                                  "isaac_simulated"])
def test_a_capability_appearing_enables_its_option_without_a_reload(server, mode):
    """An operator plugs a camera in, or starts the worker, mid-session.

    The page is never reloaded: only what the backend reports changes, and the
    next poll must make that one option selectable while the other two stay
    exactly as they were.
    """
    document = _drive(server, lateEnable=mode)
    before = _by_tag(document, "poll-1")
    assert mode in before["disabledValues"], f"{mode} started enabled"

    after = _by_tag(document, "after-capability-change")
    assert mode in after["enabledValues"], (
        f"{mode} was still disabled after the backend reported it available — "
        "an operator would have to reload the page")
    assert "unavailable" not in after["labels"][mode].lower()
    for other in ALL_MODES:
        if other in (mode, "preset"):
            continue
        assert other in after["disabledValues"], (
            f"enabling {mode} also enabled {other} — the capabilities are not "
            "independent")


def test_a_selected_mode_going_unavailable_does_not_blank_the_selector(
        server_with_camera):
    """§5. The choice is genuinely made against the real backend, and then its
    capability disappears while the page is open.

    The operator's selection is NOT moved for them — that would be a silent fall
    back to something they did not choose — so the control keeps showing it,
    stays non-blank, keeps all four modes, and the note says it cannot run.
    """
    document = _drive(server_with_camera, selectValue="planar_webcam",
                      lateDisable=True)
    assert _by_tag(document, "after-edit")["value"] == "planar_webcam", (
        "the selection did not take, so the rest of this test proves nothing")

    for tag in ("after-capability-change", "poll-2"):
        entry = _by_tag(document, tag)
        assert entry["selectedIndex"] >= 0, (
            f"the selector went blank at {tag} when the selected mode became "
            "unavailable")
        assert entry["shownText"], f"nothing is displayed at {tag}"
        assert entry["optionValues"] == ALL_MODES
        assert entry["value"] == "planar_webcam", (
            "the selection was silently moved to something the operator did "
            "not choose")
    told = _by_tag(document, "poll-2")["note"].lower()
    assert "cannot run" in told, (
        f"the note does not say the selection cannot run: {told!r}")


# --------------------------------------------------------------------------- #
# 6. Reported in references/Bugs.pdf
# --------------------------------------------------------------------------- #
#
# Each source must present ONLY the perception methods that can read it. The
# method list used to be every method the deployment can run, whatever sat
# beside it — so "Physical RGB camera" offered FoundationPose, which cannot read
# a webcam, and "Physical RGB-D camera" offered the planar detector, which
# cannot use depth. Two selectors could be put into a combination with no
# implementation, and the panel showed it as an ordinary choice.

#: source -> the methods that can read it, from `METHOD_ACQUISITIONS`.
COMPATIBLE = {
    "preset": [],
    ACQUISITION_PLANAR: ["planar_fasterrcnn"],
    ACQUISITION_REALSENSE: ["foundationpose_rgbd"],
    ACQUISITION_ISAAC: ["foundationpose_rgbd"],
}


def test_a_preset_offers_no_perception_method(server):
    """It reads no camera frame, so no method applies — said, not hidden."""
    document = _drive(server)
    method = _by_tag(document, "load")["method"]
    assert method["present"]
    assert method["disabled"] is True
    assert method["shownText"] == "Not applicable"
    assert method["enabledValues"] == [], (
        f"a preset offers {method['enabledValues']}, none of which can run")


@pytest.mark.parametrize("choice", [ACQUISITION_REALSENSE, ACQUISITION_ISAAC])
def test_an_rgbd_source_offers_only_foundationpose(server, choice):
    """It cannot be read by the planar detector, so that must not be offered."""
    document = _drive(server, state="all-available", selectValue=choice)
    method = _by_tag(document, "after-edit")["method"]
    assert method["enabledValues"] == COMPATIBLE[choice], (
        f"{choice} offers {method['enabledValues']}, expected "
        f"{COMPATIBLE[choice]}")
    assert "planar_fasterrcnn" not in method["optionValues"], (
        "the planar detector is offered for a depth source; it cannot use depth")
    assert method["value"] == "foundationpose_rgbd"


def test_the_planar_source_offers_only_faster_rcnn(server_with_camera):
    """FoundationPose cannot read a colour webcam, so it must not be offered."""
    document = _drive(server_with_camera, selectValue=ACQUISITION_PLANAR)
    method = _by_tag(document, "after-edit")["method"]
    assert method["enabledValues"] == ["planar_fasterrcnn"], (
        f"the webcam offers {method['enabledValues']}")
    assert "foundationpose_rgbd" not in method["optionValues"]
    assert method["value"] == "planar_fasterrcnn"


@pytest.mark.parametrize("choice", ["preset", ACQUISITION_REALSENSE,
                                    ACQUISITION_ISAAC])
def test_the_method_selector_is_never_blank(server, choice):
    """Filtering the list must not leave the control showing nothing."""
    document = _drive(server, state="all-available", selectValue=choice)
    method = _by_tag(document, "after-edit")["method"]
    assert method["selectedIndex"] >= 0, f"{choice} left the method blank"
    assert method["shownText"]


# --- the Scenario button must act on the SELECTED source -------------------- #


@pytest.mark.parametrize("choice", [ACQUISITION_REALSENSE, ACQUISITION_ISAAC])
def test_the_scenario_button_never_runs_the_planar_detector_for_rgbd(
        server, choice):
    """REPORTED: "Looks like the RGB – Faster R-CNN was called instead."

    The button branched on a polled `object_source` that knew only "preset or
    camera", so every camera looked alike to it and an RGB-D source ran the
    planar detector.
    """
    document = _drive(server, state="all-available", selectValue=choice,
                      clickAfterSettle=True)
    assert "detect_physical_objects" not in document["posted"], (
        f"the button ran the planar detector for {choice}: {document['posted']}")
    assert any("acquire" in call for call in document["posted"]), (
        f"the button did not reach an acquisition endpoint: {document['posted']}")


def test_switching_to_preset_and_pressing_at_once_generates_a_preset(server):
    """REPORTED: after switching back to Preset, Reset did not produce a preset.

    The button read a snapshot refreshed once a second, so pressing it before
    the next poll used the PREVIOUS source. Pressed here with no poll in
    between, which is what an operator does.
    """
    document = _drive(server, state="all-available", selectValue="preset",
                      clickImmediately=True)
    assert "detect_physical_objects" not in document["posted"], (
        f"a preset reset called the detector: {document['posted']}")
    assert "reset" in document["posted"], (
        f"no reset was sent for a preset source: {document['posted']}")


def test_the_button_label_matches_the_command_it_sends(server_with_camera):
    """A control named for one thing that does another is the whole class of
    bug this file exists for."""
    document = _drive(server_with_camera, selectValue=ACQUISITION_PLANAR,
                      clickAfterSettle=True)
    label = _by_tag(document, "after-click")["button"].lower()
    assert "detect" in label, f"the button reads {label!r}"
    assert "detect_physical_objects" in document["posted"], document["posted"]


# --- the draft must never carry a selector value as a device ---------------- #


def test_the_preset_choice_is_not_stored_as_an_acquisition_device(server):
    """`preset` is an OBJECT SOURCE, not a camera. Sending it as one put the
    string "preset" in the field that names a device, and the acquisition axis
    then reported a label for a device that does not exist."""
    document = _drive(server, selectValue="preset", clickAfterSettle=True)
    import json as _json
    import urllib.request
    with urllib.request.urlopen(server.url + "/api/state", timeout=30) as handle:
        state = _json.load(handle)
    assert state["settings"]["acquisition"] in ("", None), (
        f"settings.acquisition is {state['settings']['acquisition']!r}; a preset "
        "acquires from no device and the honest value is empty")
    assert document["samples"], "the driver produced no samples"
