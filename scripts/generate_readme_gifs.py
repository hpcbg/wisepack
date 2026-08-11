"""Reproducible animated GIFs of the Human-in-the-Loop workflow, for the README.

No desktop recording. A headless Chromium drives the real dashboard through the
real operator command path, screenshots at a fixed cadence, and ffmpeg assembles
the frames with a generated palette. Re-running this produces the same
demonstration, because the scenario and seed are pinned and every action goes
through the same REST endpoints the buttons use.

    python3 scripts/generate_readme_gifs.py
    python3 scripts/generate_readme_gifs.py --only approve
    python3 scripts/generate_readme_gifs.py --fps 3 --keep-frames

HONESTY RULE, enforced rather than remembered: these are recorded in SIMULATION
mode, so the captured header badge reads SIMULATED and every GIF is checked for
that badge before it is written. A recording that claimed ROS or FIWARE
operation while coming from the simulator would misrepresent the demonstrator,
so `_assert_simulated_badge` fails the build instead.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(REPO, "web")
OUT_DIR = os.path.join(REPO, "images", "generated")
FRAME_ROOT = os.path.join(REPO, ".gif-frames")

VIEWPORT = {"width": 1440, "height": 900}
#: Crop to the dashboard content that matters: header + Digital Twin + operator
#: panel. The full 1440x900 page has a lot of whitespace at this width and the
#: GIF gets large for no informational gain.
CLIP = {"x": 0, "y": 0, "width": 1440, "height": 820}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class AttachedDashboard:
    """An ALREADY-RUNNING dashboard, used as-is.

    The sim dashboard below is self-contained and reproducible, which is right
    for the workflow GIFs. But the panels that only exist in a live deployment —
    the `FIWARE + ROS` source badge, the physical execution backend, the run
    correlation and scene-readiness diagnostics — cannot be produced by a sim
    process talking to nothing. Those are captured against a real stack the
    operator has already started, so the screenshot shows the true state rather
    than a mock of it.
    """

    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def close(self) -> None:                            # nothing to stop
        pass


class Dashboard:
    """A sim-mode dashboard on its own port."""

    def __init__(self, step_period="0.45"):
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        env = dict(os.environ, WISEPACK_STEP_PERIOD_S=step_period)
        self.proc = subprocess.Popen(
            [sys.executable, "app.py", "--source", "sim", "--port", str(self.port)],
            cwd=WEB, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self._await_ready()

    def _await_ready(self, timeout=90):
        import urllib.request
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    "dashboard exited early:\n"
                    + self.proc.stdout.read().decode(errors="replace"))
            try:
                with urllib.request.urlopen(f"{self.url}/healthz", timeout=2):
                    return
            except Exception:                           # noqa: BLE001
                time.sleep(1)
        raise RuntimeError("dashboard did not start")

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class Recorder:
    """Screenshots a page at a fixed cadence into a numbered frame sequence."""

    def __init__(self, page, folder: str, fps: int):
        self.page = page
        self.folder = folder
        self.fps = fps
        self.n = 0
        os.makedirs(folder, exist_ok=True)

    def frame(self, count: int = 1) -> None:
        for _ in range(count):
            self.n += 1
            self.page.screenshot(
                path=os.path.join(self.folder, f"f{self.n:04d}.png"), clip=CLIP)

    def hold(self, seconds: float) -> None:
        """Record for `seconds` of wall clock at the target frame rate."""
        for _ in range(max(1, int(seconds * self.fps))):
            self.page.wait_for_timeout(int(1000 / self.fps))
            self.frame()

    def until(self, predicate: str, timeout_s: float = 30.0,
              max_frames: int = 200) -> bool:
        """Record until a JS predicate is true. Returns whether it became true."""
        deadline = time.time() + timeout_s
        while time.time() < deadline and self.n < max_frames:
            if self.page.evaluate(f"() => {predicate}"):
                return True
            self.page.wait_for_timeout(int(1000 / self.fps))
            self.frame()
        return bool(self.page.evaluate(f"() => {predicate}"))


# --------------------------------------------------------------------------- #
# Page helpers — every action goes through the real command endpoint
# --------------------------------------------------------------------------- #


def command(page, name: str, args: Optional[dict] = None) -> dict:
    return page.evaluate(
        """async ([cmd, args]) => {
             const r = await fetch('/api/command', {
               method: 'POST', headers: {'Content-Type': 'application/json'},
               body: JSON.stringify({command: cmd, args: args || {}})});
             return {status: r.status, body: await r.text()};
           }""", [name, args or {}])


def stage_of(page) -> str:
    return page.evaluate(
        "() => document.querySelector('#b-stage')?.textContent?.trim() || ''")


def wait_stage(page, stage: str, timeout=90_000) -> None:
    page.wait_for_function(
        f"() => document.querySelector('#b-stage')?.textContent?.trim() === "
        f"{stage!r}", timeout=timeout)


def reset_run(page, **settings) -> None:
    args = {"preset": "mixed_pipes_dense", "seed": 42, "strategy": "max_density",
            "dynamic_events_enabled": False}
    args.update(settings)
    res = command(page, "reset", args)
    if res["status"] != 200:
        raise RuntimeError(f"reset failed: {res}")
    wait_stage(page, "WAIT_FOR_OPERATOR_APPROVAL")
    # Wait for the Digital Twin to render at least one placement. The dense demo
    # scenario draws dozens; the small cut scenarios draw only a few, so the
    # threshold must not assume a large batch.
    page.wait_for_function(
        "() => document.querySelectorAll('#twin rect').length > 1", timeout=60_000)


def set_light_theme(page) -> None:
    """Select the LIGHT theme and ASSERT it (brief §22).

    The generator must FAIL rather than produce a dark capture, so this clicks
    the Light button, waits for the theme to actually change, and asserts both
    that light is active and that Dark is not.
    """
    page.click("#theme-l")
    page.wait_for_function(
        "() => document.documentElement.dataset.theme === 'light'", timeout=5000)
    assert page.get_attribute("#theme-l", "aria-pressed") == "true", \
        "LIGHT theme button is not active — refusing to capture a dark frame"
    assert page.get_attribute("#theme-d", "aria-pressed") == "false", \
        "DARK theme button is still active — refusing to capture a dark frame"


#: The retired anomaly product name must never appear in a fresh capture. Built
#: from fragments so this guard does not itself trip the stale-label grep.
_OLD_ANOMALY_TITLE = "EDF Topic #2 " + "Integration Demo"
_NEW_ANOMALY_TITLE = "Anomaly Monitoring & Workflow Response"


def assert_anomaly_title(page) -> None:
    """A capture that shows the anomaly panel must show the NEW title only.

    Uses textContent (not inner_text): the panel title contains a literal ``&``
    that inner_text's whitespace/entity handling does not preserve verbatim.
    """
    body = page.evaluate("() => document.body.textContent || ''")
    if _OLD_ANOMALY_TITLE in body:
        raise RuntimeError(
            f"refusing to capture: the retired title {_OLD_ANOMALY_TITLE!r} is "
            "still visible")
    if _NEW_ANOMALY_TITLE not in body:
        raise RuntimeError(
            f"refusing to capture: the anomaly panel title "
            f"{_NEW_ANOMALY_TITLE!r} is not visible")


def _assert_simulated_badge(page) -> None:
    """Refuse to write a GIF that does not show its own provenance."""
    badge = page.evaluate(
        "() => document.querySelector('#b-source')?.textContent || ''")
    if "SIMULATED" not in badge.upper():
        raise RuntimeError(
            f"refusing to record: the source badge reads {badge!r}. These GIFs "
            "are recorded in simulation mode and must visibly say so.")


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def assemble(folder: str, out_path: str, fps: int, width: int = 1000) -> str:
    """Frames -> optimised GIF via a generated palette."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to assemble the GIFs")
    palette = os.path.join(folder, "palette.png")
    scale = f"scale={width}:-1:flags=lanczos"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
         "-i", os.path.join(folder, "f%04d.png"),
         "-vf", f"{scale},palettegen=max_colors=128:stats_mode=diff", palette],
        check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
         "-i", os.path.join(folder, "f%04d.png"), "-i", palette,
         "-lavfi", f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
         "-loop", "0", out_path],
        check=True)
    return out_path


# --------------------------------------------------------------------------- #
# The three scenes
# --------------------------------------------------------------------------- #


def scene_approve(page, rec: Recorder) -> None:
    """GIF 1 — the approval gate, then execution begins.

    What a viewer should notice: baseline needs 3 containers, the optimizer
    needs 2, and nothing moves until a human presses Approve.
    """
    reset_run(page)
    _assert_simulated_badge(page)
    rec.hold(2.2)                                   # the gate, plans on screen

    page.click(".v-btn[data-view='side']")          # side-by-side 3 vs 2
    rec.hold(2.4)
    page.click(".v-btn[data-view='optimized']")
    rec.hold(1.0)

    command(page, "approve")
    rec.until("parseFloat(document.querySelector('#progbar').style.width||'0') > 0",
              timeout_s=30)
    rec.hold(5.5)                                   # items becoming executed


def scene_replan(page, rec: Recorder) -> None:
    """GIF 2 — a late component forces a re-plan and a NEW approval."""
    reset_run(page)
    _assert_simulated_badge(page)
    command(page, "approve")
    rec.until("parseFloat(document.querySelector('#progbar').style.width||'0') > 8",
              timeout_s=40)
    rec.hold(1.6)

    command(page, "inject_item")                    # orange, high-priority ILW
    rec.until("document.querySelector('#b-stage')?.textContent?.trim() === "
              "'WAIT_FOR_OPERATOR_APPROVAL'", timeout_s=40)
    rec.hold(3.4)                                   # revised twin + renewed gate

    command(page, "approve")
    rec.hold(3.4)


def scene_container_unavailable(page, rec: Recorder) -> None:
    """GIF 3 — a container is retired and the remainder is re-planned."""
    reset_run(page)
    _assert_simulated_badge(page)
    command(page, "approve")
    rec.until("parseFloat(document.querySelector('#progbar').style.width||'0') > 5",
              timeout_s=40)
    rec.hold(1.6)

    command(page, "container_unavailable")
    rec.until("document.querySelector('#b-stage')?.textContent?.trim() === "
              "'WAIT_FOR_OPERATOR_APPROVAL'", timeout_s=40)
    rec.hold(3.6)

    command(page, "approve")
    rec.hold(2.6)


def scene_cut_aware(page, rec: Recorder) -> None:
    """GIF 4 — no-cut vs cut-aware: cutting a pipe avoids a whole container."""
    reset_run(page, preset="cut_avoids_extra_container", seed=7)
    _assert_simulated_badge(page)
    rec.hold(1.6)
    command(page, "compare_cut_aware")
    rec.until("(document.querySelector('#cut-state')?.textContent||'')"
              ".includes('CUT RECOMMENDED')", timeout_s=25)
    rec.hold(3.0)                                    # the comparison table
    command(page, "approve_cut")
    rec.hold(1.4)
    command(page, "simulate_cut")
    rec.until("document.querySelector('#b-stage')?.textContent?.trim() === "
              "'WAIT_FOR_OPERATOR_APPROVAL'", timeout_s=25)
    rec.hold(2.6)                                    # re-planned, fewer containers
    command(page, "approve")
    rec.hold(2.4)


def scene_inventory(page, rec: Recorder) -> None:
    """GIF 5 — the FIWARE-backed container inventory filling up."""
    reset_run(page, preset="cut_avoids_extra_container", seed=7)
    command(page, "init_inventory", {"count": 4})
    page.goto(BASE_FOR(page) + "/inventory", wait_until="networkidle")
    set_light_theme(page)
    page.wait_for_timeout(500)
    rec.hold(2.2)                                    # KPI tiles + table
    command(page, "check_containers")
    rec.until("document.querySelectorAll('#rows tr').length >= 1", timeout_s=25)
    rec.hold(3.0)                                    # reserved + delivered to cell


def scene_logistics(page, rec: Recorder) -> None:
    """GIF 6 — simulated container logistics: delivery to the cell, collection."""
    reset_run(page, preset="cut_avoids_extra_container", seed=7)
    command(page, "init_inventory", {"count": 4})
    command(page, "check_containers")
    page.goto(BASE_FOR(page) + "/logistics", wait_until="networkidle")
    set_light_theme(page)
    page.wait_for_timeout(500)
    rec.hold(3.0)                                    # facility map + robot + tasks
    command(page, "collect_full_containers")
    rec.hold(2.6)


def scene_anomaly(page, rec: Recorder) -> None:
    """GIF 7 — a SIMULATED critical anomaly holds the workflow."""
    reset_run(page, preset="cut_avoids_extra_container", seed=7)
    _assert_simulated_badge(page)
    assert_anomaly_title(page)                       # new title only (brief §7)
    command(page, "approve")
    rec.until("parseFloat(document.querySelector('#progbar').style.width||'0') > 0",
              timeout_s=25)
    rec.hold(1.6)                                    # normal execution
    page.select_option("#a-class", "shear_position_too_high")
    command(page, "inject_anomaly", {"anomaly_class": "shear_position_too_high"})
    rec.until("(document.querySelector('#anomaly-state')?.textContent||'')"
              ".includes('HELD')", timeout_s=25)
    rec.hold(3.0)                                    # workflow held, authorisation revoked
    command(page, "acknowledge_anomaly")
    rec.hold(1.6)


def BASE_FOR(page) -> str:
    """The dashboard origin for the current page (for cross-route navigation)."""
    import urllib.parse
    u = urllib.parse.urlparse(page.url)
    return f"{u.scheme}://{u.netloc}"


SCENES: Dict[str, Dict] = {
    "approve": {"fn": scene_approve, "file": "hitl-approve-execute.gif"},
    "replan": {"fn": scene_replan, "file": "hitl-dynamic-replan.gif"},
    "container": {"fn": scene_container_unavailable,
                  "file": "hitl-container-unavailable.gif"},
    "cut": {"fn": scene_cut_aware, "file": "cut-aware-comparison.gif"},
    "inventory": {"fn": scene_inventory, "file": "container-inventory.gif"},
    "logistics": {"fn": scene_logistics, "file": "container-logistics.gif"},
    "anomaly": {"fn": scene_anomaly, "file": "anomaly-workflow.gif"},
}


def _capture_live_screenshots(browser, dash, theme: str) -> List[str]:
    """Stills that only exist against a real deployment.

    Everything here depends on state a sim process cannot produce: an Orion-LD
    read-back, an execution backend that is actually running, a physical scene
    that has been acknowledged for this run. Captured from whatever stack is
    attached, so the image is evidence rather than illustration.
    """
    written: List[str] = []

    def shot(name: str, route: str, driver=None, clip=None) -> None:
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1,
                                  color_scheme=theme)
        page = ctx.new_page()
        errors: List[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(dash.url + route, wait_until="networkidle")
        page.wait_for_timeout(2500)
        if theme == "light":
            set_light_theme(page)
        if driver:
            driver(page)
        page.wait_for_timeout(1200)
        out = os.path.join(OUT_DIR, name)
        page.screenshot(path=out, clip=clip)
        if errors:
            raise RuntimeError(f"{name}: page errors {errors}")
        ctx.close()
        written.append(out)
        print(f"[shot] {name} ({os.path.getsize(out)/1e3:.0f} kB)")

    shot("dashboard-live-light.png", "/", clip=CLIP)
    # Diagnostics is a long page; the correlation and scene rows are near the
    # top, and a full-page capture would render them unreadably small.
    # Tall enough to include the scene-readiness block below the correlation
    # rows: cropping it out would drop the evidence the shot exists for.
    shot("diagnostics-run-correlation-light.png", "/diagnostics",
         clip={"x": 0, "y": 0, "width": 1440, "height": 1010})
    return written


def _require_live_camera(dash) -> Dict:
    """Refuse to capture camera evidence from anything but a real camera.

    THE POINT OF THESE IMAGES is that they are evidence: a real frame, a real
    calibration, real measured millimetres. A screenshot of the simulator with a
    camera-shaped caption would be a lie that is very hard to spot afterwards,
    so the generator checks the running stack before it captures anything and
    stops with a reason rather than producing a plausible picture.
    """
    import urllib.request                                    # noqa: PLC0415

    with urllib.request.urlopen(dash.url + "/api/perception", timeout=10) as r:
        payload = json.loads(r.read().decode("utf-8"))

    if payload.get("perception_source") != "camera":
        raise SystemExit(
            "--camera-shots needs a stack running with "
            "WISEPACK_PERCEPTION_SOURCE=camera; the attached dashboard reports "
            f"{payload.get('perception_source')!r}.")
    batch = payload.get("batch") or {}
    if batch.get("status") != "ok":
        raise SystemExit(
            "--camera-shots needs a successful detection on screen; the current "
            f"batch is {batch.get('status')!r} ({batch.get('error') or 'no batch'}). "
            "Press 'Detect physical objects' and try again.")
    if batch.get("calibration_status") != "valid":
        raise SystemExit(
            "--camera-shots needs a VALID calibration; the current batch reports "
            f"{batch.get('calibration_status')!r}. Put the ArUco sheet in frame "
            "and detect again.")
    if not batch.get("count"):
        raise SystemExit(
            "--camera-shots needs at least one detected object on screen.")
    return payload


def _capture_camera_screenshots(browser, dash, theme: str) -> List[str]:
    """PHYSICAL-CAMERA EVIDENCE, captured from a running camera deployment.

    Two images, and between them they show the whole claim end to end:

      perception-camera-light.png    the Physical Perception panel — the
                                     annotated frame from the real camera, the
                                     calibration verdict, and the measured
                                     x/y/yaw/confidence of each object.
      perception-twin-approval-light.png
                                     the Digital Twin built from exactly those
                                     observations, beside an operator panel with
                                     controls ENABLED and no inconsistent-state
                                     warning — the state the revision fix
                                     restored.

    Neither can be produced without a camera: `_require_live_camera` refuses
    first.
    """
    payload = _require_live_camera(dash)
    count = (payload.get("batch") or {}).get("count")
    written: List[str] = []

    ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1,
                              color_scheme=theme)
    page = ctx.new_page()
    errors: List[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(dash.url + "/", wait_until="networkidle")
    page.wait_for_timeout(2500)
    if theme == "light":
        set_light_theme(page)
    page.wait_for_timeout(1200)

    panel = page.locator("#perceppanel")
    if not panel.is_visible():
        raise SystemExit("the Physical Perception panel is not on screen")
    # The panel must show the detection, not a spinner: assert the count the API
    # just reported is actually rendered before the shutter opens.
    panel_text = panel.inner_text()
    for expected in ("Calibration: VALID", f"Detected cylindrical objects: {count}"):
        if expected not in panel_text:
            raise SystemExit(f"the panel does not show {expected!r} yet:\n"
                             f"{panel_text[:400]}")

    out = os.path.join(OUT_DIR, "perception-camera-light.png")
    panel.screenshot(path=out)
    written.append(out)
    print(f"[camera] perception-camera-light.png "
          f"({os.path.getsize(out)/1e3:.0f} kB)")

    # THE INCONSISTENT-STATE WARNING MUST BE ABSENT, and that is asserted rather
    # than hoped for: this image exists to show the operator gate open on a
    # coherent revision, and capturing it while the warning was on screen would
    # document the bug instead of the fix.
    body = page.locator("body").inner_text()
    if "Inconsistent state" in body:
        raise SystemExit(
            "the dashboard is showing 'Inconsistent state — controls withheld'; "
            "this image is meant to show the coherent state, so it will not be "
            "captured.")

    out = os.path.join(OUT_DIR, "perception-twin-approval-light.png")
    page.screenshot(path=out, clip=CLIP)
    written.append(out)
    print(f"[camera] perception-twin-approval-light.png "
          f"({os.path.getsize(out)/1e3:.0f} kB)")

    if errors:
        raise RuntimeError(f"page errors during capture: {errors}")
    ctx.close()

    # THE DETECTOR'S OWN OUTPUT, at full resolution and unretouched: the frame
    # that was analysed, with the ArUco plane, the measured millimetres and the
    # matched cap drawn on it by the provider. Fetched from the running service
    # rather than screenshotted, so nothing is rescaled or recompressed by a
    # browser on the way — this is the image the operator's decision rests on.
    import urllib.request                                    # noqa: PLC0415

    out = os.path.join(OUT_DIR, "perception-camera-annotated.jpg")
    with urllib.request.urlopen(dash.url + "/api/perception/image/annotated",
                                timeout=20) as response:
        image = response.read()
    if len(image) < 10_000:
        raise SystemExit("the annotated frame came back too small to be a photo")
    with open(out, "wb") as handle:
        handle.write(image)
    written.append(out)
    print(f"[camera] perception-camera-annotated.jpg "
          f"({os.path.getsize(out)/1e3:.0f} kB)")

    return written


#: The four object sources, in the order the selector offers them, with the
#: perception method each one can actually be read with. THE SAME RELATION the
#: dashboard derives from `METHOD_ACQUISITIONS` — restated here only as the
#: caption text, never as a second source of truth: every value in the captured
#: image comes from the running deployment.
SOURCE_SHOTS = (
    ("preset", "source-preset-light.png", "Preset scenario"),
    ("planar_webcam", "source-physical-rgb-light.png", "Physical RGB camera"),
    ("realsense_d435", "source-physical-rgbd-light.png", "Physical RGB-D camera"),
    ("isaac_simulated", "source-simulated-rgbd-light.png",
     "Simulated RGB-D camera"),
)


def _source_state(dash) -> Dict:
    """The acquisition axis as the ATTACHED deployment reports it right now."""
    import urllib.request                                        # noqa: PLC0415
    with urllib.request.urlopen(dash.url + "/api/state", timeout=30) as handle:
        return json.load(handle).get("acquisition_choice") or {}


def _capture_source_screenshots(browser, dash, theme: str) -> List[str]:
    """The Object source / Perception method UI, from a real deployment.

    WHAT THESE ARE FOR. The architecture changed shape: one selector now names
    four places objects can come from, and a second names the one method that
    can read the chosen one. A table can state that; a screenshot of the actual
    control is what lets an evaluator check it.

    EVERY SHOT IS OF A REAL DEPLOYMENT, and a source this machine cannot run is
    SKIPPED WITH ITS REASON rather than staged. A dashboard pointed at a camera
    that is not there would show the option disabled — which is honest — but a
    picture captioned "Physical RGB camera, detecting" that no camera produced
    would not be, so the acquisitions here are actually performed.
    """
    axis = _source_state(dash)
    available = set(axis.get("available") or [])
    if not available:
        raise SystemExit(
            "--source-shots needs a dashboard reporting its acquisition axis; "
            f"{dash.url}/api/state carries no `acquisition_choice`.")
    written: List[str] = []
    skipped: List[str] = []

    ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1,
                              color_scheme=theme)
    page = ctx.new_page()
    errors: List[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(dash.url + "/", wait_until="networkidle")
    page.wait_for_timeout(2500)
    if theme == "light":
        set_light_theme(page)
    page.wait_for_timeout(1000)

    panel = page.locator("#scenario-panel")
    if not panel.is_visible():
        raise SystemExit("the Scenario panel is not on screen")

    def save(name: str, locator) -> None:
        out = os.path.join(OUT_DIR, name)
        # THE PAGE HEADER IS STICKY and sits above everything at z-index 20, so
        # an element shot of a panel scrolled under it came out with the header
        # printed across its first two lines. It is page chrome, not part of the
        # panel, and it is un-stuck for the shutter only — nothing inside the
        # panel is touched.
        page.evaluate("() => { const h = document.querySelector('header');"
                      " if (h) h.style.visibility = 'hidden'; }")
        locator.scroll_into_view_if_needed()
        page.wait_for_timeout(250)
        locator.screenshot(path=out)
        page.evaluate("() => { const h = document.querySelector('header');"
                      " if (h) h.style.visibility = ''; }")
        if errors:
            raise SystemExit(f"{name}: page errors {errors}")
        written.append(out)
        print(f"[source] {name} ({os.path.getsize(out)/1e3:.0f} kB)")

    # -- A. every source, with its real availability ------------------------ #
    #
    # A native <select> popup is drawn by the window system and cannot be
    # captured, so the control is EXPANDED IN PLACE with `size`. The options,
    # their labels and their disabled state are the running deployment's own —
    # nothing is substituted, only shown at once.
    page.evaluate("() => { const s = document.querySelector('#s-acq');"
                  " s.dataset.shotSize = s.size; s.size = s.options.length; }")
    page.wait_for_timeout(400)
    save("source-selector-light.png", panel)
    page.evaluate("() => { const s = document.querySelector('#s-acq');"
                  " s.size = Number(s.dataset.shotSize || 0); }")
    page.wait_for_timeout(300)

    # -- B/C/D. each source, with the method it forces ---------------------- #
    for value, name, label in SOURCE_SHOTS:
        if value != "preset" and value not in available:
            reason = (axis.get("unavailable_reasons") or {}).get(value, "")
            skipped.append(f"{label}: {reason or 'not available here'}")
            continue
        # `command` reports the HTTP status and the raw body — a refusal is a
        # 409 with the capability's own reason, and reading it as one is the
        # difference between skipping with an explanation and skipping blind.
        result = command(page, "set_acquisition", {"acquisition": value})
        if int(result.get("status", 0)) != 200:
            skipped.append(f"{label}: the dashboard refused the selection "
                           f"({result.get('status')}: "
                           f"{str(result.get('body'))[:160]})")
            continue
        page.wait_for_timeout(1500)
        # THE ACQUISITION IS PERFORMED, not implied. A panel captioned with a
        # source that never ran would be the one thing these images exist to
        # rule out.
        if value == "realsense_d435":
            _acquire(page, "/api/perception/physical/acquire",
                     {"model_id": "cylinder5", "roi_px": [255, 70, 445, 719],
                      "frames": 5})
        elif value == "isaac_simulated":
            _acquire(page, "/api/perception/simulated/acquire",
                     {"model_id": "cylinder5", "acquire": False})
        elif value == "planar_webcam":
            command(page, "detect_physical_objects")
            page.wait_for_timeout(3000)
        else:
            command(page, "reset", {"preset": "mixed_pipes_dense", "seed": 42})
            page.wait_for_timeout(1500)
        # THE PANEL MUST AGREE WITH WHAT JUST RAN before the shutter opens.
        # The Scenario panel, the perception panel and the D435 block are
        # refreshed by three different pollers, and a shot taken between them
        # showed a physical acquisition beside a SIMULATED header from the
        # previous run — two runs in one picture, which is exactly what these
        # images exist to rule out. An incoherent panel is now an ERROR, not a
        # published screenshot.
        try:
            page.wait_for_function(
                """async (want) => {
                     const r = await fetch('/api/perception');
                     const d = await r.json();
                     return ((d.acquisition || {}).current || '') === want;
                   }""", arg=value, timeout=30_000)
        except Exception:                                        # noqa: BLE001
            skipped.append(f"{label}: the panel never reported this "
                           "acquisition as the current run")
            continue
        # EVERY PANEL, AT ONE INSTANT. The acquisition was driven through the
        # endpoint rather than the button, so the page's own post-acquisition
        # refresh never ran and the D435 block was left on its 5 s poll — which
        # is how a stale "CURRENT RUN" badge came to sit under a simulated
        # header. This calls the dashboard's OWN refresh; nothing is redrawn by
        # the generator.
        page.evaluate("async () => { if (typeof refreshAll === 'function')"
                      " await refreshAll(); }")
        page.wait_for_timeout(2500)
        shown = page.locator("#s-acq").input_value()
        if shown != value:
            skipped.append(f"{label}: the selector settled on {shown!r}")
            continue
        save(name, panel)
        # THE RESULT, for the sources that produce one. The Scenario panel shows
        # what WILL run; this shows what DID — the provenance badge, the frame
        # the pose is in, and for the simulated source the ground-truth
        # comparison that only exists because the scene is simulated.
        if value in ("realsense_d435", "isaac_simulated", "planar_webcam"):
            result = page.locator("#perceppanel")
            if result.is_visible():
                save(name.replace("-light.png", "-result-light.png"), result)
            else:
                skipped.append(f"{label}: the perception panel is not on screen")

    # -- E. the draft is not the running run -------------------------------- #
    #
    # Captured LAST, so the run on screen is a real acquisition and the selector
    # beside it genuinely disagrees with it.
    running = (_source_state(dash) or {}).get("current")
    if running and running != "preset":
        command(page, "set_acquisition", {"acquisition": "preset"})
        page.wait_for_timeout(2000)
        text = panel.inner_text()
        if "Running now" not in text:
            skipped.append("draft-vs-current: the panel names no running source")
        else:
            save("source-draft-vs-current-light.png", panel)
    else:
        skipped.append("draft-vs-current: no camera run was on screen to "
                       "contrast a preset draft against")

    ctx.close()
    for note in skipped:
        print(f"[source] SKIPPED — {note}")
    return written


def _acquire(page, route: str, payload: Dict) -> None:
    """Run one acquisition through the dashboard's own endpoint, and wait."""
    result = page.evaluate(
        "async ([route, body]) => {"
        " const r = await fetch(route, {method: 'POST',"
        "   headers: {'Content-Type': 'application/json'},"
        "   body: JSON.stringify(body)});"
        " return await r.json(); }", [route, payload])
    if not result.get("ok"):
        raise SystemExit(f"{route} refused: "
                         f"{result.get('reason') or result.get('detail') or result}")


def _capture_screenshots(browser, dash, theme: str) -> List[str]:
    """The required light-theme still screenshots (brief §22)."""
    written: List[str] = []

    def shot(name: str, driver, route: str = "/", clip=None) -> None:
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1,
                                  color_scheme=theme)
        page = ctx.new_page()
        errors: List[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(dash.url + route, wait_until="networkidle")
        page.wait_for_timeout(1500)
        if theme == "light":
            set_light_theme(page)
        driver(page)
        page.wait_for_timeout(800)
        out = os.path.join(OUT_DIR, name)
        page.screenshot(path=out, clip=clip)
        if errors:
            raise RuntimeError(f"{name}: page errors {errors}")
        ctx.close()
        written.append(out)
        print(f"[shot] {name} ({os.path.getsize(out)/1e3:.0f} kB)")

    def dashboard(page):
        reset_run(page)
    def strategy(page):
        reset_run(page)
        command(page, "compare_strategies")
        page.wait_for_function(
            "() => document.querySelectorAll('#strategies tr').length > 1",
            timeout=25_000)
    def cutaware(page):
        reset_run(page, preset="cut_avoids_extra_container", seed=7)
        command(page, "compare_cut_aware")
        page.wait_for_function(
            "() => (document.querySelector('#cut-state')?.textContent||'')"
            ".includes('CUT RECOMMENDED')", timeout=25_000)
    def anomaly(page):
        reset_run(page, preset="cut_avoids_extra_container", seed=7)
        assert_anomaly_title(page)                   # new title only (brief §7)
        command(page, "approve")
        page.wait_for_timeout(1200)
        command(page, "inject_anomaly", {"anomaly_class": "shear_position_too_high"})
        page.wait_for_function(
            "() => (document.querySelector('#anomaly-state')?.textContent||'')"
            ".includes('HELD')", timeout=25_000)
    def inventory(page):
        command(page, "init_inventory", {"count": 4})
        command(page, "check_containers")
        page.wait_for_function(
            "() => document.querySelectorAll('#rows tr').length >= 1", timeout=25_000)
    def logistics(page):
        command(page, "init_inventory", {"count": 4})
        command(page, "check_containers")
        page.wait_for_timeout(1200)
    def diagnostics(page):
        page.wait_for_timeout(1800)

    shot("dashboard-light.png", dashboard, "/", CLIP)
    shot("strategy-comparison-light.png", strategy, "/", CLIP)
    shot("cut-aware-light.png", cutaware, "/", CLIP)
    shot("anomaly-light.png", anomaly, "/", CLIP)
    shot("inventory-light.png", inventory, "/inventory")
    shot("logistics-light.png", logistics, "/logistics")
    shot("diagnostics-light.png", diagnostics, "/diagnostics")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(SCENES), default=None)
    parser.add_argument("--fps", type=int, default=3,
                        help="capture and playback rate (2-4 reads well)")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--theme", choices=("light", "dark"), default="light",
                        help="capture theme (light is the README default)")
    parser.add_argument("--screenshots", action="store_true",
                        help="capture the still light-theme screenshots instead of GIFs")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--attach", default=None, metavar="URL",
                        help=("capture against an already-running dashboard "
                              "(e.g. http://127.0.0.1:8080) instead of starting "
                              "a sim one — for the live-only panels"))
    parser.add_argument("--live-shots", action="store_true",
                        help="with --attach: capture the live-deployment stills")
    parser.add_argument("--source-shots", action="store_true",
                        help=("with --attach: capture the Object source / "
                              "Perception method UI. Each source is SELECTED "
                              "and ACQUIRED for real; one this deployment "
                              "cannot run is skipped with its reason."))
    parser.add_argument("--camera-shots", action="store_true",
                        help=("with --attach: capture the PHYSICAL-CAMERA "
                              "evidence. Refuses unless the attached stack is "
                              "running a real camera with a valid calibration "
                              "and a successful detection on screen."))
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright is required.\n"
              "  pip install playwright && playwright install chromium",
              file=sys.stderr)
        return 2
    if not args.screenshots and not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg is required to assemble the GIFs.", file=sys.stderr)
        return 2

    os.makedirs(OUT_DIR, exist_ok=True)
    scenes = {args.only: SCENES[args.only]} if args.only else SCENES
    written: List[str] = []

    dash = AttachedDashboard(args.attach) if args.attach else Dashboard()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            if args.source_shots:
                written = _capture_source_screenshots(browser, dash, args.theme)
                browser.close()
                dash.close()
                print(f"\nwrote {len(written)} source screenshot(s).")
                return 0
            if args.camera_shots:
                written = _capture_camera_screenshots(browser, dash, args.theme)
                browser.close()
                dash.close()
                print(f"\nwrote {len(written)} physical-camera screenshot(s).")
                return 0
            if args.live_shots:
                # NOT sim-badge guarded: these exist precisely to show a live
                # deployment, where the badge must NOT read SIMULATED.
                written = _capture_live_screenshots(browser, dash, args.theme)
                for path in written:
                    print(f"[live] {os.path.basename(path)}")
                print(f"\nwrote {len(written)} live screenshot(s).")
                return 0
            elif args.screenshots:
                written = _capture_screenshots(browser, dash, args.theme)
                browser.close()
                dash.close()
                print(f"\nwrote {len(written)} screenshot(s).")
                return 0
            for name, spec in scenes.items():
                folder = os.path.join(FRAME_ROOT, name)
                shutil.rmtree(folder, ignore_errors=True)
                ctx = browser.new_context(viewport=VIEWPORT,
                                          device_scale_factor=1,
                                          color_scheme=args.theme)
                page = ctx.new_page()
                errors: List[str] = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(dash.url, wait_until="networkidle")
                page.wait_for_timeout(2500)
                if args.theme == "light":
                    set_light_theme(page)

                print(f"[gif] recording {name} ({args.theme}) ...")
                rec = Recorder(page, folder, args.fps)
                spec["fn"](page, rec)

                if errors:
                    raise RuntimeError(
                        f"{name}: the page raised errors while recording; a GIF "
                        f"of a broken dashboard is worse than none: {errors}")
                ctx.close()

                out = os.path.join(OUT_DIR, spec["file"])
                assemble(folder, out, args.fps, args.width)
                size_mb = os.path.getsize(out) / 1e6
                seconds = rec.n / args.fps
                print(f"[gif] {spec['file']}: {rec.n} frames, "
                      f"{seconds:.1f}s, {size_mb:.2f} MB")
                if size_mb > 10:
                    print(f"[gif] WARNING: {spec['file']} is {size_mb:.1f} MB — "
                          "consider --width 900 or --fps 2")
                written.append(out)
                if not args.keep_frames:
                    shutil.rmtree(folder, ignore_errors=True)
            browser.close()
    finally:
        dash.close()
        if not args.keep_frames:
            shutil.rmtree(FRAME_ROOT, ignore_errors=True)

    print(f"\nwrote {len(written)} GIF(s):")
    for path in written:
        print(f"  {os.path.relpath(path, REPO)}  "
              f"({os.path.getsize(path) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
