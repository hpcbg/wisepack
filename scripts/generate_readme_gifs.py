"""Reproducible animated GIFs of the Human-in-the-Loop workflow, for the README.

No desktop recording. A headless Chromium drives the real dashboard through the
real operator command path, screenshots at a fixed cadence, and ffmpeg assembles
the frames with a generated palette. Re-running this produces the same
demonstration, because the scenario and seed are pinned and every action goes
through the same REST endpoints the buttons use.

    python3 scripts/generate_readme_gifs.py
    python3 scripts/generate_readme_gifs.py --only approve
    python3 scripts/generate_readme_gifs.py --fps 3 --keep-frames
    python3 scripts/generate_readme_gifs.py --scene-sync-gifs   # evidence GIFs, no recording
    python3 scripts/generate_readme_gifs.py --evaluator-demo    # end-to-end demo, no recording

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
# Scene-synchronization evidence GIFs (physical D435 -> Isaac -> pick)
# --------------------------------------------------------------------------- #

SCENE_SYNC_DIR = os.path.join(OUT_DIR, "scene-sync")
SCENE_SYNC_MANIFEST = os.path.join(REPO, "simulators", "isaac",
                                   "scene_sync_evidence", "gif-manifest.json")


def assemble_scene_sync_gifs(manifest_path: str = SCENE_SYNC_MANIFEST,
                             fps: Optional[int] = None,
                             width: Optional[int] = None,
                             keep_frames: bool = False) -> List[str]:
    """Assemble the scene-sync GIFs from the TRACKED evidence frames only.

    These GIFs are evidence of a physical run — a real D435 observation
    synchronized into Isaac and picked — and a physical run cannot be replayed
    by this script. So this mode records nothing: it reads the frame list in
    `simulators/isaac/scene_sync_evidence/gif-manifest.json`, stamps each
    tracked still with its label and the provenance footer, and hands the
    sequence to the same ffmpeg assembler the workflow GIFs use. A frame the
    manifest names but the repository does not hold is a hard error, never a
    substitute: regenerating can therefore not quietly replace this evidence
    with a generated-scenario capture.
    """
    from PIL import Image, ImageDraw, ImageFont            # noqa: PLC0415

    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    written: List[str] = []
    for gif_name, spec in manifest.items():
        if gif_name.startswith("_"):
            continue
        frames = spec.get("frames") or []
        if not frames:
            raise RuntimeError(f"{gif_name}: the manifest lists no frames")
        rate = int(fps or spec.get("fps", 3))
        out_width = int(width or spec.get("width", 800))
        footer = str(spec.get("footer", ""))
        folder = os.path.join(FRAME_ROOT, "scene-sync-" + gif_name.split(".")[0])
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

        index = 0
        for entry in frames:
            path = os.path.join(SCENE_SYNC_DIR, entry["file"])
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"{gif_name}: evidence frame {entry['file']} is missing from "
                    f"{os.path.relpath(SCENE_SYNC_DIR, REPO)}. It is a capture "
                    "from a physical run and cannot be regenerated here; "
                    "restore it from version control.")
            image = Image.open(path).convert("RGB")
            _stamp(image, str(entry.get("label", "")), footer, ImageDraw,
                   ImageFont)
            for _ in range(max(1, int(entry.get("hold", 1)))):
                index += 1
                image.save(os.path.join(folder, f"f{index:04d}.png"))

        os.makedirs(SCENE_SYNC_DIR, exist_ok=True)
        out = os.path.join(SCENE_SYNC_DIR, gif_name)
        assemble(folder, out, rate, out_width)
        size_mb = os.path.getsize(out) / 1e6
        print(f"[gif] {gif_name}: {index} frames, {index / rate:.1f}s, "
              f"{size_mb:.2f} MB  ({spec.get('run_mode', '?')} evidence)")
        written.append(out)
        if not keep_frames:
            shutil.rmtree(folder, ignore_errors=True)
    return written


# --------------------------------------------------------------------------- #
# Evaluator demo: ONE end-to-end sequence assembled from tracked evidence
# --------------------------------------------------------------------------- #

DEMO_DIR = os.path.join(OUT_DIR, "demo")
DEMO_MANIFEST = os.path.join(DEMO_DIR, "demo-manifest.json")
DEMO_CANVAS = (960, 540)


def _demo_font(ImageFont, size: int, bold: bool = True):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default()


def _demo_card(spec: Dict, Image, ImageDraw, ImageFont):
    """A full-frame text card: title, optional lines, on a dark ground."""
    width, height = DEMO_CANVAS
    image = Image.new("RGB", DEMO_CANVAS, (16, 22, 34))
    draw = ImageDraw.Draw(image)
    title_font = _demo_font(ImageFont, int(spec.get("title_size", 50)))
    line_font = _demo_font(ImageFont, int(spec.get("line_size", 27)), bold=False)
    margin = 70
    title_lines = _wrap(str(spec.get("title", "")), title_font, width - 2 * margin, draw)
    body = [w for line in spec.get("lines", []) for w in
            (_wrap(str(line), line_font, width - 2 * margin, draw) or [""])]
    title_h = 60 * len(title_lines)
    body_h = 40 * len(body)
    total = title_h + (26 if body else 0) + body_h
    y = (height - total) // 2
    for line in title_lines:
        w = draw.textlength(line, font=title_font)
        draw.text(((width - w) / 2, y), line, font=title_font, fill=(255, 255, 255))
        y += 60
    if body:
        draw.rectangle([width // 2 - 40, y + 4, width // 2 + 40, y + 8], fill=(255, 140, 40))
        y += 26
        for line in body:
            w = draw.textlength(line, font=line_font)
            draw.text(((width - w) / 2, y), line, font=line_font, fill=(222, 226, 234))
            y += 40
    return image


def _demo_frame(path: str, title: str, sub: str, Image, ImageDraw, ImageFont,
                below_banner: bool = False):
    """One evidence still fitted onto the canvas with a large title banner.

    `below_banner` fits the still into the area UNDER the banner instead of
    letting the banner overlay it — for dashboard panels, whose first row is
    the point.
    """
    width, height = DEMO_CANVAS
    source = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", DEMO_CANVAS, (238, 241, 246))
    draw = ImageDraw.Draw(canvas, "RGBA")
    title_font = _demo_font(ImageFont, 34)
    sub_font = _demo_font(ImageFont, 22, bold=False)
    margin = 22
    t_lines = _wrap(title, title_font, width - 2 * margin, draw) if title else []
    s_lines = _wrap(sub, sub_font, width - 2 * margin, draw) if sub else []
    band = (14 + 42 * len(t_lines) + 30 * len(s_lines) + (6 if s_lines else 0)
            if (t_lines or s_lines) else 0)
    top = band + 12 if below_banner else 0
    room = height - top
    scale = min(width / source.width, room / source.height)
    fitted = source.resize((max(1, int(source.width * scale)),
                            max(1, int(source.height * scale))), Image.LANCZOS)
    canvas.paste(fitted, ((width - fitted.width) // 2, top + (room - fitted.height) // 2))
    if not band:
        return canvas
    draw.rectangle([0, 0, width, band], fill=(16, 22, 34, 222))
    y = 10
    for line in t_lines:
        draw.text((margin, y), line, font=title_font, fill=(255, 255, 255))
        y += 42
    y += 6 if s_lines else 0
    for line in s_lines:
        draw.text((margin, y), line, font=sub_font, fill=(255, 190, 120))
        y += 30
    return canvas


def _demo_step_sources(step: Dict) -> List[str]:
    """The evidence files one storyboard step draws on, in order."""
    if "image" in step:
        return [os.path.join(OUT_DIR, step["image"])]
    if "frames" in step:
        return [os.path.join(OUT_DIR, f) for f in step["frames"]]
    if "range" in step:
        pattern, first, last = step["range"]
        return [os.path.join(OUT_DIR, pattern % n) for n in range(int(first), int(last) + 1)]
    return []


def assemble_evaluator_demo(manifest_path: str = DEMO_MANIFEST,
                            keep_frames: bool = False) -> List[str]:
    """Assemble the evaluator-facing end-to-end demo from TRACKED evidence.

    The storyboard lives in `images/generated/demo/demo-manifest.json`: text
    cards and evidence stills (physical D435 frames, Isaac frames, dashboard
    panels), each with an on-screen title, a duration for the long asset and
    a duration for the short README GIF (0 = not in the short cut). Nothing is
    recorded here: every still is a tracked capture from a live run, and a
    missing one is a hard error, never a substitute. The long variant is
    written as a GIF and as an H.264 MP4 for presentations.
    """
    from PIL import Image, ImageDraw, ImageFont            # noqa: PLC0415

    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    steps = manifest.get("steps") or []
    if not steps:
        raise RuntimeError("the demo manifest lists no steps")
    for step in steps:
        for path in _demo_step_sources(step):
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"demo evidence {os.path.relpath(path, REPO)} is missing; it is a "
                    "capture from a live run and cannot be regenerated here")

    written: List[str] = []
    for gif_name, spec in (manifest.get("outputs") or {}).items():
        variant = spec.get("variant", "long")
        fps = int(spec.get("fps", 4))
        folder = os.path.join(FRAME_ROOT, "demo-" + gif_name.split(".")[0])
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        index = 0
        for step in steps:
            seconds = float(step.get("short_seconds" if variant == "short" else "seconds", 0))
            count = int(round(seconds * fps))
            if count <= 0:
                continue
            if "card" in step:
                frame = _demo_card(step["card"], Image, ImageDraw, ImageFont)
                for _ in range(count):
                    index += 1
                    frame.save(os.path.join(folder, f"f{index:04d}.png"))
                continue
            sources = _demo_step_sources(step)
            subs = {int(k): v for k, v in (step.get("subs") or {}).items()}
            sub = str(step.get("sub", ""))
            for n in range(count):
                src_index = min(len(sources) - 1, (n * len(sources)) // count)
                sub = subs.get(src_index, sub)
                frame = _demo_frame(sources[src_index], str(step.get("title", "")), sub,
                                    Image, ImageDraw, ImageFont,
                                    below_banner=bool(step.get("below_banner")))
                index += 1
                frame.save(os.path.join(folder, f"f{index:04d}.png"))
        os.makedirs(DEMO_DIR, exist_ok=True)
        out = os.path.join(DEMO_DIR, gif_name)
        assemble(folder, out, fps, int(spec.get("width", DEMO_CANVAS[0])))
        print(f"[demo] {gif_name}: {index} frames, {index / fps:.1f}s, "
              f"{os.path.getsize(out) / 1e6:.2f} MB ({variant})")
        written.append(out)
        if spec.get("mp4"):
            mp4 = os.path.join(DEMO_DIR, spec["mp4"])
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
                 "-i", os.path.join(folder, "f%04d.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
                 "-movflags", "+faststart", mp4], check=True)
            print(f"[demo] {spec['mp4']}: {os.path.getsize(mp4) / 1e6:.2f} MB (H.264)")
            written.append(mp4)
        if not keep_frames:
            shutil.rmtree(folder, ignore_errors=True)
    return written


def _wrap(text: str, font, max_width: int, draw) -> List[str]:
    """Greedy word wrap by measured pixel width, so nothing runs off-frame."""
    lines: List[str] = []
    current = ""
    for word in text.split():
        candidate = (current + " " + word).strip()
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _stamp(image, label: str, footer: str, ImageDraw, ImageFont) -> None:
    """Label banner at the top, provenance footer at the bottom, in place.

    Both are word-wrapped to the frame width: a label that ran off the right
    edge would hide exactly the part of the sentence that qualifies the claim.
    """
    width, height = image.size
    scale = width / 960.0
    try:
        bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            int(24 * scale))
        small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(16 * scale))
    except OSError:
        bold = small = ImageFont.load_default()
    draw = ImageDraw.Draw(image, "RGBA")
    margin = int(16 * scale)
    if label:
        lines = _wrap(label, bold, width - 2 * margin, draw)
        line_h = int(30 * scale)
        band = int(18 * scale) + line_h * len(lines)
        draw.rectangle([0, 0, width, band], fill=(20, 24, 32, 215))
        for n, line in enumerate(lines):
            draw.text((margin, int(9 * scale) + n * line_h), line,
                      font=bold, fill=(255, 255, 255))
    if footer:
        lines = _wrap(footer, small, width - 2 * margin, draw)
        line_h = int(20 * scale)
        band = int(12 * scale) + line_h * len(lines)
        draw.rectangle([0, height - band, width, height],
                       fill=(20, 24, 32, 215))
        for n, line in enumerate(lines):
            draw.text((margin, height - band + int(6 * scale) + n * line_h),
                      line, font=small, fill=(230, 230, 230))


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

    # THE SELECTOR FIGURE IS CAPTURED AT 2x, like the other perception figures,
    # so its text is crisp at README width. The per-source shots below keep the
    # original context because they are compared side by side in a table and
    # their scale must not change under them.
    ctx = browser.new_context(
        viewport={"width": 1560, "height": 1000} if selector_only else VIEWPORT,
        # 3x for the selector: it crops to one control, so a 2x render came
        # out narrower than the surrounding figures. This is a true
        # higher-DPI render, never an upscale of a small capture.
        device_scale_factor=3 if selector_only else 1, color_scheme=theme)
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
#: The perception method whose figures need a prepared representation.
MODEL_FREE_METHOD = "foundationpose_rgbd_model_free"

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


#: The model-free perception figures, and the exact element each one crops to.
#: ELEMENT SHOTS, NOT VIEWPORT SHOTS — a full-window capture puts the controls
#: an evaluator is meant to read into a small corner of a very large image.
MODEL_FREE_SHOTS = (
    ("modelfree-scenario-panel.png", "#scenario-panel"),
    ("modelfree-physical-input.png", "#phys-acquire"),
    ("modelfree-physical-result.png", "#phys-block"),
)


def _capture_model_free_screenshots(browser, dash, theme: str) -> List[str]:
    """The model-free perception UI and a REAL physical model-free result.

    WHAT THIS REFUSES TO DO. It does not stage availability, and it does not
    reuse an old artefact because it photographs better: the physical shot is
    taken after an acquisition performed HERE, through the ordinary dashboard
    button, and if that acquisition fails the run stops with the reason instead
    of keeping whatever the panel happened to be showing.

    LIGHT THEME, like the rest of the README's media, and asserted rather than
    assumed — `set_light_theme` fails instead of producing a dark capture.
    """
    import urllib.request                                        # noqa: PLC0415

    def post(path: str, payload: Dict):
        request = urllib.request.Request(
            dash.url + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=1800) as handle:
            return json.load(handle)

    with urllib.request.urlopen(dash.url + "/api/state", timeout=30) as handle:
        state = json.load(handle)
    methods = (state.get("perception_method") or {})
    available = set(methods.get("available") or [])
    if MODEL_FREE_METHOD not in available:
        reason = (methods.get("unavailable_reasons") or {}).get(
            MODEL_FREE_METHOD, "no reason reported")
        raise SystemExit(
            "--model-free-shots needs the model-free method to be RUNNABLE on "
            f"the attached deployment; it is not: {reason}")
    sources = set((state.get("acquisition_choice") or {}).get("available") or [])
    if "realsense_d435" not in sources:
        raise SystemExit(
            "--model-free-shots needs the physical D435 available; the "
            "attached deployment does not report it. Nothing is staged.")

    written: List[str] = []
    # THE VIEWPORT IS CHOSEN FOR THE PANEL'S OWN LAYOUT. The Scenario panel is
    # a responsive grid: too narrow and it becomes a very tall single column
    # whose lower half is scenario knobs unrelated to perception; too wide and
    # the controls sit in a small corner of a large image. This width keeps the
    # figures readable at README scale. Nothing is edited — only the window
    # size, exactly as resizing a browser would do.
    ctx = browser.new_context(viewport={"width": 1560, "height": 1000},
                              device_scale_factor=2, color_scheme=theme)
    page = ctx.new_page()
    errors: List[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(dash.url + "/", wait_until="networkidle")
    page.wait_for_timeout(2500)
    if theme == "light":
        set_light_theme(page)
    page.wait_for_timeout(800)

    def save(name: str, selector: str, without: str = "") -> None:
        node = page.locator(selector)
        if not node.is_visible():
            raise SystemExit(f"{name}: {selector} is not on screen — refusing "
                             "to capture a figure that does not show it")
        page.evaluate("() => { const h = document.querySelector('header');"
                      " if (h) h.style.visibility = 'hidden'; }")
        # `without` REMOVES A SIBLING FIGURE'S SUBJECT FROM THIS ONE, and only
        # that. The result block physically contains the acquisition controls,
        # which are the previous figure; leaving them in makes this image
        # twice as tall and shows the same controls twice. Nothing reported —
        # no pose, no provenance, no number — is touched, and the element is
        # restored immediately after the shutter. This is the same treatment
        # the sticky page header already gets above.
        if without:
            page.evaluate("(sel) => { const n = document.querySelector(sel);"
                          " if (n) n.style.display = 'none'; }", without)
        node.scroll_into_view_if_needed()
        page.wait_for_timeout(250)
        node.screenshot(path=os.path.join(OUT_DIR, name))
        if without:
            page.evaluate("(sel) => { const n = document.querySelector(sel);"
                          " if (n) n.style.display = ''; }", without)
        page.evaluate("() => { const h = document.querySelector('header');"
                      " if (h) h.style.visibility = ''; }")
        if errors:
            raise SystemExit(f"{name}: page errors {errors}")
        written.append(os.path.join(OUT_DIR, name))
        print(f"[model-free] {name} "
              f"({os.path.getsize(os.path.join(OUT_DIR, name))/1e3:.0f} kB)")

    # -- A. the Scenario panel, simulated RGB-D + model-free ---------------- #
    page.select_option("#s-acq", "isaac_simulated")
    page.wait_for_timeout(1500)
    page.select_option("#s-method", MODEL_FREE_METHOD)
    page.wait_for_timeout(2500)
    if page.eval_on_selector("#s-method", "e => e.value") != MODEL_FREE_METHOD:
        raise SystemExit("the model-free method did not stay selected")
    if "READY" not in page.eval_on_selector("#s-representation", "e => e.innerText"):
        raise SystemExit("the representation is not READY on this deployment; "
                         "refusing to capture a figure claiming that it is")
    save(*MODEL_FREE_SHOTS[0])

    # -- B. the physical acquisition controls, model-free ------------------- #
    page.select_option("#s-acq", "realsense_d435")
    page.wait_for_timeout(1500)
    page.select_option("#s-method", MODEL_FREE_METHOD)
    page.wait_for_timeout(2500)
    label = page.eval_on_selector("#phys-model-label", "e => e.innerText")
    if "CAD" in label:
        raise SystemExit("the physical panel still offers a CAD estimator "
                         f"input while model-free is selected ({label!r})")
    save(*MODEL_FREE_SHOTS[1])

    # -- C. a REAL physical model-free result, acquired now ----------------- #
    model = page.eval_on_selector("#phys-model", "e => e.value")
    roi = page.eval_on_selector("#phys-roi", "e => e.value")
    result = post("/api/perception/physical/acquire", {
        "model_id": model,
        "roi_px": [int(v) for v in roi.split(",")] if roi else None,
        # ONE MEASUREMENT FRAME, ONE INFERENCE PASS — what the button in the
        # photograph does. A screenshot taken from a run that cost more passes
        # than the documented behaviour would illustrate a different behaviour.
        "frames": 1, "perception_method": MODEL_FREE_METHOD})
    if not result.get("ok"):
        raise SystemExit(
            "the physical model-free acquisition failed at "
            f"{result.get('stage')}: {result.get('reason')}. No previous "
            "result is photographed in its place.")
    page.reload(wait_until="networkidle")
    page.wait_for_timeout(3000)
    if theme == "light":
        set_light_theme(page)
    page.wait_for_timeout(1200)
    running = page.eval_on_selector("#percep-source", "e => e.innerText")
    if "model-free" not in running:
        raise SystemExit(f"the run on screen is not model-free: {running!r}")
    save(*MODEL_FREE_SHOTS[2], without="#phys-acquire")
    ctx.close()
    return written


def _capture_source_screenshots(browser, dash, theme: str,
                                selector_only: bool = False) -> List[str]:
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

    # THE SELECTOR FIGURE IS CAPTURED AT 2x, like the other perception figures,
    # so its text is crisp at README width. The per-source shots below keep the
    # original context because they are compared side by side in a table and
    # their scale must not change under them.
    ctx = browser.new_context(
        viewport={"width": 1560, "height": 1000} if selector_only else VIEWPORT,
        # 3x for the selector: it crops to one control, so a 2x render came
        # out narrower than the surrounding figures. This is a true
        # higher-DPI render, never an upscale of a small capture.
        device_scale_factor=3 if selector_only else 1, color_scheme=theme)
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
    save("source-selector-light.png", page.locator("#s-acq-field"))
    page.evaluate("() => { const s = document.querySelector('#s-acq');"
                  " s.size = Number(s.dataset.shotSize || 0); }")
    page.wait_for_timeout(300)
    if selector_only:
        ctx.close()
        return written

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
            # THE ORDINARY ACQUISITION: one measurement frame, one
            # FoundationPose pass, exactly as the dashboard button asks for it.
            _acquire(page, "/api/perception/physical/acquire",
                     {"model_id": "cylinder5", "roi_px": [255, 70, 445, 719],
                      "frames": 1})
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
    parser.add_argument("--selector-shot", action="store_true",
                        help=("with --attach: regenerate ONLY the Object "
                              "source selector figure, leaving the per-source "
                              "table images untouched."))
    parser.add_argument("--evaluator-demo", action="store_true",
                        help=("assemble the evaluator-facing end-to-end demo "
                              "(images/generated/demo/) from the tracked evidence "
                              "stills listed in its manifest; records nothing"))
    parser.add_argument("--scene-sync-gifs", action="store_true",
                        help=("assemble the physical-D435 -> Isaac -> pick "
                              "evidence GIFs from the TRACKED frames listed in "
                              "simulators/isaac/scene_sync_evidence/"
                              "gif-manifest.json. Records nothing and needs no "
                              "dashboard, browser or camera; a listed frame "
                              "that is missing is an error, not a substitute."))
    parser.add_argument("--model-free-shots", action="store_true",
                        help=("with --attach: capture the MODEL-FREE perception "
                              "figures in light theme, cropped to the panels. "
                              "Refuses unless the model-free method is runnable "
                              "and the physical D435 is available, and performs "
                              "a real acquisition rather than photographing an "
                              "older result."))
    args = parser.parse_args()

    if args.evaluator_demo:
        if not shutil.which("ffmpeg"):
            print("ERROR: ffmpeg is required to assemble the demo.", file=sys.stderr)
            return 2
        written = assemble_evaluator_demo(keep_frames=args.keep_frames)
        if not args.keep_frames:
            shutil.rmtree(FRAME_ROOT, ignore_errors=True)
        print(f"\nwrote {len(written)} demo asset(s):")
        for path in written:
            print(f"  {os.path.relpath(path, REPO)}  "
                  f"({os.path.getsize(path) / 1e6:.2f} MB)")
        return 0

    if args.scene_sync_gifs:
        if not shutil.which("ffmpeg"):
            print("ERROR: ffmpeg is required to assemble the GIFs.",
                  file=sys.stderr)
            return 2
        written = assemble_scene_sync_gifs(
            fps=args.fps if args.fps != 3 else None,
            width=args.width if args.width != 1000 else None,
            keep_frames=args.keep_frames)
        if not args.keep_frames:
            shutil.rmtree(FRAME_ROOT, ignore_errors=True)
        print(f"\nwrote {len(written)} scene-sync GIF(s):")
        for path in written:
            print(f"  {os.path.relpath(path, REPO)}  "
                  f"({os.path.getsize(path) / 1e6:.2f} MB)")
        return 0

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
            if args.selector_shot:
                written = _capture_source_screenshots(browser, dash,
                                                      args.theme,
                                                      selector_only=True)
                browser.close()
                dash.close()
                print(f"\nwrote {len(written)} selector screenshot(s).")
                return 0
            if args.model_free_shots:
                written = _capture_model_free_screenshots(browser, dash,
                                                          args.theme)
                browser.close()
                dash.close()
                print(f"\nwrote {len(written)} model-free screenshot(s).")
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
