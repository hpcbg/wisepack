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
    page.wait_for_function(
        "() => document.querySelectorAll('#twin rect').length > 20", timeout=60_000)


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


SCENES: Dict[str, Dict] = {
    "approve": {"fn": scene_approve, "file": "hitl-approve-execute.gif"},
    "replan": {"fn": scene_replan, "file": "hitl-dynamic-replan.gif"},
    "container": {"fn": scene_container_unavailable,
                  "file": "hitl-container-unavailable.gif"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(SCENES), default=None)
    parser.add_argument("--fps", type=int, default=3,
                        help="capture and playback rate (2-4 reads well)")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright is required.\n"
              "  pip install playwright && playwright install chromium",
              file=sys.stderr)
        return 2
    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg is required to assemble the GIFs.", file=sys.stderr)
        return 2

    os.makedirs(OUT_DIR, exist_ok=True)
    scenes = {args.only: SCENES[args.only]} if args.only else SCENES
    written: List[str] = []

    dash = Dashboard()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            for name, spec in scenes.items():
                folder = os.path.join(FRAME_ROOT, name)
                shutil.rmtree(folder, ignore_errors=True)
                ctx = browser.new_context(viewport=VIEWPORT,
                                          device_scale_factor=1)
                page = ctx.new_page()
                errors: List[str] = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(dash.url, wait_until="networkidle")
                page.wait_for_timeout(2500)

                print(f"[gif] recording {name} ...")
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
