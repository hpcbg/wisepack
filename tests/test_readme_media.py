"""README media must exist, resolve, be light-theme, and be distinct.

These tests guard the deliverables of brief §22-§26: every required screenshot,
GIF and Behaviour Tree image is present, every image the README embeds actually
resolves, the screenshots are genuinely LIGHT theme (a dark capture is a failure),
the new GIFs are not accidental copies of the HitL ones, and no temporary frame
directory was left behind.
"""

from __future__ import annotations

import hashlib
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(REPO, "images", "generated")

REQUIRED_SCREENSHOTS = [
    "dashboard-light.png", "diagnostics-light.png", "anomaly-light.png",
    "cut-aware-light.png", "inventory-light.png", "logistics-light.png",
    "strategy-comparison-light.png",
]
REQUIRED_GIFS = [
    "hitl-approve-execute.gif", "hitl-dynamic-replan.gif",
    "hitl-container-unavailable.gif", "cut-aware-comparison.gif",
    "container-inventory.gif", "container-logistics.gif", "anomaly-workflow.gif",
]
REQUIRED_BT = [
    "wisepack_behaviour_tree.svg", "wisepack_behaviour_tree.png",
    "wisepack_behaviour_tree_interview.svg", "wisepack_behaviour_tree_interview.png",
]
NEW_GIFS = ["cut-aware-comparison.gif", "container-inventory.gif",
            "container-logistics.gif", "anomaly-workflow.gif"]
OLD_GIFS = ["hitl-approve-execute.gif", "hitl-dynamic-replan.gif",
            "hitl-container-unavailable.gif"]


@pytest.mark.parametrize("name", REQUIRED_SCREENSHOTS + REQUIRED_GIFS + REQUIRED_BT)
def test_required_media_exists_and_is_non_trivial(name):
    path = os.path.join(GEN, name)
    assert os.path.isfile(path), f"required media {name} is missing"
    assert os.path.getsize(path) > 1024, f"{name} is suspiciously small"


def test_all_readme_image_paths_resolve():
    with open(os.path.join(REPO, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    for ref in re.findall(r"images/generated/([\w.\-]+)", readme):
        assert os.path.isfile(os.path.join(GEN, ref)), \
            f"README references missing image {ref}"


@pytest.mark.parametrize("name", REQUIRED_SCREENSHOTS)
def test_screenshots_are_light_theme(name):
    """A corner sample must be light — the generator must fail on a dark capture."""
    Image = pytest.importorskip("PIL.Image", reason="Pillow needed for pixel check")
    img = Image.open(os.path.join(GEN, name)).convert("RGB")
    w, h = img.size
    # Sample a patch near the top-left header background, which is the light
    # panel colour (~#f4f6f9) in light theme and a dark navy in dark theme.
    xs = range(4, min(40, w), 6)
    ys = range(4, min(40, h), 6)
    samples = [img.getpixel((x, y)) for x in xs for y in ys]
    avg = sum(sum(p) for p in samples) / (len(samples) * 3)
    assert avg > 180, (
        f"{name} does not look light-theme (mean corner brightness {avg:.0f} "
        "<= 180) — refusing a dark capture")


def test_new_gifs_are_distinct_from_the_hitl_gifs():
    """The new cut/inventory/logistics/anomaly GIFs must not be copies."""
    def digest(name):
        with open(os.path.join(GEN, name), "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    old = {digest(g) for g in OLD_GIFS}
    for g in NEW_GIFS:
        assert digest(g) not in old, f"{g} is identical to a HitL GIF"
    # And the new GIFs are distinct from each other.
    new = [digest(g) for g in NEW_GIFS]
    assert len(set(new)) == len(new), "two new GIFs are identical"


def test_no_temporary_frame_directory_committed():
    assert not os.path.isdir(os.path.join(REPO, ".gif-frames")), \
        "the temporary .gif-frames directory must not be left behind"


def test_generator_defaults_to_light_theme():
    with open(os.path.join(REPO, "scripts", "generate_readme_gifs.py"),
              encoding="utf-8") as fh:
        src = fh.read()
    assert 'default="light"' in src, "README media must default to the light theme"
    assert 'color_scheme=args.theme' in src or 'color_scheme=theme' in src
