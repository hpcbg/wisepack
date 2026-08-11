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
#: PHYSICAL-CAMERA EVIDENCE. These three cannot be produced without a real
#: camera: `generate_readme_gifs.py --camera-shots` checks the attached
#: dashboard's `/api/perception` first and refuses unless the source is
#: `camera`, the batch is `ok` and the calibration is `valid`. They are listed
#: separately from the screenshots above because they are captured against a
#: LIVE deployment, not against the reproducible sim dashboard.
CAMERA_MEDIA = [
    "perception-camera-light.png",          # the Physical Perception panel
    "perception-twin-approval-light.png",   # Digital Twin + operator controls
    "perception-camera-annotated.jpg",      # the detector's own annotated frame
]

#: THE OBJECT SOURCE / PERCEPTION METHOD UI, captured from a real deployment by
#: `generate_readme_gifs.py --attach <url> --source-shots`. Listed separately
#: from the reproducible sim screenshots because each one requires the source it
#: shows to be genuinely available: the RGB-D pair needs the FoundationPose
#: worker, and the physical one needs a D435 attached.
SOURCE_MEDIA = [
    "source-selector-light.png",            # all four sources, real availability
    "source-draft-vs-current-light.png",    # a camera run, a preset drafted
    "source-physical-rgbd-light.png",       # source + forced method
    "source-physical-rgbd-result-light.png",
    "source-simulated-rgbd-light.png",
    "source-simulated-rgbd-result-light.png",
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


@pytest.mark.parametrize("name", CAMERA_MEDIA)
def test_physical_camera_media_exists_and_is_a_real_capture(name):
    """The camera evidence must be present and must be a photograph-sized file.

    A few kB would mean a blank panel or an error page saved with a 200; the
    real captures are hundreds of kB because they contain a camera frame.
    """
    path = os.path.join(GEN, name)
    assert os.path.isfile(path), (
        f"{name} is missing — regenerate it against a running camera stack:\n"
        "  ./run_wisepack_dashboard.sh          (with WISEPACK_PERCEPTION_SOURCE=camera)\n"
        "  python3 scripts/generate_readme_gifs.py --attach <url> --camera-shots")
    assert os.path.getsize(path) > 50_000, f"{name} is too small to be a capture"


def test_the_camera_media_is_referenced_by_the_readme():
    """Evidence nobody can see is not evidence."""
    readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    for name in CAMERA_MEDIA:
        assert name in readme, f"{name} is not referenced anywhere in the README"


def test_the_camera_capture_refuses_a_stack_that_is_not_on_a_camera():
    """The generator must not be able to produce this media from the simulator.

    Read from the source rather than run: the guard needs a live dashboard. What
    is asserted is that the guard EXISTS and covers all three conditions — a
    fabricated 'physical' screenshot is the one failure in this file that no
    later reader could detect.
    """
    src = open(os.path.join(REPO, "scripts", "generate_readme_gifs.py"),
               encoding="utf-8").read()
    assert "def _require_live_camera" in src
    assert "_require_live_camera(dash)" in src
    for condition in ('!= "camera"', '!= "ok"', '!= "valid"'):
        assert condition in src, (
            f"the camera-capture guard does not check {condition}")
    # And it must refuse to capture the coherent-state image while the
    # inconsistency banner is on screen.
    assert "Inconsistent state" in src


@pytest.mark.parametrize("name", REQUIRED_SCREENSHOTS
                         + [n for n in CAMERA_MEDIA if n.endswith(".png")])
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


# --------------------------------------------------------------------------- #
# The Object source / Perception method screenshots
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", SOURCE_MEDIA)
def test_the_source_media_exists_and_is_a_real_capture(name):
    """A few kB would mean an empty panel saved with a 200."""
    path = os.path.join(GEN, name)
    assert os.path.isfile(path), (
        f"{name} is missing — regenerate it against a running deployment:\n"
        "  ./run_wisepack_dashboard.sh\n"
        "  python3 scripts/generate_readme_gifs.py --attach <url> --source-shots")
    assert os.path.getsize(path) > 20_000, f"{name} is too small to be a capture"


@pytest.mark.parametrize("name", SOURCE_MEDIA)
def test_the_source_media_is_referenced_by_the_readme(name):
    """Evidence nobody can see is not evidence."""
    readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    assert name in readme, f"{name} is not referenced anywhere in the README"


def test_the_source_capture_refuses_a_source_it_cannot_actually_run():
    """The guard EXISTS and covers the whole claim.

    A source this deployment cannot run must be SKIPPED WITH ITS REASON, never
    staged: a panel captioned "Physical RGB camera, detecting" that no camera
    produced would be the one failure in this file no later reader could detect.
    """
    src = open(os.path.join(REPO, "scripts", "generate_readme_gifs.py"),
               encoding="utf-8").read()
    assert "def _capture_source_screenshots" in src
    assert "skipped.append" in src, "a source is dropped with no reason recorded"
    # It must ACQUIRE, not merely select — and it must wait for the panel to
    # agree that what it acquired is the run on screen.
    assert "/api/perception/physical/acquire" in src
    assert "/api/perception/simulated/acquire" in src
    assert "(d.acquisition || {}).current" in src, (
        "the capture does not wait for the panel to report the acquisition it "
        "just performed, so a shot can mix two runs")


def test_the_readme_names_the_current_scenario_button_wording():
    """§9: the primary Scenario action is "Reset run & …", not the older names."""
    readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    for wording in ("Reset run & generate", "Reset run & detect",
                    "Reset run & acquire"):
        assert wording in readme, f"the README never names {wording!r}"


def test_the_readme_uses_the_operator_facing_source_label():
    """The visible control is "Object source"; "Acquisition" is internal."""
    readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    for stale in ("**Acquisition** selector", "*Acquisition* selector",
                  "| Acquisition | Perception method |",
                  "Object source: Physical camera"):
        assert stale not in readme, f"stale operator-facing wording: {stale!r}"
