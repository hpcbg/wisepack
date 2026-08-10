"""The physical D435 result, shown in the dashboard without becoming a plan.

WHAT THIS PANEL IS FOR. WISEPACK can locate a real Cylinder5 with a real depth
camera, and an evaluator should see that in the ordinary dashboard rather than
in a terminal. What it must never do is let a physical pose look like an input
to planning: the physical camera has no validated camera-to-work-area
extrinsic, so the pose is real, valid and NOT placeable — three separate facts.

SOURCE-LEVEL, like the other dashboard tests here. `web/app.py` imports FastAPI,
which this host deliberately does not have (the dashboard borrows the container
for it), so the contract is asserted against the code rather than by serving it.
The behaviour behind it is exercised by running `./scripts/physical_c5.sh` and
reading `/api/perception/physical`.
"""

from __future__ import annotations

import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(REPO, "web", "app.py")
INDEX = os.path.join(REPO, "web", "index.html")
DRIVER = os.path.join(REPO, "scripts", "physical_c5.py")
LAUNCHER = os.path.join(REPO, "scripts", "physical_c5_dashboard.sh")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _function(source: str, header: str) -> str:
    """Just that function's own text, to the next top-level `def`.

    Slicing to some LATER function's header sweeps up everything between the
    two, which is how a check on one function silently starts asserting about
    its neighbours.
    """
    start = source.index(header)
    rest = source[start + len(header):]
    end = rest.find("\ndef ")
    return header + (rest if end == -1 else rest[:end])


def _physical_render() -> str:
    """The panel's own render function, comments stripped.

    The block EXPLAINS what it must not do and has to name those things to do
    so; a check that could not tell the explanation from the behaviour would
    forbid writing the reason down.
    """
    html = _read(INDEX)
    body = html[html.index("async function refreshPhysicalD435"):
                html.index("function showPhysicalImage")]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    return "\n".join(line.split("//")[0] for line in body.splitlines())


# --------------------------------------------------------------------------- #
# It reads the existing result — it does not estimate
# --------------------------------------------------------------------------- #


def test_the_dashboard_reads_the_artefact_and_runs_no_inference():
    """ONE INFERENCE PATH. The pose was produced by the real worker through the
    real provider; recomputing it here under different inputs would put a
    different number on screen from the one the run measured."""
    app = _read(APP)
    body = app[app.index("def api_perception_physical()"):
               app.index("def api_perception_physical_image")]
    assert "physical_c5.json" in app
    for forbidden in ("acquire_physical", "client.estimate", "register("):
        assert forbidden not in body, f"the route calls {forbidden}"


def test_the_physical_result_is_not_routed_through_stage_c():
    """Stage C carries a camera-to-work-area transform Isaac knows and this
    camera does not have. Feeding a physical pose through it would place a real
    object in a frame nobody measured."""
    app = _read(APP)
    body = _function(app, "def _physical_c5_document")
    assert "SIMULATED_RGBD_RESULT" not in body
    assert "stage-c" not in body
    driver = _read(DRIVER)
    assert "stage-c" not in driver and "stage_c" not in driver


def test_a_missing_result_is_reported_rather_than_faked():
    app = _read(APP)
    body = app[app.index("def api_perception_physical()"):
               app.index("def api_perception_physical_image")]
    assert '"available": False' in body
    assert "physical_c5.sh" in body, "the reason must name the command to run"


# --------------------------------------------------------------------------- #
# It says PHYSICAL, and says which kind of physical
# --------------------------------------------------------------------------- #


def test_the_panel_states_the_acquisition_and_its_provenance():
    app = _read(APP)
    assert '"acquisition": "Intel RealSense D435"' in app
    render = _physical_render()
    for required in ("Acquisition:", "Provenance:", "Perception:"):
        assert required in render


def test_live_and_replayed_are_different_claims():
    """Both are real D435 data. A recorded capture is not evidence that the
    camera worked just now, and a replay labelled live would claim a sensor the
    machine may not even have attached."""
    driver = _read(DRIVER)
    assert '"run_mode": "live" if live else "replay"' in driver
    assert "LIVE PHYSICAL D435" in driver
    assert "RECORDED PHYSICAL D435 DATA" in driver
    # The badge follows the run rather than being hard-coded to either.
    render = _physical_render()
    assert "doc.run_label" in render
    assert "LIVE PHYSICAL D435" not in render


def test_the_pose_is_shown_in_the_camera_frame_it_was_measured_in():
    """Three numbers without a frame are not a pose."""
    render = _physical_render()
    assert "doc.frame_id" in render
    assert "physical centre" in render
    assert "tube axis" in render


def test_the_simulated_work_area_centre_never_appears_in_the_physical_panel():
    """The simulated run reports a centre in `wisepack_workarea`. Rendering one
    beside a physical pose would read as though the real object had been
    located in the cell, which is exactly what has not been done."""
    render = _physical_render()
    assert "wisepack_workarea" not in render
    assert "simulated_rgbd" not in render
    assert "task_reference_point" not in render


# --------------------------------------------------------------------------- #
# It refuses to look like planning
# --------------------------------------------------------------------------- #


def test_planning_is_declared_unavailable_in_the_data_not_only_the_markup():
    """A consumer that omitted a label would otherwise render this as a
    planning input. The refusal travels with the result."""
    app = _read(APP)
    assert '"planning_available": False' in app
    assert "Work-area calibration is required before planning or execution" in app


def test_the_panel_shows_the_calibration_message():
    render = _physical_render()
    assert "planning_blocked_reason" in render


def test_the_physical_panel_drives_no_planning_or_execution_control():
    """No optimizer, no approval, no execution, no Digital Twin placement."""
    html = _read(INDEX)
    block = html[html.index('<div id="phys-block"'):
                 html.index('<img id="phys-image"')]
    for forbidden in ("c-approve", "c-execute", "c-plan", "c-detect-physical",
                      "compareStrategies", "optimiz"):
        assert forbidden not in block, f"the physical panel offers {forbidden}"


def test_the_workarea_flag_is_reported_and_not_assumed():
    app = _read(APP)
    assert '"workarea_pose_available": pose.get("workarea_pose_available")' in app
    render = _physical_render()
    assert "workarea_pose_available" in render


# --------------------------------------------------------------------------- #
# The real images, labelled as real
# --------------------------------------------------------------------------- #


def test_all_four_real_images_are_served_from_the_artefact():
    app = _read(APP)
    assert "PHYSICAL_C5_IMAGES" in app
    for kind, name in (("rgb", "rgb.jpg"), ("depth", "depth_aligned.jpg"),
                       ("mask", "mask_overlay.jpg"),
                       ("overlay", "pose_overlay.jpg")):
        assert f'"{kind}": "{name}"' in app


def test_every_image_caption_says_real_and_physical():
    """A picture of a real bench is indistinguishable from a rendered one at a
    glance. The caption is what makes the claim."""
    html = _read(INDEX)
    captions = html[html.index("const PHYS_CAPTIONS"):
                    html.index("async function refreshPhysicalD435")]
    assert captions.count("REAL / PHYSICAL D435") == 4


def test_the_pose_overlay_caption_denies_using_ground_truth():
    html = _read(INDEX)
    captions = html[html.index("const PHYS_CAPTIONS"):
                    html.index("async function refreshPhysicalD435")]
    assert "ESTIMATED pose" in captions
    assert "No simulator ground truth is used" in captions


def test_repeatability_is_never_called_accuracy():
    render = _physical_render()
    assert "NOT accuracy" in render


# --------------------------------------------------------------------------- #
# The simulated demonstration is untouched
# --------------------------------------------------------------------------- #


def test_stage_e_is_not_modified_by_the_physical_launcher():
    """`./scripts/stage_e.sh` remains the simulated perception-to-planning
    demonstration. The physical launcher must not wrap, call or alter it."""
    launcher = _read(LAUNCHER)
    assert "stage_e" not in launcher.replace("stage_e.sh`", "STAGE_E_MENTION") \
        or "does not touch it" in launcher
    assert os.access(os.path.join(REPO, "scripts", "stage_e.sh"), os.X_OK)


def test_the_launcher_starts_the_ordinary_dashboard():
    """No separate demo page. The panel lives in the normal dashboard, which is
    what an evaluator will already have open."""
    launcher = _read(LAUNCHER)
    assert "run_wisepack_dashboard.sh" in launcher
    assert ".html" not in launcher


def test_the_launcher_publishes_nothing_when_the_run_fails():
    """A dashboard panel is not a place to show a result that does not exist."""
    launcher = _read(LAUNCHER)
    assert "Nothing is published" in launcher
    assert "exit $status" in launcher


def test_the_simulated_panel_still_renders_independently():
    """Adding the physical block must not have replaced the simulated one: the
    two are the halves of the sim-to-real story."""
    html = _read(INDEX)
    assert 'id="sim-block"' in html
    assert "function renderSimulatedRGBD" in html
    assert 'id="phys-block"' in html


def test_the_two_panels_do_not_share_state():
    """A refresh of one must not clear the other. They describe different runs
    from different sensors and neither is evidence about the other."""
    html = _read(INDEX)
    assert "let PHYS_LAST" in html and "let SIM_VIEW" in html
    assert "PHYS_VIEW" in html
    # The physical panel is polled on its own, so a failure in it cannot take
    # the run view down.
    assert 'panel("physical-d435", refreshPhysicalD435)' in html


def test_the_physical_refresh_does_not_shadow_the_planar_one():
    """`refreshPhysical` already exists and belongs to the PLANAR camera panel.
    Two function declarations of one name do not collide loudly — the later one
    silently wins — and the panel that never ran would simply stay empty."""
    html = _read(INDEX)
    assert html.count("async function refreshPhysical(") == 1
    assert html.count("async function refreshPhysicalD435(") == 1
