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


def _code_only(source: str) -> str:
    """Python with docstrings and comments removed.

    These modules EXPLAIN what they must not do, and have to NAME those things
    to explain them — `physical_c5.sh`, "simulated", "planar" all appear in
    prose saying they are not used. A check that cannot tell the explanation
    from the behaviour would forbid writing the reason down.
    """
    source = re.sub(r'"""(?:.|\n)*?"""', "", source)
    return "\n".join(line.split("#")[0] for line in source.splitlines())


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
    # IN THE PIPELINE, which is where the stages now live.
    pipeline = _read(os.path.join(REPO, "perception", "physical_pipeline.py"))
    assert '"run_mode": "live" if live else "replay"' in pipeline
    assert "LIVE PHYSICAL D435" in pipeline
    assert "RECORDED PHYSICAL D435 DATA" in pipeline
    # THE PANEL SAYS HOW IT WAS ACQUIRED, IN THE PAST TENSE, and separately
    # from whether it is the current run. The badge once carried "LIVE PHYSICAL
    # D435" — true of the acquisition, and read as "now" by anyone looking at
    # the panel later beside a planar run that IS current.
    app = _read(APP)
    assert '"acquired live from the camera"' in app
    assert '"replayed from a recorded capture"' in app
    render = _physical_render()
    assert "doc.acquired_how" in render
    assert "LIVE" not in render


def test_recorded_evidence_is_never_presented_as_the_current_run():
    """THE REPORTED BUG. Switching the perception-method selector to
    FoundationPose made a cached physical result look freshly acquired, beside a
    planar run that was actually current. Nothing about the artefact changed —
    so nothing about how fresh it looks may change either.

    `is_current_run` is DERIVED from the batch on screen rather than hard-coded,
    so it becomes true by itself on the day a physical acquisition does drive a
    run — and is false today, because none does.
    """
    app = _read(APP)
    assert "current_run = getattr(batch, \"acquisition\", \"\") == ACQUISITION_REALSENSE" in app
    assert "Recorded physical D435 evidence — not the current run" in app
    render = _physical_render()
    assert "doc.is_current_run" in render
    assert "doc.status_label" in render
    # AND IT IS SAID ABOVE THE NUMBERS, not only in a badge.
    html = _read(INDEX)
    assert 'id="phys-status"' in html


def test_the_recorded_result_names_its_capture_and_time():
    """A cached result with no identity cannot be told from a fresh one by
    looking at it."""
    app = _read(APP)
    assert '"dataset": document.get("dataset", "")' in app
    render = _physical_render()
    assert "doc.dataset" in render
    assert "doc.completed_at" in render


def test_selecting_foundationpose_runs_no_inference():
    """A selector change configures the NEXT run; it must not acquire. The
    physical panel only ever reads an artefact — there is no POST in it."""
    render = _physical_render()
    assert "POST" not in render and "method:" not in render
    assert 'fetch("/api/perception/physical")' in render


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


# --------------------------------------------------------------------------- #
# Evaluator-facing presentation
# --------------------------------------------------------------------------- #


def test_the_device_serial_is_not_shown_on_the_normal_dashboard():
    """It identifies one specific unit and answers a question nobody asks of a
    demonstration. Firmware, USB version and the stream profile carry what an
    evaluator needs — that this is a real D4xx running a real profile."""
    render = _physical_render()
    assert "serial_number" not in render
    # STILL AVAILABLE where an audit would look: the API and the artefact.
    app = _read(APP)
    assert '"serial_number"' in app


def test_recorded_evidence_is_not_styled_as_a_failure():
    """Valid data from an earlier acquisition is not an error. An evaluator who
    learns to read red as "ignore this" will ignore a real failure later."""
    html = _read(INDEX)
    status = html[html.index('<div id="phys-status"'):
                  html.index('<div id="phys-provenance"')]
    assert 'class="percep-info"' in status
    assert "percep-error" not in status
    # AND THE NEUTRAL STYLE IS STILL A FULL-WIDTH BORDERED BLOCK, so the wording
    # keeps its prominence rather than fading into the notes around it.
    assert ".percep-info{" in html
    css = html[html.index(".percep-info{"):html.index(".conn::before")]
    assert "border" in css and "padding" in css


def test_the_timeline_says_whose_events_it_holds():
    """The physical panel sits directly above the timeline, and two blocks in a
    column read as one story. The camera-frame pose drove no planning, no
    approval and no execution — and cannot, with no work-area extrinsic."""
    html = _read(INDEX)
    assert 'id="log-scope"' in html
    render = _physical_render()
    assert "CURRENT WORKFLOW" in render
    assert "recorded physical D435 evidence" in render
    assert "drove no planning" in render


def test_the_timeline_note_appears_only_with_recorded_evidence():
    """The ordinary dashboard is unchanged when no physical artefact exists, and
    the note never outlives the panel it refers to."""
    html = _read(INDEX)
    scope = html[html.index('<div id="log-scope"'):html.index('id="log-filters"')]
    assert 'style="display:none' in scope, "it must start hidden"
    render = _physical_render()
    assert "doc.available && !doc.is_current_run" in render
    # Cleared on the path where the artefact has gone away.
    early = render[:render.index("PHYS_LAST = doc")]
    assert "log-scope" in early


# --------------------------------------------------------------------------- #
# Acquiring a NEW physical result from the dashboard
# --------------------------------------------------------------------------- #


PIPELINE = os.path.join(REPO, "perception", "physical_pipeline.py")


def test_there_is_one_implementation_of_the_physical_pipeline():
    """Two copies of "capture, segment, refuse, estimate" would agree only until
    somebody edited one, and a pose seen in the browser must be produced by
    exactly the code the CLI runs."""
    app = _read(APP)
    driver = _read(DRIVER)
    assert os.path.isfile(PIPELINE)
    assert "from physical_pipeline import" in app
    assert "from physical_pipeline import" in driver
    # The CLI keeps only what a terminal needs; the stages live in the module.
    for stage in ("def capture(", "def segment(", "def estimate("):
        assert stage not in driver, f"the CLI re-implements {stage}"


def test_the_dashboard_does_not_shell_out_to_the_cli():
    """The Python pipeline is importable, so spawning a shell script would add a
    second failure mode and lose the structured refusal."""
    app = _read(APP)
    body = _code_only(_function(app, "def api_perception_physical_acquire"))
    for forbidden in ("subprocess", "physical_c5.sh", "os.system"):
        assert forbidden not in body, f"the endpoint uses {forbidden}"


def test_acquisition_requires_an_explicit_model():
    """FoundationPose estimates the pose OF A KNOWN SHAPE. Which part is on the
    table is stated by the operator — never inferred from the image, the ROI or
    a measured component size."""
    app = _read(APP)
    body = _function(app, "def api_perception_physical_acquire")
    assert "`model_id` is required" in body
    assert "never inferred" in body
    # And no model is hard-coded into the generic endpoint — CODE ONLY, since
    # the comments name cylinder5 as the worked example when explaining that the
    # CAD geometry, not the pose, is what packing uses.
    assert "cylinder5" not in _code_only(body)


def test_the_model_list_comes_from_the_registry():
    app = _read(APP)
    assert "def api_perception_physical_models" in app
    pipeline = _read(PIPELINE)
    body = pipeline[pipeline.index("def eligible_models"):
                    pipeline.index("def run(")]
    assert "load_object_registry" in body
    assert 'supports("foundationpose_rgbd")' in body
    assert "mesh_exists" in body


def test_a_refusal_names_its_stage_and_substitutes_nothing():
    """"The camera is not there", "the mask is unusable" and "the estimator
    failed" send an operator to three different places."""
    pipeline = _read(PIPELINE)
    assert "class PhysicalAcquisitionError" in pipeline
    for stage in ('"camera"', '"segmentation"', '"estimation"', '"model"'):
        assert stage in pipeline
    # THE REFUSAL IS RETURNED, not swallowed, and no previous pose is used.
    app = _read(APP)
    body = _code_only(_function(app, "def api_perception_physical_acquire"))
    assert '"ok": False' in body
    for forbidden in ("_physical_c5_document", "simulated", "planar"):
        assert forbidden not in body, f"the failure path reaches for {forbidden}"


def test_no_mask_is_fabricated_when_segmentation_refuses():
    pipeline = _read(PIPELINE)
    body = pipeline[pipeline.index("def run("):]
    assert 'if not segmentation.get("mask_valid")' in body
    # The raise happens BEFORE any estimate call.
    assert body.index("mask_valid") < body.index("estimate(dataset")


def test_the_dropdown_does_not_acquire():
    """A selector configures the NEXT run. An operator who switched a dropdown
    and got a camera capture would have had no chance to check the model or the
    ROI first."""
    html = _read(INDEX)
    assert "async function acquirePhysical" in html
    assert '$("#c-phys-acquire").onclick = acquirePhysical' in html
    render = _physical_render()
    # The panel's poll shows the controls; it never presses the button.
    assert "acquirePhysical()" not in render


def test_a_fresh_acquisition_may_be_called_current_but_still_not_the_run():
    """Two different "current"s. A pose acquired seconds ago is legitimately the
    current PHYSICAL result; it still drove no planning and no approval."""
    app = _read(APP)
    assert "is_current_physical" in app
    assert "CURRENT PHYSICAL D435 RESULT" in app
    assert "is NOT part of the" in app
    render = _physical_render()
    assert "doc.is_current_physical" in render
    # THE TIMELINE NOTICE SURVIVES for either kind of physical result.
    assert "CURRENT WORKFLOW" in render
    assert "doc.available && !doc.is_current_run" in render


def test_a_new_acquisition_never_enables_planning():
    app = _read(APP)
    assert '"planning_available": False' in app
    body = _code_only(_function(app, "def api_perception_physical_acquire"))
    for forbidden in ("optimiz", "approve", "execute", "workarea_transform"):
        assert forbidden not in body, f"acquisition touches {forbidden}"
