"""One dashboard acquisition is ONE FoundationPose estimate.

THE PROBLEM THIS LOCKS SHUT. The dashboard's *Acquire & estimate* asked for five
measurement frames and therefore paid five FoundationPose passes to display one
pose. Five frames of a stationary scene is what MEASURING REPEATABILITY needs,
and that is a different question from the one the button asks — where is the part
now. The demonstration spent four inference passes on a figure it did not show.

THE DISTINCTION THAT MAKES ONE FRAME HONEST: warm-up frames are not measurement
frames. The RGB-D stream is opened per acquisition, so the worker's
`capture_dataset` discards the opening frames while auto-exposure settles and
only then writes the measurement frames. Warming the camera therefore costs no
estimate, and one warmed, aligned, segmented frame is a real measurement rather
than a rushed one. Nothing sleeps: the settling is counted in frames the device
actually delivered.

WHAT MUST NOT REGRESS ALONGSIDE IT:

* the repeatability/validation tools still take N measurement frames and produce
  N INDEPENDENT estimates — `--frames 5`, `--frames 10`, `--frames 12`;
* a one-frame result never reports a spread of zero, which would read as a
  perfect instrument rather than as an unasked question;
* both dashboard entry points behave identically, because both call one function.

SOURCE-LEVEL for the dashboard halves, like the other dashboard tests here:
`web/app.py` imports FastAPI, which this host deliberately does not have.
"""

from __future__ import annotations

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
sys.path.insert(0, os.path.join(REPO, "perception"))

APP = os.path.join(REPO, "web", "app.py")
INDEX = os.path.join(REPO, "web", "index.html")
PIPELINE = os.path.join(REPO, "perception", "physical_pipeline.py")
CAMERA = os.path.join(REPO, "perception", "foundationpose", "worker", "camera.py")
DRIVER = os.path.join(REPO, "scripts", "physical_c5.py")
PREPARE = os.path.join(REPO, "scripts", "physical_model_free_prepare.py")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _function(source: str, header: str) -> str:
    start = source.index(header)
    rest = source[start + len(header):]
    end = rest.find("\ndef ")
    return header + (rest if end == -1 else rest[:end])


def _js_function(source: str, header: str) -> str:
    """One JS function's text, to the next top-level `function`/`async`."""
    start = source.index(header)
    rest = source[start + len(header):]
    ends = [i for i in (rest.find("\nasync function "), rest.find("\nfunction "))
            if i != -1]
    return header + (rest if not ends else rest[:min(ends)])


def _code_only(source: str) -> str:
    """Python with docstrings and comments removed.

    These modules EXPLAIN the five-frame behaviour they no longer have, and have
    to name it to explain it. A check that could not tell the explanation from
    the behaviour would forbid writing the reason down.
    """
    source = re.sub(r'"""(?:.|\n)*?"""', "", source)
    return "\n".join(line.split("#")[0] for line in source.splitlines())


def _js_code_only(source: str) -> str:
    return "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("//"))


# --------------------------------------------------------------------------- #
# The dashboard asks for ONE measurement frame
# --------------------------------------------------------------------------- #


def test_the_acquire_button_asks_for_one_measurement_frame():
    """Five frames is a repeatability question. The button asks a different one,
    and paying five inference passes to display one pose is the cost of asking
    the wrong one."""
    body = _js_code_only(_js_function(_read(INDEX), "async function acquirePhysical"))
    assert "frames: 1" in body, (
        "the physical acquire button no longer requests exactly one measurement "
        "frame; every extra frame is another FoundationPose pass")
    assert "frames: 5" not in body


def test_both_dashboard_entry_points_call_the_one_acquire_function():
    """*Reset run & acquire* (Scenario) and *Acquire & estimate* (Perception
    panel) must not become two acquisition behaviours wearing one name."""
    html = _js_code_only(_read(INDEX))
    assert '$("#c-phys-acquire").onclick = acquirePhysical' in html
    # The Scenario button dispatches on the visible selector and reaches the
    # SAME function rather than a second request of its own.
    assert 'choice === "realsense_d435" ? acquirePhysical()' in html
    assert html.count("/api/perception/physical/acquire") == 1, (
        "a second caller of the acquire endpoint would be a second place for "
        "the frame count to disagree")


def test_the_endpoint_defaults_to_one_measurement_frame():
    body = _code_only(_function(_read(APP), "def api_perception_physical_acquire"))
    assert 'body.get("frames", 1)' in body, (
        "a caller that names no frame count must get one measurement frame and "
        "one inference pass, not five")


def test_the_shared_pipeline_defaults_to_one_measurement_frame():
    pipeline = _read(PIPELINE)
    assert "def run(model_id: str, roi_px: Optional[List[int]] = None, " \
           "frames: int = 1," in pipeline


# --------------------------------------------------------------------------- #
# One pass per measurement frame — never more, never fewer
# --------------------------------------------------------------------------- #


def test_one_inference_pass_per_measurement_frame():
    """`frames` is the number of estimates, exactly. Nothing re-estimates a
    frame it already estimated, and nothing estimates a frame twice to average
    it — either would make N mean something other than N measurements."""
    body = _code_only(_function(_read(PIPELINE), "def estimate("))
    assert "for index in range(frames):" in body
    assert body.count("provider.acquire_physical(") == 1, (
        "more than one estimator call inside the per-frame loop")


def test_warmup_frames_are_discarded_and_are_never_estimated():
    """The stream is opened per acquisition, so the settling frames are consumed
    inside the capture and never reach the estimator. This is what lets an
    ordinary acquisition take ONE measurement frame without taking a dark one."""
    camera = _read(CAMERA)
    body = _function(camera, "def capture_dataset(")
    warm = body.index("stream.warmup()")
    measure = body.index("for index in range(max(1, frames)):")
    assert warm < measure, (
        "the warm-up no longer precedes the measurement frames; a measurement "
        "frame would be taken while auto-exposure is still settling")
    # FRAME-BASED, NOT A SLEEP. A sleep is a guess about the device; frames are
    # what the device actually delivered.
    assert "time.sleep" not in _code_only(camera)
    assert "time.sleep" not in _code_only(_read(PIPELINE))


# --------------------------------------------------------------------------- #
# The validation tools still measure a spread
# --------------------------------------------------------------------------- #


def test_the_repeatability_tools_still_take_several_measurement_frames():
    """`--frames 5 / 10 / 12` are N INDEPENDENT measurement frames producing N
    INDEPENDENT estimates. Speeding up the demonstration must not quietly turn
    the evidence into one frame."""
    driver = _read(DRIVER)
    assert '"--frames", type=int, default=5' in driver
    prepare = _read(PREPARE)
    assert '"--frames", type=int, default=12' in prepare


def test_a_repeatability_run_estimates_every_frame_it_captured():
    """The same `frames` reaches the capture and the estimator, so N frames can
    never become N captured and one estimated."""
    body = _code_only(_function(_read(PIPELINE), "def run("))
    assert "capture(model_id, frames)" in body
    assert "estimate(dataset, model_id, frames, options" in body


# --------------------------------------------------------------------------- #
# A one-frame result does not fabricate repeatability
# --------------------------------------------------------------------------- #


def test_one_frame_reports_repeatability_as_not_measured_rather_than_zero():
    from physical_pipeline import repeatability                    # noqa: PLC0415

    spread = repeatability([])
    assert spread["measured"] is False
    assert "NOT" in spread["note"] and "zero" in spread["note"]
    # NOT A SINGLE FABRICATED NUMBER. A spread of 0.000 mm reads as a perfect
    # instrument; the absence of the question must not look like an answer.
    for key in ("model_frame_origin", "object_centre",
                "orientation_vs_first_deg", "tube_axis_line_deg"):
        assert key not in spread, f"{key} was invented for a single frame"
    # And it names what WOULD measure it, so an evaluator is not left guessing.
    assert "--frames" in spread["note"]


def test_the_panel_states_that_repeatability_was_not_measured():
    """A blank line where a figure used to be is indistinguishable from a figure
    of zero, and both read as "the instrument is perfect"."""
    html = _js_code_only(_read(INDEX))
    assert "repDoc.measured === false" in html
    assert "NOT MEASURED" in html


def test_the_measured_flag_survives_to_the_dashboard():
    app = _code_only(_read(APP))
    assert '"repeatability": document.get("repeatability", {})' in app


# --------------------------------------------------------------------------- #
# The cost is reported, not asserted
# --------------------------------------------------------------------------- #


def test_the_artefact_records_where_the_wall_clock_went():
    """"It felt slow" is not something a later reader can act on, and the number
    of inference passes is the thing this change controls — so both are recorded
    rather than claimed in a comment."""
    body = _code_only(_function(_read(PIPELINE), "def run("))
    for key in ('"capture_ms"', '"segmentation_ms"', '"inference_total_ms"',
                '"total_ms"'):
        assert key in body, f"the artefact does not record {key}"
    assert 'timing["inference_passes"] = len(batches)' in body
    assert '"timing_ms": timing' in body


def test_the_timing_reaches_both_dashboard_readers():
    app = _code_only(_read(APP))
    assert app.count('"timing_ms": document.get("timing_ms", {})') == 2, (
        "the acquire response and the panel route must both carry the timing")
    html = _js_code_only(_read(INDEX))
    assert "inference_passes" in html
