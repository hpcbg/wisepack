"""The four-way acquisition selector, and the draft rules it must obey.

WHAT THIS PINS. WISEPACK acquires objects four ways — a generated preset, a
planar webcam, a physical D435 and a simulated RGB-D camera — and the operator
chooses between them with ONE control. Three things about that control are easy
to break and expensive to notice:

    IT IS A PRESENTATION, NOT A NEW MODEL. Each choice maps onto the
    `object_source` axis the workflow already has and the `acquisition` device
    axis `wisepack_core.acquisition` already defines. A second registry would
    agree with those two only until somebody edited one of them.

    IT IS A DRAFT. Changing it acquires nothing, changes no plan, revokes no
    approval and rewrites no provenance. Only an explicit Generate / Detect /
    Acquire turns it into a run.

    A POLL MUST NOT REVERT IT. `#s-method` was once missing from the
    draft-change binding: the selection was never recorded, so the next poll put
    the selector back within a second. It looked like a broken control and was a
    missing wire. THE SAME WIRE IS ASSERTED HERE for `#s-acq`.

SOURCE-LEVEL for the dashboard parts, like the other dashboard tests here:
`web/app.py` imports FastAPI, which this host deliberately does not have.
"""

from __future__ import annotations

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))

from wisepack_core.acquisition import (                            # noqa: E402
    ACQUISITION_ISAAC, ACQUISITION_PLANAR, ACQUISITION_REALSENSE,
    METHOD_ACQUISITIONS)

APP = os.path.join(REPO, "web", "app.py")
INDEX = os.path.join(REPO, "web", "index.html")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _code_only(source: str) -> str:
    """Python with docstrings and comments removed.

    These modules EXPLAIN what they must not do and have to NAME those things to
    explain them. A check that could not tell the explanation from the behaviour
    would forbid writing the reason down.
    """
    source = re.sub(r'"""(?:.|\n)*?"""', "", source)
    return "\n".join(line.split("#")[0] for line in source.splitlines())


def _js_only(source: str) -> str:
    """JavaScript and markup with every kind of comment removed.

    HTML comments too: this panel EXPLAINS that it must not borrow physical
    wording, and has to write the word "physical" to say so. A check that could
    not tell the explanation from the behaviour would forbid the reason.
    """
    source = re.sub(r"<!--.*?-->", "", source, flags=re.S)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return "\n".join(line.split("//")[0] for line in source.splitlines())


# --------------------------------------------------------------------------- #
# A. Four choices, over the axes that already exist
# --------------------------------------------------------------------------- #


def test_the_selector_offers_exactly_four_acquisitions():
    code = _code_only(_read(APP))
    assert "ACQUISITION_CHOICES" in code
    for value in ("ACQUISITION_PRESET", "ACQUISITION_PLANAR",
                  "ACQUISITION_REALSENSE", "ACQUISITION_ISAAC"):
        assert value in code, f"{value} is not a selector choice"


def test_every_choice_maps_onto_the_existing_axes():
    """No competing model: each choice is (object_source, acquisition)."""
    source = _read(APP)
    block = source[source.index("ACQUISITION_CHOICES: Dict"):]
    block = block[:block.index("}\n")]
    # The preset is the one value that is an object source rather than a device.
    assert "PerceptionSource.SIM.value" in block
    assert "PerceptionSource.CAMERA.value" in block
    for device in (ACQUISITION_PLANAR, ACQUISITION_REALSENSE, ACQUISITION_ISAAC):
        constant = {ACQUISITION_PLANAR: "ACQUISITION_PLANAR",
                    ACQUISITION_REALSENSE: "ACQUISITION_REALSENSE",
                    ACQUISITION_ISAAC: "ACQUISITION_ISAAC"}[device]
        assert constant in block, f"{device} is not mapped"


def test_no_second_acquisition_registry_is_defined_in_the_dashboard():
    """The device names come from `wisepack_core.acquisition`, not from here."""
    code = _code_only(_read(APP))
    for literal in ('"planar_webcam"', "'planar_webcam'",
                    '"realsense_d435"', "'realsense_d435'",
                    '"isaac_simulated"', "'isaac_simulated'"):
        assert literal not in code, (
            f"{literal} is hard-coded in web/app.py; the acquisition axis "
            "already names it and two lists agree only until one is edited")


# --------------------------------------------------------------------------- #
# B. Compatibility with the perception method is EXPLICIT
# --------------------------------------------------------------------------- #


def test_method_compatibility_is_derived_from_the_existing_map():
    code = _code_only(_read(APP))
    assert "METHOD_ACQUISITIONS.items()" in code, (
        "compatibility must be derived from METHOD_ACQUISITIONS, not restated")


def test_the_expected_mappings_hold():
    """The mapping the brief specifies, asserted against the shared registry."""
    assert METHOD_ACQUISITIONS["planar_fasterrcnn"] == (ACQUISITION_PLANAR,)
    assert set(METHOD_ACQUISITIONS["foundationpose_rgbd"]) == {
        ACQUISITION_REALSENSE, ACQUISITION_ISAAC}


def test_a_forced_method_change_is_reported_not_hidden():
    """§3: make the transition explicit in the draft state and the UI."""
    source = _read(APP)
    handler = source[source.index('if command == "set_acquisition"'):
                     source.index('if command == "set_object_source"')]
    assert "perception_method_changed_to" in handler, (
        "a forced method change must be reported back to the caller")
    js = _js_only(_read(INDEX))
    assert "method_conflict" in js, (
        "the UI must be able to say that switching will change the method")
    assert "perception_method_changed_to" in _read(INDEX) or \
        "settledMethod" in js, "the UI must re-sync the method selector"


def test_an_unavailable_acquisition_is_refused_with_its_reason():
    source = _read(APP)
    handler = source[source.index('if command == "set_acquisition"'):
                     source.index('if command == "set_object_source"')]
    assert 'state["available"]' in handler
    assert "unavailable_reasons" in handler, (
        "a refusal without the capability's own reason is a dead end")


def test_unavailable_options_are_disabled_with_a_reason_not_hidden():
    js = _js_only(_read(INDEX))
    block = js[js.index("function fillAcquisitions"):
               js.index("function applyAcquisitionNote")]
    assert "node.disabled = !enabled" in block, (
        "an unavailable mode must be disabled, never removed")
    assert "opt.reason" in block, "a disabled option must carry its reason"
    assert "unavailable" in block, (
        "the option text must say so; a tooltip alone is not visible enough")


def test_all_four_modes_are_in_the_static_markup():
    """WISEPACK SUPPORTS four acquisitions, and that is not a fact about what is
    plugged in. The list must not depend on a payload arriving: it once rendered
    blank without one, and then showed only the preset."""
    html = _read(INDEX)
    block = html[html.index('<select id="s-acq">'):]
    block = block[:block.index("</select>")]
    for value in ("preset", ACQUISITION_PLANAR, ACQUISITION_REALSENSE,
                  ACQUISITION_ISAAC):
        assert f'value="{value}"' in block, (
            f"{value} is not in the static option list, so a deployment that "
            "cannot reach the backend would not show it at all")


def test_the_preset_option_is_never_disabled():
    """It needs no service, no camera, no worker and no GPU."""
    js = _js_only(_read(INDEX))
    assert "ACQ_ALWAYS_ENABLED" in js
    block = js[js.index("function fillAcquisitions"):
               js.index("function applyAcquisitionNote")]
    assert "opt.value === ACQ_ALWAYS_ENABLED" in block


def test_the_options_are_updated_in_place_rather_than_rebuilt():
    """Rebuilding made the contents of a supported control depend on a payload
    arriving, which is how it came to show one option on a four-mode build."""
    js = _js_only(_read(INDEX))
    block = js[js.index("function fillAcquisitions"):
               js.index("function applyAcquisitionNote")]
    assert "sel.textContent = \"\"" not in block, (
        "the option list is wiped; a poll that arrives without options would "
        "empty a control whose modes the build actually supports")


# --------------------------------------------------------------------------- #
# C. DRAFT vs CURRENT RUN
# --------------------------------------------------------------------------- #


def test_setting_the_acquisition_touches_no_run():
    """The handler may write settings and nothing else."""
    source = _read(APP)
    handler = _code_only(source[source.index('if command == "set_acquisition"'):
                                source.index('if command == "set_object_source"')])
    for forbidden in ("start_run(", "apply_observation_batch", "generate_plans",
                      "digital_twin_validate", "request_approval",
                      "revoke", "STATE.engine ="):
        assert forbidden not in handler, (
            f"set_acquisition must not {forbidden!r}: changing a dropdown must "
            "never acquire, re-plan or revoke an approval")


def test_current_is_read_off_the_batch_not_off_a_setting():
    source = _read(APP)
    body = source[source.index("def current_acquisition"):]
    body = body[:body.index("\n# ---")]
    assert "observation_batch" in body
    assert 'STATE.settings' not in _code_only(body), (
        "`current` must describe what PRODUCED the batch on screen, never a "
        "draft setting — a preset run acquired nothing and naming a device for "
        "it would invent a provenance")


def test_a_draft_is_never_clamped_to_availability_by_a_poll():
    source = _read(APP)
    body = source[source.index("def acquisition_state"):
                  source.index("def current_acquisition")]
    assert "selected or current" in body
    code = _code_only(body)
    assert "if selected in available" not in code, (
        "the draft must not be silently moved to an available value; an "
        "unavailable choice is offered DISABLED with its reason instead")


def test_the_selector_is_bound_to_the_draft_change_handler():
    """THE REGRESSION. `#s-method` was once missing from this list; a selection
    that notifies nothing is never recorded, and the next poll reverts it."""
    js = _js_only(_read(INDEX))
    # THE LIST THAT FEEDS `onDraftChanged`, not just any binding loop: there is
    # more than one `for (const sel of [...])` in this file and asserting
    # against the wrong one would prove nothing.
    lists = [m.group(1) for m in
             re.finditer(r'for \(const sel of \[([^\]]*)\]\)', js, re.S)
             if "onDraftChanged" in js[m.end():m.end() + 400]]
    assert lists, "the draft-binding list could not be found"
    bound = "".join(lists)
    for control in ('"#s-method"', '"#s-acq"'):
        assert control in bound, (
            f"{control} is not bound to onDraftChanged — this is exactly the "
            "bug that made the perception-method selector look broken")


def test_the_acquisition_is_carried_in_the_draft_payload():
    js = _js_only(_read(INDEX))
    draft = js[js.index("function currentDraft"):]
    draft = draft[:draft.index("\n}")]
    assert "acquisition: draftAcquisitionDevice()" in draft, (
        "the draft must carry the acquisition, or DRAFT_TOUCHED protects a "
        "value nobody stored")


def test_the_draft_carries_a_device_never_the_preset_choice():
    """`preset` is an OBJECT SOURCE, not a camera. Sending it as an acquisition
    put the string "preset" into the settings field that names a device, and the
    axis then reported a label for a device that does not exist."""
    js = _js_only(_read(INDEX))
    body = js[js.index("function draftAcquisitionDevice"):]
    body = body[:body.index("\n}")]
    assert '=== "preset" ? ""' in body, (
        "the preset choice must map to an EMPTY device, not to itself")


def test_the_poll_only_seeds_the_selector_while_untouched():
    js = _js_only(_read(INDEX))
    block = js[js.index("function fillAcquisitions"):
               js.index("function applyAcquisitionNote")]
    assert "!DRAFT_TOUCHED" in block, (
        "a poll that re-applies server state unconditionally reverts the "
        "operator's selection within a second")


def test_a_stale_device_in_the_draft_form_cannot_reselect_it():
    source = _read(APP)
    body = source[source.index("def api_draft"):]
    body = body[:body.index("\n@app.")]
    assert '"acquisition" in incoming' in body
    assert "acquisition_state().available" in body


# --------------------------------------------------------------------------- #
# D. Honest labels — a simulated result is never dressed as a physical one
# --------------------------------------------------------------------------- #


def test_each_choice_names_its_own_action():
    source = _read(APP)
    block = source[source.index("_ACQUISITION_ACTIONS"):]
    block = block[:block.index("}\n")]
    assert "Generate & plan" in block
    assert "Detect & plan" in block
    assert block.count("Acquire & estimate") == 2, (
        "both RGB-D acquisitions acquire and estimate; the planar one detects "
        "and the preset generates")


def test_the_button_label_comes_from_the_backend_per_choice():
    js = _js_only(_read(INDEX))
    block = js[js.index("function genLabelFor"):]
    block = block[:block.index("\n}")]
    assert "acquisition_choice" in block, (
        "the button must be named per acquisition, or a simulated run gets a "
        "control worded for a physical camera")


def test_the_simulated_panel_never_claims_a_physical_camera():
    html = _read(INDEX)
    block = html[html.index('id="sim-block"'):html.index('id="fp-block"')]
    lowered = _js_only(block).lower()
    for forbidden in ("physical", "real camera", "d435 result"):
        assert forbidden not in lowered, (
            f"the simulated block says {forbidden!r}; a simulated result must "
            "never be labelled as a physical one")


# --------------------------------------------------------------------------- #
# E. The operator's words, and the internal axis's words, kept apart
# --------------------------------------------------------------------------- #


def test_the_visible_label_is_object_source():
    """WHAT AN OPERATOR READS. `acquisition` stays the internal name of the axis
    — it identifies WHICH DEVICE produced a batch, which is what the API fields,
    the enums and the batch provenance are about. The control asks a simpler
    question, and asks it in the operator's words."""
    html = _read(INDEX)
    label = html[html.index('id="s-acq-field"'):]
    label = label[:label.index("\n")]
    assert "Object source" in label, f"the visible label reads {label!r}"
    assert "Acquisition" not in label


def test_the_internal_axis_keeps_its_name():
    """The rename is user-facing ONLY. Renaming the axis would rename it in the
    API, the enums, the batch provenance and the orchestrator, none of which is
    what a label change is."""
    code = _code_only(_read(APP))
    for identifier in ("acquisition_choice_state", "ACQUISITION_CHOICES",
                       "acquisition_state", "current_acquisition"):
        assert identifier in code, f"{identifier} was renamed by a label change"
    # And the wire field the frontend reads is still `acquisition_choice`.
    assert '"acquisition_choice"' in code
    js = _js_only(_read(INDEX))
    assert "acquisition_choice" in js
    assert 'command("set_acquisition"' in js


def test_the_two_selectors_ask_different_questions():
    """Object source is WHERE; perception method is HOW. Kept as two controls."""
    html = _read(INDEX)
    assert 'id="s-acq"' in html and 'id="s-method"' in html
    # The visible source control comes first, the method second.
    assert html.index('id="s-acq-field"') < html.index('id="s-method-field"')


def test_the_scenario_controls_are_in_the_documented_order():
    html = _read(INDEX)
    order = ['id="s-acq-field"', 'id="s-method-field"', 'id="s-preset-field"',
             'id="s-robot-field"', 'id="s-seed"', 'id="s-strategy"']
    positions = [html.index(marker) for marker in order]
    assert positions == sorted(positions), (
        "the Scenario panel controls are not in the order "
        "Object source, Perception method, Preset, Execution source, Seed, "
        f"Strategy: {order} sit at {positions}")
