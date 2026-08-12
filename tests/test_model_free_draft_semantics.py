"""CURRENT RUN vs NEXT-RUN DRAFT, with model-free on the method axis.

THE SEMANTICS THIS PROTECTS
---------------------------
    CURRENT RUN   what actually produced the workflow on screen.
    NEXT-RUN      what the operator has selected for the next acquisition.

Changing a selector must not acquire, infer, replan, rewrite the current run's
provenance or revoke approval — and a poll landing mid-edit must not revert what
the operator just chose. Only the explicit Scenario action starts a run, and it
must dispatch from the controls the operator can SEE.

Adding a second FoundationPose method makes each of these easier to get wrong:
two RGB-D methods are interchangeable to the compatibility logic, so a
draft/current mix-up now produces a plausible-looking run measured the other
way rather than an obvious error.

WHY SOURCE-LEVEL CHECKS FOR THE FRONTEND. The real-DOM suites in this
repository need FastAPI and jsdom; this machine has neither in its host
interpreter, so those skip here. These assertions are the fallback that still
fails loudly if the rules are removed, and they are weaker than an execution
test rather than a replacement for one.
"""

from __future__ import annotations

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from wisepack_core.perception import (                            # noqa: E402
    DEFAULT_PERCEPTION_METHOD, PerceptionMethod, PerceptionMethodState,
    resolve_perception_method_selection)

CAD = PerceptionMethod.FOUNDATIONPOSE_RGBD.value
MODEL_FREE = PerceptionMethod.FOUNDATIONPOSE_RGBD_MODEL_FREE.value
PLANAR = PerceptionMethod.PLANAR_FASTERRCNN.value

PAGE = os.path.join(REPO, "web", "index.html")
APP = os.path.join(REPO, "web", "app.py")


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _slice_function(body: str, header: str) -> str:
    """One function's text, from its header to the next top-level `function`.

    Sliced on code rather than on a comment, because `_js_body` strips comments
    — a marker that vanishes with them would make the test pass by accident.
    """
    start = body.index(header)
    rest = body[start + len(header):]
    end = rest.find("\nfunction ")
    return body[start:] if end < 0 else body[start:start + len(header) + end]


def _js_body():
    """The page without comment lines, so a rule cannot pass by being described."""
    return "\n".join(line for line in _read(PAGE).splitlines()
                     if not line.lstrip().startswith("//"))


# --------------------------------------------------------------------------- #
# 1. The draft and the current run are separate values
# --------------------------------------------------------------------------- #


def test_a_model_free_draft_does_not_describe_the_running_batch():
    state = PerceptionMethodState(current=CAD, selected=MODEL_FREE,
                                  available=[PLANAR, CAD, MODEL_FREE])
    document = state.to_dict()
    assert document["current"] == CAD
    assert document["current_label"].endswith("(CAD)")
    assert document["selected"] == MODEL_FREE
    # AND THE PANEL IS TOLD THEY DIFFER, so it can say so rather than implying
    # the run on screen was measured the new way.
    assert document["changes_next_run"] is True


def test_the_current_geometry_describes_the_run_not_the_draft():
    """The field a panel keys on to decide whether to show a representation."""
    state = PerceptionMethodState(current=CAD, selected=MODEL_FREE,
                                  available=[PLANAR, CAD, MODEL_FREE])
    document = state.to_dict()
    assert document["current_estimator_geometry"] == "cad"
    assert document["current_requires_representation"] is False
    assert document["selected_estimator_geometry"] == "learned_representation"


def test_a_run_that_measured_nothing_claims_no_geometry():
    """A preset run has no method and therefore no estimator geometry. Empty,
    not defaulted — naming one would invent a provenance."""
    state = PerceptionMethodState(current="", selected=MODEL_FREE,
                                  available=[PLANAR, CAD, MODEL_FREE])
    document = state.to_dict()
    assert document["current"] == ""
    assert document["current_label"] == ""
    assert document["current_estimator_geometry"] == ""
    assert document["current_requires_representation"] is False


def test_an_unavailable_model_free_draft_falls_back_but_never_to_the_other_rgbd():
    """A DRAFT naming a method that cannot run must not become a run measured
    the other way. The fallback is the planar default, which the panel then
    displays — visibly different, rather than a silent swap between two RGB-D
    methods that would look identical on screen."""
    resolved = resolve_perception_method_selection(
        MODEL_FREE, [PLANAR, CAD], fallback=DEFAULT_PERCEPTION_METHOD)
    assert resolved == PLANAR
    assert resolved != CAD


def test_an_available_model_free_draft_is_kept():
    assert resolve_perception_method_selection(
        MODEL_FREE, [PLANAR, CAD, MODEL_FREE]) == MODEL_FREE


def test_every_method_is_offered_with_a_reason_when_unavailable():
    """VISIBLE AND DISABLED, never hidden: an operator must be able to see that
    the method exists and what it needs."""
    state = PerceptionMethodState(
        current="", selected=PLANAR, available=[PLANAR],
        unavailable_reasons={CAD: "worker down",
                             MODEL_FREE: "representation not built"})
    options = {o["value"]: o for o in state.to_dict()["options"]}
    assert set(options) == {PLANAR, CAD, MODEL_FREE}
    assert options[MODEL_FREE]["available"] is False
    assert options[MODEL_FREE]["reason"] == "representation not built"


# --------------------------------------------------------------------------- #
# 2. Selecting performs nothing
# --------------------------------------------------------------------------- #


def test_setting_the_draft_method_does_not_acquire_or_replan():
    """The command handler writes a setting and returns. If it ever acquired,
    an operator changing a dropdown would start a camera capture."""
    body = _read(APP)
    handler = body[body.index('if command == "set_perception_method":'):]
    handler = handler[:handler.index('if command == "acquire_simulated_rgbd":')]
    for forbidden in ("_acquire_simulated_rgbd", "run_physical", "run_simulated",
                      "replan", "reset(", "approve"):
        assert forbidden not in handler, forbidden
    assert 'STATE.settings["perception_method"]' in handler


def test_the_draft_endpoint_drops_an_unavailable_method_instead_of_selecting_it():
    body = _read(APP)
    draft = body[body.index("def api_draft("):]
    draft = draft[:draft.index("\n@app.")] if "\n@app." in draft else draft
    assert 'incoming.pop("perception_method")' in draft


# --------------------------------------------------------------------------- #
# 3. The action dispatches from what is visible
# --------------------------------------------------------------------------- #


def test_both_acquire_buttons_send_the_visible_method():
    body = _js_body()
    assert "function visibleMethod()" in body
    # BOTH RGB-D BUTTONS, because an operator can reach either.
    assert body.count("perception_method: visibleMethod()") == 2


def test_the_visible_method_comes_from_the_selector_not_from_polled_state():
    body = _js_body()
    function = _slice_function(body, "function visibleMethod()")
    assert '$("#s-method")' in function
    # NOT the last polled payload: reading the draft out of a poll would
    # dispatch a run the panel never displayed.
    for forbidden in ("perception_method.selected", "LAST_STATE", "STATE."):
        assert forbidden not in function, forbidden


def test_the_server_refuses_rather_than_substituting_a_method():
    body = _read(APP)
    resolver = body[body.index("def _resolve_requested_method("):]
    resolver = resolver[:resolver.index("\ndef ")]
    # THREE REFUSALS, and no fallback to another method anywhere in it.
    assert resolver.count('return "", {') == 3
    assert "requires_representation" in resolver
    for forbidden in ("FOUNDATIONPOSE_RGBD.value  #", "= CAD", "fallback"):
        assert forbidden not in resolver, forbidden


def test_a_representation_refusal_names_its_own_stage():
    """`stage` is what the panel prints. "representation" and "method" send an
    operator to different places."""
    body = _read(APP)
    resolver = body[body.index("def _resolve_requested_method("):]
    resolver = resolver[:resolver.index("\ndef ")]
    assert '"stage": "representation"' in resolver
    assert '"stage": "method"' in resolver


# --------------------------------------------------------------------------- #
# 4. Polling must not revert an edit
# --------------------------------------------------------------------------- #


def test_the_method_selector_is_not_overwritten_while_the_operator_is_editing():
    body = _js_body()
    assert re.search(r"if \(!DRAFT_TOUCHED && pm\.selected", body), \
        "the poll no longer guards the method selector with DRAFT_TOUCHED"


def test_the_representation_row_follows_the_selector_not_the_poll():
    """It renders from the CONTROL's current value, so a poll cannot make the
    row describe a method the operator is no longer choosing."""
    body = _js_body()
    function = _slice_function(body, "function applyRepresentationRow(")
    assert 'const method = $("#s-method")' in function
    assert "method.value" in function


# --------------------------------------------------------------------------- #
# 5. No stale state across runs
# --------------------------------------------------------------------------- #


def test_the_current_representation_is_read_from_the_runs_own_artefact():
    """Not from the registry: rebuilding a representation later must not change
    what an already-completed run reports."""
    body = _read(APP)
    function = body[body.index("def _current_run_representation("):]
    function = function[:function.index("\ndef ")]
    assert "_simulated_rgbd_document()" in function
    assert "_physical_c5_document()" in function
    # AND ONLY WHEN THE RUN'S METHOD MATCHES, so a CAD run cannot pick up the
    # representation a previous model-free run recorded.
    assert 'document.get("perception_method") != current_method' in function


def test_a_cad_run_reports_no_representation():
    body = _read(APP)
    function = body[body.index("def _current_run_representation("):]
    function = function[:function.index("\ndef ")]
    assert "requires_representation" in function
    assert "return {}" in function


def test_both_pipelines_record_what_the_estimator_was_given():
    for module in ("simulated_rgbd_pipeline.py", "physical_pipeline.py"):
        body = _read(os.path.join(REPO, "perception", module))
        assert '"estimator_geometry"' in body, module
        assert '"representation"' in body, module


def test_the_panel_shows_one_geometry_or_the_other_never_both():
    """Switching methods swaps the answer rather than leaving a representation
    on screen beside a CAD run."""
    body = _js_body()
    function = _slice_function(body, "function applyRepresentationRow(")
    assert function.count("field.style.display = \"none\"") >= 2
    assert "requires_representation" in function
    assert "requires_object_model" in function
