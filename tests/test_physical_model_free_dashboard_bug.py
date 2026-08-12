"""Regression for the demo-blocking Physical RGB-D + model-free dashboard bug.

WHAT WENT WRONG, AND WHY IT LOOKED LIKE A BACKEND FAILURE
---------------------------------------------------------
Selecting *Physical RGB-D camera* + *FoundationPose (model-free)* and pressing
acquire appeared to do nothing. It was NOT a dispatch or backend failure: the
request carried the right method, the worker ran, and the pose was produced.

Two independent UI defects hid it, and both were caused by code that named ONE
FoundationPose method at a time when two existed:

  1. `#phys-acquire` — the block holding the object selector, the ROI, the
     Acquire button, the busy note AND the success message — was displayed only
     when the method was exactly `foundationpose_rgbd`. With model-free
     selected the whole block was `display:none`, so the acquisition ran and
     completed entirely invisibly.

  2. The `#s-acq` change handler is async and, after its awaits, assigned
     `#s-method` from a `STATE` captured before those awaits finished. Changing
     the device and then the method — the exact demo sequence — let the stale
     value overwrite the operator's newer choice, and the run dispatched the
     method they had just moved away from.

A third, milder instance of the same mistake made a model-free run fail the
"is this an RGB-D run?" test, so the panel reported planar-detector errors
about a webcam the run never used.

THE LESSON THESE TESTS ENCODE: a UI condition must ask what a method NEEDS
(`requires_depth`, `requires_representation`), never what it is CALLED. Every
assertion below fails if a method name is compared against directly again.
"""

from __future__ import annotations

import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(REPO, "web", "index.html")
APP = os.path.join(REPO, "web", "app.py")

CAD = "foundationpose_rgbd"
MODEL_FREE = "foundationpose_rgbd_model_free"


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _js():
    """The page with comment lines removed, so a rule cannot pass by being
    described in prose."""
    return "\n".join(line for line in _read(PAGE).splitlines()
                     if not line.lstrip().startswith("//"))


def _fn(body, header):
    start = body.index(header)
    rest = body[start + len(header):]
    end = rest.find("\nfunction ")
    return body[start:] if end < 0 else body[start:start + len(header) + end]


# --------------------------------------------------------------------------- #
# 1. The acquire block must not be hidden for one RGB-D method
# --------------------------------------------------------------------------- #


def test_the_physical_acquire_block_is_not_gated_on_one_method_name():
    """THE BUG. `wantsPhysical` decided whether the Acquire button, the busy
    note and the result message were on screen at all."""
    body = _js()
    assert "const wantsPhysical" in body
    line = next(l for l in body.splitlines() if "const wantsPhysical" in l)
    assert '"foundationpose_rgbd"' not in line, line
    assert '"foundationpose_rgbd_model_free"' not in line, line
    # It follows the DEVICE, which every RGB-D method reads.
    assert "realsense_d435" in line


def test_the_object_list_is_filled_whenever_the_block_is_shown():
    """Filling it for one method only left `#phys-model` empty when model-free
    was chosen from a fresh page, and the acquisition then posted an empty
    `model_id` and was refused."""
    body = _js()
    block = body[body.index("const wantsPhysical"):]
    block = block[:block.index("const scope")]
    assert "fillPhysicalModels()" in block
    assert "if (wantsPhysical)" in block


def test_no_ui_condition_compares_against_a_bare_foundationpose_method_name():
    """The general rule. Conditions ask what a method NEEDS, not what it is
    called — the property that survives a third RGB-D method being added."""
    body = _js()
    offenders = [l.strip() for l in body.splitlines()
                 if re.search(r'===\s*"foundationpose_rgbd(_model_free)?"', l)
                 or re.search(r'!==\s*"foundationpose_rgbd(_model_free)?"', l)]
    # `#fp-capability` legitimately checks that the CAD option EXISTS at all,
    # which is a question about the deployment, not about the selected run.
    offenders = [l for l in offenders if "options || []" not in l
                 and "pm.options" not in l]
    assert offenders == [], offenders


def test_the_rgbd_test_is_derived_from_requires_depth():
    body = _js()
    helper = _fn(body, "function isRgbdMethod(")
    assert "requires_depth" in helper
    assert '"foundationpose_rgbd"' not in helper


def test_the_run_is_recognised_as_rgbd_for_either_method():
    body = _js()
    assert "isRgbdMethod(doc, currentMethod)" in body


# --------------------------------------------------------------------------- #
# 2. A stale selector value must not clobber a newer choice
# --------------------------------------------------------------------------- #


def test_the_acquisition_handler_only_moves_the_method_when_it_must():
    """The async clobber. It assigned `#s-method` unconditionally from a
    `STATE` read before its own awaits completed."""
    body = _js()
    handler = body[body.index("acq.addEventListener(\"change\""):]
    handler = handler[:handler.index("const objsrc")]
    assert "mustMove" in handler
    assert "selected_methods" in handler
    # The assignment is now conditional on the displayed value being unusable.
    assert re.search(r"if \(method && settledMethod && \(mustMove \|\| !displayed\)",
                     handler), handler


def test_the_dispatch_still_reads_the_visible_selector():
    body = _js()
    assert "function visibleMethod()" in body
    assert body.count("perception_method: visibleMethod()") == 2


# --------------------------------------------------------------------------- #
# 3. The estimator input shown must match the selected method
# --------------------------------------------------------------------------- #


def test_the_panel_shows_representation_input_for_model_free():
    body = _js()
    helper = _fn(body, "function applyPhysicalEstimatorInput(")
    assert "requires_representation" in helper
    assert "Reference representation" in helper
    assert "no CAD supplied to the estimator" in helper


def test_the_panel_shows_cad_input_for_the_cad_method():
    body = _js()
    helper = _fn(body, "function applyPhysicalEstimatorInput(")
    assert '"Object model (CAD)"' in helper


def test_the_cad_label_is_not_static_markup():
    """It used to be literal text in the label, so it could not change with the
    method. It is now a span the code retargets."""
    page = _read(PAGE)
    assert 'id="phys-model-label"' in page
    assert 'id="phys-representation"' in page


def test_the_representation_lookup_does_not_lag_the_visible_selector():
    """Keyed by METHOD rather than by the server's stored draft: the stored
    draft trails the selector by a poll, which rendered every field as an
    em-dash and the status as NOT READY for about a second after selecting."""
    body = _js()
    helper = _fn(body, "function applyPhysicalEstimatorInput(")
    assert "representations || {})[method]" in helper
    app = _read(APP)
    assert 'document["representations"]' in app
    assert "requires_representation" in app


def test_no_object_name_appears_in_the_physical_panel_code():
    helper = _fn(_js(), "function applyPhysicalEstimatorInput(")
    assert "cylinder" not in helper.lower()
    assert "9e59f851" not in helper


# --------------------------------------------------------------------------- #
# 4. Busy / result / failure states
# --------------------------------------------------------------------------- #


def test_the_busy_state_is_set_before_the_request_and_cleared_after():
    body = _js()
    fn = body[body.index("async function acquirePhysical("):]
    fn = fn[:fn.index("\nasync function ")] if "\nasync function " in fn else fn
    assert 'btn.textContent = "Acquiring…"' in fn
    assert "finally" in fn
    cleared = fn[fn.index("finally"):]
    assert 'btn.textContent = "Acquire & estimate"' in cleared
    assert "btn.disabled = false" in cleared


def test_a_failure_reports_its_stage_and_does_not_leave_a_stale_result():
    """A previous successful result must never make a failed acquisition look
    like it succeeded."""
    body = _js()
    fn = body[body.index("async function acquirePhysical("):]
    fn = fn[:fn.index("\nasync function ")] if "\nasync function " in fn else fn
    assert "REFUSED at ${doc.stage}" in fn
    # The panel re-reads the artefact on EVERY outcome, in `finally`.
    assert "refreshPhysicalD435()" in fn[fn.index("finally"):]


def test_a_completed_acquisition_refreshes_the_runs_own_provenance():
    """Refreshing only this panel left the CURRENT RUN header naming the
    PREVIOUS method until the slower poll caught up — so straight after a
    model-free acquisition it could still read "(CAD)"."""
    body = _js()
    fn = body[body.index("async function acquirePhysical("):]
    fn = fn[:fn.index("\nasync function ")] if "\nasync function " in fn else fn
    tail = fn[fn.index("finally"):]
    assert "refreshPerception()" in tail
    assert "refreshState()" in tail


# --------------------------------------------------------------------------- #
# 5. What the estimator is actually given — unchanged by any of the above
# --------------------------------------------------------------------------- #


def test_the_model_free_request_still_carries_the_representation_only():
    """THE UI FIX MUST NOT HAVE TOUCHED THE GEOMETRY. Re-asserted here so this
    regression file fails if a later UI change reaches into the provider."""
    import sys
    for path in (os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"),
                 os.path.join(REPO, "perception")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from wisepack_core.representation import load_representation_registry
    from providers.foundationpose_rgbd import FoundationPoseProvider

    class Spy:
        def __init__(self):
            self.requests = []

        def estimate(self, request):
            self.requests.append(dict(request))
            return None, "spy"

    provider = FoundationPoseProvider(
        representations=load_representation_registry(repo_root=REPO))
    provider.client = Spy()
    provider.acquire_physical(dataset="d", model_id="cylinder5",
                              depth_scale_mm=1.0, method=MODEL_FREE)
    mesh = provider.client.requests[0]["mesh_path"]
    assert mesh.endswith("model.obj")
    assert ".stl" not in mesh.lower() and "CAD-Models" not in mesh


def test_packing_geometry_semantics_were_not_changed_by_the_ui_fix():
    from providers import foundationpose_rgbd as module
    source = _read(module.__file__)
    assert 'geometry_source="cad_model"' in source
    assert source.count("geometry_source=") == 1
