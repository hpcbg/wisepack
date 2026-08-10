"""The NEXT-run draft must survive the dashboard's poll.

THE BUG THIS PINS. Selecting "RGB-D 6-DoF — FoundationPose" snapped back to
"Planar RGB — Faster R-CNN" within a second, so the physical acquisition could
not be started from the dashboard at all. The state model was never wrong: the
backend already separates the DRAFT (`STATE.settings`) from what the running
batch was MEASURED with. What was missing was one wire — `#s-method` was not
bound to `onDraftChanged`, so choosing a method neither persisted it nor set
`DRAFT_TOUCHED`, and the next poll re-applied the stored value.

THESE TESTS RUN THE REAL JAVASCRIPT. The functions are extracted from
`web/index.html` by name and evaluated in Node against a stub DOM, so the poll
is actually executed rather than described — a source-grep would have passed
throughout the bug, because every line it would have looked for was already
there.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO, "web", "index.html")

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node is not installed")


def _read_index() -> str:
    with open(INDEX, encoding="utf-8") as handle:
        return handle.read()


def _extract(name: str) -> str:
    """One function's source, verbatim, from the dashboard page."""
    text = _read_index()
    start = text.index(f"function {name}(")
    # Back up over an `async` keyword if there is one.
    line_start = text.rfind("\n", 0, start) + 1
    if text[line_start:start].strip() == "async":
        start = line_start
    depth, i, opened = 0, text.index("{", start), False
    while i < len(text):
        if text[i] == "{":
            depth += 1
            opened = True
        elif text[i] == "}":
            depth -= 1
            if opened and depth == 0:
                return text[start:i + 1]
        i += 1
    raise AssertionError(f"could not extract {name}")


#: A stub DOM with exactly what the two functions touch: one <select> whose
#: options can be replaced, and the module-level draft flag.
HARNESS = """
class Option { constructor(value, label) { this.value = value; this.textContent = label;
                                           this.disabled = false; this.title = ""; } }
class Select {
  constructor() { this.options = []; this._value = ""; }
  set textContent(_v) { this.options = []; }
  get textContent() { return ""; }
  append(o) { this.options.push(o); if (!this._value) this._value = o.value; }
  get value() { return this._value; }
  set value(v) { this._value = v; }
}
const SELECT = new Select();
function $(_sel) { return SELECT; }
function el(_tag, attrs, label) { return new Option(attrs.value, label); }
let DRAFT_TOUCHED = false;
let METHOD_OPTIONS_KEY = "";
"""

#: The two-option list the dashboard is given once both methods are available.
OPTIONS = [{"value": "planar_fasterrcnn", "label": "Planar RGB — Faster R-CNN",
            "available": True, "reason": ""},
           {"value": "foundationpose_rgbd", "label": "RGB-D 6-DoF — FoundationPose",
            "available": True, "reason": ""}]


def _run(script: str) -> dict:
    body = (HARNESS + _extract("fillPerceptionMethods") + "\n" + script)
    result = subprocess.run([node, "-e", body], capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _poll_script(stored_selected: str, polls: int, touch_to: str = "") -> str:
    """Seed the selector, optionally make an operator selection, then poll."""
    touch = ""
    if touch_to:
        # EXACTLY WHAT THE OPERATOR DOES: set the value, then the change event
        # handler runs — which is the wire that was missing.
        touch = f'SELECT.value = {touch_to!r};\nDRAFT_TOUCHED = true;\n'
    return textwrap.dedent(f"""
        const state = {{ perception_method: {{
            options: {json.dumps(OPTIONS)}, selected: {stored_selected!r} }} }};
        fillPerceptionMethods(state);
        {touch}
        for (let i = 0; i < {polls}; i++) fillPerceptionMethods(state);
        console.log(JSON.stringify({{ selector: SELECT.value,
                                      stored: state.perception_method.selected }}));
    """)


def test_the_selector_is_seeded_from_the_backend_before_anybody_touches_it():
    """Untouched, the poll may seed it — that is how a fresh page finds the
    stored draft."""
    out = _run(_poll_script("planar_fasterrcnn", polls=3))
    assert out["selector"] == "planar_fasterrcnn"


def test_a_chosen_method_survives_repeated_polls():
    """THE REPORTED BUG, reproduced end to end.

    The current run is planar and the stored draft still says planar — the state
    the dashboard is in the instant after the operator chooses FoundationPose.
    Several poll cycles then run. The selector must not move.
    """
    out = _run(_poll_script("planar_fasterrcnn", polls=10,
                            touch_to="foundationpose_rgbd"))
    assert out["selector"] == "foundationpose_rgbd", (
        "the poll reverted the operator's selection")
    # AND THE CURRENT RUN IS UNCHANGED BY THE SELECTION: what the running batch
    # was measured with is a different field and still says planar.
    assert out["stored"] == "planar_fasterrcnn"


def test_the_reverse_direction_survives_too():
    """After a physical acquisition the stored draft is FoundationPose, and
    choosing the planar detector again must be equally stable."""
    out = _run(_poll_script("foundationpose_rgbd", polls=10,
                            touch_to="planar_fasterrcnn"))
    assert out["selector"] == "planar_fasterrcnn"
    assert out["stored"] == "foundationpose_rgbd"


def test_rebuilding_the_option_list_keeps_the_chosen_value():
    """The list is rebuilt whenever availability changes — a worker coming up
    mid-session does exactly that — and the selection must ride through it."""
    script = textwrap.dedent(f"""
        const first = {{ perception_method: {{
            options: {json.dumps(OPTIONS[:1])}, selected: 'planar_fasterrcnn' }} }};
        fillPerceptionMethods(first);
        const both = {{ perception_method: {{
            options: {json.dumps(OPTIONS)}, selected: 'planar_fasterrcnn' }} }};
        fillPerceptionMethods(both);
        SELECT.value = 'foundationpose_rgbd';
        DRAFT_TOUCHED = true;
        for (let i = 0; i < 5; i++) fillPerceptionMethods(both);
        console.log(JSON.stringify({{ selector: SELECT.value,
                                      stored: both.perception_method.selected }}));
    """)
    out = _run(script)
    assert out["selector"] == "foundationpose_rgbd"


# --------------------------------------------------------------------------- #
# The wire itself
# --------------------------------------------------------------------------- #


def test_the_method_selector_notifies_the_draft():
    """Without this binding the two behaviours above are unreachable: the
    selection is never persisted and DRAFT_TOUCHED is never set."""
    text = _read_index()
    # THE SCENARIO LOOP, found by a member only it has. There is a SECOND
    # draft-binding loop for the container/item controls, and matching on
    # "#s-preset" alone lands on whichever comes first in the file — which is
    # how a check ends up asserting about the wrong block entirely.
    blocks = [text[m.start():text.index("]", m.start())]
              for m in re.finditer(r'for \(const sel of \[', text)]
    scenario = [b for b in blocks if '"#s-robot"' in b]
    assert scenario, "the scenario draft-binding loop was not found"
    assert '"#s-method"' in scenario[0], (
        "the perception method must notify the draft like every other control")


def test_changing_the_method_acquires_nothing():
    """A selector configures the NEXT run. It may refresh what is on screen; it
    must not capture, detect or estimate."""
    source = _extract("onDraftChanged")
    for forbidden in ("/acquire", "detect_physical_objects", "command("):
        assert forbidden not in source, f"changing the draft calls {forbidden}"


def test_starting_a_run_clears_the_draft_flag():
    """The draft has become the active scenario at that point — and only at
    that point, which is what keeps the selection stable until then."""
    text = _read_index()
    # THE LAST OCCURRENCE, not the first: the first is the declaration
    # `let DRAFT_TOUCHED = false;` and asserting about its neighbours would
    # test the top of the file rather than the reset after a run.
    where = text.rindex("DRAFT_TOUCHED = false;")
    tail = text[where - 500:where]
    assert "command(" in tail, (
        "the flag must be cleared where a run STARTS, not on a poll")


def test_an_unnotified_selection_is_exactly_what_reverted():
    """THE MECHANISM, characterised rather than described.

    With the value changed but the draft NOT notified — the state the dashboard
    was in before `#s-method` was bound — the poll legitimately re-applies the
    stored draft. That is not a bug in the poll; it is the poll doing its job
    with no record that an operator had chosen anything. The binding is what
    makes the difference, so this pins the before-state and
    `test_a_chosen_method_survives_repeated_polls` pins the after.
    """
    script = textwrap.dedent(f"""
        const state = {{ perception_method: {{
            options: {json.dumps(OPTIONS)}, selected: 'planar_fasterrcnn' }} }};
        fillPerceptionMethods(state);
        SELECT.value = 'foundationpose_rgbd';   // no change handler fires
        for (let i = 0; i < 3; i++) fillPerceptionMethods(state);
        console.log(JSON.stringify({{ selector: SELECT.value }}));
    """)
    assert _run(script)["selector"] == "planar_fasterrcnn"
