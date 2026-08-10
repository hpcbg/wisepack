"""The FoundationPose self-test is a diagnostic, not part of a WISEPACK run.

WHAT IT WAS. The control sat inside the live RGB-D 6-DoF panel on the main
dashboard, and its result was written into that panel's pose readout and image
viewer — which nothing else ever populated. So the only pose that block ever
displayed was an offline tutorial BOLT, shown beside a live capability badge,
in a demonstrator that packs pipe sections.

Moving it changed where the result is SHOWN and nothing else: same endpoint,
same payload, same provider call.
"""

from __future__ import annotations

import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO, "web", "index.html")
DIAGNOSTICS = os.path.join(REPO, "web", "diagnostics.html")
APP = os.path.join(REPO, "web", "app.py")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _markup_only(html: str) -> str:
    """HTML with comments stripped.

    Both pages EXPLAIN the move in a comment, naming the thing that moved. A
    check that cannot tell the explanation from the control would forbid
    writing the reason down — a trap this suite has fallen into before.
    """
    return re.sub(r"<!--.*?-->", "", html, flags=re.S)


def _script_only(html: str) -> str:
    """Just the script bodies, with // and /* */ comments removed."""
    scripts = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
    scripts = re.sub(r"/\*.*?\*/", "", scripts, flags=re.S)
    return "\n".join(line.split("//")[0] for line in scripts.splitlines())


# --------------------------------------------------------------------------- #
# It is gone from the live dashboard
# --------------------------------------------------------------------------- #


def test_the_live_dashboard_has_no_self_test_control():
    markup = _markup_only(_read(INDEX))
    assert "c-fp-reference" not in markup
    assert "c-fp-selftest" not in markup
    assert "Run reference/offline regression" not in markup


def test_the_live_dashboard_cannot_render_a_self_test_result():
    """The readout and the image viewer moved with it. Leaving them behind
    would leave the exact surface the bolt used to appear on."""
    markup = _markup_only(_read(INDEX))
    for element in ('id="fp-pose"', 'id="fp-image"', 'id="fp-views"'):
        assert element not in markup, element


def test_the_live_dashboard_never_calls_the_self_test_endpoint():
    script = _script_only(_read(INDEX))
    assert "reference-regression" not in script
    assert "foundationpose/image/" not in script


def test_the_live_panel_keeps_its_capability_status():
    """The move must not cost the operator the LIVE readiness information,
    which is a different thing entirely from the self-test."""
    markup = _markup_only(_read(INDEX))
    for element in ('id="fp-block"', 'id="fp-badge"', 'id="fp-status"',
                    'id="fp-blocked"'):
        assert element in markup, element


def test_the_live_simulated_rgbd_panel_is_untouched():
    """§: must not replace the current live RGB-D images/status. Those live in
    the separate simulated-RGB-D block, which this change never touched."""
    markup = _markup_only(_read(INDEX))
    for element in ('id="sim-block"', 'id="sim-views"', 'id="sim-provenance"'):
        assert element in markup, element


# --------------------------------------------------------------------------- #
# It is present, and labelled, under Diagnostics
# --------------------------------------------------------------------------- #


def test_diagnostics_hosts_the_self_test():
    markup = _markup_only(_read(DIAGNOSTICS))
    assert "c-fp-selftest" in markup
    assert "FoundationPose self-test" in markup


def test_the_diagnostics_control_names_the_dataset():
    """§: 'Offline tutorial bolt dataset'. An operator must not have to know
    that 'reference regression' means a bolt."""
    markup = _markup_only(_read(DIAGNOSTICS)).lower()
    assert "offline tutorial bolt dataset" in markup


def test_the_diagnostics_panel_says_it_is_not_part_of_a_run():
    markup = _markup_only(_read(DIAGNOSTICS)).lower()
    assert "not part of any wisepack run" in markup


def test_diagnostics_owns_its_own_result_state():
    """Its own state, so a self-test result cannot leak into a live view."""
    script = _script_only(_read(DIAGNOSTICS))
    assert "FP_LAST" in script
    assert "runFoundationPoseSelfTest" in script


# --------------------------------------------------------------------------- #
# The functionality itself is unchanged
# --------------------------------------------------------------------------- #


def test_the_endpoint_and_payload_are_unchanged():
    """§: keep the functionality unchanged — same endpoint, same explicit
    depth scale, which must never acquire a default."""
    script = _script_only(_read(DIAGNOSTICS))
    assert "/api/perception/foundationpose/reference-regression" in script
    assert "depth_scale_mm" in script
    app = _read(APP)
    assert '@app.post("/api/perception/foundationpose/reference-regression")' in app


def test_the_server_label_still_marks_the_result_offline():
    """'offline' is the load-bearing word and survives the rename."""
    app = _read(APP)
    start = app.index('"label": "FoundationPose self-test')
    label = app[start:app.index("\n", start)]
    assert "offline" in label.lower()
    assert "bolt" in label.lower()


def test_the_result_still_declares_itself_not_live():
    app = _read(APP)
    start = app.index('"label": "FoundationPose self-test')
    assert '"live": False' in app[start:start + 200]
