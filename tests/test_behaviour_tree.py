"""The generated Behaviour Tree diagrams must reflect the implementation.

The generator derives its node set from ``wisepack_core.events.Stage`` and the
anomaly reactions, so these tests assert the diagram is regenerated
deterministically and contains the required nodes — including the anomaly hold
and acknowledgement, which are what make the Topic #2 integration visible in the
workflow.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(REPO, "scripts", "generate_behaviour_tree_images.py")
OUT = os.path.join(REPO, "images", "generated")
FULL_SVG = os.path.join(OUT, "wisepack_behaviour_tree.svg")
INTERVIEW_SVG = os.path.join(OUT, "wisepack_behaviour_tree_interview.svg")


@pytest.fixture(scope="module")
def generated():
    r = subprocess.run([sys.executable, GEN], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_generation_is_deterministic(generated):
    """Same inputs, byte-identical SVG."""
    with open(FULL_SVG, encoding="utf-8") as fh:
        first = fh.read()
    subprocess.run([sys.executable, GEN], capture_output=True, text=True, cwd=REPO)
    with open(FULL_SVG, encoding="utf-8") as fh:
        second = fh.read()
    assert first == second


def test_expected_files_exist(generated):
    for name in ("wisepack_behaviour_tree.svg",
                 "wisepack_behaviour_tree_interview.svg"):
        assert os.path.isfile(os.path.join(OUT, name)), f"{name} not generated"


def test_full_tree_contains_required_nodes(generated):
    with open(FULL_SVG, encoding="utf-8") as fh:
        svg = fh.read()
    for stage in ("DIGITAL_TWIN_VALIDATE", "WAIT_FOR_OPERATOR_APPROVAL",
                  "PICK_ITEM", "VERIFY_PLACEMENT", "REPLAN"):
        assert f'data-node="{stage}"' in svg, \
            f"required stage {stage} missing from the diagram"
    # Anomaly hold and acknowledgement must be present.
    for label in ("Anomaly HOLD", "Acknowledge anomaly", "Anomaly PAUSE"):
        assert label in svg, f"required anomaly node '{label}' missing"


def test_full_tree_contains_cut_inventory_logistics_nodes(generated):
    """The expanded tree (brief §20) must show the whole-process branches."""
    with open(FULL_SVG, encoding="utf-8") as fh:
        svg = fh.read()
    for stage in ("GENERATE_CUT_ALTERNATIVES", "DIGITAL_TWIN_VALIDATE_CUT_PLAN",
                  "WAIT_FOR_CUT_APPROVAL", "CUT_REQUESTED", "CUT_COMPLETED",
                  "REGISTER_DERIVED_ITEMS", "REPLAN_AFTER_CUT",
                  "CHECK_CONTAINER_AVAILABILITY", "RESERVE_CONTAINER",
                  "WAIT_FOR_CONTAINER", "COLLECT_FULL_CONTAINER"):
        assert f'data-node="{stage}"' in svg, \
            f"required whole-process stage {stage} missing from the diagram"


def test_interview_tree_tells_the_whole_process_story(generated):
    with open(INTERVIEW_SVG, encoding="utf-8") as fh:
        svg = fh.read()
    for phrase in ("Read container inventory", "cut-aware optimize",
                   "Human cut decision", "cutting skill",
                   "Reserve / deliver container", "Collect full container",
                   "FIWARE analytics"):
        assert phrase in svg, f"interview tree missing '{phrase}'"


def test_diagram_states_the_core_invariant(generated):
    for path in (FULL_SVG, INTERVIEW_SVG):
        with open(path, encoding="utf-8") as fh:
            svg = fh.read()
        assert "independently validated and explicitly approved" in svg, (
            f"{os.path.basename(path)} must state the core safety invariant")


def test_readme_references_resolve(generated):
    """Any behaviour-tree image the README embeds must actually exist."""
    with open(os.path.join(REPO, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    import re
    for ref in re.findall(r"images/generated/(wisepack_behaviour_tree[\w.]*)", readme):
        assert os.path.isfile(os.path.join(OUT, ref)), \
            f"README references missing {ref}"
