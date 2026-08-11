"""Which CAD model a control offers first, and why it is declared.

WHAT WENT WRONG. The physical acquisition control opened on whatever the model
list happened to put first, and then remembered whatever ran last. Both are
orderings, not decisions, and together they produced a live D435 acquisition
against `tutorial_bolt` — a reference asset from the FoundationPose tutorial that
WISEPACK does not package. Registering a bolt CAD onto a photograph of tubes does
not fail: it returns a pose, with 13.5 mm of centre spread and 56 deg of
axis spread, beside a caption that read "the Cylinder5 CAD reprojected".

So the default is DECLARED in the registry, reference assets are marked as such
by their declared `object_type`, and neither ordering nor memory can put one in
front of an operator acquiring a workpiece.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
sys.path.insert(0, os.path.join(REPO, "perception"))

from wisepack_core.rgbd import (ObjectModelRegistry, RGBDError,   # noqa: E402
                                load_object_registry)


@pytest.fixture(scope="module")
def registry() -> ObjectModelRegistry:
    loaded = load_object_registry(repo_root=REPO)
    assert not loaded.error, loaded.error
    return loaded


def test_the_registry_declares_a_default(registry):
    assert registry.default_model_id, (
        "no default_model_id is configured, so a control has nothing to open on "
        "but the first entry in an alphabetical list")


def test_the_default_is_cylinder5(registry):
    """The part every physical and simulated run here was validated on."""
    assert registry.default_model_id == "cylinder5"


def test_the_preferred_model_is_the_declared_one(registry):
    assert registry.preferred("foundationpose_rgbd") == "cylinder5"


def test_the_tutorial_bolt_is_marked_as_a_reference_asset(registry):
    bolt = registry.models["tutorial_bolt"]
    assert bolt.is_reference, (
        "the bolt is not marked as a reference asset, so nothing downstream can "
        "keep it out of an operator's way")


def test_the_workpieces_are_not_reference_assets(registry):
    for model_id, model in registry.models.items():
        if model_id == "tutorial_bolt":
            continue
        assert not model.is_reference, f"{model_id} is marked as a reference asset"


def test_a_reference_asset_is_never_the_fallback_default():
    """With no declaration, the fallback still must not choose a bolt."""
    loaded = load_object_registry(repo_root=REPO)
    loaded.default_model_id = ""
    preferred = loaded.preferred("foundationpose_rgbd")
    assert preferred, "no fallback default was resolved"
    assert not loaded.models[preferred].is_reference, (
        f"the fallback chose {preferred}, a reference asset")


def test_a_default_that_names_nothing_is_refused():
    """A default nothing can select is worse than none: it fails silently."""
    with pytest.raises(RGBDError):
        ObjectModelRegistry.from_dict(
            {"default_model_id": "no_such_model", "objects": []})


# --------------------------------------------------------------------------- #
# The list an operator sees
# --------------------------------------------------------------------------- #


def test_reference_assets_are_offered_last_and_say_what_they_are():
    from physical_pipeline import eligible_models                  # noqa: PLC0415

    models = eligible_models(REPO)
    assert models, "no models are eligible; the rest of this proves nothing"
    assert models[-1]["model_id"] == "tutorial_bolt", (
        f"the list opens toward {models[-1]['model_id']}; workpieces come first "
        "and reference assets last")
    bolt = models[-1]
    assert bolt["is_reference"] is True
    assert "reference asset" in bolt["label"].lower(), (
        f"the bolt reads {bolt['label']!r} — indistinguishable from a part")
    for entry in models[:-1]:
        assert entry["is_reference"] is False
        assert "reference asset" not in entry["label"].lower()


def test_the_endpoint_reports_the_default_and_not_a_sticky_last_model():
    """`last_model_id` is provenance. It must not be what preselects a control:
    that memory is what made one mistaken choice permanent."""
    app_source = open(os.path.join(REPO, "web", "app.py"), encoding="utf-8").read()
    body = app_source[app_source.index("def api_perception_physical_models"):]
    body = body[:body.index("\n@app.")]
    assert '"default_model_id": default' in body
    assert "preferred(" in body, "the default must come from the registry"

    index = open(os.path.join(REPO, "web", "index.html"), encoding="utf-8").read()
    fill = index[index.index("async function fillPhysicalModels"):]
    fill = fill[:fill.index("\n}")]
    assert "doc.default_model_id" in fill
    assert "doc.last_model_id" not in fill, (
        "the control still preselects the last model used; that is the sticky "
        "memory that kept the tutorial bolt selected")


def test_both_model_dropdowns_open_on_the_same_declared_default():
    """TWO CONTROLS, ONE REGISTRY. A physical dropdown and a simulated one that
    opened on different parts would be two answers to one question."""
    index = open(os.path.join(REPO, "web", "index.html"), encoding="utf-8").read()
    for function in ("fillPhysicalModels", "fillSimulatedModels"):
        body = index[index.index(f"async function {function}"):]
        body = body[:body.index("\n}")]
        assert "doc.default_model_id" in body, (
            f"{function} does not use the registry's declared default")
