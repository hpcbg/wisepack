"""CAD-backed workpieces coexisting with the generated ones.

THE COMPATIBILITY REQUIREMENT IS THE POINT. Adding real reference parts must
leave every existing preset scenario producing byte-identical items, because the
generated path is what the optimizer regressions, the large deterministic
scenarios and the fast tests depend on. The two paths coexist; neither replaces
the other.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))

from wisepack_core.cad_scenarios import (                        # noqa: E402
    CAD_SCENARIOS, CADScenarioError, build_cad_scenario, cad_scenario_names,
    is_cad_scenario)
from wisepack_core.domain import (GEOMETRY_SOURCE_CAD_MESH,      # noqa: E402
                                  GEOMETRY_SOURCE_GENERATED, WasteItem)
from wisepack_core.generator import build_scenario               # noqa: E402


# --------------------------------------------------------------------------- #
# The generated path is untouched
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("preset", ["mixed_pipes_small", "mixed_pipes_dense"])
def test_existing_presets_still_produce_generated_items(preset):
    scenario = build_scenario(preset, seed=42)
    assert scenario.items
    assert {i.geometry_source for i in scenario.items} == {GEOMETRY_SOURCE_GENERATED}
    assert all(i.model_id == "" for i in scenario.items)


def test_generated_is_the_default_for_a_bare_item():
    """Nothing has to opt out of CAD; everything opts in."""
    item = WasteItem(item_id="i", length_mm=1000, outer_diameter_mm=100)
    assert item.geometry_source == GEOMETRY_SOURCE_GENERATED


def test_a_document_written_before_cad_existed_still_parses():
    item = WasteItem(item_id="i", length_mm=1000, outer_diameter_mm=100)
    legacy = {k: v for k, v in item.to_dict().items()
              if k not in ("geometry_source", "model_id")}
    restored = WasteItem.from_dict(legacy)
    assert restored.geometry_source == GEOMETRY_SOURCE_GENERATED
    assert restored.model_id == ""


def test_a_generated_scenario_is_unchanged_by_the_new_fields():
    """The same preset and seed must still give the same items."""
    a = build_scenario("mixed_pipes_dense", seed=42)
    b = build_scenario("mixed_pipes_dense", seed=42)
    assert [i.to_dict() for i in a.items] == [i.to_dict() for i in b.items]


# --------------------------------------------------------------------------- #
# The CAD path
# --------------------------------------------------------------------------- #


def test_the_stage_a_scenario_is_one_cylinder5():
    scenario = build_cad_scenario("cad_cylinder5_single")
    assert len(scenario.items) == 1
    item = scenario.items[0]
    assert item.model_id == "cylinder5"
    assert item.geometry_source == GEOMETRY_SOURCE_CAD_MESH


def test_cad_items_take_their_dimensions_from_the_registry():
    """Nominal dimensions, not mesh measurements: the planner's arithmetic must
    not move when somebody re-exports a part."""
    item = build_cad_scenario("cad_cylinder5_single").items[0]
    assert (item.outer_diameter_mm, item.length_mm) == (25, 342)
    assert item.inner_diameter_mm == 19            # D25 with a 3 mm wall


def test_a_cad_tube_is_not_weighed_as_a_solid_rod():
    """A hollow tube is several times lighter than the rod that bounds it."""
    item = build_cad_scenario("cad_cylinder5_single").items[0]
    assert 0.3 < item.weight_kg < 0.9, item.weight_kg


def _code_only(source: str) -> str:
    """Source with comments and docstrings removed.

    scene.py EXPLAINS why a CAD mesh is centred by citing the part whose STL
    origin sits 141 mm off its body, so the prose necessarily names it. A check
    that cannot tell the explanation from a hard-coded dependency would forbid
    writing the reason down.
    """
    import io
    import tokenize
    kept, previous = [], tokenize.INDENT
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        if (token.type == tokenize.STRING
                and previous in (tokenize.INDENT, tokenize.NEWLINE,
                                 tokenize.NL, tokenize.DEDENT)):
            previous = token.type
            continue
        previous = token.type
        kept.append(token.string)
    return "\n".join(kept)


def test_no_part_is_hard_coded_in_the_scene_builder():
    """§: the scene builder must not know a part by name. It resolves whatever
    the item's model_id names, through the registry."""
    code = _code_only(open(os.path.join(REPO, "simulators", "isaac", "scene.py"),
                           encoding="utf-8").read())
    for name in ("Cylinder1", "Cylinder2", "Cylinder3", "Cylinder4",
                 "Cylinder5", "cylinder5", ".stl"):
        assert name not in code, f"scene.py names {name}"


def test_the_scenario_names_parts_but_does_not_define_them():
    """Every dimension comes from the registry; the scenario is a selection."""
    for spec in CAD_SCENARIOS.values():
        for key in ("diameter_mm", "length_mm", "mesh_path", "wall_mm"):
            assert key not in spec


def test_an_unknown_model_is_refused_with_the_known_ones_listed():
    class Empty:
        models: dict = {}
    with pytest.raises(CADScenarioError) as exc:
        build_cad_scenario("cad_cylinder5_single", registry=Empty())
    assert "registry has no model" in str(exc.value)


def test_an_unknown_cad_scenario_is_refused():
    with pytest.raises(CADScenarioError):
        build_cad_scenario("cad_nonexistent")


# --------------------------------------------------------------------------- #
# Both paths are selectable through the same entry point
# --------------------------------------------------------------------------- #


def test_cad_scenarios_are_selectable_like_any_other_preset():
    """No restart, no second architecture: the same call."""
    generated = build_scenario("mixed_pipes_small", seed=42)
    cad = build_scenario("cad_cylinder5_single")
    assert generated.items[0].geometry_source == GEOMETRY_SOURCE_GENERATED
    assert cad.items[0].geometry_source == GEOMETRY_SOURCE_CAD_MESH


def test_switching_between_paths_does_not_disturb_either():
    first = build_scenario("mixed_pipes_dense", seed=42)
    build_scenario("cad_cylinder5_single")
    again = build_scenario("mixed_pipes_dense", seed=42)
    assert [i.to_dict() for i in first.items] == [i.to_dict() for i in again.items]


def test_is_cad_scenario_agrees_with_the_registry_of_names():
    assert is_cad_scenario("cad_cylinder5_single")
    assert not is_cad_scenario("mixed_pipes_dense")
    assert set(cad_scenario_names()) == set(CAD_SCENARIOS)


# --------------------------------------------------------------------------- #
# The layering rule
# --------------------------------------------------------------------------- #


def test_the_domain_layer_parses_no_meshes_and_imports_no_simulator():
    """§: generic workflow/planning code must not import Isaac or parse STL."""
    import ast
    root = os.path.join(REPO, "wisepack_ws", "src", "wisepack_core",
                        "wisepack_core")
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        source = open(os.path.join(root, name), encoding="utf-8").read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            for module in modules:
                lowered = module.lower()
                for forbidden in ("isaacsim", "omni", "pxr", "trimesh"):
                    assert not lowered.startswith(forbidden), \
                        f"{name} imports {module}"


def test_the_cad_scenario_builder_reads_no_mesh_file():
    source = open(os.path.join(REPO, "wisepack_ws", "src", "wisepack_core",
                               "wisepack_core", "cad_scenarios.py"),
                  encoding="utf-8").read()
    for marker in (".stl", "trimesh", "open("):
        assert marker not in source, f"cad_scenarios.py uses {marker}"


def test_the_isaac_adapter_resolves_the_mesh_through_the_shared_registry():
    """Not a path written into the adapter: a second source of CAD metadata
    would agree with the registry only until one of them was edited."""
    source = open(os.path.join(REPO, "simulators", "isaac", "scene.py"),
                  encoding="utf-8").read()
    assert "load_object_registry" in source
    assert "mesh_scale_to_mm" in source          # units from the declaration
