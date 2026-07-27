"""Task generator: determinism, validity and round-tripping."""

from __future__ import annotations

import json

import pytest

from wisepack_core.domain import GeometryType, Scenario
from wisepack_core.generator import (
    CONTAINER_SPECS, MATERIALS, PRESETS, build_curated_scenario, build_scenario,
    generate_scenario, inject_item, preset_config,
)


ALL_PRESETS = sorted(PRESETS)

#: Curated presets are hand-built and deterministic — they contain no random
#: draw, so a seed cannot and must not change them. Detected structurally via the
#: scenario's ``curated`` flag rather than a hard-coded name list.
CURATED_PRESETS = {p for p in ALL_PRESETS if build_scenario(p, seed=1).curated}
SEEDED_PRESETS = [p for p in ALL_PRESETS if p not in CURATED_PRESETS]


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_same_seed_gives_identical_items(preset):
    """The determinism contract: identical input, byte-identical output."""
    a = build_scenario(preset, seed=42)
    b = build_scenario(preset, seed=42)
    assert json.dumps(a.to_dict(), sort_keys=True) == \
           json.dumps(b.to_dict(), sort_keys=True)


@pytest.mark.parametrize("preset", SEEDED_PRESETS)
def test_different_seed_changes_the_scenario(preset):
    """A different seed must actually change something.

    The curated preset is excluded by design: it contains no random draw at all,
    so a seed cannot and must not change it.
    """
    a = build_scenario(preset, seed=42)
    b = build_scenario(preset, seed=1234)
    assert [i.to_dict() for i in a.items] != [i.to_dict() for i in b.items]


def test_curated_scenario_ignores_the_seed():
    """The curated dataset is hand-built, so seeds are irrelevant to it."""
    a = build_curated_scenario(seed=1)
    b = build_curated_scenario(seed=999)
    assert [i.to_dict() for i in a.items] == [i.to_dict() for i in b.items]
    assert a.curated is True


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_generated_dimensions_are_valid(preset):
    scenario = build_scenario(preset, seed=7)
    assert scenario.items, "generator produced no items"
    for item in scenario.items:
        assert item.length_mm > 0
        assert item.outer_diameter_mm > 0
        if item.inner_diameter_mm is not None:
            assert 0 < item.inner_diameter_mm < item.outer_diameter_mm
        assert item.weight_kg >= 0
        assert item.material_volume_mm3 > 0
        # The bounding box must never under-state the material inside it.
        assert item.material_volume_mm3 <= item.occupied_volume_mm3 + 1e-6
        assert item.permitted_axes


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_every_item_fits_its_container_in_some_orientation(preset):
    """A scenario whose items cannot fit measures the container, not the packer."""
    scenario = build_scenario(preset, seed=7)
    inner = scenario.container_template.inner_size
    for item in scenario.items:
        fits = any(
            item.size_for_axis(axis).x <= inner.x
            and item.size_for_axis(axis).y <= inner.y
            and item.size_for_axis(axis).z <= inner.z
            for axis in item.permitted_axes)
        assert fits, f"{item.item_id} ({item.length_mm}x{item.outer_diameter_mm}) " \
                     f"fits no orientation of {inner.as_tuple()}"


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_segregation_groups_are_valid(preset):
    known = {group for _, _, group in MATERIALS}
    scenario = build_scenario(preset, seed=3)
    for item in scenario.items:
        assert item.segregation_group in known
        assert item.dose_class in {"VLLW", "LLW", "ILW", None}


def test_segregated_preset_really_has_several_groups():
    scenario = build_scenario("segregated_materials", seed=42)
    assert len(scenario.segregation_groups) >= 2
    assert scenario.container_template.segregation_locking is True


def test_json_round_trip_preserves_everything():
    original = build_scenario("mixed_geometries", seed=11)
    restored = Scenario.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.scenario_id == original.scenario_id
    assert restored.seed == original.seed
    assert len(restored.items) == len(original.items)
    for a, b in zip(original.items, restored.items):
        assert a.to_dict() == b.to_dict()
    assert restored.container_template.to_dict() == \
           original.container_template.to_dict()


def test_csv_export_has_a_row_per_item():
    scenario = build_scenario("mixed_pipes_small", seed=5)
    header, rows = scenario.csv_rows()
    assert len(rows) == len(scenario.items)
    assert len(rows[0]) == len(header)
    assert "occupied_volume_mm3" in header
    assert "material_volume_mm3" in header


def test_mixed_geometries_marks_approximated_items():
    scenario = build_scenario("mixed_geometries", seed=42)
    approx = [i for i in scenario.items if i.is_approximated]
    exact = [i for i in scenario.items if not i.is_approximated]
    assert approx and exact, "preset should contain both exact and approximated"
    for item in approx:
        assert item.geometry_type is not GeometryType.TUBE
    for item in exact:
        assert item.geometry_type is GeometryType.TUBE


def test_tube_material_volume_matches_the_annulus_formula():
    import math
    scenario = build_scenario("mixed_pipes_small", seed=42)
    item = next(i for i in scenario.items if i.geometry_type is GeometryType.TUBE)
    expected = (math.pi / 4.0) * (item.outer_diameter_mm ** 2
                                  - (item.inner_diameter_mm or 0) ** 2) \
        * item.length_mm
    assert item.material_volume_mm3 == pytest.approx(expected)


def test_hollow_pipes_are_mostly_air():
    """The premise of the whole project: bounding box >> material volume."""
    scenario = build_scenario("mixed_pipes_dense", seed=42)
    tubes = [i for i in scenario.items if i.geometry_type is GeometryType.TUBE]
    assert tubes
    assert all(i.void_fraction_pct > 50.0 for i in tubes)


def test_item_count_override_is_honoured():
    scenario = generate_scenario(preset_config("mixed_pipes_small", 42,
                                               item_count=5))
    assert len(scenario.items) == 5


def test_injected_item_is_flagged_and_numbered_next():
    scenario = build_scenario("mixed_pipes_small", seed=42)
    before = len(scenario.items)
    item = inject_item(scenario, {"length_mm": 900, "outer_diameter_mm": 150,
                                  "inner_diameter_mm": 120})
    assert len(scenario.items) == before + 1
    assert item.injected is True
    assert item.weight_kg > 0, "weight should be derived when not supplied"
    assert item.item_id == f"item-{before + 1:03d}"


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="unknown preset"):
        build_scenario("no_such_preset", seed=1)


def test_container_specs_are_self_consistent():
    for name, spec in CONTAINER_SPECS.items():
        assert spec["inner_width_mm"] > 0
        assert spec["inner_depth_mm"] > 0
        assert spec["inner_height_mm"] > 0
        assert spec["max_payload_kg"] > 0
        assert "description" in spec, f"{name} must document its dimensions"


def test_invalid_generator_config_is_rejected():
    with pytest.raises(ValueError):
        preset_config("mixed_pipes_small", 42, item_count=0).validate()
    with pytest.raises(ValueError):
        preset_config("mixed_pipes_small", 42,
                      length_range_mm=(900, 100)).validate()
    with pytest.raises(ValueError):
        preset_config("mixed_pipes_small", 42,
                      wall_fraction_range=(0.1, 0.9)).validate()
