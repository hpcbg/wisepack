"""CAD-backed scenarios — real reference parts instead of generated tubes.

TWO PATHS, COEXISTING, NEITHER REPLACING THE OTHER
--------------------------------------------------
    generated   parametric tubes from `generator.py`. The preset scenarios, the
                optimizer regressions and every test that needs no CAD. Fast,
                deterministic, large. UNCHANGED by this module's existence.
    cad_mesh    the actual reference parts — Cylinder1..5 — for perception,
                FoundationPose and sim-to-real work, where the exact geometry
                (hollow bore, saddle ends) is the entire point.

A scenario built here differs from a generated one in exactly one way: each item
carries `geometry_source="cad_mesh"` and a `model_id`. Everything downstream —
packing, the workflow, the Digital Twin, the approval gate — sees an ordinary
`WasteItem` and needs no knowledge of where the shape came from.

WHERE THE GEOMETRY IS RESOLVED, AND WHERE IT IS NOT
---------------------------------------------------
This module reads NOMINAL DIMENSIONS from the object-model registry
(`config/perception_objects.yaml`), which is the one place CAD metadata lives.
It does NOT read the mesh: no STL is parsed here, and nothing in the domain
layer imports a simulator. Whoever needs the actual triangles — the Isaac
adapter loading it into USD, the FoundationPose provider sending it to the
worker — resolves the path through the same registry.

That split is what lets planning code stay geometry-agnostic while the parts
stay exact.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .domain import (GEOMETRY_SOURCE_CAD_MESH, Axis, GeometryType, Scenario,
                     Vec3, WasteItem)

#: The reference parts, by the registry's own model ids. Named here as a
#: SELECTION, not a definition: every dimension comes from the registry.
CAD_SCENARIOS: Dict[str, Dict[str, Any]] = {
    # Stage A: one part, one container, the simplest closed loop that exercises
    # the whole path. Deliberately not a bin-picking scene.
    "cad_cylinder5_single": {
        "models": ["cylinder5"],
        "container_spec": "compact_box",
        "max_containers": 1,
        "description": "One Cylinder5 reference tube — first closed-loop test.",
    },
    # Stage G: several separated parts, still one per instance and still not
    # clutter.
    "cad_cylinders_mixed": {
        "models": ["cylinder2", "cylinder3", "cylinder4", "cylinder5"],
        "container_spec": "compact_box",
        "max_containers": 2,
        "description": "Four separated reference tubes — multi-object stage.",
    },
}

#: Density used when the registry does not carry a mass. Carbon steel, the same
#: figure `generator.py` uses, so a CAD item and a generated item of the same
#: size weigh the same.
DEFAULT_DENSITY_KG_M3 = 7850.0


class CADScenarioError(ValueError):
    """A CAD scenario that cannot be built, with the reason."""


def cad_scenario_names() -> List[str]:
    return sorted(CAD_SCENARIOS)


def is_cad_scenario(preset: str) -> bool:
    return preset in CAD_SCENARIOS


def build_cad_scenario(preset: str, scenario_id: Optional[str] = None,
                       registry: Any = None) -> Scenario:
    """Build a scenario whose items are real reference parts.

    DETERMINISTIC AND UNSEEDED. There is no randomness to seed: the parts are
    named, their dimensions come from the registry, and their placement is
    computed by the simulator's own layout. A regression that changes between
    runs would be useless for the sim-to-real comparison this exists for.
    """
    if preset not in CAD_SCENARIOS:
        raise CADScenarioError(
            f"unknown CAD scenario {preset!r}; known: "
            + ", ".join(cad_scenario_names()))
    spec = CAD_SCENARIOS[preset]

    if registry is None:
        from .rgbd import load_object_registry                # noqa: PLC0415
        registry = load_object_registry()

    items: List[WasteItem] = []
    for index, model_id in enumerate(spec["models"], start=1):
        model = registry.models.get(model_id)
        if model is None:
            raise CADScenarioError(
                f"{preset}: the object registry has no model {model_id!r}. "
                "CAD scenarios name parts; they do not define them.")
        items.append(_item_from_model(model, index))

    # THE SAME CONTAINER BUILDER the generated scenarios use. A CAD scenario
    # differs in its ITEMS, not in its containers or its workflow.
    from .generator import make_container                    # noqa: PLC0415
    return Scenario(
        scenario_id=scenario_id or preset,
        preset=preset,
        # SEEDED WITH ZERO because nothing here is random. The field is part of
        # the Scenario contract; carrying a meaningful-looking seed would imply
        # a generator that could have produced something else.
        seed=0,
        items=items,
        container_template=make_container(spec["container_spec"]),
        max_containers=int(spec["max_containers"]),
        description=spec["description"],
    )


def _item_from_model(model: Any, index: int) -> WasteItem:
    """One registry model -> one domain item. NO MESH IS READ.

    The packing layer works in integer millimetres from nominal dimensions, and
    those are what the registry declares. Deriving them from the mesh instead
    would make the planner's arithmetic move whenever somebody re-exported a
    part, which is precisely what `diameter_mm`/`length_mm` exist to prevent.
    """
    diameter = model.diameter_mm
    length = model.length_mm
    if not diameter or not length:
        raise CADScenarioError(
            f"{model.model_id} declares no nominal diameter/length. A CAD part "
            "still needs its nominal dimensions for packing: the mesh is the "
            "geometry, not the specification.")

    item = WasteItem(
        item_id=f"item-{index:03d}",
        length_mm=int(length),
        outer_diameter_mm=int(diameter),
        geometry_type=GeometryType.TUBE,
        inner_diameter_mm=(int(model.inner_diameter_mm)
                           if getattr(model, "inner_diameter_mm", None) else None),
        material="carbon_steel",
        segregation_group="A",
        source_position=Vec3(),
        priority=0,
        # THE TWO FIELDS THAT MAKE IT CAD-BACKED. Everything else on this item
        # is what a generated tube of the same size would carry.
        geometry_source=GEOMETRY_SOURCE_CAD_MESH,
        model_id=model.model_id,
        # A straight tube may lie along X or Y on the table but is never stood
        # on end for this demonstration — the same constraint the camera
        # observations use.
        permitted_axes=(Axis.X, Axis.Y),
    )
    if item.weight_kg == 0.0:
        item.weight_kg = round(
            item.material_volume_mm3 * 1e-9 * DEFAULT_DENSITY_KG_M3, 3)
    return item


__all__ = ["CAD_SCENARIOS", "CADScenarioError", "build_cad_scenario",
           "cad_scenario_names", "is_cad_scenario"]
