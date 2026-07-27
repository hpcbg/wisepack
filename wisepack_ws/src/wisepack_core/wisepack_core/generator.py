"""Deterministic nuclear-waste task generator.

Pattern adapted from HARVEST's ``task_generator.py``: a single ``random.Random``
seeded per run, dataclass output, and templates weighted to produce a realistic
mix rather than a uniform one. The subject matter is different (dismantled
metallic components, not farm tasks) so this is a re-implementation of the
approach, not a copy of the code.

The determinism contract is exact: the same (preset, seed, counts) produce
byte-identical scenario JSON on any machine and any Python 3.10+. That means

  * one RNG, drawn from in a fixed order — no set iteration, no dict ordering,
    no ``id()``, no time, no ``hash()`` of a str (which is salted per process);
  * every derived value rounded at the point of creation, not at print time;
  * item ids assigned from a counter, never from a shuffled pool.

Presets are declared here rather than in YAML because the acceptance demo must
be able to run with no config file present. config/scenarios/*.yaml can override
any field of a preset; see scenarios.py.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .domain import (
    Axis, Container, GeometryType, Scenario, Vec3, WasteItem,
)

# --------------------------------------------------------------------------- #
# Materials
# --------------------------------------------------------------------------- #

#: (name, density kg/m^3, segregation group). Densities are ordinary engineering
#: values for the alloys named; they drive the weight field and therefore the
#: payload constraint. They are NOT nuclear data of any kind.
MATERIALS: Sequence[Tuple[str, float, str]] = (
    ("carbon_steel",    7850.0, "A"),
    ("stainless_304",   8000.0, "A"),
    ("stainless_316L",  8000.0, "B"),
    ("inconel_600",     8470.0, "B"),
    ("zircaloy_4",      6560.0, "C"),
)

#: Simulated dose classes. There is no radiation model in this repository —
#: these are labels used to exercise the priority and segregation machinery, and
#: every surface that displays one marks it "simulated".
DOSE_CLASSES: Sequence[Tuple[str, float]] = (
    ("VLLW", 0.55),     # very low level waste
    ("LLW",  0.35),     # low level
    ("ILW",  0.10),     # intermediate level
)


# --------------------------------------------------------------------------- #
# Container specifications
# --------------------------------------------------------------------------- #

#: Inner dimensions in mm.
#:
#: SIZING NOTE — these are chosen so the packing problem is actually contested.
#: A container much larger than the batch makes every algorithm need exactly one
#: container, so baseline and optimized tie at 0% improvement and the demo shows
#: nothing; a container much smaller makes items unplaceable, which measures the
#: container, not the algorithm. Each spec below is paired with a preset whose
#: total occupied volume is roughly 0.4-1.2x the container capacity, which is the
#: band where container count is genuinely decided by packing quality. The
#: dimensions remain plausible for decommissioning hardware, and every preset's
#: longest item fits the container's longest inner dimension.
CONTAINER_SPECS: Dict[str, Dict[str, Any]] = {
    "standard_box": {
        "inner_width_mm": 1500, "inner_depth_mm": 800, "inner_height_mm": 600,
        "max_payload_kg": 1500.0,
        "description": "Standard metallic waste box (inner 1500x800x600 mm, 0.72 m3)",
    },
    "compact_box": {
        "inner_width_mm": 1250, "inner_depth_mm": 800, "inner_height_mm": 400,
        "max_payload_kg": 900.0,
        "description": "Compact waste box (inner 1250x800x400 mm, 0.40 m3)",
    },
    "long_box": {
        "inner_width_mm": 1550, "inner_depth_mm": 700, "inner_height_mm": 700,
        "max_payload_kg": 1500.0,
        "description": "Long-format waste box (inner 1550x700x700 mm, 0.76 m3)",
    },
    "iso_half_height": {
        "inner_width_mm": 1600, "inner_depth_mm": 1200, "inner_height_mm": 1100,
        "max_payload_kg": 3000.0,
        "description": "ISO-style half-height box (inner 1600x1200x1100 mm, 2.11 m3)",
    },
    # PHYSICAL SMOKE TEST ONLY. Sized to a Franka Emika Panda on a table, not to
    # a decommissioning cell: the whole container has to sit inside the arm's
    # reach envelope at drop height, which the industrial specs above do not.
    # It is deliberately kept out of every benchmark scenario so the measured
    # baseline-versus-optimized comparison is unaffected by its existence.
    "isaac_smoke_bin": {
        "inner_width_mm": 300, "inner_depth_mm": 220, "inner_height_mm": 150,
        "max_payload_kg": 30.0,
        "description": ("Bench-scale open-top bin for the Isaac Sim physical "
                        "smoke test (inner 300x220x150 mm, 0.010 m3)"),
    },
}


def make_container(spec: str = "standard_box", container_id: str = "CNT",
                   allowed_groups: Sequence[str] = (),
                   overrides: Optional[Dict[str, Any]] = None,
                   segregation_locking: bool = False) -> Container:
    """Build a container template from a named specification."""
    if spec not in CONTAINER_SPECS:
        raise ValueError(f"unknown container spec {spec!r}; "
                         f"known: {sorted(CONTAINER_SPECS)}")
    data = dict(CONTAINER_SPECS[spec])
    data.pop("description", None)
    data.update(overrides or {})
    return Container(
        container_id=container_id,
        inner_width_mm=int(data["inner_width_mm"]),
        inner_depth_mm=int(data["inner_depth_mm"]),
        inner_height_mm=int(data["inner_height_mm"]),
        max_payload_kg=float(data["max_payload_kg"]),
        allowed_segregation_groups=tuple(allowed_groups),
        segregation_locking=segregation_locking,
    )


# --------------------------------------------------------------------------- #
# Generation parameters
# --------------------------------------------------------------------------- #


@dataclass
class GeneratorConfig:
    """Everything the generator needs. Fully overridable from YAML or the UI."""

    preset: str = "mixed_pipes_small"
    seed: int = 42
    item_count: int = 16
    length_range_mm: Tuple[int, int] = (300, 1400)
    diameter_range_mm: Tuple[int, int] = (60, 220)
    #: Wall thickness as a fraction of the outer diameter, sampled per item.
    wall_fraction_range: Tuple[float, float] = (0.06, 0.16)
    materials: Sequence[str] = ()               # empty == all of MATERIALS
    container_spec: str = "standard_box"
    container_overrides: Dict[str, Any] = field(default_factory=dict)
    max_containers: int = 8
    #: When true, each container locks to the segregation group of its first
    #: item, making segregation a binding hard constraint rather than a label.
    segregation_locking: bool = False
    #: Fraction of items drawn from the five approximated geometry classes.
    #: 0.0 keeps the scenario to exact straight tubes only.
    approximated_fraction: float = 0.0
    #: Quantisation of generated dimensions, in mm. Real stock comes in discrete
    #: sizes, and quantised dimensions also make good packing visibly possible —
    #: a continuous distribution produces pathological slivers that no algorithm
    #: can exploit, which would understate the optimizer for the wrong reason.
    length_step_mm: int = 50
    diameter_step_mm: int = 10
    #: Orientations the packer and the validator may use for these items. The
    #: default is the full axis-aligned set. A preset restricts it when the
    #: EXECUTING hardware cannot achieve an orientation — the Isaac smoke preset
    #: drops "z" because a top-down parallel gripper cannot stand a pipe on its
    #: end without a regrasp, and planning a placement no backend can perform
    #: would make the physical run fail for a planning reason.
    permitted_axes: Tuple[str, ...] = ("x", "y", "z")
    description: str = ""

    def validate(self) -> None:
        if self.item_count <= 0:
            raise ValueError(f"item_count must be > 0, got {self.item_count}")
        lo, hi = self.length_range_mm
        if lo <= 0 or hi < lo:
            raise ValueError(f"invalid length_range_mm {self.length_range_mm}")
        dlo, dhi = self.diameter_range_mm
        if dlo <= 0 or dhi < dlo:
            raise ValueError(f"invalid diameter_range_mm {self.diameter_range_mm}")
        wlo, whi = self.wall_fraction_range
        if not (0.0 < wlo <= whi < 0.5):
            raise ValueError(
                f"wall_fraction_range must satisfy 0 < lo <= hi < 0.5, "
                f"got {self.wall_fraction_range}")
        if not 0.0 <= self.approximated_fraction <= 1.0:
            raise ValueError("approximated_fraction must be in [0, 1]")
        if not self.permitted_axes:
            raise ValueError("permitted_axes must name at least one axis")
        unknown = [a for a in self.permitted_axes if a not in ("x", "y", "z")]
        if unknown:
            raise ValueError(f"permitted_axes contains unknown axes {unknown}")


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


def _quantise(value: float, step: int, lo: int, hi: int) -> int:
    """Round to the nearest ``step`` and clamp into [lo, hi]."""
    q = int(round(value / step)) * step
    return max(lo, min(hi, q))


def _weighted(rng: random.Random, items: Sequence[Tuple[Any, float]]) -> Any:
    """Weighted choice with a fixed traversal order (no dict, no set)."""
    r = rng.random()
    cumulative = 0.0
    for value, weight in items:
        cumulative += weight
        if r <= cumulative:
            return value
    return items[-1][0]


def generate_scenario(config: GeneratorConfig,
                      scenario_id: Optional[str] = None) -> Scenario:
    """Generate a reproducible scenario. Same config in, same scenario out."""
    config.validate()
    rng = random.Random(config.seed)

    pool = [m for m in MATERIALS
            if not config.materials or m[0] in config.materials]
    if not pool:
        raise ValueError(f"no known materials in {list(config.materials)}; "
                         f"known: {[m[0] for m in MATERIALS]}")

    l_lo, l_hi = config.length_range_mm
    d_lo, d_hi = config.diameter_range_mm

    items: List[WasteItem] = []
    for n in range(1, config.item_count + 1):
        material, density, group = pool[rng.randrange(len(pool))]

        # Lengths are drawn from a triangular distribution biased short: a real
        # bin holds many offcuts and a few long runs, and that asymmetry is
        # precisely what a shelf packer handles badly and an optimizer handles
        # well. A uniform draw would flatter the baseline.
        raw_length = rng.triangular(l_lo, l_hi, l_lo + (l_hi - l_lo) * 0.35)
        length = _quantise(raw_length, config.length_step_mm, l_lo, l_hi)

        raw_diameter = rng.uniform(d_lo, d_hi)
        outer = _quantise(raw_diameter, config.diameter_step_mm, d_lo, d_hi)

        wall = rng.uniform(*config.wall_fraction_range)
        inner = _quantise(outer * (1.0 - 2.0 * wall), 2, 2, outer - 2)
        if inner >= outer:                          # pragma: no cover - defensive
            inner = outer - 2

        is_approx = rng.random() < config.approximated_fraction
        if is_approx:
            geometry = _weighted(rng, (
                (GeometryType.BENT_TUBE, 0.35),
                (GeometryType.FLAT_SHEET, 0.20),
                (GeometryType.CURVED_SHEET, 0.20),
                (GeometryType.CURVED_PANEL, 0.10),
                (GeometryType.I_BEAM, 0.15),
            ))
            # The declared solid fraction of the enclosing box. Reporting only —
            # packing always uses the full box (see WasteItem docstring).
            fill_ratio = round(rng.uniform(0.25, 0.60), 3)
            inner_d: Optional[int] = None
            material_mm3 = length * outer * outer * fill_ratio
        else:
            geometry = GeometryType.TUBE
            fill_ratio = 1.0
            inner_d = inner
            material_mm3 = (math.pi / 4.0) * (outer ** 2 - inner ** 2) * length

        weight = round(material_mm3 * 1e-9 * density, 3)

        dose = _weighted(rng, DOSE_CLASSES)
        # ILW items get handling priority — the machinery the real system needs,
        # exercised here on simulated labels.
        priority = {"ILW": 2, "LLW": 1, "VLLW": 0}[dose]

        items.append(WasteItem(
            item_id=f"item-{n:03d}",
            length_mm=length,
            outer_diameter_mm=outer,
            geometry_type=geometry,
            inner_diameter_mm=inner_d,
            material=material,
            segregation_group=group,
            weight_kg=weight,
            source_position=Vec3(
                x=int(rng.uniform(0, 900)),
                y=int(rng.uniform(0, 700)),
                z=int(rng.uniform(0, 250))),
            priority=priority,
            dose_class=dose,
            permitted_axes=tuple(Axis(a) for a in config.permitted_axes),
            profile_fill_ratio=fill_ratio,
        ))

    groups = sorted({i.segregation_group for i in items})
    template = make_container(
        config.container_spec, "CNT", (), config.container_overrides,
        segregation_locking=config.segregation_locking)

    return Scenario(
        scenario_id=scenario_id or f"{config.preset}-s{config.seed}",
        preset=config.preset,
        seed=config.seed,
        items=items,
        container_template=template,
        max_containers=config.max_containers,
        description=config.description or _PRESET_DESCRIPTIONS.get(config.preset, ""),
    )


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #

_PRESET_DESCRIPTIONS = {
    "mixed_pipes_small": (
        "Presentation scenario: a small mixed batch of straight pipes into a "
        "single container specification. Fast to plan and easy to read on screen."),
    "mixed_pipes_dense": (
        "Dense batch across a wide length and diameter range. This is the "
        "scenario the baseline-vs-optimized container-count comparison uses."),
    "segregated_materials": (
        "Two segregation groups that may not share a container. Exercises the "
        "hard segregation constraint and the segregation strategy."),
    "late_arrival_replan": (
        "Starts from a valid plan; a high-priority component arrives after "
        "execution has begun, forcing a controlled pause and re-plan."),
    "curated_volume_reduction": (
        "CURATED DEMONSTRATION DATASET — hand-built to expose the specific "
        "weakness of arrival-order shelf packing. Not a random sample and not a "
        "general performance claim."),
    "mixed_geometries": (
        "Includes the five approximated EDF/CEA geometry classes alongside "
        "straight tubes. Approximated items are packed by conservative bounding "
        "box and are labelled as such."),
    "isaac_cylinders_smoke": (
        "PHYSICAL SMOKE TEST for the Isaac Sim execution backend. Four short "
        "bench-scale pipe segments into one open-top bin, sized for a Franka "
        "Emika Panda rather than for a decommissioning cell. It is NOT a "
        "packing benchmark and contributes nothing to the measured "
        "baseline-versus-optimized result."),
    "cut_avoids_extra_container": (
        "CURATED CUT-AWARE DATASET — hand-built so one cuttable pipe forces a "
        "second container whole but its segments fit residual cavities in the "
        "first. The container saving is COMPUTED by the packer, not asserted."),
    "cut_not_worthwhile": (
        "CURATED CUT-AWARE DATASET — a cuttable pipe that already packs into one "
        "container. Cutting can only add process cost, so the whole-process "
        "planner recommends NO cut. Same untouched scoring as the saving case."),
    "cut_result_deviation": (
        "CURATED CUT-AWARE DATASET — the saving geometry, used to demonstrate an "
        "ACTUAL cut result that deviates from the proposal: lineage is updated "
        "from the real segment sizes, packing is re-planned and re-approved."),
}


#: The default preset for a physical Isaac Sim run. Named once here so the
#: launcher, the orchestrator and the smoke validator cannot disagree about it.
ISAAC_SMOKE_PRESET = "isaac_cylinders_smoke"


def preset_config(preset: str, seed: int = 42, **overrides: Any) -> GeneratorConfig:
    """Build a GeneratorConfig for a named preset, with optional overrides."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
    data: Dict[str, Any] = dict(PRESETS[preset])
    data.update(overrides)
    data["preset"] = preset
    data["seed"] = seed
    return GeneratorConfig(**data)


PRESETS: Dict[str, Dict[str, Any]] = {
    # ~16 pipes into a compact box. Small enough to read on screen, tight enough
    # that container count is still decided by packing quality.
    "mixed_pipes_small": dict(
        item_count=16,
        length_range_mm=(300, 1200),
        diameter_range_mm=(60, 200),
        container_spec="compact_box",
        max_containers=4,
    ),
    # The baseline-vs-optimized container-count comparison. 40 items at roughly
    # 1.2x one container's capacity, so both algorithms must open more than one
    # and the question is how many.
    "mixed_pipes_dense": dict(
        item_count=40,
        length_range_mm=(250, 1500),
        diameter_range_mm=(50, 240),
        container_spec="standard_box",
        max_containers=12,
    ),
    # Three materials spanning three segregation groups, with containers that
    # lock to the first group placed. A container may therefore never mix.
    "segregated_materials": dict(
        item_count=28,
        length_range_mm=(300, 1400),
        diameter_range_mm=(60, 220),
        materials=("carbon_steel", "stainless_316L", "zircaloy_4"),
        container_spec="standard_box",
        segregation_locking=True,
        max_containers=12,
    ),
    "late_arrival_replan": dict(
        item_count=18,
        length_range_mm=(300, 1200),
        diameter_range_mm=(60, 200),
        container_spec="compact_box",
        max_containers=6,
    ),
    "mixed_geometries": dict(
        item_count=24,
        length_range_mm=(300, 1400),
        diameter_range_mm=(80, 240),
        approximated_fraction=0.5,
        container_spec="standard_box",
        max_containers=10,
    ),
    # THE ISAAC PHYSICAL SMOKE SCENARIO. Every number here is a hardware
    # constraint, not a packing choice:
    #
    #   item_count 4        one robotic pick-and-place cycle takes ~25 s of
    #                       simulated time, so a 40-item benchmark run is
    #                       twenty minutes of watching an arm. Four items
    #                       exercise the whole sequence, including a second
    #                       item landing on the first.
    #   diameter <= 70 mm   the Panda's parallel gripper opens to 80 mm. A
    #                       larger pipe cannot be grasped at all, and the run
    #                       would fail for a reason that has nothing to do with
    #                       WISEPACK.
    #   length <= 250 mm    must fit the 300 mm bin in at least one horizontal
    #                       orientation with clearance, and must not overhang
    #                       far enough to strike a wall on the way down.
    #   permitted_axes x,y  a top-down gripper cannot stand a pipe on its end
    #                       without a regrasp; see GeneratorConfig.permitted_axes.
    #
    # Kept entirely separate from the benchmark presets above so it can never
    # affect the measured optimization result.
    "isaac_cylinders_smoke": dict(
        item_count=4,
        length_range_mm=(150, 250),
        diameter_range_mm=(40, 70),
        container_spec="isaac_smoke_bin",
        permitted_axes=("x", "y"),
        max_containers=2,
    ),
    # curated_volume_reduction is not generated from ranges — see
    # build_curated_scenario(), which is used instead.
    "curated_volume_reduction": dict(
        item_count=1,               # placeholder, unused
        container_spec="long_box",
        max_containers=6,
    ),
    # The cut-aware curated scenarios are hand-built too — see
    # build_cut_scenario(). These placeholders only satisfy the preset registry.
    "cut_avoids_extra_container": dict(
        item_count=1, container_spec="standard_box", max_containers=8),
    "cut_not_worthwhile": dict(
        item_count=1, container_spec="standard_box", max_containers=8),
    "cut_result_deviation": dict(
        item_count=1, container_spec="standard_box", max_containers=8),
}


# --------------------------------------------------------------------------- #
# Curated scenario
# --------------------------------------------------------------------------- #


def build_curated_scenario(seed: int = 7,
                           scenario_id: str = "curated_volume_reduction") -> Scenario:
    """A transparent, hand-built dataset that exposes the shelf packer's weakness.

    THE CONSTRUCTION IS STATED OPENLY, because a curated demo that hides its
    construction is indistinguishable from a rigged one:

      * The container is the ``long_box``: 1550 x 700 x 700 mm inside.
      * 20 items alternate in arrival order between LONG-THIN (1500 x 100 mm)
        and SHORT-FAT (400 x 300 mm).
      * The shelf baseline takes them in that alternating order at one fixed
        orientation, so every 300 mm-tall short-fat item sets the height of a
        whole shelf level, and the 100 mm-tall long-thin items sharing that
        level waste 200 mm of height each. It also cannot rotate the long items
        to exploit the 700 mm depth.
      * The optimizer sorts by volume, rotates freely, and nests the thin items
        into the residual space beside and above the fat ones.

    NO NUMBER IS ASSERTED HERE. Whatever reduction results is whatever the two
    algorithms compute on this input, and the KPI module records it as a curated
    demonstration measurement rather than a general claim.

    Sizing rationale (published for the same reason): 20 items is the largest
    count for which the optimizer still reaches a SINGLE container, which makes
    the comparison unambiguous — one container is the hard floor, so the
    optimizer cannot be accused of leaving anything on the table. Larger counts
    push both algorithms up by one container and the ratio gets *worse*, not
    better; this was measured across 6/8/10/12/14/16 pairs and several
    thin/fat geometry variants before settling here. It was not seed-fished:
    the dataset is deterministic and contains no random draw at all.
    """
    items: List[WasteItem] = []
    n = 0

    def add(length: int, outer: int, inner: int, material: str, density: float,
            group: str, dose: str, priority: int) -> None:
        nonlocal n
        n += 1
        volume = (math.pi / 4.0) * (outer ** 2 - inner ** 2) * length
        items.append(WasteItem(
            item_id=f"item-{n:03d}",
            length_mm=length,
            outer_diameter_mm=outer,
            geometry_type=GeometryType.TUBE,
            inner_diameter_mm=inner,
            material=material,
            segregation_group=group,
            weight_kg=round(volume * 1e-9 * density, 3),
            source_position=Vec3(x=120 * (n % 7), y=90 * (n % 5), z=40),
            priority=priority,
            dose_class=dose,
        ))

    # 10 alternating pairs: long-thin then short-fat.
    for _ in range(10):
        add(1500, 100, 84, "carbon_steel", 7850.0, "A", "VLLW", 0)
        add(400, 300, 268, "stainless_304", 8000.0, "A", "LLW", 1)

    template = make_container("long_box", "CNT")
    return Scenario(
        scenario_id=scenario_id,
        preset="curated_volume_reduction",
        seed=seed,
        items=items,
        container_template=template,
        max_containers=6,
        description=_PRESET_DESCRIPTIONS["curated_volume_reduction"],
        curated=True,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


_CUT_SCENARIOS = frozenset({
    "cut_avoids_extra_container", "cut_not_worthwhile", "cut_result_deviation"})


def build_cut_scenario(preset: str, seed: int = 7,
                       scenario_id: Optional[str] = None) -> Scenario:
    """Hand-built cut-aware datasets. Curated and disclosed, never seed-fished.

    ``cut_avoids_extra_container`` / ``cut_result_deviation`` (same geometry):
      * container is the ``standard_box`` (inner 1500 x 800 x 600 mm);
      * two 800 mm pipes (OD 360) sit end-on, leaving two 700 mm floor cavities;
      * one cuttable pipe C (1300 mm, OD 340) fits WHOLE only by opening a second
        container, but split into ~two 650 mm segments it drops into the cavities.
      No container count is asserted: the no-cut plan needs 2 and the cut plan
      needs 1 because that is what the geometry-aware packer computes on this
      input (verified in tests/test_cut_optimizer.py).

    ``cut_not_worthwhile``:
      * one cuttable pipe (1400 mm, OD 300) that already packs into ONE container.
      * cutting cannot reduce the container count, so the whole-process planner —
        using the SAME scoring — charges the cut its cost and recommends no cut.
    """
    def pipe(item_id, length, od, *, group="A", cut=False, mc=0, minseg=None):
        density = 7850.0
        volume = (math.pi / 4.0) * (od ** 2 - (od - 40) ** 2) * length
        return WasteItem(
            item_id=item_id, length_mm=length, outer_diameter_mm=od,
            inner_diameter_mm=od - 40, geometry_type=GeometryType.TUBE,
            material="carbon_steel", segregation_group=group,
            weight_kg=round(volume * 1e-9 * density, 3),
            source_position=Vec3(x=80, y=60, z=40), priority=1, dose_class="LLW",
            cut_allowed=cut, maximum_number_of_cuts=mc,
            minimum_segment_length_mm=minseg, protected_end_length_mm=20)

    template = make_container("standard_box", "CNT")
    if preset == "cut_not_worthwhile":
        # A cuttable pipe that already fits one box alongside a small filler.
        # Cutting cannot drop the container count below one, so it never pays.
        items = [pipe("pipe-A", 1400, 300, cut=True, mc=2, minseg=400),
                 pipe("pipe-B", 400, 200)]
        max_containers = 4
    else:
        items = [pipe("pipe-A", 800, 360), pipe("pipe-B", 800, 360),
                 pipe("pipe-C", 1300, 340, cut=True, mc=2, minseg=400)]
        max_containers = 8
    return Scenario(
        scenario_id=scenario_id or preset, preset=preset, seed=seed, items=items,
        container_template=template, max_containers=max_containers,
        description=_PRESET_DESCRIPTIONS[preset], curated=True)


def build_scenario(preset: str, seed: int = 42,
                   scenario_id: Optional[str] = None,
                   **overrides: Any) -> Scenario:
    """Build any preset by name, curated ones included."""
    if preset == "curated_volume_reduction":
        return build_curated_scenario(
            seed=seed, scenario_id=scenario_id or "curated_volume_reduction")
    if preset in _CUT_SCENARIOS:
        return build_cut_scenario(preset, seed=seed, scenario_id=scenario_id)
    return generate_scenario(preset_config(preset, seed, **overrides), scenario_id)


def inject_item(scenario: Scenario, spec: Dict[str, Any],
                item_id: Optional[str] = None) -> WasteItem:
    """Add a late-arriving item to a scenario (the ``item_inject`` event).

    The id continues the existing numbering so the audit trail reads naturally,
    and ``injected`` is set so the dashboard can mark it on the timeline and in
    the Digital Twin view.
    """
    existing = len(scenario.items)
    item = WasteItem(
        item_id=item_id or f"item-{existing + 1:03d}",
        length_mm=int(spec.get("length_mm", 1400)),
        outer_diameter_mm=int(spec.get("outer_diameter_mm", 220)),
        geometry_type=GeometryType(spec.get("geometry_type", "tube")),
        inner_diameter_mm=spec.get("inner_diameter_mm"),
        material=spec.get("material", "stainless_316L"),
        segregation_group=spec.get("segregation_group", "A"),
        weight_kg=float(spec.get("weight_kg", 0.0)),
        source_position=Vec3.from_dict(spec.get("source_position")),
        priority=int(spec.get("priority", 5)),
        dose_class=spec.get("dose_class", "ILW"),
        injected=True,
        profile_fill_ratio=float(spec.get("profile_fill_ratio", 1.0)),
    )
    if item.weight_kg == 0.0:
        density = next((d for name, d, _ in MATERIALS if name == item.material), 7850.0)
        item.weight_kg = round(item.material_volume_mm3 * 1e-9 * density, 3)
    scenario.items.append(item)
    return item


__all__ = [
    "MATERIALS", "DOSE_CLASSES", "CONTAINER_SPECS", "make_container",
    "GeneratorConfig", "generate_scenario", "PRESETS", "preset_config",
    "build_curated_scenario", "build_cut_scenario", "build_scenario", "inject_item",
]
