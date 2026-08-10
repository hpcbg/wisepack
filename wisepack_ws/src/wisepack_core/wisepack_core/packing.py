"""Packing algorithms: the industrial baseline and the geometry-aware optimizer.

Both produce a PackingPlan over the same Scenario, and both are checked by the
same independent validator (validator.py) before anything is reported.

BASELINE — ``arrival_order_shelf``
    Items in arrival order, one fixed orientation, filled as shelves: left to
    right along a row, rows front to back on a level, levels bottom to top, new
    container when the current one cannot take the next item. This is a real
    algorithm used in real material handling, not a strawman: it is O(n), needs
    no lookahead, and a human with a clipboard can execute it. It is called
    ``arrival_order_shelf`` and not "manual industry average" because no evidence
    for the latter exists in this repository.

OPTIMIZED — ``geometry_aware_ep_bfd``
    Best-fit-decreasing over extreme points: candidate positions are maintained
    per container, every permitted axis-aligned orientation is evaluated at every
    candidate, feasibility uses the same hard-constraint rules as the validator,
    and the best (item, container, position, axis) is chosen by a deterministic
    fit score. Multi-start over several seeded orderings, then a container-
    consolidation improvement pass, then the best-scoring solution wins.

Determinism is a requirement, not a nicety: the acceptance demo must produce the
same containers on the reviewer's laptop as on the presenter's. Every random
choice is drawn from a ``random.Random`` seeded from the scenario, iteration is
over sorted collections, and ties are broken by explicit rules rather than by
dict or set ordering.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .domain import (
    Axis, Box, Container, ContainerStatus, PackingPlan, Placement, Scenario,
    Strategy, Vec3, WasteItem,
)
from .validator import DEFAULT_VALIDATION, PlacementValidator, ValidationConfig

BASELINE_ALGORITHM = "arrival_order_shelf"
OPTIMIZED_ALGORITHM = "geometry_aware_ep_bfd"


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights of the ranking objective.

    HARD CONSTRAINTS ARE NOT IN HERE. Container bounds, collisions, payload and
    segregation are enforced by feasibility filtering; they can never be traded
    away for density no matter how the weights are set. What remains are genuine
    preferences between feasible plans.

        score = packing_density
              - container_count_penalty
              - unplaced_volume_penalty
              - segregation_penalty
              - retrievability_penalty
              - excessive_clearance_penalty

    ``segregation_penalty`` is non-zero even though mixing is already forbidden:
    it discourages *spreading* one group thinly across many containers, which is
    legal but operationally poor.
    """

    container_count: float = 0.35
    unplaced_volume: float = 2.00
    segregation_mixing: float = 0.20
    retrievability: float = 0.10
    excessive_clearance: float = 0.05

    def to_dict(self) -> Dict[str, float]:
        return {
            "container_count": self.container_count,
            "unplaced_volume": self.unplaced_volume,
            "segregation_mixing": self.segregation_mixing,
            "retrievability": self.retrievability,
            "excessive_clearance": self.excessive_clearance,
        }


#: The three operator-selectable strategies from the proposal. They differ ONLY
#: in these weights — same constraints, same search, same validator.
STRATEGY_WEIGHTS: Dict[Strategy, ObjectiveWeights] = {
    # Fewest containers, tightest fill. The default and the KPI4 configuration.
    Strategy.MAX_DENSITY: ObjectiveWeights(
        container_count=0.35, unplaced_volume=2.00, segregation_mixing=0.20,
        retrievability=0.10, excessive_clearance=0.05),
    # Penalise burying items deep: heavier retrievability term, lighter density
    # pressure. Expect this to use more containers — that is the trade-off, and
    # the dashboard shows it rather than hiding it.
    Strategy.RETRIEVABILITY: ObjectiveWeights(
        container_count=0.15, unplaced_volume=2.00, segregation_mixing=0.20,
        retrievability=0.60, excessive_clearance=0.05),
    # Keep each segregation group consolidated, even at a container cost.
    Strategy.SEGREGATION: ObjectiveWeights(
        container_count=0.20, unplaced_volume=2.00, segregation_mixing=1.00,
        retrievability=0.10, excessive_clearance=0.05),
}


def score_plan(plan: PackingPlan, scenario: Scenario,
               weights: ObjectiveWeights) -> float:
    """Rank a feasible plan. Higher is better. Deterministic.

    All terms are normalised to roughly [0, 1] so the weights read as relative
    importances rather than as unit conversions.
    """
    used = plan.containers_used
    if not used:
        # Nothing placed: the unplaced term should dominate and stay negative.
        return -weights.unplaced_volume

    density = plan.utilization_pct / 100.0

    # Container count, normalised by the information-free lower bound (total
    # occupied volume / one container's capacity). A plan that hits the bound
    # pays ~1 unit of penalty * weight; one that doubles it pays ~2.
    capacity_each = used[0].capacity_mm3
    lower_bound = max(1.0, scenario.total_occupied_volume_mm3 / capacity_each)
    container_term = len(used) / lower_bound

    total_vol = max(1, scenario.total_occupied_volume_mm3)
    unplaced_vol = sum(
        (scenario.item(i).occupied_volume_mm3 if scenario.item(i) else 0)
        for i in plan.unplaced_item_ids)
    unplaced_term = unplaced_vol / total_vol

    # Segregation spread: how many (group, container) pairs exist beyond the
    # minimum of one container per group.
    pairs = {(scenario.item(p.item_id).segregation_group, p.container_id)
             for p in plan.placements if scenario.item(p.item_id)}
    groups = {g for g, _ in pairs}
    spread_term = 0.0 if not groups else (len(pairs) - len(groups)) / max(1, len(pairs))

    # Retrievability: how deeply buried the average item is, as a fraction of
    # container height. An item at the floor under a full stack scores worst.
    burial: List[float] = []
    for container in used:
        placements = plan.placements_for(container.container_id)
        for p in placements:
            above = sum(1 for q in placements
                        if q is not p
                        and q.position.z >= p.top_z_mm
                        and p.box.footprint_overlap_mm2(q.box) > 0)
            burial.append(min(1.0, above / 4.0))
    retrieval_term = (sum(burial) / len(burial)) if burial else 0.0

    # Excessive clearance: gaps large enough to have held another item. Uses the
    # smallest item as the yardstick, so "excessive" is scenario-relative.
    clearance_term = 0.0
    if plan.placements and scenario.items:
        smallest = min(i.occupied_volume_mm3 for i in scenario.items)
        wasted = plan.unused_capacity_mm3
        clearance_term = min(1.0, (wasted / smallest) / max(1, len(scenario.items)))

    return (density
            - weights.container_count * container_term
            - weights.unplaced_volume * unplaced_term
            - weights.segregation_mixing * spread_term
            - weights.retrievability * retrieval_term
            - weights.excessive_clearance * clearance_term)


# --------------------------------------------------------------------------- #
# Shared container bookkeeping
# --------------------------------------------------------------------------- #


class _ContainerState:
    """One container under construction, plus its extreme-point candidates."""

    def __init__(self, container: Container) -> None:
        self.container = container
        self.contents: List[Tuple[WasteItem, Box]] = []
        self.placements: List[Placement] = []
        self.payload_kg = 0.0
        # Extreme points: candidate min-corners for the next box. Seeded with the
        # container origin; each placement contributes three new projections.
        self.points: List[Vec3] = [Vec3(0, 0, 0)]

    @property
    def occupied_mm3(self) -> int:
        return sum(b.volume_mm3 for _, b in self.contents)

    def add(self, item: WasteItem, axis: Axis, position: Vec3,
            order: int) -> Placement:
        # A locking container commits to this item's segregation group. Done
        # here, in the one place a placement is ever recorded, so no packer can
        # forget it — and the validator then re-derives the same conclusion.
        self.container.lock_to_group(item.segregation_group)
        size = item.size_for_axis(axis)
        box = Box(position, size)
        self.contents.append((item, box))
        self.payload_kg += item.weight_kg
        placement = Placement(
            item_id=item.item_id, container_id=self.container.container_id,
            position=position, axis=axis, size=size, placement_order=order)
        self.placements.append(placement)
        self._extend_points(box)
        return placement

    def _extend_points(self, box: Box) -> None:
        """Add the three face-projected extreme points of a newly placed box.

        Points already occupied by the new box are dropped. Keeping them would
        cost a feasibility check per candidate forever; dropping them is safe
        because any position they could still serve is dominated by one of the
        new points or by an existing one.
        """
        m = box.max_corner
        self.points = [p for p in self.points if not _inside(p, box)]
        for cand in (Vec3(m.x, box.origin.y, box.origin.z),
                     Vec3(box.origin.x, m.y, box.origin.z),
                     Vec3(box.origin.x, box.origin.y, m.z)):
            if cand not in self.points:
                self.points.append(cand)
        # Deterministic candidate order: bottom-most, then front, then left. The
        # search relies on this for reproducibility, and it also biases toward
        # low, stable placements, which is what a real gripper prefers.
        self.points.sort(key=lambda v: (v.z, v.y, v.x))


def _inside(point: Vec3, box: Box) -> bool:
    m = box.max_corner
    return (box.origin.x <= point.x < m.x
            and box.origin.y <= point.y < m.y
            and box.origin.z <= point.z < m.z)


def _new_container(template: Container, index: int) -> Container:
    """Allocate container ``index`` from the template's id stem.

    Indices are global to a run, not per-plan: a re-plan must never re-issue the
    id of a container that has been retired (marked unavailable), or the
    retirement is silently undone and the packer fills a box that is out of
    service. The engine advances the offset across re-plans for exactly that
    reason.
    """
    return template.respec(f"{template.container_id}-{index:02d}")


# --------------------------------------------------------------------------- #
# Baseline: arrival-order shelf
# --------------------------------------------------------------------------- #


@dataclass
class ShelfCursor:
    """The three-level cursor of a shelf packer."""

    x: int = 0
    y: int = 0
    z: int = 0
    row_depth: int = 0      # deepest item in the current row
    level_height: int = 0   # tallest item on the current level


def pack_baseline(scenario: Scenario, *,
                  axis: Axis = Axis.X,
                  validation: ValidationConfig = DEFAULT_VALIDATION,
                  plan_id: Optional[str] = None,
                  container_index_offset: int = 0) -> PackingPlan:
    """Arrival-order shelf packing with one fixed orientation.

    Deliberately simple, but deliberately NOT a strawman. What it lacks is
    optimization, not competence:
      * items are taken in the order they arrive — no sorting, no lookahead,
        no reconsideration of an item once placed;
      * every item lies along the same axis — no orientation search;
      * within a container: a new row when the current one is full, a new level
        when the rows are exhausted, resting on a shelf plate;
      * it WILL put an item in whichever already-open container accepts it —
        that is what an operator with several open boxes does, and denying it
        would inflate the baseline's container count on segregated scenarios
        without telling us anything about packing quality;
      * an item that fits no open container and no fresh one is reported
        unplaced rather than silently dropped.

    The gap to the optimizer is therefore genuinely about geometry: sorting,
    orientation, and using the space between and above items instead of paying
    for a full shelf level per item height.
    """
    started = time.perf_counter()
    template = scenario.container_template
    if template is None:
        raise ValueError("scenario has no container_template")

    states: List[_ContainerState] = []
    cursors: List[ShelfCursor] = []
    unplaced: List[str] = []
    order = 0

    def open_container() -> _ContainerState:
        state = _ContainerState(_new_container(
            template, container_index_offset + len(states) + 1))
        states.append(state)
        cursors.append(ShelfCursor())
        return state

    open_container()

    def try_place(index: int, item: WasteItem) -> bool:
        """Advance container ``index``'s shelf cursor and place the item if it fits."""
        state, cur = states[index], cursors[index]
        size = item.size_for_axis(axis)
        inner = state.container.inner_size

        if not state.container.is_usable or \
                not state.container.accepts_group(item.segregation_group):
            return False
        if state.payload_kg + item.weight_kg > state.container.max_payload_kg:
            return False

        # Advance the cursor: next row, then next level. Two passes are enough
        # — row exhaustion can cascade into level exhaustion, but no further.
        for _ in range(2):
            if (cur.x + size.x <= inner.x and cur.y + size.y <= inner.y
                    and cur.z + size.z <= inner.z):
                break
            if cur.x + size.x > inner.x:            # next row
                cur.x = 0
                cur.y += cur.row_depth
                cur.row_depth = 0
            if cur.y + size.y > inner.y:            # next level
                cur.x = 0
                cur.y = 0
                cur.row_depth = 0
                cur.z += cur.level_height
                cur.level_height = 0

        if not (cur.x + size.x <= inner.x and cur.y + size.y <= inner.y
                and cur.z + size.z <= inner.z):
            return False

        nonlocal order
        order += 1
        state.add(item, axis, Vec3(cur.x, cur.y, cur.z), order)
        # A level above the floor stands on a shelf plate. Record it so the
        # validator's support check knows the level is physically held — and so
        # the plan is honest about *how* it stands up.
        if cur.z > 0:
            levels = set(state.container.shelf_levels_mm)
            levels.add(cur.z)
            state.container.shelf_levels_mm = tuple(sorted(levels))
        cur.x += size.x
        cur.row_depth = max(cur.row_depth, size.y)
        cur.level_height = max(cur.level_height, size.z)
        return True

    for item in scenario.items:
        # Any already-open container that accepts this item, oldest first. This
        # is not lookahead — it never revisits a placement or reorders an item.
        if any(try_place(i, item) for i in range(len(states))):
            continue
        open_container()
        if not try_place(len(states) - 1, item):
            unplaced.append(item.item_id)

    plan = _assemble_plan(
        plan_id or f"plan-baseline-{scenario.scenario_id}",
        scenario, BASELINE_ALGORITHM, Strategy.MAX_DENSITY, states, unplaced,
        (time.perf_counter() - started) * 1000.0)
    plan.details.update({
        "fixed_axis": axis.value,
        "sorting": "arrival order (none)",
        "lookahead": False,
        "uses_shelf_plates": True,
        "shelf_levels_mm": {c.container_id: list(c.shelf_levels_mm)
                            for c in plan.containers_used},
        "note": ("Deliberately simple industrial baseline. Named for what it "
                 "does; no claim is made that it represents any particular "
                 "site's current practice. Levels rest on zero-thickness shelf "
                 "plates, which is the assumption most favourable to it."),
    })
    PlacementValidator(validation).validate_plan(plan, scenario)
    return plan


# --------------------------------------------------------------------------- #
# Optimizer: extreme-point best-fit-decreasing, multi-start
# --------------------------------------------------------------------------- #


@dataclass
class OptimizerConfig:
    """Search budget and behaviour. All of it deterministic given ``seed``."""

    strategy: Strategy = Strategy.MAX_DENSITY
    #: Number of seeded starting orderings to try. Each start is a full pack.
    restarts: int = 6
    #: Wall-clock ceiling. On expiry the best complete solution so far is
    #: returned and ``details.timed_out`` is set — never a partial plan.
    time_budget_ms: float = 4000.0
    #: Run the container-consolidation improvement pass.
    improve: bool = True
    #: Max consolidation attempts per solution.
    improve_rounds: int = 3
    seed: int = 0
    #: First container index to issue. Advanced by the workflow across re-plans
    #: so retired container ids are never re-used.
    container_index_offset: int = 0
    validation: ValidationConfig = field(default=DEFAULT_VALIDATION)

    def __post_init__(self) -> None:
        # Configs arrive from YAML, JSON and dashboard query strings, where a
        # strategy is a plain string. Coerce once here rather than leaving every
        # consumer to guess whether it holds an enum or a str.
        self.strategy = Strategy(self.strategy)


def pack_optimized(scenario: Scenario, *,
                   config: Optional[OptimizerConfig] = None,
                   plan_id: Optional[str] = None) -> PackingPlan:
    """Geometry-aware packing. Returns the best-scoring feasible plan found."""
    cfg = config or OptimizerConfig(seed=scenario.seed)
    template = scenario.container_template
    if template is None:
        raise ValueError("scenario has no container_template")

    started = time.perf_counter()
    deadline = started + cfg.time_budget_ms / 1000.0
    weights = STRATEGY_WEIGHTS[cfg.strategy]
    validator = PlacementValidator(cfg.validation)

    best: Optional[PackingPlan] = None
    best_score = float("-inf")
    attempts: List[Dict[str, Any]] = []
    timed_out = False

    for restart, ordering in enumerate(_orderings(scenario, cfg)):
        if time.perf_counter() > deadline and restart > 0:
            timed_out = True
            break

        states, unplaced = _pack_ordering(
            ordering, template, scenario, validator, deadline,
            cfg.container_index_offset)

        if cfg.improve:
            states, unplaced = _consolidate(
                states, unplaced, template, scenario, validator, cfg, deadline)

        candidate = _assemble_plan(
            f"{plan_id or f'plan-optimized-{scenario.scenario_id}'}-r{restart}",
            scenario, OPTIMIZED_ALGORITHM, cfg.strategy, states, unplaced, 0.0)
        report = validator.validate_plan(candidate, scenario)
        score = score_plan(candidate, scenario, weights) if report.valid else float("-inf")

        attempts.append({
            "restart": restart,
            "ordering": ordering.name,
            "containers": candidate.containers_required,
            "unplaced": len(candidate.unplaced_item_ids),
            "utilization_pct": round(candidate.utilization_pct, 2),
            "score": round(score, 5) if score != float("-inf") else None,
            "valid": report.valid,
        })

        if score > best_score:
            best_score, best = score, candidate

    if best is None:                                # pragma: no cover - defensive
        raise RuntimeError("optimizer produced no candidate solutions")

    best.plan_id = plan_id or f"plan-optimized-{scenario.scenario_id}"
    best.computation_time_ms = (time.perf_counter() - started) * 1000.0
    best.objective_score = best_score
    best.details.update({
        "strategy": cfg.strategy.value,
        "objective_weights": weights.to_dict(),
        "restarts_run": len(attempts),
        "restarts_configured": cfg.restarts,
        "timed_out": timed_out,
        "time_budget_ms": cfg.time_budget_ms,
        "seed": cfg.seed,
        "attempts": attempts,
        "support_check": cfg.validation.min_support_fraction,
    })
    validator.validate_plan(best, scenario)
    return best


# -- orderings --------------------------------------------------------------- #


@dataclass
class _Ordering:
    name: str
    items: List[WasteItem]


def _orderings(scenario: Scenario, cfg: OptimizerConfig) -> Iterable[_Ordering]:
    """Deterministic starting orderings, most promising first.

    The first three are the classical decreasing rules. The remainder are seeded
    perturbations of the volume-decreasing order: enough randomness to escape a
    bad tie-break, none of the irreproducibility.
    """
    items = list(scenario.items)

    def by(key, name) -> _Ordering:
        # item_id is always the final tie-break, so equal items never depend on
        # the input list order (which a dynamic event can change).
        return _Ordering(name, sorted(items, key=lambda i: (key(i), i.item_id)))

    yield by(lambda i: (-i.priority, -i.occupied_volume_mm3), "priority_then_volume_desc")
    yield by(lambda i: (-i.occupied_volume_mm3, -i.length_mm), "volume_desc")
    yield by(lambda i: (-i.length_mm, -i.outer_diameter_mm), "length_desc")
    yield by(lambda i: (-i.outer_diameter_mm, -i.length_mm), "diameter_desc")
    # Group-major: keeps segregation groups together from the start.
    yield by(lambda i: (i.segregation_group, -i.occupied_volume_mm3), "group_then_volume_desc")

    base = sorted(items, key=lambda i: (-i.occupied_volume_mm3, i.item_id))
    for k in range(max(0, cfg.restarts - 5)):
        rng = random.Random(cfg.seed * 1000 + k)
        shuffled = list(base)
        # Local perturbation: swap a few neighbouring pairs. A full shuffle
        # destroys the decreasing property that makes BFD work at all.
        #
        # A SINGLE ITEM HAS NO NEIGHBOURING PAIR. The earlier `max(1, len - 1)`
        # guard stopped `randrange` seeing an empty range but then produced
        # index 0 and read `shuffled[1]`, so a one-item scenario raised
        # IndexError — which is exactly what a single perceived object is.
        if len(shuffled) > 1:
            for _ in range(max(1, len(shuffled) // 6)):
                i = rng.randrange(0, len(shuffled) - 1)
                shuffled[i], shuffled[i + 1] = shuffled[i + 1], shuffled[i]
        yield _Ordering(f"volume_desc_perturbed_{k}", shuffled)


# -- one full pass ----------------------------------------------------------- #


def _pack_ordering(ordering: _Ordering, template: Container, scenario: Scenario,
                   validator: PlacementValidator, deadline: float,
                   index_offset: int = 0
                   ) -> Tuple[List[_ContainerState], List[str]]:
    """Place every item of ``ordering`` using best-fit over extreme points."""
    states: List[_ContainerState] = [
        _ContainerState(_new_container(template, index_offset + 1))]
    unplaced: List[str] = []
    order = 0

    for item in ordering.items:
        best_choice: Optional[Tuple[float, int, Axis, Vec3]] = None

        for idx, state in enumerate(states):
            if not state.container.is_usable:
                continue
            if not state.container.accepts_group(item.segregation_group):
                continue
            if state.payload_kg + item.weight_kg > state.container.max_payload_kg:
                continue
            for axis in item.permitted_axes:
                for point in state.points:
                    if not validator.placement_is_feasible(
                            item, axis, point, state.container,
                            state.contents, state.payload_kg):
                        continue
                    fit = _fit_score(item, axis, point, state)
                    key = (fit, idx, axis, point)
                    if best_choice is None or _better(key, best_choice):
                        best_choice = key
            # Time check per container, not per candidate: cheap, and it cannot
            # abandon an item mid-evaluation and leave it half-placed.
            if time.perf_counter() > deadline:
                break

        if best_choice is None:
            # No open container can take it: open one and try there. If a fresh
            # container cannot either, the item genuinely does not fit.
            fresh = _ContainerState(
                _new_container(template, index_offset + len(states) + 1))
            choice = None
            for axis in item.permitted_axes:
                for point in fresh.points:
                    if validator.placement_is_feasible(
                            item, axis, point, fresh.container, fresh.contents, 0.0):
                        choice = (axis, point)
                        break
                if choice:
                    break
            if choice is None:
                unplaced.append(item.item_id)
                continue
            states.append(fresh)
            order += 1
            fresh.add(item, choice[0], choice[1], order)
            continue

        _, idx, axis, point = best_choice
        order += 1
        states[idx].add(item, axis, point, order)

    return states, unplaced


def _better(a: Tuple[float, int, Axis, Vec3],
            b: Tuple[float, int, Axis, Vec3]) -> bool:
    """Strict ordering with explicit tie-breaks — never relies on dict order."""
    if a[0] != b[0]:
        return a[0] > b[0]
    if a[1] != b[1]:
        return a[1] < b[1]                          # prefer the earlier container
    pa, pb = a[3], b[3]
    if pa.z != pb.z:
        return pa.z < pb.z                          # prefer lower
    if pa.y != pb.y:
        return pa.y < pb.y
    if pa.x != pb.x:
        return pa.x < pb.x
    return a[2].value < b[2].value                  # stable axis tie-break


def _fit_score(item: WasteItem, axis: Axis, point: Vec3,
               state: _ContainerState) -> float:
    """How good is this candidate placement? Higher is better.

    This is a FIXED-BIN problem, not strip packing, so the dominant term is
    contact — hugging walls and existing items is what eliminates the unusable
    slivers that force an extra container. Height is only a secondary tie-break.

    An earlier version weighted height twice as heavily as contact. That is the
    right objective when minimising the height of an open-topped stack, and the
    wrong one here: it made the optimizer lay one flat layer and then spill
    sideways into a second container rather than build upward inside the first.

      * contact ratio  — shared face area with walls and neighbours (primary)
      * height cost    — mild preference for keeping the stack low (secondary)
      * corner bias    — deterministic fill direction among equals (tie-break)
    """
    size = item.size_for_axis(axis)
    inner = state.container.inner_size
    box = Box(point, size)

    current_top = max((b.max_corner.z for _, b in state.contents), default=0)
    new_top = max(current_top, box.max_corner.z)
    height_cost = new_top / max(1, inner.z)

    # Contact area with walls.
    contact = 0
    if point.x == 0:
        contact += size.y * size.z
    if point.y == 0:
        contact += size.x * size.z
    if point.z == 0:
        contact += size.x * size.y
    if box.max_corner.x == inner.x:
        contact += size.y * size.z
    if box.max_corner.y == inner.y:
        contact += size.x * size.z
    # Contact area with already-placed items (touching faces).
    for _, other in state.contents:
        if box.gap_to(other) == 0:
            contact += _touch_area(box, other)
    surface = 2 * (size.x * size.y + size.y * size.z + size.x * size.z)
    contact_ratio = contact / max(1, surface)

    corner_bias = 1.0 - ((point.x + point.y + point.z)
                         / max(1, inner.x + inner.y + inner.z))

    return 1.0 * contact_ratio - 0.35 * height_cost + 0.05 * corner_bias


def _touch_area(a: Box, b: Box) -> int:
    """Area of the shared face between two touching, non-overlapping boxes."""
    am, bm = a.max_corner, b.max_corner
    ox = min(am.x, bm.x) - max(a.origin.x, b.origin.x)
    oy = min(am.y, bm.y) - max(a.origin.y, b.origin.y)
    oz = min(am.z, bm.z) - max(a.origin.z, b.origin.z)
    if am.x == b.origin.x or bm.x == a.origin.x:
        return max(0, oy) * max(0, oz)
    if am.y == b.origin.y or bm.y == a.origin.y:
        return max(0, ox) * max(0, oz)
    if am.z == b.origin.z or bm.z == a.origin.z:
        return max(0, ox) * max(0, oy)
    return 0


# -- improvement: container consolidation ------------------------------------ #


def _consolidate(states: List[_ContainerState], unplaced: List[str],
                 template: Container, scenario: Scenario,
                 validator: PlacementValidator, cfg: OptimizerConfig,
                 deadline: float) -> Tuple[List[_ContainerState], List[str]]:
    """Try to empty the least-full container into the others.

    This is the pass that turns "one item spilled into a second container" into
    "one container", which is the difference the KPI actually measures. It is
    strictly improving: if the emptying fails, the original arrangement is kept
    untouched.
    """
    for _ in range(cfg.improve_rounds):
        if len(states) < 2 or time.perf_counter() > deadline:
            break
        # Emptiest container first — fewest items to rehome.
        victim_idx = min(range(len(states)), key=lambda i: states[i].occupied_mm3)
        victim = states[victim_idx]
        if not victim.contents:
            states.pop(victim_idx)
            continue

        others = [s for i, s in enumerate(states) if i != victim_idx]
        # Rebuild the survivors from scratch so the extreme points are clean.
        rebuilt = [_rebuild(s) for s in others]
        moved: List[Placement] = []
        # Largest first: if the big ones do not fit elsewhere, stop early.
        for item, _box in sorted(victim.contents,
                                 key=lambda c: (-c[1].volume_mm3, c[0].item_id)):
            choice = _find_slot(item, rebuilt, validator)
            if choice is None:
                break
            idx, axis, point = choice
            moved.append(rebuilt[idx].add(item, axis, point,
                                          len(moved) + 1))
        if len(moved) == len(victim.contents):
            # Every item rehomed — the victim container is gone.
            states = rebuilt
            _renumber(states, cfg.container_index_offset)
        else:
            break
    return states, unplaced


def _rebuild(state: _ContainerState) -> _ContainerState:
    fresh = _ContainerState(state.container.respec(state.container.container_id))
    for order, (item, box) in enumerate(
            sorted(state.contents, key=lambda c: (c[1].origin.z, c[1].origin.y,
                                                  c[1].origin.x, c[0].item_id)), 1):
        axis = _axis_of(item, box.size)
        fresh.add(item, axis, box.origin, order)
    return fresh


def _axis_of(item: WasteItem, size: Vec3) -> Axis:
    for axis in (Axis.X, Axis.Y, Axis.Z):
        if item.size_for_axis(axis).as_tuple() == size.as_tuple():
            return axis
    return item.permitted_axes[0]                   # pragma: no cover - defensive


def _find_slot(item: WasteItem, states: Sequence[_ContainerState],
               validator: PlacementValidator) -> Optional[Tuple[int, Axis, Vec3]]:
    best: Optional[Tuple[float, int, Axis, Vec3]] = None
    for idx, state in enumerate(states):
        if not state.container.is_usable or \
                not state.container.accepts_group(item.segregation_group):
            continue
        for axis in item.permitted_axes:
            for point in state.points:
                if validator.placement_is_feasible(
                        item, axis, point, state.container, state.contents,
                        state.payload_kg):
                    key = (_fit_score(item, axis, point, state), idx, axis, point)
                    if best is None or _better(key, best):
                        best = key
    return None if best is None else (best[1], best[2], best[3])


def _renumber(states: List[_ContainerState], offset: int = 0) -> None:
    """Give surviving containers contiguous ids after a consolidation."""
    for n, state in enumerate(states, offset + 1):
        base = state.container.container_id.rsplit("-", 1)[0]
        new_id = f"{base}-{n:02d}"
        state.container.container_id = new_id
        for p in state.placements:
            p.container_id = new_id


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _assemble_plan(plan_id: str, scenario: Scenario, algorithm: str,
                   strategy: Strategy, states: Sequence[_ContainerState],
                   unplaced: Sequence[str], elapsed_ms: float) -> PackingPlan:
    containers: List[Container] = []
    placements: List[Placement] = []
    for state in states:
        container = state.container
        container.placements = list(state.placements)
        container.status = (ContainerStatus.FILLING if state.placements
                            else ContainerStatus.AVAILABLE)
        containers.append(container)
        placements.extend(state.placements)

    # Execution order: container by container, then bottom-up within a container.
    # The robot empties one container before starting the next, and never has to
    # reach under something it already placed.
    placements.sort(key=lambda p: (p.container_id, p.position.z, p.position.y,
                                   p.position.x, p.item_id))
    for n, p in enumerate(placements, 1):
        p.placement_order = n

    return PackingPlan(
        plan_id=plan_id, scenario_id=scenario.scenario_id, algorithm=algorithm,
        strategy=strategy, containers=containers, placements=placements,
        unplaced_item_ids=list(unplaced), computation_time_ms=elapsed_ms)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def select_plan(baseline: PackingPlan, optimized: PackingPlan,
                scenario: Scenario,
                weights: Optional[ObjectiveWeights] = None) -> Tuple[PackingPlan, str]:
    """Choose the plan to execute. Returns (plan, human-readable reason).

    The optimized plan does NOT win by default. It wins on the objective, and if
    it loses the baseline is selected and the fact is reported — the alternative
    is a demo that claims an improvement it did not achieve, which is the exact
    failure mode this function exists to prevent.
    """
    weights = weights or STRATEGY_WEIGHTS[optimized.strategy]

    if not optimized.is_valid and baseline.is_valid:
        return baseline, ("optimized plan failed validation "
                          f"({len(optimized.constraint_violations)} violations); "
                          "baseline selected")
    if not baseline.is_valid and optimized.is_valid:
        return optimized, "baseline plan failed validation; optimized selected"
    if not baseline.is_valid and not optimized.is_valid:
        return baseline, "BOTH plans failed validation — nothing is executable"

    # Unplaced items dominate: a denser plan that abandons waste is not better.
    if len(optimized.unplaced_item_ids) != len(baseline.unplaced_item_ids):
        if len(optimized.unplaced_item_ids) < len(baseline.unplaced_item_ids):
            return optimized, (f"optimized leaves "
                               f"{len(optimized.unplaced_item_ids)} unplaced vs "
                               f"{len(baseline.unplaced_item_ids)} for baseline")
        return baseline, (f"optimized leaves more items unplaced "
                          f"({len(optimized.unplaced_item_ids)} vs "
                          f"{len(baseline.unplaced_item_ids)}); baseline selected")

    opt_score = score_plan(optimized, scenario, weights)
    base_score = score_plan(baseline, scenario, weights)
    if opt_score >= base_score:
        return optimized, (f"optimized scores {opt_score:.4f} >= baseline "
                           f"{base_score:.4f} "
                           f"({optimized.containers_required} vs "
                           f"{baseline.containers_required} containers)")
    return baseline, (f"optimized scored WORSE ({opt_score:.4f} < "
                      f"{base_score:.4f}); baseline retained as selected plan")


__all__ = [
    "BASELINE_ALGORITHM", "OPTIMIZED_ALGORITHM", "ObjectiveWeights",
    "STRATEGY_WEIGHTS", "score_plan", "pack_baseline", "pack_optimized",
    "OptimizerConfig", "select_plan",
]
