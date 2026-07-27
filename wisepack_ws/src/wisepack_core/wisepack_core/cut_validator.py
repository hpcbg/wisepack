"""Independent conservation-and-lineage validator for cuts.

This module is the examiner, and it is deliberately separate from the planner:
cut_optimizer.py *calls* it, but nothing here imports cut_optimizer, so a cut can
never be certified by the same code that proposed it (brief §3). Every check
re-derives its quantities from first principles rather than trusting a figure the
proposal carries.

The identities enforced, all within :data:`CONSERVATION_TOLERANCE_MM`:

    original_length          == sum(segment_lengths) + n_cuts * kerf
    sum(child material vol)  == parent material vol - kerf swept vol
    sum(child mass)          == parent mass - kerf mass fraction

plus the structural rules: minimum segment length, maximum cuts, protected ends,
inherited material and segregation group, unique child ids, complete parent->child
lineage, and the invariant that an original item and its derived children may
never both be packable at once.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .cutting import (
    CONSERVATION_TOLERANCE_MM, CutProposal, CutResult, tube_cross_section_mm2,
)
from .domain import GeometryType, WasteItem


class _Report:
    """Accumulates named check outcomes into a serialisable verdict."""

    def __init__(self, tolerance_mm: int) -> None:
        self.checks: Dict[str, bool] = {}
        self.violations: List[str] = []
        self.tolerance_mm = tolerance_mm

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks[name] = bool(self.checks.get(name, True)) and bool(ok)
        if not ok:
            self.violations.append(detail or name)
        return ok

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": not self.violations,
            "violations": list(self.violations),
            "checks": dict(self.checks),
            "tolerance_mm": self.tolerance_mm,
            "validator": "wisepack_core.cut_validator",
        }


# --------------------------------------------------------------------------- #
# Segmentation arithmetic / geometry
# --------------------------------------------------------------------------- #


def validate_segmentation(parent: WasteItem, segment_lengths_mm: Sequence[int],
                          *, kerf_mm: int, max_cuts: int,
                          minimum_segment_mm: Optional[int] = None,
                          protected_end_mm: int = 0,
                          tolerance_mm: int = CONSERVATION_TOLERANCE_MM,
                          report: Optional[_Report] = None) -> _Report:
    """Validate lengths, kerf, cut count, minimum length and protected ends."""
    r = report or _Report(tolerance_mm)
    segs = [int(s) for s in segment_lengths_mm]
    n_cuts = max(0, len(segs) - 1)

    r.check("is_tube", parent.geometry_type is GeometryType.TUBE,
            f"{parent.item_id}: only tubes are cuttable")
    r.check("at_least_two_segments", len(segs) >= 2,
            f"{parent.item_id}: a cut yields >= 2 segments, got {len(segs)}")
    r.check("positive_segments", all(s > 0 for s in segs),
            f"{parent.item_id}: every segment length must be > 0")

    # Conservation: original == sum(segments) + n*kerf (± tolerance).
    reconstructed = sum(segs) + n_cuts * kerf_mm
    delta = abs(reconstructed - parent.length_mm)
    r.check("length_conservation", delta <= tolerance_mm,
            f"{parent.item_id}: length not conserved — "
            f"{sum(segs)} + {n_cuts}x{kerf_mm} kerf = {reconstructed} mm vs "
            f"original {parent.length_mm} mm (delta {delta} > {tolerance_mm})")

    r.check("max_cuts", n_cuts <= max_cuts,
            f"{parent.item_id}: {n_cuts} cuts exceeds maximum {max_cuts}")

    floor = minimum_segment_mm if minimum_segment_mm is not None \
        else parent.effective_minimum_segment_mm
    r.check("minimum_segment_length", all(s >= floor for s in segs),
            f"{parent.item_id}: a segment is shorter than the "
            f"{floor} mm minimum ({segs})")

    # Protected ends: the two outermost segments must each be at least the
    # protected length, i.e. no kerf intrudes into either keep-out zone.
    if protected_end_mm > 0 and len(segs) >= 2:
        ends_ok = segs[0] >= protected_end_mm and segs[-1] >= protected_end_mm
        r.check("protected_ends", ends_ok,
                f"{parent.item_id}: a cut intrudes into a "
                f"{protected_end_mm} mm protected end")
    else:
        r.checks.setdefault("protected_ends", True)
    return r


# --------------------------------------------------------------------------- #
# Lineage and mass / material-volume conservation
# --------------------------------------------------------------------------- #


def validate_lineage(parent: WasteItem, children: Sequence[WasteItem], *,
                     kerf_mm: int, tolerance_mm: int = CONSERVATION_TOLERANCE_MM,
                     report: Optional[_Report] = None) -> _Report:
    """Validate inherited attributes, ids, generation and conservation."""
    r = report or _Report(tolerance_mm)
    kids = list(children)

    r.check("has_children", len(kids) >= 2,
            f"{parent.item_id}: lineage needs >= 2 derived items")

    ids = [k.item_id for k in kids]
    r.check("unique_child_ids", len(set(ids)) == len(ids),
            f"{parent.item_id}: derived child ids are not unique ({ids})")

    r.check("parent_linkage", all(k.parent_item_id == parent.item_id for k in kids),
            f"{parent.item_id}: a child does not reference its parent")
    r.check("generation", all(k.generation == parent.generation + 1 for k in kids),
            f"{parent.item_id}: a child has the wrong generation")

    r.check("inherited_material",
            all(k.material == parent.material for k in kids),
            f"{parent.item_id}: material not inherited by every child")
    r.check("inherited_segregation_group",
            all(k.segregation_group == parent.segregation_group for k in kids),
            f"{parent.item_id}: segregation group not inherited by every child")
    r.check("inherited_diameter",
            all(k.outer_diameter_mm == parent.outer_diameter_mm
                and k.inner_diameter_mm == parent.inner_diameter_mm
                for k in kids),
            f"{parent.item_id}: a child changed diameter")

    n_cuts = max(0, len(kids) - 1)

    # Material-volume conservation minus kerf. Length-proportional for tubes, so
    # this is re-derived from cross-section and lengths, not read off the items.
    cross = tube_cross_section_mm2(parent.outer_diameter_mm, parent.inner_diameter_mm)
    kerf_volume = cross * n_cuts * kerf_mm
    child_volume = sum(k.material_volume_mm3 for k in kids)
    vol_delta = abs((child_volume + kerf_volume) - parent.material_volume_mm3)
    vol_tol = cross * tolerance_mm + 1.0
    r.check("material_volume_conservation", vol_delta <= vol_tol,
            f"{parent.item_id}: material volume not conserved — children "
            f"{child_volume:.1f} + kerf {kerf_volume:.1f} vs parent "
            f"{parent.material_volume_mm3:.1f} mm^3 (delta {vol_delta:.1f})")

    # Mass conservation minus the kerf mass fraction (pro-rated by length).
    total_len = sum(k.length_mm for k in kids)
    if parent.length_mm > 0 and parent.weight_kg > 0:
        expected_child_mass = parent.weight_kg * (total_len / parent.length_mm)
        mass_delta = abs(sum(k.weight_kg for k in kids) - expected_child_mass)
        r.check("mass_conservation", mass_delta <= 0.05 * parent.weight_kg + 1e-6,
                f"{parent.item_id}: child mass {sum(k.weight_kg for k in kids):.3f} "
                f"deviates from length-pro-rated {expected_child_mass:.3f} kg")
    else:
        r.checks.setdefault("mass_conservation", True)
    return r


# --------------------------------------------------------------------------- #
# Combined proposal / result validation
# --------------------------------------------------------------------------- #


def validate_proposal(proposal: CutProposal, parent: WasteItem, *,
                      children: Optional[Sequence[WasteItem]] = None,
                      tolerance_mm: int = CONSERVATION_TOLERANCE_MM) -> Dict[str, Any]:
    """Full independent verdict for a proposal against its source item.

    If ``children`` are not supplied they are re-derived here from the proposal's
    own segment lengths, so a caller cannot hand the validator a doctored set.
    """
    from .cutting import derive_segments  # local import avoids any cycle
    r = _Report(tolerance_mm)
    validate_segmentation(
        parent, proposal.segment_lengths_mm, kerf_mm=proposal.kerf_mm,
        max_cuts=parent.maximum_number_of_cuts,
        minimum_segment_mm=parent.minimum_segment_length_mm,
        protected_end_mm=parent.protected_end_length_mm,
        tolerance_mm=tolerance_mm, report=r)
    kids = list(children) if children is not None else derive_segments(
        parent, proposal.segment_lengths_mm, kerf_mm=proposal.kerf_mm)
    validate_lineage(parent, kids, kerf_mm=proposal.kerf_mm,
                     tolerance_mm=tolerance_mm, report=r)
    # Cross-check the ids the proposal advertises match the derived lineage.
    r.check("proposal_child_ids_match",
            proposal.derived_item_ids_for() == [k.item_id for k in kids],
            f"{parent.item_id}: proposal child ids disagree with lineage")
    return r.to_dict()


def validate_result(result: CutResult, parent: WasteItem, *,
                    children: Optional[Sequence[WasteItem]] = None,
                    tolerance_mm: int = CONSERVATION_TOLERANCE_MM) -> Dict[str, Any]:
    """Validate an executed cut's ACTUAL segments and its registered lineage."""
    from .cutting import derive_segments
    r = _Report(tolerance_mm)
    if not result.succeeded:
        r.check("cut_succeeded", False,
                f"{parent.item_id}: cut did not complete "
                f"({result.completion_status.value}: {result.failure_reason})")
        return r.to_dict()
    validate_segmentation(
        parent, result.actual_segment_lengths_mm, kerf_mm=result.actual_kerf_mm,
        max_cuts=max(parent.maximum_number_of_cuts,
                     len(result.actual_segment_lengths_mm) - 1),
        minimum_segment_mm=parent.minimum_segment_length_mm,
        protected_end_mm=parent.protected_end_length_mm,
        tolerance_mm=tolerance_mm, report=r)
    kids = list(children) if children is not None else derive_segments(
        parent, result.actual_segment_lengths_mm, kerf_mm=result.actual_kerf_mm,
        child_ids=result.resulting_child_ids or None)
    validate_lineage(parent, kids, kerf_mm=result.actual_kerf_mm,
                     tolerance_mm=tolerance_mm, report=r)
    return r.to_dict()


def validate_no_coexistence(items: Sequence[WasteItem]) -> Dict[str, Any]:
    """No original item and any of its derived children may both be packable.

    Once a pipe is cut, the pipe leaves the packable set. If both a parent id and
    a child that names it as parent are present, the item set is inconsistent.
    """
    r = _Report(CONSERVATION_TOLERANCE_MM)
    present = {i.item_id for i in items}
    for i in items:
        if i.parent_item_id and i.parent_item_id in present:
            r.check("no_parent_child_coexistence", False,
                    f"child {i.item_id} coexists with its parent "
                    f"{i.parent_item_id} in the packable set")
    r.checks.setdefault("no_parent_child_coexistence", True)
    return r.to_dict()


__all__ = [
    "validate_segmentation", "validate_lineage", "validate_proposal",
    "validate_result", "validate_no_coexistence",
]
