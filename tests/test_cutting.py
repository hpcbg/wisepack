"""Cut-aware domain + independent validator.

These tests pin the conservation and lineage guarantees (brief §3) and the honesty
labels. They deliberately exercise the validator against HAND-BROKEN inputs — a
wrong material, a stolen millimetre, a duplicate id — because a validator that
only ever sees good data proves nothing.
"""

from __future__ import annotations

import pytest

from wisepack_core.cutting import (
    CONSERVATION_TOLERANCE_MM, CutConfig, CutProposal, CutResult, CutState,
    derive_segments, positions_from_segment_lengths, segment_lengths_from_positions,
    tube_cross_section_mm2,
)
from wisepack_core import cut_validator as cv
from wisepack_core.domain import DomainError, GeometryType, Source, WasteItem


def tube(item_id="pipe-1", length_mm=3000, od=200, idia=180, **kw):
    kw.setdefault("weight_kg", 30.0)
    return WasteItem(item_id, length_mm=length_mm, outer_diameter_mm=od,
                     inner_diameter_mm=idia, cut_allowed=kw.pop("cut_allowed", True),
                     maximum_number_of_cuts=kw.pop("maximum_number_of_cuts", 2),
                     minimum_segment_length_mm=kw.pop("minimum_segment_length_mm", 400),
                     protected_end_length_mm=kw.pop("protected_end_length_mm", 50), **kw)


# --------------------------------------------------------------------------- #
# Domain metadata + backward compatibility
# --------------------------------------------------------------------------- #

def test_cut_metadata_round_trips_and_defaults_off():
    t = tube()
    back = WasteItem.from_dict(t.to_dict())
    assert back.cut_allowed and back.maximum_number_of_cuts == 2
    assert back.minimum_segment_length_mm == 400
    # A legacy dict without any cut fields keeps the old behaviour exactly.
    legacy = WasteItem.from_dict({"item_id": "p", "length_mm": 1000,
                                  "outer_diameter_mm": 100})
    assert legacy.cut_allowed is False and legacy.generation == 0
    assert not legacy.is_cuttable


def test_only_tubes_are_cuttable():
    with pytest.raises(DomainError):
        WasteItem("s", length_mm=1000, outer_diameter_mm=100,
                  geometry_type=GeometryType.FLAT_SHEET, cut_allowed=True)


def test_effective_minimum_falls_back_to_diameter():
    t = WasteItem("p", length_mm=1000, outer_diameter_mm=120, cut_allowed=True,
                  maximum_number_of_cuts=1)
    assert t.effective_minimum_segment_mm == 120


# --------------------------------------------------------------------------- #
# Kerf / segment arithmetic
# --------------------------------------------------------------------------- #

def test_segment_positions_round_trip():
    segs = [1400, 800, 794]      # 3000 with 2 x 3 mm kerf
    pos = positions_from_segment_lengths(segs, 3)
    assert segment_lengths_from_positions(3000, pos, 3) == segs


def test_proposal_conserves_length_by_construction():
    cfg = CutConfig(kerf_mm=3)
    t = tube()
    segs = [1400, 800, 3000 - 1400 - 800 - 2 * 3]
    prop = CutProposal.for_segments("cmp-1", t, segs, config=cfg)
    assert sum(prop.segment_lengths_mm) + prop.total_kerf_mm == t.length_mm
    assert prop.n_cuts == 2 and prop.n_segments == 3


def test_kerf_material_loss_is_cross_section_times_total_kerf():
    cfg = CutConfig(kerf_mm=4)
    t = tube()
    prop = CutProposal.for_segments("cmp", t, [1500, 3000 - 1500 - 4], config=cfg)
    cross = tube_cross_section_mm2(t.outer_diameter_mm, t.inner_diameter_mm)
    assert prop.kerf_material_loss_mm3 == pytest.approx(cross * 4)


# --------------------------------------------------------------------------- #
# Independent conservation + lineage validation
# --------------------------------------------------------------------------- #

def test_valid_proposal_passes_all_checks():
    cfg = CutConfig(kerf_mm=3)
    t = tube()
    segs = [1400, 800, 3000 - 1400 - 800 - 6]
    prop = CutProposal.for_segments("cmp", t, segs, config=cfg)
    v = cv.validate_proposal(prop, t)
    assert v["valid"], v["violations"]
    assert all(v["checks"].values())


def test_length_theft_is_caught():
    """A proposal whose segments do not add up (plus kerf) must be rejected."""
    cfg = CutConfig(kerf_mm=3)
    t = tube()
    prop = CutProposal.for_segments("cmp", t, [1400, 800, 794], config=cfg)
    prop.segment_lengths_mm = [1400, 800, 700]      # 94 mm vanish
    v = cv.validate_proposal(prop, t)
    assert not v["valid"]
    assert any("conserv" in x for x in v["violations"])


def test_minimum_segment_enforced():
    cfg = CutConfig(kerf_mm=3)
    t = tube(minimum_segment_length_mm=400)
    prop = CutProposal.for_segments("cmp", t, [100, 3000 - 100 - 3], config=cfg)
    v = cv.validate_proposal(prop, t)
    assert not v["valid"]
    assert not v["checks"]["minimum_segment_length"]


def test_maximum_cuts_enforced():
    cfg = CutConfig(kerf_mm=3)
    t = tube(maximum_number_of_cuts=2)
    segs = [700, 700, 700, 3000 - 2100 - 9]
    prop = CutProposal.for_segments("cmp", t, segs, config=cfg)
    v = cv.validate_proposal(prop, t)
    assert not v["valid"] and not v["checks"]["max_cuts"]


def test_protected_ends_enforced():
    cfg = CutConfig(kerf_mm=3)
    t = tube(protected_end_length_mm=500, minimum_segment_length_mm=100)
    # first segment 100 < 500 protected end
    prop = CutProposal.for_segments("cmp", t, [100, 3000 - 100 - 3], config=cfg)
    v = cv.validate_proposal(prop, t)
    assert not v["checks"]["protected_ends"]


def test_inherited_material_and_segregation_are_required():
    t = tube(material="stainless", segregation_group="B")
    kids = derive_segments(t, [1500, 3000 - 1500 - 3], kerf_mm=3)
    assert all(k.material == "stainless" and k.segregation_group == "B"
               for k in kids)
    kids[0].material = "carbon_steel"                 # break inheritance
    v = cv.validate_lineage(t, kids, kerf_mm=3).to_dict()
    assert not v["valid"] and not v["checks"]["inherited_material"]


def test_mass_and_material_volume_conserved_minus_kerf():
    t = tube()
    segs = [1400, 800, 3000 - 1400 - 800 - 6]
    kids = derive_segments(t, segs, kerf_mm=3)
    v = cv.validate_lineage(t, kids, kerf_mm=3).to_dict()
    assert v["checks"]["mass_conservation"]
    assert v["checks"]["material_volume_conservation"]
    # Children carry strictly less mass than the parent (kerf becomes swarf).
    assert sum(k.weight_kg for k in kids) < t.weight_kg


def test_duplicate_child_ids_rejected():
    t = tube()
    with pytest.raises(DomainError):
        derive_segments(t, [1500, 1497], kerf_mm=3, child_ids=["x", "x"])


def test_no_parent_child_coexistence():
    t = tube()
    kids = derive_segments(t, [1500, 1497], kerf_mm=3)
    assert not cv.validate_no_coexistence([t] + kids)["valid"]
    assert cv.validate_no_coexistence(kids)["valid"]


def test_lineage_is_complete_and_one_generation_deeper():
    t = tube()
    kids = derive_segments(t, [1500, 1497], kerf_mm=3)
    for k in kids:
        assert k.parent_item_id == t.item_id
        assert k.generation == t.generation + 1
        assert k.is_derived and not k.cut_allowed        # bounded recursion
        assert k.cut_history and k.cut_history[-1]["parent_item_id"] == t.item_id


# --------------------------------------------------------------------------- #
# Cut result / deviation
# --------------------------------------------------------------------------- #

def test_deviated_result_still_validates_when_conserving():
    """Actual segments differ from the proposal but still conserve length."""
    t = tube()
    result = CutResult(
        proposal_id="cmp", source_item_id=t.item_id,
        actual_segment_lengths_mm=[1450, 800, 3000 - 1450 - 800 - 6],
        resulting_child_ids=["pipe-1-s1", "pipe-1-s2", "pipe-1-s3"],
        actual_kerf_mm=3, completion_status=CutState.COMPLETED)
    v = cv.validate_result(result, t)
    assert v["valid"], v["violations"]


def test_failed_cut_is_flagged_not_silently_accepted():
    t = tube()
    result = CutResult(proposal_id="cmp", source_item_id=t.item_id,
                       actual_segment_lengths_mm=[], resulting_child_ids=[],
                       actual_kerf_mm=3, completion_status=CutState.FAILED,
                       failure_reason="blade jam (simulated)")
    v = cv.validate_result(result, t)
    assert not v["valid"] and not v["checks"]["cut_succeeded"]


# --------------------------------------------------------------------------- #
# Honesty
# --------------------------------------------------------------------------- #

def test_all_cut_objects_are_labelled_simulated():
    t = tube()
    prop = CutProposal.for_segments("cmp", t, [1500, 1497], config=CutConfig())
    assert prop.to_dict()["source"] == Source.SIMULATED.value
    assert "SIMULATED" in prop.to_dict()["label"]
    result = CutResult(proposal_id="cmp", source_item_id=t.item_id,
                       actual_segment_lengths_mm=[1500, 1497],
                       resulting_child_ids=["pipe-1-s1", "pipe-1-s2"],
                       actual_kerf_mm=3)
    assert result.to_dict()["source"] == Source.SIMULATED.value
    assert "SIMULATED" in result.to_dict()["label"]
