"""FIWARE-backed container inventory — lifecycle, reservations, selection.

Enforces the transition table (brief §10) against the exact valid/invalid
examples the brief lists, checks that illegal moves are rejected AND logged, and
that inventory-aware selection excludes the states §14 forbids.
"""

from __future__ import annotations

import pytest

from wisepack_core.generator import make_container
from wisepack_core.inventory import (
    ALLOWED_TRANSITIONS, ContainerInventory, ContainerLifecycleState as S,
    InvalidTransition, is_valid_transition,
)


def _inv(n=3, spec="standard_box"):
    inv = ContainerInventory(simulated=True)
    for i in range(n):
        cid = f"CNT-{i:02d}"
        inv.register(make_container(spec, cid))
        inv.mark_available(cid)
    return inv


# --------------------------------------------------------------------------- #
# Transition table
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("src,dst,ok", [
    (S.AVAILABLE, S.RESERVED, True),
    (S.RESERVED, S.REQUESTED_FOR_DELIVERY, True),
    (S.SEALED, S.AVAILABLE, False),
    (S.DISPATCHED, S.FILLING, False),
    (S.RETIRED, S.RESERVED, False),
])
def test_transition_table_matches_brief(src, dst, ok):
    assert is_valid_transition(src, dst) is ok


def test_terminal_states_have_no_exits():
    assert ALLOWED_TRANSITIONS[S.DISPATCHED] == frozenset()
    assert ALLOWED_TRANSITIONS[S.RETIRED] == frozenset()


def test_invalid_transition_is_rejected_and_logged():
    inv = _inv(1)
    with pytest.raises(InvalidTransition):
        inv.mark_dispatched("CNT-00")           # AVAILABLE -> DISPATCHED
    last = inv.get("CNT-00").history[-1]
    assert last["rejected"] and not last["applied"]
    # State is unchanged after a rejected transition.
    assert inv.get("CNT-00").state is S.AVAILABLE


def test_valid_full_lifecycle_walk():
    inv = _inv(1)
    cid = "CNT-00"
    inv.reserve(cid, holder="plan-1", segregation_group="A")
    inv.request_delivery(cid)
    inv.mark_in_transit_to_cell(cid)
    inv.mark_at_cell(cid)
    inv.mark_filling(cid)
    inv.mark_full(cid)
    inv.request_quality_check(cid)
    inv.mark_sealed(cid)
    inv.request_collection(cid)
    inv.mark_in_transit_from_cell(cid)
    inv.mark_dispatched(cid)
    assert inv.get(cid).state is S.DISPATCHED
    # Revision advanced monotonically across the walk.
    revs = [h["revision"] for h in inv.get(cid).history if h["applied"]]
    assert revs == sorted(revs)


# --------------------------------------------------------------------------- #
# Reservations
# --------------------------------------------------------------------------- #

def test_reserve_and_release():
    inv = _inv(1)
    inv.reserve("CNT-00", holder="plan-1")
    assert inv.get("CNT-00").state is S.RESERVED
    assert inv.get("CNT-00").reservation is not None
    inv.release_reservation("CNT-00")
    assert inv.get("CNT-00").state is S.AVAILABLE
    assert inv.get("CNT-00").reservation is None


def test_reservation_conflict_excludes_from_other_group_selection():
    inv = _inv(2, spec="standard_box")
    inv.reserve("CNT-00", holder="plan-A", segregation_group="A")
    # A plan for group B cannot use the container reserved for A.
    sel_b = {ic.container_id for ic in inv.selectable_for("B")}
    assert "CNT-00" not in sel_b
    # But the same group A can still use its own reservation.
    sel_a = {ic.container_id for ic in inv.selectable_for("A")}
    assert "CNT-00" in sel_a


# --------------------------------------------------------------------------- #
# Inventory-aware selection (brief §14)
# --------------------------------------------------------------------------- #

def test_selection_excludes_unavailable_full_sealed_dispatched_retired():
    inv = _inv(6)
    inv.mark_unavailable("CNT-00")
    # Drive CNT-01 to FULL and CNT-02 to SEALED via the valid path.
    for cid in ("CNT-01", "CNT-02"):
        inv.reserve(cid, holder="p"); inv.request_delivery(cid)
        inv.mark_in_transit_to_cell(cid); inv.mark_at_cell(cid)
        inv.mark_filling(cid); inv.mark_full(cid)
    inv.mark_sealed("CNT-02")
    inv.retire("CNT-03")
    sel = {ic.container_id for ic in inv.selectable_for("A")}
    for excluded in ("CNT-00", "CNT-01", "CNT-02", "CNT-03"):
        assert excluded not in sel
    assert {"CNT-04", "CNT-05"} <= sel


def test_segregation_incompatible_container_is_not_selectable():
    inv = ContainerInventory(simulated=True)
    inv.register(make_container("standard_box", "ONLY-B", allowed_groups=("B",)))
    inv.mark_available("ONLY-B")
    assert inv.selectable_for("A") == []
    assert len(inv.selectable_for("B")) == 1


def test_compatible_capacity_and_shortage():
    inv = _inv(2)
    cap = inv.compatible_capacity_mm3("A")
    assert cap == sum(ic.remaining_capacity_mm3 for ic in inv.all())
    ev = inv.record_shortage("A", needed_mm3=cap + 1)
    assert inv.summary()["forecast_shortage"] is True
    assert ev["needed_mm3"] == cap + 1


# --------------------------------------------------------------------------- #
# FIWARE projection (brief §11)
# --------------------------------------------------------------------------- #

def test_semantic_state_is_compact_and_has_no_placement_geometry():
    inv = _inv(1)
    st = inv.get("CNT-00").semantic_state()
    assert st["entity_id"] == "urn:ngsi-ld:WISEPACKContainer:CNT-00"
    assert st["entity_type"] == "WISEPACKContainer"
    # Required semantic fields present...
    for f in ("lifecycle_state", "utilization_pct", "remaining_capacity_mm3",
              "availability", "location", "revision", "source"):
        assert f in st
    # ...and NO placement geometry leaks into FIWARE.
    assert "placements" not in st and "position" not in st


def test_contents_update_bumps_revision_and_utilization():
    inv = _inv(1)
    before = inv.get("CNT-00").revision
    cap = inv.get("CNT-00").capacity_mm3
    inv.get("CNT-00").apply_contents(occupied_volume_mm3=cap // 2, payload_kg=100,
                                     item_count=4, plan_id="plan-1")
    ic = inv.get("CNT-00")
    assert ic.revision > before
    assert 49 < ic.utilization_pct < 51 and ic.item_count == 4


def test_summary_counts_states():
    inv = _inv(3)
    inv.reserve("CNT-00", holder="p")
    c = inv.summary()
    assert c["total"] == 3 and c["available"] == 2 and c["reserved"] == 1
