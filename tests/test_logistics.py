"""Simulated container logistics — deterministic tasks and robot motion.

The simulation is a pure function of tick count (brief §16), so these tests
assert exact, repeatable outcomes: a delivery drives the container to the cell, a
collection dispatches it, movement is identical across runs, and failures are
handled rather than swallowed.
"""

from __future__ import annotations

import pytest

from wisepack_core.generator import make_container
from wisepack_core.inventory import ContainerInventory, ContainerLifecycleState as S
from wisepack_core.logistics import (
    LogisticsSimulator, TransportTaskState, TransportTaskType as TT,
)


def _inv(n=2):
    inv = ContainerInventory(simulated=True)
    for i in range(n):
        cid = f"CNT-{i:02d}"
        inv.register(make_container("standard_box", cid))
        inv.mark_available(cid)
    return inv


def _reserved_ready(inv, cid):
    inv.reserve(cid, holder="plan-1", segregation_group="A")
    inv.request_delivery(cid)
    return inv


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #

def test_delivery_task_moves_container_to_cell():
    inv = _reserved_ready(_inv(1), "CNT-00")
    log = LogisticsSimulator(inventory=inv)
    task = log.request("CNT-00", TT.DELIVER_EMPTY_CONTAINER, plan="plan-1")
    assert task.status is TransportTaskState.QUEUED
    log.run_to_quiescence()
    assert log.tasks[task.task_id].status is TransportTaskState.COMPLETED
    assert inv.get("CNT-00").state is S.AT_PACKING_CELL
    assert inv.get("CNT-00").location == "packing_cell"


def test_collection_task_dispatches_full_container():
    inv = _reserved_ready(_inv(1), "CNT-00")
    log = LogisticsSimulator(inventory=inv)
    log.request("CNT-00", TT.DELIVER_EMPTY_CONTAINER)
    log.run_to_quiescence()
    for op in ("mark_filling", "mark_full", "mark_sealed", "request_collection",
               "mark_in_transit_from_cell"):
        getattr(inv, op)("CNT-00")
    log.request("CNT-00", TT.REMOVE_FULL_CONTAINER)
    log.run_to_quiescence()
    assert inv.get("CNT-00").state is S.DISPATCHED


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def test_movement_is_deterministic_across_runs():
    def run():
        inv = _reserved_ready(_inv(1), "CNT-00")
        log = LogisticsSimulator(inventory=inv)
        log.request("CNT-00", TT.DELIVER_EMPTY_CONTAINER)
        ticks = log.run_to_quiescence()
        return ticks, log.robot.x, log.robot.y
    assert run() == run()


def test_two_tasks_run_in_priority_order():
    inv = _inv(2)
    for c in ("CNT-00", "CNT-01"):
        _reserved_ready(inv, c)
    log = LogisticsSimulator(inventory=inv)
    low = log.request("CNT-00", TT.DELIVER_EMPTY_CONTAINER, priority=9)
    high = log.request("CNT-01", TT.DELIVER_EMPTY_CONTAINER, priority=1)
    # Step just enough to assign one task; the higher priority must go first.
    log.step(1)
    assert log.robot.current_task_id == high.task_id


# --------------------------------------------------------------------------- #
# Failure / cancel
# --------------------------------------------------------------------------- #

def test_failure_is_handled_and_counted():
    inv = _reserved_ready(_inv(1), "CNT-00")
    log = LogisticsSimulator(inventory=inv)
    t = log.request("CNT-00", TT.DELIVER_EMPTY_CONTAINER)
    log.fail(t.task_id, "aisle blocked (simulated)")
    assert log.tasks[t.task_id].status is TransportTaskState.FAILED
    assert log.tasks[t.task_id].failure_reason
    assert log.analytics()["failed_tasks"] == 1
    assert log.robot.current_task_id is None


def test_cancel_before_completion():
    inv = _reserved_ready(_inv(1), "CNT-00")
    log = LogisticsSimulator(inventory=inv)
    t = log.request("CNT-00", TT.DELIVER_EMPTY_CONTAINER)
    log.cancel(t.task_id)
    assert log.tasks[t.task_id].status is TransportTaskState.CANCELLED


def test_replace_unavailable_container_task_type():
    inv = _inv(1)
    log = LogisticsSimulator(inventory=inv)
    t = log.request("CNT-00", TT.REPLACE_UNAVAILABLE_CONTAINER)
    assert t.task_type is TT.REPLACE_UNAVAILABLE_CONTAINER
    assert t.source_location == "container_storage"
    assert t.destination_location == "packing_cell"


# --------------------------------------------------------------------------- #
# Analytics + facility map
# --------------------------------------------------------------------------- #

def test_analytics_report_deliveries_collections_and_utilization():
    inv = _reserved_ready(_inv(1), "CNT-00")
    log = LogisticsSimulator(inventory=inv)
    log.request("CNT-00", TT.DELIVER_EMPTY_CONTAINER)
    log.run_to_quiescence()
    a = log.analytics()
    assert a["delivery_requests"] == 1
    assert a["tasks_completed"] == 1
    assert 0 < a["robot_utilization_pct"] <= 100
    assert a["avg_task_duration_ticks"] >= 0


def test_facility_map_labels_simulation_and_lists_locations():
    inv = _inv(1)
    log = LogisticsSimulator(inventory=inv)
    fm = log.facility_map()
    assert "no physical mobile robot" in fm["label"]
    for loc in ("container_storage", "packing_cell", "inspection_station",
                "dispatch_area"):
        assert loc in fm["layout"]
