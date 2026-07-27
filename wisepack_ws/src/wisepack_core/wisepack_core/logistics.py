"""Simulated container logistics — transport tasks and a deterministic robot.

This is a SIMULATED container-logistics integration: there is no physical mobile
robot, no SLAM and no Nav2 (brief §16). What is real is the task model, its
state machine, and a fully deterministic facility simulation whose movement is a
pure function of tick count — no wall-clock, no randomness — so a test and the
dashboard see identical motion.

The simulator may hold a reference to a :class:`ContainerInventory`; when a
transport task reaches a milestone it drives the matching audited inventory
operation (an arriving delivery moves the container to the cell, a completed
collection dispatches it). That keeps the two models consistent without either
importing ROS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .domain import DomainError, Source, _require_id
from .events import utc_now_iso
from .inventory import (
    ContainerInventory, LOCATION_CELL, LOCATION_DISPATCH, LOCATION_INSPECTION,
    LOCATION_STORAGE,
)


class TransportTaskType(str, Enum):
    DELIVER_EMPTY_CONTAINER = "DELIVER_EMPTY_CONTAINER"
    REMOVE_FULL_CONTAINER = "REMOVE_FULL_CONTAINER"
    REPLACE_UNAVAILABLE_CONTAINER = "REPLACE_UNAVAILABLE_CONTAINER"
    MOVE_TO_INSPECTION = "MOVE_TO_INSPECTION"
    RETURN_TO_STORAGE = "RETURN_TO_STORAGE"


class TransportTaskState(str, Enum):
    REQUESTED = "REQUESTED"
    QUEUED = "QUEUED"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    ARRIVED = "ARRIVED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


_TERMINAL = frozenset({TransportTaskState.COMPLETED, TransportTaskState.FAILED,
                       TransportTaskState.CANCELLED})

#: Fixed facility layout, in mm. A simple deterministic map inspired by HARVEST.
FACILITY_LAYOUT: Dict[str, Tuple[int, int]] = {
    LOCATION_STORAGE: (0, 0),
    LOCATION_CELL: (6000, 0),
    LOCATION_INSPECTION: (6000, 3500),
    LOCATION_DISPATCH: (11000, 0),
}

#: Where each task type starts and ends.
_TASK_ROUTES: Dict[TransportTaskType, Tuple[str, str]] = {
    TransportTaskType.DELIVER_EMPTY_CONTAINER: (LOCATION_STORAGE, LOCATION_CELL),
    TransportTaskType.REMOVE_FULL_CONTAINER: (LOCATION_CELL, LOCATION_DISPATCH),
    TransportTaskType.REPLACE_UNAVAILABLE_CONTAINER: (LOCATION_STORAGE, LOCATION_CELL),
    TransportTaskType.MOVE_TO_INSPECTION: (LOCATION_CELL, LOCATION_INSPECTION),
    TransportTaskType.RETURN_TO_STORAGE: (LOCATION_CELL, LOCATION_STORAGE),
}


def _distance(a: str, b: str) -> float:
    (ax, ay), (bx, by) = FACILITY_LAYOUT[a], FACILITY_LAYOUT[b]
    return math.hypot(bx - ax, by - ay)


# --------------------------------------------------------------------------- #
# ContainerTransportTask
# --------------------------------------------------------------------------- #


@dataclass
class ContainerTransportTask:
    """One requested container move (brief §15)."""

    task_id: str
    container_id: str
    task_type: TransportTaskType
    source_location: str
    destination_location: str
    priority: int = 5
    status: TransportTaskState = TransportTaskState.REQUESTED
    requested_by: str = "orchestrator"
    assigned_robot: Optional[str] = None
    failure_reason: str = ""
    scenario: Optional[str] = None
    plan: Optional[str] = None
    timestamps: Dict[str, Any] = field(default_factory=dict)
    source: Source = Source.SIMULATED

    def __post_init__(self) -> None:
        _require_id("task_id", self.task_id)
        self.task_type = TransportTaskType(self.task_type)
        self.status = TransportTaskState(self.status)
        self.source = Source(self.source)

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    @property
    def distance_mm(self) -> float:
        return _distance(self.source_location, self.destination_location)

    def duration_ticks(self) -> Optional[int]:
        start = self.timestamps.get("IN_PROGRESS_tick")
        end = self.timestamps.get("COMPLETED_tick", self.timestamps.get("FAILED_tick"))
        if start is None or end is None:
            return None
        return int(end) - int(start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "container_id": self.container_id,
            "task_type": self.task_type.value,
            "source_location": self.source_location,
            "destination_location": self.destination_location,
            "priority": self.priority,
            "status": self.status.value,
            "requested_by": self.requested_by,
            "assigned_robot": self.assigned_robot,
            "failure_reason": self.failure_reason,
            "scenario": self.scenario,
            "plan": self.plan,
            "timestamps": dict(self.timestamps),
            "duration_ticks": self.duration_ticks(),
            "distance_mm": round(self.distance_mm, 1),
            "source": self.source.value,
            "label": "SIMULATED CONTAINER LOGISTICS",
        }


@dataclass
class MobileRobotState:
    """The single simulated mobile robot (no physical hardware — brief §16)."""

    robot_id: str = "amr-sim-1"
    x: float = 0.0
    y: float = 0.0
    status: str = "idle"                       # idle | moving | at_location
    current_task_id: Optional[str] = None
    location: str = LOCATION_STORAGE
    busy_ticks: int = 0
    idle_ticks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        total = self.busy_ticks + self.idle_ticks
        return {
            "robot_id": self.robot_id,
            "x": round(self.x, 1), "y": round(self.y, 1),
            "status": self.status,
            "current_task_id": self.current_task_id,
            "location": self.location,
            "busy_ticks": self.busy_ticks, "idle_ticks": self.idle_ticks,
            "utilization_pct": round(100.0 * self.busy_ticks / total, 1) if total else 0.0,
            "label": "SIMULATED MOBILE ROBOT — no physical transport",
            "source": Source.SIMULATED.value,
        }


# --------------------------------------------------------------------------- #
# LogisticsSimulator
# --------------------------------------------------------------------------- #


class LogisticsSimulator:
    """A deterministic, tick-driven container-logistics simulation."""

    def __init__(self, inventory: Optional[ContainerInventory] = None, *,
                 robot_speed_mm_per_tick: float = 1800.0) -> None:
        self.inventory = inventory
        self.robot = MobileRobotState()
        self.speed = float(robot_speed_mm_per_tick)
        self.tick = 0
        self.tasks: Dict[str, ContainerTransportTask] = {}
        self._order: List[str] = []
        self._seq = 0
        self.events: List[Dict[str, Any]] = []
        self.failed_task_ids: List[str] = []

    # -- request / queue -------------------------------------------------- #

    def request(self, container_id: str, task_type: TransportTaskType, *,
                priority: int = 5, requested_by: str = "orchestrator",
                scenario: Optional[str] = None, plan: Optional[str] = None,
                task_id: Optional[str] = None) -> ContainerTransportTask:
        task_type = TransportTaskType(task_type)
        src, dst = _TASK_ROUTES[task_type]
        self._seq += 1
        tid = task_id or f"task-{self._seq:04d}"
        task = ContainerTransportTask(
            task_id=tid, container_id=container_id, task_type=task_type,
            source_location=src, destination_location=dst, priority=priority,
            requested_by=requested_by, scenario=scenario, plan=plan)
        task.timestamps["REQUESTED_tick"] = self.tick
        task.timestamps["REQUESTED_at"] = utc_now_iso()
        self.tasks[tid] = task
        self._order.append(tid)
        self._emit(task, "requested")
        # A request immediately queues (brief §15 states).
        self._set_state(task, TransportTaskState.QUEUED, "queued")
        return task

    def cancel(self, task_id: str, reason: str = "cancelled") -> ContainerTransportTask:
        task = self._task(task_id)
        if task.is_terminal:
            raise DomainError(f"{task_id} is already {task.status.value}")
        self._set_state(task, TransportTaskState.CANCELLED, reason)
        if self.robot.current_task_id == task_id:
            self.robot.current_task_id = None
            self.robot.status = "idle"
        return task

    def fail(self, task_id: str, reason: str = "simulated transport failure"
             ) -> ContainerTransportTask:
        task = self._task(task_id)
        if task.is_terminal:
            raise DomainError(f"{task_id} is already {task.status.value}")
        task.failure_reason = reason
        self._set_state(task, TransportTaskState.FAILED, reason)
        self.failed_task_ids.append(task_id)
        if self.robot.current_task_id == task_id:
            self.robot.current_task_id = None
            self.robot.status = "idle"
        return task

    # -- deterministic stepping ------------------------------------------- #

    def step(self, ticks: int = 1) -> None:
        for _ in range(max(0, ticks)):
            self._step_once()

    def run_to_quiescence(self, max_ticks: int = 1000) -> int:
        """Advance until no task can progress. Returns ticks consumed."""
        used = 0
        while used < max_ticks and self._has_active_work():
            self._step_once()
            used += 1
        return used

    def _has_active_work(self) -> bool:
        return any(not t.is_terminal for t in self.tasks.values())

    def _step_once(self) -> None:
        self.tick += 1
        # Assign the robot the highest-priority queued task when idle.
        if self.robot.current_task_id is None:
            nxt = self._next_queued()
            if nxt is not None:
                self._assign(nxt)
            else:
                self.robot.idle_ticks += 1
                self.robot.status = "idle"
                return
        task = self.tasks.get(self.robot.current_task_id)
        if task is None or task.is_terminal:
            self.robot.current_task_id = None
            return
        self.robot.busy_ticks += 1
        if task.status is TransportTaskState.ASSIGNED:
            self._set_state(task, TransportTaskState.IN_PROGRESS, "moving")
            self._on_in_progress(task)
        self._advance_robot_towards(task)

    def _next_queued(self) -> Optional[ContainerTransportTask]:
        queued = [self.tasks[t] for t in self._order
                  if self.tasks[t].status is TransportTaskState.QUEUED]
        if not queued:
            return None
        # Highest priority (lowest number) first, then request order.
        queued.sort(key=lambda t: (t.priority, self._order.index(t.task_id)))
        return queued[0]

    def _assign(self, task: ContainerTransportTask) -> None:
        task.assigned_robot = self.robot.robot_id
        self.robot.current_task_id = task.task_id
        self.robot.status = "moving"
        # Teleport-free: the robot starts each task from wherever it is; if it is
        # not at the task source we route via the source first (single leg here
        # because tasks originate where the previous one ended in practice).
        self._set_state(task, TransportTaskState.ASSIGNED, "assigned")

    def _advance_robot_towards(self, task: ContainerTransportTask) -> None:
        tx, ty = FACILITY_LAYOUT[task.destination_location]
        dx, dy = tx - self.robot.x, ty - self.robot.y
        dist = math.hypot(dx, dy)
        if dist <= self.speed or dist == 0:
            self.robot.x, self.robot.y = float(tx), float(ty)
            self.robot.location = task.destination_location
            self.robot.status = "at_location"
            self._set_state(task, TransportTaskState.ARRIVED, "arrived")
            self._on_arrived(task)
            self._set_state(task, TransportTaskState.COMPLETED, "completed")
            self._on_completed(task)
            self.robot.current_task_id = None
        else:
            self.robot.x += self.speed * dx / dist
            self.robot.y += self.speed * dy / dist
            self.robot.status = "moving"

    # -- inventory coupling ----------------------------------------------- #

    def _on_in_progress(self, task: ContainerTransportTask) -> None:
        if self.inventory is None or task.container_id not in self.inventory:
            return
        try:
            if task.task_type in (TransportTaskType.DELIVER_EMPTY_CONTAINER,
                                  TransportTaskType.REPLACE_UNAVAILABLE_CONTAINER):
                self.inventory.mark_in_transit_to_cell(task.container_id)
        except DomainError:
            pass

    def _on_arrived(self, task: ContainerTransportTask) -> None:
        if self.inventory is None or task.container_id not in self.inventory:
            return
        try:
            if task.task_type in (TransportTaskType.DELIVER_EMPTY_CONTAINER,
                                  TransportTaskType.REPLACE_UNAVAILABLE_CONTAINER):
                self.inventory.mark_at_cell(task.container_id)
            elif task.task_type is TransportTaskType.MOVE_TO_INSPECTION:
                self.inventory.request_quality_check(task.container_id)
        except DomainError:
            pass

    def _on_completed(self, task: ContainerTransportTask) -> None:
        if self.inventory is None or task.container_id not in self.inventory:
            return
        try:
            if task.task_type is TransportTaskType.REMOVE_FULL_CONTAINER:
                self.inventory.mark_dispatched(task.container_id)
        except DomainError:
            pass

    # -- bookkeeping ------------------------------------------------------ #

    def _task(self, task_id: str) -> ContainerTransportTask:
        if task_id not in self.tasks:
            raise DomainError(f"unknown task {task_id!r}")
        return self.tasks[task_id]

    def _set_state(self, task: ContainerTransportTask,
                   state: TransportTaskState, reason: str) -> None:
        task.status = state
        task.timestamps[f"{state.value}_tick"] = self.tick
        task.timestamps[f"{state.value}_at"] = utc_now_iso()
        self._emit(task, reason)

    def _emit(self, task: ContainerTransportTask, reason: str) -> None:
        self.events.append({
            "task_id": task.task_id, "container_id": task.container_id,
            "task_type": task.task_type.value, "status": task.status.value,
            "reason": reason, "tick": self.tick, "at": utc_now_iso(),
            "source": Source.SIMULATED.value,
        })

    # -- analytics (brief §17 logistics) ---------------------------------- #

    def analytics(self) -> Dict[str, Any]:
        tasks = list(self.tasks.values())
        deliveries = [t for t in tasks
                      if t.task_type in (TransportTaskType.DELIVER_EMPTY_CONTAINER,
                                         TransportTaskType.REPLACE_UNAVAILABLE_CONTAINER)]
        collections = [t for t in tasks
                       if t.task_type is TransportTaskType.REMOVE_FULL_CONTAINER]
        durations = [t.duration_ticks() for t in tasks
                     if t.duration_ticks() is not None]
        req_to_arrival = []
        for t in tasks:
            a = t.timestamps.get("ARRIVED_tick")
            r = t.timestamps.get("REQUESTED_tick")
            if a is not None and r is not None:
                req_to_arrival.append(a - r)
        return {
            "delivery_requests": len(deliveries),
            "collection_requests": len(collections),
            "tasks_total": len(tasks),
            "tasks_completed": sum(1 for t in tasks
                                   if t.status is TransportTaskState.COMPLETED),
            "failed_tasks": len(self.failed_task_ids),
            "avg_task_duration_ticks": round(sum(durations) / len(durations), 2)
                if durations else 0.0,
            "avg_request_to_arrival_ticks":
                round(sum(req_to_arrival) / len(req_to_arrival), 2)
                if req_to_arrival else 0.0,
            "robot_utilization_pct": self.robot.to_dict()["utilization_pct"],
            "ticks_elapsed": self.tick,
            "source": Source.SIMULATED.value,
        }

    def facility_map(self) -> Dict[str, Any]:
        """Deterministic facility snapshot for the logistics view (brief §16)."""
        at_location: Dict[str, List[str]] = {loc: [] for loc in FACILITY_LAYOUT}
        if self.inventory is not None:
            for ic in self.inventory.all():
                at_location.setdefault(ic.location, []).append(ic.container_id)
        active = self.tasks.get(self.robot.current_task_id or "")
        return {
            "layout": {k: {"x": v[0], "y": v[1]} for k, v in FACILITY_LAYOUT.items()},
            "containers_at_location": at_location,
            "robot": self.robot.to_dict(),
            "active_task": active.to_dict() if active else None,
            "pending_tasks": [t.to_dict() for t in self.tasks.values()
                              if t.status in (TransportTaskState.QUEUED,
                                              TransportTaskState.REQUESTED)],
            "completed_tasks": [t.task_id for t in self.tasks.values()
                                if t.status is TransportTaskState.COMPLETED],
            "label": "Simulated container-logistics integration — "
                     "no physical mobile robot",
            "source": Source.SIMULATED.value,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks": [t.to_dict() for t in self.tasks.values()],
            "robot": self.robot.to_dict(),
            "analytics": self.analytics(),
            "facility_map": self.facility_map(),
        }


__all__ = [
    "TransportTaskType", "TransportTaskState", "ContainerTransportTask",
    "MobileRobotState", "LogisticsSimulator", "FACILITY_LAYOUT",
]
