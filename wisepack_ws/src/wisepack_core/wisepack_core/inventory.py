"""FIWARE-backed operational container inventory — ROS-free core.

This is a real, validated operational model, not a static demo table (brief §9).
Like domain.py it imports no ROS and no FastAPI: the same inventory object backs
sim mode, the ROS nodes and the FIWARE read-back, so the numbers cannot diverge
between them.

Two rules make the lifecycle trustworthy:

  * every state change goes through an explicit transition table
    (:data:`ALLOWED_TRANSITIONS`); an illegal move (``SEALED -> AVAILABLE``,
    ``DISPATCHED -> FILLING``, ``RETIRED -> RESERVED``) raises
    :class:`InvalidTransition` and is recorded, never silently applied;
  * there is no public field setter for state — you call an audited operation,
    which validates the transition, bumps the container's revision, appends a
    history record with actor/reason/timestamp, and returns an event descriptor
    for the integration layer to publish to ROS + FIWARE.

The FIWARE projection (:meth:`InventoryContainer.semantic_state`) carries only
compact semantic state (brief §11). Full placement geometry stays on ROS, in the
Digital Twin and in the immutable result artefacts — never inside FIWARE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .domain import Container, DomainError, Source, _require_id
from .events import utc_now_iso


class ContainerLifecycleState(str, Enum):
    """The 16 validated lifecycle states of brief §10."""

    REGISTERED = "REGISTERED"
    AVAILABLE = "AVAILABLE"
    RESERVED = "RESERVED"
    REQUESTED_FOR_DELIVERY = "REQUESTED_FOR_DELIVERY"
    IN_TRANSIT_TO_CELL = "IN_TRANSIT_TO_CELL"
    AT_PACKING_CELL = "AT_PACKING_CELL"
    FILLING = "FILLING"
    TEMPORARILY_UNAVAILABLE = "TEMPORARILY_UNAVAILABLE"
    FULL = "FULL"
    QUALITY_CHECK = "QUALITY_CHECK"
    SEALED = "SEALED"
    READY_FOR_COLLECTION = "READY_FOR_COLLECTION"
    IN_TRANSIT_FROM_CELL = "IN_TRANSIT_FROM_CELL"
    DISPATCHED = "DISPATCHED"
    MAINTENANCE = "MAINTENANCE"
    RETIRED = "RETIRED"


_S = ContainerLifecycleState

#: The explicit transition table. Anything not listed here is invalid.
ALLOWED_TRANSITIONS: Dict[ContainerLifecycleState, frozenset] = {
    _S.REGISTERED: frozenset({_S.AVAILABLE, _S.MAINTENANCE, _S.RETIRED}),
    _S.AVAILABLE: frozenset({_S.RESERVED, _S.MAINTENANCE, _S.RETIRED,
                             _S.TEMPORARILY_UNAVAILABLE}),
    _S.RESERVED: frozenset({_S.AVAILABLE, _S.REQUESTED_FOR_DELIVERY}),
    _S.REQUESTED_FOR_DELIVERY: frozenset({_S.IN_TRANSIT_TO_CELL, _S.RESERVED}),
    _S.IN_TRANSIT_TO_CELL: frozenset({_S.AT_PACKING_CELL,
                                      _S.TEMPORARILY_UNAVAILABLE}),
    _S.AT_PACKING_CELL: frozenset({_S.FILLING, _S.TEMPORARILY_UNAVAILABLE}),
    _S.FILLING: frozenset({_S.FULL, _S.TEMPORARILY_UNAVAILABLE}),
    _S.TEMPORARILY_UNAVAILABLE: frozenset({_S.AVAILABLE, _S.AT_PACKING_CELL,
                                           _S.FILLING, _S.MAINTENANCE}),
    _S.FULL: frozenset({_S.QUALITY_CHECK, _S.SEALED, _S.TEMPORARILY_UNAVAILABLE}),
    _S.QUALITY_CHECK: frozenset({_S.SEALED, _S.FILLING}),
    _S.SEALED: frozenset({_S.READY_FOR_COLLECTION}),
    _S.READY_FOR_COLLECTION: frozenset({_S.IN_TRANSIT_FROM_CELL}),
    _S.IN_TRANSIT_FROM_CELL: frozenset({_S.DISPATCHED}),
    _S.DISPATCHED: frozenset(),                       # terminal (collected)
    _S.MAINTENANCE: frozenset({_S.AVAILABLE, _S.RETIRED}),
    _S.RETIRED: frozenset(),                          # terminal
}

#: States in which a container can accept newly-planned items.
_FILLABLE = frozenset({_S.AT_PACKING_CELL, _S.FILLING})
#: States that count as "occupied by an active commitment" for planning.
_ACTIVE_RESERVED = frozenset({
    _S.RESERVED, _S.REQUESTED_FOR_DELIVERY, _S.IN_TRANSIT_TO_CELL,
    _S.AT_PACKING_CELL, _S.FILLING})


class InvalidTransition(DomainError):
    """Raised when a lifecycle transition is not in the transition table."""


def is_valid_transition(src: ContainerLifecycleState,
                        dst: ContainerLifecycleState) -> bool:
    return dst in ALLOWED_TRANSITIONS.get(src, frozenset())


# --------------------------------------------------------------------------- #
# Reservation
# --------------------------------------------------------------------------- #


@dataclass
class Reservation:
    """A hold placed on a container by one active plan/scenario."""

    reservation_id: str
    holder: str                       # plan id or scenario id
    segregation_group: str = ""
    created_at: str = ""
    actor: str = "operator"

    def __post_init__(self) -> None:
        _require_id("reservation_id", self.reservation_id)
        if not self.created_at:
            self.created_at = utc_now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reservation_id": self.reservation_id,
            "holder": self.holder,
            "segregation_group": self.segregation_group,
            "created_at": self.created_at,
            "actor": self.actor,
        }


# --------------------------------------------------------------------------- #
# InventoryContainer
# --------------------------------------------------------------------------- #

#: The four facility locations of the simulated cell (see logistics.py).
LOCATION_STORAGE = "container_storage"
LOCATION_CELL = "packing_cell"
LOCATION_INSPECTION = "inspection_station"
LOCATION_DISPATCH = "dispatch_area"


@dataclass
class InventoryContainer:
    """One physical container tracked through its operational lifecycle."""

    container: Container
    state: ContainerLifecycleState = ContainerLifecycleState.REGISTERED
    location: str = LOCATION_STORAGE
    workstation: Optional[str] = None
    reservation: Optional[Reservation] = None
    scenario_id: Optional[str] = None
    plan_id: Optional[str] = None
    transport_task_id: Optional[str] = None
    occupied_volume_mm3: int = 0
    current_payload_kg: float = 0.0
    item_count: int = 0
    sealed: bool = False
    inspection_state: str = "not_inspected"
    plan_digest: str = ""
    contents_digest: str = ""
    revision: int = 0
    last_update: str = ""
    source: Source = Source.SIMULATED
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.state = ContainerLifecycleState(self.state)
        self.source = Source(self.source)
        if not self.last_update:
            self.last_update = utc_now_iso()

    # -- identity / capacity ---------------------------------------------- #

    @property
    def container_id(self) -> str:
        return self.container.container_id

    @property
    def capacity_mm3(self) -> int:
        return self.container.capacity_mm3

    @property
    def remaining_capacity_mm3(self) -> int:
        return max(0, self.capacity_mm3 - self.occupied_volume_mm3)

    @property
    def utilization_pct(self) -> float:
        return (0.0 if self.capacity_mm3 == 0
                else 100.0 * self.occupied_volume_mm3 / self.capacity_mm3)

    @property
    def active_segregation_group(self) -> Optional[str]:
        groups = self.container.allowed_segregation_groups
        return groups[0] if len(groups) == 1 else None

    @property
    def is_available(self) -> bool:
        return self.state is ContainerLifecycleState.AVAILABLE \
            and self.reservation is None

    # -- transitions (internal) ------------------------------------------- #

    def _transition(self, dst: ContainerLifecycleState, actor: str, reason: str,
                    timestamp: Optional[str] = None) -> Dict[str, Any]:
        src = self.state
        if src is dst:
            # Idempotent no-op is allowed but still audited.
            record = self._record(src, dst, actor, reason or "noop", timestamp,
                                   applied=False)
            return record
        if not is_valid_transition(src, dst):
            self._record(src, dst, actor, reason, timestamp, applied=False,
                         rejected=True)
            raise InvalidTransition(
                f"{self.container_id}: {src.value} -> {dst.value} is not a "
                f"permitted lifecycle transition")
        self.state = dst
        self.revision += 1
        return self._record(src, dst, actor, reason, timestamp, applied=True)

    def _record(self, src, dst, actor, reason, timestamp, *,
                applied: bool, rejected: bool = False) -> Dict[str, Any]:
        ts = timestamp or utc_now_iso()
        self.last_update = ts
        rec = {
            "from": src.value, "to": dst.value, "actor": actor,
            "reason": reason, "timestamp": ts, "revision": self.revision,
            "applied": applied, "rejected": rejected,
        }
        self.history.append(rec)
        return rec

    # -- contents --------------------------------------------------------- #

    def apply_contents(self, *, occupied_volume_mm3: int, payload_kg: float,
                       item_count: int, plan_id: Optional[str] = None,
                       contents_digest: str = "") -> None:
        """Record the packed contents assigned to this container by a plan."""
        self.occupied_volume_mm3 = int(occupied_volume_mm3)
        self.current_payload_kg = float(payload_kg)
        self.item_count = int(item_count)
        if plan_id:
            self.plan_id = plan_id
        if contents_digest:
            self.contents_digest = contents_digest
        self.revision += 1
        self.last_update = utc_now_iso()

    # -- FIWARE projection ------------------------------------------------ #

    def semantic_state(self) -> Dict[str, Any]:
        """Compact NGSI-LD-friendly state (brief §11). No placement geometry."""
        c = self.container
        return {
            "entity_id": f"urn:ngsi-ld:WISEPACKContainer:{self.container_id}",
            "entity_type": "WISEPACKContainer",
            "container_id": self.container_id,
            "container_type": c.__class__.__name__,
            "inner_width_mm": c.inner_width_mm,
            "inner_depth_mm": c.inner_depth_mm,
            "inner_height_mm": c.inner_height_mm,
            "max_payload_kg": c.max_payload_kg,
            "current_payload_kg": round(self.current_payload_kg, 3),
            "capacity_mm3": self.capacity_mm3,
            "occupied_volume_mm3": self.occupied_volume_mm3,
            "utilization_pct": round(self.utilization_pct, 2),
            "remaining_capacity_mm3": self.remaining_capacity_mm3,
            "compatible_segregation_groups":
                list(c.allowed_segregation_groups),
            "active_segregation_group": self.active_segregation_group,
            "lifecycle_state": self.state.value,
            "availability": self.is_available,
            "location": self.location,
            "workstation": self.workstation,
            "reservation": self.reservation.reservation_id
                if self.reservation else None,
            "scenario": self.scenario_id,
            "plan": self.plan_id,
            "transport_task": self.transport_task_id,
            "item_count": self.item_count,
            "sealed": self.sealed,
            "inspection_state": self.inspection_state,
            "plan_digest": self.plan_digest,
            "contents_digest": self.contents_digest,
            "revision": self.revision,
            "last_update": self.last_update,
            "source": self.source.value,
        }

    def to_dict(self) -> Dict[str, Any]:
        d = self.semantic_state()
        d["history"] = list(self.history)
        d["spec"] = self.container.to_dict()
        return d


# --------------------------------------------------------------------------- #
# ContainerInventory
# --------------------------------------------------------------------------- #

_SUMMARY_STATES = [
    "total", "available", "reserved", "at_packing_cell", "filling", "full",
    "unavailable", "ready_for_collection", "dispatched",
]


@dataclass
class InventoryEvent:
    """The audited outcome of an inventory operation (for the integration layer)."""

    operation: str
    container_id: str
    from_state: str
    to_state: str
    actor: str
    reason: str
    timestamp: str
    revision: int
    applied: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation, "container_id": self.container_id,
            "from_state": self.from_state, "to_state": self.to_state,
            "actor": self.actor, "reason": self.reason,
            "timestamp": self.timestamp, "revision": self.revision,
            "applied": self.applied, "source": Source.SIMULATED.value,
        }


class ContainerInventory:
    """A collection of tracked containers with audited operations."""

    def __init__(self, simulated: bool = True) -> None:
        self._by_id: Dict[str, InventoryContainer] = {}
        self.simulated = simulated
        self.revision = 0
        self._reservation_seq = 0
        self.shortage_events: List[Dict[str, Any]] = []

    # -- population ------------------------------------------------------- #

    def __contains__(self, container_id: str) -> bool:
        return container_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, container_id: str) -> InventoryContainer:
        if container_id not in self._by_id:
            raise DomainError(f"unknown container {container_id!r}")
        return self._by_id[container_id]

    def all(self) -> List[InventoryContainer]:
        return list(self._by_id.values())

    def _bump(self) -> None:
        self.revision += 1

    def _event(self, op: str, ic: InventoryContainer,
               rec: Dict[str, Any]) -> InventoryEvent:
        self._bump()
        return InventoryEvent(
            operation=op, container_id=ic.container_id,
            from_state=rec["from"], to_state=rec["to"], actor=rec["actor"],
            reason=rec["reason"], timestamp=rec["timestamp"],
            revision=self.revision, applied=rec["applied"])

    # -- operations (brief §13) — each validates, audits, bumps revision -- #

    def register(self, container: Container, *, location: str = LOCATION_STORAGE,
                 actor: str = "operator", reason: str = "commissioned",
                 timestamp: Optional[str] = None) -> InventoryEvent:
        if container.container_id in self._by_id:
            raise DomainError(f"container {container.container_id} already registered")
        ic = InventoryContainer(
            container=container, state=ContainerLifecycleState.REGISTERED,
            location=location,
            source=Source.SIMULATED if self.simulated else Source.MEASURED)
        ic._record(ContainerLifecycleState.REGISTERED,
                   ContainerLifecycleState.REGISTERED, actor, reason, timestamp,
                   applied=True)
        self._by_id[container.container_id] = ic
        self._bump()
        return InventoryEvent("register", ic.container_id, "-",
                              ic.state.value, actor, reason,
                              ic.last_update, self.revision, True)

    def _op(self, op: str, container_id: str,
            dst: ContainerLifecycleState, actor: str, reason: str,
            timestamp: Optional[str]) -> InventoryEvent:
        ic = self.get(container_id)
        rec = ic._transition(dst, actor, reason, timestamp)
        return self._event(op, ic, rec)

    def mark_available(self, cid, actor="operator", reason="made available",
                       timestamp=None) -> InventoryEvent:
        return self._op("mark_available", cid, _S.AVAILABLE, actor, reason, timestamp)

    def reserve(self, cid, holder: str, *, segregation_group: str = "",
                actor="operator", reason="reserved for plan",
                timestamp=None) -> InventoryEvent:
        ic = self.get(cid)
        ev = self._op("reserve", cid, _S.RESERVED, actor, reason, timestamp)
        self._reservation_seq += 1
        ic.reservation = Reservation(
            reservation_id=f"res-{self._reservation_seq:04d}", holder=holder,
            segregation_group=segregation_group, actor=actor)
        ic.scenario_id = ic.scenario_id or holder
        return ev

    def release_reservation(self, cid, actor="operator",
                            reason="reservation released",
                            timestamp=None) -> InventoryEvent:
        ic = self.get(cid)
        ev = self._op("release_reservation", cid, _S.AVAILABLE, actor, reason,
                      timestamp)
        ic.reservation = None
        return ev

    def request_delivery(self, cid, *, workstation: str = LOCATION_CELL,
                         actor="operator", reason="delivery requested",
                         timestamp=None) -> InventoryEvent:
        ic = self.get(cid)
        ev = self._op("request_delivery", cid, _S.REQUESTED_FOR_DELIVERY, actor,
                      reason, timestamp)
        ic.workstation = workstation
        return ev

    def mark_in_transit_to_cell(self, cid, actor="logistics_sim",
                                reason="en route to cell",
                                timestamp=None) -> InventoryEvent:
        return self._op("mark_in_transit_to_cell", cid, _S.IN_TRANSIT_TO_CELL,
                        actor, reason, timestamp)

    def mark_at_cell(self, cid, actor="logistics_sim", reason="arrived at cell",
                     timestamp=None) -> InventoryEvent:
        ic = self.get(cid)
        ev = self._op("mark_at_cell", cid, _S.AT_PACKING_CELL, actor, reason,
                      timestamp)
        ic.location = LOCATION_CELL
        return ev

    def mark_filling(self, cid, actor="orchestrator", reason="filling started",
                     timestamp=None) -> InventoryEvent:
        return self._op("mark_filling", cid, _S.FILLING, actor, reason, timestamp)

    def mark_unavailable(self, cid, actor="operator",
                         reason="temporarily unavailable",
                         timestamp=None) -> InventoryEvent:
        return self._op("mark_unavailable", cid, _S.TEMPORARILY_UNAVAILABLE,
                        actor, reason, timestamp)

    def restore(self, cid, *, to: ContainerLifecycleState = _S.AVAILABLE,
                actor="operator", reason="restored", timestamp=None) -> InventoryEvent:
        return self._op("restore", cid, to, actor, reason, timestamp)

    def mark_full(self, cid, actor="orchestrator", reason="container full",
                  timestamp=None) -> InventoryEvent:
        return self._op("mark_full", cid, _S.FULL, actor, reason, timestamp)

    def request_quality_check(self, cid, actor="operator",
                              reason="quality check", timestamp=None) -> InventoryEvent:
        ic = self.get(cid)
        ev = self._op("request_quality_check", cid, _S.QUALITY_CHECK, actor,
                      reason, timestamp)
        ic.inspection_state = "in_progress"
        ic.location = LOCATION_INSPECTION
        return ev

    def mark_sealed(self, cid, actor="operator", reason="sealed",
                    timestamp=None) -> InventoryEvent:
        ic = self.get(cid)
        ev = self._op("mark_sealed", cid, _S.SEALED, actor, reason, timestamp)
        ic.sealed = True
        ic.inspection_state = "passed"
        return ev

    def request_collection(self, cid, actor="operator",
                           reason="collection requested",
                           timestamp=None) -> InventoryEvent:
        return self._op("request_collection", cid, _S.READY_FOR_COLLECTION,
                        actor, reason, timestamp)

    def mark_in_transit_from_cell(self, cid, actor="logistics_sim",
                                  reason="collected from cell",
                                  timestamp=None) -> InventoryEvent:
        return self._op("mark_in_transit_from_cell", cid, _S.IN_TRANSIT_FROM_CELL,
                        actor, reason, timestamp)

    def mark_dispatched(self, cid, actor="logistics_sim", reason="dispatched",
                        timestamp=None) -> InventoryEvent:
        ic = self.get(cid)
        ev = self._op("mark_dispatched", cid, _S.DISPATCHED, actor, reason,
                      timestamp)
        ic.location = LOCATION_DISPATCH
        return ev

    def send_to_maintenance(self, cid, actor="operator", reason="maintenance",
                            timestamp=None) -> InventoryEvent:
        return self._op("send_to_maintenance", cid, _S.MAINTENANCE, actor,
                        reason, timestamp)

    def retire(self, cid, actor="operator", reason="retired",
               timestamp=None) -> InventoryEvent:
        return self._op("retire", cid, _S.RETIRED, actor, reason, timestamp)

    # -- inventory-aware selection (brief §14) ---------------------------- #

    def selectable_for(self, segregation_group: str, *,
                       workstation: Optional[str] = None) -> List[InventoryContainer]:
        """Containers the optimizer MAY use for a plan of ``segregation_group``.

        A container qualifies only when it is available (or already at the cell
        and not reserved by another plan), accepts the group, and is not
        unavailable / full / sealed / dispatched / retired.
        """
        out: List[InventoryContainer] = []
        for ic in self._by_id.values():
            if ic.state in (_S.FULL, _S.SEALED, _S.DISPATCHED, _S.RETIRED,
                            _S.TEMPORARILY_UNAVAILABLE, _S.MAINTENANCE,
                            _S.READY_FOR_COLLECTION, _S.IN_TRANSIT_FROM_CELL):
                continue
            if not ic.container.accepts_group(segregation_group):
                continue
            if ic.reservation is not None \
                    and ic.reservation.segregation_group not in ("", segregation_group):
                continue
            if workstation and ic.workstation and ic.workstation != workstation:
                continue
            out.append(ic)
        return out

    def compatible_capacity_mm3(self, segregation_group: str) -> int:
        return sum(ic.remaining_capacity_mm3
                   for ic in self.selectable_for(segregation_group))

    def record_shortage(self, segregation_group: str, needed_mm3: int, *,
                        actor="orchestrator") -> Dict[str, Any]:
        ev = {
            "segregation_group": segregation_group,
            "needed_mm3": int(needed_mm3),
            "available_mm3": self.compatible_capacity_mm3(segregation_group),
            "timestamp": utc_now_iso(), "actor": actor,
            "source": Source.SIMULATED.value,
        }
        self.shortage_events.append(ev)
        self._bump()
        return ev

    # -- summary / KPIs (brief §12) --------------------------------------- #

    def counts(self) -> Dict[str, int]:
        c = {k: 0 for k in _SUMMARY_STATES}
        c["total"] = len(self._by_id)
        for ic in self._by_id.values():
            s = ic.state
            if s is _S.AVAILABLE and ic.reservation is None:
                c["available"] += 1
            if ic.reservation is not None or s in _ACTIVE_RESERVED:
                if s is not _S.FILLING and s is not _S.AT_PACKING_CELL:
                    c["reserved"] += 1
            if s is _S.AT_PACKING_CELL:
                c["at_packing_cell"] += 1
            if s is _S.FILLING:
                c["filling"] += 1
            if s is _S.FULL:
                c["full"] += 1
            if s is _S.TEMPORARILY_UNAVAILABLE:
                c["unavailable"] += 1
            if s is _S.READY_FOR_COLLECTION:
                c["ready_for_collection"] += 1
            if s is _S.DISPATCHED:
                c["dispatched"] += 1
        return c

    def summary(self, segregation_groups: Sequence[str] = ()) -> Dict[str, Any]:
        groups = list(segregation_groups) or sorted({
            g for ic in self._by_id.values()
            for g in (ic.container.allowed_segregation_groups or ("A",))})
        counts = self.counts()
        return {
            **counts,
            "compatible_capacity_mm3": {
                g: self.compatible_capacity_mm3(g) for g in groups},
            "compatible_capacity_available_mm3": sum(
                self.compatible_capacity_mm3(g) for g in groups),
            "forecast_shortage": len(self.shortage_events) > 0,
            "shortage_events": len(self.shortage_events),
            "revision": self.revision,
            "simulated": self.simulated,
            "source": (Source.SIMULATED if self.simulated
                       else Source.MEASURED).value,
        }

    def semantic_states(self) -> List[Dict[str, Any]]:
        return [ic.semantic_state() for ic in self._by_id.values()]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "containers": [ic.to_dict() for ic in self._by_id.values()],
            "summary": self.summary(),
            "shortage_events": list(self.shortage_events),
        }


__all__ = [
    "ContainerLifecycleState", "ALLOWED_TRANSITIONS", "is_valid_transition",
    "InvalidTransition", "Reservation", "InventoryContainer", "InventoryEvent",
    "ContainerInventory", "LOCATION_STORAGE", "LOCATION_CELL",
    "LOCATION_INSPECTION", "LOCATION_DISPATCH",
]
