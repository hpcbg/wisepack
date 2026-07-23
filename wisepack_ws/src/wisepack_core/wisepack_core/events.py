"""ActionEvent — the WISEPACK audit trail record, and the dynamic-event model.

Every workflow action produces exactly one ActionEvent. That event is published
on ``/wisepack/action/event`` and reaches FIWARE through the Orion-LD DDS bridge;
it is the regulatory-traceability artefact the proposal describes ("every robotic
action and operator intervention must be recorded in a standards-compliant audit
trail"). Two properties matter more than the payload:

  * the sequence number is monotonic and gap-free within a run, so a consumer can
    prove nothing was lost rather than assume it;
  * ``source`` is never guessed. A simulated pick outcome is labelled simulated
    even when everything around it is measured.

There is no ROS and no HTTP in this module. The transport lives in
wisepack_orchestration; this file only defines the record and the sequencer.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .domain import SCHEMA_VERSION, Source

# --------------------------------------------------------------------------- #
# Workflow stages and actions
# --------------------------------------------------------------------------- #


class Stage(str, Enum):
    """Behaviour-tree stages. The order here IS the documented workflow order."""

    IDLE = "IDLE"
    GENERATE_OR_LOAD_SCENARIO = "GENERATE_OR_LOAD_SCENARIO"
    SCAN_SOURCE_BIN = "SCAN_SOURCE_BIN"
    DETECT_ITEMS = "DETECT_ITEMS"
    GENERATE_BASELINE_PLAN = "GENERATE_BASELINE_PLAN"
    GENERATE_OPTIMIZED_PLAN = "GENERATE_OPTIMIZED_PLAN"
    DIGITAL_TWIN_VALIDATE = "DIGITAL_TWIN_VALIDATE"
    WAIT_FOR_OPERATOR_APPROVAL = "WAIT_FOR_OPERATOR_APPROVAL"
    PICK_ITEM = "PICK_ITEM"
    VERIFY_PICK = "VERIFY_PICK"
    PLACE_ITEM = "PLACE_ITEM"
    VERIFY_PLACEMENT = "VERIFY_PLACEMENT"
    UPDATE_CONTAINER_STATE = "UPDATE_CONTAINER_STATE"
    NEXT_ITEM = "NEXT_ITEM"
    REPLAN = "REPLAN"
    COMPLETE = "COMPLETE"
    PAUSED = "PAUSED"
    DEGRADED = "DEGRADED"


#: Stages in which no physical action has yet been authorised. The orchestrator
#: asserts that PICK_ITEM is never entered while the plan sits in one of these.
PRE_APPROVAL_STAGES = frozenset({
    Stage.IDLE, Stage.GENERATE_OR_LOAD_SCENARIO, Stage.SCAN_SOURCE_BIN,
    Stage.DETECT_ITEMS, Stage.GENERATE_BASELINE_PLAN,
    Stage.GENERATE_OPTIMIZED_PLAN, Stage.DIGITAL_TWIN_VALIDATE,
    Stage.WAIT_FOR_OPERATOR_APPROVAL,
})


class Result(str, Enum):
    OK = "ok"
    FAILED = "failed"
    RETRY = "retry"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    PENDING = "pending"


class Actor(str, Enum):
    """Which module produced the event. Mirrors the proposal's layer model."""

    TASK_GENERATOR = "task_generator"
    PERCEPTION_SIM = "perception_simulator"
    OPTIMIZER = "packing_optimizer"
    DIGITAL_TWIN = "digital_twin_validator"
    ORCHESTRATOR = "hitl_orchestrator"
    ROBOT_SIM = "robot_simulator"
    OPERATOR = "operator"
    EVENT_INJECTOR = "dynamic_event_injector"
    SYSTEM = "system"


# --------------------------------------------------------------------------- #
# ActionEvent
# --------------------------------------------------------------------------- #


def utc_now_iso() -> str:
    """RFC3339 UTC timestamp with milliseconds — the NGSI-LD-friendly form."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z")


@dataclass
class ActionEvent:
    """One recorded workflow action.

    ``details`` must stay JSON-serialisable and small: the whole event is carried
    as a single ``std_msgs/String`` over DDS into one NGSI-LD attribute, and an
    unbounded blob there is what turns a working audit trail into a truncated
    one. ``MAX_SERIALISED_BYTES`` is enforced on serialisation rather than left
    to chance.
    """

    #: Conservative ceiling for one event on the DDS path. Fast DDS defaults are
    #: far larger, but an event that needs fragmenting is an event that will
    #: intermittently fail to reach the broker, which is worse than a truncated
    #: details block that says it was truncated.
    MAX_SERIALISED_BYTES = 8192

    stage: Stage
    action: str
    actor: Actor
    result: Result = Result.OK
    sequence: int = 0
    event_id: str = ""
    timestamp: str = field(default_factory=utc_now_iso)
    scenario_id: str = ""
    cycle_id: str = ""
    item_id: Optional[str] = None
    container_id: Optional[str] = None
    duration_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    source: Source = Source.MEASURED

    def __post_init__(self) -> None:
        self.stage = Stage(self.stage)
        self.actor = Actor(self.actor)
        self.result = Result(self.result)
        self.source = Source(self.source)
        if not self.event_id:
            self.event_id = f"evt-{uuid.uuid4().hex[:12]}"
        if not self.message:
            self.message = f"{self.stage.value}/{self.action}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "scenario_id": self.scenario_id,
            "cycle_id": self.cycle_id,
            "stage": self.stage.value,
            "action": self.action,
            "actor": self.actor.value,
            "item_id": self.item_id,
            "container_id": self.container_id,
            "result": self.result.value,
            "duration_ms": round(self.duration_ms, 3),
            "message": self.message,
            "details": self.details,
            "source": self.source.value,
        }

    def to_json(self) -> str:
        """Compact JSON for the DDS payload, truncating ``details`` if oversized.

        Truncation is visible: the details block is replaced by an explicit
        marker rather than silently trimmed, because a consumer that cannot tell
        it received a partial record cannot rely on the audit trail at all.
        """
        payload = self.to_dict()
        blob = json.dumps(payload, separators=(",", ":"), default=str)
        if len(blob.encode("utf-8")) <= self.MAX_SERIALISED_BYTES:
            return blob
        dropped = sorted(payload["details"].keys())
        payload["details"] = {
            "_truncated": True,
            "_dropped_keys": dropped,
            "_reason": f"event exceeded {self.MAX_SERIALISED_BYTES} B on the DDS path",
        }
        return json.dumps(payload, separators=(",", ":"), default=str)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ActionEvent":
        return ActionEvent(
            stage=Stage(d["stage"]),
            action=d["action"],
            actor=Actor(d["actor"]),
            result=Result(d.get("result", "ok")),
            sequence=int(d.get("sequence", 0)),
            event_id=d.get("event_id", ""),
            timestamp=d.get("timestamp", utc_now_iso()),
            scenario_id=d.get("scenario_id", ""),
            cycle_id=d.get("cycle_id", ""),
            item_id=d.get("item_id"),
            container_id=d.get("container_id"),
            duration_ms=float(d.get("duration_ms", 0.0)),
            message=d.get("message", ""),
            details=dict(d.get("details", {})),
            source=Source(d.get("source", "measured")),
        )

    @staticmethod
    def csv_header() -> Sequence[str]:
        return ("sequence", "timestamp", "event_id", "scenario_id", "cycle_id",
                "stage", "action", "actor", "item_id", "container_id", "result",
                "duration_ms", "source", "message")

    def csv_row(self) -> Sequence[Any]:
        return (self.sequence, self.timestamp, self.event_id, self.scenario_id,
                self.cycle_id, self.stage.value, self.action, self.actor.value,
                self.item_id or "", self.container_id or "", self.result.value,
                round(self.duration_ms, 3), self.source.value, self.message)


# --------------------------------------------------------------------------- #
# Sequencer / recorder
# --------------------------------------------------------------------------- #


class ActionLog:
    """Assigns monotonic sequence numbers and keeps the in-process history.

    Thread-safe because in live mode the ROS executor thread and the FastAPI
    event loop both emit into it. The sequence counter is the audit trail's
    integrity guarantee, and a duplicated or skipped number from a benign race
    would destroy exactly the property the demo claims.
    """

    def __init__(self, scenario_id: str = "", cycle_id: str = "",
                 max_history: int = 5000) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._events: List[ActionEvent] = []
        self._max_history = max_history
        self.scenario_id = scenario_id
        self.cycle_id = cycle_id
        #: Callables invoked (inside the lock) for each event — the ROS publisher
        #: and the dashboard cache attach here.
        self._sinks: List[Any] = []

    # -- wiring ------------------------------------------------------------- #

    def add_sink(self, fn) -> None:
        with self._lock:
            self._sinks.append(fn)

    def set_context(self, scenario_id: Optional[str] = None,
                    cycle_id: Optional[str] = None) -> None:
        with self._lock:
            if scenario_id is not None:
                self.scenario_id = scenario_id
            if cycle_id is not None:
                self.cycle_id = cycle_id

    # -- emission ----------------------------------------------------------- #

    def emit(self, stage: Stage, action: str, actor: Actor,
             result: Result = Result.OK, *, item_id: Optional[str] = None,
             container_id: Optional[str] = None, duration_ms: float = 0.0,
             message: str = "", details: Optional[Dict[str, Any]] = None,
             source: Source = Source.MEASURED) -> ActionEvent:
        """Create, number and record one event. Returns it for the caller to publish."""
        with self._lock:
            self._seq += 1
            event = ActionEvent(
                stage=stage, action=action, actor=actor, result=result,
                sequence=self._seq, scenario_id=self.scenario_id,
                cycle_id=self.cycle_id, item_id=item_id,
                container_id=container_id, duration_ms=duration_ms,
                message=message, details=dict(details or {}), source=source)
            self._events.append(event)
            if len(self._events) > self._max_history:
                del self._events[:len(self._events) - self._max_history]
            sinks = list(self._sinks)
        for sink in sinks:
            # A broken sink must not silence the audit trail for the others, but
            # it must not be invisible either.
            try:
                sink(event)
            except Exception as exc:                    # noqa: BLE001
                print(f"[action-log] sink {sink!r} raised: {exc!r}")
        return event

    # -- inspection --------------------------------------------------------- #

    @property
    def count(self) -> int:
        with self._lock:
            return self._seq

    def events(self, limit: Optional[int] = None,
               newest_first: bool = False) -> List[ActionEvent]:
        with self._lock:
            evts = list(self._events)
        if newest_first:
            evts.reverse()
        return evts[:limit] if limit else evts

    def by_stage(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self.events():
            counts[e.stage.value] = counts.get(e.stage.value, 0) + 1
        return counts

    def by_action(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self.events():
            counts[e.action] = counts.get(e.action, 0) + 1
        return counts

    def duration_by_stage_ms(self) -> Dict[str, float]:
        """Total recorded duration per stage — the analytics 'action duration' chart."""
        totals: Dict[str, float] = {}
        for e in self.events():
            if e.duration_ms:
                totals[e.stage.value] = round(
                    totals.get(e.stage.value, 0.0) + e.duration_ms, 3)
        return totals

    def sequence_is_monotonic(self) -> Tuple[bool, str]:
        """(ok, explanation) — verifies 1..N with no gaps and no repeats."""
        seqs = [e.sequence for e in self.events()]
        expected = list(range(1, len(seqs) + 1))
        if seqs == expected:
            return True, f"{len(seqs)} events, sequence 1..{len(seqs)} contiguous"
        return False, f"expected {expected[:5]}... got {seqs[:5]}..."

    # -- export ------------------------------------------------------------- #

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_dict(), separators=(",", ":"),
                                    default=str) for e in self.events())

    def csv_rows(self) -> Tuple[Sequence[str], List[Sequence[Any]]]:
        return ActionEvent.csv_header(), [e.csv_row() for e in self.events()]


# --------------------------------------------------------------------------- #
# Dynamic events
# --------------------------------------------------------------------------- #


class DynamicEventType(str, Enum):
    ITEM_INJECT = "item_inject"
    ITEM_REMOVED = "item_removed"
    ITEM_RECLASSIFIED = "item_reclassified"
    CONTAINER_UNAVAILABLE = "container_unavailable"
    CONTAINER_RESTORED = "container_restored"
    OPERATOR_REJECT = "operator_reject"
    GRASP_FAILURE = "grasp_failure"
    SEGREGATION_RULE_CHANGE = "segregation_rule_change"
    OPTIMIZER_TIMEOUT = "optimizer_timeout"


#: Event types that invalidate the current plan and force a controlled re-plan.
#: A grasp failure deliberately does NOT: it is a retry at the execution layer,
#: and re-planning the whole container because one grasp slipped would be both
#: wasteful and a misleading demonstration of what triggers re-planning.
REPLAN_TRIGGERS = frozenset({
    DynamicEventType.ITEM_INJECT,
    DynamicEventType.ITEM_REMOVED,
    DynamicEventType.ITEM_RECLASSIFIED,
    DynamicEventType.CONTAINER_UNAVAILABLE,
    DynamicEventType.OPERATOR_REJECT,
    DynamicEventType.SEGREGATION_RULE_CHANGE,
})

_TRIGGER_RE = re.compile(
    r"^(?:"
    r"stage:(?P<stage>[A-Z_]+)"
    r"|placement:(?P<placement>\d+)"
    r"|t:(?P<seconds>\d+(?:\.\d+)?)"
    r")$")


@dataclass
class DynamicEvent:
    """A scripted disturbance, reproducible from the scenario configuration.

    ``trigger`` is one of three deterministic forms — deliberately not wall-clock
    time, which would make the demo non-reproducible on a slower machine:

        ``stage:PICK_ITEM``   fire on first entry to that workflow stage
        ``placement:3``       fire before executing the 4th placement (0-based)
        ``t:12.5``            fire 12.5 s after execution starts (wall clock)

    The ``t:`` form exists because HARVEST's scenarios are time-driven and it
    reads naturally in a config file, but ``placement:`` is preferred for the
    acceptance demo since it is machine-speed independent.
    """

    event_type: DynamicEventType
    trigger: str
    label: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    fired: bool = False

    def __post_init__(self) -> None:
        self.event_type = DynamicEventType(self.event_type)
        if not _TRIGGER_RE.match(self.trigger):
            raise ValueError(
                f"dynamic event trigger {self.trigger!r} is not one of "
                "'stage:<STAGE>', 'placement:<n>' or 't:<seconds>'")
        if not self.label:
            self.label = self.event_type.value.replace("_", " ")

    @property
    def forces_replan(self) -> bool:
        return self.event_type in REPLAN_TRIGGERS

    def matches(self, *, stage: Optional[Stage] = None,
                placement_index: Optional[int] = None,
                elapsed_s: Optional[float] = None) -> bool:
        """True when this event's trigger condition is met by the given state."""
        if self.fired:
            return False
        m = _TRIGGER_RE.match(self.trigger)
        assert m is not None                     # guaranteed by __post_init__
        if m.group("stage") is not None:
            return stage is not None and stage.value == m.group("stage")
        if m.group("placement") is not None:
            return (placement_index is not None
                    and placement_index == int(m.group("placement")))
        return elapsed_s is not None and elapsed_s >= float(m.group("seconds"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "trigger": self.trigger,
            "label": self.label,
            "payload": dict(self.payload),
            "fired": self.fired,
            "forces_replan": self.forces_replan,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DynamicEvent":
        return DynamicEvent(
            event_type=DynamicEventType(d["event_type"]),
            trigger=d["trigger"],
            label=d.get("label", ""),
            payload=dict(d.get("payload", {})),
            fired=bool(d.get("fired", False)),
        )


def load_dynamic_events(raw: Iterable[Dict[str, Any]]) -> List[DynamicEvent]:
    """Parse the ``dynamic_events:`` list from a scenario YAML/JSON block."""
    return [DynamicEvent.from_dict(d) for d in raw]


class Stopwatch:
    """Monotonic millisecond timer for ``duration_ms`` on action events.

    ``time.monotonic`` rather than ``time.time``: an NTP step mid-demo must not
    be able to produce a negative duration in the audit trail.
    """

    def __init__(self) -> None:
        self._t0 = time.monotonic()

    def reset(self) -> None:
        self._t0 = time.monotonic()

    @property
    def ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000.0

    def lap(self) -> float:
        elapsed = self.ms
        self.reset()
        return elapsed


__all__ = [
    "Stage", "PRE_APPROVAL_STAGES", "Result", "Actor", "ActionEvent",
    "ActionLog", "DynamicEventType", "REPLAN_TRIGGERS", "DynamicEvent",
    "load_dynamic_events", "Stopwatch", "utc_now_iso",
]
