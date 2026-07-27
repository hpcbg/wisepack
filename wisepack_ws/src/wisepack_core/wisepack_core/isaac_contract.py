"""The WISEPACK <-> Isaac Sim transport contract.

WHY A NEW CONTRACT AT ALL
-------------------------
Everything above the domain core already speaks the canonical topic contract in
``wisepack_bringup.topics``, and this module does NOT replace it. The
orchestrator still owns and publishes every state, plan, KPI and action-event
topic exactly as before. What is missing from that contract is a *duplex*
channel for one specific thing: "physically execute this one accepted placement"
and "here is what physically happened". That is what these two messages are.

The same three constraints that shaped the canonical contract apply here, and
for the same measured reasons (see ``wisepack_bringup.topics``):

  * the payload is a versioned JSON document inside ``std_msgs/String``, because
    a ``wisepack_interfaces`` package would be unbridgeable on the Orion-LD DDS
    path and — just as importantly here — would have to be built and then
    imported by Isaac Sim's BUNDLED Python, which is a different interpreter
    with a different ROS 2 build. Standard messages need no such coupling.
  * neither topic's final segment is the reserved ``status`` leaf.
  * exactly one writer per topic: WISEPACK writes commands, Isaac writes
    feedback. Never both.

NOTHING HERE IS ISAAC-SPECIFIC
------------------------------
The name says Isaac because Isaac is the first implementation, not because the
messages assume one. There is no simulator concept in this file: no USD path, no
PhysX setting, no renderer, no joint. A command says "pick this item, which is
here, and put it there"; a report says "this is what physically happened". A real
robot cell answering the same two topics is a drop-in replacement for the
simulator, and the orchestrator side — ``wisepack_orchestration.isaac_bridge`` —
would not change at all, because it also contains no simulator imports.

Everything that IS Isaac-specific lives under ``simulators/isaac/``, and nothing
outside that directory imports ``isaacsim``, ``omni``, ``carb`` or ``pxr``. That
boundary is asserted by a test rather than left as an intention (see
``tests/test_isaac_backend.py::test_simulator_imports_are_confined_to_the_adapter``).

THIS MODULE IMPORTS NOTHING
---------------------------
Not from ROS, not from the rest of ``wisepack_core``, not from Isaac. It is
plain-stdlib Python so that the identical file is the contract on both sides of
the wire: the orchestrator imports it under the Vulcanexus interpreter inside
Docker, and the Isaac application imports it under Isaac Sim's bundled
interpreter on the host. One definition, so the two cannot drift.

UNITS
-----
Every length in this contract is **millimetres**, matching the WISEPACK domain
model, and every pose is expressed in a NAMED FRAME. Conversion into Isaac's
metres/Z-up world happens in exactly one place — ``wisepack_core.isaac_transform``
— and nowhere else. See that module for why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

#: Bump the MINOR part for backwards-compatible additions (a new optional
#: field), the MAJOR part when a consumer written against the old version would
#: mis-read the new one. Receivers reject a mismatched MAJOR outright rather
#: than guessing, because a silently mis-parsed pose is a robot moving somewhere
#: nobody asked it to.
SCHEMA_VERSION = "wisepack-isaac/1.0"


def schema_major(version: str) -> str:
    """``"wisepack-isaac/1.0"`` -> ``"wisepack-isaac/1"``."""
    return version.rsplit(".", 1)[0]


class ContractError(ValueError):
    """Raised when a message is malformed, mis-versioned or internally inconsistent."""


def utc_now_iso() -> str:
    """RFC3339 UTC timestamp with milliseconds — same form as the audit trail."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z")


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


class IsaacCommandType(str, Enum):
    """WISEPACK -> Isaac. The complete command vocabulary.

    Deliberately tiny. Isaac is an execution backend, not a second workflow
    engine: it is never told to plan, approve, re-plan or choose an item order.
    """

    #: Open a run. Carries preset/seed so Isaac can build the same scene the
    #: orchestrator planned against, and is the point at which a new run_id
    #: becomes the only one Isaac will accept.
    RUN_BEGIN = "RUN_BEGIN"
    #: Execute exactly one accepted placement: pick this item, place it there.
    EXECUTE_ITEM = "EXECUTE_ITEM"
    #: The plan is finished. Isaac stops accepting EXECUTE_ITEM for this run.
    RUN_END = "RUN_END"
    #: Stop whatever is in progress and fail the current item. Used when the
    #: operator rejects, pauses into a re-plan, or the orchestrator goes away.
    RUN_ABORT = "RUN_ABORT"


class IsaacState(str, Enum):
    """Isaac -> WISEPACK. Physical execution feedback.

    These are PHYSICAL states, and they are mapped onto the existing WISEPACK
    workflow ``Stage`` values by ``wisepack_core.execution.stage_for_isaac_state``
    rather than shown as a parallel state machine. The dashboard timeline, the
    audit trail and the FIWARE ``stage`` attribute keep their existing
    vocabulary; this enum adds physical resolution underneath it.
    """

    READY = "READY"                                # simulator up, scene built
    MOVING_TO_PICK = "MOVING_TO_PICK"
    GRASPING = "GRASPING"
    LIFTING = "LIFTING"
    MOVING_TO_CONTAINER = "MOVING_TO_CONTAINER"
    RELEASING = "RELEASING"
    SETTLING = "SETTLING"                          # PhysX is resolving the drop
    ITEM_COMPLETED = "ITEM_COMPLETED"
    ITEM_FAILED = "ITEM_FAILED"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"


#: States that terminate work on one item. Exactly one of these arrives per
#: EXECUTE_ITEM command, and the orchestrator advances only on these.
ITEM_TERMINAL_STATES = frozenset({IsaacState.ITEM_COMPLETED, IsaacState.ITEM_FAILED})

#: States that terminate the whole run.
RUN_TERMINAL_STATES = frozenset({IsaacState.RUN_COMPLETED, IsaacState.RUN_FAILED})

#: Progress states, in the order a healthy item passes through them. Used by the
#: tests to assert the simulator never reports progress backwards.
ITEM_PROGRESS_ORDER = (
    IsaacState.MOVING_TO_PICK,
    IsaacState.GRASPING,
    IsaacState.LIFTING,
    IsaacState.MOVING_TO_CONTAINER,
    IsaacState.RELEASING,
    IsaacState.SETTLING,
)


# --------------------------------------------------------------------------- #
# Geometry payloads
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Dimensions:
    """The item's geometry, in millimetres. Mirrors ``domain.WasteItem``."""

    length_mm: int
    outer_diameter_mm: int
    inner_diameter_mm: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length_mm": int(self.length_mm),
            "outer_diameter_mm": int(self.outer_diameter_mm),
            "inner_diameter_mm": (None if self.inner_diameter_mm is None
                                  else int(self.inner_diameter_mm)),
        }

    @staticmethod
    def from_dict(d: Any) -> "Dimensions":
        if not isinstance(d, dict):
            raise ContractError(f"dimensions must be an object, got {type(d).__name__}")
        try:
            return Dimensions(
                length_mm=int(d["length_mm"]),
                outer_diameter_mm=int(d["outer_diameter_mm"]),
                inner_diameter_mm=(None if d.get("inner_diameter_mm") is None
                                   else int(d["inner_diameter_mm"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"malformed dimensions: {exc}") from exc


@dataclass(frozen=True)
class Pose:
    """A pose in millimetres, in a NAMED frame, with an axis-aligned orientation.

    ``frame`` is not decoration. The same numbers mean different places in the
    two frames this contract uses:

        ``table``            the item's pick-up pose on the table, expressed in
                             the WISEPACK table frame (origin at the table
                             layout origin, mm, Z up).
        ``container:<id>``   a placement inside that container, expressed in the
                             container's INNER frame — origin at the inner
                             min corner, exactly as ``domain.Placement.position``
                             defines it.

    ``position_mm`` is the CENTRE of the item's bounding box, not its min corner.
    Placements in the domain model carry the min corner; the conversion happens
    once, in ``isaac_transform.placement_pose``, and this contract only ever
    carries centres — a robot is commanded to a centre, never to a corner.

    ``axis`` is the WISEPACK ``Axis`` value ("x"/"y"/"z") the item's length
    points along in ``frame``. Orientation is restricted to those three
    axis-aligned choices because that is the complete set the packing model
    produces (see ``domain.Axis``).
    """

    x_mm: float
    y_mm: float
    z_mm: float
    axis: str = "x"
    frame: str = "table"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_mm": {"x": round(float(self.x_mm), 3),
                            "y": round(float(self.y_mm), 3),
                            "z": round(float(self.z_mm), 3)},
            "axis": self.axis,
            "frame": self.frame,
        }

    @staticmethod
    def from_dict(d: Any) -> "Pose":
        if not isinstance(d, dict):
            raise ContractError(f"pose must be an object, got {type(d).__name__}")
        position = d.get("position_mm")
        if not isinstance(position, dict):
            raise ContractError("pose is missing its position_mm object")
        axis = str(d.get("axis", "x")).lower()
        if axis not in ("x", "y", "z"):
            raise ContractError(f"pose axis must be x, y or z; got {axis!r}")
        try:
            return Pose(
                x_mm=float(position["x"]), y_mm=float(position["y"]),
                z_mm=float(position["z"]), axis=axis,
                frame=str(d.get("frame", "table")))
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"malformed pose position_mm: {exc}") from exc

    def distance_mm(self, other: "Pose") -> float:
        """Euclidean separation. Used to report target-vs-actual placement error."""
        return ((self.x_mm - other.x_mm) ** 2
                + (self.y_mm - other.y_mm) ** 2
                + (self.z_mm - other.z_mm) ** 2) ** 0.5


def _pose_or_none(d: Any) -> Optional[Pose]:
    return None if d is None else Pose.from_dict(d)


def _check_envelope(doc: Any, kind: str) -> Dict[str, Any]:
    """Shared validation for both message types.

    A mismatched schema MAJOR is refused rather than best-effort parsed. The
    alternative — reading a v2 pose with v1 field meanings — moves a robot.
    """
    if not isinstance(doc, dict):
        raise ContractError(f"{kind} must be a JSON object, got {type(doc).__name__}")
    version = doc.get("schema_version")
    if not isinstance(version, str):
        raise ContractError(f"{kind} is missing schema_version")
    if schema_major(version) != schema_major(SCHEMA_VERSION):
        raise ContractError(
            f"{kind} schema_version {version!r} is incompatible with "
            f"{SCHEMA_VERSION!r}; refusing to guess the field meanings")
    run_id = doc.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ContractError(f"{kind} is missing run_id")
    return doc


# --------------------------------------------------------------------------- #
# Command: WISEPACK -> Isaac
# --------------------------------------------------------------------------- #


@dataclass
class IsaacCommand:
    """One instruction from the orchestrator to the Isaac execution backend.

    ``sequence_index`` is the placement's 0-based index within the accepted
    plan's execution order. It is the de-duplication key: Isaac executes each
    ``(run_id, sequence_index)`` at most once, so a command republished by a
    transient-local latch after a reconnect cannot make the robot pick the same
    item twice.
    """

    command: IsaacCommandType
    run_id: str
    sequence_index: int = -1
    #: Which ATTEMPT at ``sequence_index`` this is, 0-based.
    #:
    #: Without it, de-duplication and retry are the same event. A physical
    #: attempt that fails is legitimately re-commanded for the SAME placement —
    #: the simulated backend does exactly this, with the same retry budget — and
    #: keying the duplicate guard on ``sequence_index`` alone silently discarded
    #: the retry as a replay. Observed end to end: the arm failed to reach a
    #: pre-grasp pose, the orchestrator dispatched attempt 1, and the simulator
    #: dropped it as already-executed.
    attempt: int = 0
    item_id: Optional[str] = None
    dimensions: Optional[Dimensions] = None
    source_pose: Optional[Pose] = None
    target_pose: Optional[Pose] = None
    container_id: Optional[str] = None
    #: Inner size of the target container in mm, so Isaac can validate the
    #: settled object against the same footprint the optimizer planned into.
    container_inner_mm: Optional[Dict[str, int]] = None
    preset: str = ""
    seed: int = 0
    plan_id: str = ""
    total_items: int = 0
    timestamp: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.command = IsaacCommandType(self.command)
        if self.command is IsaacCommandType.EXECUTE_ITEM:
            missing = [name for name, value in (
                ("item_id", self.item_id), ("dimensions", self.dimensions),
                ("source_pose", self.source_pose),
                ("target_pose", self.target_pose)) if value is None]
            if missing:
                raise ContractError(
                    f"EXECUTE_ITEM for {self.item_id!r} is missing {missing}; a "
                    "physical pick cannot be commanded without a full pose pair")
            if self.sequence_index < 0:
                raise ContractError(
                    f"EXECUTE_ITEM for {self.item_id!r} needs a sequence_index "
                    ">= 0; it is the duplicate-execution guard")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "command": self.command.value,
            "sequence_index": self.sequence_index,
            "attempt": int(self.attempt),
            "item_id": self.item_id,
            "container_id": self.container_id,
            "dimensions": self.dimensions.to_dict() if self.dimensions else None,
            "source_pose": self.source_pose.to_dict() if self.source_pose else None,
            "target_pose": self.target_pose.to_dict() if self.target_pose else None,
            "container_inner_mm": (dict(self.container_inner_mm)
                                   if self.container_inner_mm else None),
            "preset": self.preset,
            "seed": int(self.seed),
            "total_items": int(self.total_items),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), default=str)

    @staticmethod
    def from_dict(doc: Any) -> "IsaacCommand":
        doc = _check_envelope(doc, "isaac command")
        try:
            command = IsaacCommandType(doc["command"])
        except (KeyError, ValueError) as exc:
            raise ContractError(
                f"unknown or missing isaac command {doc.get('command')!r}") from exc
        dims = doc.get("dimensions")
        return IsaacCommand(
            command=command,
            run_id=doc["run_id"],
            sequence_index=int(doc.get("sequence_index", -1)),
            attempt=int(doc.get("attempt", 0)),
            item_id=doc.get("item_id"),
            dimensions=Dimensions.from_dict(dims) if dims else None,
            source_pose=_pose_or_none(doc.get("source_pose")),
            target_pose=_pose_or_none(doc.get("target_pose")),
            container_id=doc.get("container_id"),
            container_inner_mm=doc.get("container_inner_mm"),
            preset=str(doc.get("preset", "")),
            seed=int(doc.get("seed", 0)),
            plan_id=str(doc.get("plan_id", "")),
            total_items=int(doc.get("total_items", 0)),
            timestamp=str(doc.get("timestamp", utc_now_iso())),
            schema_version=str(doc["schema_version"]),
        )

    @staticmethod
    def from_json(blob: str) -> "IsaacCommand":
        try:
            return IsaacCommand.from_dict(json.loads(blob))
        except json.JSONDecodeError as exc:
            raise ContractError(f"isaac command is not valid JSON: {exc}") from exc


# --------------------------------------------------------------------------- #
# Feedback: Isaac -> WISEPACK
# --------------------------------------------------------------------------- #


@dataclass
class IsaacFeedback:
    """One physical-execution report from Isaac Sim.

    ``actual_pose`` and ``position_error_mm`` are populated on ITEM_COMPLETED and
    ITEM_FAILED and are MEASURED from the settled rigid body, never copied from
    the target. A dropped cylinder does not land exactly where the optimizer
    planned, and reporting the target back as though it were the outcome would
    turn a physics demonstration into an animation.
    """

    state: IsaacState
    run_id: str
    sequence_index: int = -1
    item_id: Optional[str] = None
    dimensions: Optional[Dimensions] = None
    source_pose: Optional[Pose] = None
    target_pose: Optional[Pose] = None
    actual_pose: Optional[Pose] = None
    position_error_mm: Optional[float] = None
    container_id: Optional[str] = None
    message: str = ""
    #: Free-form measurements: settle time, residual velocities, containment
    #: checks. Kept small — it rides in one std_msgs/String on DDS.
    detail: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.state = IsaacState(self.state)
        if self.state in ITEM_TERMINAL_STATES and not self.item_id:
            raise ContractError(
                f"{self.state.value} must name the item it terminated")

    @property
    def is_item_terminal(self) -> bool:
        return self.state in ITEM_TERMINAL_STATES

    @property
    def is_run_terminal(self) -> bool:
        return self.state in RUN_TERMINAL_STATES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "state": self.state.value,
            "sequence_index": self.sequence_index,
            "item_id": self.item_id,
            "container_id": self.container_id,
            "dimensions": self.dimensions.to_dict() if self.dimensions else None,
            "source_pose": self.source_pose.to_dict() if self.source_pose else None,
            "target_pose": self.target_pose.to_dict() if self.target_pose else None,
            "actual_pose": self.actual_pose.to_dict() if self.actual_pose else None,
            "position_error_mm": (None if self.position_error_mm is None
                                  else round(float(self.position_error_mm), 2)),
            "message": self.message,
            "detail": dict(self.detail),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), default=str)

    @staticmethod
    def from_dict(doc: Any) -> "IsaacFeedback":
        doc = _check_envelope(doc, "isaac feedback")
        try:
            state = IsaacState(doc["state"])
        except (KeyError, ValueError) as exc:
            raise ContractError(
                f"unknown or missing isaac state {doc.get('state')!r}") from exc
        dims = doc.get("dimensions")
        error = doc.get("position_error_mm")
        return IsaacFeedback(
            state=state,
            run_id=doc["run_id"],
            sequence_index=int(doc.get("sequence_index", -1)),
            item_id=doc.get("item_id"),
            dimensions=Dimensions.from_dict(dims) if dims else None,
            source_pose=_pose_or_none(doc.get("source_pose")),
            target_pose=_pose_or_none(doc.get("target_pose")),
            actual_pose=_pose_or_none(doc.get("actual_pose")),
            position_error_mm=None if error is None else float(error),
            container_id=doc.get("container_id"),
            message=str(doc.get("message", "")),
            detail=dict(doc.get("detail", {}) or {}),
            timestamp=str(doc.get("timestamp", utc_now_iso())),
            schema_version=str(doc["schema_version"]),
        )

    @staticmethod
    def from_json(blob: str) -> "IsaacFeedback":
        try:
            return IsaacFeedback.from_dict(json.loads(blob))
        except json.JSONDecodeError as exc:
            raise ContractError(f"isaac feedback is not valid JSON: {exc}") from exc


# --------------------------------------------------------------------------- #
# Run gating
# --------------------------------------------------------------------------- #


class RunGate:
    """Accepts messages for exactly one run, and each item at most once.

    Both ends need this and for different failure modes, so it lives here once:

      * ISAAC needs it because both operator topics and this command topic are
        TRANSIENT_LOCAL. A latched EXECUTE_ITEM is redelivered whenever Isaac
        re-subscribes — after a reconnect, or when a second reader joins — and
        acting on the redelivery means picking an item that is already in the
        container.
      * THE ORCHESTRATOR needs it because an Isaac process left over from a
        previous invocation would otherwise report ITEM_COMPLETED into a run it
        knows nothing about, and the engine would mark a placement executed that
        no robot touched.

    ``adopt`` is the only way the current run changes. Anything for another
    run_id is rejected, never silently adopted.
    """

    def __init__(self, run_id: str = "") -> None:
        self.run_id = run_id
        self._seen: set = set()

    def adopt(self, run_id: str) -> None:
        """Start gating a new run. Clears the per-item history."""
        if not run_id:
            raise ContractError("cannot adopt an empty run_id")
        self.run_id = run_id
        self._seen = set()

    def accepts_run(self, run_id: str) -> bool:
        return bool(self.run_id) and run_id == self.run_id

    def reject_reason(self, run_id: str, sequence_index: int,
                      attempt: int = 0) -> str:
        """"" when the message should be acted on, else why it must not be.

        The key is ``(sequence_index, attempt)``, not ``sequence_index`` alone.
        Both are needed and they guard different things: a REPLAY of attempt 0
        must be dropped, while a genuine RETRY of the same placement — attempt 1
        after a physical failure — must be executed. Keying on the index alone
        made those two indistinguishable and silently discarded every retry.
        """
        if not self.run_id:
            return "no run has been opened yet (RUN_BEGIN not received)"
        if run_id != self.run_id:
            return f"run_id {run_id!r} is not the current run {self.run_id!r}"
        if sequence_index >= 0 and (sequence_index, attempt) in self._seen:
            return (f"sequence_index {sequence_index} attempt {attempt} was "
                    f"already executed in run {self.run_id} — refusing a duplicate")
        return ""

    def mark_done(self, sequence_index: int, attempt: int = 0) -> None:
        if sequence_index >= 0:
            self._seen.add((sequence_index, attempt))

    @property
    def completed_count(self) -> int:
        """Distinct PLACEMENTS resolved, not attempts — retries are not progress."""
        return len({index for index, _ in self._seen})


__all__ = [
    "SCHEMA_VERSION", "schema_major", "ContractError", "utc_now_iso",
    "IsaacCommandType", "IsaacState", "ITEM_TERMINAL_STATES",
    "RUN_TERMINAL_STATES", "ITEM_PROGRESS_ORDER", "Dimensions", "Pose",
    "IsaacCommand", "IsaacFeedback", "RunGate",
]
