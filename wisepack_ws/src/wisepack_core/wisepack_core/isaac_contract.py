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
from typing import Any, Dict, List, Optional

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
    #: VERIFY OR BUILD the scene for one exact run and scenario revision, then
    #: acknowledge it. Sent on EVERY run, not only after a reset.
    #:
    #: The bug this closes: the initial scene used to be trusted because the
    #: launcher passed Isaac the same (preset, seed) the run was planned from.
    #: But "built from the right preset" is not "built for THIS run", and the
    #: trust was applied inside `open_run`, which only executes after approval —
    #: while approval itself waits for the scene. Isaac sat there with a correct
    #: four-cylinder scene on screen, the dashboard said the scene had not been
    #: rebuilt, and Approve stayed disabled with no way forward but a reset.
    #:
    #: Distinct from RESET_SCENE: this one is allowed to answer "already
    #: correct" after verifying, without destroying and recreating a scene that
    #: matches. RESET_SCENE always rebuilds.
    SYNC_SCENE = "SYNC_SCENE"
    #: REBUILD THE PHYSICAL SCENE for a new scenario, and do not report ready
    #: until it is genuinely rebuilt.
    #:
    #: THE SAFETY COMMAND. Generating a new software scenario does NOT reset a
    #: physical backend: the objects from the previous run are still lying in
    #: the container, while the new plan assumes every one of them is back at
    #: its source pose. Without this handshake the robot is sent to pick items
    #: that are not there, which is uncontrolled motion, not a failed pick.
    RESET_SCENE = "RESET_SCENE"


class IsaacState(str, Enum):
    """Isaac -> WISEPACK. Physical execution feedback.

    These are PHYSICAL states, and they are mapped onto the existing WISEPACK
    workflow ``Stage`` values by ``wisepack_core.execution.stage_for_isaac_state``
    rather than shown as a parallel state machine. The dashboard timeline, the
    audit trail and the FIWARE ``stage`` attribute keep their existing
    vocabulary; this enum adds physical resolution underneath it.
    """

    #: SIMULATOR-level readiness: the Isaac process, its ROS bridge and the
    #: physics application are up. It says NOTHING about which run the scene
    #: corresponds to, and on its own it never authorises physical execution —
    #: see SCENE_READY. Kept spelled ``READY`` on the wire for compatibility.
    READY = "READY"
    #: Explicit synonym of READY, for code and UI that must not blur the two
    #: readiness levels. Accepted inbound; the simulator may publish either.
    SIMULATOR_READY = "SIMULATOR_READY"
    #: Scene-reset lifecycle. SCENE_READY is the ONLY thing that re-authorises
    #: physical execution after a new scenario is generated, and it carries the
    #: scenario revision it rebuilt for, so it cannot satisfy a later one.
    RESET_REQUESTED = "RESET_REQUESTED"
    RESETTING = "RESETTING"
    SCENE_READY = "SCENE_READY"
    RESET_FAILED = "RESET_FAILED"
    #: THE ROBOT MODEL DID NOT VALIDATE. The configured asset is missing, a
    #: required prim is absent, the configured joints do not match the loaded
    #: articulation, the end effector could not be resolved, or the selected
    #: robot does not support the active preset.
    #:
    #: A separate state from RUN_FAILED because it is not a failure OF a run —
    #: it happens before any run may open, it is a configuration fault rather
    #: than a physical one, and the response is different: the backend goes
    #: DEGRADED and approval is disabled rather than an item being retried. A
    #: partially loaded robot must never reach execution.
    ROBOT_MODEL_INVALID = "ROBOT_MODEL_INVALID"
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

#: The scene-reset lifecycle. These concern the SCENE, not an item and not a
#: run's execution, so they are handled on their own path.
RESET_STATES = frozenset({IsaacState.RESET_REQUESTED, IsaacState.RESETTING,
                          IsaacState.SCENE_READY, IsaacState.RESET_FAILED})

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
    #: Which SCENARIO the physical scene must correspond to. Incremented by the
    #: orchestrator whenever a new scenario is generated, so a SCENE_READY for
    #: revision 1 can never authorise execution of revision 2.
    scenario_revision: int = 0
    #: WHICH ROBOT this run is for. Carried on every command, not only on the
    #: scene handshake: the simulator refuses a command addressed to a robot it
    #: is not running rather than executing it with whatever arm it happens to
    #: have loaded. Empty means "the sender did not say", which an older
    #: orchestrator may legitimately do and which is therefore not treated as a
    #: mismatch — see the receiver in simulators/isaac/wisepack_isaac.py.
    robot_id: str = ""
    #: WHICH SIMULATOR INSTANCE. Incremented by the host supervisor every time
    #: it starts an Isaac process.
    #:
    #: The robot id alone cannot tell two instances apart. A robot switch stops
    #: one simulator and starts another, and for a few seconds both may be on
    #: the DDS domain — the old one still publishing as it dies, the new one
    #: coming up. Worse, switching A -> B -> A returns to the SAME robot id
    #: while being a different process with a different scene. A generation
    #: makes "this is the instance I asked for" checkable instead of inferred
    #: from timing. 0 means unstamped, and is never treated as a mismatch.
    simulator_generation: int = 0
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
            "scenario_revision": int(self.scenario_revision),
            "robot_id": self.robot_id,
            "simulator_generation": int(self.simulator_generation),
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
            scenario_revision=int(doc.get("scenario_revision", 0)),
            robot_id=str(doc.get("robot_id", "") or ""),
            simulator_generation=int(doc.get("simulator_generation", 0) or 0),
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
class SceneAcknowledgement:
    """WHAT Isaac actually built, and for WHICH run — the SCENE_READY payload.

    "The simulator is up" and "the world in front of the robot is the world this
    plan was written against" are different claims, and only the second one may
    authorise a pick. Everything here exists so the orchestrator can check the
    second claim instead of inferring it:

      * ``run_id`` / ``scenario_id`` / ``scenario_revision`` — WHICH run. An
        acknowledgement from the previous run is not evidence about this one, and
        without these the only way to tell them apart is timing.
      * ``preset`` / ``seed`` — the generator inputs. Right preset, wrong seed is
        a different set of objects with the same names.
      * ``scene_fingerprint`` — a deterministic digest over preset, seed, object
        ids, dimensions, initial source poses and the container specification.
        The ids and the count can match while the geometry does not; this is what
        catches that.
      * ``object_ids`` / ``object_count`` — what is actually in the scene, so a
        mismatch names the missing objects rather than just failing.
      * ``robot_home_verified`` / ``container_empty_verified`` — the two physical
        preconditions for a first pick, MEASURED by the simulator rather than
        assumed by the orchestrator.
      * ``robot_id`` / ``robot_profile_revision`` — WHICH ARM is standing in
        that world, and under which configuration. An acknowledgement from a
        Panda cannot authorise an xArm run and vice versa: the two have
        different envelopes, different bin positions and different tool frames,
        so a plan validated for one is not a plan the other may execute. The
        revision is carried as well as the id because "the right robot" and
        "the right robot, configured the same way" are different claims — an
        edited home pose or tool-centre-point is a different machine as far as
        a validated plan is concerned.

    Deliberately NOT inferable from: the process being up, a camera stream
    existing, four objects being visible, or an older SCENE_READY.
    """

    run_id: str = ""
    scenario_id: str = ""
    scenario_revision: int = 0
    preset: str = ""
    seed: int = 0
    robot_id: str = ""
    robot_profile_revision: str = ""
    #: The simulator instance that built this scene. A switch A -> B -> A comes
    #: back to the same robot id; only the generation separates the scene the
    #: first instance built from the one the third did.
    simulator_generation: int = 0
    scene_fingerprint: str = ""
    object_ids: List[str] = field(default_factory=list)
    object_count: int = 0
    robot_home_verified: bool = False
    container_empty_verified: bool = False
    #: True when the existing scene was verified as already correct rather than
    #: destroyed and rebuilt. A correct scene is not rebuilt for the sake of it.
    verified_without_rebuild: bool = False
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "scenario_revision": int(self.scenario_revision),
            "preset": self.preset,
            "seed": int(self.seed),
            "robot_id": self.robot_id,
            "robot_profile_revision": self.robot_profile_revision,
            "simulator_generation": int(self.simulator_generation),
            "scene_fingerprint": self.scene_fingerprint,
            "object_ids": list(self.object_ids),
            "object_count": int(self.object_count),
            "robot_home_verified": bool(self.robot_home_verified),
            "container_empty_verified": bool(self.container_empty_verified),
            "verified_without_rebuild": bool(self.verified_without_rebuild),
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(doc: Any) -> Optional["SceneAcknowledgement"]:
        if not isinstance(doc, dict):
            return None
        return SceneAcknowledgement(
            run_id=str(doc.get("run_id", "")),
            scenario_id=str(doc.get("scenario_id", "")),
            scenario_revision=int(doc.get("scenario_revision", 0) or 0),
            preset=str(doc.get("preset", "")),
            seed=int(doc.get("seed", 0) or 0),
            robot_id=str(doc.get("robot_id", "") or ""),
            robot_profile_revision=str(doc.get("robot_profile_revision", "") or ""),
            simulator_generation=int(doc.get("simulator_generation", 0) or 0),
            scene_fingerprint=str(doc.get("scene_fingerprint", "")),
            object_ids=[str(i) for i in (doc.get("object_ids") or [])],
            object_count=int(doc.get("object_count", 0) or 0),
            robot_home_verified=bool(doc.get("robot_home_verified", False)),
            container_empty_verified=bool(doc.get("container_empty_verified", False)),
            verified_without_rebuild=bool(doc.get("verified_without_rebuild", False)),
            timestamp=str(doc.get("timestamp", "")),
        )

    def mismatches(self, *, run_id: str, scenario_id: str, revision: int,
                   preset: str, seed: int, fingerprint: str,
                   object_count: int, robot_id: str = "",
                   robot_profile_revision: str = "",
                   simulator_generation: int = 0) -> List[str]:
        """Every reason this acknowledgement does not describe the given run.

        Returns sentences an operator can act on, not booleans. "scene not
        ready" is not actionable; "acknowledged fingerprint 4f2a… but this run
        expects 9b71…" is.
        """
        out: List[str] = []
        if run_id and self.run_id and self.run_id != run_id:
            out.append(f"acknowledged run {self.run_id} but this run is {run_id}")
        if scenario_id and self.scenario_id and self.scenario_id != scenario_id:
            out.append(f"acknowledged scenario {self.scenario_id} but this run "
                       f"is {scenario_id}")
        if self.scenario_revision != revision:
            out.append(f"acknowledged scenario revision {self.scenario_revision} "
                       f"but this run is at {revision}")
        if preset and self.preset and self.preset != preset:
            out.append(f"acknowledged preset {self.preset} but this run planned "
                       f"{preset}")
        if self.seed and seed and self.seed != seed:
            out.append(f"acknowledged seed {self.seed} but this run planned {seed}")
        # THE ROBOT IS NOT NEGOTIABLE. A missing id is tolerated — an older
        # simulator does not send one — but a DIFFERENT id is refused outright:
        # a Panda's acknowledgement describes a world with the bin 80 mm further
        # out and a different reach envelope, and authorising an xArm pick from
        # it would send the arm at coordinates nothing was validated against.
        if robot_id and self.robot_id and self.robot_id != robot_id:
            out.append(f"acknowledged robot {self.robot_id} but this run "
                       f"selected {robot_id}")
        if (robot_profile_revision and self.robot_profile_revision
                and self.robot_profile_revision != robot_profile_revision):
            out.append(
                f"acknowledged robot profile revision "
                f"{self.robot_profile_revision} but this run is configured for "
                f"{robot_profile_revision} — the same robot, described "
                "differently, is a different machine to a validated plan")
        # THE INSTANCE, not only the robot. A switch that returns to a robot
        # already used comes back to the same id, and the scene an earlier
        # instance built is not the scene this run planned against.
        if (simulator_generation and self.simulator_generation
                and self.simulator_generation != simulator_generation):
            out.append(
                f"acknowledged simulator generation "
                f"{self.simulator_generation} but this run is waiting for "
                f"generation {simulator_generation} — that scene was built by a "
                "previous Isaac process")
        if fingerprint and self.scene_fingerprint \
                and self.scene_fingerprint != fingerprint:
            out.append(f"acknowledged scene fingerprint "
                       f"{self.scene_fingerprint[:12]} but this run expects "
                       f"{fingerprint[:12]}")
        if object_count and self.object_count != object_count:
            out.append(f"acknowledged {self.object_count} object(s) but this run "
                       f"has {object_count}")
        if not self.robot_home_verified:
            out.append("the simulator did not verify that the robot is home")
        if not self.container_empty_verified:
            out.append("the simulator did not verify that the container is empty")
        return out


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
    #: For SCENE_READY / RESET_* this is the scenario revision the physical
    #: scene now corresponds to. The orchestrator refuses to authorise
    #: execution until this matches the ACTIVE revision exactly.
    scenario_revision: int = 0
    #: WHICH ROBOT produced this report. On EVERY state, not only the scene
    #: handshake: a simulator left over from a previous run — restarted for a
    #: different arm, or simply never shut down — publishes onto the same topic,
    #: and "this came from the robot this run selected" must be checkable
    #: without inferring it from timing. Empty means the sender did not say.
    robot_id: str = ""
    #: WHICH SIMULATOR INSTANCE. Incremented by the host supervisor every time
    #: it starts an Isaac process.
    #:
    #: The robot id alone cannot tell two instances apart. A robot switch stops
    #: one simulator and starts another, and for a few seconds both may be on
    #: the DDS domain — the old one still publishing as it dies, the new one
    #: coming up. Worse, switching A -> B -> A returns to the SAME robot id
    #: while being a different process with a different scene. A generation
    #: makes "this is the instance I asked for" checkable instead of inferred
    #: from timing. 0 means unstamped, and is never treated as a mismatch.
    simulator_generation: int = 0
    #: SCENE ACKNOWLEDGEMENT, present on SCENE_READY. Everything needed to
    #: decide whether the world Isaac built is the world this run planned
    #: against — see SceneAcknowledgement for why each field is here.
    scene: Optional["SceneAcknowledgement"] = None
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
            "scenario_revision": int(self.scenario_revision),
            "robot_id": self.robot_id,
            "simulator_generation": int(self.simulator_generation),
            "dimensions": self.dimensions.to_dict() if self.dimensions else None,
            "source_pose": self.source_pose.to_dict() if self.source_pose else None,
            "target_pose": self.target_pose.to_dict() if self.target_pose else None,
            "actual_pose": self.actual_pose.to_dict() if self.actual_pose else None,
            "position_error_mm": (None if self.position_error_mm is None
                                  else round(float(self.position_error_mm), 2)),
            "message": self.message,
            "scene": self.scene.to_dict() if self.scene else None,
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
            scenario_revision=int(doc.get("scenario_revision", 0)),
            robot_id=str(doc.get("robot_id", "") or ""),
            simulator_generation=int(doc.get("simulator_generation", 0) or 0),
            scene=SceneAcknowledgement.from_dict(doc.get("scene")),
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
