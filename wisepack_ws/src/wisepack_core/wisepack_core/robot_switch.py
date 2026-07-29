"""THE ROBOT-SWITCH PROTOCOL — one vocabulary, both ends.

A robot switch is a PROCESS operation, not a scene operation. The adapter and the
USD model are chosen when Isaac starts and cannot be changed afterwards, so
"switch to the other arm" means stop one simulator and start another. Only the
host launcher can do that; the orchestrator runs in a container and must not be
able to.

This module is the narrow seam between them: the phase vocabulary, the request
document and the status document. Both ends import it, so neither can invent a
phase the other does not understand.

WHAT THE CONTAINER IS ALLOWED TO SAY
------------------------------------
One verb, ``switch_robot``, and a set of identifiers the supervisor COMPARES
rather than executes. There is no command string, no path, no signal and no
process id anywhere in the request. The requested robot is validated against the
tracked registry on the host before anything is stopped, so a request naming an
unknown arm costs a refusal rather than a dead simulator.

WHY A FILE AND NOT A SOCKET
---------------------------
The boundary is already a bind mount, so a file needs no new listener, no port
and no daemon lifecycle. It is also inspectable after the fact: a request that
produced a bad restart is still on disk, with its id, when someone comes to ask
why. Requests are written atomically (temp + rename) and consumed exactly once.

Pure stdlib, like the rest of the contract: the identical file is imported by the
orchestrator under Vulcanexus inside Docker and by the supervisor under the host
interpreter.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

#: The complete verb list. The supervisor refuses anything else and records it.
SWITCH_OP = "switch_robot"
ALLOWED_OPS = (SWITCH_OP,)

#: Where the control directory is, when the launcher created one.
CONTROL_DIR_ENV = "WISEPACK_CONTROL_DIR"
STATUS_FILENAME = "supervisor-status.json"

# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #
#
# Named individually rather than collapsed into "switching", because a switch
# that fails must say WHERE. "The robot switch failed" is not actionable;
# "the previous simulator did not fully stop" and "the new simulator did not
# report READY" need different responses from an operator.

PHASE_IDLE = "idle"
PHASE_REQUESTED = "ROBOT_SWITCH_REQUESTED"
PHASE_STOPPING = "STOPPING_OLD_ROBOT"
PHASE_STARTING = "STARTING_NEW_ROBOT"
PHASE_WAITING_READY = "WAITING_SIMULATOR_READY"
PHASE_READY = "SIMULATOR_READY"
PHASE_FAILED = "ROBOT_SWITCH_FAILED"

#: In the order they occur, for a progress display.
SWITCH_PHASES = (PHASE_REQUESTED, PHASE_STOPPING, PHASE_STARTING,
                 PHASE_WAITING_READY, PHASE_READY)

#: Phases during which nothing may be approved and nothing may be executed.
IN_FLIGHT_PHASES = frozenset({PHASE_REQUESTED, PHASE_STOPPING, PHASE_STARTING,
                              PHASE_WAITING_READY})

#: Operator-facing wording for each phase. One sentence, present tense, naming
#: the robots rather than the mechanism.
PHASE_LABELS = {
    PHASE_IDLE: "Idle",
    PHASE_REQUESTED: "Robot switch requested",
    PHASE_STOPPING: "Stopping the {previous} simulator",
    PHASE_STARTING: "Starting the {requested} simulator",
    PHASE_WAITING_READY: "Loading the {requested} model",
    PHASE_READY: "{requested} model loaded",
    PHASE_FAILED: "Robot switch FAILED",
}


def describe_phase(phase: str, previous: str = "", requested: str = "") -> str:
    """One operator-readable line for a phase."""
    template = PHASE_LABELS.get(phase, phase)
    return template.format(previous=previous or "the previous robot",
                           requested=requested or "the selected robot")


def control_dir(explicit: Optional[str] = None) -> str:
    """The launcher-owned control directory, or "" when there is none.

    Empty is a normal answer: a logical run has no simulator to switch, and a
    launcher that predates this protocol simply does not create the directory.
    Callers treat "" as "web-initiated switching is unavailable" and say so,
    rather than pretending a switch happened.
    """
    return explicit or os.environ.get(CONTROL_DIR_ENV, "") or ""


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


@dataclass
class SwitchRequest:
    """One allowlisted request from the orchestrator to the host supervisor."""

    requested_robot_id: str
    requested_profile_revision: str = ""
    run_id: str = ""
    scenario_revision: int = 0
    op: str = SWITCH_OP
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "request_id": self.request_id,
            "requested_robot_id": self.requested_robot_id,
            "requested_profile_revision": self.requested_profile_revision,
            "run_id": self.run_id,
            "scenario_revision": int(self.scenario_revision),
            "timestamp": self.timestamp,
        }


@dataclass
class SupervisorStatus:
    """What the host supervisor says is ACTUALLY running.

    ``robot_id`` and ``requested_robot_id`` are kept apart on purpose: they
    differ exactly while a switch is in flight or has failed, and that is the
    interval in which the dashboard must not claim the new robot is active.
    """

    present: bool = False
    robot_id: str = ""
    requested_robot_id: str = ""
    simulator_generation: int = 0
    isaac_running: bool = False
    simulator_ready: bool = False
    phase: str = PHASE_IDLE
    request_id: str = ""
    last_error: str = ""
    generated_at: str = ""
    age_s: Optional[float] = None

    @property
    def switch_in_flight(self) -> bool:
        return self.phase in IN_FLIGHT_PHASES

    @property
    def failed(self) -> bool:
        return self.phase == PHASE_FAILED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "present": self.present,
            "robot_id": self.robot_id,
            "requested_robot_id": self.requested_robot_id,
            "simulator_generation": int(self.simulator_generation),
            "isaac_running": self.isaac_running,
            "simulator_ready": self.simulator_ready,
            "phase": self.phase,
            "phase_label": describe_phase(self.phase, requested=self.requested_robot_id),
            "switch_in_flight": self.switch_in_flight,
            "failed": self.failed,
            "request_id": self.request_id,
            "last_error": self.last_error,
            "generated_at": self.generated_at,
            "age_s": self.age_s,
        }

    @staticmethod
    def from_dict(doc: Any, *, age_s: Optional[float] = None) -> "SupervisorStatus":
        if not isinstance(doc, dict):
            return SupervisorStatus()
        return SupervisorStatus(
            present=True,
            robot_id=str(doc.get("robot_id", "") or ""),
            requested_robot_id=str(doc.get("requested_robot_id", "") or ""),
            simulator_generation=int(doc.get("simulator_generation", 0) or 0),
            isaac_running=bool(doc.get("isaac_running", False)),
            simulator_ready=bool(doc.get("simulator_ready", False)),
            phase=str(doc.get("phase", PHASE_IDLE) or PHASE_IDLE),
            request_id=str(doc.get("request_id", "") or ""),
            last_error=str(doc.get("last_error", "") or ""),
            generated_at=str(doc.get("generated_at", "") or ""),
            age_s=age_s)


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class RobotSwitchClient:
    """Writes requests and reads status. Never starts or stops anything."""

    def __init__(self, directory: Optional[str] = None) -> None:
        self.directory = control_dir(directory)

    @property
    def available(self) -> bool:
        return bool(self.directory) and os.path.isdir(self.directory)

    def unavailable_reason(self) -> str:
        """"" when a switch can be requested, else why it cannot."""
        if not self.directory:
            return ("this stack was started without a host supervisor, so the "
                    "robot cannot be changed from the web application — restart "
                    "the launcher with the robot you want")
        if not os.path.isdir(self.directory):
            return (f"the control directory named by {CONTROL_DIR_ENV} does not "
                    "exist; restart the launcher")
        if not os.access(self.directory, os.W_OK):
            return ("the control directory is not writable, so no request can "
                    "be delivered to the host supervisor")
        return ""

    def status(self) -> SupervisorStatus:
        """What the supervisor last reported. Never raises."""
        if not self.directory:
            return SupervisorStatus()
        path = os.path.join(self.directory, STATUS_FILENAME)
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
            age = round(time.time() - os.path.getmtime(path), 1)
        except (OSError, ValueError):
            return SupervisorStatus()
        return SupervisorStatus.from_dict(doc, age_s=age)

    def request_switch(self, request: SwitchRequest) -> str:
        """Deliver one request atomically. Returns its id.

        Written to a temporary name and renamed into place, so the supervisor
        can never read a half-written document and act on it.
        """
        reason = self.unavailable_reason()
        if reason:
            raise RuntimeError(reason)
        payload = json.dumps(request.to_dict(), indent=2) + "\n"
        final = os.path.join(self.directory,
                             f"request-{request.timestamp}-{request.request_id}.json")
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".request-",
                                   suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            # WORLD-READABLE, and that is required rather than lax.
            #
            # The writer is the orchestrator inside the container, running as
            # root; the reader is the supervisor on the host, running as the
            # invoking user. mkstemp creates 0600, so the supervisor could not
            # read its own control directory's requests and rejected every one
            # of them with "Permission denied" — a switch that silently never
            # happened. The document contains four identifiers and a timestamp;
            # there is nothing in it to protect.
            os.chmod(tmp, 0o644)
            os.replace(tmp, final)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return request.request_id


__all__ = [
    "SWITCH_OP", "ALLOWED_OPS", "CONTROL_DIR_ENV", "STATUS_FILENAME",
    "PHASE_IDLE", "PHASE_REQUESTED", "PHASE_STOPPING", "PHASE_STARTING",
    "PHASE_WAITING_READY", "PHASE_READY", "PHASE_FAILED", "SWITCH_PHASES",
    "IN_FLIGHT_PHASES", "PHASE_LABELS", "describe_phase", "control_dir",
    "SwitchRequest", "SupervisorStatus", "RobotSwitchClient",
]
