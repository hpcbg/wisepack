#!/usr/bin/env python3
"""THE HOST-SIDE ISAAC SUPERVISOR — the only thing that may restart a simulator.

WHY THIS EXISTS
---------------
The robot is chosen when the Isaac process starts. The adapter is built from the
profile, the USD model is referenced into the stage, and neither can be changed
afterwards — so "switch robot" is a PROCESS operation, not a scene operation.

Before this, a cross-robot "Reset run & generate" re-bound the orchestrator's
view of the robot and then sent RESET_SCENE to the still-running old simulator.
That simulator dutifully rebuilt the objects with the arm it had, and stamped the
acknowledgement with its OWN robot id. The gate correctly refused it, so nothing
unsafe happened — but the operator watched the workcell reset while the wrong
robot stayed on the stage, which is a worse kind of wrong than an error message.

So the launcher no longer starts Isaac directly. It starts THIS, which owns the
Isaac process group and is the one component able to stop one and start another.

WHAT THE WEB CONTAINER IS GIVEN, AND WHAT IT IS NOT
---------------------------------------------------
It is given ONE directory, and in it may write ONE kind of file: a request naming
an allowlisted operation. It is not given the Docker socket, a host shell, or any
ability to name a command, a path or a signal. The only verb is `switch_robot`,
its robot argument is validated against the tracked registry BEFORE anything is
stopped, and every other field is an identifier this supervisor compares rather
than executes.

A file protocol rather than a socket because the boundary is already a bind mount
and a file is inspectable after the fact: a request that produced a bad restart is
still sitting there, with its id, when someone comes to ask why.

GENERATIONS
-----------
Every started process gets an incrementing generation, passed to Isaac and
stamped on everything it publishes. The robot id alone cannot distinguish two
instances — switching A -> B -> A returns to the same id — and for a few seconds
during a switch both the dying and the starting simulator are on the DDS domain.

NOTHING HERE RESTARTS ANYTHING BY ITSELF. A simulator that dies stays dead and is
reported; an operator decides what happens next. A supervisor that quietly
restarted a failing robot would replace one clear diagnosis with a scrolling one.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _pkg in ("wisepack_core",):
    _path = os.path.join(REPO, "wisepack_ws", "src", _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)

LOG = "[isaac-supervisor]"

# ONE VOCABULARY, BOTH ENDS. The verbs and the phase names come from the shared
# protocol module rather than being restated here — two copies of a phase list
# agree only until one of them is edited, and the dashboard would then be
# rendering a phase this supervisor never reports.
from wisepack_core.robot_switch import (                          # noqa: E402
    ALLOWED_OPS, PHASE_FAILED, PHASE_IDLE, PHASE_READY, PHASE_REQUESTED,
    PHASE_STARTING, PHASE_STOPPING, PHASE_WAITING_READY, STATUS_FILENAME,
)

#: Bounded, every one of them. A switch that hangs must fail with a named phase
#: rather than leave the operator in front of a dashboard that says "switching".
STOP_TIMEOUT_S = float(os.environ.get("WISEPACK_SWITCH_STOP_TIMEOUT", "60"))
START_TIMEOUT_S = float(os.environ.get("WISEPACK_SWITCH_START_TIMEOUT", "300"))


class Supervisor:
    def __init__(self, control_dir: str, robot_id: str, log_dir: str,
                 poll_s: float = 1.0) -> None:
        self.control_dir = control_dir
        self.status_path = os.path.join(control_dir, STATUS_FILENAME)
        self.log_dir = log_dir
        self.poll_s = poll_s
        self.robot_id = robot_id
        self.generation = 0
        self.pgid = 0
        self.isaac_log = ""
        self.phase = PHASE_IDLE
        self.last_error = ""
        self.ready = False
        self.request_id = ""
        self.requested_robot = ""
        self._stopping = False
        os.makedirs(control_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #

    def write_status(self) -> None:
        """Atomic replace, so a reader never sees half a document."""
        doc = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "supervisor_pid": os.getpid(),
            "control_dir": self.control_dir,
            "allowed_operations": list(ALLOWED_OPS),
            # WHAT IS ACTUALLY RUNNING. Never what was asked for.
            "robot_id": self.robot_id,
            "simulator_generation": self.generation,
            "isaac_pgid": self.pgid,
            "isaac_running": self.isaac_alive(),
            "simulator_ready": self.ready,
            "phase": self.phase,
            # WHAT WAS ASKED FOR, kept apart. They differ exactly while a switch
            # is in flight or has failed, which is when the difference matters.
            "request_id": self.request_id,
            "requested_robot_id": self.requested_robot,
            "last_error": self.last_error,
        }
        try:
            fd, tmp = tempfile.mkstemp(dir=self.control_dir, prefix=".status-",
                                       suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=2)
                fh.write("\n")
            os.replace(tmp, self.status_path)
            os.chmod(self.status_path, 0o644)
        except OSError as exc:
            # Reporting is best-effort; a simulator that is running must not be
            # taken down because its status file could not be written.
            print(f"{LOG} WARNING: cannot write status: {exc}", file=sys.stderr)

    def set_phase(self, phase: str, error: str = "") -> None:
        self.phase = phase
        if error:
            self.last_error = error
        print(f"{LOG} phase: {phase}" + (f" — {error}" if error else ""),
              flush=True)
        self.write_status()

    # ------------------------------------------------------------------ #
    # The Isaac process group
    # ------------------------------------------------------------------ #

    def isaac_alive(self) -> bool:
        if not self.pgid:
            return False
        try:
            os.killpg(self.pgid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def group_size(self) -> int:
        """LIVE processes remaining in the group. Kit outlives its leader.

        ZOMBIES DO NOT COUNT, and that is not a detail. A `<defunct>` child
        still carries its process-group id, so a group whose every member has
        exited still measured as size 1 until the parent reaped it — and this
        loop waited for zero. Measured: a robot switch sat in STOPPING_OLD_ROBOT
        with one defunct `python.sh` in the group, and every switch would have
        ended in "the previous simulator did not fully stop".

        Reaping (below) removes the direct child; this filter covers any
        grandchild whose own parent went first.
        """
        if not self.pgid:
            return 0
        self._reap()
        try:
            out = subprocess.run(["ps", "-eo", "pgid=,pid=,stat="],
                                 capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return 0
        live = 0
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 3 or parts[0] != str(self.pgid):
                continue
            if parts[2].startswith("Z"):
                continue                      # defunct: exited, not yet reaped
            live += 1
        return live

    def _reap(self) -> None:
        """Reap the direct child so it stops holding a slot in its own group."""
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        try:
            proc.poll()
        except OSError:
            pass

    def start_isaac(self, robot_id: str) -> bool:
        """Start one Isaac process group for ``robot_id``. Returns False on failure."""
        self.generation += 1
        self.robot_id = robot_id
        self.ready = False
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.isaac_log = os.path.join(
            self.log_dir, f"isaac-gen{self.generation}-{robot_id}-{stamp}.log")

        env = dict(os.environ)
        env["WISEPACK_ISAAC_ROBOT"] = robot_id
        # THE GENERATION TRAVELS WITH THE PROCESS. Isaac stamps it on every
        # report, which is what lets the orchestrator tell this instance's
        # SCENE_READY from the one the previous instance published as it died.
        env["WISEPACK_ISAAC_GENERATION"] = str(self.generation)

        print(f"{LOG} starting Isaac generation {self.generation} with "
              f"{robot_id} (log: {self.isaac_log})", flush=True)
        try:
            with open(self.isaac_log, "w", encoding="utf-8") as out:
                proc = subprocess.Popen(
                    [os.path.join(REPO, "scripts", "run_wisepack_isaac.sh")],
                    stdout=out, stderr=subprocess.STDOUT, env=env,
                    # Its OWN session, so the whole tree can be signalled as a
                    # group and nothing outside it ever is.
                    start_new_session=True, cwd=REPO)
        except OSError as exc:
            self.pgid = 0
            self.set_phase(PHASE_FAILED, f"could not start Isaac: {exc}")
            return False
        self.pgid = os.getpgid(proc.pid)
        self.proc = proc
        self.phase = PHASE_WAITING_READY
        self.write_status()
        return True

    def stop_isaac(self, timeout_s: float = STOP_TIMEOUT_S) -> bool:
        """Stop the owned group and WAIT until every member is gone.

        Waits on GROUP MEMBERSHIP, not on the leader. Kit spawns children that
        outlive their parent during shutdown, so the leader can be gone while
        the simulator still holds the GPU — and starting the replacement then
        races the corpse for the device and the DDS topic.
        """
        if not self.pgid:
            return True
        print(f"{LOG} stopping Isaac process group {self.pgid}", flush=True)
        try:
            os.killpg(self.pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.group_size() == 0:
                break
            time.sleep(0.5)
        if self.group_size() > 0:
            print(f"{LOG} group {self.pgid} did not exit on TERM — sending KILL",
                  file=sys.stderr, flush=True)
            try:
                os.killpg(self.pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            hard = time.time() + 15
            while time.time() < hard and self.group_size() > 0:
                time.sleep(0.5)
        # Reap the direct child explicitly before the final count, so the
        # verdict is about processes that are actually still running.
        proc = getattr(self, "proc", None)
        if proc is not None:
            try:
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass
        remaining = self.group_size()
        if remaining:
            # VERIFIED, not assumed. Reporting "stopped" while the GPU is still
            # held is how a switch ends with two simulators on one domain.
            print(f"{LOG} ERROR: {remaining} process(es) remain in group "
                  f"{self.pgid}", file=sys.stderr, flush=True)
            return False
        self.pgid = 0
        self.ready = False
        return True

    def note_ready(self) -> bool:
        """Has the CURRENT generation's log announced READY?"""
        if not self.isaac_log or not os.path.isfile(self.isaac_log):
            return False
        try:
            with open(self.isaac_log, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            return False
        if "ROBOT_MODEL_INVALID" in text:
            self.last_error = "the robot model did not validate"
        return "[isaac-app] READY" in text

    def log_tail(self, lines: int = 20) -> str:
        try:
            with open(self.isaac_log, encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-lines:])
        except OSError:
            return ""

    # ------------------------------------------------------------------ #
    # Requests
    # ------------------------------------------------------------------ #

    def _requests(self):
        try:
            names = sorted(n for n in os.listdir(self.control_dir)
                           if n.startswith("request-") and n.endswith(".json"))
        except OSError:
            return []
        out = []
        for name in names:
            path = os.path.join(self.control_dir, name)
            try:
                with open(path, encoding="utf-8") as fh:
                    doc = json.load(fh)
            except (OSError, ValueError) as exc:
                self._reject(path, {}, f"unreadable request: {exc}")
                continue
            out.append((path, doc))
        return out

    def _reject(self, path: str, doc: dict, reason: str) -> None:
        print(f"{LOG} REJECTED request {doc.get('request_id', '?')}: {reason}",
              file=sys.stderr, flush=True)
        self.request_id = str(doc.get("request_id", ""))
        self.last_error = reason
        self.write_status()
        self._consume(path)

    def _consume(self, path: str) -> None:
        """Move the request out of the way so it is handled exactly once.

        Renamed rather than deleted: a request that produced a bad restart is
        still there, with its id, when someone comes to ask why.
        """
        try:
            os.replace(path, path + ".done")
        except OSError:
            try:
                os.unlink(path)
            except OSError:
                pass

    def handle(self, path: str, doc: dict) -> None:
        op = str(doc.get("op", ""))
        if op not in ALLOWED_OPS:
            self._reject(path, doc, f"operation {op!r} is not allowed; "
                                    f"allowed: {list(ALLOWED_OPS)}")
            return
        wanted = str(doc.get("requested_robot_id", "")).strip().lower()
        self.request_id = str(doc.get("request_id", ""))
        self.requested_robot = wanted

        # VALIDATED AGAINST THE TRACKED REGISTRY BEFORE ANYTHING IS STOPPED.
        # A switch to a robot that does not exist must cost nothing: the running
        # simulator keeps running and the request is refused.
        try:
            from wisepack_core.robots import RobotConfigError, load_registry
            profile = load_registry(reload=True).get(wanted)
        except Exception as exc:                            # noqa: BLE001
            self._reject(path, doc, f"{wanted!r} is not a usable robot: {exc}")
            self.set_phase(PHASE_FAILED, f"{wanted!r} is not a usable robot")
            return

        requested_rev = str(doc.get("requested_profile_revision", ""))
        if requested_rev and requested_rev != profile.revision:
            self._reject(path, doc,
                         f"requested profile revision {requested_rev} but the "
                         f"registry has {profile.revision}")
            self.set_phase(PHASE_FAILED, "robot profile revision mismatch")
            return

        if wanted == self.robot_id and self.isaac_alive() and self.ready:
            # Not an error, and not a restart either. A same-robot request is a
            # scene reset the orchestrator should have handled in-process.
            print(f"{LOG} {wanted} is already running — no restart needed",
                  flush=True)
            self._consume(path)
            self.set_phase(PHASE_READY)
            return

        print(f"{LOG} switch_robot {self.robot_id or '-'} -> {wanted} "
              f"(request {self.request_id}, run {doc.get('run_id', '?')})",
              flush=True)
        self._consume(path)
        self.switch(profile.robot_id)

    def switch(self, robot_id: str) -> None:
        """The bounded transaction. Every phase reported, every phase timed."""
        self.set_phase(PHASE_REQUESTED)

        self.set_phase(PHASE_STOPPING)
        if not self.stop_isaac():
            self.set_phase(PHASE_FAILED,
                           "the previous simulator did not fully stop; refusing "
                           "to start a second one on the same GPU and DDS domain")
            return

        self.set_phase(PHASE_STARTING)
        if not self.start_isaac(robot_id):
            return

        self.set_phase(PHASE_WAITING_READY)
        deadline = time.time() + START_TIMEOUT_S
        while time.time() < deadline:
            if not self.isaac_alive():
                self.set_phase(
                    PHASE_FAILED,
                    f"the {robot_id} simulator exited during startup"
                    + (f": {self.last_error}" if self.last_error else ""))
                print(self.log_tail(), file=sys.stderr, flush=True)
                return
            if self.note_ready():
                self.ready = True
                self.set_phase(PHASE_READY)
                print(f"{LOG} {robot_id} ready as generation {self.generation}",
                      flush=True)
                return
            time.sleep(1.0)
        self.set_phase(PHASE_FAILED,
                       f"the {robot_id} simulator did not report READY within "
                       f"{START_TIMEOUT_S:.0f}s")

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self) -> int:
        def _term(_signum, _frame):
            self._stopping = True
        signal.signal(signal.SIGTERM, _term)
        signal.signal(signal.SIGINT, _term)

        if not self.start_isaac(self.robot_id):
            return 5
        self.set_phase(PHASE_WAITING_READY)

        while not self._stopping:
            if self.phase == PHASE_WAITING_READY:
                if self.note_ready():
                    self.ready = True
                    self.set_phase(PHASE_READY)
                elif not self.isaac_alive():
                    self.set_phase(
                        PHASE_FAILED,
                        f"the {self.robot_id} simulator exited during startup"
                        + (f": {self.last_error}" if self.last_error else ""))
                    print(self.log_tail(), file=sys.stderr, flush=True)
            elif self.phase == PHASE_READY and not self.isaac_alive():
                # DEAD, NOT RESTARTED. See the module docstring.
                self.ready = False
                self.set_phase(PHASE_FAILED,
                               f"the {self.robot_id} simulator exited")
                print(self.log_tail(), file=sys.stderr, flush=True)

            for path, doc in self._requests():
                self.handle(path, doc)
                if self._stopping:
                    break

            self.write_status()
            time.sleep(self.poll_s)

        print(f"{LOG} shutting down", flush=True)
        self.stop_isaac()
        self.phase = PHASE_IDLE
        self.ready = False
        self.write_status()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--control-dir", required=True,
                        help="Launcher-owned directory for requests and status")
    parser.add_argument("--robot", required=True,
                        help="Robot id to start with; already resolved")
    parser.add_argument("--log-dir", default="",
                        help="Where per-generation Isaac logs are written")
    parser.add_argument("--poll", type=float, default=1.0)
    args = parser.parse_args()

    log_dir = args.log_dir or os.path.join(args.control_dir, "logs")
    supervisor = Supervisor(args.control_dir, args.robot, log_dir, args.poll)
    try:
        return supervisor.run()
    except KeyboardInterrupt:
        supervisor.stop_isaac()
        return 130


if __name__ == "__main__":
    sys.exit(main())
