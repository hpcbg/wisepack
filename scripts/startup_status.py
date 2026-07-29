#!/usr/bin/env python3
"""WHAT ACTUALLY STARTED, and what died — the file Diagnostics reads.

THE PROBLEM THIS SOLVES
-----------------------
`docker ps` said Up. The dashboard answered. And the WISEPACK stack inside was
not running at all: `ros2 launch` had exited on its first line, the wrapper
waited out a fixed timeout for a topic that was never going to appear, printed
"WISEPACK stack up", and started the dashboard anyway. Every symptom an operator
could see — stage IDLE, no run_id, topics WAITING, components "starting" — was a
consequence, and none of them named the cause.

So process liveness is RECORDED rather than inferred. Two writers, each owning
its own file so they never race:

    results/startup-host.json    the launcher's own children (Isaac, its watcher)
    results/startup-stack.json   the container's children (the ROS launch)

`web/diagnostics.py` merges both into one table. Nothing here talks to Docker,
reads an environment, or inspects a process it does not own.

WHY A FILE RATHER THAN A TOPIC
------------------------------
Because the failure being reported is "the ROS stack is not there". A liveness
signal that travels over ROS cannot report its own transport being absent, and a
diagnostic that goes quiet in exactly the case it exists for is worse than none.

USAGE (called from shell; every subcommand is read-modify-write and atomic)

    startup_status.py init  --out F --scope host --mode isaac \\
                            [--robot panda --robot-source environment ...]
    startup_status.py proc  --out F --name isaac-sim --pid 1234 --expected 1
    startup_status.py proc  --out F --name isaac-sim --running 0 --exit-code 5 \\
                            --error "exited during startup"
    startup_status.py degrade --out F --reason "ros2 launch exited with 1"
    startup_status.py beat  --out F --name isaac-sim
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import sys
import tempfile
import time

#: Every process the two launchers are expected to own, in the order an operator
#: reads them. A name absent from a status file still appears in Diagnostics as
#: "expected, not reported", which is the state a crashed-before-first-write
#: process leaves behind and is worth seeing.
HOST_PROCESSES = ("isaac-sim", "isaac-watcher", "wisepack-container")
STACK_PROCESSES = ("ros-launch", "orchestrator", "perception-sim",
                   "twin-validator", "anomaly-simulator", "dashboard")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@contextlib.contextmanager
def _locked(path: str):
    """Serialise read-modify-write across writers. LOST UPDATES ARE REAL HERE.

    A scope has more than one writer: the container wrapper records the
    dashboard while a background heartbeat is already ticking, and the launcher
    records Isaac while its watcher beats. Both do read-modify-write on the same
    document, so without a lock one silently discards the other's entry —
    observed as the dashboard row reading "unknown" on a run where it had just
    been recorded as running.

    Best-effort: if the lock file cannot be created the write still proceeds.
    Losing an update is better than losing the whole diagnostic.
    """
    handle = None
    try:
        handle = open(path + ".lock", "a+", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except OSError:
        handle = None
    try:
        yield
    finally:
        if handle is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _load(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(path: str, doc: dict) -> None:
    """Atomic replace. A dashboard polling this file must never read half of it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    doc["generated_at"] = _now()
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)) or ".",
                               prefix=".startup-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _process(doc: dict, name: str) -> dict:
    for entry in doc.setdefault("processes", []):
        if entry.get("name") == name:
            return entry
    entry = {"name": name, "pid": None, "expected": True, "running": None,
             "exit_code": None, "last_heartbeat": None, "last_error": ""}
    doc["processes"].append(entry)
    return entry


def cmd_init(args) -> int:
    doc = {
        "scope": args.scope,
        "mode": args.mode,
        "started_at": _now(),
        "degraded": False,
        "degraded_reason": "",
        "processes": [],
    }
    if args.scope == "host":
        expected = HOST_PROCESSES if args.mode.startswith("isaac") else ()
    else:
        expected = STACK_PROCESSES
    for name in expected:
        _process(doc, name)
    if args.robot:
        doc["robot"] = {
            "effective": args.robot,
            "source": args.robot_source or "unknown",
            "profile_revision": args.robot_revision or "",
            "registry_path": args.registry_path or "",
            "registry_default": args.registry_default or "",
            "registry_loaded": True,
        }
    _save(args.out, doc)
    return 0


def cmd_proc(args) -> int:
    with _locked(args.out):
        doc = _load(args.out)
        entry = _process(doc, args.name)
        if args.pid is not None:
            entry["pid"] = args.pid
        if args.expected is not None:
            entry["expected"] = bool(args.expected)
        if args.running is not None:
            entry["running"] = bool(args.running)
        if args.exit_code is not None:
            entry["exit_code"] = args.exit_code
        if args.error:
            entry["last_error"] = args.error
        entry["last_heartbeat"] = _now()
        _save(args.out, doc)
        return 0


def cmd_beat(args) -> int:
    with _locked(args.out):
        doc = _load(args.out)
        entry = _process(doc, args.name)
        entry["running"] = True
        entry["last_heartbeat"] = _now()
        _save(args.out, doc)
        return 0


def cmd_degrade(args) -> int:
    """Mark the whole scope DEGRADED. Never silently reversed.

    A launcher that recovers a process must say so explicitly (`--clear`);
    otherwise an intermittent restart would erase the evidence of the failure
    that made it necessary.
    """
    with _locked(args.out):
        doc = _load(args.out)
        if args.clear:
            doc["degraded"] = False
            doc["degraded_reason"] = ""
        else:
            doc["degraded"] = True
            previous = doc.get("degraded_reason", "")
            doc["degraded_reason"] = (f"{previous}; {args.reason}"
                                      if previous and args.reason not in previous
                                      else args.reason)
        _save(args.out, doc)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init")
    p.add_argument("--out", required=True)
    p.add_argument("--scope", required=True, choices=("host", "stack"))
    p.add_argument("--mode", default="ros")
    p.add_argument("--robot", default="")
    p.add_argument("--robot-source", default="")
    p.add_argument("--robot-revision", default="")
    p.add_argument("--registry-path", default="")
    p.add_argument("--registry-default", default="")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("proc")
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--pid", type=int)
    p.add_argument("--expected", type=int)
    p.add_argument("--running", type=int)
    p.add_argument("--exit-code", type=int)
    p.add_argument("--error", default="")
    p.set_defaults(func=cmd_proc)

    p = sub.add_parser("beat")
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_beat)

    p = sub.add_parser("degrade")
    p.add_argument("--out", required=True)
    p.add_argument("--reason", default="")
    p.add_argument("--clear", action="store_true")
    p.set_defaults(func=cmd_degrade)

    args = parser.parse_args()
    try:
        return args.func(args)
    except OSError as exc:
        # A status file that cannot be written must never take the launcher
        # down with it — the run is still valid, only the reporting is blind.
        print(f"[startup-status] WARNING: {exc}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
