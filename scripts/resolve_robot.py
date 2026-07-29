#!/usr/bin/env python3
"""Resolve the effective Isaac robot BEFORE anything is launched.

WHY THIS IS A SEPARATE STEP, AND WHY IT RUNS FIRST
--------------------------------------------------
The launcher used to print

    robot : <configured default>

and hand an EMPTY string to both the simulator and the orchestrator, letting
each resolve it independently. Two things were wrong with that. A placeholder is
not a runtime diagnostic — an operator reading it cannot tell which arm is about
to move, and neither can a bug report. And an unresolved value travelling into
``ros2 launch`` as ``robot:=`` is a malformed argument that kills the whole ROS
stack while the container it runs in stays Up, which is exactly how a dashboard
ends up sitting at IDLE with no run and no visible reason.

So the answer is computed ONCE, here, on the host, before a process is started,
and the same concrete value is then handed to every consumer. This script is the
only place that decision is made for a launch.

OUTPUT is one tab-separated line on stdout, for `read` in shell:

    <robot_id>\\t<source>\\t<profile_revision>\\t<registry_path>\\t<registry_default>

EXIT CODES
    0   resolved; the value on stdout is concrete and validated
    5   could not resolve — the reason is on stderr and NOTHING should start

Never prints a placeholder. If it cannot produce a real robot id it fails.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _pkg in ("wisepack_core",):
    _path = os.path.join(REPO, "wisepack_ws", "src", _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)


#: Anything that looks like a placeholder rather than an identifier. Checked on
#: the way OUT as well as the way in: the whole point of this script is that a
#: literal "<configured default>" can never reach a launch argument.
def _is_placeholder(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    return ("<" in text or ">" in text or " " in text
            or text.lower() in {"none", "null", "default", "unset", "unresolved"})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve the effective Isaac robot id, concretely.")
    parser.add_argument("--robot", default="",
                        help="Explicit robot id; wins over the environment")
    parser.add_argument("--draft", default="",
                        help="Scenario draft selection, if one is known")
    parser.add_argument("--registry", default="",
                        help="Path to isaac_robots.yaml (default: resolved)")
    parser.add_argument("--preset", default="",
                        help="Refuse a robot that cannot run this preset")
    args = parser.parse_args()

    try:
        from wisepack_core.robots import (
            ROBOT_ENV_VAR, RobotConfigError, load_registry,
        )
    except ImportError as exc:                              # pragma: no cover
        print(f"[resolve-robot] ERROR: cannot import the robot registry: {exc}",
              file=sys.stderr)
        return 5

    try:
        registry = load_registry(args.registry or None, reload=True)
    except RobotConfigError as exc:
        print(f"[resolve-robot] ERROR: {exc}", file=sys.stderr)
        return 5
    except Exception as exc:                                # noqa: BLE001
        print(f"[resolve-robot] ERROR: the robot registry could not be read: "
              f"{exc!r}", file=sys.stderr)
        return 5

    # A placeholder ARRIVING is refused too, rather than being treated as
    # "nothing was chosen". `WISEPACK_ISAAC_ROBOT='<configured default>'` is a
    # mistake somebody made, not a request for the default.
    for label, value in (("--robot", args.robot),
                         (ROBOT_ENV_VAR, os.environ.get(ROBOT_ENV_VAR, "")),
                         ("--draft", args.draft)):
        if value and _is_placeholder(value):
            print(f"[resolve-robot] ERROR: {label}={value!r} is not a robot id. "
                  f"Configured robots: {', '.join(sorted(registry.profiles))}",
                  file=sys.stderr)
            return 5

    try:
        profile, source = registry.resolve_with_source(
            explicit=args.robot or None, draft=args.draft or None)
    except RobotConfigError as exc:
        print(f"[resolve-robot] ERROR: {exc}", file=sys.stderr)
        return 5

    if args.preset:
        refusal = profile.preset_refusal(args.preset)
        if refusal:
            # Refused HERE, before Isaac spends a minute loading an asset it
            # will not be allowed to use, and before the operator is shown an
            # approval gate that can never open.
            print(f"[resolve-robot] ERROR: {refusal}", file=sys.stderr)
            return 5

    if _is_placeholder(profile.robot_id):                   # pragma: no cover
        print(f"[resolve-robot] ERROR: the registry resolved to "
              f"{profile.robot_id!r}, which is not a usable id", file=sys.stderr)
        return 5

    print("\t".join([profile.robot_id, source, profile.revision,
                     registry.source_path, registry.default_robot_id]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
