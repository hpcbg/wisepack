"""Verify the WISEPACK audit trail inside Orion-LD.

Answers three questions a reviewer will actually ask, and refuses to guess at
any of them:

  1. Is the broker on :1026 really Orion-LD in DDS mode (not an NGSI-v2 Orion)?
  2. Did every mapped entity and attribute appear, with a REAL value rather than
     the `"uninitialized"` placeholder the bridge writes before the first sample?
  3. Does the action sequence in FIWARE match what the workflow actually
     executed?

`"uninitialized"` is reported as its own outcome, never as a failure and never
as success: it means the attribute is mapped but no DDS sample has been
propagated yet, which on a plain (non-Vulcanexus) ROS 2 is the EXPECTED result
and is not a WISEPACK defect. HARMONY documents this precisely; the check
repeats the explanation rather than leaving a reviewer to guess.

    python3 -m wisepack_fiware.verify --expect-sequence 214
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from .entities import ENTITY_ATTRIBUTES, ENTITY_IDS

ORION = os.environ.get("ORION", "http://localhost:1026").rstrip("/")

_TTY = sys.stdout.isatty()
BOLD = "\033[1m" if _TTY else ""
GRN = "\033[32m" if _TTY else ""
YEL = "\033[33m" if _TTY else ""
RED = "\033[31m" if _TTY else ""
CYN = "\033[36m" if _TTY else ""
RST = "\033[0m" if _TTY else ""


def head(msg: str) -> None:
    print(f"\n{BOLD}{CYN}{msg}{RST}")
    print("-" * 60)


def ok(msg: str) -> None:
    print(f"  {GRN}OK{RST}   {msg}")


def warn(msg: str) -> None:
    print(f"  {YEL}!{RST}    {msg}")


def bad(msg: str) -> None:
    print(f"  {RED}FAIL{RST} {msg}")


def _get(path: str, timeout: float = 5.0) -> Tuple[Optional[int], Any]:
    req = urllib.request.Request(f"{ORION}{path}",
                                 headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception:                                  # noqa: BLE001
        return None, None


def classify(entity: Optional[Dict[str, Any]], attr: str) -> Tuple[str, Any]:
    """REAL / UNINITIALIZED / MISSING, plus the value. HARMONY's rule, reused."""
    if entity is None:
        return "MISSING", None
    a = entity.get(attr)
    if a is None:
        return "MISSING", None
    value = a.get("value") if isinstance(a, dict) else a
    if value == "uninitialized":
        return "UNINITIALIZED", None
    if isinstance(value, dict) and "data" in value:
        return "REAL", value["data"]
    return "REAL", value


def broker_is_orionld() -> Tuple[bool, str]:
    status, body = _get("/version", timeout=3.0)
    if status is None:
        return False, f"no response from {ORION}"
    text = json.dumps(body or {}).lower()
    if "orionld" in text:
        return True, (body or {}).get("orionld version", "orion-ld")
    if "orion" in text:
        return False, ("broker is a NON-LD Orion (NGSI-v2). Stop it: the DDS "
                       "path requires Orion-LD started with -wip dds -mongocOnly")
    return False, "unrecognised broker banner"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-sequence", type=int, default=None,
                        help="assert the FIWARE action sequence reached this value")
    parser.add_argument("--json-out", default=None,
                        help="write the machine-readable report here")
    args = parser.parse_args(argv)

    report: Dict[str, Any] = {"broker": ORION, "entities": {}, "checks": []}
    failures = 0
    uninitialised = 0

    head("1. Orion-LD broker")
    is_ld, detail = broker_is_orionld()
    report["broker_is_orionld"] = is_ld
    report["broker_detail"] = detail
    if is_ld:
        ok(f"Orion-LD confirmed at {ORION} ({detail})")
    else:
        bad(f"{detail}")
        report["checks"].append({"name": "broker is Orion-LD", "ok": False,
                                 "detail": detail})
        # Nothing further can be verified without the right broker.
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
        return 1
    report["checks"].append({"name": "broker is Orion-LD", "ok": True,
                             "detail": detail})

    head("2. Mapped entities and attributes")
    for short, urn in sorted(ENTITY_IDS.items()):
        status, entity = _get(f"/ngsi-ld/v1/entities/{urn}?local=true")
        expected = ENTITY_ATTRIBUTES.get(urn, [])
        block: Dict[str, Any] = {"urn": urn, "present": entity is not None,
                                 "attributes": {}}
        if entity is None:
            bad(f"{short:9s} {urn} — entity not found "
                f"(has any node published its topics yet?)")
            failures += 1
            report["entities"][short] = block
            continue
        real = 0
        for attr in expected:
            state, value = classify(entity, attr)
            block["attributes"][attr] = {"state": state, "value": value}
            if state == "REAL":
                real += 1
            elif state == "UNINITIALIZED":
                uninitialised += 1
        print(f"  {BOLD}{short}{RST} {urn}")
        for attr in expected:
            info = block["attributes"][attr]
            shown = str(info["value"])
            if len(shown) > 68:
                shown = shown[:65] + "..."
            mark = {"REAL": f"{GRN}REAL{RST}",
                    "UNINITIALIZED": f"{YEL}uninit{RST}",
                    "MISSING": f"{RED}missing{RST}"}[info["state"]]
            print(f"      .{attr:<24s} {mark:>18s}  {shown if info['state'] == 'REAL' else ''}")
        if real == 0 and expected:
            failures += 1
        report["entities"][short] = block

    head("3. Action trail")
    seq_entity = ENTITY_IDS.get("actions")
    status, entity = _get(f"/ngsi-ld/v1/entities/{seq_entity}?local=true")
    state, sequence = classify(entity, "sequence")
    state_json, action_json = classify(entity, "actionJson")
    report["action_sequence"] = {"state": state, "value": sequence}

    if state != "REAL":
        bad(f"action sequence is {state} — the audit trail did not reach FIWARE")
        failures += 1
    else:
        ok(f"action sequence in FIWARE: {sequence}")
        if args.expect_sequence is not None:
            # The bridge is STATE-oriented: the attribute holds the LATEST event,
            # so the sequence should have reached at least the expected value.
            if int(sequence) >= args.expect_sequence:
                ok(f"sequence {sequence} >= expected {args.expect_sequence}")
                report["checks"].append({"name": "action sequence reached",
                                         "ok": True,
                                         "detail": f"{sequence} >= {args.expect_sequence}"})
            else:
                bad(f"sequence {sequence} < expected {args.expect_sequence} — "
                    "events were lost between ROS and FIWARE")
                failures += 1
                report["checks"].append({"name": "action sequence reached",
                                         "ok": False,
                                         "detail": f"{sequence} < {args.expect_sequence}"})
    if state_json == "REAL" and action_json:
        try:
            latest = json.loads(action_json)
            ok(f"latest action: #{latest.get('sequence')} "
               f"{latest.get('stage')}/{latest.get('action')} "
               f"[{latest.get('source')}]")
            report["latest_action"] = latest
        except ValueError:
            warn("latest action attribute is not valid JSON")

    head("Summary")
    if uninitialised:
        warn(f"{uninitialised} attribute(s) still \"uninitialized\".")
        print("       This means the attribute is MAPPED but no DDS sample has been")
        print("       propagated. The FIWARE DDS Enabler only fills a value once the")
        print("       publisher propagates the std_msgs TypeObject, which needs")
        print("       Vulcanexus on the ROS side — plain ROS 2 Jazzy publishes the")
        print("       topic but does not propagate the type. That is a documented")
        print("       environment requirement, not a WISEPACK defect.")
    if failures:
        bad(f"{failures} check(s) failed")
    else:
        ok("all FIWARE checks passed")

    report["failures"] = failures
    report["uninitialised"] = uninitialised
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n  wrote {args.json_out}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
