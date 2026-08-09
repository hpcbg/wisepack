#!/usr/bin/env python3
"""Resolve the FoundationPose network weights — from the OFFICIAL source only.

    python3 scripts/foundationpose_weights.py            # fetch if missing
    python3 scripts/foundationpose_weights.py --check    # report, fetch nothing

WHY THIS IS ITS OWN SCRIPT
--------------------------
The weights are the one part of the FoundationPose runtime that cannot be built
reproducibly. They are ~1 GB, are covered by the NVIDIA Source Code License, and
are published through a Google Drive folder — there is no versioned URL to pin,
no checksum published upstream, and no release artefact. So they are fetched
into a WISEPACK-owned, git-ignored cache and mounted read-only into the worker,
and this script is the single place that knows how.

OFFICIAL SOURCE ONLY, AND THAT IS A DELIBERATE CONSTRAINT
---------------------------------------------------------
Community re-uploads of these checkpoints exist on model hubs. Using one would
mean WISEPACK's pose results depended on weights whose provenance nobody can
attest to — an unattributable dependency inside a licensed pipeline. When the
official source is rate-limited (which it frequently is), the correct behaviour
is to REPORT AND WAIT, not to substitute.

WHAT IS RECORDED
----------------
For each checkpoint that arrives: filename, exact byte size and SHA-256. Upstream
publishes no reference hashes, so these are OBSERVED values — they let a later
run prove it used the same bytes, which is a different and weaker claim than
verifying against upstream. `provenance.json` says so in as many words.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

#: The official folder named in the FoundationPose README at the pinned
#: revision. Re-checked each time this script is touched; see UPSTREAM_NOTES.md.
OFFICIAL_WEIGHTS_FOLDER_ID = "1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i"
OFFICIAL_WEIGHTS_URL = (
    f"https://drive.google.com/drive/folders/{OFFICIAL_WEIGHTS_FOLDER_ID}")

#: Upstream's own directory names. The refiner and the scorer are separate
#: checkpoints and either can be present without the other.
CHECKPOINTS = {
    "refiner": "2023-10-28-18-33-37",
    "scorer": "2024-01-11-20-02-45",
}
CHECKPOINT_FILE = "model_best.pth"

#: Below this it is not a checkpoint. A rate-limited Drive response is an HTML
#: page, and it saves happily under the right filename.
MIN_PLAUSIBLE_BYTES = 1_000_000

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(REPO, ".cache-perception", "foundationpose", "weights")

#: The downloader gets its OWN environment, and that is the whole point.
#:
#: `gdown` is needed by nothing else in WISEPACK. Installing it into
#: `.venv-perception` would put a new dependency into the environment the
#: WORKING planar Faster R-CNN detector runs in, to serve an optional feature —
#: an unnecessary risk to the thing that already works. Installing it with
#: `pip --user` would put it on the host's Python for every project on the
#: machine. So: a throwaway venv beside the weights cache, git-ignored,
#: deletable at any time without consequence.
DOWNLOADER_VENV = os.path.join(REPO, ".cache-perception", "foundationpose",
                               "downloader")


def downloader_python() -> str:
    """Path to an interpreter that has `gdown`, creating one if need be.

    Returns "" if it cannot be built — offline, or no venv module. That is
    reported, never worked around by reaching for a different weight source.
    """
    candidate = os.path.join(DOWNLOADER_VENV, "bin", "python")
    if os.path.isfile(candidate):
        probe = subprocess.run([candidate, "-c", "import gdown"],
                               capture_output=True)
        if probe.returncode == 0:
            return candidate
    print(f"creating the downloader environment in {DOWNLOADER_VENV}")
    try:
        subprocess.run([sys.executable, "-m", "venv", DOWNLOADER_VENV],
                       check=True, capture_output=True)
        subprocess.run([candidate, "-m", "pip", "install", "--quiet",
                        "--upgrade", "pip"], check=True, capture_output=True)
        subprocess.run([candidate, "-m", "pip", "install", "--quiet", "gdown"],
                       check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        print(f"  could not build it: {detail.decode(errors='replace').strip()[:400]}",
              file=sys.stderr)
        return ""
    return candidate if os.path.isfile(candidate) else ""


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(directory: str) -> Dict[str, Any]:
    """What is present right now, with sizes and hashes for what is."""
    report: Dict[str, Any] = {"directory": directory, "checkpoints": {}}
    for role, folder in CHECKPOINTS.items():
        path = os.path.join(directory, folder, CHECKPOINT_FILE)
        entry: Dict[str, Any] = {"expected_path": path, "present": False}
        if os.path.isfile(path):
            size = os.path.getsize(path)
            entry["size_bytes"] = size
            if size < MIN_PLAUSIBLE_BYTES:
                entry["problem"] = (
                    f"only {size} bytes — not a checkpoint. A rate-limited "
                    "download saves an HTML page under the right name; delete "
                    "it and fetch again.")
            else:
                entry["present"] = True
                entry["sha256"] = sha256_of(path)
        report["checkpoints"][role] = entry
    report["complete"] = all(e["present"] for e in report["checkpoints"].values())
    return report


def write_provenance(directory: str, report: Dict[str, Any]) -> str:
    """Record what was obtained, from where, and what it hashes to."""
    path = os.path.join(directory, "provenance.json")
    document = {
        "source": "official",
        "source_url": OFFICIAL_WEIGHTS_URL,
        "licence": ("NVIDIA Source Code License — non-commercial research use. "
                    "FoundationPose weights are third-party; WISEPACK ships "
                    "none of them."),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hash_note": ("SHA-256 values are OBSERVED, not verified against an "
                      "upstream reference: upstream publishes no checksums. "
                      "They prove a later run used the same bytes; they do not "
                      "prove those bytes are upstream's."),
        "checkpoints": report["checkpoints"],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    return path


def fetch(directory: str) -> str:
    """Attempt the official download. Returns "" on success, else the reason."""
    python = downloader_python()
    if not python:
        return ("no interpreter with `gdown` could be prepared, and gdown is "
                "the only practical client for a Google Drive folder.")
    os.makedirs(directory, exist_ok=True)
    program = (
        "import gdown, sys\n"
        f"gdown.download_folder(id={OFFICIAL_WEIGHTS_FOLDER_ID!r}, "
        f"output={directory!r}, quiet=False, use_cookies=False)\n")
    result = subprocess.run([python, "-c", program], capture_output=True,
                            text=True)
    if result.returncode == 0:
        return ""
    output = f"{result.stdout}\n{result.stderr}"
    lowered = output.lower()
    # THE RATE LIMIT IS NAMED, because it is the expected failure and it is
    # temporary. Anything else gets reported verbatim rather than guessed at.
    if ("too many users" in lowered or "quota" in lowered
            or "cannot retrieve the folder" in lowered):
        return ("the official Google Drive folder is rate-limited right now "
                "(\"too many users have viewed or downloaded this file "
                "recently\"). This clears on its own, usually within 24 hours. "
                "WISEPACK does NOT substitute an unofficial mirror.")
    lines = [line for line in output.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "the download failed without a message"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=DEFAULT_DIR,
                        help="the WISEPACK-owned weights cache")
    parser.add_argument("--check", action="store_true",
                        help="report what is present; download nothing")
    args = parser.parse_args(argv)

    report = inspect(args.dir)
    if report["complete"]:
        print(f"FoundationPose weights: COMPLETE in {args.dir}")
        for role, entry in report["checkpoints"].items():
            print(f"  {role:<8} {entry['size_bytes']:>12,} bytes  "
                  f"sha256 {entry['sha256']}")
        print(f"  provenance -> {write_provenance(args.dir, report)}")
        return 0

    missing = [r for r, e in report["checkpoints"].items() if not e["present"]]
    if args.check:
        print(f"FoundationPose weights: INCOMPLETE in {args.dir}")
        for role in missing:
            entry = report["checkpoints"][role]
            print(f"  {role:<8} MISSING"
                  + (f" — {entry['problem']}" if entry.get("problem") else ""))
        print(f"  official source: {OFFICIAL_WEIGHTS_URL}")
        return 1

    print(f"fetching the missing checkpoints ({', '.join(missing)}) from the "
          f"official source")
    reason = fetch(args.dir)
    report = inspect(args.dir)
    if report["complete"]:
        print("FoundationPose weights: COMPLETE")
        for role, entry in report["checkpoints"].items():
            print(f"  {role:<8} {entry['size_bytes']:>12,} bytes  "
                  f"sha256 {entry['sha256']}")
        print(f"  provenance -> {write_provenance(args.dir, report)}")
        return 0

    print("FoundationPose weights: STILL INCOMPLETE", file=sys.stderr)
    if reason:
        print(f"  reason: {reason}", file=sys.stderr)
    print(f"  official source: {OFFICIAL_WEIGHTS_URL}", file=sys.stderr)
    print("  The worker still builds and still starts; it reports "
          "inference_available=false until these arrive.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
