#!/usr/bin/env bash
# Generate presentation evidence: run artefacts under results/ and SVG figures
# under images/generated/. Pure Python — no ROS, no Docker, no FIWARE needed.
set -eu
REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
exec python3 "$REPO/scripts/generate_demo_artifacts.py" "$@"
