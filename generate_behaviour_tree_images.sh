#!/usr/bin/env bash
# Generate WISEPACK Behaviour Tree diagrams from the implementation.
# Pure Python — no ROS, no Docker. PNG requires cairosvg (optional).
set -eu
REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
# Prefer a Python that has cairosvg (for the .png outputs); fall back to python3
# (SVGs are always produced; PNG is optional).
PYBIN="python3"
for cand in python3 python; do
    if "$cand" -c "import cairosvg" >/dev/null 2>&1; then PYBIN="$cand"; break; fi
done
exec "$PYBIN" "$REPO/scripts/generate_behaviour_tree_images.py" "$@"
