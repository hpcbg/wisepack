#!/usr/bin/env bash
# Record the README's Human-in-the-Loop GIFs from the real dashboard.
#
# Headless Chromium drives the live sim-mode dashboard through the same REST
# command path the buttons use, screenshots at a fixed cadence, and ffmpeg
# assembles the frames with a generated palette. No desktop recording, no manual
# steps, deterministic preset and seed.
#
#     ./generate_readme_gifs.sh
#     ./generate_readme_gifs.sh --only approve --fps 2
#
# Requires: playwright (+ chromium) and ffmpeg.
set -eu
REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"

if ! python3 -c 'import playwright' >/dev/null 2>&1; then
    echo "ERROR: playwright is not installed." >&2
    echo "       pip install playwright && playwright install chromium" >&2
    exit 2
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg is not installed (needed to assemble the GIFs)." >&2
    exit 2
fi

exec python3 "$REPO/scripts/generate_readme_gifs.py" "$@"
