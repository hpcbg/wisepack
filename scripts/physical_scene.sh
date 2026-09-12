#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# physical_scene.sh — the whole bench from the physical D435, as ONE batch.
#
#     ./scripts/physical_scene.sh                          # live camera
#     ./scripts/physical_scene.sh --dataset scene-20260912-101010
#     ./scripts/physical_scene.sh --roi 200,0,1280,720     # restrict the scene
#
# THE SAME CODE THE DASHBOARD RUNS. `perception/scene_pipeline.run_scene` is
# what the *Acquire scene* button calls; this wrapper only parses arguments
# and prints the table. Every object standing on the fitted work plane is
# segmented from depth and colour, sized by its footprint, matched to a demo
# object class (config/scene_demo_classes.yaml) and written to
# .cache-perception/physical-scene/ with its images. Nothing is submitted to a
# run from here — that is the dashboard's job, through the same batch.
#
# IDENTITY BY SIZE, POSE ON THE PLANE. Say so wherever the result is quoted.
# ---------------------------------------------------------------------------
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2
exec python3 - "$@" <<'PY'
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "perception"))
sys.path.insert(0, os.path.join(os.getcwd(), "wisepack_ws", "src", "wisepack_core"))
from physical_pipeline import PhysicalAcquisitionError
from scene_pipeline import OUT, run_scene

parser = argparse.ArgumentParser(description="whole-scene physical D435 acquisition")
parser.add_argument("--dataset", default="", help="replay a recorded capture instead of the camera")
parser.add_argument("--roi", default="", help="x0,y0,x1,y1 colour pixels; restricts the scene")
parser.add_argument("--json", action="store_true", help="print the document instead of the table")
args = parser.parse_args()
roi = [int(v) for v in args.roi.split(",")] if args.roi else None
try:
    result = run_scene(roi_px=roi, dataset=args.dataset, log=lambda m: print(f"[scene] {m}"))
except PhysicalAcquisitionError as exc:
    print(f"[scene] REFUSED at {exc.stage}: {exc.reason}", file=sys.stderr)
    sys.exit(1)
doc = result.document
if args.json:
    print(json.dumps(doc, indent=2, default=str)); sys.exit(0)
print(f"[scene] {doc['run_label']} — capture {doc['dataset']} — {doc['counts']}")
print(f"{'#':>3} {'status':<12} {'class':<15} {'model':<10} {'L x W x H mm':<18} note")
for o in doc["objects"]:
    fp = f"{o['length_mm']:.0f} x {o['width_mm']:.0f} x {o['height_mm']:.0f}"
    print(f"{o['index']:>3} {o['status']:<12} {o['demo_type'] or '-':<15} {o['model_id'] or '-':<10} {fp:<18} {o['reason'][:60]}")
print(f"[scene] images and physical_scene.json in {OUT}")
PY
