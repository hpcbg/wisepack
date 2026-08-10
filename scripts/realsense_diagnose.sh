#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# realsense_diagnose.sh — is the RGB-D camera usable by the FoundationPose
# worker, and if not, WHICH LAYER is broken?
#
#     ./scripts/realsense_diagnose.sh
#
# WHY A CHAIN AND NOT A BOOLEAN
# -----------------------------
# "pyrealsense2 imports" is not "the camera works". Between a camera on a desk
# and a pose estimate there are seven independent things, each of which fails on
# its own and each of which has a DIFFERENT fix:
#
#   1. the HOST sees the device on USB          -> cable, port, power
#   2. the CONTAINER sees that same device      -> Docker USB passthrough
#   3. pyrealsense2 in the container enumerates -> SDK/permissions in the image
#   4. model and serial are readable            -> device identity
#   5. RGB and depth streams both start         -> USB bandwidth, profiles
#   6. depth is aligned to colour               -> alignment actually applied
#   7. intrinsics and depth scale read          -> calibrated acquisition
#
# Reporting one "camera unavailable" for all seven is what sends someone to
# re-seat a cable when the real problem is a container that was started before
# the camera was plugged in.
#
# STEP 2 IS THE ONE WORTH SEPARATING. If the host sees the camera and the
# container does not, that is a Docker passthrough problem — restart the worker
# so it picks up the device node — and NOT a reason to move acquisition back to
# the host.
#
# /dev/video0 IS NOT CONSULTED ANYWHERE HERE. That is the planar webcam's path;
# a RealSense presents several such nodes and none of them carry intrinsics, a
# depth scale or alignment.
# ---------------------------------------------------------------------------

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${WISEPACK_FP_CONTAINER:-wisepack-foundationpose-worker}"
INTEL_VENDOR_ID="8086"

pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILED=1; }
info() { printf '        %s\n' "$1"; }

FAILED=0
step() { printf '\n%s\n' "$1"; }

# --- 1. the host -------------------------------------------------------------

step "1. HOST sees a RealSense on the USB bus"
HOST_NODES=""
for sysfs in /sys/bus/usb/devices/*/; do
    vendor="$(cat "${sysfs}idVendor" 2>/dev/null)"
    [ "$vendor" = "$INTEL_VENDOR_ID" ] || continue
    bus="$(cat "${sysfs}busnum" 2>/dev/null)"
    dev="$(cat "${sysfs}devnum" 2>/dev/null)"
    product="$(cat "${sysfs}product" 2>/dev/null)"
    speed="$(cat "${sysfs}speed" 2>/dev/null)"
    node="$(printf '/dev/bus/usb/%03d/%03d' "$bus" "$dev")"
    HOST_NODES="$HOST_NODES $node"
    pass "$product at $node (${speed:-?} Mbps)"
    # THE V4L2 NODES OF THIS DEVICE ARE PART OF THE CHAIN. librealsense's Linux
    # backend enumerates through /sys/class/video4linux and then opens
    # /dev/videoN; without those nodes step 3 finds no device at all. They are
    # discovered by walking down from THIS Intel device, never by index, so the
    # planar webcam's own nodes are never among them.
    for v4l in "${sysfs}"*/video4linux/video*; do
        v4l_node="/dev/$(basename "$v4l")"
        [ -e "$v4l_node" ] || continue
        HOST_NODES="$HOST_NODES $v4l_node"
        pass "$product also owns $v4l_node (librealsense V4L2 backend)"
    done
    # A D4xx on USB 2 silently loses its higher resolutions and frame rates,
    # which reads as a broken camera rather than a cable in the wrong socket.
    case "$speed" in
        480|12|1.5) info "WARNING: negotiated USB 2 speed — a D4xx needs USB 3 for its full stream set" ;;
    esac
done
if [ -z "$HOST_NODES" ]; then
    fail "no Intel (vendor $INTEL_VENDOR_ID) USB device on this host"
    info "The camera is not on the USB bus. Check the cable and the port."
    info "Everything below depends on this, so the remaining steps are skipped."
    exit 1
fi

# --- 2. the container --------------------------------------------------------

step "2. CONTAINER sees the same device node"
if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    fail "the worker container '$CONTAINER' does not exist"
    info "Start it: ./scripts/setup_foundationpose.sh --run"
    exit 1
fi
for node in $HOST_NODES; do
    if docker exec "$CONTAINER" test -e "$node" 2>/dev/null; then
        pass "$node is visible inside the container"
    else
        fail "$node is NOT visible inside the container"
        info "This is a DOCKER USB PASSTHROUGH problem, not a perception one."
        info "The node is passed with --device when the worker starts, so a"
        info "camera plugged in AFTER the container started is not visible."
        info "Fix: ./scripts/setup_foundationpose.sh --no-build --run"
    fi
done
[ "$FAILED" = "1" ] && exit 1

# --- 3-7. the SDK, inside the container --------------------------------------

step "3-7. librealsense inside the container"
# `-i` IS LOAD-BEARING. Without it `docker exec` gives python3 no stdin, so
# `python3 -` reads an empty program, prints nothing and exits 0 — five checks
# silently skipped and reported as a pass.
docker exec -i "$CONTAINER" python3 - <<'PY'
import sys
sys.path.insert(0, "/opt/wisepack-fp-worker")

def pass_(m): print(f"  \033[32mPASS\033[0m  {m}")
def fail_(m): print(f"  \033[31mFAIL\033[0m  {m}"); sys.exit(1)

try:
    import pyrealsense2 as rs
except Exception as exc:
    fail_(f"3. pyrealsense2 does not import in the container: {exc}")

devices = list(rs.context().query_devices())
if not devices:
    fail_("3. pyrealsense2 enumerates NO device although the node is visible. "
          "Usually a permissions problem on the USB node inside the container.")
pass_(f"3. pyrealsense2 enumerates {len(devices)} device(s)")

# 4. identity
from camera import RGBDStream, describe                      # noqa: E402
document = describe()
name = document["name"]
serial = document["serial_number"]
if not name or not serial:
    fail_("4. the device did not report a model and serial")
pass_(f"4. {name}  serial {serial}  firmware {document['firmware_version']}  "
      f"USB {document['usb_type_descriptor']}")

options = document["synchronised_profiles"]
if not options:
    fail_("5. no colour(BGR8)+depth(Z16) combination at a shared size and rate")
pass_(f"5. {len(options)} synchronised colour+depth profiles; best "
      f"{options[0]['width']}x{options[0]['height']}@{options[0]['fps']}")

# 5-7. actually start the streams and read the calibrated quantities.
try:
    with RGBDStream(serial=serial, align=True) as stream:
        stream.warmup(10)
        colour, depth, _meta = stream.frame()
        state = stream.state()
except Exception as exc:
    fail_(f"5. the streams did not start: {exc}")
pass_(f"5. RGB {colour.shape} and depth {depth.shape} both started")

if state["alignment_verified"]:
    pass_("6. depth IS aligned to colour (verified against the colour intrinsics)")
else:
    fail_("6. alignment was requested but NOT verified — depth does not share "
          "the colour camera's geometry. A dataset must not claim alignment.")

k = state["colour_intrinsics"]
scale = state["depth_scale_mm_per_unit"]
if not k or not scale:
    fail_("7. intrinsics or depth scale could not be read from the device")
pass_(f"7. intrinsics fx={k['fx']:.2f} fy={k['fy']:.2f} "
      f"cx={k['cx']:.2f} cy={k['cy']:.2f} at {k['width']}x{k['height']}")
pass_(f"7. depth scale {scale} mm per unit (read from the device)")
print("\n  All seven checks passed — the worker can acquire calibrated RGB-D.")
PY
exit $?
