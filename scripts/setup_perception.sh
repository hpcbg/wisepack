#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_perception.sh — create the WISEPACK-OWNED perception environment.
#
#     ./scripts/setup_perception.sh              # venv + dependencies
#     ./scripts/setup_perception.sh --model      # ... and fetch the weights
#     ./scripts/setup_perception.sh --isolated   # do not reuse host packages
#     ./scripts/setup_perception.sh --check      # report, change nothing
#
# YOU DO NOT NORMALLY RUN THIS. With WISEPACK_PERCEPTION_SOURCE=camera the
# launcher runs it for you on the first camera start; from then on it finds the
# environment and starts the service. Simulated perception (the default) never
# touches it.
#
# WHY AN ENVIRONMENT OF OUR OWN
# -----------------------------
# The perception service needs torch, torchvision, OpenCV, fastapi and uvicorn.
# Three things it must NOT do to get them:
#
#   * install into the system Python — that is the operator's machine, not ours;
#   * borrow another project's virtualenv — that made a foreign checkout a
#     runtime dependency of a WISEPACK feature, which is exactly what this
#     replaces;
#   * be activated into the launcher's shell — activation puts this interpreter
#     and its site-packages ahead of everything WISEPACK runs afterwards.
#
# So: a git-ignored `.venv-perception/` inside the working directory, and the
# launcher invokes `.venv-perception/bin/python` directly.
#
# REUSING WHAT THE HOST ALREADY HAS, BY DEFAULT
# ---------------------------------------------
# The venv is created with `--system-site-packages`, so a host that already has
# a CUDA-matched torch keeps it instead of downloading a second ~2.5 GB copy
# that may not match its driver. Anything missing is installed INTO THE VENV;
# nothing is ever installed outside it. `--isolated` opts out and builds a fully
# self-contained environment.
# ---------------------------------------------------------------------------

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${WISEPACK_PERCEPTION_VENV:-$REPO/.venv-perception}"
REQUIREMENTS="$REPO/perception/requirements.txt"
PYTHON_BIN="$VENV/bin/python"

#: Every module the service and its providers import. The single source of truth
#: for "is this environment usable" — checked by this script AND by the launcher,
#: so the two can never disagree about what "ready" means.
PERCEPTION_IMPORT_CHECK='import torch, torchvision, cv2, cv2.aruco, numpy, PIL, fastapi, uvicorn'

WANT_MODEL=0
ISOLATED=0
CHECK_ONLY=0
QUIET=0

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model)     WANT_MODEL=1 ;;
        --isolated)  ISOLATED=1 ;;
        --check)     CHECK_ONLY=1 ;;
        --quiet)     QUIET=1 ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "setup_perception.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

say() { [ "$QUIET" = "1" ] || echo "[perception-setup] $*"; }

# --- is the environment already usable? -------------------------------------

perception_env_ok() {
    [ -x "$PYTHON_BIN" ] || return 1
    "$PYTHON_BIN" -c "$PERCEPTION_IMPORT_CHECK" >/dev/null 2>&1
}

if [ "$CHECK_ONLY" = "1" ]; then
    if perception_env_ok; then
        echo "python : $PYTHON_BIN"
        echo "status : ready"
        exit 0
    fi
    echo "python : ${PYTHON_BIN}"
    echo "status : not ready"
    if [ -x "$PYTHON_BIN" ]; then
        echo "reason : the environment exists but cannot import its dependencies:"
        "$PYTHON_BIN" -c "$PERCEPTION_IMPORT_CHECK" 2>&1 | sed 's/^/         /'
    else
        echo "reason : $VENV does not exist"
    fi
    echo "fix    : ./scripts/setup_perception.sh"
    exit 1
fi

if perception_env_ok && [ "$WANT_MODEL" = "0" ]; then
    say "environment already usable: $PYTHON_BIN"
    exit 0
fi

# --- create it --------------------------------------------------------------

if [ ! -f "$REQUIREMENTS" ]; then
    echo "[perception-setup] ERROR: $REQUIREMENTS is missing" >&2
    exit 1
fi

BOOTSTRAP_PYTHON="${WISEPACK_PERCEPTION_BOOTSTRAP_PYTHON:-python3}"
if ! command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1; then
    echo "[perception-setup] ERROR: $BOOTSTRAP_PYTHON not found — a Python 3 is" >&2
    echo "[perception-setup]        needed to create the perception environment." >&2
    exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
    if [ "$ISOLATED" = "1" ]; then
        say "creating $VENV (isolated — nothing is reused from this host)"
        VENV_FLAGS=""
    else
        say "creating $VENV (reusing host packages where they already exist)"
        VENV_FLAGS="--system-site-packages"
    fi
    # Unquoted on purpose: it is either empty or one literal flag, never a path.
    # shellcheck disable=SC2086
    if ! "$BOOTSTRAP_PYTHON" -m venv $VENV_FLAGS "$VENV"; then
        echo "[perception-setup] ERROR: could not create $VENV." >&2
        echo "[perception-setup]        On Debian/Ubuntu this usually means the" >&2
        echo "[perception-setup]        python3-venv package is missing." >&2
        exit 1
    fi
fi

# --- dependencies -----------------------------------------------------------
#
# Only when something is actually missing. On a host that already has a
# CUDA-matched torch this skips pip entirely, which is both faster and safer
# than reinstalling a wheel that may not match the driver.

if ! "$PYTHON_BIN" -c "$PERCEPTION_IMPORT_CHECK" >/dev/null 2>&1; then
    say "installing perception dependencies from perception/requirements.txt"
    say "(this can take a while the first time — torch is a large download)"
    "$PYTHON_BIN" -m pip install --upgrade pip >/dev/null 2>&1 || true
    if ! "$PYTHON_BIN" -m pip install -r "$REQUIREMENTS"; then
        echo "[perception-setup] ERROR: installing $REQUIREMENTS failed." >&2
        echo "[perception-setup]        Nothing was installed outside $VENV." >&2
        exit 1
    fi
fi

if ! "$PYTHON_BIN" -c "$PERCEPTION_IMPORT_CHECK" >/dev/null 2>&1; then
    echo "[perception-setup] ERROR: the environment still cannot import its" >&2
    echo "[perception-setup]        dependencies after installation:" >&2
    "$PYTHON_BIN" -c "$PERCEPTION_IMPORT_CHECK" 2>&1 | sed 's/^/[perception-setup]        /' >&2
    exit 1
fi

say "environment ready: $PYTHON_BIN"

# --- the weights (optional) -------------------------------------------------
#
# Separate from the environment on purpose: the service resolves and fetches
# them itself on first use, so this is only for pre-seeding a host — a demo
# machine that will be offline, say.

if [ "$WANT_MODEL" = "1" ]; then
    say "resolving detector weights"
    if ! "$PYTHON_BIN" "$REPO/perception/model_store.py"; then
        echo "[perception-setup] ERROR: the detector weights could not be" >&2
        echo "[perception-setup]        resolved or downloaded (see above)." >&2
        exit 1
    fi
fi

exit 0
