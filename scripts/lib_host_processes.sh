#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# lib_host_processes.sh — ONE composable cleanup mechanism for host children.
#
#     . "$REPO/scripts/lib_host_processes.sh"
#     host_register_cleanup isaac_cleanup
#     host_register_cleanup harmony_cleanup
#     host_run_foreground docker run ...        # or exec, when nothing is owned
#
# WHY THIS EXISTS
# ---------------
# The launcher can own more than one host process at a time: Isaac Sim, and now
# the HARMONY perception service. `trap 'isaac_cleanup' EXIT INT TERM` followed
# by `trap 'harmony_cleanup' EXIT INT TERM` does NOT compose — the second call
# silently replaces the first, and whichever process was registered earlier is
# leaked on Ctrl-C. So there is exactly one trap, installed once, over a list of
# hooks that each clean up only what they own.
#
# THE `exec` PROBLEM, which is the other half of the same thing.
# ------------------------------------------------------------
# This launcher historically ended in `exec python3 web/app.py` or
# `exec docker run ...`. `exec` REPLACES this shell, so its EXIT trap never
# runs and any host child it started is orphaned — it would survive Ctrl-C and
# keep holding the camera. Only the Isaac path avoided that, by running the
# container in the foreground instead.
#
# `host_run_foreground` generalises that decision rather than duplicating it:
#
#   * nothing owned  -> `exec`, exactly the historical behaviour, one less
#                       process in the tree and no wrapper shell hanging around;
#   * something owned -> run in the foreground, then clean up and exit with the
#                       child's status.
#
# So a mode that owns no host child is byte-for-byte what it was, and a mode
# that owns one is guaranteed to clean it up.
# ---------------------------------------------------------------------------

#: Space-separated names of cleanup functions, in registration order. Cleanup
#: runs in that order, so a process registered earlier is stopped earlier.
HOST_CLEANUP_HOOKS="${HOST_CLEANUP_HOOKS:-}"

host_register_cleanup() {
    # Idempotent: registering twice must not clean up twice, because a cleanup
    # that runs after its own teardown is where "no such process" noise on
    # Ctrl-C comes from.
    case " $HOST_CLEANUP_HOOKS " in
        *" $1 "*) return 0 ;;
    esac
    HOST_CLEANUP_HOOKS="${HOST_CLEANUP_HOOKS}${HOST_CLEANUP_HOOKS:+ }$1"
    # Installed on FIRST registration and never replaced. Re-installing the same
    # handler is harmless; installing a different one per subsystem is the bug
    # this file exists to prevent.
    trap 'host_cleanup' EXIT INT TERM
}

host_cleanup() {
    local hook
    for hook in $HOST_CLEANUP_HOOKS; do
        # A hook may be declared in the launcher AFTER a mode that never reaches
        # its definition (sim exits long before the Isaac block). Skip what is
        # not defined rather than emitting "command not found" during teardown.
        if declare -F "$hook" >/dev/null 2>&1; then
            "$hook" || true
        fi
    done
    return 0
}

#: True when this shell owns at least one host child that must be cleaned up.
host_owns_children() {
    [ -n "${HOST_CLEANUP_HOOKS// /}" ]
}

host_run_foreground() {
    if ! host_owns_children; then
        exec "$@"
    fi
    "$@"
    local status=$?
    host_cleanup
    # Cleared before exiting so the EXIT trap cannot run the hooks a second
    # time against processes that are already gone.
    trap - EXIT INT TERM
    exit "$status"
}
