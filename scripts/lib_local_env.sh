#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# lib_local_env.sh — host-specific settings, resolved in one documented order.
#
#     source "$REPO/scripts/lib_local_env.sh"
#     wisepack_load_local_env "$REPO"
#     wisepack_resolve_ssh_port          # sets WISEPACK_SSH_PORT
#
# WHY THIS EXISTS
# ---------------
# The Isaac WebRTC stream is unauthenticated, so the documented way to reach it
# from another machine is an SSH port-forward. Building that command needs this
# server's SSH port — which is host-specific, is not always 22, and has no
# business being written into a tracked file, a README, a test fixture or a log.
#
# So it is resolved at runtime, from the most explicit source available:
#
#   1. an already-exported WISEPACK_SSH_PORT   — an operator override wins
#   2. config/local.env                        — this host, git-ignored
#   3. the SERVER-side port of $SSH_CONNECTION — its FOURTH field, present when
#                                                the caller is in an SSH session
#   4. nothing                                 — left UNRESOLVED, with a
#                                                diagnostic. Never defaulted.
#
# WHY NOT DEFAULT TO 22. A wrong port produces a port-forward command that looks
# right, runs without error, and connects to the wrong place or to nothing. An
# unresolved placeholder makes the operator supply the value they alone know.
# ---------------------------------------------------------------------------

#: The literal used when the port could not be resolved. Deliberately not a
#: number, so it cannot be mistaken for one if it ends up in a copied command.
WISEPACK_SSH_PORT_PLACEHOLDER="<ssh-port>"

wisepack_load_local_env() {
    # Read config/local.env WITHOUT executing it: it is a plain KEY=VALUE file,
    # and sourcing host-local files as shell code is how a stray backtick in a
    # config becomes a command execution.
    local repo="${1:-.}"
    local file="$repo/config/local.env"
    [ -f "$file" ] || return 0
    local line key value
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*) continue ;; esac
        key="${line%%=*}"
        value="${line#*=}"
        # ALLOWLIST. Only keys this project owns, so a stray or malicious line
        # cannot set PATH, LD_PRELOAD or anything else. Note this is a parser,
        # not a shell: the file is never sourced, so a backtick or $(...) in a
        # value is data.
        case "$key" in
            WISEPACK_SSH_PORT)
                # An explicit export always wins over the file.
                [ -n "${WISEPACK_SSH_PORT:-}" ] || export WISEPACK_SSH_PORT="$value"
                ;;
            WISEPACK_ISAAC_STREAM_HOST)
                # The ADVERTISED endpoint for a remote native client. Host-
                # specific and therefore tedious to retype, but it is not a
                # prerequisite for WebRTC: unset simply means the descriptor
                # advertises loopback and says so.
                [ -n "${WISEPACK_ISAAC_STREAM_HOST:-}" ] \
                    || export WISEPACK_ISAAC_STREAM_HOST="$value"
                ;;
            # -- PHYSICAL PERCEPTION (README §15a) -----------------------
            # All WISEPACK-owned: perception runs entirely from this repository
            # and no setting here names another project, its checkout or its
            # Python. Host-specific by nature: which camera is plugged in, where
            # the weights live, how big the printed calibration board is, and
            # how big the objects on this table actually are. Listed here for
            # the same reason as the two above — an allowlisted key is honoured,
            # and a key that is NOT listed is silently ignored, which would
            # make a documented variable quietly do nothing.
            WISEPACK_PERCEPTION_SOURCE|WISEPACK_PERCEPTION_DETECTOR|\
            WISEPACK_PERCEPTION_SERVICE_URL|WISEPACK_PERCEPTION_CAMERA|\
            WISEPACK_PERCEPTION_READY_TIMEOUT|WISEPACK_PERCEPTION_LOG|\
            WISEPACK_PERCEPTION_MODEL_PATH|WISEPACK_PERCEPTION_MODEL_CACHE|\
            WISEPACK_PERCEPTION_MODEL_DOWNLOAD|WISEPACK_PERCEPTION_VENV|\
            WISEPACK_PERCEPTION_AUTO_SETUP|WISEPACK_PERCEPTION_CONFIDENCE|\
            WISEPACK_PERCEPTION_WIDTH|WISEPACK_PERCEPTION_HEIGHT|\
            WISEPACK_PERCEPTION_SET_RESOLUTION|\
            WISEPACK_PERCEPTION_CALIBRATION_MARKERS|\
            WISEPACK_PERCEPTION_CALIBRATION_EXTENT_MM|\
            WISEPACK_PERCEPTION_CALIBRATION_DICTIONARY|\
            WISEPACK_PHYSICAL_PROXY_DIAMETER_MM|WISEPACK_PHYSICAL_PROXY_LENGTH_MM|\
            WISEPACK_PHYSICAL_PROXY_WALL_MM|WISEPACK_PHYSICAL_PROXY_MATERIAL|\
            WISEPACK_PHYSICAL_PROXY_GROUP|WISEPACK_PHYSICAL_FRAME_ID|\
            WISEPACK_PHYSICAL_WORKAREA_WIDTH_MM|WISEPACK_PHYSICAL_WORKAREA_DEPTH_MM)
                # An explicit export always wins over the file, exactly as above.
                # INDIRECT EXPANSION, not eval: `$key` has already been matched
                # against the literal allowlist above, but `${!key}` keeps the
                # promise made at the top of this function — the file is parsed,
                # never executed, in every branch.
                [ -n "${!key:-}" ] || export "$key=$value"
                ;;
        esac
    done < "$file"
}

#: Placeholders from the template. Present means "not filled in", which must be
#: treated exactly like absent — otherwise the literal string is advertised as
#: an endpoint and a client tries to resolve it.
wisepack_clear_placeholders() {
    case "${WISEPACK_ISAAC_STREAM_HOST:-}" in
        YOUR_REACHABLE_SERVER_ADDRESS|'') unset WISEPACK_ISAAC_STREAM_HOST ;;
    esac
    case "${WISEPACK_SSH_PORT:-}" in
        YOUR_SSH_PORT) unset WISEPACK_SSH_PORT ;;
    esac
}

wisepack_stream_host_source() {
    # Which resolution step supplied the advertised host — for the launcher's
    # printed configuration. Never prints the value itself.
    if [ -n "${WISEPACK_ISAAC_STREAM_HOST:-}" ]; then
        echo "explicit (environment or config/local.env)"
    else
        echo "default 127.0.0.1 (local/forwarded)"
    fi
}

wisepack_resolve_ssh_port() {
    # 1. already exported, or loaded from config/local.env
    if [ -n "${WISEPACK_SSH_PORT:-}" ] \
            && [ "$WISEPACK_SSH_PORT" != "YOUR_SSH_PORT" ]; then
        export WISEPACK_SSH_PORT
        return 0
    fi
    # 2. the SERVER-side port of the current SSH session (4th field)
    if [ -n "${SSH_CONNECTION:-}" ]; then
        local detected
        detected="$(printf '%s' "$SSH_CONNECTION" | awk '{print $NF}')"
        case "$detected" in
            ''|*[!0-9]*) ;;                       # not a number: ignore
            *) export WISEPACK_SSH_PORT="$detected"; return 0 ;;
        esac
    fi
    # 3. unresolved — say so once, and keep going. This is not fatal: it only
    #    affects the wording of a documentation hint.
    export WISEPACK_SSH_PORT="$WISEPACK_SSH_PORT_PLACEHOLDER"
    return 1
}

wisepack_ssh_port_hint() {
    # A one-line diagnostic for the unresolved case, suitable for stderr.
    if [ "${WISEPACK_SSH_PORT:-}" = "$WISEPACK_SSH_PORT_PLACEHOLDER" ]; then
        echo "this host's SSH port could not be detected — set WISEPACK_SSH_PORT," \
             "or copy config/local.env.example to config/local.env and fill it in"
    fi
}
