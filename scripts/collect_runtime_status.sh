#!/usr/bin/env bash
# Collect ALLOWLISTED WISEPACK container facts into results/runtime-status.json.
#
# SECURITY: this script does NOT mount or use the Docker socket from inside any
# container. It runs on the HOST and inspects ONLY the three named WISEPACK
# containers, extracting a fixed set of non-sensitive fields. It never dumps the
# environment, mounts, labels or any unrelated container.
#
# The diagnostics page reads the generated file; it does not run docker itself.
set -eu
REPO="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
OUT="${WISEPACK_RESULTS_DIR:-$REPO/results}/runtime-status.json"
mkdir -p "$(dirname "$OUT")"

# Fixed allowlist of container names and their known roles.
declare -A ROLE=(
    [wisepack-dashboard]=dashboard
    [wisepack-orion-ld]=orion
    [wisepack-mongo-dds]=mongo
)

emit() {
    printf '{\n  "generated_at": "%s",\n  "containers": [\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    local first=1
    for name in wisepack-dashboard wisepack-orion-ld wisepack-mongo-dds; do
        if ! command -v docker >/dev/null 2>&1; then continue; fi
        # Only the allowlisted fields. --format restricts output to exactly these.
        local line
        line="$(docker inspect "$name" \
            --format '{{.Name}}|{{.Config.Image}}|{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}|{{.Created}}|{{.State.StartedAt}}|{{.RestartCount}}' \
            2>/dev/null)" || continue
        [ -z "$line" ] && continue
        IFS='|' read -r cname image state health created started restarts <<<"$line"
        cname="${cname#/}"
        [ "$first" -eq 0 ] && printf ',\n'
        first=0
        printf '    {"name": "%s", "image": "%s", "state": "%s", "health": "%s", "created": "%s", "started": "%s", "restart_count": %s, "known_role": "%s"}' \
            "$cname" "$image" "$state" "$health" "$created" "$started" "${restarts:-0}" "${ROLE[$cname]:-}"
    done
    printf '\n  ]\n}\n'
}

emit > "$OUT"
echo "wrote $OUT"
