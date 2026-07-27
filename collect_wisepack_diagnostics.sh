#!/usr/bin/env bash
# Generate a timestamped, SAFE WISEPACK support bundle.
#
# SECURITY — the bundle is built from an ALLOWLIST. It NEVER includes:
#   credentials, .env, private keys, access tokens, full environment dumps,
#   unrelated logs, or unrelated Docker/container data.
# It includes only: runtime status (allowlisted containers), the ROS topic +
# QoS summary, the FIWARE broker/version + entity summary, the current bridge
# mapping, the latest results artefacts, and configuration checksums.
set -eu
REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
OUTDIR="${WISEPACK_RESULTS_DIR:-$REPO/results}/diagnostics-$STAMP"
ORION="${ORION:-http://localhost:1026}"
mkdir -p "$OUTDIR"

echo "[diag] collecting into $OUTDIR"

# 1. Allowlisted container runtime status.
"$REPO/scripts/collect_runtime_status.sh" >/dev/null 2>&1 || true
cp -f "$REPO/results/runtime-status.json" "$OUTDIR/" 2>/dev/null || true

# 2. Topic + QoS summary (from the contract — no live ros2 calls needed).
python3 - "$OUTDIR/topic-qos-summary.json" "$REPO" <<'PY'
import json, sys, importlib.util, os
repo = sys.argv[2]
src = os.path.join(repo, "wisepack_ws", "src")
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
T = load("t", os.path.join(src, "wisepack_bringup", "wisepack_bringup", "topics.py"))
json.dump({"topics": T.all_topics(),
           "inbound": list(T.INBOUND_TOPICS),
           "operator_commands": list(T.OPERATOR_COMMANDS),
           "reserved_leaf_violations": T.reserved_leaf_violations()},
          open(sys.argv[1], "w"), indent=2)
print("wrote", sys.argv[1])
PY

# 3. FIWARE broker/version + entity summary (public NGSI-LD reads only).
{
    echo "{"
    echo "  \"broker\": \"$ORION\","
    printf '  "version": '
    curl -s --max-time 3 "$ORION/version" 2>/dev/null | python3 -c "import json,sys;print(json.dumps((sys.stdin.read() or '{}')))" 2>/dev/null || echo '"unreachable"'
    echo ","
    printf '  "entity_types": '
    curl -s --max-time 3 "$ORION/ngsi-ld/v1/types?local=true" -H 'Accept: application/json' 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(json.dumps(d.get('typeList', d) if isinstance(d,dict) else d))" 2>/dev/null || echo '[]'
    echo "}"
} > "$OUTDIR/fiware-summary.json" 2>/dev/null || echo '{"broker":"unreachable"}' > "$OUTDIR/fiware-summary.json"

# 4. Current bridge mapping (the generated, non-sensitive config).
cp -f "$REPO/wisepack_ws/src/wisepack_fiware/dds/context_broker_config.json" "$OUTDIR/" 2>/dev/null || true

# 5. Latest results artefacts (allowlisted patterns only).
for pat in wisepack-run-*.json wisepack-kpis-*.json wisepack-validation-*.md \
           wisepack-fiware-validation-*.md wisepack-dds-fiware-latency-*.json; do
    f="$(ls -1t "$REPO/results/"$pat 2>/dev/null | head -1)"
    [ -n "$f" ] && cp -f "$f" "$OUTDIR/" 2>/dev/null || true
done

# 6. Configuration checksums + software/test versions.
{
    echo "# WISEPACK diagnostics bundle $STAMP"
    echo
    echo "## Configuration checksums"
    for f in wisepack_ws/src/wisepack_fiware/config/bridge_config.yaml \
             wisepack_ws/src/wisepack_fiware/dds/context_broker_config.json \
             config/wisepack.yaml; do
        [ -f "$REPO/$f" ] && echo "$(sha256sum "$REPO/$f" 2>/dev/null | cut -d' ' -f1)  $f"
    done
    echo
    echo "## Software"
    echo "python: $(python3 --version 2>&1)"
    echo "docker: $(docker --version 2>&1 || echo 'not available')"
    echo "git: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
    echo
    echo "## Tests (last known)"
    echo "run: python3 -m pytest tests/ -q"
} > "$OUTDIR/manifest.md"

# 7. EXPLICIT allowlist assertion: fail loudly if anything sensitive slipped in.
if grep -rilE 'password|secret|token|private[_-]?key|BEGIN [A-Z ]*PRIVATE KEY' "$OUTDIR" 2>/dev/null | head -1 | grep -q .; then
    echo "[diag] ERROR: a sensitive-looking string was found in the bundle; aborting." >&2
    rm -rf "$OUTDIR"
    exit 1
fi

echo "[diag] bundle ready: $OUTDIR"
ls -1 "$OUTDIR" | sed 's/^/    /'
