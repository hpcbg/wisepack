# WISEPACK dashboard

FastAPI backend (`app.py`) plus a single self-contained page (`index.html`).
No build step, no bundler, no CDN — the page is one HTML file with inline CSS
and JS, because a demo that needs network access to render is a demo that fails
on venue wifi.

## Running

```bash
../run_wisepack_dashboard.sh sim      # no ROS, no FIWARE, no Docker needed
../run_wisepack_dashboard.sh          # live ROS 2 / DDS stack
../run_wisepack_dashboard.sh fiware   # live + Orion-LD; state read back over NGSI-LD
```

Or directly, if the host has fastapi:

```bash
python3 app.py --source sim --port 8080 --preset mixed_pipes_dense --seed 42
```

## The three sources, and what each one proves

The header badge always names the source in use. No figure is ever shown
without it.

| `--source` | Needs | Packing figures | Execution figures | What it proves |
|---|---|---|---|---|
| `sim` | nothing | **measured** | simulated | the algorithms and KPIs, with zero infrastructure |
| `ros` | Docker | **measured** | simulated | the ROS 2 / DDS node graph and the operator command path |
| `fiware` | Docker + Orion-LD | **measured** | simulated | the full audit path: a value only renders if it crossed DDS into NGSI-LD |

`sim` mode is **not** a separate animation. It drives the same
`wisepack_core.WorkflowEngine`, the same generator, the same optimizer and the
same independent validator that the live stack runs. Only the transport differs.
That is why the container counts and utilization percentages are identical in
all three modes for a given preset and seed.

What is *always* simulated, in every mode: perception confidence, grasp success
and therefore end-to-end success. There is no vision model, no robot and no
physics in this repository, and every such figure carries a `simulated` badge.

## API

REST renders a complete initial state — the page is fully usable with the
WebSocket blocked.

| Endpoint | Purpose |
|---|---|
| `GET /api/state` | everything needed for the first render |
| `GET /api/plans` | baseline, optimized (the comparison pair) and the execution plan |
| `GET /api/kpis` | every KPI with its provenance, plus the proposal-target assessment |
| `GET /api/strategies` | all three packing strategies, run and compared |
| `GET /api/events` | the action log, newest first |
| `GET /api/analytics` | actions by type/stage, durations, re-plans, latency artefact |
| `GET /api/topology` | the position-free system graph plus live node status |
| `GET /api/fiware` | live NGSI-LD read-back (live modes only) |
| `POST /api/command` | operator actions |
| `WS /ws` | optional push; the page polls regardless |

### Commands

`approve`, `reject`, `alternative_strategy`, `inject_item`,
`container_unavailable`, `grasp_failure`, `pause`, `resume`, `step`, `reset`,
`write_artifacts`.

**In live mode these do not touch Python state.** They are published on
`/wisepack/operator/approval` and `/wisepack/operator/command`, which is the
same path an external NGSI-LD client uses when it PATCHes the mapped attribute:

```bash
curl -X PATCH \
  'http://localhost:1026/ngsi-ld/v1/entities/urn:ngsi-ld:WISEPACKSystem:main/attrs/approval' \
  -H 'Content-Type: application/json' \
  -d '{"type":"Property","value":{"data":"APPROVE"}}'
```

Verified: that PATCH returns HTTP 204 and moves the workflow from
`WAIT_FOR_OPERATOR_APPROVAL` to executing. Mutating the engine in-process would
demonstrate a control path that does not exist in the real system.

## The Digital Twin view

SVG, drawn from real placement geometry, offline-capable. Each container gets a
top projection (X-Y) and a side projection (X-Z) at a shared scale, so the
baseline and optimized panels are directly comparable — identical containers
must render at identical size or the picture misleads.

- **colour** — segregation group (blue / amber / magenta, chosen to survive the
  common colour-vision deficiencies rather than relying on red-vs-green)
- **solid fill** — executed; **faded, dashed** — planned but not yet placed
- **red dashed** — a placement the independent validator rejected
- **orange outline** — an item introduced by a dynamic event
- **orange dashed line** — a shelf plate, drawn only for the baseline because it
  is the thing that explains its wasted height

## Falling back

The WebSocket is an enhancement. Polling runs unconditionally and carries the
page on its own; if the socket never connects or dies mid-demo, nothing is lost
but update latency.
