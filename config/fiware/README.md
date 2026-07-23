# FIWARE configuration and example responses

The **authoritative** bridge configuration lives with the code that generates and
consumes it:

| File | Where |
|---|---|
| `bridge_config.yaml` — the single source of truth | [`wisepack_ws/src/wisepack_fiware/config/`](../../wisepack_ws/src/wisepack_fiware/config/bridge_config.yaml) |
| `generate_config.py` — the mapping generator | [`wisepack_ws/src/wisepack_fiware/dds/`](../../wisepack_ws/src/wisepack_fiware/dds/generate_config.py) |
| `context_broker_config.json` — generated, mounted into Orion-LD | [`wisepack_ws/src/wisepack_fiware/dds/`](../../wisepack_ws/src/wisepack_fiware/dds/context_broker_config.json) |
| `docker-compose.dds.yml` — the broker stack | [`wisepack_ws/src/wisepack_fiware/dds/`](../../wisepack_ws/src/wisepack_fiware/dds/docker-compose.dds.yml) |

They are *not* duplicated here. The broker mounts
`context_broker_config.json` by relative path from its compose file, so a second
copy in this directory would be a copy that is never read and silently drifts.

This directory holds **example NGSI-LD responses** — real output captured from a
running broker, for anyone building a consumer without starting the stack.

- [`example-responses.json`](example-responses.json)

## Regenerating the mapping

```bash
cd ../../wisepack_ws/src/wisepack_fiware/dds
python3 generate_config.py --domain 0     # must equal ROS_DOMAIN_ID
python3 generate_config.py --check        # validate without writing
```

The generator **refuses** to emit a config that drops a topic. On this repository
the DDS path is the only audit path, so a silently skipped topic would be a
silently missing section of a regulatory record.

## Reading a value

Every attribute is `<attr>.value.data`, for strings and numbers alike:

```bash
E=urn:ngsi-ld:WISEPACKKPI:current
curl -s "http://localhost:1026/ngsi-ld/v1/entities/$E?local=true" \
  -H 'Accept: application/json' | jq '.volumeReductionPct.value.data'
```

A value of the plain string `"uninitialized"` means the attribute is mapped but
no DDS sample has been propagated yet — treat it as "no data". See the README's
note on why Vulcanexus is required for that to resolve.
