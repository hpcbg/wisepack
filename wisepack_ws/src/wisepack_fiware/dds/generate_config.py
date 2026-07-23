#!/usr/bin/env python3
"""Generate the Orion-LD DDS-bridge config from the WISEPACK mapping YAML.

ADAPTED FROM HARMONY — ros2-xarm-pack-bottle/ros2_ws/src/fiware_bridge/dds/
generate_config.py (MIT, Copyright (c) 2026 Kaloyan Yovchev). The eligibility
rules, the `rt/` topic-name rule, the reserved-`status`-leaf exclusion and the
output document shape are HARMONY's and are reused because they encode
end-to-end-validated knowledge about what the bridge actually accepts.

What is different here: WISEPACK has no NGSI-v2 node backend to keep in step
with, so a topic that cannot bridge is a HARD ERROR rather than a warning. On
this repository the DDS path is the *only* audit path — a silently skipped topic
would be a silently missing section of a regulatory record, so the generator
refuses to emit a config that drops one.

Usage:
    python3 generate_config.py                    # reads ../config/bridge_config.yaml
    python3 generate_config.py --domain 0 --check # verify without writing
"""

import argparse
import json
import os
import sys

try:
    import yaml
except ImportError:                                  # pragma: no cover
    sys.stderr.write("ERROR: pyyaml is required.  pip install pyyaml\n")
    sys.exit(2)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "..", "config", "bridge_config.yaml")
DEFAULT_OUTPUT = os.path.join(HERE, "context_broker_config.json")

# Scalar std_msgs the Orion-LD DDS bridge maps as `.<attribute>.value.data`.
# Validated end to end in HARMONY on fiware/orion-ld:1.13.0-PRE-1835:
# String -> JSON string, Bool -> JSON bool, Int32 -> JSON number, both directions.
ELIGIBLE_STD_MSGS = {
    "std_msgs/String", "std_msgs/Bool", "std_msgs/Int32",
    "std_msgs/Int64", "std_msgs/Float32", "std_msgs/Float64",
}

# Orion-LD's DDS module reserves the topic leaf `status` (it collides with ROS 2
# action status / action_msgs/GoalStatusArray). Such a topic never bridges.
RESERVED_LEAF = "status"


def ngsild_ids(entity_id, entity_type):
    """(entityType, urn) for a short id, without doubling an existing prefix."""
    if entity_type:
        if entity_id.split(":", 1)[0] == entity_type:
            return entity_type, f"urn:ngsi-ld:{entity_id}"
        return entity_type, f"urn:ngsi-ld:{entity_type}:{entity_id}"
    return entity_id.split(":", 1)[0], f"urn:ngsi-ld:{entity_id}"


def dds_topic(ros_topic):
    """ROS 2 `/a/b`  ->  DDS `rt/a/b`."""
    return "rt/" + ros_topic.lstrip("/")


def build_topics(cfg):
    topics, problems = {}, []

    def add(mapping, direction):
        ros_topic = mapping["ros_topic"]
        msg_type = mapping.get("ros_msg_type", "")
        if msg_type not in ELIGIBLE_STD_MSGS:
            problems.append(
                f"{ros_topic} ({msg_type or 'unknown'}, {direction}): not a scalar "
                f"std_msgs type. The DDS bridge cannot represent it, and WISEPACK "
                f"has no second path — change the topic to a scalar std_msgs.")
            return
        dt = dds_topic(ros_topic)
        if dt.rsplit("/", 1)[-1] == RESERVED_LEAF:
            problems.append(
                f"{ros_topic} ({direction}): the leaf '{RESERVED_LEAF}' is reserved "
                f"by Orion-LD's DDS module and is silently dropped. Rename the ROS "
                f"leaf to '{RESERVED_LEAF}_json' (the FIWARE attribute can stay).")
            return
        if dt in topics and topics[dt] != {
                "entityType": ngsild_ids(mapping["fiware_entity"],
                                         mapping.get("fiware_entity_type"))[0],
                "entityId": ngsild_ids(mapping["fiware_entity"],
                                       mapping.get("fiware_entity_type"))[1],
                "attribute": mapping["fiware_attribute"]}:
            problems.append(f"{ros_topic}: mapped twice to different attributes")
            return
        entity_type, urn = ngsild_ids(mapping["fiware_entity"],
                                      mapping.get("fiware_entity_type"))
        topics[dt] = {"entityType": entity_type, "entityId": urn,
                      "attribute": mapping["fiware_attribute"]}

    for m in cfg.get("ros_to_fiware", []):
        add(m, "pub->FIWARE")
    for m in cfg.get("fiware_to_ros", []):
        add(m, "FIWARE->sub")
    return topics, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--domain", type=int,
                        default=int(os.environ.get("ROS_DOMAIN_ID", "0")),
                        help="DDS domain id; MUST equal ROS_DOMAIN_ID (default 0)")
    parser.add_argument("--transport", default="udp")
    parser.add_argument("--check", action="store_true",
                        help="validate the mapping without writing the output")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    topics, problems = build_topics(cfg)

    if problems:
        print("ERROR: the mapping contains topics that cannot bridge:\n",
              file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("\nWISEPACK's audit trail has no fallback path, so this is fatal "
              "rather than a warning.", file=sys.stderr)
        return 1
    if not topics:
        print("ERROR: no topics to bridge.", file=sys.stderr)
        return 1

    doc = {
        "dds": {
            "ddsmodule": {"dds": {"domain": args.domain,
                                  "transport": args.transport}},
            "ngsild": {"topics": topics},
        }
    }

    if args.check:
        print(f"OK: {len(topics)} topics bridge cleanly (domain {args.domain}).")
        return 0

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")

    print(f"Wrote {args.output}")
    print(f"  domain: {args.domain}  transport: {args.transport}")
    print(f"  DDS-eligible topics: {len(topics)}")
    entities = {}
    for dds_t, spec in sorted(topics.items()):
        entities.setdefault(spec["entityId"], []).append(
            (dds_t, spec["attribute"]))
    for urn, attrs in sorted(entities.items()):
        print(f"    {urn}")
        for dds_t, attr in attrs:
            print(f"      .{attr:<24s} <- {dds_t}")
    print("\n  Read every value as  <attr>.value.data  (String and numeric alike).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
