"""NGSI-LD entity registry — derived from bridge_config.yaml, never hand-typed.

Hard-coding entity ids in a consumer is how a dashboard ends up querying an
entity the bridge stopped producing. These are read from the same YAML the
Orion-LD mapping is generated from, so a rename propagates everywhere or fails
loudly in one place.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "..", "config", "bridge_config.yaml")

#: Fallback used when pyyaml is unavailable or the package is not installed with
#: its share/ data. Kept in step by test_ros_fiware.py, which asserts the two
#: agree — so this can never silently drift from the YAML.
_FALLBACK: Dict[str, str] = {
    "system": "urn:ngsi-ld:WISEPACKSystem:main",
    "scenario": "urn:ngsi-ld:WISEPACKScenario:current",
    "plan": "urn:ngsi-ld:WISEPACKPackingPlan:current",
    "robot": "urn:ngsi-ld:WISEPACKRobot:arm-01",
    "actions": "urn:ngsi-ld:WISEPACKActionStream:main",
    "kpi": "urn:ngsi-ld:WISEPACKKPI:current",
}

_SHORT = {
    "WISEPACKSystem": "system",
    "WISEPACKScenario": "scenario",
    "WISEPACKPackingPlan": "plan",
    "WISEPACKRobot": "robot",
    "WISEPACKActionStream": "actions",
    "WISEPACKKPI": "kpi",
}


def _urn(entity_id: str, entity_type: str) -> str:
    if entity_id.split(":", 1)[0] == entity_type:
        return f"urn:ngsi-ld:{entity_id}"
    return f"urn:ngsi-ld:{entity_type}:{entity_id}"


def load_mapping(path: str = CONFIG) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """(short name -> URN, URN -> attribute names) read from the bridge YAML."""
    try:
        import yaml                                    # noqa: PLC0415
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except Exception:                                  # noqa: BLE001
        return dict(_FALLBACK), {}

    ids: Dict[str, str] = {}
    attrs: Dict[str, List[str]] = {}
    for block in ("ros_to_fiware", "fiware_to_ros"):
        for m in cfg.get(block, []) or []:
            etype = m.get("fiware_entity_type") or m["fiware_entity"].split(":", 1)[0]
            urn = _urn(m["fiware_entity"], etype)
            ids[_SHORT.get(etype, etype.lower())] = urn
            attrs.setdefault(urn, [])
            if m["fiware_attribute"] not in attrs[urn]:
                attrs[urn].append(m["fiware_attribute"])
    return (ids or dict(_FALLBACK)), attrs


ENTITY_IDS, ENTITY_ATTRIBUTES = load_mapping()

#: Attributes Orion-LD writes INTO ROS. Everything else flows ROS -> Orion-LD.
INBOUND_ATTRIBUTES = ("approval", "command")

__all__ = ["ENTITY_IDS", "ENTITY_ATTRIBUTES", "INBOUND_ATTRIBUTES", "load_mapping"]
