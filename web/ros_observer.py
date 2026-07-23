"""ROS 2 observer and operator-command publisher for the WISEPACK dashboard.

Adapted from TEMPO's web/ros_observer.py, with one deliberate difference: TEMPO's
observer publishes NOTHING because its dashboard is a pure observer. WISEPACK's
dashboard has to command — Human-in-the-Loop approval is the point — so this
module also owns the two operator topics.

It owns EXACTLY those two and nothing else. Single-writer discipline is what
makes the Deadline/Liveliness monitoring on the state topics unambiguous, so the
dashboard must never publish a state, plan or KPI topic; it observes those.

QOS IS IMPORTED FROM wisepack_bringup.qos, NEVER REDECLARED. Requested QoS must
be compatible with what a publisher offers, and when it is not, rclpy does not
raise — the subscription silently receives nothing, forever. Two specific traps
in this contract:

  * state topics are offered RELIABLE + TRANSIENT_LOCAL. Requesting BEST_EFFORT
    is compatible but loses the latch, so a dashboard that connects mid-run sees
    an empty scenario until something changes.
  * operator topics are offered TRANSIENT_LOCAL so an approval written by
    Orion-LD is not lost in the gap before the orchestrator subscribes. A
    VOLATILE publisher here would drop exactly the message that matters.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from typing import Any, Dict, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PUBLISHERS: Dict[str, Any] = {}
_NODE = None
_LOCK = threading.Lock()


async def start_ros_observer(state) -> None:
    """Spin rclpy on a background thread and mirror the contract into ``state``.

    Must be a coroutine: app.py schedules it with asyncio.create_task().
    """
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32, Int32, String

    from wisepack_bringup import topics as T
    from wisepack_bringup.qos import qos_for
    from wisepack_core.events import ActionEvent

    class Observer(Node):
        def __init__(self) -> None:
            super().__init__("wisepack_dashboard_observer")
            self.mirror: Dict[str, Any] = {
                "stage": "IDLE", "progress_pct": 0.0, "readiness": False,
                "current_item_id": None, "current_container_id": None,
                "detected_count": 0, "scenario": None, "baseline": None,
                "optimized": None, "selected": None, "plan_status": None,
                "kpi": {}, "action_sequence": 0,
            }

            S = self.create_subscription
            S(String, T.EXECUTION_STATE, self._stage, qos_for(T.EXECUTION_STATE))
            S(Float32, T.EXECUTION_PROGRESS_PCT, self._progress,
              qos_for(T.EXECUTION_PROGRESS_PCT))
            S(Bool, T.SYSTEM_READINESS, self._ready, qos_for(T.SYSTEM_READINESS))
            S(String, T.EXECUTION_CURRENT_ITEM, self._item,
              qos_for(T.EXECUTION_CURRENT_ITEM))
            S(String, T.EXECUTION_CURRENT_CONTAINER, self._container,
              qos_for(T.EXECUTION_CURRENT_CONTAINER))
            S(Int32, T.WASTE_DETECTED_COUNT, self._detected,
              qos_for(T.WASTE_DETECTED_COUNT))
            S(String, T.SCENARIO_STATE, self._scenario, qos_for(T.SCENARIO_STATE))
            S(String, T.PLAN_BASELINE, self._baseline, qos_for(T.PLAN_BASELINE))
            S(String, T.PLAN_OPTIMIZED, self._optimized, qos_for(T.PLAN_OPTIMIZED))
            S(String, T.PLAN_SELECTED, self._selected, qos_for(T.PLAN_SELECTED))
            S(String, T.PLAN_STATUS, self._plan_status, qos_for(T.PLAN_STATUS))
            S(String, T.ACTION_EVENT, self._event, qos_for(T.ACTION_EVENT))

            for topic, key in (
                    (T.KPI_CONTAINERS_BASELINE, "containers_baseline"),
                    (T.KPI_CONTAINERS_OPTIMIZED, "containers_optimized")):
                S(Int32, topic, self._kpi_int(key), qos_for(topic))
            for topic, key in (
                    (T.KPI_UTILIZATION_BASELINE_PCT, "utilization_baseline_pct"),
                    (T.KPI_UTILIZATION_OPTIMIZED_PCT, "utilization_optimized_pct"),
                    (T.KPI_VOLUME_REDUCTION_PCT, "volume_reduction_pct"),
                    (T.KPI_OPTIMIZATION_MS, "optimization_ms"),
                    (T.KPI_PICK_SUCCESS_PCT, "pick_success_pct"),
                    (T.KPI_END_TO_END_SUCCESS_PCT, "end_to_end_success_pct")):
                S(Float32, topic, self._kpi_float(key), qos_for(topic))

            # The dashboard's ONLY publishers.
            with _LOCK:
                _PUBLISHERS[T.OPERATOR_APPROVAL] = self.create_publisher(
                    String, T.OPERATOR_APPROVAL, qos_for(T.OPERATOR_APPROVAL))
                _PUBLISHERS[T.OPERATOR_COMMAND] = self.create_publisher(
                    String, T.OPERATOR_COMMAND, qos_for(T.OPERATOR_COMMAND))

        # -- handlers ------------------------------------------------------ #
        def _stage(self, m):
            self.mirror["stage"] = m.data

        def _progress(self, m):
            self.mirror["progress_pct"] = round(float(m.data), 1)

        def _ready(self, m):
            self.mirror["readiness"] = bool(m.data)

        def _item(self, m):
            self.mirror["current_item_id"] = m.data or None

        def _container(self, m):
            self.mirror["current_container_id"] = m.data or None

        def _detected(self, m):
            self.mirror["detected_count"] = int(m.data)

        def _json_into(self, key):
            def handler(m):
                try:
                    self.mirror[key] = json.loads(m.data)
                except ValueError:
                    self.get_logger().warn(f"malformed JSON on {key}")
            return handler

        def _scenario(self, m):
            self._json_into("scenario")(m)

        def _baseline(self, m):
            self._json_into("baseline")(m)

        def _optimized(self, m):
            self._json_into("optimized")(m)

        def _selected(self, m):
            self._json_into("selected")(m)

        def _plan_status(self, m):
            self._json_into("plan_status")(m)

        def _kpi_int(self, key):
            def handler(m):
                self.mirror["kpi"][key] = int(m.data)
            return handler

        def _kpi_float(self, key):
            def handler(m):
                self.mirror["kpi"][key] = round(float(m.data), 4)
            return handler

        def _event(self, m):
            try:
                event = ActionEvent.from_dict(json.loads(m.data))
            except (ValueError, KeyError):
                return
            self.mirror["action_sequence"] = event.sequence
            with state.lock:
                state.events.append(event.to_dict())
                if len(state.events) > 4000:
                    del state.events[:len(state.events) - 4000]

    def spin() -> None:
        global _NODE
        rclpy.init()
        _NODE = Observer()
        with state.lock:
            state.ros_mirror = _NODE.mirror
        try:
            rclpy.spin(_NODE)
        finally:
            _NODE.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    threading.Thread(target=spin, daemon=True).start()
    await asyncio.Event().wait()          # hold the lifespan task open


def publish_operator_command(command: str, args: Optional[Dict[str, Any]] = None
                             ) -> Tuple[bool, str]:
    """Send an operator decision down the documented ROS/DDS command path.

    Returns (ok, human-readable detail). Approval and rejection go on the
    dedicated approval topic (which is the one Orion-LD maps to the `approval`
    attribute), everything else on the JSON command topic. This is the SAME path
    an external NGSI-LD client uses when it PATCHes the entity, which is exactly
    why the dashboard does not mutate engine state directly in live mode.
    """
    from std_msgs.msg import String

    from wisepack_bringup import topics as T

    with _LOCK:
        approval_pub = _PUBLISHERS.get(T.OPERATOR_APPROVAL)
        command_pub = _PUBLISHERS.get(T.OPERATOR_COMMAND)

    if approval_pub is None or command_pub is None:
        return False, ("ROS publishers are not up yet — the observer is still "
                       "connecting")

    if command == "approve":
        approval_pub.publish(String(data=T.APPROVE))
        return True, f"published {T.APPROVE} on {T.OPERATOR_APPROVAL}"
    if command == "reject":
        approval_pub.publish(String(data=T.REJECT))
        return True, f"published {T.REJECT} on {T.OPERATOR_APPROVAL}"
    if command not in T.OPERATOR_COMMANDS:
        return False, f"unknown command {command!r}"

    payload = json.dumps({"command": command, "args": args or {}})
    command_pub.publish(String(data=payload))
    return True, f"published {command} on {T.OPERATOR_COMMAND}"
