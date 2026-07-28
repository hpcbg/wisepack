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
import time
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
            # The mirror must be COMPLETE enough to rebuild the whole dashboard.
            # Anything missing here becomes an empty panel in live mode, which is
            # exactly how the ROS dashboard used to render blank while every node
            # reported healthy.
            self.mirror: Dict[str, Any] = {
                "stage": "IDLE", "progress_pct": 0.0, "readiness": False,
                "current_item_id": None, "current_container_id": None,
                "detected_count": 0, "scenario": None, "scenario_config": None,
                "items": [], "baseline": None, "optimized": None,
                "selected": None, "plan_status": None, "plan_summary": None,
                "strategy_comparison": None, "anomaly": None,
                # whole-process (cutting / inventory / logistics)
                "cut": None, "inventory_summary": None,
                "inventory_containers": [], "logistics_map": None,
                "logistics_robot": None,
                "kpi": {}, "action_sequence": 0, "heartbeat": 0,
                "heartbeat_at": 0.0,
                "run_id": None, "dynamic_events": [],
                # WHO EXECUTED. Absent until the orchestrator publishes it, and
                # deliberately not defaulted to "simulated": an unknown backend
                # and a known-simulated one are different states, and rendering
                # them identically is how a physical run gets mislabelled.
                "execution_backend": None,
                # Per-item physical outcomes, newest last. Bounded — this is a
                # diagnostic surface, not a second audit trail.
                "isaac_results": [],
                "isaac_state": None,
            }

            S = self.create_subscription
            S(String, T.EXECUTION_STATE, self._stage, qos_for(T.EXECUTION_STATE))
            S(Int32, T.SYSTEM_HEARTBEAT, self._heartbeat,
              qos_for(T.SYSTEM_HEARTBEAT))
            S(String, T.SCENARIO_CONFIG, self._scenario_config,
              qos_for(T.SCENARIO_CONFIG))
            S(String, T.WASTE_ITEMS, self._items, qos_for(T.WASTE_ITEMS))
            S(String, T.PLAN_SUMMARY, self._plan_summary, qos_for(T.PLAN_SUMMARY))
            S(String, T.DYNAMIC_EVENT, self._dynamic, qos_for(T.DYNAMIC_EVENT))
            S(Int32, T.ACTION_SEQUENCE, self._sequence, qos_for(T.ACTION_SEQUENCE))
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
            S(String, T.PLAN_STRATEGY_COMPARISON, self._comparison,
              qos_for(T.PLAN_STRATEGY_COMPARISON))
            S(String, T.ANOMALY_STATE, self._anomaly, qos_for(T.ANOMALY_STATE))
            # whole-process: cutting comparison + inventory + logistics
            S(String, T.CUTTING_PROPOSAL, self._json_into("cut"),
              qos_for(T.CUTTING_PROPOSAL))
            S(String, T.INVENTORY_SUMMARY, self._json_into("inventory_summary"),
              qos_for(T.INVENTORY_SUMMARY))
            S(String, T.INVENTORY_CONTAINER_STATE,
              self._json_into("inventory_containers"),
              qos_for(T.INVENTORY_CONTAINER_STATE))
            S(String, T.LOGISTICS_CONTAINER_TASK,
              self._json_into("logistics_map"), qos_for(T.LOGISTICS_CONTAINER_TASK))
            S(String, T.LOGISTICS_MOBILE_ROBOT_STATE,
              self._json_into("logistics_robot"),
              qos_for(T.LOGISTICS_MOBILE_ROBOT_STATE))
            S(String, T.PLAN_STATUS, self._plan_status, qos_for(T.PLAN_STATUS))
            S(String, T.ACTION_EVENT, self._event, qos_for(T.ACTION_EVENT))
            S(String, T.OPERATOR_APPROVAL, self._noop,
              qos_for(T.OPERATOR_APPROVAL))
            S(String, T.EXECUTION_BACKEND, self._backend,
              qos_for(T.EXECUTION_BACKEND))
            # The raw physical feedback, for the diagnostics panel only. The
            # workflow itself is driven entirely by the orchestrator: the
            # dashboard reading this topic is an OBSERVATION, and nothing here
            # publishes on it or acts on it.
            S(String, T.ISAAC_FEEDBACK, self._isaac_feedback,
              qos_for(T.ISAAC_FEEDBACK))

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

        def _heartbeat(self, m):
            """Watchdog input.

            DDS-level liveliness is unusable here (see qos.watchdog_subscribe_qos),
            so staleness is judged from the counter and its arrival time. A
            counter that stops advancing IS a dead orchestrator.
            """
            self.mirror["heartbeat"] = int(m.data)
            self.mirror["heartbeat_at"] = time.time()

        def _sequence(self, m):
            self.mirror["action_sequence"] = int(m.data)

        def _scenario_config(self, m):
            self._json_into("scenario_config")(m)

        def _plan_summary(self, m):
            self._json_into("plan_summary")(m)

        def _anomaly(self, m):
            self._json_into("anomaly")(m)

        def _comparison(self, m):
            """Latest strategy comparison. Stamped with its scenario revision so
            the snapshot can refuse to render one from a superseded batch."""
            try:
                doc = json.loads(m.data)
            except ValueError:
                self.mirror["strategy_comparison"] = {
                    "status": "error", "error": "malformed comparison payload",
                    "results": []}
                return
            doc["_received_at"] = time.time()
            doc["source"] = "ros"
            self.mirror["strategy_comparison"] = doc

        def _dynamic(self, m):
            try:
                self.mirror.setdefault("dynamic_events", []).append(
                    json.loads(m.data))
            except ValueError:
                self.get_logger().warn("malformed dynamic event")

        def _items(self, m):
            """The full waste-item list.

            The Digital Twin needs every item's geometry, segregation group and
            injected flag to colour and label a placement — a plan alone is not
            enough, because a placement only carries an item_id.
            """
            try:
                items = json.loads(m.data)
            except ValueError:
                self.get_logger().warn("malformed item list")
                return
            self.mirror["items"] = items
            # Splice the items into the scenario the dashboard renders, so a
            # single `scenario` object carries what the sim-mode one does.
            scenario = self.mirror.get("scenario") or {}
            scenario["items"] = items
            groups = sorted({i.get("segregation_group", "A") for i in items})
            totals = dict(scenario.get("totals") or {})
            totals.setdefault("items", len(items))
            totals["segregation_groups"] = groups
            totals["has_approximated_items"] = any(
                i.get("is_approximated") for i in items)
            scenario["totals"] = totals
            self.mirror["scenario"] = scenario

        def _record_stamp(self, key, doc):
            # The stamp is also where the mirror learns the run_id. It used to
            # come only from an ActionEvent's details, so before the first event
            # the dashboard had no idea which run it was showing — and a
            # correlation check with no reference run silently compares nothing.
            """Remember WHICH run and revision this component described.

            These topics are latched and arrive independently, so the mirror is
            always a mixture of whatever each publisher last said. Keeping the
            stamp per component is what lets the snapshot notice that the
            scenario on screen and the plan on screen came from different runs
            instead of silently rendering both.
            """
            if not isinstance(doc, dict):
                return
            self.mirror.setdefault("stamps", {})[key] = {
                "run_id": doc.get("run_id"),
                "scenario_revision": doc.get("scenario_revision"),
                "scenario_id": doc.get("scenario_id"),
            }
            if doc.get("run_id"):
                self.mirror["run_id"] = doc["run_id"]

        def _json_into(self, key):
            def handler(m):
                try:
                    doc = json.loads(m.data)
                except ValueError:
                    self.get_logger().warn(f"malformed JSON on {key}")
                    return
                self.mirror[key] = doc
                self._record_stamp(key, doc)
            return handler

        def _scenario(self, m):
            """Scenario summary. Merged, never clobbering an item list."""
            try:
                doc = json.loads(m.data)
            except ValueError:
                self.get_logger().warn("malformed scenario state")
                return
            scenario = dict(self.mirror.get("scenario") or {})
            items = scenario.get("items") or self.mirror.get("items") or []
            totals = dict(scenario.get("totals") or {})
            _stamp_keys = ("scenario_id", "run_id", "scenario_revision")
            totals.update({k: v for k, v in doc.items() if k not in _stamp_keys})
            scenario.update({"scenario_id": doc.get("scenario_id"),
                             "items": items, "totals": totals})
            scenario.setdefault("preset", (self.mirror.get("scenario_config") or {})
                                .get("preset"))
            scenario.setdefault("seed", (self.mirror.get("scenario_config") or {})
                                .get("seed"))
            scenario.setdefault("container_template",
                                (self.mirror.get("scenario_config") or {})
                                .get("container_template"))
            self.mirror["scenario"] = scenario
            self._record_stamp("scenario", doc)

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

        def _backend(self, m):
            """Which execution backend is authoritative for this run."""
            self._json_into("execution_backend")(m)

        def _isaac_feedback(self, m):
            """Physical outcomes, for the diagnostics panel.

            Only the item-terminal reports are retained. The intermediate states
            already reach the dashboard as ordinary action events on the audit
            trail — keeping both would render every physical step twice, once in
            the timeline and once here.
            """
            try:
                doc = json.loads(m.data)
            except ValueError:
                self.get_logger().warn("malformed isaac feedback")
                return
            state = doc.get("state")
            self.mirror["isaac_state"] = state
            if state not in ("ITEM_COMPLETED", "ITEM_FAILED"):
                return
            results = self.mirror.setdefault("isaac_results", [])
            results.append({
                "item_id": doc.get("item_id"),
                "state": state,
                "container_id": doc.get("container_id"),
                "target_pose": doc.get("target_pose"),
                "actual_pose": doc.get("actual_pose"),
                "position_error_mm": doc.get("position_error_mm"),
                "message": doc.get("message", ""),
                "detail": doc.get("detail", {}),
            })
            if len(results) > 64:
                del results[:len(results) - 64]

        def _noop(self, _m):
            """Subscribed only so `ros2 topic info` shows the dashboard endpoint.

            Publishing and subscribing the same topic from one node is harmless
            here (the orchestrator is the consumer) and makes the operator path
            visible when debugging the live graph.
            """

        def _event(self, m):
            try:
                event = ActionEvent.from_dict(json.loads(m.data))
            except (ValueError, KeyError):
                return
            self.mirror["action_sequence"] = max(
                self.mirror.get("action_sequence", 0), event.sequence)
            if event.action == "scenario_ready":
                self.mirror["run_id"] = (event.details or {}).get("run_id") \
                    or self.mirror.get("run_id")
            with state.lock:
                state.events.append(event.to_dict())
                if len(state.events) > 4000:
                    del state.events[:len(state.events) - 4000]

    def spin() -> None:
        global _NODE
        # On Ctrl+C, rclpy raises ExternalShutdownException out of spin(). It is
        # the EXPECTED shutdown signal, not an error, and printing its traceback
        # made a clean exit look like a crash. Catch exactly that, destroy the
        # node once, and shut rclpy down only if it is still up — while still
        # letting any UNEXPECTED exception surface.
        from rclpy.executors import ExternalShutdownException     # noqa: PLC0415
        rclpy.init()
        _NODE = Observer()
        with state.lock:
            state.ros_mirror = _NODE.mirror
        destroyed = False
        try:
            rclpy.spin(_NODE)
        except (ExternalShutdownException, KeyboardInterrupt):
            pass                                        # expected on Ctrl+C
        except Exception as exc:                        # noqa: BLE001
            print(f"[ros-observer] unexpected error: {exc!r}")
        finally:
            try:
                _NODE.destroy_node()
                destroyed = True
            except Exception:                           # noqa: BLE001
                pass
            # Only this thread owns rclpy here, so shutting it down once is safe.
            if rclpy.ok():
                try:
                    rclpy.shutdown()
                except Exception:                       # noqa: BLE001
                    pass
        if not destroyed:
            print("[ros-observer] node teardown skipped")

    _thread = threading.Thread(target=spin, daemon=True)
    _thread.start()
    try:
        await asyncio.Event().wait()      # hold the lifespan task open
    finally:
        # Lifespan is being torn down (server stopping). Ask rclpy to unwind the
        # spin thread cleanly rather than leaving it to a hard process exit.
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:                               # noqa: BLE001
            pass
        _thread.join(timeout=3.0)


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
