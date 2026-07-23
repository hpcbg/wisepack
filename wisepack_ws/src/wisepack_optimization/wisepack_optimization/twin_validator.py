"""Digital Twin validator node — an INDEPENDENT second opinion, in its own process.

This node subscribes to the plan geometry the orchestrator publishes and re-runs
``wisepack_core.PlacementValidator`` on it from scratch. It shares no state with
the orchestrator: it rebuilds every box from the item dimensions and the recorded
axis, exactly as the validator does when the tests feed it hand-made broken plans.

Why a separate node rather than a function call:

  * it is the architecture the proposal describes — the Digital Twin evaluates
    geometric feasibility, collision avoidance and segregation *before* physical
    execution, as a distinct service;
  * a validator running in the optimizer's own process shares the optimizer's
    assumptions, its rounding and its bugs. This one only sees serialised JSON
    that crossed DDS, so it can disagree — and if it does, the disagreement is
    published rather than lost.

It publishes its verdict on the plan-status topic. The leaf is `status_json`,
not `status`: Orion-LD's DDS module reserves the bare `status` leaf and would
silently drop the topic.
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from wisepack_bringup import topics as T
from wisepack_bringup.qos import qos_for
from wisepack_core.domain import Container, PackingPlan, Placement, Scenario, WasteItem
from wisepack_core.validator import PlacementValidator, ValidationConfig


class TwinValidator(Node):
    def __init__(self) -> None:
        super().__init__("wisepack_twin_validator")
        self.declare_parameter("min_support_fraction", 0.70)
        self.declare_parameter("min_clearance_mm", 0)
        self.validator = PlacementValidator(ValidationConfig(
            min_support_fraction=float(
                self.get_parameter("min_support_fraction").value),
            min_clearance_mm=int(self.get_parameter("min_clearance_mm").value)))

        self._items: dict = {}
        self._scenario_id = "unknown"

        self.pub = self.create_publisher(String, T.PLAN_STATUS,
                                         qos_for(T.PLAN_STATUS))
        self.create_subscription(String, T.WASTE_ITEMS, self._on_items,
                                 qos_for(T.WASTE_ITEMS))
        self.create_subscription(String, T.PLAN_GEOMETRY, self._on_geometry,
                                 qos_for(T.PLAN_GEOMETRY))
        self.get_logger().info(
            "Digital Twin validator up — independent re-validation over DDS")

    def _on_items(self, msg: String) -> None:
        try:
            items = [WasteItem.from_dict(d) for d in json.loads(msg.data)]
        except (ValueError, KeyError) as exc:
            self.get_logger().error(f"malformed item list: {exc}")
            return
        self._items = {i.item_id: i for i in items}
        self.get_logger().info(f"item catalogue: {len(self._items)} items")

    def _on_geometry(self, msg: String) -> None:
        if not self._items:
            self.get_logger().warn("plan received before the item catalogue — "
                                   "cannot validate yet")
            return
        try:
            doc = json.loads(msg.data)
            containers = [Container.from_dict(c) for c in doc["containers"]]
            placements = [Placement.from_dict(p) for p in doc["placements"]]
        except (ValueError, KeyError) as exc:
            self.get_logger().error(f"malformed plan geometry: {exc}")
            self._publish({"valid": False, "error": f"malformed geometry: {exc}"})
            return

        # Rebuild the scenario from the catalogue this node received over DDS —
        # never from anything the orchestrator handed over in-process.
        scenario = Scenario(
            scenario_id=doc.get("scenario_id", self._scenario_id),
            preset="received", seed=0,
            items=list(self._items.values()),
            container_template=containers[0] if containers else None)
        plan = PackingPlan(
            plan_id=doc.get("plan_id", "received"),
            scenario_id=scenario.scenario_id,
            algorithm="received", containers=containers, placements=placements)

        report = self.validator.validate_plan(plan, scenario, mark=False)
        verdict = {
            "plan_id": plan.plan_id,
            "valid": report.valid,
            "placements_checked": report.placements_checked,
            "placements_valid": report.placements_valid,
            "violations": [v.to_dict() for v in report.violations[:20]],
            "violation_count": len(report.violations),
            "validator": "wisepack_twin_validator (independent process)",
            "config": report.to_dict()["config"],
        }
        level = self.get_logger().info if report.valid else self.get_logger().error
        level(f"{plan.plan_id}: {'VALID' if report.valid else 'INVALID'} — "
              f"{report.placements_valid}/{report.placements_checked} placements")
        self._publish(verdict)

    def _publish(self, verdict: dict) -> None:
        self.pub.publish(String(data=json.dumps(verdict, separators=(",", ":"))))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TwinValidator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
