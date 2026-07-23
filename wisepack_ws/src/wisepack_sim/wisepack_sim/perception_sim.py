"""Simulated perception node.

THIS IS NOT A PERCEPTION SYSTEM. There is no camera, no RGB-D input, no detector
and no pose estimator anywhere in this repository. This node reads the ground
truth published by the task generator, attaches a seeded confidence to each item,
optionally drops some, and republishes the result labelled `simulated`.

It exists so the topic graph, the timing and the downstream consumers are the
real shape the perception layer will plug into — the proposal's YOLOv12-OBB +
SAM2 + FPFH/ICP pipeline replaces exactly this node and nothing else. The
extension point is the `detect()` method: swap its body for real inference and
the rest of WISEPACK is unchanged.

Every message it publishes carries `"source": "simulated"`, and the dashboard
renders detection figures with a `simulated` badge, because reporting a
confidence produced by `random.uniform` as a detection rate would be fabrication.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String

from wisepack_bringup import topics as T
from wisepack_bringup.qos import qos_for


class PerceptionSim(Node):
    def __init__(self) -> None:
        super().__init__("wisepack_perception_sim")
        self.declare_parameter("seed", 42)
        self.declare_parameter("confidence_min", 0.82)
        self.declare_parameter("confidence_max", 0.99)
        self.declare_parameter("miss_probability", 0.0)

        self._rng = random.Random(int(self.get_parameter("seed").value))
        self._cmin = float(self.get_parameter("confidence_min").value)
        self._cmax = float(self.get_parameter("confidence_max").value)
        self._miss = float(self.get_parameter("miss_probability").value)

        self.pub_count = self.create_publisher(
            Int32, T.WASTE_DETECTED_COUNT, qos_for(T.WASTE_DETECTED_COUNT))
        self.create_subscription(String, T.WASTE_ITEMS, self._on_items,
                                 qos_for(T.WASTE_ITEMS))
        self.get_logger().info(
            "perception simulator up — SIMULATED detections, no vision model")

    def detect(self, items: List[Dict[str, Any]]) -> Dict[str, float]:
        """EXTENSION POINT. Replace this body with real inference.

        Contract for a real implementation: return {item_id: confidence} for
        everything detected. Nothing else in WISEPACK needs to change.
        """
        out: Dict[str, float] = {}
        for item in items:
            if self._rng.random() < self._miss:
                continue
            out[item["item_id"]] = round(
                self._rng.uniform(self._cmin, self._cmax), 3)
        return out

    def _on_items(self, msg: String) -> None:
        try:
            items = json.loads(msg.data)
        except ValueError as exc:
            self.get_logger().error(f"malformed item list: {exc}")
            return
        detected = self.detect(items)
        self.pub_count.publish(Int32(data=len(detected)))
        self.get_logger().info(
            f"detected {len(detected)}/{len(items)} items (SIMULATED)")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionSim()
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
