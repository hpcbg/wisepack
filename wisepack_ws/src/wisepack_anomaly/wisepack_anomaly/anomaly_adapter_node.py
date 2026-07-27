"""Anomaly adapter — accepts Topic #2-compatible events from a FUTURE detector.

The simulator publishes AnomalyEvent JSON directly. A real external detector may
speak a slightly different dialect, so this adapter is the documented seam: it
subscribes to a raw external topic, validates and normalises the payload to the
WISEPACK AnomalyEvent schema, and republishes it on the canonical
``/wisepack/anomaly/event``.

In the interview demo it does nothing but re-stamp and forward, because the
simulator already emits the canonical schema. It exists so that integrating a
real detector is a matter of pointing it at `/wisepack/anomaly/external` rather
than changing the orchestrator or the optimizer.
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from wisepack_bringup import topics as T
from wisepack_bringup.qos import qos_for
from wisepack_core.anomaly import AnomalyEvent

VENDOR_TOPIC = "/wisepack/anomaly/vendor_raw"


class AnomalyAdapterNode(Node):
    def __init__(self) -> None:
        super().__init__("wisepack_anomaly_adapter")
        self.pub = self.create_publisher(
            String, T.ANOMALY_EXTERNAL, qos_for(T.ANOMALY_EXTERNAL))
        self.create_subscription(
            String, VENDOR_TOPIC, self._on_external, qos_for(T.ANOMALY_EXTERNAL))
        self.get_logger().info(
            f"anomaly adapter up — normalises {VENDOR_TOPIC} to "
            f"{T.ANOMALY_EXTERNAL} (future external-detector seam)")

    def _on_external(self, msg: String) -> None:
        try:
            event = AnomalyEvent.from_dict(json.loads(msg.data))
        except (ValueError, KeyError) as exc:
            self.get_logger().warn(f"rejected malformed external anomaly: {exc}")
            return
        # Mark provenance as an adapter passthrough; still simulated in the demo.
        event.source_module = event.source_module or "external_adapter"
        self.pub.publish(String(data=json.dumps(event.to_dict())))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AnomalyAdapterNode()
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
