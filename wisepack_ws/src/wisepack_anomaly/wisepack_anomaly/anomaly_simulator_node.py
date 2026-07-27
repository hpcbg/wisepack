"""Simulated anomaly source — an ARCHITECTURE DEMONSTRATION.

This node publishes deterministic, clearly-labelled SIMULATED anomaly events on
``/wisepack/anomaly/event``. It stands in for an independent, application-agnostic
anomaly detector to show that such a detector can publish structured OK/NOK events
over ROS 2 and drive a deterministic workflow reaction — without any change to the
packing optimizer. (Its cutting-position / tool-closure / camera-loss examples are
aligned with EDF Pilot Topic #2 monitoring needs; see the README relevance note.)

It is NOT a detector. There is no camera, no cutting tool and no perception. It
does not run autonomously by default: it publishes only when asked, via the
``/wisepack/operator/command`` `inject_anomaly` command handled by the
orchestrator, or (for standalone testing) on a fixed timer when
``auto_interval_s`` > 0.

The ``anomaly_adapter_node`` counterpart accepts the same schema from a future
external detector, so replacing this simulator with a real one is a drop-in.
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from wisepack_bringup import topics as T
from wisepack_bringup.qos import qos_for
from wisepack_core.anomaly import AnomalyClass, AnomalyEvent


class AnomalySimulatorNode(Node):
    def __init__(self) -> None:
        super().__init__("wisepack_anomaly_simulator")
        self.declare_parameter("auto_interval_s", 0.0)   # 0 == on-demand only
        self.declare_parameter("auto_class", "camera_view_lost")
        self._seq = 0
        self.pub = self.create_publisher(
            String, T.ANOMALY_EXTERNAL, qos_for(T.ANOMALY_EXTERNAL))
        # Standalone-test convenience: emit on a timer if configured.
        interval = float(self.get_parameter("auto_interval_s").value)
        if interval > 0:
            self.create_timer(interval, self._auto_emit)
        self.get_logger().info(
            f"SIMULATED anomaly source up on {T.ANOMALY_EXTERNAL} — "
            "no real detector, "
            "no physical cutting")

    def emit(self, anomaly_class: str, severity: str = "",
             confidence: float = -1.0) -> AnomalyEvent:
        """Publish one deterministic simulated anomaly and return it."""
        self._seq += 1
        event = AnomalyEvent.simulate(
            anomaly_class, sequence=self._seq,
            severity=severity or None,
            confidence=confidence if confidence >= 0 else None)
        self.pub.publish(String(data=json.dumps(event.to_dict())))
        self.get_logger().info(
            f"published SIMULATED anomaly {event.anomaly_class.value} "
            f"({event.severity.value})")
        return event

    def _auto_emit(self) -> None:
        self.emit(str(self.get_parameter("auto_class").value))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AnomalySimulatorNode()
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
