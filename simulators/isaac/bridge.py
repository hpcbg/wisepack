"""ROS 2 transport for the Isaac Sim side of the WISEPACK execution backend.

WHICH rclpy THIS IS
-------------------
Isaac Sim's own. Enabling the ``isaacsim.ros2.bridge`` extension puts Isaac's
internally-built ROS 2 Jazzy libraries on the path, and ``import rclpy`` then
resolves to a build compiled against Isaac's Python ABI — not the one in
/opt/ros/jazzy. That is the isolation this integration depends on:

    * Isaac keeps its bundled interpreter and its own ROS 2 build;
    * the WISEPACK stack keeps Vulcanexus Jazzy inside its container;
    * they meet on the DDS wire, on a shared ROS_DOMAIN_ID, and nowhere else.

Sourcing /opt/ros/jazzy/setup.bash into this process is therefore not merely
unnecessary, it is actively harmful: it puts a second, ABI-incompatible rclpy
ahead of Isaac's on PYTHONPATH, and the failure is an import-time crash deep
inside rclpy's C extension. The launcher scrubs the ROS environment for exactly
this reason — see scripts/run_wisepack_isaac.sh.

WHAT IT CARRIES
---------------
Two ``std_msgs/String`` topics and no custom message types, so nothing here needs
a colcon build to be importable by Isaac's interpreter. The payload schema lives
in ``wisepack_core.isaac_contract``, which is pure-stdlib Python imported by BOTH
ends from the same file — the orchestrator under Vulcanexus and this process
under Isaac. One definition, so the two cannot drift.

The QoS profiles are likewise imported from ``wisepack_bringup.qos`` rather than
restated here. Requested QoS that is incompatible with what a publisher offers
does not raise in rclpy: the subscription silently receives nothing, forever.
Two copies of a profile is exactly how that ends up happening.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from wisepack_bringup import topics as T
from wisepack_bringup.qos import qos_for
from wisepack_core.isaac_contract import (
    ContractError, IsaacCommand, IsaacFeedback, IsaacState, Pose, RunGate,
)

from .config import LOG_BRIDGE


class IsaacRosBridge(Node):
    """Receives WISEPACK commands, publishes physical feedback.

    Single-writer discipline: this node publishes the feedback topic and NOTHING
    else. It does not touch /wisepack/execution/state or any other canonical
    topic — the orchestrator owns those, and a second writer on a latched state
    topic means a reader sees whichever wrote last.
    """

    def __init__(self, on_command: Callable[[IsaacCommand], None]) -> None:
        super().__init__("wisepack_isaac_bridge")
        self._on_command = on_command
        self._lock = threading.Lock()
        #: Gates inbound commands. The command topic is TRANSIENT_LOCAL, so a
        #: latched EXECUTE_ITEM is redelivered every time this node
        #: re-subscribes; acting on the redelivery means picking an item that is
        #: already in the container.
        self.gate = RunGate()
        self.commands_received = 0
        self.commands_rejected = 0

        self.publisher = self.create_publisher(
            String, T.ISAAC_FEEDBACK, qos_for(T.ISAAC_FEEDBACK))
        self.create_subscription(String, T.ISAAC_COMMAND, self._handle,
                                 qos_for(T.ISAAC_COMMAND))
        self.get_logger().info(
            f"{LOG_BRIDGE} listening on {T.ISAAC_COMMAND}, reporting on "
            f"{T.ISAAC_FEEDBACK}")

    # -- inbound ------------------------------------------------------------ #

    def _handle(self, msg: String) -> None:
        try:
            command = IsaacCommand.from_json(msg.data or "")
        except ContractError as exc:
            # Loud, not swallowed: a malformed command is either a version skew
            # or another publisher on the topic, and both need a human.
            self.commands_rejected += 1
            self.get_logger().error(f"{LOG_BRIDGE} rejected command: {exc}")
            return
        self.commands_received += 1
        self.get_logger().info(
            f"{LOG_BRIDGE} <- {command.command.value}"
            + (f" {command.item_id} #{command.sequence_index}"
               if command.item_id else f" run={command.run_id}"))
        self._on_command(command)

    # -- outbound ----------------------------------------------------------- #

    def publish(self, state: IsaacState, run_id: str, *,
                item_id: Optional[str] = None, sequence_index: int = -1,
                container_id: Optional[str] = None,
                scenario_revision: int = 0,
                scene=None,
                dimensions=None, source_pose: Optional[Pose] = None,
                target_pose: Optional[Pose] = None,
                actual_pose: Optional[Pose] = None,
                position_error_mm: Optional[float] = None,
                message: str = "",
                detail: Optional[Dict[str, Any]] = None) -> IsaacFeedback:
        """Publish one physical-execution report and return it."""
        feedback = IsaacFeedback(
            state=state, run_id=run_id, item_id=item_id,
            sequence_index=sequence_index, container_id=container_id,
            scenario_revision=scenario_revision, scene=scene,
            dimensions=dimensions, source_pose=source_pose,
            target_pose=target_pose, actual_pose=actual_pose,
            position_error_mm=position_error_mm, message=message,
            detail=dict(detail or {}))
        with self._lock:
            self.publisher.publish(String(data=feedback.to_json()))
        self.get_logger().info(
            f"{LOG_BRIDGE} -> {state.value}"
            + (f" {item_id}" if item_id else "")
            + (f" ({message})" if message else ""))
        return feedback

    def spin_once(self) -> None:
        """Service callbacks without blocking the render loop.

        ``timeout_sec=0`` on purpose: this is called once per simulation frame
        and must never stall the simulator waiting for a message that may be
        minutes away (an operator studying a plan before approving it).
        """
        rclpy.spin_once(self, timeout_sec=0.0)


def init_ros() -> None:
    if not rclpy.ok():
        rclpy.init()


def shutdown_ros(node: Optional[Node]) -> None:
    if node is not None:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


__all__ = ["IsaacRosBridge", "init_ros", "shutdown_ros"]
