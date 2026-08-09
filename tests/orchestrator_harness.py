"""Run the REAL orchestrator node, with ROS replaced by recording stubs.

WHY THIS EXISTS
---------------
The orchestrator is where the topic contract is produced, and some defects live
in the SEQUENCING of its publishes rather than in the engine underneath it: which
document is written, in which order, stamped with which revision. A test that
drives `WorkflowEngine` directly cannot see any of that — it has no publishers —
and a test that greps the source can only check that a call site exists, not that
the state it produced was coherent.

So this stands up the actual `HitLOrchestrator`, with the actual behaviour tree,
against stub implementations of the three things it needs from the environment:

    rclpy / rclpy.node.Node   parameters, publishers, subscriptions, timers, log
    std_msgs.msg              Bool / Float32 / Int32 / String
    py_trees                  Behaviour, Status, Sequence

Everything else — the engine, the tree, the publish methods, the stamps — is the
production code.

WHAT THE STUBS DELIBERATELY DO NOT DO
-------------------------------------
TIMERS NEVER FIRE. `create_timer` records the callback and returns; the test
drives `_tick()` itself. A test whose outcome depends on when a background timer
happened to run is a test that fails on a loaded machine, and the state
transition under examination is not a timing question.

PUBLISHERS RECORD, THEY DO NOT TRANSPORT. Each publisher keeps every message it
was given, so a test can ask what the LATEST document on a topic says — which is
exactly what a latched topic gives a late-joining dashboard, and therefore
exactly what the dashboard's consistency gate sees.

NOTHING HERE IS A SECOND IMPLEMENTATION OF THE NODE. If a test can be written
against `WorkflowEngine` alone, write it there instead.
"""

from __future__ import annotations

import json
import os
import sys
import types
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for _pkg in ("wisepack_core", "wisepack_bringup", "wisepack_orchestration"):
    _path = os.path.join(REPO, "wisepack_ws", "src", _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)


# --------------------------------------------------------------------------- #
# Recording publisher
# --------------------------------------------------------------------------- #


class RecordingPublisher:
    """Keeps every message published on one topic, newest last."""

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.messages: List[Any] = []

    def publish(self, message: Any) -> None:
        self.messages.append(getattr(message, "data", message))

    # -- what a latched-topic consumer would see ---------------------------- #

    @property
    def last(self) -> Any:
        return self.messages[-1] if self.messages else None

    @property
    def last_json(self) -> Optional[Any]:
        raw = self.last
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    @property
    def count(self) -> int:
        return len(self.messages)


# --------------------------------------------------------------------------- #
# The stub environment
# --------------------------------------------------------------------------- #


#: Module names this harness put into `sys.modules` and must take back out.
#:
#: THEY ARE REMOVED AGAIN AFTER THE IMPORT, and that is not tidiness. Other
#: tests in this suite legitimately do `pytest.importorskip("rclpy")` to skip
#: themselves where ROS 2 is absent; leaving a stub behind would make those
#: tests RUN against it and fail on the first attribute the stub does not have.
#: The orchestrator module has already bound the names it needs by then, so
#: removing the entries costs it nothing.
_INSTALLED: List[str] = []


def _install_stub_modules() -> None:
    """Put stub `rclpy`, `std_msgs` and `py_trees` in `sys.modules`.

    Only installed when the real package is absent: on a machine that HAS ROS 2
    the real modules are used, and this harness then exercises the genuine
    article.
    """
    del _INSTALLED[:]
    if "rclpy" not in sys.modules:
        _install_rclpy()
    if "std_msgs.msg" not in sys.modules:
        _install_std_msgs()
    if "py_trees" not in sys.modules:
        _install_py_trees()


def _remove_stub_modules() -> None:
    for name in reversed(_INSTALLED):
        sys.modules.pop(name, None)
    del _INSTALLED[:]


def _install_rclpy() -> None:
    rclpy = types.ModuleType("rclpy")

    class _Parameter:
        def __init__(self, value: Any) -> None:
            self.value = value

    class _Logger:
        def __init__(self) -> None:
            self.lines: List[str] = []

        def _log(self, level: str, message: str) -> None:
            self.lines.append(f"{level}: {message}")

        def info(self, message): self._log("info", message)
        def warn(self, message): self._log("warn", message)
        def warning(self, message): self._log("warn", message)
        def error(self, message): self._log("error", message)
        def debug(self, message): self._log("debug", message)

    class Node:
        """Enough of `rclpy.node.Node` for the orchestrator to be constructed."""

        #: Parameter values a test wants the node to come up with. Seeded here
        #: rather than assigned after construction because the orchestrator
        #: READS them during `__init__` — which is where the perception source,
        #: the execution backend and the preset are all decided. This is the
        #: stand-in for what the launch file passes.
        preseed: Dict[str, Any] = {}

        def __init__(self, name: str) -> None:
            self._name = name
            self._parameters: Dict[str, Any] = dict(Node.preseed)
            self.publishers_by_topic: Dict[str, RecordingPublisher] = {}
            self.subscriptions_by_topic: Dict[str, Any] = {}
            self.timers: List[Any] = []
            self._logger = _Logger()

        # parameters
        def declare_parameter(self, name: str, value: Any = None):
            self._parameters.setdefault(name, value)
            return _Parameter(self._parameters[name])

        def get_parameter(self, name: str):
            return _Parameter(self._parameters.get(name))

        def set_parameters_for_test(self, **values: Any) -> None:
            self._parameters.update(values)

        # pub/sub
        def create_publisher(self, _msg_type, topic, _qos):
            publisher = RecordingPublisher(topic)
            self.publishers_by_topic[topic] = publisher
            return publisher

        def create_subscription(self, _msg_type, topic, callback, _qos):
            self.subscriptions_by_topic[topic] = callback
            return callback

        # timers NEVER fire — see the module docstring
        def create_timer(self, period, callback):
            timer = types.SimpleNamespace(period=period, callback=callback)
            self.timers.append(timer)
            return timer

        def get_logger(self):
            return self._logger

        def destroy_node(self) -> None:
            pass

    node_module = types.ModuleType("rclpy.node")
    node_module.Node = Node

    duration_module = types.ModuleType("rclpy.duration")

    class Duration:
        def __init__(self, seconds: float = 0.0, nanoseconds: int = 0) -> None:
            self.seconds = seconds
            self.nanoseconds = nanoseconds

    duration_module.Duration = Duration

    qos_module = types.ModuleType("rclpy.qos")

    class _Policy:
        """Stands in for the QoS policy enums. Any attribute is a distinct value."""

        def __init__(self, name: str) -> None:
            self._name = name
            self._values: Dict[str, str] = {}

        def __getattr__(self, item: str) -> str:
            return self._values.setdefault(item, f"{self._name}.{item}")

    class QoSProfile:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    qos_module.QoSProfile = QoSProfile
    for policy in ("DurabilityPolicy", "HistoryPolicy", "LivelinessPolicy",
                   "ReliabilityPolicy"):
        setattr(qos_module, policy, _Policy(policy))

    rclpy.node = node_module
    rclpy.duration = duration_module
    rclpy.qos = qos_module
    rclpy.init = lambda *a, **k: None
    rclpy.shutdown = lambda *a, **k: None
    rclpy.spin = lambda *a, **k: None
    rclpy.ok = lambda: True

    for name, module in (("rclpy", rclpy), ("rclpy.node", node_module),
                         ("rclpy.duration", duration_module),
                         ("rclpy.qos", qos_module)):
        sys.modules[name] = module
        _INSTALLED.append(name)


def _install_std_msgs() -> None:
    std_msgs = sys.modules.get("std_msgs") or types.ModuleType("std_msgs")
    msg_module = types.ModuleType("std_msgs.msg")

    def _message(name: str):
        def __init__(self, data=None):
            self.data = data
        return type(name, (), {"__init__": __init__})

    for name in ("Bool", "Float32", "Int32", "String"):
        setattr(msg_module, name, _message(name))

    std_msgs.msg = msg_module
    for name, module in (("std_msgs", std_msgs), ("std_msgs.msg", msg_module)):
        sys.modules[name] = module
        _INSTALLED.append(name)


def _install_py_trees() -> None:
    py_trees = types.ModuleType("py_trees")

    class Status:
        SUCCESS = "SUCCESS"
        RUNNING = "RUNNING"
        FAILURE = "FAILURE"
        INVALID = "INVALID"

    common = types.ModuleType("py_trees.common")
    common.Status = Status

    behaviour = types.ModuleType("py_trees.behaviour")

    class Behaviour:
        def __init__(self, name: str = "") -> None:
            self.name = name
            self.status = Status.INVALID
            self._initialised = False

        def setup(self, **kwargs): return True
        def initialise(self): pass
        def update(self): return Status.SUCCESS
        def terminate(self, new_status): pass

        def tick_once(self):
            """One tick, with the real initialise/update contract.

            `initialise()` runs when the behaviour was not already RUNNING —
            which is precisely py_trees' rule, and precisely what makes
            `AwaitApproval.initialise` fire on re-entry to the gate.
            """
            if self.status != Status.RUNNING:
                self.initialise()
            self.status = self.update()
            if self.status != Status.RUNNING:
                self.terminate(self.status)
            return self.status

        def setup_with_descendants(self):
            self.setup()

    behaviour.Behaviour = Behaviour

    composites = types.ModuleType("py_trees.composites")

    class Sequence(Behaviour):
        """`memory=True` sequence: resumes at the first non-SUCCESS child."""

        def __init__(self, name: str = "", memory: bool = True) -> None:
            super().__init__(name)
            self.memory = memory
            self.children: List[Behaviour] = []

        def add_children(self, children):
            self.children.extend(children)

        def setup_with_descendants(self):
            for child in self.children:
                child.setup_with_descendants()

        def update(self):
            for child in self.children:
                status = child.tick_once()
                if status != Status.SUCCESS:
                    return status
            return Status.SUCCESS

    composites.Sequence = Sequence

    py_trees.common = common
    py_trees.behaviour = behaviour
    py_trees.composites = composites

    for name, module in (("py_trees", py_trees), ("py_trees.common", common),
                         ("py_trees.behaviour", behaviour),
                         ("py_trees.composites", composites)):
        sys.modules[name] = module
        _INSTALLED.append(name)


# --------------------------------------------------------------------------- #
# The orchestrator under test
# --------------------------------------------------------------------------- #


def orchestrator_module():
    """Import (once) and return the real orchestrator module.

    The stub environment exists only for the duration of the import; see
    `_INSTALLED` for why it is taken back out immediately afterwards.
    """
    _install_stub_modules()
    try:
        from wisepack_orchestration import hitl_orchestrator   # noqa: PLC0415
    finally:
        _remove_stub_modules()
    return hitl_orchestrator


class HarnessedOrchestrator:
    """A constructed `HitLOrchestrator` plus helpers to read what it published."""

    def __init__(self, node, module) -> None:
        self.node = node
        self.module = module

    # -- topics -------------------------------------------------------------- #

    def topic(self, name: str) -> RecordingPublisher:
        publisher = self.node.publishers_by_topic.get(name)
        assert publisher is not None, (
            f"nothing publishes {name!r}; known topics: "
            + ", ".join(sorted(self.node.publishers_by_topic)))
        return publisher

    def latest(self, name: str) -> Optional[Any]:
        """The newest document on a topic — what a latched consumer would read."""
        return self.topic(name).last_json

    def mirror(self) -> Dict[str, Any]:
        """The dashboard's `ros_mirror`, built the way `web/ros_observer.py` does.

        Only the parts the consistency gate reads: the three stamped components
        and the plan-summary/selected fields it cross-checks. Reproducing the
        observer's stamping here (rather than importing it, which needs rclpy
        for real) is what makes this a test of the ORCHESTRATOR's output.
        """
        from wisepack_bringup import topics as T           # noqa: PLC0415

        mirror: Dict[str, Any] = {"stamps": {}}
        for key, topic in (("scenario", T.SCENARIO_STATE),
                           ("selected", T.PLAN_SELECTED),
                           ("plan_summary", T.PLAN_SUMMARY)):
            doc = self.latest(topic)
            if not isinstance(doc, dict):
                continue
            mirror[key] = doc
            mirror["stamps"][key] = {
                "run_id": doc.get("run_id"),
                "scenario_revision": doc.get("scenario_revision"),
                "scenario_id": doc.get("scenario_id"),
            }
        return mirror

    def inconsistency(self) -> str:
        """The dashboard's verdict on what this orchestrator has published.

        THE PRODUCTION GATE, imported and run unchanged — the test must not
        contain its own opinion of what "consistent" means.
        """
        sys.path.insert(0, os.path.join(REPO, "web"))
        from snapshot import _identity_conflict            # noqa: PLC0415
        reason, _identity = _identity_conflict(self.mirror())
        return reason

    # -- driving ------------------------------------------------------------- #

    def tick(self, times: int = 1) -> None:
        for _ in range(times):
            self.node._tick()

    def tick_until_gate(self, limit: int = 40) -> None:
        """Tick until the engine sits at the approval gate, or fail loudly."""
        from wisepack_core.events import Stage             # noqa: PLC0415
        for _ in range(limit):
            self.node._tick()
            if self.node.engine.stage is Stage.WAIT_FOR_OPERATOR_APPROVAL:
                return
        raise AssertionError(
            f"the workflow never reached the approval gate (stage "
            f"{self.node.engine.stage.value})")


def build_orchestrator(*, perception_source: str = "sim",
                       preset: str = "isaac_cylinders_smoke",
                       seed: int = 42,
                       auto_approve: bool = False,
                       env: Optional[Dict[str, str]] = None,
                       monkeypatch=None) -> HarnessedOrchestrator:
    """Construct the real node with the stub environment in place.

    `perception_source` is set through the ENVIRONMENT, exactly as the launcher
    sets it, so the node's own resolution runs rather than being bypassed.
    """
    module = orchestrator_module()

    previous: Dict[str, Optional[str]] = {}
    settings = {"WISEPACK_PERCEPTION_SOURCE": perception_source}
    settings.update(env or {})
    for key, value in settings.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value

    seeded = getattr(module.Node, "preseed", None)
    if seeded is None:
        raise RuntimeError(
            "the real rclpy is installed; this harness only drives the stub "
            "environment. Run these tests without a sourced ROS 2.")
    try:
        module.Node.preseed = {"preset": preset, "seed": seed,
                               "auto_approve": auto_approve,
                               "dynamic_events": False,
                               "pick_failure_probability": 0.0}
        node = module.HitLOrchestrator()
    finally:
        module.Node.preseed = seeded
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return HarnessedOrchestrator(node, module)
