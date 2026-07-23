"""DDS QoS profiles for the WISEPACK topic contract.

Adapted from TEMPO's tempo_bringup/qos.py, which makes the same argument: the
robustness layer is DDS QoS itself, not application-level heartbeat code. These
profiles are constructed in rclpy and passed to every create_publisher and
create_subscription call, so they are genuinely in force on the wire — there is
no XML profile file that nothing loads.

FOUR PROFILES, and the reason each exists:

  event_qos    Action events are the audit trail. RELIABLE with a deep history,
               because a dropped event is a hole in a regulatory record. This is
               the only place a deep queue is justified. Note the split: the
               event STREAM is volatile and deep, while the sequence COUNTER is
               latched state (see qos_for) — "here is an event" and "how many
               events so far" are different kinds of thing.

  state_qos    Latched state (scenario, plans, stage, readiness). RELIABLE +
               TRANSIENT_LOCAL so a dashboard or `ros2 topic echo` that joins
               late sees the CURRENT value immediately instead of waiting for
               the next change. Most WISEPACK state changes only a few times per
               run, so without the latch a late joiner would render an empty UI.

  command_qos  Operator approval and commands. RELIABLE + TRANSIENT_LOCAL, and
               deliberately NO Deadline and NO Liveliness. Transient-local
               matters here because Orion-LD writes an approval by PATCHing an
               attribute, and the orchestrator must see it even if it subscribed
               a moment later.

               WHY NO DEADLINE HERE — this was a real bug, found by running the
               live stack. A subscription that REQUESTS a Deadline only matches
               a publisher that OFFERS a period at least as short. Every generic
               publisher — `ros2 topic pub`, and critically Orion-LD's DDS
               bridge — offers an infinite deadline, so a requested 2 s deadline
               makes them INCOMPATIBLE and the subscription silently never
               matches. rclpy does not raise; the topic simply stays dead. That
               would have broken the entire FIWARE -> ROS operator path while
               every node reported itself healthy.

               A deadline is also wrong here on its own terms: an operator
               taking five minutes to study a plan is the system working, not a
               fault to detect.

  telemetry_qos  High-rate progress and KPI values. BEST_EFFORT — freshness beats
               completeness, and a lost progress sample is replaced 200 ms later.

DEGRADATION. The orchestrator publishes its heartbeat on the execution-state
topic every ORCHESTRATOR_PERIOD_S. Its subscribers request a Deadline; when the
orchestrator stops publishing, the deadline-missed event fires and the consumer
transitions to a paused/degraded state. WISEPACK never simulates unsafe
autonomous continuation, so degraded means HELD, not "carry on".

A DEADLINE IS ONLY DECLARED WHERE IT MEANS SOMETHING. Plans and events are
event-driven — they legitimately go quiet for a minute while an operator thinks —
so putting a deadline on them would produce constant false alarms. Only the
periodic heartbeat and command topics carry one.
"""

from rclpy.duration import Duration
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    LivelinessPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

#: How often the orchestrator republishes its state, even when unchanged.
ORCHESTRATOR_PERIOD_S = 0.5

#: Deadline must exceed the publish period or a healthy loop reports misses.
STATE_DEADLINE_S = 2.0

#: How long a consumer tolerates a silent (or dead) orchestrator before holding.
LIVELINESS_LEASE_S = 4.0

#: Action-event history depth. One full 40-item cycle emits ~230 events; a
#: subscriber that stalls briefly must not lose the middle of the audit trail.
EVENT_HISTORY_DEPTH = 200


def event_qos() -> QoSProfile:
    """Action events: RELIABLE, deep KEEP_LAST. The audit trail must not drop."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=EVENT_HISTORY_DEPTH,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def state_qos() -> QoSProfile:
    """Latched state: RELIABLE, TRANSIENT_LOCAL, KEEP_LAST(1)."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def heartbeat_qos() -> QoSProfile:
    """Execution state: latched AND deadline-monitored.

    This is the topic whose silence means "the orchestrator is gone". It carries
    both the latch (so late joiners see the stage) and the Deadline/Liveliness
    pair (so its absence is detectable rather than merely invisible).
    """
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        deadline=Duration(seconds=STATE_DEADLINE_S),
        liveliness=LivelinessPolicy.AUTOMATIC,
        liveliness_lease_duration=Duration(seconds=LIVELINESS_LEASE_S),
    )


def command_qos() -> QoSProfile:
    """Operator commands: RELIABLE, TRANSIENT_LOCAL, KEEP_LAST(1). No deadline.

    Transient-local is essential rather than convenient: Orion-LD delivers an
    approval by writing a mapped attribute, and the orchestrator must receive it
    even when its subscription matched a moment after the write.

    NO Deadline and NO Liveliness — see the module docstring. Requesting either
    here makes this subscription incompatible with every generic publisher,
    including the Orion-LD DDS bridge, and rclpy reports that by silently
    delivering nothing.
    """
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def telemetry_qos() -> QoSProfile:
    """Progress and KPI values: BEST_EFFORT, KEEP_LAST(5)."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def qos_for(topic: str) -> QoSProfile:
    """Pick the right profile for a canonical topic.

    Nodes call THIS rather than choosing a profile by hand. Requested QoS must be
    compatible with what the publisher offers, and when it is not, rclpy does not
    raise — the subscription simply receives nothing, forever, in silence. TEMPO
    documents being bitten by exactly that; centralising the choice makes the
    mismatch impossible.
    """
    from . import topics as T

    if topic in (T.ACTION_EVENT, T.DYNAMIC_EVENT):
        return event_qos()
    if topic == T.ACTION_SEQUENCE:
        # The sequence COUNTER is state ("how many actions so far"), not an
        # event. Latching it lets a late-joining dashboard, a validation script
        # or Orion-LD read the current count immediately instead of waiting for
        # the next action — which, once a run has finished, never comes.
        return state_qos()
    if topic in (T.OPERATOR_APPROVAL, T.OPERATOR_COMMAND):
        return command_qos()
    if topic == T.EXECUTION_STATE:
        return heartbeat_qos()
    if topic in (T.EXECUTION_PROGRESS_PCT,) or topic.startswith("/wisepack/kpi/"):
        return telemetry_qos()
    return state_qos()
