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

  telemetry_qos  High-rate progress ONLY. BEST_EFFORT — freshness beats
               completeness, and a lost progress sample is replaced shortly.
               KPI topics deliberately do NOT use this: they are latched state.

LATE ATTACHMENT IS THE NORMAL CASE. The dashboard connects after the
orchestrator has already planned, so anything it must render has to be
TRANSIENT_LOCAL. Both the KPI topics and the action-event stream were VOLATILE
and produced an empty dashboard against a healthy stack; both are latched now.

DEGRADATION. The orchestrator publishes `/wisepack/system/heartbeat` every
ORCHESTRATOR_PERIOD_S. Its subscribers request a Deadline; when the
orchestrator stops publishing, the deadline-missed event fires and the consumer
transitions to a paused/degraded state. WISEPACK never simulates unsafe
autonomous continuation, so degraded means HELD, not "carry on".

A DEADLINE IS ONLY DECLARED WHERE IT MEANS SOMETHING. Plans and events are
event-driven — they legitimately go quiet for a minute while an operator thinks —
so putting a deadline on them would produce constant false alarms. Only the
periodic heartbeat topic carries one; the command topics deliberately do not
(see command_qos).
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

#: Application-level staleness threshold for the heartbeat counter. A consumer
#: that has seen no new heartbeat for this long treats the orchestrator as lost.
#: This is what actually detects a dead orchestrator — see
#: watchdog_subscribe_qos() for why the DDS-level route is unavailable here.
HEARTBEAT_STALE_S = 6.0

#: Action-event history depth. One full 40-item cycle emits ~230 events; a
#: subscriber that stalls briefly must not lose the middle of the audit trail.
EVENT_HISTORY_DEPTH = 200


def event_qos() -> QoSProfile:
    """Action events: RELIABLE, TRANSIENT_LOCAL, deep KEEP_LAST.

    TRANSIENT_LOCAL matters as much as RELIABLE here. The dashboard attaches
    AFTER the orchestrator has planned — that is the normal case, not an edge
    case — and with VOLATILE durability it received an empty timeline while the
    orchestrator had already emitted the whole planning sequence. Measured:
    `/api/events` returned 0 events against an action count of 7.

    Retaining the last EVENT_HISTORY_DEPTH events for late joiners is also the
    right semantics for an audit trail: a regulatory record that only exists for
    whoever happened to be listening is not a record.
    """
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=EVENT_HISTORY_DEPTH,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
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
    """`/wisepack/system/heartbeat` ONLY: latched AND deadline-monitored.

    This is the topic whose silence means "the orchestrator is gone". It is the
    single place a Deadline/Liveliness pair is declared, and it is deliberately
    NOT bridged to FIWARE.

    DO NOT apply this profile to a topic anything else might publish on. A
    subscription requesting a liveliness lease is incompatible with any
    publisher offering an infinite one — which is every generic publisher,
    Orion-LD's DDS bridge included — and rclpy reports that by silently
    receiving nothing. `/wisepack/execution/state` used this profile and the
    live dashboard rendered empty as a direct result.
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


def watchdog_subscribe_qos() -> QoSProfile:
    """SUBSCRIBER side of the heartbeat: latched state, NO liveliness request.

    Asymmetric on purpose, and the asymmetry is the whole point.

    The orchestrator OFFERS a Deadline and a Liveliness lease on the heartbeat
    (see heartbeat_qos). Offering is free and well-defined. REQUESTING them is
    not: a reader only matches a writer whose offered lease is at least as
    short, and Orion-LD's DDS enabler creates a bare-DDS publisher on every
    topic it discovers — not merely the mapped ones — each offering an INFINITE
    lease. Measured on this machine: with the enabler running, a subscription
    requesting a 4 s lease logs

        New publisher discovered on topic '/wisepack/system/heartbeat',
        offering incompatible QoS. No messages will be received from it.
        Last incompatible policy: LIVELINESS

    ...even after the state topic was split out. So DDS-level liveliness
    detection is simply unavailable to a subscriber in this deployment.

    Detection is therefore done where it still works: the heartbeat carries a
    monotonically increasing counter, and a consumer that sees it stop advancing
    for longer than HEARTBEAT_STALE_S declares the orchestrator lost. That is
    application-level, it is honest about being so, and unlike the QoS route it
    actually fires.
    """
    return state_qos()


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
    """High-rate progress only: BEST_EFFORT, KEEP_LAST(5).

    NOT for KPI values. A KPI changes a handful of times per run and a late
    subscriber must see the current one; BEST_EFFORT + VOLATILE gave the live
    dashboard a full set of empty KPI tiles because every value had been
    published before it connected. KPIs are latched state — see qos_for.
    """
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
    if topic in (T.OPERATOR_APPROVAL, T.OPERATOR_COMMAND,
                 T.CUTTING_APPROVAL, T.INVENTORY_REQUEST):
        return command_qos()
    if topic in T.CORRELATION_TOPICS.values():
        # State, and latched: which run a projection describes must be readable
        # by a consumer that attached long after the run started. A correlation
        # that arrives only on the next change would leave every entity
        # unverifiable — and therefore withheld — until something moved.
        return state_qos()
    if topic == T.ISAAC_COMMAND:
        # A physical instruction must not be lost, and Isaac Sim takes ~30 s to
        # boot — far longer than the orchestrator takes to publish RUN_BEGIN. So
        # this is latched: the simulator receives the current command whenever it
        # finishes subscribing, rather than missing it and waiting forever.
        #
        # Latching is exactly why the receiving side must de-duplicate. A
        # TRANSIENT_LOCAL command is redelivered on every re-subscribe, and
        # acting on a redelivery means picking an item that is already in the
        # container. wisepack_core.isaac_contract.RunGate is that guard, and it
        # is used on BOTH ends.
        return command_qos()
    if topic == T.ISAAC_FEEDBACK:
        # Physical execution feedback is an event STREAM, not a current value:
        # a run emits several states per item and the orchestrator advances on
        # the terminal one. Losing an ITEM_COMPLETED would strand the run, so
        # RELIABLE with a deep history, same reasoning as the audit trail.
        return event_qos()
    if topic == T.SYSTEM_HEARTBEAT:
        # NOTE: this is the SUBSCRIBER-safe profile. The orchestrator publishes
        # the heartbeat with heartbeat_qos() explicitly, so it still OFFERS the
        # Deadline/Liveliness pair.
        return watchdog_subscribe_qos()
    if topic == T.EXECUTION_PROGRESS_PCT:
        # Genuinely high-rate: a lost sample is replaced 700 ms later.
        return telemetry_qos()
    if topic in (T.PERCEPTION_OBJECTS, T.PERCEPTION_STATUS):
        # The current physical observation is STATE, not a stream: there is
        # exactly one current batch and a re-detection replaces it. Latched, so
        # an orchestrator or dashboard that attaches after the operator ran a
        # detection sees the objects that are on the table rather than waiting
        # for the next scan that may never come.
        return state_qos()
    if topic == T.HARMONY_DETECTION_COMMAND:
        # HARMONY's inbound trigger. Deliberately NOT latched: a transient-local
        # START would be redelivered every time the detector re-subscribed, and
        # each redelivery would run an unrequested inference.
        return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                          reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.VOLATILE)
    if topic.startswith("/wisepack/kpi/"):
        # KPIs are STATE, not telemetry. Latched, so a dashboard attaching
        # mid-run renders the current values instead of "not measured".
        return state_qos()
    return state_qos()
