"""EDF Topic #2 anomaly-integration model — an ARCHITECTURE DEMONSTRATION.

WISEPACK addresses EDF Topic #1 (volume-optimized packaging). This module does
NOT implement or validate a Topic #2 anomaly detector. It demonstrates one thing
and says so everywhere: an independent anomaly-detection module can publish
structured OK/NOK events through ROS 2, the WISEPACK workflow reacts to them
deterministically, and the same event flows through the existing DDS -> FIWARE
analytics layer — all without touching the packing optimizer.

Every event this module produces is labelled ``SIMULATED ANOMALY INTEGRATION
EVENT`` and its ``source`` is ``simulated``. Nothing here claims a validated
detector, a measured detection accuracy, or a physical cutting operation.

There is no ROS in this module. The transport lives in the wisepack_anomaly ROS
package; this file defines the record and the deterministic reaction so the
same behaviour runs in the no-ROS dashboard and in the live stack.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from .domain import Source
from .events import utc_now_iso

#: Shown on every anomaly surface. Not decoration — it is the honesty contract.
#: Domain-neutral by design: the anomaly module is application-independent (see
#: the "Relevance to the JARVIS EDF pilot" README subsection).
SIMULATED_LABEL = "SIMULATED ANOMALY INTEGRATION EVENT"


class Severity(str, Enum):
    """Drives the deterministic workflow reaction. See ``reaction_for``."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AnomalyClass(str, Enum):
    """Compatible anomaly classes for a pipe-cutting skill.

    The names mirror the kind of NOK conditions a real cutting cell reports.
    They are labels for the demonstration; no detector produces them here.
    """

    OPERATION_OK = "operation_ok"
    SHEAR_POSITION_TOO_HIGH = "shear_position_too_high"
    SHEAR_POSITION_TOO_LOW = "shear_position_too_low"
    SHEAR_CLOSED_BEFORE_CONTACT = "shear_closed_before_contact"
    CAMERA_VIEW_LOST = "camera_view_lost"
    TOOL_POSE_DEVIATION = "tool_pose_deviation"


#: Default severity per class. INFO for the healthy case; the rest are chosen to
#: exercise all three reaction paths deterministically in a demo.
_DEFAULT_SEVERITY: Dict[AnomalyClass, Severity] = {
    AnomalyClass.OPERATION_OK: Severity.INFO,
    AnomalyClass.CAMERA_VIEW_LOST: Severity.WARNING,
    AnomalyClass.TOOL_POSE_DEVIATION: Severity.WARNING,
    AnomalyClass.SHEAR_POSITION_TOO_HIGH: Severity.CRITICAL,
    AnomalyClass.SHEAR_POSITION_TOO_LOW: Severity.CRITICAL,
    AnomalyClass.SHEAR_CLOSED_BEFORE_CONTACT: Severity.CRITICAL,
}

#: Recommended operator action per class (advisory text only).
_RECOMMENDED_ACTION: Dict[AnomalyClass, str] = {
    AnomalyClass.OPERATION_OK: "continue",
    AnomalyClass.CAMERA_VIEW_LOST: "pause_and_check_camera",
    AnomalyClass.TOOL_POSE_DEVIATION: "pause_and_verify_tool_pose",
    AnomalyClass.SHEAR_POSITION_TOO_HIGH: "pause_and_reposition",
    AnomalyClass.SHEAR_POSITION_TOO_LOW: "pause_and_reposition",
    AnomalyClass.SHEAR_CLOSED_BEFORE_CONTACT: "hold_and_inspect",
}


def default_severity(cls: AnomalyClass) -> Severity:
    return _DEFAULT_SEVERITY.get(cls, Severity.WARNING)


def recommended_action(cls: AnomalyClass) -> str:
    return _RECOMMENDED_ACTION.get(cls, "review")


class Reaction(str, Enum):
    """What the workflow does in response to a severity."""

    CONTINUE = "continue"       # info: record only
    PAUSE = "pause"             # warning: pause, require acknowledgement
    HOLD = "hold"               # critical: revoke authorisation, require decision


def reaction_for(severity: Severity) -> Reaction:
    if severity is Severity.INFO:
        return Reaction.CONTINUE
    if severity is Severity.WARNING:
        return Reaction.PAUSE
    return Reaction.HOLD


@dataclass
class AnomalyEvent:
    """One structured anomaly, as an external detector would publish it.

    ``source`` is always ``simulated`` in this demonstrator. The dashboard and
    the KPIs treat it as such, and the official Topic #2 detection KPI is never
    marked achieved from these events.
    """

    anomaly_class: AnomalyClass
    status: str = "NOK"                          # OK | NOK
    severity: Severity = Severity.WARNING
    confidence: float = 0.9
    skill: str = "pipe_cutting"
    recommended_action: str = ""
    source_module: str = "simulated_anomaly_detector"
    source: Source = Source.SIMULATED
    event_id: str = ""
    sequence: int = 0
    timestamp: str = field(default_factory=utc_now_iso)
    scenario_id: str = ""
    scenario_revision: int = 0
    cycle_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.anomaly_class = AnomalyClass(self.anomaly_class)
        self.severity = Severity(self.severity)
        self.source = Source(self.source)
        if not self.event_id:
            self.event_id = f"anom-{uuid.uuid4().hex[:10]}"
        if not self.recommended_action:
            self.recommended_action = recommended_action(self.anomaly_class)
        if self.anomaly_class is AnomalyClass.OPERATION_OK:
            self.status = "OK"

    @property
    def reaction(self) -> Reaction:
        return reaction_for(self.severity)

    @property
    def is_ok(self) -> bool:
        return self.status == "OK"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "label": SIMULATED_LABEL,
            "event_id": self.event_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "scenario_id": self.scenario_id,
            "scenario_revision": self.scenario_revision,
            "cycle_id": self.cycle_id,
            "skill": self.skill,
            "status": self.status,
            "anomaly_class": self.anomaly_class.value,
            "severity": self.severity.value,
            "confidence": round(self.confidence, 3),
            "recommended_action": self.recommended_action,
            "reaction": self.reaction.value,
            "source_module": self.source_module,
            "source": self.source.value,
            "details": dict(self.details),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AnomalyEvent":
        return AnomalyEvent(
            anomaly_class=AnomalyClass(d["anomaly_class"]),
            status=d.get("status", "NOK"),
            severity=Severity(d.get("severity",
                                    default_severity(AnomalyClass(d["anomaly_class"])).value)),
            confidence=float(d.get("confidence", 0.9)),
            skill=d.get("skill", "pipe_cutting"),
            recommended_action=d.get("recommended_action", ""),
            source_module=d.get("source_module", "simulated_anomaly_detector"),
            source=Source(d.get("source", "simulated")),
            event_id=d.get("event_id", ""),
            sequence=int(d.get("sequence", 0)),
            timestamp=d.get("timestamp", utc_now_iso()),
            scenario_id=d.get("scenario_id", ""),
            scenario_revision=int(d.get("scenario_revision", 0)),
            cycle_id=d.get("cycle_id", ""),
            details=dict(d.get("details", {})),
        )

    @staticmethod
    def simulate(anomaly_class: str, sequence: int = 0,
                 scenario_id: str = "", scenario_revision: int = 0,
                 cycle_id: str = "", confidence: Optional[float] = None,
                 severity: Optional[str] = None) -> "AnomalyEvent":
        """Build a deterministic simulated event for a class name."""
        cls = AnomalyClass(anomaly_class)
        sev = Severity(severity) if severity else default_severity(cls)
        # Deterministic confidence derived from the class, so a demo is repeatable.
        conf = confidence if confidence is not None else (
            0.99 if cls is AnomalyClass.OPERATION_OK else 0.94)
        return AnomalyEvent(
            anomaly_class=cls, severity=sev, confidence=conf, sequence=sequence,
            scenario_id=scenario_id, scenario_revision=scenario_revision,
            cycle_id=cycle_id, source=Source.SIMULATED,
            details={"note": SIMULATED_LABEL})


__all__ = [
    "SIMULATED_LABEL", "Severity", "AnomalyClass", "Reaction",
    "AnomalyEvent", "reaction_for", "default_severity", "recommended_action",
]
