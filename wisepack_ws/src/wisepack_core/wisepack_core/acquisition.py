"""How a physical frame is ACQUIRED — a fourth axis, independent of the rest.

WHY THIS EXISTS. WISEPACK already separates three questions:

    dashboard data source   sim | ros | fiware      where the UI reads
    object source           preset | camera         where objects come from
    perception method       planar | rgbd 6-DoF     how a frame is read

"Camera" turned out to hide a fourth: WHICH camera. A planar USB webcam, a
physical RealSense D435 and a simulated RGB-D camera in Isaac are three
different devices with three different availability answers, and collapsing
them made a working simulated run report "no perception service is answering"
about a webcam it never used — and made a missing RealSense block an inference
that needed no RealSense at all.

    ACQUISITION SOURCE      planar_webcam | realsense_d435 | isaac_simulated

Each answers "can I acquire a frame right now" for itself. Nothing downstream
branches on it: the frame, the mask and the intrinsics reach FoundationPose in
the same generic form whichever produced them. What the axis carries is
AVAILABILITY and PROVENANCE — which of them is running, and whether what it
produced is measured or synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: The planar Faster R-CNN provider's colour webcam, through the perception
#: service. Measured.
ACQUISITION_PLANAR = "planar_webcam"
#: A physical Intel RealSense D435, owned by the FoundationPose worker. Measured.
ACQUISITION_REALSENSE = "realsense_d435"
#: A simulated RGB-D camera rendered by Isaac Sim. SYNTHETIC, and labelled so
#: everywhere it appears — it is not a measurement of the world.
ACQUISITION_ISAAC = "isaac_simulated"

ACQUISITION_SOURCES = (ACQUISITION_PLANAR, ACQUISITION_REALSENSE,
                       ACQUISITION_ISAAC)

#: Which acquisition sources can feed which perception method. A planar detector
#: cannot use depth; FoundationPose cannot work without it.
METHOD_ACQUISITIONS = {
    "planar_fasterrcnn": (ACQUISITION_PLANAR,),
    "foundationpose_rgbd": (ACQUISITION_REALSENSE, ACQUISITION_ISAAC),
}

_LABELS = {
    ACQUISITION_PLANAR: "Planar webcam",
    ACQUISITION_REALSENSE: "Intel RealSense D435",
    ACQUISITION_ISAAC: "Isaac Sim (simulated D435-compatible RGB-D)",
}

#: `measured` or `simulated`, and nothing else. ONE WORD PER SIDE, because two
#: words for the same fact is how a panel ends up saying "synthetic" where a
#: report says "simulated" and a reader has to decide whether they mean the same
#: thing. `synthetic` survives where it describes the CONTENT of a frame — the
#: mask a renderer knew rather than measured — which is a different claim from
#: where the acquisition came from.
_PROVENANCE = {
    ACQUISITION_PLANAR: "measured",
    ACQUISITION_REALSENSE: "measured",
    ACQUISITION_ISAAC: "simulated",
}

_DETAILS = {
    ACQUISITION_PLANAR: ("a colour frame from the USB webcam the perception "
                         "service owns"),
    ACQUISITION_REALSENSE: ("colour and aligned depth from the physical D435, "
                            "read by the FoundationPose worker"),
    ACQUISITION_ISAAC: ("colour and depth rendered by Isaac Sim through a "
                        "camera built to the D435's published geometry. "
                        "SYNTHETIC — no sensor noise and no real materials."),
}


def acquisition_label(source: str) -> str:
    return _LABELS.get(source, source)


def acquisition_provenance(source: str) -> str:
    """`measured` or `synthetic`. Never inferred from context at the call site."""
    return _PROVENANCE.get(source, "unknown")


def acquisition_detail(source: str) -> str:
    return _DETAILS.get(source, "")


def acquisitions_for(method: str) -> tuple:
    return METHOD_ACQUISITIONS.get(method, ())


@dataclass
class AcquisitionState:
    """available / selected / current — the same three states as everything else.

    `current` is the source that produced the batch ON SCREEN. It is empty when
    no physical batch exists, which is not the same as "planar": a preset run
    acquired nothing, and naming a source would invent a provenance.
    """

    current: str = ""
    selected: str = ""
    available: List[str] = field(default_factory=list)
    #: Why each unavailable source is unavailable. A source offered with no
    #: reason is a dead end for the operator.
    unavailable_reasons: Dict[str, str] = field(default_factory=dict)

    def is_available(self, source: str) -> bool:
        return source in self.available

    @property
    def any_camera_available(self) -> bool:
        """Whether ANY camera can acquire — what the object-source selector
        needs. A missing webcam must not veto a simulated run, and a missing
        RealSense must not veto either of the others."""
        return bool(self.available)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current": self.current,
            "current_label": (acquisition_label(self.current)
                              if self.current else ""),
            "current_provenance": (acquisition_provenance(self.current)
                                   if self.current else ""),
            "selected": self.selected,
            "selected_label": (acquisition_label(self.selected)
                               if self.selected else ""),
            "available": list(self.available),
            "unavailable_reasons": dict(self.unavailable_reasons),
            "options": [
                {"value": source,
                 "label": acquisition_label(source),
                 "detail": acquisition_detail(source),
                 "provenance": acquisition_provenance(source),
                 "available": source in self.available,
                 "reason": self.unavailable_reasons.get(source, "")}
                for source in ACQUISITION_SOURCES
            ],
        }


__all__ = ["ACQUISITION_PLANAR", "ACQUISITION_REALSENSE", "ACQUISITION_ISAAC",
           "ACQUISITION_SOURCES", "METHOD_ACQUISITIONS", "AcquisitionState",
           "acquisition_label", "acquisition_provenance", "acquisition_detail",
           "acquisitions_for"]
