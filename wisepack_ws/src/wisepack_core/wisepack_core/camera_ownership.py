"""Who holds the camera, and what switching perception method would require.

THE PROBLEM THIS EXISTS FOR
---------------------------
`cv2.VideoCapture` and an RGB-D SDK do not survive two owners on one device: the
second gets nothing, or the first stops receiving frames. The planar provider
opens the colour device inside the perception service; a FoundationPose worker
that acquires RGB-D from the same physical camera would be a second owner.

So ownership is MODELLED rather than left to whoever opens the device first.
This module is that model: a small, explicit state machine that answers "who
holds it", "would switching require a handover", and "what would have to happen"
— in terms a dashboard can render and a test can assert.

WHAT IT DELIBERATELY IS NOT
---------------------------
* NOT a process manager. Nothing here kills, signals, greps or matches
  processes, and nothing sleeps waiting for a device to settle. `pkill`,
  `killall`, broad device globbing and "sleep 2 and hope" are all ways of
  guessing that a handover finished; they fail silently on a busy machine and
  they can stop something this project does not own. A handover is performed by
  the owner RELEASING and the acquirer OPENING, each reporting success.
* NOT a claim that any handover has been tested. Every state below is reachable
  and asserted in tests, and the physical transfer between a planar colour
  capture and an RGB-D acquisition has never been performed on any deployment.
  `handover_tested` is False and says so — and it stays False now that a D435 is
  attached, because attaching a camera is not performing a handover.

HONEST STATES BEAT A HOPEFUL API. The useful thing this can do is say precisely
what is and is not known — which is why `SHARED_DEVICE_UNKNOWN` is a state
rather than an assumption in either direction: whether the depth camera's colour
stream can also feed the planar detector has not been established on the one
deployment that now has both, and either answer would be a guess.

NOTHING HERE OPENS A DEVICE. Availability is reported BY ITS OWNER — the planar
provider for colour, the FoundationPose worker for RGB-D — and passed in. A
probe of its own here would be a second opener of the very device this module
exists to keep single-owned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Nothing holds the device.
FREE = "free"
#: A method holds it and is using it.
HELD = "held"
#: The device is not present at all, so ownership is not a question yet.
ABSENT = "absent"

OWNERSHIP_STATES = (FREE, HELD, ABSENT)

#: Whether one physical device serves both methods. THREE-VALUED on purpose.
SHARED_DEVICE_YES = "shared"
SHARED_DEVICE_NO = "separate"
SHARED_DEVICE_UNKNOWN = "unknown"


@dataclass
class CameraOwnership:
    """Who holds which camera, and what a method switch would cost.

    `holder` is a perception-method name or "" — the same vocabulary the
    selector uses, so a dashboard needs no translation table.
    """

    #: The colour device the planar provider uses.
    colour_state: str = ABSENT
    colour_holder: str = ""
    #: The RGB-D device FoundationPose would use.
    depth_state: str = ABSENT
    depth_holder: str = ""
    #: Whether the two are the same physical device. Unknown until one exists.
    shared_device: str = SHARED_DEVICE_UNKNOWN
    #: Why a device is absent or unavailable, per device.
    reasons: Dict[str, str] = field(default_factory=dict)
    #: NEVER SET TRUE BY ANYTHING THAT HAS NOT ACTUALLY DONE IT. No RGB-D
    #: camera is attached, so no handover has been performed, and a UI that
    #: implied otherwise would be claiming a validated capability.
    handover_tested: bool = False

    def __post_init__(self) -> None:
        for name, value in (("colour_state", self.colour_state),
                            ("depth_state", self.depth_state)):
            if value not in OWNERSHIP_STATES:
                raise ValueError(
                    f"{name}={value!r} is not one of {OWNERSHIP_STATES}")
        if self.shared_device not in (SHARED_DEVICE_YES, SHARED_DEVICE_NO,
                                      SHARED_DEVICE_UNKNOWN):
            raise ValueError(f"shared_device={self.shared_device!r} is unknown")
        if self.colour_state != HELD:
            self.colour_holder = ""
        if self.depth_state != HELD:
            self.depth_holder = ""

    # -- the questions a switch has to answer ------------------------------- #

    def requires_handover(self, to_method: str) -> bool:
        """Would switching to `to_method` need the other method to let go?

        ONLY WHEN THE DEVICE IS GENUINELY SHARED. Two separate cameras need no
        handover, and an UNKNOWN sharing relationship is not treated as shared:
        asserting a handover that is not needed would block a switch that would
        have worked, which is the same class of error as performing one that
        was needed and failing silently.
        """
        if self.shared_device != SHARED_DEVICE_YES:
            return False
        holder = self.colour_holder or self.depth_holder
        return bool(holder) and holder != to_method

    def plan_for(self, to_method: str) -> List[str]:
        """The steps a switch to `to_method` would take, in order.

        Returned as text because that is honestly all this is: a description an
        operator can read, not an executed sequence. When the hardware exists
        the steps become calls, each of which reports its own success.
        """
        if not self.requires_handover(to_method):
            return []
        holder = self.colour_holder or self.depth_holder
        return [
            f"{holder} releases the camera and confirms it has closed the device",
            f"{to_method} opens the camera and confirms it has frames",
            "the switch is reported only after that confirmation — never after "
            "a fixed delay",
        ]

    def blocked_reason(self, to_method: str) -> str:
        """Why `to_method` cannot take the camera right now, or ""."""
        needs_depth = to_method == "foundationpose_rgbd"
        state = self.depth_state if needs_depth else self.colour_state
        device = "RGB-D" if needs_depth else "colour"
        if state == ABSENT:
            return (self.reasons.get(device.lower())
                    or f"no {device} camera is attached")
        if self.requires_handover(to_method) and not self.handover_tested:
            holder = self.colour_holder or self.depth_holder
            return (f"the camera is held by {holder} and the handover to "
                    f"{to_method} has never been performed on this deployment — "
                    "it is modelled, not validated")
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "colour": {"state": self.colour_state, "holder": self.colour_holder},
            "depth": {"state": self.depth_state, "holder": self.depth_holder},
            "shared_device": self.shared_device,
            "reasons": dict(self.reasons),
            "handover_tested": self.handover_tested,
            "note": self.note,
        }

    @property
    def note(self) -> str:
        """What this deployment can and cannot claim about ownership.

        THE ABSENCE OF A CAMERA AND THE ABSENCE OF A TESTED HANDOVER ARE TWO
        DIFFERENT FACTS. This note once asserted both at once — it said "no
        RGB-D camera is attached to this deployment" unconditionally — so a
        deployment with a D435 on the bus reported that no RGB-D camera existed
        directly beneath a capability saying one did. Whichever an operator
        believed, the panel was wrong. The device claim now follows
        `depth_state`; the handover claim does not, because no transfer has been
        performed whether or not hardware is present.
        """
        modelled = ("Camera ownership is modelled so two providers cannot both "
                    "open one device.")
        if self.depth_state == ABSENT:
            return (f"{modelled} No RGB-D camera is attached to this "
                    "deployment, so no handover has been performed and none is "
                    "claimed.")
        held = (f" It is held by {self.depth_holder}." if self.depth_holder
                else " Nothing is holding it.")
        return (f"{modelled} An RGB-D camera is attached.{held} Nothing has "
                "been transferred between the planar and RGB-D methods on this "
                "deployment, so no handover has been performed and none is "
                "claimed.")


def current_ownership(colour_available: bool = False,
                      depth_available: bool = False,
                      colour_holder: str = "",
                      depth_holder: str = "",
                      reasons: Optional[Dict[str, str]] = None
                      ) -> CameraOwnership:
    """Ownership as it stands, from what the capability probes reported."""
    document = dict(reasons or {})
    if not depth_available:
        document.setdefault(
            "rgb-d", "no RGB-D camera is attached to this host")
    return CameraOwnership(
        colour_state=(HELD if colour_available and colour_holder
                      else FREE if colour_available else ABSENT),
        colour_holder=colour_holder,
        depth_state=(HELD if depth_available and depth_holder
                     else FREE if depth_available else ABSENT),
        depth_holder=depth_holder,
        # UNKNOWN UNTIL A DEPTH CAMERA EXISTS. Whether its colour stream can
        # also feed the planar detector is a property of a device nobody has
        # plugged in, and either assumption would be a guess.
        shared_device=SHARED_DEVICE_UNKNOWN,
        reasons=document)


__all__ = ["CameraOwnership", "current_ownership", "FREE", "HELD", "ABSENT",
           "OWNERSHIP_STATES", "SHARED_DEVICE_YES", "SHARED_DEVICE_NO",
           "SHARED_DEVICE_UNKNOWN"]
