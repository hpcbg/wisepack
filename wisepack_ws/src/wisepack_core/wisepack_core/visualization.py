"""How a viewer WATCHES an execution backend — described without naming one.

THE SEPARATION THIS FILE EXISTS TO ENFORCE
------------------------------------------
WISEPACK keeps three concerns apart, and they travel by different routes on
purpose:

    robot execution        who moves the item        (wisepack_core.execution)
    state telemetry        what happened, and when   (ROS 2 topics, NGSI-LD)
    rendered visualization what it LOOKS like        (this descriptor + a stream)

Rendered frames never travel on the telemetry path. Not through
``std_msgs/String``, not through an NGSI-LD attribute, not through the
dashboard's polling API. Those carry state and metadata; a video transport
carries video. Pushing frames through the audit path would bloat the regulatory
record with data that has no audit value and would make the dashboard's poll
loop as slow as the renderer.

WHAT THIS DESCRIPTOR IS
-----------------------
A small, backend-neutral answer to "can I watch this, and how?". The dashboard
consumes ONLY this. It contains no Isaac concept — no extension name, no kit
setting, no USD path — so the Simulator View does not have to know which backend
is running, and a backend that offers no video is a first-class case rather than
an error.

    ``simulated``      -> transport NONE. There is nothing to watch; the
                          simulated robot model has no renderer at all.
    ``isaac``          -> transport WEBRTC (Isaac Sim's livestream), or DESKTOP
                          when Isaac renders to a physical desktop that is
                          viewed by an externally-managed remote-desktop tool.
    a real robot cell  -> WEBRTC from an approved gateway, RTSP, MJPEG, or NONE.

Isaac-specific discovery — which extensions exist, which ports they bind, how
the camera is pinned — lives in ``simulators/isaac``. It produces one of these
and nothing else crosses the boundary.

XR READINESS
------------
Deliberately NOT an XR client, and deliberately not in the way of one. A future
XR viewer needs three things, and all three already exist independently of the
HTML dashboard:

  * the execution-state contracts (``wisepack_core.isaac_contract``) — item ids,
    stages, timestamps;
  * timestamped robot/object poses in NAMED frames with a documented transform
    (``wisepack_core.isaac_transform``) — so a headset can place geometry
    without scraping pixels out of a video stream;
  * this descriptor, to find a spectator stream if it wants one.

So XR support is a new CONSUMER of existing contracts, not a change to them. No
XR dependency belongs in ``wisepack_core``, in the orchestration layer, or in the
Isaac adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class VisualizationTransport(str, Enum):
    """How rendered frames reach a viewer, if at all."""

    #: No visual stream. The honest answer for the simulated backend, for a
    #: headless Isaac run with streaming switched off, and for a real cell with
    #: no approved camera feed. It is a normal state, not a failure.
    NONE = "none"
    #: Isaac Sim's WebRTC livestream, or a real cell's approved WebRTC gateway.
    WEBRTC = "webrtc"
    #: The backend renders to a physical desktop that someone watches with an
    #: externally-managed remote-desktop tool (NoMachine, Sunshine/Moonlight,
    #: VNC). WISEPACK neither starts, configures nor stops those — it only
    #: reports that this is how the run is visible, and where.
    DESKTOP = "desktop"
    #: A camera stream from a real cell, behind an approved gateway.
    RTSP = "rtsp"
    MJPEG = "mjpeg"


class VisualizationStatus(str, Enum):
    """Connection lifecycle, as the dashboard renders it.

    Every one of these is an EXPLICIT state with its own wording. The failure
    this enum exists to prevent is a viewer that shows an empty box or a
    permanent spinner, leaving an operator unable to tell "not offered" from
    "starting" from "broken".
    """

    #: The backend offers no visualization at all.
    UNAVAILABLE = "unavailable"
    #: Offered, but the stream server is not up yet.
    STARTING = "starting"
    #: The endpoint is up and a viewer may connect.
    READY = "ready"
    #: A viewer is attached.
    CONNECTED = "connected"
    #: It was up and is not any more.
    DISCONNECTED = "disconnected"
    #: It was offered and failed. ``message`` says why.
    ERROR = "error"


#: Human wording for each status. Kept beside the enum so the dashboard, the
#: tests and the API all use the same phrases.
STATUS_LABEL: Dict[VisualizationStatus, str] = {
    VisualizationStatus.UNAVAILABLE: "Stream unavailable",
    VisualizationStatus.STARTING: "Stream starting",
    VisualizationStatus.READY: "Ready to connect",
    VisualizationStatus.CONNECTED: "Connected",
    VisualizationStatus.DISCONNECTED: "Disconnected",
    VisualizationStatus.ERROR: "Stream error",
}


@dataclass
class VisualizationDescriptor:
    """What a viewer needs to know, and nothing about how it was produced."""

    backend: str = "simulated"
    available: bool = False
    transport: VisualizationTransport = VisualizationTransport.NONE
    status: VisualizationStatus = VisualizationStatus.UNAVAILABLE
    #: Where a viewer connects. None when there is nothing to connect to.
    viewer_url: Optional[str] = None
    stream_id: Optional[str] = None
    #: Which camera the stream shows. Named so a viewer can say "spectator view
    #: of the workcell" rather than "some viewport".
    camera_name: Optional[str] = None
    #: True when the viewer can drive the view (orbit, zoom). Isaac's livestream
    #: is interactive; a fixed cell camera is not. The dashboard words its call
    #: to action differently for each.
    interactive: bool = False
    #: Whether the stream can be embedded in an iframe. FALSE for Isaac Sim
    #: 6.0.1: the installed livestream package ships no browser client at all
    #: (verified by inspection — no HTML or JS in omni.kit.livestream.*), so the
    #: stream is consumed by NVIDIA's native streaming client. The dashboard
    #: therefore links out instead of rendering an iframe that could only ever
    #: stay blank.
    embeddable: bool = False
    #: Operator-facing explanation. Always set for UNAVAILABLE and ERROR, so the
    #: dashboard never has to invent a reason.
    message: str = ""
    #: How to actually watch it — e.g. which client to use, or the SSH
    #: port-forward to run first. Free text, shown verbatim.
    client_hint: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.transport = VisualizationTransport(self.transport)
        self.status = VisualizationStatus(self.status)
        # `available` and a NONE transport are contradictory; resolve rather than
        # letting a viewer render a connect button that cannot work.
        if self.transport is VisualizationTransport.NONE:
            self.available = False
            if self.status is not VisualizationStatus.ERROR:
                self.status = VisualizationStatus.UNAVAILABLE

    @property
    def status_label(self) -> str:
        return STATUS_LABEL[self.status]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "available": bool(self.available),
            "transport": self.transport.value,
            "status": self.status.value,
            "status_label": self.status_label,
            "viewer_url": self.viewer_url,
            "stream_id": self.stream_id,
            "camera_name": self.camera_name,
            "interactive": bool(self.interactive),
            "embeddable": bool(self.embeddable),
            "message": self.message,
            "client_hint": self.client_hint,
            "detail": dict(self.detail),
        }

    @staticmethod
    def from_dict(doc: Any) -> "VisualizationDescriptor":
        """Parse, tolerating absence and refusing to invent availability.

        Anything unparseable becomes an explicit UNAVAILABLE with a reason,
        because the alternative — raising into a dashboard poll loop — takes out
        the whole page for a panel that is optional by design.
        """
        if not isinstance(doc, dict):
            return unavailable("simulated",
                               "no visualization descriptor was published")
        try:
            transport = VisualizationTransport(doc.get("transport", "none"))
        except ValueError:
            return unavailable(
                str(doc.get("backend", "unknown")),
                f"unknown visualization transport {doc.get('transport')!r}")
        try:
            status = VisualizationStatus(doc.get("status", "unavailable"))
        except ValueError:
            status = VisualizationStatus.ERROR
        return VisualizationDescriptor(
            backend=str(doc.get("backend", "simulated")),
            available=bool(doc.get("available", False)),
            transport=transport,
            status=status,
            viewer_url=doc.get("viewer_url"),
            stream_id=doc.get("stream_id"),
            camera_name=doc.get("camera_name"),
            interactive=bool(doc.get("interactive", False)),
            embeddable=bool(doc.get("embeddable", False)),
            message=str(doc.get("message", "")),
            client_hint=str(doc.get("client_hint", "")),
            detail=dict(doc.get("detail", {}) or {}),
        )


def unavailable(backend: str, message: str) -> VisualizationDescriptor:
    """The explicit "nothing to watch" descriptor.

    Used for the simulated backend, for headless runs with streaming off, and
    for a real cell with no approved feed. All three are normal.
    """
    return VisualizationDescriptor(
        backend=backend, available=False,
        transport=VisualizationTransport.NONE,
        status=VisualizationStatus.UNAVAILABLE, message=message)


__all__ = [
    "VisualizationTransport", "VisualizationStatus", "STATUS_LABEL",
    "VisualizationDescriptor", "unavailable",
]
