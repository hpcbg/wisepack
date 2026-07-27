"""Isaac Sim 6.0.1 WebRTC livestreaming — the ONLY place that knows how.

Everything Isaac-specific about visualization lives here: which extensions to
enable, which settings namespace carries the ports, which camera the stream is
pinned to. What leaves this module is a
``wisepack_core.visualization.VisualizationDescriptor`` and nothing else, so the
dashboard never learns that Isaac exists.

VALIDATED AGAINST THE INSTALLED 6.0.1 PACKAGE, not against an older release:

  * ``omni.kit.livestream.app`` (10.1.1) captures the application framebuffer;
  * ``omni.kit.livestream.webrtc`` (10.3.2) is the WebRTC server it drives;
  * settings live under ``/exts/omni.kit.livestream.app/primaryStream/`` —
    ``signalPort`` (default 49100, TCP, negotiation), ``streamPort`` (default
    47998, UDP, media), ``publicIp``, ``streamType`` and ``targetFps``.

The enable sequence is the one in the shipped standalone example
``standalone_examples/api/isaacsim.simulation_app/livestream.py``: launch
``SimulationApp`` with ``headless=True`` and ``hide_ui=False``, then
``enable_extension("omni.kit.livestream.app")``. Older Isaac releases used a
``omni.services.livestream.webrtc`` extension and a ``/app/livestream/enabled``
setting; NEITHER is present in this install, and writing code against them would
fail at runtime rather than at import.

NO BROWSER CLIENT IS SHIPPED, and that shapes the whole design. There is no HTML
or JavaScript anywhere in the installed ``omni.kit.livestream.*`` extensions —
verified by inspection — because NVIDIA moved to a native "Isaac Sim WebRTC
Streaming Client" application. So the descriptor reports ``embeddable=False``
and the dashboard offers an "Open live simulator" action plus the endpoint,
rather than an iframe that could only ever render blank. That is the
better-integrated option the alternative — proxying frames through a custom
video path — would have traded reliability for.

SECURITY, stated as measured rather than as intended. The stream has no
authentication and no encryption of its own, and — verified on this install —
Kit BINDS THE SIGNAL PORT ON 0.0.0.0, every interface, regardless of what host
WISEPACK advertises. `WISEPACK_ISAAC_STREAM_HOST` therefore controls the URL the
dashboard publishes, NOT the bind address; it cannot restrict who can reach the
port.

So access control is necessarily external, and the honest defaults are: advertise
loopback, never contact an IP-discovery service to learn a public address, and
tell the operator plainly that reaching it from elsewhere means an SSH
port-forward, a firewall rule scoped to one client address, or an authenticated
reverse proxy. See the README.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from wisepack_core.visualization import (
    VisualizationDescriptor, VisualizationStatus, VisualizationTransport,
    unavailable,
)

from .config import LOG_APP

#: The extensions this backend requires, in the installed 6.0.1 package.
REQUIRED_EXTENSIONS: Tuple[str, ...] = (
    "omni.kit.livestream.app",
    "omni.kit.livestream.webrtc",
)

#: Settings path prefix for the primary stream, per the installed extension's
#: own documentation. Stated once so a rename is a one-line change.
PRIMARY_STREAM_SETTING = "/exts/omni.kit.livestream.app/primaryStream"

#: The camera the stream is pinned to. A stable, framed spectator view of the
#: whole workcell — NOT whatever the development viewport happened to be looking
#: at, which on a fresh stage points away from the table entirely.
SPECTATOR_CAMERA = "/World/DemoCamera"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc


@dataclass
class StreamingConfig:
    """Streaming tunables, all overridable, none containing a public address."""

    enabled: bool = False
    #: Interface the operator will point a client at. Loopback by default: the
    #: stream is unauthenticated, so binding it to a routable address must be a
    #: deliberate act, never the default.
    host: str = "127.0.0.1"
    signal_port: int = 49100          # TCP, connection negotiation
    stream_port: int = 47998          # UDP, media
    #: Where the dashboard sends an operator. Derived from host/signal_port
    #: unless explicitly overridden (e.g. behind a reverse proxy).
    viewer_url: str = ""
    #: Optional separate viewer/UI port, for deployments that front the stream
    #: with their own page. 0 = not used.
    viewer_port: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30

    @staticmethod
    def from_env() -> "StreamingConfig":
        cfg = StreamingConfig(
            enabled=_env_flag("WISEPACK_ISAAC_STREAMING", False),
            host=os.environ.get("WISEPACK_ISAAC_STREAM_HOST", "127.0.0.1"),
            signal_port=_env_int("WISEPACK_ISAAC_SIGNAL_PORT", 49100),
            stream_port=_env_int("WISEPACK_ISAAC_STREAM_PORT", 47998),
            viewer_port=_env_int("WISEPACK_ISAAC_VIEWER_PORT", 0),
            viewer_url=os.environ.get("WISEPACK_ISAAC_STREAM_URL", ""),
            target_fps=_env_int("WISEPACK_ISAAC_STREAM_FPS", 30),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        problems = []
        for name, port in (("signal_port", self.signal_port),
                           ("stream_port", self.stream_port)):
            if not 1 <= port <= 65535:
                problems.append(f"{name}={port} is not a valid TCP/UDP port")
        if self.signal_port == self.stream_port:
            problems.append("signal_port and stream_port must differ")
        if problems:
            raise ValueError("Isaac streaming configuration is invalid:\n  - "
                             + "\n  - ".join(problems))

    def resolved_viewer_url(self) -> str:
        """The endpoint an operator points a client at.

        An explicit ``WISEPACK_ISAAC_STREAM_URL`` always wins — that is how a
        reverse proxy or an SSH-forwarded port is expressed. Otherwise it is
        built from the configured host and signal port. The server's public IP is
        never discovered or guessed.
        """
        if self.viewer_url:
            return self.viewer_url
        port = self.viewer_port or self.signal_port
        return f"http://{self.host}:{port}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled, "host": self.host,
            "signal_port": self.signal_port, "stream_port": self.stream_port,
            "viewer_port": self.viewer_port or None,
            "width": self.width, "height": self.height,
            "target_fps": self.target_fps,
        }


def port_is_free(port: int, host: str = "0.0.0.0") -> bool:
    """True when nothing is already listening on ``port`` (TCP).

    Checked BEFORE enabling the stream. Kit's livestream extension will happily
    fall back to "an unoccupied port" when its configured one is taken, which
    means the viewer URL WISEPACK publishes would point at a different, older
    stream server — the worst kind of wrong, because it shows a picture and the
    picture is of something else.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
            return True
        except OSError:
            return False


def missing_extensions(available: Any) -> List[str]:
    """Which required extensions the running Kit does not have.

    ``available`` is a predicate ``name -> bool`` so this stays testable without
    Isaac; the caller passes Kit's own extension-manager lookup.
    """
    return [name for name in REQUIRED_EXTENSIONS if not available(name)]


def describe(config: StreamingConfig, *,
             status: VisualizationStatus = VisualizationStatus.READY,
             message: str = "",
             stream_id: Optional[str] = None,
             extra: Optional[Dict[str, Any]] = None) -> VisualizationDescriptor:
    """Build the backend-neutral descriptor for a configured Isaac stream."""
    if not config.enabled:
        return unavailable(
            "isaac",
            "Isaac Sim is running without WebRTC streaming "
            "(set WISEPACK_ISAAC_STREAMING=1 to enable it)")
    return VisualizationDescriptor(
        backend="isaac",
        available=status in (VisualizationStatus.READY,
                             VisualizationStatus.CONNECTED),
        transport=VisualizationTransport.WEBRTC,
        status=status,
        viewer_url=config.resolved_viewer_url(),
        stream_id=stream_id or f"isaac-primary-{config.signal_port}",
        camera_name=SPECTATOR_CAMERA,
        interactive=True,
        # See the module docstring: the installed package ships no browser
        # client, so an iframe would render nothing.
        embeddable=False,
        message=message,
        client_hint=(
            "Open with the NVIDIA Isaac Sim WebRTC Streaming Client — the "
            "installed Isaac Sim 6.0.1 livestream package ships no in-browser "
            "client (an HTTP GET to the signal port returns 501). For a remote "
            "machine, forward the port over SSH and connect to localhost:\n"
            f'    ssh -p "${{WISEPACK_SSH_PORT}}" '
            f"-L {config.signal_port}:127.0.0.1:{config.signal_port} "
            f"<user>@<host>\n"
            "NOTE: Kit binds this port on ALL interfaces, so restricting access "
            "is a firewall decision — the stream is unauthenticated."),
        detail={**(extra or {}), "stream": config.to_dict(),
                "required_extensions": list(REQUIRED_EXTENSIONS)},
        # Ports are surfaced separately because the operator needs BOTH: the
        # client negotiates on TCP and receives media on UDP, so an SSH TCP
        # tunnel alone cannot carry the video.
        signal_port=config.signal_port,
        media_port=config.stream_port,
        # "the TCP port is listening" is NOT "a client is attached" and is NOT
        # "frames were rendered". None means not reported rather than false.
        client_connected=None,
        frames_verified=False)


def desktop_descriptor(display: str, note: str = "") -> VisualizationDescriptor:
    """Isaac rendering to a physical desktop watched by an external tool.

    NoMachine and Sunshine/Moonlight are EXTERNALLY MANAGED. WISEPACK does not
    install, start, restart, reconfigure or stop them — it only reports that a
    GUI run is visible this way, so the Simulator View can say something true
    instead of "unavailable" when the operator is already watching the desktop.
    """
    return VisualizationDescriptor(
        backend="isaac", available=True,
        transport=VisualizationTransport.DESKTOP,
        status=VisualizationStatus.READY,
        viewer_url=None,
        stream_id=f"isaac-desktop-{display}",
        camera_name=SPECTATOR_CAMERA, interactive=True, embeddable=False,
        message=f"Isaac Sim is rendering to display {display}",
        client_hint=("Watch the host desktop with your existing remote-desktop "
                     "tool (NoMachine, Sunshine/Moonlight, VNC). WISEPACK does "
                     "not manage those services." + (f" {note}" if note else "")),
        detail={"display": display})


__all__ = [
    "REQUIRED_EXTENSIONS", "PRIMARY_STREAM_SETTING", "SPECTATOR_CAMERA",
    "StreamingConfig", "port_is_free", "missing_extensions", "describe",
    "desktop_descriptor",
]
