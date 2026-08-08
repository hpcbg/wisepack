"""Client for the WISEPACK perception service.

Used by BOTH consumers of the host perception service:

  * the dashboard, which renders health and images;
  * `wisepack_orchestration.hitl_orchestrator`, inside the container, which
    fetches batches and republishes them on `/wisepack/perception/*` over the
    validated Vulcanexus / Fast DDS runtime.

Neither opens the camera and neither imports torch. They ask the service
(`perception/perception_service.py`), which is the single camera owner, and use
what comes back. That split is what lets both keep running when the camera is
unplugged, the weights are missing, or the detector process was never started —
§5 requires exactly that.

EVERY METHOD ANSWERS. None of them raise for an unreachable service: a
perception failure is a state to render, not an exception to propagate into a
FastAPI handler that would turn a missing camera into a 500 on the whole page.

Only the standard library is used for HTTP. `requests` is not a dependency of
this repository and adding one so a dashboard can GET a JSON document would be a
poor trade.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from .perception import ObservationBatch, PerceptionHealth, PerceptionSource

#: Where the perception service listens. A DOCUMENTED DEFAULT for the ordinary
#: single-host demo, overridable for a detector on the machine the camera is
#: actually plugged into.
DEFAULT_SERVICE_URL = "http://127.0.0.1:22101"


def service_url() -> str:
    return os.environ.get("WISEPACK_PERCEPTION_SERVICE_URL",
                          DEFAULT_SERVICE_URL).rstrip("/")


class PerceptionClient:
    """Talks to one perception service. Cheap to construct, holds no state."""

    def __init__(self, url: Optional[str] = None, timeout_s: float = 5.0,
                 detect_timeout_s: float = 60.0,
                 health_timeout_s: float = 2.0) -> None:
        self.url = (url or service_url()).rstrip("/")
        self.timeout_s = timeout_s
        # HEALTH IS POLLED, so it must fail FAST. A refused connection returns
        # immediately, but a host that silently drops packets (a firewall, a
        # machine that went away) does not — and blocking the Physical
        # Perception panel for seconds on every refresh would make a dead
        # detector look like a broken dashboard.
        self.health_timeout_s = health_timeout_s
        # A COLD DETECTION IS SLOW AND THAT IS NORMAL. The first request imports
        # torch and loads a 159 MB Faster R-CNN; on CPU that is tens of seconds.
        # A 5 s timeout would report "detector unavailable" for a detector that
        # was working perfectly and merely starting up.
        self.detect_timeout_s = detect_timeout_s

    # -- transport ---------------------------------------------------------- #

    def _request(self, path: str, method: str = "GET",
                 timeout_s: Optional[float] = None) -> Tuple[Optional[int], Any, str]:
        """(status, parsed_json_or_bytes, error). Never raises."""
        request = urllib.request.Request(f"{self.url}{path}", method=method,
                                         headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(
                    request, timeout=timeout_s or self.timeout_s) as response:
                body = response.read()
                content_type = response.headers.get("Content-Type", "")
                if "json" in content_type:
                    return response.status, json.loads(body.decode("utf-8")), ""
                return response.status, body, ""
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:500]
            except Exception:                                # noqa: BLE001
                detail = exc.reason or ""
            return exc.code, None, f"HTTP {exc.code}: {detail}"
        except urllib.error.URLError as exc:
            return None, None, (f"perception service unreachable at {self.url} "
                                f"({exc.reason})")
        except Exception as exc:                             # noqa: BLE001
            return None, None, f"perception service error: {exc}"

    # -- queries ------------------------------------------------------------ #

    def health(self) -> Dict[str, Any]:
        """Every §5 health field, whether or not the service is reachable."""
        status, body, error = self._request("/health",
                                            timeout_s=self.health_timeout_s)
        if status == 200 and isinstance(body, dict):
            body["service_url"] = self.url
            body["service_reachable"] = True
            return body
        # UNREACHABLE IS NOT THE SAME AS BROKEN, and the tri-state fields say so:
        # `camera_available=None` means "not known from here", never "no camera".
        health = PerceptionHealth(
            source=PerceptionSource.CAMERA.value,
            service_url=self.url,
            service_reachable=False,
            last_error=error or "perception service did not answer",
        ).to_dict()
        health["note"] = (
            "The HARMONY perception service is not answering. Start it with "
            "`python3 perception/harmony_perception_service.py`, or point "
            "WISEPACK_PERCEPTION_SERVICE_URL at the host running it.")
        return health

    def detect(self) -> ObservationBatch:
        """Request ONE detection. A failure comes back as a failed batch."""
        status, body, error = self._request("/api/v1/detect", method="POST",
                                            timeout_s=self.detect_timeout_s)
        return self._batch_from(status, body, error)

    def last_detection(self) -> Optional[ObservationBatch]:
        """The service's current batch, or None when it has never detected."""
        status, body, error = self._request("/api/v1/camera/last-detection")
        if status != 200 or not isinstance(body, dict):
            return None
        if body.get("status") == "none":
            return None
        try:
            return ObservationBatch.from_dict(body)
        except Exception:                                    # noqa: BLE001
            return None

    def image(self, kind: str = "annotated") -> Tuple[Optional[bytes], str]:
        """(jpeg_bytes, error). ``kind`` is annotated | raw | snapshot."""
        path = {
            "annotated": "/api/v1/detection/image/annotated",
            "raw": "/api/v1/detection/image/raw",
            "snapshot": "/api/v1/camera/snapshot",
        }.get(kind)
        if path is None:
            return None, f"unknown image kind {kind!r}"
        status, body, error = self._request(path)
        if status == 200 and isinstance(body, (bytes, bytearray)):
            return bytes(body), ""
        return None, error or f"no {kind} image available"

    @property
    def live_url(self) -> str:
        """The raw MJPEG preview, for an <img> tag. Served by the detector."""
        return f"{self.url}/api/v1/camera/live"

    # -- helpers ------------------------------------------------------------ #

    def _batch_from(self, status: Optional[int], body: Any,
                    error: str) -> ObservationBatch:
        if status == 200 and isinstance(body, dict):
            try:
                return ObservationBatch.from_dict(body)
            except Exception as exc:                         # noqa: BLE001
                return ObservationBatch.failed(
                    batch_id="batch-error",
                    source=PerceptionSource.CAMERA.value,
                    error=("the perception service returned a document this "
                           f"build cannot read: {exc}"))
        return ObservationBatch.failed(
            batch_id="batch-error",
            source=PerceptionSource.CAMERA.value,
            error=error or f"perception service returned HTTP {status}")


def make_observation_provider(client: Optional[PerceptionClient] = None):
    """A provider for ``WorkflowEngine.observation_provider``.

    The engine calls this with no arguments and gets a batch — a failed one when
    the detector is unhappy. It never learns that HTTP was involved.
    """
    client = client or PerceptionClient()
    return client.detect


__all__ = ["DEFAULT_SERVICE_URL", "service_url", "PerceptionClient",
           "make_observation_provider"]
