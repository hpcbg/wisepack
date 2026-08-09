"""Client for the WISEPACK-managed FoundationPose worker.

The same shape, and the same rules, as `perception_client.py`:

  * only the standard library — no `requests`, no torch, no OpenCV;
  * EVERY METHOD ANSWERS. None raise for an unreachable worker, because a
    missing GPU container is a state to render, not an exception to propagate
    into a FastAPI handler that would turn it into a 500 on the whole page;
  * capability is a LIVE question, asked repeatedly. An operator can start the
    worker at any moment and the dashboard must notice without a restart.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not interpret a pose. It fetches JSON and reports reachability. Turning
a worker response into a `PhysicalObservation` — validating the quaternion,
applying the declared symmetry, deciding what was actually measured — is the
PROVIDER's job, in `perception/providers/foundationpose_rgbd.py`. Keeping the
transport ignorant of the domain is what lets the dashboard poll health without
loading the object registry.

Nothing FoundationPose-specific escapes past the provider: no matrices, no
tensors, no masks, no CUDA objects. This module's vocabulary is HTTP and dicts.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

#: Where the worker listens. Bound to loopback by
#: `scripts/setup_foundationpose.sh`; overridable for a worker on the machine the
#: depth camera is actually plugged into.
DEFAULT_WORKER_URL = "http://127.0.0.1:22201"

WORKER_URL_ENV = "WISEPACK_FOUNDATIONPOSE_URL"

#: The capability fields the worker reports, in the order an operator needs
#: them: each is a separate prerequisite with its own fix, and `inference_ready`
#: is their conjunction. Collapsing them into one boolean produces the dashboard
#: that says "unavailable" while nobody can tell whether to plug in a GPU,
#: rebuild an image, or wait for a download.
CAPABILITY_FIELDS = (
    "worker_reachable",
    "worker_ready",
    "gpu_available",
    "foundationpose_runtime_available",
    "scorer_weights_available",
    "refiner_weights_available",
    "rgbd_camera_available",
    "inference_ready",
)


def worker_url() -> str:
    return os.environ.get(WORKER_URL_ENV, DEFAULT_WORKER_URL).rstrip("/")


class FoundationPoseClient:
    """Talks to one FoundationPose worker. Cheap to construct, holds no state."""

    def __init__(self, url: Optional[str] = None,
                 timeout_s: float = 5.0,
                 health_timeout_s: float = 2.0,
                 estimate_timeout_s: float = 300.0) -> None:
        self.url = (url or worker_url()).rstrip("/")
        self.timeout_s = timeout_s
        # HEALTH IS POLLED, so it must fail FAST — a host that silently drops
        # packets would otherwise block the panel for seconds on every refresh
        # and make a dead worker look like a broken dashboard.
        self.health_timeout_s = health_timeout_s
        # A COLD REGISTRATION IS SLOW AND THAT IS NORMAL: the first request
        # builds the estimator, loads two checkpoints and compiles CUDA kernels.
        # The measured bolt registration is ~12 s warm and considerably more
        # cold. A short timeout would report "worker unavailable" for a worker
        # that was working perfectly and merely starting up.
        self.estimate_timeout_s = estimate_timeout_s

    # -- transport ---------------------------------------------------------- #

    def _request(self, path: str, method: str = "GET",
                 payload: Optional[Dict[str, Any]] = None,
                 timeout_s: Optional[float] = None
                 ) -> Tuple[Optional[int], Any, str]:
        """(status, parsed body, error). NEVER raises."""
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(f"{self.url}{path}", data=data,
                                         method=method, headers=headers)
        try:
            with urllib.request.urlopen(
                    request, timeout=timeout_s or self.timeout_s) as response:
                body = response.read()
                if "json" in response.headers.get("Content-Type", ""):
                    return response.status, json.loads(body.decode("utf-8")), ""
                return response.status, body, ""
        except urllib.error.HTTPError as exc:
            detail: Any = ""
            try:
                raw = exc.read().decode("utf-8")
                # THE WORKER'S REFUSALS ARE STRUCTURED and worth keeping: a 409
                # carries `blocked_by`, which is the entire diagnosis. Flattening
                # it to a string here would throw away the reason.
                parsed = json.loads(raw)
                detail = parsed.get("detail", parsed)
            except Exception:                                # noqa: BLE001
                detail = exc.reason or ""
            return exc.code, detail, f"HTTP {exc.code}: {_summarise(detail)}"
        except urllib.error.URLError as exc:
            return None, None, (f"FoundationPose worker unreachable at "
                                f"{self.url} ({exc.reason})")
        except Exception as exc:                             # noqa: BLE001
            return None, None, f"FoundationPose worker error: {exc}"

    # -- queries ------------------------------------------------------------ #

    def health(self) -> Dict[str, Any]:
        """The worker's capability snapshot, whether or not it is reachable.

        UNREACHABLE IS NOT THE SAME AS BROKEN. When nothing answers, every
        capability is reported False with a reason that says the worker did not
        answer — not, for instance, "no GPU", which would send an operator to
        buy hardware they already have.
        """
        status, body, error = self._request("/health",
                                            timeout_s=self.health_timeout_s)
        if status == 200 and isinstance(body, dict):
            document = dict(body)
            document["worker_reachable"] = True
            document["worker_url"] = self.url
            return document
        return {
            "worker_reachable": False,
            "worker_url": self.url,
            "worker_ready": False,
            "gpu_available": False,
            "foundationpose_runtime_available": False,
            "scorer_weights_available": False,
            "refiner_weights_available": False,
            "inference_available": False,
            "last_error": error or "the worker did not answer",
            "blocked_by": [error or "the worker did not answer"],
            "note": (
                "The WISEPACK FoundationPose worker is not answering. It is "
                "OPT-IN and is not started by the ordinary launcher; start it "
                "with `./scripts/setup_foundationpose.sh --run`, or point "
                f"{WORKER_URL_ENV} at the host running it."),
        }

    def capability(self, health: Optional[Dict[str, Any]] = None
                   ) -> Tuple[bool, str]:
        """(FoundationPose can estimate a pose right now, reason if not).

        NEVER raises, and never answers True merely because a container exists.
        """
        document = self.health() if health is None else health
        if not document.get("worker_reachable"):
            return False, str(document.get("last_error")
                              or "the FoundationPose worker did not answer")
        if document.get("inference_available"):
            return True, ""
        blocked = document.get("blocked_by") or []
        if blocked:
            return False, "; ".join(str(b) for b in blocked)
        return False, "the worker reports inference is not available"

    def datasets(self) -> Tuple[List[Dict[str, Any]], str]:
        """The reference datasets the worker can see. ([], reason) when it cannot."""
        status, body, error = self._request("/datasets")
        if status == 200 and isinstance(body, dict):
            return list(body.get("datasets") or []), ""
        return [], error or "the worker did not list its datasets"

    def estimate(self, request: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        """One registration. (result, "") or (None, reason). NEVER raises."""
        status, body, error = self._request(
            "/estimate", method="POST", payload=request,
            timeout_s=self.estimate_timeout_s)
        if status == 200 and isinstance(body, dict):
            return body, ""
        if status == 409:
            # THE REFUSAL IS THE USEFUL ANSWER. 409 is the worker declining
            # because a prerequisite is missing, and `blocked_by` names which.
            return None, _summarise(body) or "inference is not available"
        return None, error or f"the worker refused the estimate (HTTP {status})"

    def last_result(self) -> Tuple[Optional[Dict[str, Any]], str]:
        status, body, error = self._request("/last-result")
        if status == 200 and isinstance(body, dict):
            if body.get("status") == "none":
                return None, str(body.get("message") or "no estimate yet")
            return body, ""
        return None, error or "the worker has no result"

    def image(self, kind: str = "overlay") -> Tuple[Optional[bytes], str]:
        """A diagnostic image as bytes. (None, reason) when there is none."""
        status, body, error = self._request(f"/image/{kind}")
        if status == 200 and isinstance(body, (bytes, bytearray)):
            return bytes(body), ""
        return None, error or f"no {kind} image available"


def _summarise(detail: Any) -> str:
    """A worker refusal, as one readable line, keeping the reasons."""
    if isinstance(detail, dict):
        blocked = detail.get("blocked_by")
        if blocked:
            return "; ".join(str(b) for b in blocked)
        return str(detail.get("error") or detail)
    if isinstance(detail, (list, tuple)):
        return "; ".join(str(d) for d in detail)
    return str(detail or "")


__all__ = ["FoundationPoseClient", "worker_url", "DEFAULT_WORKER_URL",
           "WORKER_URL_ENV", "CAPABILITY_FIELDS"]
