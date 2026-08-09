"""Detector weights: where WISEPACK keeps them, and how it gets them.

THE WEIGHTS ARE NEVER COMMITTED. They are ~159 MB of binary that changes when
the model is retrained, and a repository is not a model registry. So they are
RESOLVED at start-up, in a documented order, and the answer is reported by
`/health` rather than assumed:

    1. WISEPACK_PERCEPTION_MODEL_PATH      an explicit choice always wins
    2. /data/arise/models/best_model.pth   the shared copy on the ARISE host
    3. <repo>/.cache-perception/models/    the WISEPACK-owned cache
    4. download into (3) from Hugging Face

Steps 1-3 are a filesystem question and live in `wisepack_core.perception`, so
they can be answered before torch is imported at all. Step 4 lives here, because
fetching is a host-side perception concern and the domain package must not grow a
downloader.

NO FOREIGN CHECKOUT IS EVER CONSULTED. An earlier revision searched a HARMONY
clone's `models/` directory; that made another repository's layout part of
WISEPACK's runtime contract. The cache is WISEPACK's own, inside the working
directory, git-ignored, and safe to delete — the next start re-fetches it.

ATTRIBUTION. The published weights are the bottle detector trained in the HARMONY
project (`hpcbg/harmony-bottle-detector` on Hugging Face). WISEPACK uses that
model as-is and says so in `/health` (`model_origin`) and in NOTICE.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Callable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_CORE = os.path.join(_REPO, "wisepack_ws", "src", "wisepack_core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from wisepack_core.perception import (                             # noqa: E402
    HUGGINGFACE_MODEL_URL, HUGGINGFACE_REPO, ModelResolution,
    resolve_model_path,
)

#: The WISEPACK-owned model cache, inside the working directory and git-ignored.
#: A sibling of `.venv-perception/` and equally disposable: deleting it costs one
#: download, never a broken checkout.
DEFAULT_CACHE_DIRNAME = ".cache-perception"

#: The file name the published checkpoint has upstream. Kept identical so a
#: manually placed copy and a downloaded one are interchangeable.
MODEL_FILENAME = "best_model.pth"

#: Below this, whatever arrived is not a Faster R-CNN checkpoint — most likely an
#: HTML error page saved with a 200. Downloading garbage and then failing inside
#: `torch.load` hides which of the two actually went wrong.
MIN_PLAUSIBLE_MODEL_BYTES = 10 * 1024 * 1024


def repo_root() -> str:
    return _REPO


def default_cache_dir(env: Optional[dict] = None) -> str:
    """Where WISEPACK caches detector weights on this host."""
    env = os.environ if env is None else env
    configured = str(env.get("WISEPACK_PERCEPTION_MODEL_CACHE", "") or "").strip()
    if configured:
        return os.path.abspath(configured)
    return os.path.join(_REPO, DEFAULT_CACHE_DIRNAME, "models")


def download_enabled(env: Optional[dict] = None) -> bool:
    """Automatic fetching, on by default.

    Off is a legitimate choice for an air-gapped host, and it must produce the
    resolution diagnostic rather than a network timeout at the first detection.
    """
    env = os.environ if env is None else env
    raw = str(env.get("WISEPACK_PERCEPTION_MODEL_DOWNLOAD", "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def ensure_model(configured: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 env: Optional[dict] = None,
                 allow_download: Optional[bool] = None,
                 log: Optional[Callable[[str], None]] = None) -> ModelResolution:
    """Resolve the weights, downloading into the WISEPACK cache if need be.

    ABSENCE IS A DIAGNOSTIC, NOT A CRASH: a failed download comes back as an
    unavailable `ModelResolution` carrying the reason, so `/health` can show it
    and the camera preview still works while an operator sorts the network out.
    """
    env = os.environ if env is None else env
    cache_dir = cache_dir or default_cache_dir(env)
    log = log or (lambda _message: None)

    resolution = resolve_model_path(configured=configured, cache_dir=cache_dir,
                                    env=env)
    if resolution.available:
        return resolution

    # An EXPLICIT path that does not exist is never papered over with a download:
    # the operator named a file, and quietly loading a different one is worse
    # than reporting the miss.
    explicit = (configured if configured is not None
                else env.get("WISEPACK_PERCEPTION_MODEL_PATH", ""))
    if str(explicit or "").strip():
        return resolution

    if allow_download is None:
        allow_download = download_enabled(env)
    if not allow_download:
        return resolution

    destination = os.path.join(cache_dir, MODEL_FILENAME)
    log(f"[perception] detector weights not found locally — downloading "
        f"{HUGGINGFACE_REPO} (~159 MB) into {destination}")
    error = download_model(destination)
    if error:
        resolution.message = (
            f"{resolution.message}\n\nThe automatic download also failed: "
            f"{error}")
        return resolution

    log("[perception] detector weights downloaded")
    return ModelResolution(destination, "downloaded", True,
                           list(resolution.searched) + [HUGGINGFACE_MODEL_URL])


def download_model(destination: str, url: str = HUGGINGFACE_MODEL_URL,
                   timeout_s: float = 60.0) -> str:
    """Fetch the checkpoint to `destination`. Returns "" or the reason it failed.

    ATOMIC: written to a temporary file in the same directory and renamed, so an
    interrupted download can never leave a half-file that `torch.load` reports as
    a corrupt model on the next twenty start-ups.
    """
    import urllib.error                                      # noqa: PLC0415
    import urllib.request                                    # noqa: PLC0415

    directory = os.path.dirname(os.path.abspath(destination)) or "."
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        return f"the cache directory {directory} could not be created ({exc})"

    handle = None
    temporary = ""
    try:
        fd, temporary = tempfile.mkstemp(prefix=".download-", dir=directory)
        with os.fdopen(fd, "wb") as out:
            with urllib.request.urlopen(url, timeout=timeout_s) as response:
                while True:
                    chunk = response.read(1024 * 512)
                    if not chunk:
                        break
                    out.write(chunk)
        size = os.path.getsize(temporary)
        if size < MIN_PLAUSIBLE_MODEL_BYTES:
            os.unlink(temporary)
            return (f"the download from {url} produced only {size} bytes, which "
                    "is not a model checkpoint — check the URL and any proxy")
        os.replace(temporary, destination)
        return ""
    except (urllib.error.URLError, OSError, ValueError) as exc:
        if temporary and os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass
        return f"{exc}"
    finally:
        if handle is not None:
            handle.close()


def main() -> int:
    """`python perception/model_store.py` — resolve, fetch if needed, report."""
    resolution = ensure_model(log=lambda message: print(message))
    print(f"path      : {resolution.path}")
    print(f"origin    : {resolution.origin}")
    print(f"available : {resolution.available}")
    if not resolution.available:
        print(resolution.message)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CACHE_DIRNAME", "MODEL_FILENAME", "default_cache_dir",
    "download_enabled", "download_model", "ensure_model", "repo_root",
]
