"""WISEPACK camera capture — the one process that opens the device.

ADAPTED FROM HARMONY (`ai-bottle-detector-fiware/camera.py`, MIT). The threaded
grab-newest-frame design is reused because it is the behaviour the detector was
validated against, not because WISEPACK cannot open a `VideoCapture`. The file
lives here, in WISEPACK, and the HARMONY repository is NOT a runtime dependency:
this module imports nothing outside WISEPACK and OpenCV.

WHY A THREAD AND NOT A PLAIN `read()`
-------------------------------------
`cv2.VideoCapture.read()` returns the NEXT frame in the driver's queue, not the
newest one. A service that reads on demand therefore hands the detector a frame
from however long ago the last read was — seconds, on a USB camera left idle
between one-shot detections — and the resulting observation describes a scene
that has already moved. A background thread that keeps draining the queue means
`get_frame()` always returns something recent.

ONE OWNER. `cv2.VideoCapture` does not survive two processes on one device: the
second gets `-1` frames, or the first stops receiving them. The perception
service is the only thing in WISEPACK that constructs this class, and the
dashboard never does.

ADDED HERE, BEYOND THE ORIGINAL: `opened`, `last_error` and a bounded
`wait_for_frame()`. The service reports camera trouble as a HEALTH FIELD rather
than as a traceback, and it cannot do that if the capture object silently keeps
failing in a daemon thread.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

#: How long the capture loop sleeps between grabs. 50 ms ≈ 20 fps, which is
#: ahead of any USB webcam this runs on and cheap enough to leave running while
#: nobody is looking at the preview.
CAPTURE_INTERVAL_S = 0.05

#: How long a consecutive run of failed reads is tolerated before it is reported.
#: A single failed read is normal (the device is still warming up); a continuous
#: minute of them means the camera is gone.
FAILURE_REPORT_AFTER = 20


class Camera:
    """A background frame grabber over `cv2.VideoCapture`.

    NEVER RAISES FROM THE THREAD. A camera that disappears mid-run leaves
    `get_frame()` returning None and `last_error` explaining why, because the
    service's contract is that every perception failure is a reported state.
    """

    def __init__(self, camera: Any, set_resolution: bool = False,
                 width: int = 1600, height: int = 1200) -> None:
        import cv2                                           # noqa: PLC0415

        self.device = camera
        self.last_error: str = ""
        self._cv2 = cv2
        self.cap = cv2.VideoCapture(camera)
        if set_resolution:
            # Best effort by design: a driver that refuses the request keeps its
            # own resolution and the detector still works — the markers just have
            # to be nearer. Failing the whole camera over a rejected format would
            # be a worse outcome than a smaller frame.
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.last_frame = None
        self.lock = threading.Lock()
        self.running = True
        self._failures = 0

        if not self.cap.isOpened():
            # NOT an exception: `camera 3 could not be opened` is a health field
            # the dashboard shows, and a service that died here could not show
            # anything at all.
            self.last_error = (
                f"camera {camera!r} could not be opened — is it connected, and "
                "is no other process (including a second copy of this service) "
                "holding it?")

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    # -- the loop ---------------------------------------------------------- #

    def _capture_loop(self) -> None:
        while self.running:
            try:
                ok, frame = self.cap.read()
            except Exception as exc:                         # noqa: BLE001
                ok, frame = False, None
                self.last_error = f"camera read failed: {exc}"
            if not ok or frame is None:
                self._failures += 1
                if self._failures == FAILURE_REPORT_AFTER and not self.last_error:
                    self.last_error = (
                        f"camera {self.device!r} is open but delivered no frame")
                time.sleep(CAPTURE_INTERVAL_S)
                continue

            self._failures = 0
            self.last_error = ""
            with self.lock:
                self.last_frame = frame

            time.sleep(CAPTURE_INTERVAL_S)

    # -- reading ----------------------------------------------------------- #

    def get_frame(self):
        """The newest frame, COPIED, or None.

        The copy is not paranoia: the caller annotates the frame in place, and
        the capture thread would be overwriting the same array underneath it.
        """
        with self.lock:
            if self.last_frame is None:
                return None
            return self.last_frame.copy()

    def wait_for_frame(self, timeout_s: float = 3.0):
        """The newest frame, waiting up to `timeout_s` for the first one.

        A camera needs a moment after `open()` before it produces anything, and a
        one-shot detection arriving in that window must not be reported as "no
        camera". BOUNDED, so an absent device is an answer rather than a hang.
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        frame = self.get_frame()
        while frame is None and time.monotonic() < deadline:
            time.sleep(CAPTURE_INTERVAL_S)
            frame = self.get_frame()
        return frame

    @property
    def opened(self) -> bool:
        try:
            return bool(self.cap.isOpened())
        except Exception:                                    # noqa: BLE001
            return False

    def release(self) -> None:
        self.running = False
        if self.thread.is_alive():
            # Bounded: a driver wedged inside `read()` must not stop the launcher
            # from exiting, and the process is going away anyway.
            self.thread.join(timeout=2.0)
        try:
            self.cap.release()
        except Exception:                                    # noqa: BLE001
            pass


__all__ = ["Camera", "CAPTURE_INTERVAL_S"]
