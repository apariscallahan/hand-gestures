"""Webcam capture on a background thread that always exposes the newest frame.

Reading frames on their own thread means a slow processing step never works
through a backlog of stale frames, which keeps gesture latency low.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)

BACKENDS = {
    # Media Foundation gets the full 30 fps from webcams that DirectShow only
    # runs at 15 fps; DirectShow is the fallback for cameras MSMF can't open.
    "msmf": (cv2.CAP_MSMF, cv2.CAP_DSHOW),
    "dshow": (cv2.CAP_DSHOW, cv2.CAP_MSMF),
    "any": (cv2.CAP_ANY,),
}


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30,
                 backend: str = "msmf") -> None:
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.backend = backend
        self.size = (width, height)
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._seq = 0
        self._running = False
        self._error: str | None = None

    def open(self) -> None:
        for api in BACKENDS[self.backend]:
            cap = cv2.VideoCapture(self.index, api)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if ok and frame is not None:
                self._cap = cap
                self.size = (frame.shape[1], frame.shape[0])
                log.info("Camera %d open: %dx%d via %s", self.index, *self.size, cap.getBackendName())
                break
            cap.release()
        else:
            raise CameraError(
                f"Could not open camera {self.index}. Is it plugged in, or in use by another app "
                f"(Teams, Zoom, the Camera app)? Also check Settings > Privacy & security > Camera."
            )
        self._running = True
        self._error = None
        self._thread = threading.Thread(target=self._reader, name="camera", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        failures = 0
        offset: float | None = None  # camera clock -> time.monotonic()
        last_capture = 0.0  # seconds, on the camera's clock
        while self._running:
            ok, frame = self._cap.read()
            arrived = time.monotonic()
            if not ok or frame is None:
                failures += 1
                if failures > 30:
                    with self._cond:
                        self._error = "The camera stopped sending frames (was it unplugged?)"
                        self._cond.notify_all()
                    return
                time.sleep(0.05)
                continue
            failures = 0

            # Frames can queue up in the driver and then arrive in a burst, so
            # arrival time is a poor record of when a frame was taken - and hand
            # speeds are computed from these times. Media Foundation reports
            # the real capture time; map it onto our clock using the smallest
            # arrival delay seen (allowed to creep up slowly to absorb drift).
            # DirectShow doesn't report it, but it drops frames rather than queueing.
            stamp = arrived
            capture = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if capture <= 0 or capture <= last_capture:
                offset = None  # no capture clock, or it restarted: re-sync on the next frame
            else:
                offset = arrived - capture if offset is None else min(offset + 1e-4, arrived - capture)
                stamp = capture + offset
            last_capture = capture
            with self._cond:
                self._frame, self._stamp = frame, max(stamp, self._stamp + 1e-4)
                self._seq += 1
                self._cond.notify_all()

    def read(self, after: int, timeout: float = 1.0) -> tuple[int, np.ndarray, float] | None:
        """Wait for a frame newer than sequence number `after`.

        Returns (sequence, BGR frame, capture time) or None on timeout.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._seq != after or self._error is not None, timeout)
            if self._error is not None:
                raise CameraError(self._error)
            if self._seq == after:
                return None
            return self._seq, self._frame, self._stamp

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def list_cameras(backend: str = "msmf", max_index: int = 6) -> list[tuple[int, int, int]]:
    """(index, width, height) of each camera that can be opened. Backends can
    number cameras differently, so this uses the one the program will use."""
    found = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, BACKENDS[backend][0])
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                found.append((index, frame.shape[1], frame.shape[0]))
        cap.release()
    return found
