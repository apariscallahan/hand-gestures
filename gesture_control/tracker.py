"""Hand tracking with Google's MediaPipe gesture recognizer.

One model gives us, per hand: 21 landmarks in image and world coordinates,
which hand it is, and a classification into a handful of common gestures.
It runs locally on the CPU; nothing leaves the computer.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

from .features import HandObservation

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)
MIN_MODEL_BYTES = 1_000_000


def ensure_model(path: Path) -> Path:
    """Download the model on first run (about 8 MB, from Google's model storage)."""
    if path.exists() and path.stat().st_size >= MIN_MODEL_BYTES:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    log.info("Downloading the hand-tracking model (about 8 MB) - first run only...")
    try:
        urllib.request.urlretrieve(MODEL_URL, partial)
        if partial.stat().st_size < MIN_MODEL_BYTES:
            raise OSError("downloaded file is too small")
        partial.replace(path)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the hand-tracking model: {exc}\n"
            f"Check your internet connection, or download it manually from\n  {MODEL_URL}\n"
            f"and save it as\n  {path}"
        ) from None
    return path


@contextlib.contextmanager
def _quiet_stderr():
    """Hide MediaPipe's C++ start-up chatter, which is written straight to the
    stderr file descriptor. Real failures still surface as Python exceptions."""
    try:
        fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):  # no console (pythonw)
        yield
        return
    sys.stderr.flush()
    saved = os.dup(fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)


class HandTracker:
    """Wraps MediaPipe's GestureRecognizer in video mode (it tracks hands
    between frames, which is faster and steadier than detecting every frame)."""

    def __init__(self, model_path: Path, *, num_hands: int = 1, min_detection: float = 0.6,
                 min_presence: float = 0.5, min_tracking: float = 0.5, mirrored: bool = True) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._mirrored = mirrored
        options = vision.GestureRecognizerOptions(
            # Passing the bytes avoids MediaPipe's trouble with non-ASCII paths on Windows.
            base_options=BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        )
        self._last_ms = -1
        with _quiet_stderr():
            self._recognizer = vision.GestureRecognizer.create_from_options(options)
            self.process(np.zeros((240, 320, 3), np.uint8), 0.0)  # the first run logs too

    def process(self, rgb: np.ndarray, timestamp: float) -> list[HandObservation]:
        """Find hands in an RGB frame taken at `timestamp` (seconds, increasing)."""
        ms = max(int(timestamp * 1000), self._last_ms + 1)  # MediaPipe needs strictly increasing times
        self._last_ms = ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize_for_video(image, ms)

        hands = []
        for i, landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].category_name if result.handedness[i] else ""
            if not self._mirrored:
                # MediaPipe assumes a mirrored selfie view when naming hands.
                handedness = {"Left": "Right", "Right": "Left"}.get(handedness, handedness)
            gesture = result.gestures[i][0] if i < len(result.gestures) and result.gestures[i] else None
            hands.append(HandObservation(
                landmarks=np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float64),
                world=np.array([(p.x, p.y, p.z) for p in result.hand_world_landmarks[i]], dtype=np.float64),
                handedness=handedness,
                gesture=gesture.category_name if gesture else "None",
                gesture_score=float(gesture.score) if gesture else 0.0,
            ))
        return hands

    def close(self) -> None:
        self._recognizer.close()
