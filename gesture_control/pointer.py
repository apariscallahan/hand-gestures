"""Finger mouse: while the hand points, the index fingertip steers the cursor,
and pressing the thumb in against the hand clicks (keep it pressed to drag).

Pure logic like the gesture engine: it turns per-frame hand features into a
cursor position (as a fraction of the screen) and mouse-button events.
"""

from __future__ import annotations

import math
from collections import deque

from .config import MouseConfig
from .engine import GestureEvent, OneEuroFilter
from .features import EXTENDED_BELOW, HandFeatures, Pose

ARM_DELAY = 0.1  # seconds after pointing starts before the thumb can click
REF_DECAY = 0.3  # palm lengths/s at which "how far out the thumb was" is forgotten
MIN_DROP = 0.2  # palm lengths the thumb must move in to click...
PRESS_DROP = 0.3  # ...or this share of how far out it was, whichever is more
PRESS_GAP = 0.4  # and it must end up this close to the hand (palm lengths)
MIN_RISE, RELEASE_RISE = 0.12, 0.4  # how far back out it must come to let go
PRESS_REWIND = 0.12  # click where the cursor was before the thumb started moving
RELEASE_FREEZE = 0.15  # hold the cursor still briefly after letting go, too
DRAG_START = 0.025  # screen fraction the finger must travel, thumb pressed, to drag
EXIT_REWIND = 0.15  # when pointing stops, undo the last moments of cursor motion
STAY_INDEX_CURL = 0.5  # once pointing, the index may relax this much before it stops


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class PointerTracker:
    def __init__(self, cfg: MouseConfig) -> None:
        self.cfg = cfg
        # Higher smoothing = lower cutoff frequency = steadier but laggier cursor.
        # With simulated fingertip jitter, the default leaves ~2.5 px of shake
        # (from ~14 px raw) and trails a moving finger by ~25 ms.
        min_cutoff = 0.3 + 2.5 * (1.0 - cfg.smoothing)
        self._fx = OneEuroFilter(min_cutoff=min_cutoff, beta=8.0)
        self._fy = OneEuroFilter(min_cutoff=min_cutoff, beta=8.0)
        self.area = (0.0, 0.0, 1.0, 1.0)  # steering area in normalised image coordinates
        self.set_geometry(4 / 3, 16 / 9)
        self.active = False
        self.pressed = False
        self.position: tuple[float, float] | None = None  # cursor, as a fraction of the screen
        self._history: deque[tuple[float, float, float]] = deque()  # recent cursor positions
        self._frozen: tuple[float, float] | None = None
        self._freeze_until = 0.0
        self._press_at = (0.0, 0.0)
        self._ref = 0.0  # how far out the thumb has been recently
        self._low = 0.0  # closest the thumb got during the current press
        self._ready_at = 0.0
        self._last_t = 0.0

    def set_geometry(self, frame_aspect: float, screen_aspect: float) -> None:
        """Give the steering area the screen's shape, so the cursor moves as far
        horizontally as vertically for the same finger movement."""
        w = self.cfg.area_width
        h = min(1.0, w * frame_aspect / screen_aspect)
        cx = _clamp(self.cfg.area_center_x, w / 2, 1 - w / 2)
        cy = _clamp(self.cfg.area_center_y, h / 2, 1 - h / 2)
        self.area = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def update(self, t: float, hand: HandFeatures | None, pose: Pose | None) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        if not self._pointing(hand, pose):
            if self.active:
                self._stop(t, events)
            else:
                self.position = None
            return events
        if not self.active:
            self._start(t, hand)

        x0, y0, x1, y1 = self.area
        tx, ty = hand.tip
        here = (self._fx(t, _clamp((tx - x0) / (x1 - x0), 0.0, 1.0)),
                self._fy(t, _clamp((ty - y0) / (y1 - y0), 0.0, 1.0)))
        if self.cfg.click:
            self._update_thumb(t, hand.thumb_gap, here, events)
        if self._frozen is not None:
            if self.pressed:
                if math.dist(here, self._press_at) > DRAG_START:
                    self._frozen = None  # moving with the thumb pressed: it's a drag
            elif t >= self._freeze_until:
                self._frozen = None
        self.position = self._frozen or here
        self._history.append((t, *self.position))
        while t - self._history[0][0] > 0.5:
            self._history.popleft()
        self._last_t = t
        return events

    def cancel(self, t: float) -> list[GestureEvent]:
        """Stop steering and let go of the button (used when pausing)."""
        events: list[GestureEvent] = []
        if self.active:
            self._stop(t, events)
        self.position = None
        return events

    def _pointing(self, hand: HandFeatures | None, pose: Pose | None) -> bool:
        if hand is None:
            return False
        if pose is Pose.POINTING:
            return True
        # Once steering, tolerate a slightly relaxed hand so the cursor doesn't
        # drop out mid-movement: the index must stay fairly straight and at most
        # one other finger may straighten.
        return (self.active and hand.curls[0] < STAY_INDEX_CURL
                and sum(c < EXTENDED_BELOW for c in hand.curls[1:]) <= 1)

    def _start(self, t: float, hand: HandFeatures) -> None:
        self.active = True
        self._fx.reset()
        self._fy.reset()
        self._history.clear()
        self._frozen = None
        self._ref = hand.thumb_gap
        self._ready_at = t + ARM_DELAY
        self._last_t = t

    def _stop(self, t: float, events: list[GestureEvent]) -> None:
        if self.pressed:
            self.pressed = False
            events.append(GestureEvent("mouse", "up"))
        self.active = False
        self._frozen = None
        # Curling the finger to stop pointing drags the fingertip (and cursor)
        # down for a frame or two before the pose changes: put the cursor back.
        self.position = self._rewind(t - EXIT_REWIND) if self._history else None

    def _rewind(self, when: float) -> tuple[float, float]:
        """Where the cursor was at time `when` (or as far back as remembered)."""
        for ts, u, v in reversed(self._history):
            if ts <= when:
                return u, v
        return self._history[0][1], self._history[0][2]

    def _update_thumb(self, t: float, gap: float, here: tuple[float, float],
                      events: list[GestureEvent]) -> None:
        """The thumb works like a button: pressing it in against the hand is a
        press, letting it spring back out is a release. Judged relative to where
        the thumb has recently been, so it adapts to how each person holds it."""
        if not self.pressed:
            self._ref = max(gap, self._ref - REF_DECAY * max(0.0, t - self._last_t))
            if (t >= self._ready_at and self._history and gap <= PRESS_GAP
                    and self._ref - gap >= max(MIN_DROP, PRESS_DROP * self._ref)):
                self.pressed = True
                self._low = gap
                self._frozen = self._rewind(t - PRESS_REWIND)
                self._press_at = here
                events.append(GestureEvent("mouse", "down"))
        else:
            self._low = min(self._low, gap)
            if gap - self._low >= max(MIN_RISE, RELEASE_RISE * (self._ref - self._low)):
                self.pressed = False
                self._ref = gap
                if self._frozen is None:  # end of a drag: drop where the finger was before letting go
                    self._frozen = self._rewind(t - PRESS_REWIND)
                self._freeze_until = t + RELEASE_FREEZE
                events.append(GestureEvent("mouse", "up"))
