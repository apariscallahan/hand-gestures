"""Gestures over time: swipes, held poses and the fist-grab app switcher.

The engine is pure logic. It is fed one HandFeatures (or None when no hand
is visible) per camera frame, with a timestamp in seconds, and returns the
GestureEvents that frame completed. It knows nothing about cameras or
Windows, so it can be tested with synthetic hand movements.

Distances are measured in palm lengths where it matters, so gestures feel
the same whether you sit close to the camera or further away.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .config import TuningConfig
from .features import HandFeatures, Pose

LOST_RESET = 0.3  # hand gone this long: its return counts as a new hand
JUMP_DISTANCE = 0.35  # frame heights in a single frame: the tracker switched hands
INTENT_WINDOW = 1.5  # held gestures must begin within this long of an open hand
HOLD_MAX_MOTION = 0.6  # palm lengths the hand may drift while holding a pose
COOLDOWN_AFTER_SWITCHER = 0.5
SWITCHER_LOST_CANCEL = 0.8  # hand gone this long while switching: cancel
SWITCHER_TIMEOUT = 20.0
STEP_HYSTERESIS = 0.15  # in steps; stops the selection flickering at a boundary
RELEASE_TIME = 0.15  # an open hand held this long picks the app...
RELAXED_RELEASE_TIME = 0.6  # ...as does any other non-fist pose held this long

# Swipe detection
ARM_TIME = 0.3  # a newly seen hand must be tracked this long...
SETTLE_TIME = 0.12  # ...and be still this long before it can swipe
VELOCITY_WINDOW = 0.07
STROKE_LOOKBACK = 0.1  # a stroke is noticed a little after it really starts
MIN_SWIPE, MAX_SWIPE = 0.12, 0.5  # frame heights
AXIS_RATIO = 1.5  # travel along the main axis vs across it
OPEN_FRACTION = 0.5  # share of a stroke's frames the hand must be open in
OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class GestureEvent:
    kind: str  # "swipe", "hold" or "switcher"
    name: str  # e.g. "swipe_left", "pinch_hold"; for the switcher "open", "step", "commit", "cancel"
    step: int = 0  # +1 / -1 for switcher steps


class OneEuroFilter:
    """The "1 euro" filter (Casiez et al., 2012): smooths jitter when the hand
    is still without adding noticeable lag when it moves."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 3.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None
        self._dx = 0.0
        self._t = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x: float) -> float:
        if self._x is None:
            self._x, self._t = x, t
            return x
        dt = t - self._t
        if dt <= 0:
            return self._x
        self._t = t
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * (x - self._x) / dt + (1 - a_d) * self._dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self._dx), dt)
        self._x = a * x + (1 - a) * self._x
        return self._x


class PoseFilter:
    """Debounces the per-frame pose: a new pose must persist before it counts."""

    def __init__(self, confirm_time: float) -> None:
        self.confirm_time = confirm_time
        self.reset()

    def reset(self) -> None:
        self.stable: Pose | None = None
        self._candidate: Pose | None = None
        self._since = 0.0

    def update(self, t: float, raw: Pose) -> Pose | None:
        if raw != self._candidate:
            self._candidate, self._since = raw, t
        if self._candidate != self.stable and t - self._since >= self.confirm_time:
            self.stable = self._candidate
        return self.stable


@dataclass
class _Stroke:
    t0: float
    x0: float
    y0: float
    frames: int = 0
    open_frames: int = 0
    spent: bool = False  # already judged; ignore until the hand slows down


class SwipeDetector:
    """Finds quick, mostly straight movements of an open hand.

    A stroke starts when the hand speeds up past `swipe_speed` and ends when it
    slows to half of that. It becomes a swipe the moment it has travelled
    `swipe_distance` along one axis. Afterwards the hand has to come to rest
    before the next swipe, and the opposite direction stays blocked for a
    moment, so bringing the hand back doesn't count as a swipe of its own.
    """

    def __init__(self, tuning: TuningConfig) -> None:
        self.tuning = tuning
        self.samples: deque[tuple[float, float, float]] = deque()
        self.last: tuple[str, float] | None = None  # (direction, time) of the latest swipe
        self._blocked: str | None = None
        self._blocked_until = 0.0
        self._cooldown_until = 0.0
        self.reset()

    def reset(self) -> None:
        self.samples.clear()
        self.stroke: _Stroke | None = None
        self.armed = False
        self._calm_since: float | None = None

    def update(self, t: float, center: tuple[float, float], unit: float, openish: bool,
               track_age: float) -> str | None:
        """Feed one frame; returns "left"/"right"/"up"/"down" when a swipe completes."""
        x, y = center
        self.samples.append((t, x, y))
        while t - self.samples[0][0] > 0.4:
            self.samples.popleft()
        speed = self._speed(t, x, y)
        v_start = self.tuning.swipe_speed * unit
        v_end = 0.5 * v_start

        if t < self._cooldown_until:
            self.stroke = None
            self._calm_since = None
            return None

        if not self.armed:
            if speed >= v_end:
                self._calm_since = None
                return None
            if self._calm_since is None:
                self._calm_since = t
            self.armed = track_age >= ARM_TIME and t - self._calm_since >= SETTLE_TIME
            return None

        stroke = self.stroke
        if stroke is None:
            if speed < v_start:
                return None
            stroke = self.stroke = _Stroke(*self._sample_at(t - STROKE_LOOKBACK))

        stroke.frames += 1
        stroke.open_frames += openish
        if speed < v_end or t - stroke.t0 > self.tuning.swipe_max_duration:
            self.stroke = None
            return None
        if stroke.spent:
            return None

        dx, dy = x - stroke.x0, y - stroke.y0
        need = _clamp(self.tuning.swipe_distance * unit, MIN_SWIPE, MAX_SWIPE)
        if abs(dx) >= need and abs(dx) >= AXIS_RATIO * abs(dy):
            direction = "right" if dx > 0 else "left"
        elif abs(dy) >= need and abs(dy) >= AXIS_RATIO * abs(dx):
            direction = "down" if dy > 0 else "up"
        else:
            return None

        stroke.spent = True
        if stroke.open_frames < OPEN_FRACTION * stroke.frames:
            return None
        if direction == self._blocked and t < self._blocked_until:
            return None
        self.last = (direction, t)
        self._cooldown_until = t + self.tuning.swipe_cooldown
        self._blocked = OPPOSITE[direction]
        self._blocked_until = t + self.tuning.swipe_return_block
        self.armed = False
        self._calm_since = None
        self.stroke = None
        return direction

    def _speed(self, t: float, x: float, y: float) -> float:
        """Speed over roughly the last VELOCITY_WINDOW seconds (at least one frame)."""
        ref = None
        for sample in reversed(self.samples):
            ref = sample
            if t - sample[0] >= VELOCITY_WINDOW:
                break
        if ref is None or t <= ref[0]:
            return 0.0
        return math.hypot(x - ref[1], y - ref[2]) / (t - ref[0])

    def _sample_at(self, when: float) -> tuple[float, float, float]:
        """The newest sample taken at or before `when` (or the oldest one kept)."""
        for sample in reversed(self.samples):
            if sample[0] <= when:
                return sample
        return self.samples[0]


class GestureEngine:
    """Turns a stream of per-frame hand features into gesture events.

    `hold_times` maps each held pose that has an action (pinch, fist, thumbs
    up, ...) to how many seconds it must be held. With `switcher` on, holding
    a fist opens the Alt+Tab switcher instead of firing a one-shot event.
    """

    def __init__(self, tuning: TuningConfig, hold_times: dict[Pose, float], switcher: bool = True) -> None:
        self.tuning = tuning
        self.hold_times = dict(hold_times)
        self.switcher_enabled = switcher and Pose.FIST in self.hold_times
        self.swipe = SwipeDetector(tuning)
        self._pose_filter = PoseFilter(tuning.pose_confirm_time)
        self._fx, self._fy = OneEuroFilter(), OneEuroFilter()

        self.hand_visible = False
        self.lost_since: float | None = None
        self.palm = 0.14
        self.center = (0.0, 0.0)  # smoothed palm centre
        self.pose: Pose | None = None  # debounced pose
        self._pose_since = 0.0
        self._track_start = 0.0
        self._last_raw: tuple[float, float] | None = None
        self._last_open: float | None = None
        self._cooldown_until = 0.0

        self.hold_pose: Pose | None = None
        self.hold_progress = 0.0
        self.hold_fired = False
        self.hold_blocked = False  # the pose didn't start from an open hand
        self._hold_start = 0.0
        self._hold_anchor = (0.0, 0.0)

        self.switcher_active = False
        self.switcher_steps = 0
        self.switcher_offset = 0.0  # hand position relative to where the fist closed, in steps
        self._switcher_start = 0.0
        self._switcher_anchor = 0.0
        self._release_start: float | None = None

    @property
    def unit(self) -> float:
        """One palm length in frame heights, kept within sane bounds."""
        return _clamp(self.palm, 0.06, 0.25)

    def update(self, t: float, hand: HandFeatures | None) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        if hand is None:
            self._hand_missing(t, events)
            return events

        if not self.hand_visible:
            self.hand_visible = True
            if self.lost_since is None or t - self.lost_since > LOST_RESET:
                self._new_track(t, hand)
            self.lost_since = None
        elif self._last_raw is not None and math.dist(hand.center, self._last_raw) > JUMP_DISTANCE:
            # Nobody moves that fast: the tracker has latched onto a different hand.
            if self.switcher_active:
                self._end_switcher(t, events, "cancel")
            self._new_track(t, hand)
        self._last_raw = hand.center

        self.palm += 0.2 * (hand.palm - self.palm)
        self.center = (self._fx(t, hand.center[0]), self._fy(t, hand.center[1]))
        openish = hand.extended >= 3
        if openish:
            self._last_open = t
        pose = self._pose_filter.update(t, hand.pose)
        if pose != self.pose:
            self.pose, self._pose_since = pose, t

        if self.switcher_active:
            self._update_switcher(t, hand, events)
        elif t >= self._cooldown_until:
            direction = self.swipe.update(t, hand.center, self.unit, openish, t - self._track_start)
            if direction:
                self._clear_hold()
                events.append(GestureEvent("swipe", f"swipe_{direction}"))
            else:
                self._update_hold(t, events)
        return events

    def cancel(self, t: float) -> list[GestureEvent]:
        """Abandon whatever is in progress (used when pausing)."""
        events: list[GestureEvent] = []
        if self.switcher_active:
            self._end_switcher(t, events, "cancel")
        self.hand_visible = False
        self.lost_since = None
        self._reset_tracking()
        return events

    # -- tracking ----------------------------------------------------------

    def _new_track(self, t: float, hand: HandFeatures) -> None:
        self._reset_tracking()
        self._track_start = t
        self.palm = hand.palm

    def _reset_tracking(self) -> None:
        self._fx.reset()
        self._fy.reset()
        self.swipe.reset()
        self._pose_filter.reset()
        self.pose = None
        self._last_raw = None
        self._last_open = None
        self._clear_hold()

    def _hand_missing(self, t: float, events: list[GestureEvent]) -> None:
        if self.hand_visible:
            self.hand_visible = False
            self.lost_since = t
        if self.lost_since is None:
            return
        gone = t - self.lost_since
        if self.switcher_active and gone > SWITCHER_LOST_CANCEL:
            self._end_switcher(t, events, "cancel")
        if gone > LOST_RESET and not self.switcher_active:
            self._reset_tracking()

    # -- held poses --------------------------------------------------------

    def _update_hold(self, t: float, events: list[GestureEvent]) -> None:
        pose = self.pose
        hold_time = self.hold_times.get(pose)
        if hold_time is None:
            self._clear_hold()
            return
        if pose != self.hold_pose:
            self.hold_pose = pose
            self.hold_fired = False
            self.hold_progress = 0.0
            self._hold_start, self._hold_anchor = t, self.center
            # Deliberate gestures start from an open hand. This filters out
            # things like holding a pen (looks like a pinch) or a mug (a fist).
            self.hold_blocked = self.tuning.require_open_hand_first and not (
                self._last_open is not None and self._pose_since - self._last_open <= INTENT_WINDOW
            )
        if self.hold_fired or self.hold_blocked:
            return
        if math.dist(self.center, self._hold_anchor) > HOLD_MAX_MOTION * self.unit:
            self._hold_start, self._hold_anchor = t, self.center  # still moving: start over
        self.hold_progress = min(1.0, (t - self._hold_start) / hold_time)
        if self.hold_progress < 1.0:
            return
        self.hold_fired = True
        if pose is Pose.FIST and self.switcher_enabled:
            self._start_switcher(t, events)
        else:
            events.append(GestureEvent("hold", f"{pose.value}_hold"))

    def _clear_hold(self) -> None:
        self.hold_pose = None
        self.hold_progress = 0.0
        self.hold_fired = False
        self.hold_blocked = False

    # -- app switcher ------------------------------------------------------

    def _start_switcher(self, t: float, events: list[GestureEvent]) -> None:
        self.switcher_active = True
        self.switcher_steps = 0
        self.switcher_offset = 0.0
        self._switcher_start = t
        self._switcher_anchor = self.center[0]
        self._release_start = None
        self.hold_progress = 0.0
        self.swipe.reset()
        events.append(GestureEvent("switcher", "open"))

    def _update_switcher(self, t: float, hand: HandFeatures, events: list[GestureEvent]) -> None:
        step = _clamp(self.tuning.switcher_step * self.unit, 0.03, 0.15)
        self.switcher_offset = (self.center[0] - self._switcher_anchor) / step
        while self.switcher_offset > self.switcher_steps + 0.5 + STEP_HYSTERESIS:
            self.switcher_steps += 1
            events.append(GestureEvent("switcher", "step", +1))
        while self.switcher_offset < self.switcher_steps - 0.5 - STEP_HYSTERESIS:
            self.switcher_steps -= 1
            events.append(GestureEvent("switcher", "step", -1))

        if self.pose in (Pose.FIST, None):
            self._release_start = None
        else:
            if self._release_start is None:
                self._release_start = t
            wait = RELEASE_TIME if hand.extended >= 3 else RELAXED_RELEASE_TIME
            if t - self._release_start >= wait:
                self._end_switcher(t, events, "commit")
                return
        if t - self._switcher_start > SWITCHER_TIMEOUT:
            self._end_switcher(t, events, "cancel")

    def _end_switcher(self, t: float, events: list[GestureEvent], how: str) -> None:
        self.switcher_active = False
        self._release_start = None
        self._cooldown_until = t + COOLDOWN_AFTER_SWITCHER
        self.swipe.reset()
        # Whatever pose the hand is in now has to change before it can trigger anything.
        self.hold_pose, self.hold_fired, self.hold_progress = self.pose, True, 0.0
        events.append(GestureEvent("switcher", how))
