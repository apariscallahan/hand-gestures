"""Gesture engine tests using synthetic hand movements.

Positions are in frame heights (a 640x480 frame is 1.33 wide, 1.0 tall) and
the simulated camera runs at 30 fps.
"""

import unittest

from gesture_control.config import TuningConfig
from gesture_control.engine import GestureEngine
from gesture_control.features import HandFeatures, Pose

EXTENDED_FINGERS = {Pose.OPEN: 4, Pose.FIST: 0, Pose.PINCH: 0, Pose.THUMB_UP: 0, Pose.VICTORY: 2}


class Sim:
    """Feeds a GestureEngine a scripted hand, frame by frame."""

    FPS = 30

    def __init__(self, tuning=None, *, switcher=True, palm=0.14, pointer=None, holds=None):
        tuning = tuning or TuningConfig()
        hold_times = holds or {Pose.PINCH: tuning.pinch_hold_time, Pose.FIST: tuning.fist_hold_time}
        self.engine = GestureEngine(tuning, hold_times, switcher=switcher, pointer=pointer)
        self.palm = palm
        self.t = 0.0
        self.pos = (0.66, 0.5)
        self.log = []  # (time, event)

    @property
    def events(self):
        return [e for _, e in self.log]

    def _frame(self, hand):
        self.t += 1 / self.FPS
        self.log += [(self.t, e) for e in self.engine.update(self.t, hand)]

    def _hand(self, pose, extended):
        if extended is None:
            extended = EXTENDED_FINGERS.get(pose, 1)
        return HandFeatures(pose=pose, center=self.pos, palm=self.palm, curls=(0.0,) * 4,
                            extended=extended, pinch=1.0, thumb_out=0.5, thumb_gap=0.7, obs=None)

    def hold(self, seconds, pose, at=None, extended=None):
        if at is not None:
            self.pos = at
        for _ in range(round(seconds * self.FPS)):
            self._frame(self._hand(pose, extended))
        return self

    def move(self, seconds, pose, dx=0.0, dy=0.0, extended=None):
        frames = round(seconds * self.FPS)
        x0, y0 = self.pos
        for i in range(1, frames + 1):
            self.pos = (x0 + dx * i / frames, y0 + dy * i / frames)
            self._frame(self._hand(pose, extended))
        return self

    def gone(self, seconds):
        for _ in range(round(seconds * self.FPS)):
            self._frame(None)
        return self

    def names(self):
        return [e.name for e in self.events]

    def time_of(self, name):
        return next(t for t, e in self.log if e.name == name)


class SwipeTests(unittest.TestCase):
    def test_quick_open_hand_movement_is_a_swipe(self):
        sim = Sim().hold(0.6, Pose.OPEN).move(0.3, Pose.OPEN, dx=-0.5)
        self.assertEqual(sim.names(), ["swipe_left"])

    def test_swipe_right_up_down(self):
        for dx, dy, name in ((0.5, 0, "swipe_right"), (0, -0.5, "swipe_up"), (0, 0.5, "swipe_down")):
            with self.subTest(name):
                sim = Sim().hold(0.6, Pose.OPEN, at=(0.66, 0.5)).move(0.3, Pose.OPEN, dx=dx, dy=dy)
                self.assertEqual(sim.names(), [name])

    def test_swipe_works_moments_after_the_hand_appears(self):
        sim = Sim().hold(0.15, Pose.OPEN).move(0.3, Pose.OPEN, dx=-0.5)
        self.assertEqual(sim.names(), ["swipe_left"])

    def test_swipe_fires_early_in_the_movement(self):
        sim = Sim().hold(0.3, Pose.OPEN).move(0.3, Pose.OPEN, dx=-0.5)
        self.assertLess(sim.time_of("swipe_left") - 0.3, 0.17)

    def test_long_sweep_counts_once(self):
        sim = Sim().hold(0.3, Pose.OPEN).move(0.6, Pose.OPEN, dx=-1.0)
        self.assertEqual(sim.names(), ["swipe_left"])

    def test_quick_repeated_swipes(self):
        # swipe, bring the hand back, swipe again - with only natural pauses in between
        sim = Sim().hold(0.3, Pose.OPEN)
        sim.move(0.25, Pose.OPEN, dx=-0.4).hold(0.1, Pose.OPEN)
        sim.move(0.25, Pose.OPEN, dx=0.4).hold(0.1, Pose.OPEN)
        sim.move(0.25, Pose.OPEN, dx=-0.4)
        self.assertEqual(sim.names(), ["swipe_left", "swipe_left"])

    def test_fast_flick_needs_less_travel(self):
        fast = Sim().hold(0.3, Pose.OPEN).move(0.1, Pose.OPEN, dx=-0.2)
        slow = Sim().hold(0.3, Pose.OPEN).move(0.3, Pose.OPEN, dx=-0.2)
        self.assertEqual(fast.names(), ["swipe_left"])
        self.assertEqual(slow.names(), [])

    def test_bringing_the_hand_back_is_not_a_swipe(self):
        sim = Sim().hold(0.6, Pose.OPEN).move(0.3, Pose.OPEN, dx=-0.5)
        sim.hold(0.1, Pose.OPEN).move(0.3, Pose.OPEN, dx=0.5)
        self.assertEqual(sim.names(), ["swipe_left"])

    def test_bringing_the_hand_back_after_a_short_rest_is_not_a_swipe(self):
        sim = Sim().hold(0.6, Pose.OPEN).move(0.3, Pose.OPEN, dx=-0.5)
        sim.hold(0.5, Pose.OPEN).move(0.3, Pose.OPEN, dx=0.5)
        self.assertEqual(sim.names(), ["swipe_left"])

    def test_can_swipe_back_the_other_way_after_a_pause(self):
        sim = Sim().hold(0.6, Pose.OPEN).move(0.3, Pose.OPEN, dx=-0.5)
        sim.hold(1.0, Pose.OPEN).move(0.3, Pose.OPEN, dx=0.5)
        self.assertEqual(sim.names(), ["swipe_left", "swipe_right"])

    def test_slow_drift_is_not_a_swipe(self):
        sim = Sim().hold(0.6, Pose.OPEN).move(3.0, Pose.OPEN, dx=-0.6)
        self.assertEqual(sim.names(), [])

    def test_short_flick_is_not_a_swipe(self):
        sim = Sim().hold(0.6, Pose.OPEN).move(0.1, Pose.OPEN, dx=-0.15)
        self.assertEqual(sim.names(), [])

    def test_closed_hand_does_not_swipe(self):
        sim = Sim().hold(0.6, Pose.OTHER).move(0.3, Pose.OTHER, dx=-0.5)
        self.assertEqual(sim.names(), [])

    def test_diagonal_movement_is_not_a_swipe(self):
        sim = Sim().hold(0.6, Pose.OPEN).move(0.3, Pose.OPEN, dx=-0.4, dy=0.4)
        self.assertEqual(sim.names(), [])

    def test_hand_moving_into_view_does_not_swipe(self):
        # Raised from the bottom edge in one continuous movement: it never
        # pauses, so it is never armed, however far it travels.
        sim = Sim().hold(1 / 30, Pose.OPEN, at=(0.66, 1.0)).move(0.6, Pose.OPEN, dy=-0.7)
        self.assertEqual(sim.names(), [])

    def test_hand_sweeping_in_from_the_side_does_not_swipe(self):
        # e.g. reaching across: it never pauses, so it is never armed
        sim = Sim().hold(1 / 30, Pose.OPEN, at=(0.0, 0.5)).move(0.4, Pose.OPEN, dx=0.8)
        self.assertEqual(sim.names(), [])

    def test_brief_pause_while_raising_the_hand_is_not_a_swipe_up(self):
        sim = Sim().hold(1 / 30, Pose.OPEN, at=(0.66, 1.0)).move(0.35, Pose.OPEN, dy=-0.3)
        sim.hold(5 / 30, Pose.OPEN).move(0.4, Pose.OPEN, dy=-0.5)
        self.assertEqual(sim.names(), [])

    def test_sideways_swipe_right_after_raising_the_hand(self):
        sim = Sim().hold(1 / 30, Pose.OPEN, at=(0.66, 1.0)).move(0.35, Pose.OPEN, dy=-0.3)
        sim.hold(5 / 30, Pose.OPEN).move(0.3, Pose.OPEN, dx=-0.5)
        self.assertEqual(sim.names(), ["swipe_left"])

    def test_tracker_jumping_to_another_hand_is_not_a_swipe(self):
        sim = Sim().hold(0.6, Pose.OPEN, at=(0.3, 0.5)).hold(0.5, Pose.OPEN, at=(1.2, 0.5))
        self.assertEqual(sim.names(), [])

    def test_distance_scales_with_hand_size(self):
        # 0.2 frame heights is ~1.4 palm lengths up close (too short) but ~2.9 far away.
        # Moved at a steady pace, so the quick-flick discount barely applies.
        near = Sim(palm=0.14).hold(0.6, Pose.OPEN).move(0.4, Pose.OPEN, dx=-0.2)
        far = Sim(palm=0.07).hold(0.6, Pose.OPEN).move(0.4, Pose.OPEN, dx=-0.2)
        self.assertEqual(near.names(), [])
        self.assertEqual(far.names(), ["swipe_left"])


class HoldTests(unittest.TestCase):
    def test_pinch_hold_fires_once(self):
        sim = Sim().hold(0.5, Pose.OPEN).hold(2.0, Pose.PINCH)
        self.assertEqual(sim.names(), ["pinch_hold"])

    def test_pinch_fires_about_a_third_of_a_second_after_pinching(self):
        sim = Sim().hold(0.5, Pose.OPEN).hold(0.25, Pose.PINCH)
        self.assertEqual(sim.names(), [])
        self.assertGreater(sim.engine.hold_progress, 0.5)
        sim.hold(0.1, Pose.PINCH)
        self.assertEqual(sim.names(), ["pinch_hold"])

    def test_releasing_early_cancels(self):
        sim = Sim().hold(0.5, Pose.OPEN).hold(0.2, Pose.PINCH).hold(0.5, Pose.OPEN)
        self.assertEqual(sim.names(), [])
        self.assertEqual(sim.engine.hold_progress, 0.0)

    def test_pinch_can_repeat_after_letting_go(self):
        sim = Sim().hold(0.5, Pose.OPEN).hold(0.5, Pose.PINCH).hold(0.3, Pose.OPEN).hold(0.5, Pose.PINCH)
        self.assertEqual(sim.names(), ["pinch_hold", "pinch_hold"])

    def test_pinch_that_did_not_start_from_an_open_hand_is_ignored(self):
        sim = Sim().hold(2.0, Pose.PINCH)
        self.assertEqual(sim.names(), [])
        self.assertTrue(sim.engine.hold_blocked)

    def test_open_hand_requirement_can_be_turned_off(self):
        sim = Sim(TuningConfig(require_open_hand_first=False)).hold(2.0, Pose.PINCH)
        self.assertEqual(sim.names(), ["pinch_hold"])

    def test_moving_pinch_does_not_fire(self):
        sim = Sim().hold(0.5, Pose.OPEN).move(2.0, Pose.PINCH, dx=0.6)
        self.assertEqual(sim.names(), [])

    def test_single_frame_glitch_does_not_reset_the_hold(self):
        sim = Sim().hold(0.5, Pose.OPEN).hold(0.15, Pose.PINCH)
        sim.hold(1 / 30, Pose.OTHER).hold(0.25, Pose.PINCH)
        self.assertEqual(sim.names(), ["pinch_hold"])

    def test_pose_without_an_action_is_ignored(self):
        sim = Sim().hold(0.5, Pose.OPEN).hold(2.0, Pose.THUMB_UP)
        self.assertEqual(sim.names(), [])

    def test_fist_hold_is_a_plain_gesture_when_switcher_is_off(self):
        sim = Sim(switcher=False).hold(0.5, Pose.OPEN).hold(1.0, Pose.FIST)
        self.assertEqual(sim.names(), ["fist_hold"])

    def test_victory_hold(self):
        tuning = TuningConfig()
        sim = Sim(holds={Pose.VICTORY: tuning.gesture_hold_time}).hold(0.5, Pose.OPEN).hold(0.5, Pose.VICTORY)
        self.assertEqual(sim.names(), ["victory_hold"])


class SwitcherTests(unittest.TestCase):
    def grab(self):
        return Sim().hold(0.5, Pose.OPEN).hold(0.35, Pose.FIST)

    def steps(self, sim):
        return [e.step for e in sim.events if e.name == "step"]

    def test_open_hand_then_fist_opens_the_switcher(self):
        sim = self.grab()
        self.assertEqual(sim.names(), ["open"])
        self.assertTrue(sim.engine.switcher_active)

    def test_switcher_opens_within_a_quarter_second(self):
        sim = Sim().hold(0.5, Pose.OPEN)
        first_fist_frame = sim.t + 1 / Sim.FPS
        sim.hold(0.35, Pose.FIST)
        self.assertLessEqual(sim.time_of("open") - first_fist_frame, 0.25)

    def test_move_right_then_open_hand_picks(self):
        sim = self.grab().move(1.0, Pose.FIST, dx=0.18)  # step is 0.55 palm = 0.077
        self.assertEqual(self.steps(sim), [1, 1])
        sim.hold(0.2, Pose.OPEN)
        self.assertEqual(sim.names()[-1], "commit")
        self.assertFalse(sim.engine.switcher_active)

    def test_move_left_steps_backwards(self):
        sim = self.grab().move(1.0, Pose.FIST, dx=-0.18)
        self.assertEqual(self.steps(sim), [-1, -1])

    def test_small_wobble_does_not_step(self):
        sim = self.grab().move(0.3, Pose.FIST, dx=0.03).move(0.3, Pose.FIST, dx=-0.06)
        self.assertEqual(self.steps(sim), [])

    def test_losing_the_hand_cancels(self):
        sim = self.grab().gone(1.0)
        self.assertEqual(sim.names(), ["open", "cancel"])

    def test_brief_dropout_keeps_switching(self):
        sim = self.grab().gone(0.2).hold(0.3, Pose.FIST)
        self.assertEqual(sim.names(), ["open"])
        self.assertTrue(sim.engine.switcher_active)

    def test_fast_fist_movement_is_not_a_swipe(self):
        sim = self.grab().move(0.3, Pose.FIST, dx=-0.5)
        self.assertNotIn("swipe_left", sim.names())

    def test_fist_that_did_not_start_from_an_open_hand_is_ignored(self):
        sim = Sim().hold(2.0, Pose.FIST)
        self.assertEqual(sim.names(), [])

    def test_no_swipe_right_after_picking(self):
        sim = self.grab().hold(0.2, Pose.OPEN).move(0.3, Pose.OPEN, dx=-0.5)
        self.assertEqual(sim.names(), ["open", "commit"])

    def test_cancel_ends_the_switcher(self):
        sim = self.grab()
        self.assertEqual([e.name for e in sim.engine.cancel(sim.t)], ["cancel"])
        self.assertFalse(sim.engine.switcher_active)


if __name__ == "__main__":
    unittest.main()
