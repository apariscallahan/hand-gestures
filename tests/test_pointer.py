"""Finger mouse tests with synthetic hands (30 fps).

Fingertip positions are normalised camera coordinates; with the defaults the
steering area spans x 0.25..0.75 and y 0.2625..0.6375 of a 4:3 image mapped
onto a 16:9 screen.
"""

import random
import statistics
import unittest

import numpy as np

from gesture_control.config import MouseConfig, TuningConfig
from gesture_control.engine import GestureEngine
from gesture_control.features import HandFeatures, HandObservation, Pose
from gesture_control.pointer import PointerTracker

POINTING_CURLS = (0.0, 1.0, 1.0, 1.0)
CENTER = (0.5, 0.45)


def hand(tip, gap, pose=Pose.POINTING, curls=POINTING_CURLS):
    landmarks = np.zeros((21, 3))
    landmarks[8, :2] = tip
    obs = HandObservation(landmarks=landmarks, world=np.zeros((21, 3)), handedness="Right",
                          gesture="None", gesture_score=0.0)
    return HandFeatures(pose=pose, center=(tip[0] * 4 / 3, tip[1] + 0.2), palm=0.14, curls=curls,
                        extended=sum(c < 0.4 for c in curls), pinch=1.0, thumb_out=0.8,
                        thumb_gap=gap, obs=obs)


class PointerSim:
    FPS = 30

    def __init__(self, **mouse):
        self.pointer = PointerTracker(MouseConfig(**mouse))
        self.pointer.set_geometry(4 / 3, 16 / 9)
        self.t = 0.0
        self.tip = CENTER
        self.gap = 0.7  # thumb sticking out
        self.log = []  # (time, event name, cursor position)
        self.positions = []

    def _frame(self, h, pose):
        self.t += 1 / self.FPS
        events = self.pointer.update(self.t, h, pose)
        self.positions.append(self.pointer.position)
        self.log += [(self.t, e.name, self.pointer.position) for e in events]

    def point(self, seconds, tip=None, pose=Pose.POINTING, curls=POINTING_CURLS, jitter=0.0):
        if tip is not None:
            self.tip = tip
        for _ in range(round(seconds * self.FPS)):
            noisy = (self.tip[0] + random.gauss(0, jitter), self.tip[1] + random.gauss(0, jitter))
            self._frame(hand(noisy, self.gap, pose, curls), pose)
        return self

    def move(self, seconds, dx=0.0, dy=0.0, gap=None):
        """Move the fingertip (and optionally the thumb) in a straight line."""
        frames = round(seconds * self.FPS)
        (x0, y0), g0 = self.tip, self.gap
        for i in range(1, frames + 1):
            self.tip = (x0 + dx * i / frames, y0 + dy * i / frames)
            if gap is not None:
                self.gap = g0 + (gap - g0) * i / frames
            self._frame(hand(self.tip, self.gap), Pose.POINTING)
        return self

    def thumb(self, seconds, gap):
        return self.move(seconds, gap=gap)

    def gone(self, seconds):
        for _ in range(round(seconds * self.FPS)):
            self._frame(None, None)
        return self

    def names(self):
        return [name for _, name, _ in self.log]


class SteeringTests(unittest.TestCase):
    def test_fingertip_position_maps_onto_the_screen(self):
        for tip, expected in [(CENTER, (0.5, 0.5)), ((0.25, 0.2625), (0.0, 0.0)), ((0.75, 0.6375), (1.0, 1.0))]:
            with self.subTest(tip=tip):
                sim = PointerSim().point(0.5, tip)
                for got, want in zip(sim.pointer.position, expected):
                    self.assertAlmostEqual(got, want, places=3)

    def test_positions_outside_the_area_stick_to_the_edge(self):
        sim = PointerSim().point(0.5, (0.05, 0.95))
        self.assertEqual(sim.pointer.position, (0.0, 1.0))

    def test_not_pointing_leaves_the_mouse_alone(self):
        sim = PointerSim().point(0.5, CENTER, pose=Pose.OPEN, curls=(0.0,) * 4)
        self.assertFalse(sim.pointer.active)
        self.assertIsNone(sim.pointer.position)

    def test_cursor_is_steadier_than_the_raw_fingertip(self):
        random.seed(3)
        sim = PointerSim().point(3.0, CENTER, jitter=0.0025)
        xs = [p[0] for p in sim.positions[30:]]
        shake = statistics.fmean(abs(b - a) for a, b in zip(xs, xs[1:]))
        raw_shake = 0.0025 / 0.5 * 2 / np.sqrt(np.pi) * np.sqrt(2)  # mean |difference| of the raw mapped signal
        self.assertLess(shake, raw_shake / 3)

    def test_cursor_keeps_up_with_the_finger(self):
        sim = PointerSim().point(0.5, (0.3, 0.45)).move(0.5, dx=0.4)
        # the finger moved 0.8 screen widths in half a second; 0.1 s later the cursor is there
        sim.point(0.1)
        self.assertAlmostEqual(sim.pointer.position[0], 0.9, delta=0.02)

    def test_relaxed_hand_keeps_steering(self):
        sim = PointerSim().point(0.3, CENTER)
        sim.point(0.3, pose=Pose.OTHER, curls=(0.3, 0.3, 1.0, 1.0))  # middle finger straightened a bit
        self.assertTrue(sim.pointer.active)
        sim.point(0.1, pose=Pose.OPEN, curls=(0.1, 0.1, 0.1, 0.1))
        self.assertFalse(sim.pointer.active)

    def test_stopping_undoes_the_last_moment_of_motion(self):
        # Curling the finger to stop pointing drags the tip down for a frame or two.
        sim = PointerSim().point(0.5, CENTER).move(0.07, dy=0.1)
        sim.point(1 / 30, pose=Pose.FIST, curls=(1.0,) * 4)
        self.assertAlmostEqual(sim.pointer.position[1], 0.5, delta=0.02)
        sim.point(1 / 30, pose=Pose.FIST, curls=(1.0,) * 4)
        self.assertIsNone(sim.pointer.position)


class ClickTests(unittest.TestCase):
    def test_pressing_the_thumb_in_clicks(self):
        sim = PointerSim().point(0.5).thumb(0.1, 0.2).point(0.1).thumb(0.1, 0.7)
        self.assertEqual(sim.names(), ["down", "up"])

    def test_click_is_quick(self):
        sim = PointerSim().point(0.5)
        start = sim.t
        sim.thumb(0.12, 0.2)
        self.assertLess(sim.log[0][0] - start, 0.12)

    def test_two_quick_clicks(self):
        sim = PointerSim().point(0.5)
        for _ in range(2):
            sim.thumb(0.1, 0.2).thumb(0.1, 0.7)
        self.assertEqual(sim.names(), ["down", "up", "down", "up"])
        positions = {pos for _, _, pos in sim.log}
        self.assertEqual(len(positions), 1)  # all at exactly the same spot, so Windows sees a double-click

    def test_slowly_tucking_the_thumb_does_not_click(self):
        sim = PointerSim().point(0.5).thumb(3.0, 0.2)
        self.assertEqual(sim.names(), [])

    def test_pointing_with_the_thumb_already_tucked_does_not_click(self):
        sim = PointerSim()
        sim.gap = 0.2
        sim.point(1.0)
        self.assertEqual(sim.names(), [])

    def test_small_thumb_twitch_does_not_click(self):
        sim = PointerSim().point(0.5).thumb(0.1, 0.55).thumb(0.1, 0.7)
        self.assertEqual(sim.names(), [])

    def test_click_lands_where_the_cursor_was_before_the_thumb_moved(self):
        # Pressing the thumb also nudges the fingertip down a little.
        sim = PointerSim().point(0.5, CENTER)
        before = sim.pointer.position
        sim.move(0.1, dy=0.02, gap=0.2)
        _, name, where = sim.log[0]
        self.assertEqual(name, "down")
        self.assertAlmostEqual(where[1], before[1], delta=0.005)

    def test_moving_with_the_thumb_pressed_drags(self):
        sim = PointerSim().point(0.5, CENTER).thumb(0.1, 0.2)
        sim.move(0.5, dx=0.15)
        self.assertEqual(sim.names(), ["down"])
        self.assertGreater(sim.pointer.position[0], 0.7)  # the cursor followed
        sim.thumb(0.1, 0.7)
        self.assertEqual(sim.names(), ["down", "up"])

    def test_letting_the_hand_drop_releases_the_button(self):
        sim = PointerSim().point(0.5).thumb(0.1, 0.2).gone(0.1)
        self.assertEqual(sim.names(), ["down", "up"])

    def test_opening_the_hand_releases_the_button(self):
        sim = PointerSim().point(0.5).thumb(0.1, 0.2)
        sim.point(0.1, pose=Pose.OPEN, curls=(0.0,) * 4)
        self.assertEqual(sim.names(), ["down", "up"])

    def test_no_click_right_as_pointing_starts(self):
        sim = PointerSim()
        sim.point(1 / 30)
        sim.thumb(0.05, 0.2)
        self.assertEqual(sim.names(), [])

    def test_clicking_can_be_turned_off(self):
        sim = PointerSim(click=False).point(0.5).thumb(0.1, 0.2).thumb(0.1, 0.7)
        self.assertEqual(sim.names(), [])


class EngineIntegrationTests(unittest.TestCase):
    def test_pointing_steers_and_blocks_other_gestures(self):
        tuning = TuningConfig()
        pointer = PointerTracker(MouseConfig())
        engine = GestureEngine(tuning, {Pose.VICTORY: 0.3}, pointer=pointer)
        t, events = 0.0, []
        for pose, curls, seconds in [(Pose.OPEN, (0.0,) * 4, 0.5), (Pose.POINTING, POINTING_CURLS, 0.3),
                                     (Pose.VICTORY, (0.0, 0.0, 1.0, 1.0), 0.6)]:
            for _ in range(round(seconds * 30)):
                t += 1 / 30
                events += engine.update(t, hand(CENTER, 0.7, pose, curls))
        # The victory sign came out of the pointing hand, so it is still steering
        # the mouse rather than counting as a held gesture.
        self.assertTrue(pointer.active)
        self.assertEqual(events, [])

    def test_pausing_releases_the_mouse_button(self):
        pointer = PointerTracker(MouseConfig())
        engine = GestureEngine(TuningConfig(), {}, pointer=pointer)
        t = 0.0
        for gap in [0.7] * 15 + [0.45, 0.2]:
            t += 1 / 30
            engine.update(t, hand(CENTER, gap))
        self.assertTrue(pointer.pressed)
        self.assertEqual([e.name for e in engine.cancel(t)], ["up"])


if __name__ == "__main__":
    unittest.main()
