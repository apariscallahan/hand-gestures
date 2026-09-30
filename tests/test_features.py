"""Pose classification tests on real MediaPipe landmarks.

fixtures/hands.json holds landmarks MediaPipe produced for Google's sample
gesture photos (victory, pointing up, thumbs up/down, two open hands). Some
tests edit those landmarks to build poses the photos don't show.
"""

import json
import unittest
from pathlib import Path

import numpy as np

from gesture_control.features import HandObservation, Pose, PoseClassifier

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "hands.json").read_text())
ASPECT = 4 / 3


def observation(name, *, model_label=None, world=None):
    """A fixture hand; model_label overrides what MediaPipe's classifier said."""
    f = FIXTURES[name]
    return HandObservation(
        landmarks=np.array(f["image"]),
        world=np.array(f["world"]) if world is None else world,
        handedness=f["handedness"],
        gesture=f["gesture"] if model_label is None else model_label,
        gesture_score=f["gesture_score"] if model_label is None else 0.9,
    )


def classify(name, **kwargs):
    return PoseClassifier()(observation(name, **kwargs), ASPECT)


def world(name):
    return np.array(FIXTURES[name]["world"])


def palm_length(w):
    return float(np.linalg.norm(w[9] - w[0]))


class ModelLabelTests(unittest.TestCase):
    def test_gestures_recognised_by_the_model(self):
        expected = {
            "victory_0": Pose.VICTORY,
            "pointing_up_0": Pose.POINTING_UP,
            "thumbs_up_0": Pose.THUMB_UP,
            "thumbs_down_0": Pose.THUMB_DOWN,
        }
        for name, pose in expected.items():
            with self.subTest(name):
                self.assertEqual(classify(name).pose, pose)

    def test_open_hands(self):
        for name in ("woman_hands_0", "woman_hands_1"):
            with self.subTest(name):
                features = classify(name)
                self.assertEqual(features.extended, 4)
                self.assertEqual(features.pose, Pose.OPEN)


class GeometryTests(unittest.TestCase):
    """The same hands with the model's label removed, so only geometry decides."""

    def test_counts_straight_fingers(self):
        for name, count in (("victory_0", 2), ("pointing_up_0", 1), ("thumbs_up_0", 0), ("thumbs_down_0", 0)):
            with self.subTest(name):
                self.assertEqual(classify(name, model_label="None").extended, count)

    def test_curled_fingers_with_thumb_sticking_out_is_not_a_fist(self):
        for name in ("thumbs_up_0", "thumbs_down_0"):
            with self.subTest(name):
                self.assertEqual(classify(name, model_label="None").pose, Pose.OTHER)

    def test_curled_fingers_with_thumb_tucked_in_is_a_fist(self):
        w = world("thumbs_down_0")
        w[4] = (w[6] + w[10]) / 2  # thumb tip across the index and middle fingers
        w[3] = (w[2] + w[4]) / 2
        self.assertEqual(classify("thumbs_down_0", model_label="None", world=w).pose, Pose.FIST)

    def test_model_fist_label_accepts_a_loose_thumb(self):
        self.assertEqual(classify("thumbs_down_0", model_label="Closed_Fist").pose, Pose.FIST)

    def test_thumb_touching_index_tip_is_a_pinch(self):
        w = world("woman_hands_0")
        w[4] = w[8] + np.array([0.005, 0.0, 0.0])
        features = classify("woman_hands_0", world=w)
        self.assertEqual(features.pose, Pose.PINCH)
        self.assertLess(features.pinch, 0.1)

    def test_fist_is_never_a_pinch(self):
        # Thumb resting right on a folded index finger: the index is curled away, so no pinch.
        w = world("thumbs_down_0")
        w[4] = w[8] + np.array([0.005, 0.0, 0.0])
        self.assertNotEqual(classify("thumbs_down_0", model_label="None", world=w).pose, Pose.PINCH)

    def test_pinch_has_hysteresis(self):
        pinched, between = world("woman_hands_0"), world("woman_hands_0")
        pinched[4] = pinched[8] + np.array([0.005, 0.0, 0.0])
        gap = between[4] - between[8]
        between[4] = between[8] + gap / np.linalg.norm(gap) * 0.3 * palm_length(between)

        fresh = PoseClassifier(pinch_threshold=0.25)
        self.assertNotEqual(fresh(observation("woman_hands_0", world=between), ASPECT).pose, Pose.PINCH)

        held = PoseClassifier(pinch_threshold=0.25)
        held(observation("woman_hands_0", world=pinched), ASPECT)
        self.assertEqual(held(observation("woman_hands_0", world=between), ASPECT).pose, Pose.PINCH)

    def test_palm_position_and_size(self):
        features = classify("victory_0")
        x, y = features.center
        self.assertTrue(0 <= x <= ASPECT and 0 <= y <= 1)
        self.assertTrue(0.05 < features.palm < 1.0)


if __name__ == "__main__":
    unittest.main()
