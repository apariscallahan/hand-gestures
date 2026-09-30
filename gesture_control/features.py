"""Per-frame hand geometry: finger curl, pinch distance, palm position and a pose label.

Pose decisions use MediaPipe's *world* landmarks (3-D, in metres), so they
don't depend on how far the hand is from the camera or how it is rotated.
Positions for motion tracking use the image landmarks instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17
FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))  # index..pinky: MCP, PIP, DIP, tip
PALM_POINTS = (0, 5, 9, 13, 17)

EXTENDED_BELOW = 0.4  # finger curl under this counts as a straight finger
CURLED_ABOVE = 0.55  # finger curl over this counts as folded
THUMB_TUCKED_BELOW = 0.85  # thumb tip within this many palm lengths of the middle knuckle
PINCH_INDEX_MAX_CURL = 0.85  # in a fist the index is folded away; in a pinch it is only bent
MODEL_MIN_SCORE = 0.55


class Pose(StrEnum):
    OPEN = "open"
    FIST = "fist"
    PINCH = "pinch"
    THUMB_UP = "thumb_up"
    THUMB_DOWN = "thumb_down"
    VICTORY = "victory"
    POINTING_UP = "pointing_up"
    LOVE_YOU = "love_you"
    OTHER = "other"


# Poses taken straight from MediaPipe's gesture classifier (its label -> our pose).
MODEL_POSES = {
    "Thumb_Up": Pose.THUMB_UP,
    "Thumb_Down": Pose.THUMB_DOWN,
    "Victory": Pose.VICTORY,
    "Pointing_Up": Pose.POINTING_UP,
    "ILoveYou": Pose.LOVE_YOU,
}


@dataclass
class HandObservation:
    """One hand as reported by the tracker for one frame."""

    landmarks: np.ndarray  # (21, 3) normalised image coordinates; x, y in 0..1
    world: np.ndarray  # (21, 3) metres, origin near the centre of the hand
    handedness: str  # "Left" / "Right", from the user's point of view
    gesture: str  # MediaPipe's gesture label, e.g. "Closed_Fist" or "None"
    gesture_score: float


@dataclass
class HandFeatures:
    pose: Pose
    center: tuple[float, float]  # palm centre in frame-height units: (x * aspect, y)
    palm: float  # palm length (wrist to middle knuckle) in frame-height units
    curls: tuple[float, float, float, float]  # index..pinky; 0 = straight, 1 = fully folded
    extended: int  # how many of the four fingers are straight
    pinch: float  # thumb-tip to index-tip gap, in palm lengths
    thumb_out: float  # thumb-tip to middle knuckle, in palm lengths
    obs: HandObservation

    @property
    def pinch_point(self) -> tuple[float, float]:
        """Midpoint of thumb and index tips in normalised image coordinates."""
        lm = self.obs.landmarks
        return (float(lm[THUMB_TIP, 0] + lm[INDEX_TIP, 0]) / 2, float(lm[THUMB_TIP, 1] + lm[INDEX_TIP, 1]) / 2)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(a @ b) / (float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def finger_curl(world: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> float:
    """0 for a straight finger, 1 for one folded into the palm.

    Averages two cues: the total bend across the three finger joints, and how
    far the fingertip reaches from the wrist compared with the middle joint.
    """
    wrist = world[WRIST]
    bend = (
        _angle(world[mcp] - wrist, world[pip] - world[mcp])
        + _angle(world[pip] - world[mcp], world[dip] - world[pip])
        + _angle(world[dip] - world[pip], world[tip] - world[dip])
    )
    reach = np.linalg.norm(world[tip] - wrist) / (np.linalg.norm(world[pip] - wrist) + 1e-9)
    from_bend = np.clip((bend - 60.0) / 120.0, 0.0, 1.0)  # ~60 deg straight .. ~180 deg folded
    from_reach = np.clip((1.35 - reach) / 0.6, 0.0, 1.0)  # tip well past the joint .. tip back at the palm
    return float((from_bend + from_reach) / 2)


class PoseClassifier:
    """Turns a HandObservation into HandFeatures. Keeps a little state for
    pinch hysteresis, so call reset() when the hand is lost."""

    def __init__(self, pinch_threshold: float = 0.25) -> None:
        self.pinch_on = pinch_threshold
        self.pinch_off = pinch_threshold * 1.45
        self._pinching = False

    def reset(self) -> None:
        self._pinching = False

    def __call__(self, obs: HandObservation, aspect: float) -> HandFeatures:
        world = obs.world
        palm_len = float(np.linalg.norm(world[MIDDLE_MCP] - world[WRIST])) + 1e-9
        curls = tuple(finger_curl(world, *joints) for joints in FINGERS)
        extended = sum(c < EXTENDED_BELOW for c in curls)
        pinch = float(np.linalg.norm(world[THUMB_TIP] - world[INDEX_TIP])) / palm_len
        thumb_out = float(np.linalg.norm(world[THUMB_TIP] - world[MIDDLE_MCP])) / palm_len

        limit = self.pinch_off if self._pinching else self.pinch_on
        self._pinching = pinch < limit and curls[0] < PINCH_INDEX_MAX_CURL

        model_pose = MODEL_POSES.get(obs.gesture) if obs.gesture_score >= MODEL_MIN_SCORE else None
        model_fist = obs.gesture == "Closed_Fist" and obs.gesture_score >= MODEL_MIN_SCORE
        folded = all(c > CURLED_ABOVE for c in curls)

        if self._pinching:
            pose = Pose.PINCH
        elif model_pose is not None:
            pose = model_pose
        elif folded and (thumb_out < THUMB_TUCKED_BELOW or model_fist):
            pose = Pose.FIST
        elif extended == 4 or (extended == 3 and obs.gesture == "Open_Palm"):
            pose = Pose.OPEN
        else:
            pose = Pose.OTHER

        # Motion is tracked in image space, in units of the frame height so
        # horizontal and vertical distances are comparable.
        pts = obs.landmarks[:, :2] * np.array([aspect, 1.0])
        center = pts[list(PALM_POINTS)].mean(axis=0)
        # Palm length shrinks when the hand tilts toward the camera; knuckle
        # width shrinks when it turns sideways. The larger of the two is stable.
        palm = max(
            float(np.linalg.norm(pts[WRIST] - pts[MIDDLE_MCP])),
            1.25 * float(np.linalg.norm(pts[INDEX_MCP] - pts[PINKY_MCP])),
        )
        return HandFeatures(
            pose=pose,
            center=(float(center[0]), float(center[1])),
            palm=palm,
            curls=curls,
            extended=extended,
            pinch=pinch,
            thumb_out=thumb_out,
            obs=obs,
        )
