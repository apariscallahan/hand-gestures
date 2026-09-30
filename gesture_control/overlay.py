"""The preview window: the camera image with the hand skeleton and gesture feedback on top."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import win32
from .config import PreviewConfig
from .engine import GestureEngine, SwipeDetector
from .features import HandFeatures, Pose

WINDOW = "Gesture Control"
FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA
BASE_WIDTH = 420  # default preview width at 100% display scaling


def _rgb(r: int, g: int, b: int) -> tuple[int, int, int]:
    return (b, g, r)  # OpenCV wants BGR


TEXT = _rgb(240, 240, 240)
MUTED = _rgb(165, 170, 178)
PANEL = _rgb(20, 22, 26)
TRACK = _rgb(70, 72, 78)
GREEN = _rgb(90, 205, 120)
BLUE = _rgb(80, 170, 255)
PINK = _rgb(240, 110, 200)
AMBER = _rgb(255, 190, 60)
RED = _rgb(240, 85, 85)

POSE_STYLE = {
    Pose.OPEN: ("open hand", GREEN),
    Pose.FIST: ("fist", BLUE),
    Pose.PINCH: ("pinch", PINK),
    Pose.THUMB_UP: ("thumbs up", AMBER),
    Pose.THUMB_DOWN: ("thumbs down", AMBER),
    Pose.VICTORY: ("victory", AMBER),
    Pose.POINTING_UP: ("pointing up", AMBER),
    Pose.LOVE_YOU: ("love you", AMBER),
    Pose.OTHER: ("hand", TEXT),
}

GESTURE_NAMES = {
    "swipe_left": "Swipe left",
    "swipe_right": "Swipe right",
    "swipe_up": "Swipe up",
    "swipe_down": "Swipe down",
    "pinch_hold": "Pinch + hold",
    "fist_hold": "Open hand > fist",
    "thumb_up_hold": "Thumbs up",
    "thumb_down_hold": "Thumbs down",
    "victory_hold": "Victory sign",
    "pointing_up_hold": "Point up",
    "love_you_hold": "Love-you sign",
}

SKELETON = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
FINGERTIPS = frozenset((4, 8, 12, 16, 20))
ARROWS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}


@dataclass
class HudInfo:
    features: HandFeatures | None = None
    fps: float = 0.0
    infer_ms: float = 0.0
    dry_run: bool = False
    desktop: tuple[int | None, int] | None = None
    toast: str | None = None
    labels: dict[str, str] = field(default_factory=dict)  # gesture -> action label, mapped gestures only
    details: bool = False


# -- drawing primitives -----------------------------------------------------

def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")  # Hershey fonts are ASCII-only


def _size(text: str, scale: float, thickness: int = 1) -> tuple[int, int]:
    (w, h), _ = cv2.getTextSize(_ascii(text), FONT, scale, thickness)
    return w, h


def _put(img, text: str, x: float, y: float, scale: float, colour, thickness: int = 1) -> None:
    cv2.putText(img, _ascii(text), (int(x), int(y)), FONT, scale, colour, thickness, AA)


def _blend_rect(img, x0: float, y0: float, x1: float, y1: float, colour, alpha: float) -> None:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha) + np.array(colour) * alpha).astype(np.uint8)


def _wrap(text: str, scale: float, max_width: float) -> list[str]:
    """Greedy word wrap using real text widths; keeps explicit line breaks."""
    lines = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if current and _size(candidate, scale)[0] > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines


def _pill(img, text: str, cx: float, cy: float, scale: float, colour=TEXT, alpha: float = 0.7) -> None:
    """Text centred on (cx, cy) over a translucent box, kept inside the image."""
    w, h = _size(text, scale)
    pad = h * 0.7 + 2
    img_h, img_w = img.shape[:2]
    cx = min(max(cx, w / 2 + pad), img_w - w / 2 - pad)
    cy = min(max(cy, h / 2 + pad), img_h - h / 2 - pad)
    _blend_rect(img, cx - w / 2 - pad, cy - h / 2 - pad, cx + w / 2 + pad, cy + h / 2 + pad, PANEL, alpha)
    _put(img, text, cx - w / 2, cy + h / 2, scale, colour)


# -- the window -------------------------------------------------------------

class Preview:
    def __init__(self, cfg: PreviewConfig, frame_size: tuple[int, int]) -> None:
        self.aspect = frame_size[0] / frame_size[1]
        self.position = cfg.position
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW, np.zeros((90, 120, 3), np.uint8))
        cv2.waitKey(1)  # lets the native window come into existence
        self.hwnd = win32.find_window(WINDOW)
        self.dpi = win32.dpi_scale(self.hwnd)
        left, _, right, _ = win32.work_area()
        self.width = cfg.width or int(min(BASE_WIDTH * self.dpi, 0.4 * (right - left)))
        cv2.resizeWindow(WINDOW, self.width, round(self.width / self.aspect))
        if self.hwnd:
            win32.style_preview_window(self.hwnd, topmost=cfg.always_on_top, all_desktops=cfg.show_on_all_desktops)
            win32.place_window(self.hwnd, self.position)

    def set_frame_size(self, width: int, height: int) -> None:
        """Match the window to the camera's real aspect ratio once it is known."""
        aspect = width / height
        if abs(aspect - self.aspect) > 0.02:
            self.aspect = aspect
            w, _ = self._window_size()
            cv2.resizeWindow(WINDOW, w, round(w / aspect))
            if self.hwnd:
                win32.place_window(self.hwnd, self.position)

    def is_open(self) -> bool:
        try:
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def poll_key(self, delay_ms: int = 1) -> int:
        key = cv2.waitKey(delay_ms)
        return -1 if key < 0 else key & 0xFF

    def close(self) -> None:
        cv2.destroyAllWindows()

    def _window_size(self) -> tuple[int, int]:
        try:
            _, _, w, h = cv2.getWindowImageRect(WINDOW)
        except cv2.error:
            w = h = 0
        if w < 120 or h < 90:  # not laid out yet, or minimised
            w, h = self.width, round(self.width / self.aspect)
        return w, h

    def _canvas(self, frame: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """An image the size of the window, plus the UI scale factor for it."""
        w, h = self._window_size()
        if frame is None:
            img = np.full((h, w, 3), 30, np.uint8)
        else:
            fh, fw = frame.shape[:2]
            scale = min(w / fw, h / fh)
            size = (max(1, round(fw * scale)), max(1, round(fh * scale)))
            img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        return img, self.dpi * img.shape[1] / self.width

    # -- screens ------------------------------------------------------------

    def show(self, frame: np.ndarray, engine: GestureEngine, info: HudInfo, now: float) -> None:
        """Draw a frame; `now` is its capture time, the clock the engine runs on."""
        img, s = self._canvas(frame)
        draw_hud(img, s, engine, info, self.aspect, now)
        cv2.imshow(WINDOW, img)

    def show_message(self, title: str, body: str = "", colour=TEXT) -> None:
        img, s = self._canvas()
        draw_message(img, s, title, body, colour)
        cv2.imshow(WINDOW, img)


# -- whole screens ----------------------------------------------------------

def draw_hud(img, s: float, engine: GestureEngine, info: HudInfo, aspect: float, now: float) -> None:
    """Draw all gesture feedback onto a camera image (in place). `s` is the UI scale."""
    feats = info.features
    if feats is not None and engine.pose is not None:
        pose_name, colour = POSE_STYLE[engine.pose]
    elif feats is not None:
        pose_name, colour = "hand", TEXT
    else:
        pose_name, colour = "no hand", MUTED

    if feats is not None:
        if not engine.switcher_active:
            _draw_trail(img, engine.swipe, aspect, s, now)
        _draw_skeleton(img, feats.obs.landmarks, colour, s)
        _draw_hold(img, engine, feats, info, aspect, s)
    if engine.switcher_active:
        _draw_switcher(img, engine, s)
    _draw_swipe_flash(img, engine.swipe, info, s, now)

    status, status_colour = ("DRY RUN", AMBER) if info.dry_run else ("ACTIVE", GREEN)
    _draw_top_bar(img, s, status, status_colour, pose_name, colour, _desktop_text(info.desktop))
    if info.details:
        _draw_details(img, s, feats, engine, info)
    elif feats is None:
        _draw_legend(img, s, info.labels)
    if info.toast:
        _draw_toast(img, s, info.toast)
    _draw_help(img, s)


def draw_message(img, s: float, title: str, body: str = "", colour=TEXT) -> None:
    """A centred title and wrapped explanation, e.g. for "PAUSED" (in place)."""
    h, w = img.shape[:2]
    tw, _ = _size(title, 0.9 * s, 2)
    _put(img, title, (w - tw) / 2, h * 0.38, 0.9 * s, colour, 2)
    for i, line in enumerate(_wrap(body, 0.45 * s, w - 30 * s)):
        lw, _ = _size(line, 0.45 * s)
        _put(img, line, (w - lw) / 2, h * 0.38 + (32 + 20 * i) * s, 0.45 * s, MUTED)
    _draw_help(img, s)


# -- HUD pieces -------------------------------------------------------------

def _desktop_text(desktop: tuple[int | None, int] | None) -> str:
    if desktop is None:
        return ""
    index, count = desktop
    return f"desktop {index}/{count}" if index else f"{count} desktops"


def _draw_top_bar(img, s: float, status: str, status_colour, centre: str, centre_colour, right: str) -> None:
    w = img.shape[1]
    bar = 28 * s
    _blend_rect(img, 0, 0, w, bar, PANEL, 0.65)
    base = bar * 0.68
    cv2.circle(img, (int(13 * s), int(bar / 2)), max(3, int(5 * s)), status_colour, -1, AA)
    _put(img, status, 24 * s, base, 0.45 * s, TEXT)
    cw, _ = _size(centre, 0.5 * s)
    _put(img, centre, (w - cw) / 2, base, 0.5 * s, centre_colour)
    rw, _ = _size(right, 0.42 * s)
    _put(img, right, w - rw - 10 * s, base, 0.42 * s, MUTED)


def _draw_toast(img, s: float, text: str) -> None:
    """A short message above the help line, wrapped to fit the window."""
    h, w = img.shape[:2]
    scale, line_h = 0.48 * s, 19 * s
    lines = _wrap(text, scale, w - 36 * s)
    widths = [_size(line, scale)[0] for line in lines]
    box_w = max(widths) + 20 * s
    bottom = h - 28 * s
    top = bottom - line_h * len(lines) - 8 * s
    _blend_rect(img, (w - box_w) / 2, top, (w + box_w) / 2, bottom, PANEL, 0.75)
    for i, (line, lw) in enumerate(zip(lines, widths)):
        _put(img, line, (w - lw) / 2, top + line_h * (i + 1), scale, TEXT)


def _draw_help(img, s: float) -> None:
    h, w = img.shape[:2]
    text = "P pause   D details   Q quit"
    tw, _ = _size(text, 0.38 * s)
    _blend_rect(img, 0, h - 20 * s, w, h, PANEL, 0.55)
    _put(img, text, (w - tw) / 2, h - 6 * s, 0.38 * s, MUTED)


def _draw_skeleton(img, landmarks: np.ndarray, colour, s: float) -> None:
    h, w = img.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks]
    thickness = max(1, int(2 * s))
    for a, b in SKELETON:
        cv2.line(img, pts[a], pts[b], colour, thickness, AA)
    for i, p in enumerate(pts):
        cv2.circle(img, p, max(2, int((4 if i in FINGERTIPS else 3) * s)), TEXT if i in FINGERTIPS else colour, -1, AA)


def _draw_trail(img, swipe: SwipeDetector, aspect: float, s: float, now: float) -> None:
    h, w = img.shape[:2]
    pts = [(int(x / aspect * w), int(y * h)) for t, x, y in swipe.samples if now - t < 0.35]
    if len(pts) >= 2:
        colour = TEXT if swipe.stroke is not None else MUTED
        cv2.polylines(img, [np.array(pts, np.int32)], False, colour, max(1, int(3 * s)), AA)


def _draw_hold(img, engine: GestureEngine, feats: HandFeatures, info: HudInfo, aspect: float, s: float) -> None:
    """Progress ring around the hand while a held gesture counts down."""
    pose = engine.hold_pose
    label = info.labels.get(f"{pose.value}_hold") if pose else None
    if label is None or engine.switcher_active or not (engine.hold_progress > 0 or engine.hold_blocked):
        return
    h, w = img.shape[:2]
    if pose is Pose.PINCH:
        x, y = feats.pinch_point
    else:
        x, y = engine.center[0] / aspect, engine.center[1]
    centre = (int(x * w), int(y * h))
    radius = int(max(18 * s, feats.palm * h * 0.6))
    thickness = max(2, int(5 * s))
    if engine.hold_blocked:
        cv2.circle(img, centre, radius, MUTED, max(1, int(2 * s)), AA)
        _pill(img, "start from an open hand", centre[0], centre[1] - radius - 16 * s, 0.42 * s, MUTED)
        return
    colour = POSE_STYLE[pose][1]
    cv2.circle(img, centre, radius, TRACK, thickness, AA)
    cv2.ellipse(img, centre, (radius, radius), -90, 0, 360 * engine.hold_progress, colour, thickness, AA)
    _pill(img, label, centre[0], centre[1] - radius - 16 * s, 0.48 * s, colour)


def _draw_switcher(img, engine: GestureEngine, s: float) -> None:
    """A row of dots, one per app step, with a marker following the fist.
    Drawn near the top, where it is least likely to cover the hand."""
    w = img.shape[1]
    y = 76 * s
    _blend_rect(img, 0, y - 34 * s, w, y + 16 * s, PANEL, 0.7)
    title = "APP SWITCHER - move fist, open hand to pick"
    tw, _ = _size(title, 0.42 * s)
    _put(img, title, (w - tw) / 2, y - 16 * s, 0.42 * s, BLUE)
    spacing, cx = 30 * s, w / 2
    reach = int((w / 2 - 20 * s) // spacing)
    for k in range(-reach, reach + 1):
        selected = k == engine.switcher_steps
        cv2.circle(img, (int(cx + k * spacing), int(y)), max(2, int((7 if selected else 3) * s)),
                   BLUE if selected else MUTED, -1, AA)
    marker = cx + max(-reach - 0.4, min(reach + 0.4, engine.switcher_offset)) * spacing
    cv2.line(img, (int(marker), int(y - 11 * s)), (int(marker), int(y + 11 * s)), TEXT, max(1, int(2 * s)), AA)


def _draw_swipe_flash(img, swipe: SwipeDetector, info: HudInfo, s: float, now: float) -> None:
    """A big fading arrow right after a swipe that has an action."""
    if swipe.last is None:
        return
    direction, t = swipe.last
    age = now - t
    if age > 0.6 or f"swipe_{direction}" not in info.labels:
        return
    h, w = img.shape[:2]
    dx, dy = ARROWS[direction]
    half = 0.16 * (w if dx else h)
    p0 = (int(w / 2 - dx * half), int(h / 2 - dy * half))
    p1 = (int(w / 2 + dx * half), int(h / 2 + dy * half))
    overlay = img.copy()
    cv2.arrowedLine(overlay, p0, p1, TEXT, max(3, int(12 * s)), AA, tipLength=0.4)
    alpha = 0.8 * (1 - age / 0.6)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_legend(img, s: float, labels: dict[str, str]) -> None:
    """What each gesture does - shown while no hand is in view."""
    rows = [(GESTURE_NAMES.get(g, g), a) for g, a in labels.items()]
    if not rows:
        return
    scale = 0.4 * s
    line = 17 * s
    col = max(_size(name, scale)[0] for name, _ in rows) + 14 * s
    width = col + max(_size(action, scale)[0] for _, action in rows) + 20 * s
    top = 36 * s
    _blend_rect(img, 8 * s, top, 8 * s + width, top + line * len(rows) + 10 * s, PANEL, 0.6)
    for i, (name, action) in enumerate(rows):
        y = top + line * (i + 1)
        _put(img, name, 16 * s, y, scale, MUTED)
        _put(img, action, 16 * s + col, y, scale, TEXT)


def _draw_details(img, s: float, feats: HandFeatures | None, engine: GestureEngine, info: HudInfo) -> None:
    """Live numbers for tuning (toggle with D)."""
    lines = [f"{info.fps:4.1f} fps   model {info.infer_ms:.0f} ms"]
    if feats is not None:
        curls = "  ".join(f"{n} {c:.2f}" for n, c in zip("IMRP", feats.curls))
        lines += [
            f"model says {feats.obs.gesture} {feats.obs.gesture_score:.2f}  ({feats.obs.handedness} hand)",
            f"curl  {curls}",
            f"fingers out {feats.extended}   pinch {feats.pinch:.2f}   thumb {feats.thumb_out:.2f}",
            f"palm {engine.palm:.3f}   swipe {'armed' if engine.swipe.armed else 'waiting'}",
        ]
    scale = 0.38 * s
    line = 16 * s
    width = max(_size(t, scale)[0] for t in lines) + 16 * s
    top = 34 * s
    _blend_rect(img, 6 * s, top, 6 * s + width, top + line * len(lines) + 8 * s, PANEL, 0.65)
    for i, text in enumerate(lines):
        _put(img, text, 14 * s, top + line * (i + 1), scale, TEXT)
