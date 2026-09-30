"""Entry point: wires the camera, hand tracker, gesture engine, actions and preview together."""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import time
from dataclasses import fields
from pathlib import Path

import cv2

from . import __version__, actions, win32
from .camera import Camera, CameraError, list_cameras
from .config import PROJECT_DIR, Config, ConfigError, load_config
from .engine import GestureEngine, GestureEvent
from .features import HandFeatures, HandObservation, Pose, PoseClassifier
from .overlay import AMBER, GESTURE_NAMES, RED, TEXT, HudInfo, Preview
from .tracker import HandTracker, ensure_model

log = logging.getLogger("gesture_control")

IDLE_AFTER = 2.0  # seconds without a hand before tracking drops to a lower rate
IDLE_INTERVAL = 0.1  # ...of at most 10 frames per second, to save CPU
TOAST_SECONDS = 2.5
CAMERA_RETRY = 3.0

HOLD_POSES = {
    "pinch_hold": Pose.PINCH,
    "fist_hold": Pose.FIST,
    "thumb_up_hold": Pose.THUMB_UP,
    "thumb_down_hold": Pose.THUMB_DOWN,
    "victory_hold": Pose.VICTORY,
    "pointing_up_hold": Pose.POINTING_UP,
    "love_you_hold": Pose.LOVE_YOU,
}


class GestureApp:
    def __init__(self, cfg: Config, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.mapping = {f.name: getattr(cfg.actions, f.name) for f in fields(cfg.actions)}
        try:
            for gesture, action in self.mapping.items():
                actions.validate(action, gesture)
            if cfg.general.pause_hotkey:
                win32.parse_hotkey(cfg.general.pause_hotkey)
        except ValueError as exc:
            raise ConfigError(str(exc)) from None

        self.actions = actions.Actions(dry_run)
        self.labels = {g: actions.describe(a) for g, a in self.mapping.items() if a != "none"}
        tuning = cfg.tuning
        hold_times = {
            pose: {Pose.PINCH: tuning.pinch_hold_time, Pose.FIST: tuning.fist_hold_time}.get(
                pose, tuning.gesture_hold_time)
            for gesture, pose in HOLD_POSES.items()
            if self.mapping[gesture] != "none"
        }
        self.engine = GestureEngine(tuning, hold_times, switcher=self.mapping["fist_hold"] == "app_switcher")
        self.classifier = PoseClassifier(tuning.pinch_threshold)
        self.hud = HudInfo(dry_run=dry_run, labels=self.labels, details=cfg.preview.details)
        self.commands: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.paused = cfg.general.start_paused
        self.hotkey_name = ""
        self._toast_until = 0.0
        self._desktop_checked = 0.0
        self._announce_at = 0.0  # when to report the window the app switcher landed on

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> int:
        cfg = self.cfg
        model = ensure_model(_resolve(cfg.tracking.model_path))
        log.info("Loading the hand-tracking model...")
        tracker = HandTracker(
            model,
            num_hands=1 if cfg.tracking.hand == "any" else 2,
            min_detection=cfg.tracking.min_detection_confidence,
            min_presence=cfg.tracking.min_presence_confidence,
            min_tracking=cfg.tracking.min_tracking_confidence,
            mirrored=cfg.camera.mirror,
        )
        preview = Preview(cfg.preview, (cfg.camera.width, cfg.camera.height)) if cfg.preview.enabled else None
        hotkey = self._start_hotkey()
        win32.on_console_close(self.actions.release_all)
        self._print_banner()
        try:
            self._loop(tracker, preview)
        except KeyboardInterrupt:
            pass
        finally:
            self.actions.release_all()
            if hotkey is not None:
                hotkey.stop()
            tracker.close()
            if preview is not None:
                preview.close()
            log.info("Stopped.")
        return 0

    def _start_hotkey(self) -> win32.GlobalHotkey | None:
        combo = self.cfg.general.pause_hotkey
        if not combo:
            return None
        hotkey = win32.GlobalHotkey(combo, lambda: self.commands.put("pause"))
        if not hotkey.start():
            log.warning("Couldn't use %s as the pause hotkey - another program has it. "
                        "Pick another in config.toml.", actions.pretty_combo(combo))
            return None
        self.hotkey_name = actions.pretty_combo(combo)
        return hotkey

    def _print_banner(self) -> None:
        mode = " in DRY-RUN mode (gestures are shown but nothing is pressed or closed)" if self.hud.dry_run else ""
        print(f"\nGesture Control {__version__} is running{mode}.\n")
        for gesture, label in self.labels.items():
            if gesture == "fist_hold" and self.mapping[gesture] == "app_switcher":
                label += " (move the fist to choose, open your hand to switch)"
            print(f"  {GESTURE_NAMES[gesture]:<18} {label}")
        pause = f"{self.hotkey_name} (anywhere) or P" if self.hotkey_name else "P in the preview window"
        print(f"\n  Pause / resume     {pause}")
        print("  Quit               Q in the preview window, or Ctrl+C here\n", flush=True)

    # -- main loop -----------------------------------------------------------

    def _loop(self, tracker: HandTracker, preview: Preview | None) -> None:
        camera: Camera | None = None
        seq = 0
        retry_at = 0.0
        last_run = last_hand = 0.0
        try:
            while True:
                if not self._pump(preview):
                    return
                now = time.monotonic()

                if self.paused:
                    if camera is not None and self.cfg.general.release_camera_when_paused:
                        camera.close()
                        camera = None
                    resume = f"Press P or {self.hotkey_name} to resume" if self.hotkey_name else "Press P to resume"
                    self._message(preview, "PAUSED", ("Camera is off. " if camera is None else "") + resume, AMBER)
                    time.sleep(0.05)
                    continue

                if camera is None:
                    if now < retry_at:
                        time.sleep(0.05)
                        continue
                    self._message(preview, "STARTING CAMERA", "One moment...")
                    if not self._pump(preview):  # paints the message before the camera blocks us
                        return
                    try:
                        camera = self._open_camera()
                    except CameraError as exc:
                        log.error("%s", exc)
                        self._message(preview, "NO CAMERA", f"{exc}\nRetrying every few seconds.", RED)
                        retry_at = now + CAMERA_RETRY
                        continue
                    if preview is not None:
                        preview.set_frame_size(*camera.size)
                    seq = 0

                try:
                    item = camera.read(seq, timeout=0.5)
                except CameraError as exc:
                    log.error("%s", exc)
                    self._abandon_gestures()
                    camera.close()
                    camera = None
                    retry_at = now + CAMERA_RETRY
                    continue
                if item is None:
                    continue
                seq, frame, stamp = item
                if self.cfg.camera.mirror:
                    frame = cv2.flip(frame, 1)

                # With no hand around, track less often; the preview still shows every frame.
                if stamp - last_hand > IDLE_AFTER and stamp - last_run < IDLE_INTERVAL:
                    features = None
                else:
                    if last_run:
                        self.hud.fps += 0.1 * (1 / max(stamp - last_run, 1e-3) - self.hud.fps)
                    last_run = stamp
                    features = self._track(tracker, frame, stamp)
                    if features is not None:
                        last_hand = stamp
                    for event in self.engine.update(stamp, features):
                        self._dispatch(event)

                if preview is not None:
                    self._refresh_hud(features)
                    preview.show(frame, self.engine, self.hud, stamp)
        finally:
            if camera is not None:
                camera.close()

    def _pump(self, preview: Preview | None) -> bool:
        """Handle hotkey presses and preview-window keys. False means quit."""
        while not self.commands.empty():
            command = self.commands.get()
            if command == "quit":
                return False
            if command == "pause":
                self._set_paused(not self.paused)
        if preview is None:
            return True
        key = preview.poll_key(1)
        if key == ord("q") or not preview.is_open():
            return False
        if key in (ord("p"), ord(" ")):
            self._set_paused(not self.paused)
        elif key == ord("d"):
            self.hud.details = not self.hud.details
        return True

    def _open_camera(self) -> Camera:
        c = self.cfg.camera
        camera = Camera(c.index, c.width, c.height, c.fps, c.backend)
        camera.open()
        return camera

    def _track(self, tracker: HandTracker, frame, stamp: float) -> HandFeatures | None:
        start = time.perf_counter()
        hands = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), stamp)
        self.hud.infer_ms = (time.perf_counter() - start) * 1000
        obs = self._pick_hand(hands)
        if obs is None:
            self.classifier.reset()
            return None
        h, w = frame.shape[:2]
        return self.classifier(obs, w / h)

    def _pick_hand(self, hands: list[HandObservation]) -> HandObservation | None:
        wanted = self.cfg.tracking.hand
        for obs in hands:
            if wanted == "any" or obs.handedness.lower() == wanted:
                return obs
        return None

    # -- reacting to gestures ----------------------------------------------

    def _dispatch(self, event: GestureEvent) -> None:
        if event.kind == "switcher":
            switcher = self.actions.switcher
            if event.name == "open":
                switcher.open()
            elif event.name == "step":
                switcher.step(event.step)
            elif event.name == "commit":
                switcher.commit()
                if self.hud.dry_run:
                    self._toast("Would switch to the selected app")
                else:
                    self._announce_at = time.monotonic() + 0.3  # once Windows has switched
            else:
                switcher.cancel()
                self._toast("App switcher cancelled")
            if event.name != "step":
                log.info("App switcher: %s", event.name)
            return
        action = self.mapping.get(event.name, "none")
        if action == "none":
            return
        message = self.actions.run(action)
        log.info("%-18s -> %s", GESTURE_NAMES.get(event.name, event.name), message)
        self._toast(message)

    def _set_paused(self, paused: bool) -> None:
        if paused == self.paused:
            return
        self.paused = paused
        if paused:
            self._abandon_gestures()
            log.info("Paused.")
        else:
            log.info("Resumed.")

    def _abandon_gestures(self) -> None:
        for event in self.engine.cancel(time.monotonic()):
            self._dispatch(event)
        self.actions.release_all()
        self.classifier.reset()

    # -- preview helpers -----------------------------------------------------

    def _toast(self, message: str) -> None:
        if self.hud.dry_run and not message.startswith("Would"):
            message = f"[dry run] {message}"
        self.hud.toast = message
        self._toast_until = time.monotonic() + TOAST_SECONDS

    def _refresh_hud(self, features: HandFeatures | None) -> None:
        now = time.monotonic()
        self.hud.features = features
        if now - self._desktop_checked > 0.5:
            self.hud.desktop = win32.virtual_desktop_info()
            self._desktop_checked = now
        if self._announce_at and now >= self._announce_at:
            self._announce_at = 0.0
            win = win32.foreground_window()
            self._toast(f"Switched to {actions.shorten(win.title)}" if win and win.title else "Switched app")
        if now > self._toast_until:
            self.hud.toast = None

    @staticmethod
    def _message(preview: Preview | None, title: str, body: str, colour=TEXT) -> None:
        if preview is not None:
            preview.show_message(title, body, colour)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gesture_control", description="Control Windows with hand gestures seen by your webcam.")
    parser.add_argument("--config", type=Path, help="settings file (default: config.toml next to the program)")
    parser.add_argument("--camera", type=int, metavar="N", help="use camera number N (see --list-cameras)")
    parser.add_argument("--no-preview", action="store_true", help="run without the preview window")
    parser.add_argument("--dry-run", action="store_true",
                        help="recognise gestures but don't press keys or close anything")
    parser.add_argument("--details", action="store_true", help="show live tuning numbers in the preview")
    parser.add_argument("--list-cameras", action="store_true", help="list the cameras that can be opened, then exit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # window titles can contain any character
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error("Settings problem - %s", exc)
        return 2

    if args.list_cameras:
        cameras = list_cameras(cfg.camera.backend)
        for index, w, h in cameras:
            print(f"  camera {index}: {w}x{h}")
        if not cameras:
            print("  No cameras could be opened.")
        return 0

    try:
        if args.camera is not None:
            cfg.camera.index = args.camera
        if args.no_preview:
            cfg.preview.enabled = False
        if args.details:
            cfg.preview.details = True
        app = GestureApp(cfg, dry_run=args.dry_run)
    except ConfigError as exc:
        log.error("Settings problem - %s", exc)
        return 2

    win32.enable_dpi_awareness()
    try:
        return app.run()
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
