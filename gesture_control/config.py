"""Configuration: defaults, loading config.toml, and validation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.toml"


class ConfigError(ValueError):
    """A missing, malformed or invalid configuration file."""


@dataclass
class GeneralConfig:
    pause_hotkey: str = "ctrl+alt+g"
    start_paused: bool = False
    release_camera_when_paused: bool = True


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = True
    backend: str = "msmf"


@dataclass
class TrackingConfig:
    hand: str = "any"
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_path: str = "models/gesture_recognizer.task"


@dataclass
class ActionsConfig:
    swipe_left: str = "next_desktop"
    swipe_right: str = "previous_desktop"
    swipe_up: str = "task_view"
    swipe_down: str = "none"
    pinch_hold: str = "close_window"
    fist_hold: str = "app_switcher"
    victory_hold: str = "app_picker"
    thumb_up_hold: str = "none"
    thumb_down_hold: str = "none"
    love_you_hold: str = "none"


@dataclass
class MouseConfig:
    enabled: bool = True
    click: bool = True
    area_width: float = 0.5
    area_center_x: float = 0.5
    area_center_y: float = 0.45
    smoothing: float = 0.6
    screen: str = "primary"


@dataclass
class AppPickerConfig:
    favorites: list[str] = field(default_factory=list)


@dataclass
class TuningConfig:
    require_open_hand_first: bool = True
    swipe_distance: float = 2.0
    swipe_speed: float = 3.0
    swipe_max_duration: float = 0.7
    swipe_cooldown: float = 0.2
    swipe_return_block: float = 1.0
    pinch_hold_time: float = 0.25
    fist_hold_time: float = 0.15
    gesture_hold_time: float = 0.3
    switcher_step: float = 0.55
    pinch_threshold: float = 0.25
    pose_confirm_time: float = 0.05


@dataclass
class PreviewConfig:
    enabled: bool = True
    always_on_top: bool = True
    show_on_all_desktops: bool = True
    position: str = "bottom-right"
    width: int = 0
    details: bool = False


@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    app_picker: AppPickerConfig = field(default_factory=AppPickerConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)


CHOICES = {
    ("camera", "backend"): ("msmf", "dshow", "any"),
    ("tracking", "hand"): ("any", "left", "right"),
    ("mouse", "screen"): ("primary", "all"),
    ("preview", "position"): ("top-left", "top-right", "bottom-left", "bottom-right"),
}


def load_config(path: Path | str | None = None) -> Config:
    """Load settings from a TOML file on top of the built-in defaults.

    A missing default config.toml just means "use the defaults"; a file that
    was named explicitly must exist.
    """
    cfg = Config()
    explicit = path is not None
    path = Path(path) if explicit else DEFAULT_CONFIG_PATH
    if not path.exists():
        if explicit:
            raise ConfigError(f"config file not found: {path}")
        return cfg
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path.name}: {exc}") from None

    for section_name, values in data.items():
        section = getattr(cfg, section_name, None)
        if section is None or not isinstance(values, dict):
            raise ConfigError(f"{path.name}: unknown section [{section_name}]")
        known = {f.name for f in fields(section)}
        for key, value in values.items():
            if key not in known:
                raise ConfigError(f"{path.name}: unknown setting '{key}' in [{section_name}]")
            where = f"{path.name}: [{section_name}] {key}"
            setattr(section, key, _coerce(value, getattr(section, key), where))
    _validate(cfg, path.name)
    return cfg


def _coerce(value, default, where: str):
    # bool is a subclass of int, so it has to be checked first and excluded below.
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
    elif isinstance(default, int):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif isinstance(default, float):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif isinstance(default, str):
        if isinstance(value, str):
            return value.strip()
    elif isinstance(default, list):
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return [item.strip() for item in value if item.strip()]
    kind = {bool: "true/false", int: "whole number", float: "number", str: "text in quotes",
            list: 'list of names in quotes, like ["Word", "Spotify"]'}[type(default)]
    raise ConfigError(f"{where} should be a {kind}, got {value!r}")


def _validate(cfg: Config, source: str) -> None:
    for (section_name, key), allowed in CHOICES.items():
        section = getattr(cfg, section_name)
        value = getattr(section, key).lower()
        if value not in allowed:
            raise ConfigError(f"{source}: [{section_name}] {key} must be one of {', '.join(allowed)}")
        setattr(section, key, value)
    for f in fields(cfg.tuning):
        value = getattr(cfg.tuning, f.name)
        if isinstance(value, float) and value <= 0:
            raise ConfigError(f"{source}: [tuning] {f.name} must be greater than 0")
    if cfg.camera.width <= 0 or cfg.camera.height <= 0 or cfg.camera.fps <= 0:
        raise ConfigError(f"{source}: [camera] width, height and fps must be positive")
    mouse = cfg.mouse
    if not 0.1 <= mouse.area_width <= 1.0:
        raise ConfigError(f"{source}: [mouse] area_width must be between 0.1 and 1.0")
    for name in ("area_center_x", "area_center_y", "smoothing"):
        if not 0.0 <= getattr(mouse, name) <= 1.0:
            raise ConfigError(f"{source}: [mouse] {name} must be between 0 and 1")
