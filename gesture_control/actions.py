"""What gestures can do. Actions are referred to by name in config.toml."""

from __future__ import annotations

import time

from . import win32

HOTKEY_PREFIX = "hotkey:"

# Actions that are nothing more than a keyboard shortcut.
SHORTCUTS = {
    "next_desktop": "ctrl+win+right",
    "previous_desktop": "ctrl+win+left",
    "new_desktop": "ctrl+win+d",
    "close_desktop": "ctrl+win+f4",
    "task_view": "win+tab",
    "show_desktop": "win+d",
}

LABELS = {
    "next_desktop": "Next desktop",
    "previous_desktop": "Previous desktop",
    "new_desktop": "New desktop",
    "close_desktop": "Close desktop",
    "task_view": "Task View",
    "show_desktop": "Show desktop",
    "close_window": "Close window",
    "minimize_window": "Minimize window",
    "maximize_window": "Maximize window",
    "app_switcher": "App switcher",
    "app_picker": "App picker",
    "none": "Nothing",
}


def pretty_combo(combo: str) -> str:
    return "+".join(part.strip().capitalize() for part in combo.split("+"))


def describe(action: str) -> str:
    if action.startswith(HOTKEY_PREFIX):
        return pretty_combo(action[len(HOTKEY_PREFIX):])
    return LABELS.get(action, action)


def validate(action: str, gesture: str) -> None:
    """Raise ValueError if `action` can't be used for `gesture`."""
    if action.startswith(HOTKEY_PREFIX):
        try:
            win32.parse_combo(action[len(HOTKEY_PREFIX):])
        except ValueError as exc:
            raise ValueError(f"{gesture}: {exc}") from None
        return
    if action not in LABELS:
        raise ValueError(f"{gesture}: unknown action '{action}'. Choose one of: "
                         f"{', '.join(LABELS)}, or hotkey:<keys> such as hotkey:ctrl+c")
    if action == "app_switcher" and gesture != "fist_hold":
        raise ValueError(f"{gesture}: app_switcher only works with fist_hold")


def shorten(text: str, limit: int = 34) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


class AppSwitcher:
    """Drives the Alt+Tab switcher: hold Alt, tap Tab to move, let go of Alt to pick."""

    def __init__(self, keyboard: win32.Keyboard, dry_run: bool = False) -> None:
        self.keyboard = keyboard
        self.dry_run = dry_run
        self.active = False

    def open(self) -> None:
        if self.active:
            return
        self.active = True
        if not self.dry_run:
            self.keyboard.press(win32.VK_MENU)
            time.sleep(0.03)  # let the shell register Alt before Tab arrives
            self.keyboard.tap(win32.VK_TAB)

    def step(self, direction: int) -> None:
        if not self.active or self.dry_run:
            return
        if direction > 0:
            self.keyboard.tap(win32.VK_TAB)
        else:
            self.keyboard.tap(win32.VK_SHIFT, win32.VK_TAB)

    def commit(self) -> None:
        if not self.active:
            return
        self.active = False
        if not self.dry_run:
            self.keyboard.release(win32.VK_MENU)

    def cancel(self) -> None:
        if not self.active:
            return
        self.active = False
        if not self.dry_run:
            self.keyboard.tap(win32.VK_ESCAPE)  # closes the switcher without switching
            self.keyboard.release(win32.VK_MENU)


class Actions:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.keyboard = win32.Keyboard()
        self.mouse = win32.Mouse()
        self.switcher = AppSwitcher(self.keyboard, dry_run)

    def move_mouse(self, x: int, y: int) -> None:
        if not self.dry_run:
            self.mouse.move_to(x, y)

    def mouse_button(self, down: bool) -> None:
        if self.dry_run:
            return
        if down:
            self.mouse.press()
        else:
            self.mouse.release()

    def run(self, action: str) -> str:
        """Perform an action and return a short message saying what happened."""
        if action.startswith(HOTKEY_PREFIX):
            return self._press(action[len(HOTKEY_PREFIX):], "Pressed " + describe(action))
        if action in ("next_desktop", "previous_desktop"):
            return self._switch_desktop(action)
        if action in SHORTCUTS:
            return self._press(SHORTCUTS[action], LABELS[action])
        if action == "close_window":
            return self._window_command("close", lambda win: (win32.SC_CLOSE, "Closed"))
        if action == "minimize_window":
            return self._window_command("minimize", lambda win: (win32.SC_MINIMIZE, "Minimized"))
        if action == "maximize_window":
            return self._window_command("maximize", self._maximize_or_restore)
        return ""

    def release_all(self) -> None:
        """Let go of every key and mouse button we are holding (on exit, on pause, after errors)."""
        self.switcher.active = False
        self.keyboard.release_all()
        self.mouse.release_all()

    def _press(self, combo: str, message: str) -> str:
        if not self.dry_run:
            self.keyboard.combo(combo)
        return message

    def _switch_desktop(self, action: str) -> str:
        message = self._press(SHORTCUTS[action], LABELS[action])
        info = win32.virtual_desktop_info()
        if info is not None and info[1] == 1:
            message = "Only one desktop - press Win+Ctrl+D to add one"
        return message

    @staticmethod
    def _maximize_or_restore(win: win32.WindowInfo) -> tuple[int, str]:
        if win32.is_maximized(win.hwnd):
            return win32.SC_RESTORE, "Restored"
        return win32.SC_MAXIMIZE, "Maximized"

    def _window_command(self, verb: str, command_for) -> str:
        """Send a title-bar command to the active window. `command_for(win)`
        returns the SC_* command and the past tense to report."""
        win = win32.foreground_window()
        if win is None or win32.is_shell_window(win):
            return f"No window to {verb}"
        if win32.is_own_window(win):
            return f"Won't {verb} Gesture Control itself - click another window first"
        title = shorten(win.title or win.class_name)
        command, done = command_for(win)
        if self.dry_run:
            return f"Would {verb}: {title}"
        if not win32.post_syscommand(win.hwnd, command):
            return f"Couldn't {verb} {title} (is it running as administrator?)"
        return f"{done}: {title}"
