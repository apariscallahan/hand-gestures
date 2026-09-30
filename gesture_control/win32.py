"""Thin ctypes wrappers for the Win32 APIs this program needs.

Everything Windows-specific lives here: synthesized keyboard input, window
queries, DPI awareness, styling the preview window, the global pause hotkey
and virtual-desktop info.
"""

from __future__ import annotations

import ctypes
import os
import threading
import winreg
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LONG_PTR = ctypes.c_ssize_t
ULONG_PTR = ctypes.c_size_t

# ---------------------------------------------------------------------------
# Keyboard input
# ---------------------------------------------------------------------------

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
MAPVK_VK_TO_VSC = 0
# Written to dwExtraInfo so our synthesized events can be told apart from real ones.
INJECTED_TAG = 0x47455354

VK_TAB = 0x09
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # Alt
VK_ESCAPE = 0x1B
VK_LWIN = 0x5B


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    # MOUSEINPUT is the largest member; it must be present for sizeof(INPUT) to be right.
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT
user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
user32.MapVirtualKeyW.restype = wintypes.UINT

VK_NAMES: dict[str, int] = {
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "return": 0x0D,
    "shift": 0x10, "ctrl": 0x11, "control": 0x11, "alt": 0x12,
    "pause": 0x13, "capslock": 0x14, "esc": 0x1B, "escape": 0x1B, "space": 0x20,
    "pageup": 0x21, "pagedown": 0x22, "end": 0x23, "home": 0x24,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "printscreen": 0x2C, "insert": 0x2D, "delete": 0x2E, "del": 0x2E,
    "win": 0x5B, "lwin": 0x5B, "rwin": 0x5C, "apps": 0x5D,
    "numlock": 0x90, "scrolllock": 0x91,
    "browserback": 0xA6, "browserforward": 0xA7, "browserrefresh": 0xA8,
    "volumemute": 0xAD, "volumedown": 0xAE, "volumeup": 0xAF,
    "nexttrack": 0xB0, "prevtrack": 0xB1, "stopmedia": 0xB2, "playpause": 0xB3,
    ";": 0xBA, "plus": 0xBB, "=": 0xBB, ",": 0xBC, "minus": 0xBD, "-": 0xBD,
    ".": 0xBE, "/": 0xBF, "`": 0xC0, "[": 0xDB, "\\": 0xDC, "]": 0xDD, "'": 0xDE,
}

# Keys that live on the "extended" part of the keyboard. Without the extended
# flag Windows reads the arrow keys as numpad keys and Win+X combos can fail.
EXTENDED_VKS = frozenset(
    {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2C, 0x2D, 0x2E,
     0x5B, 0x5C, 0x5D, 0x6F, 0x90, 0xA3, 0xA5, *range(0xA6, 0xB8)}
)


def vk_from_name(name: str) -> int:
    key = name.strip().lower()
    if key in VK_NAMES:
        return VK_NAMES[key]
    if len(key) == 1 and key.isascii() and key.isalnum():
        return ord(key.upper())
    if key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        return 0x6F + int(key[1:])
    raise ValueError(f"unknown key name {name!r}")


def parse_combo(combo: str) -> list[int]:
    """'ctrl+win+right' -> [VK_CONTROL, VK_LWIN, VK_RIGHT]."""
    parts = [p for p in combo.replace(" ", "").split("+") if p]
    if not parts:
        raise ValueError("empty key combination")
    return [vk_from_name(p) for p in parts]


class Keyboard:
    """Synthesizes key presses and remembers which keys it is holding down,
    so they can always be released on exit (a stuck Alt key is miserable)."""

    def __init__(self) -> None:
        self._held: list[int] = []
        self._lock = threading.Lock()

    def _send(self, events: list[tuple[int, bool]]) -> bool:
        inputs = (INPUT * len(events))()
        for item, (vk, up) in zip(inputs, events):
            flags = KEYEVENTF_KEYUP if up else 0
            if vk in EXTENDED_VKS:
                flags |= KEYEVENTF_EXTENDEDKEY
            item.type = INPUT_KEYBOARD
            item.ki = KEYBDINPUT(vk, user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC), flags, 0, INJECTED_TAG)
        return user32.SendInput(len(events), inputs, ctypes.sizeof(INPUT)) == len(events)

    def press(self, vk: int) -> bool:
        with self._lock:
            self._held.append(vk)
            return self._send([(vk, False)])

    def release(self, vk: int) -> bool:
        with self._lock:
            if vk in self._held:
                self._held.remove(vk)
            return self._send([(vk, True)])

    def tap(self, *vks: int) -> bool:
        """Press the keys in order and release them in reverse, as one atomic batch."""
        with self._lock:
            return self._send([(vk, False) for vk in vks] + [(vk, True) for vk in reversed(vks)])

    def combo(self, combo: str) -> bool:
        return self.tap(*parse_combo(combo))

    def release_all(self) -> None:
        with self._lock:
            held, self._held = self._held, []
            if held:
                self._send([(vk, True) for vk in reversed(held)])


# ---------------------------------------------------------------------------
# Windows (as in top-level windows)
# ---------------------------------------------------------------------------

WM_SYSCOMMAND = 0x0112
SC_CLOSE = 0xF060
SC_MINIMIZE = 0xF020
SC_MAXIMIZE = 0xF030
SC_RESTORE = 0xF120
GW_OWNER = 4
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
SW_HIDE = 0
SW_SHOWNOACTIVATE = 4
HWND_TOPMOST = wintypes.HWND(-1)
HWND_NOTOPMOST = wintypes.HWND(-2)
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
SPI_GETWORKAREA = 0x0030

# The desktop background and the taskbar. Alt+F4 on these opens the
# "Shut Down Windows" dialog, so close/minimize must never target them.
SHELL_CLASSES = frozenset({"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"})

user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
user32.PostMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
user32.PostMessageW.restype = wintypes.BOOL
user32.IsZoomed.argtypes = (wintypes.HWND,)
user32.GetWindow.argtypes = (wintypes.HWND, wintypes.UINT)
user32.GetWindow.restype = wintypes.HWND
user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
user32.FindWindowW.restype = wintypes.HWND
user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
user32.SetWindowPos.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, wintypes.UINT)
user32.SystemParametersInfoW.argtypes = (wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT)
kernel32.GetConsoleWindow.restype = wintypes.HWND

# 32-bit Windows has no *LongPtr functions; the plain versions are equivalent there.
_GetWindowLongPtr = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
_SetWindowLongPtr = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
_GetWindowLongPtr.argtypes = (wintypes.HWND, ctypes.c_int)
_GetWindowLongPtr.restype = LONG_PTR
_SetWindowLongPtr.argtypes = (wintypes.HWND, ctypes.c_int, LONG_PTR)
_SetWindowLongPtr.restype = LONG_PTR


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    pid: int


def window_info(hwnd: int) -> WindowInfo:
    cls = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, cls, len(cls))
    title = ctypes.create_unicode_buffer(user32.GetWindowTextLengthW(hwnd) + 1)
    user32.GetWindowTextW(hwnd, title, len(title))
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return WindowInfo(hwnd, title.value, cls.value, pid.value)


def foreground_window() -> WindowInfo | None:
    hwnd = user32.GetForegroundWindow()
    return window_info(hwnd) if hwnd else None


def is_shell_window(win: WindowInfo) -> bool:
    return win.class_name in SHELL_CLASSES


def is_own_window(win: WindowInfo) -> bool:
    """True for our preview window and for the console window we run in."""
    if win.pid == os.getpid():
        return True
    console = kernel32.GetConsoleWindow()
    if console:
        # In Windows Terminal the console window is hidden and owned by the terminal window.
        return win.hwnd in (console, user32.GetWindow(console, GW_OWNER))
    return False


def post_syscommand(hwnd: int, command: int) -> bool:
    """Ask a window to close/minimize/etc. exactly as its title-bar buttons do."""
    return bool(user32.PostMessageW(hwnd, WM_SYSCOMMAND, command, 0))


def is_maximized(hwnd: int) -> bool:
    return bool(user32.IsZoomed(hwnd))


def find_window(title: str) -> int | None:
    return user32.FindWindowW(None, title) or None


def work_area() -> tuple[int, int, int, int]:
    """Primary monitor's usable area (left, top, right, bottom), excluding the taskbar."""
    rect = wintypes.RECT()
    if user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
        return rect.left, rect.top, rect.right, rect.bottom
    return 0, 0, user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def dpi_scale(hwnd: int | None = None) -> float:
    """Display scaling for a window (or the system), e.g. 1.5 at 150%."""
    try:
        if hwnd:
            user32.GetDpiForWindow.argtypes = (wintypes.HWND,)
            dpi = user32.GetDpiForWindow(hwnd)
        else:
            dpi = user32.GetDpiForSystem()
    except AttributeError:
        return 1.0
    return dpi / 96 if dpi else 1.0


def enable_dpi_awareness() -> None:
    """Report real pixel sizes so window placement is right on scaled displays."""
    try:
        user32.SetProcessDpiAwarenessContext.argtypes = (ctypes.c_void_p,)
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):  # PER_MONITOR_AWARE_V2
            return
    except AttributeError:
        pass
    try:
        ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        pass


def style_preview_window(hwnd: int, *, topmost: bool, all_desktops: bool) -> None:
    """Keep the preview above other windows and, by making it a tool window,
    out of the taskbar and Alt+Tab list and visible on every virtual desktop."""
    if all_desktops:
        style = _GetWindowLongPtr(hwnd, GWL_EXSTYLE)
        user32.ShowWindow(hwnd, SW_HIDE)  # the taskbar only notices the change on re-show
        _SetWindowLongPtr(hwnd, GWL_EXSTYLE, (style | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW)
        user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
    user32.SetWindowPos(hwnd, HWND_TOPMOST if topmost else HWND_NOTOPMOST, 0, 0, 0, 0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED)


def place_window(hwnd: int, corner: str, margin: int = 16) -> None:
    """Move a window (by its outer frame) into a corner of the primary work area."""
    left, top, right, bottom = work_area()
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return
    w, h = rect.right - rect.left, rect.bottom - rect.top
    x = left + margin if "left" in corner else right - w - margin
    y = top + margin if "top" in corner else bottom - h - margin
    user32.SetWindowPos(hwnd, None, x, y, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOZORDER)


# ---------------------------------------------------------------------------
# Virtual desktops
# ---------------------------------------------------------------------------

_VD_KEY = r"Software\Microsoft\Windows\CurrentVersion\Explorer\VirtualDesktops"


def virtual_desktop_info() -> tuple[int | None, int] | None:
    """(current desktop number or None, desktop count), best effort.

    Windows has no public API for this, but Explorer mirrors it in the
    registry. It only writes the values once there is more than one desktop.
    Returns None if the registry can't be read at all.
    """
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _VD_KEY) as key:
            try:
                ids = winreg.QueryValueEx(key, "VirtualDesktopIDs")[0]
            except FileNotFoundError:
                return 1, 1
            current = _query(key, "CurrentVirtualDesktop")
        if current is None:  # Windows 10 keeps it per session
            session = wintypes.DWORD()
            kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(session))
            sub = rf"Software\Microsoft\Windows\CurrentVersion\Explorer\SessionInfo\{session.value}\VirtualDesktops"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub) as key:
                current = _query(key, "CurrentVirtualDesktop")
    except OSError:
        return None
    guids = [bytes(ids[i:i + 16]) for i in range(0, len(ids) - 15, 16)]
    count = max(1, len(guids))
    index = guids.index(bytes(current)) + 1 if current and bytes(current) in guids else None
    return index, count


def _query(key, name: str):
    try:
        return winreg.QueryValueEx(key, name)[0]
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Global hotkey (works no matter which window has focus)
# ---------------------------------------------------------------------------

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000
MODIFIER_FLAGS = {"alt": MOD_ALT, "ctrl": MOD_CONTROL, "control": MOD_CONTROL, "shift": MOD_SHIFT, "win": MOD_WIN}
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

user32.RegisterHotKey.argtypes = (wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT)
user32.UnregisterHotKey.argtypes = (wintypes.HWND, ctypes.c_int)
user32.GetMessageW.argtypes = (ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT)
user32.PeekMessageW.argtypes = (ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT,
                                wintypes.UINT, wintypes.UINT)
user32.PostThreadMessageW.argtypes = (wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


def parse_hotkey(combo: str) -> tuple[int, int]:
    """'ctrl+alt+g' -> (MOD_CONTROL | MOD_ALT, ord('G'))."""
    *mods, key = [p for p in combo.replace(" ", "").lower().split("+") if p] or [""]
    if not key or key in MODIFIER_FLAGS:
        raise ValueError(f"hotkey {combo!r} needs a non-modifier key, e.g. ctrl+alt+g")
    flags = 0
    for mod in mods:
        if mod not in MODIFIER_FLAGS:
            raise ValueError(f"unknown modifier {mod!r} in hotkey {combo!r}")
        flags |= MODIFIER_FLAGS[mod]
    return flags, vk_from_name(key)


class GlobalHotkey:
    """Calls `callback` (on a background thread) whenever the hotkey is pressed."""

    def __init__(self, combo: str, callback: Callable[[], None]) -> None:
        self.combo = combo
        self._mods, self._vk = parse_hotkey(combo)
        self._callback = callback
        self._thread = threading.Thread(target=self._run, name="hotkey", daemon=True)
        self._ready = threading.Event()
        self._thread_id = 0
        self.registered = False

    def start(self) -> bool:
        self._thread.start()
        self._ready.wait(timeout=2)
        return self.registered

    def stop(self) -> None:
        if self._thread_id:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)

    def _run(self) -> None:
        # Hotkey messages go to the registering thread's queue, so this thread
        # registers the key and then pumps messages until it is told to quit.
        self._thread_id = kernel32.GetCurrentThreadId()
        msg = wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)  # creates the message queue
        self.registered = bool(user32.RegisterHotKey(None, 1, self._mods | MOD_NOREPEAT, self._vk))
        self._ready.set()
        if not self.registered:
            return
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY:
                    self._callback()
        finally:
            user32.UnregisterHotKey(None, 1)


# ---------------------------------------------------------------------------
# Console close
# ---------------------------------------------------------------------------

_HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
_console_handler = None  # keeps the callback alive


def on_console_close(callback: Callable[[], None]) -> None:
    """Run `callback` if the console window is closed or the user logs off.

    Python gets no chance to run `finally` blocks in those cases, which would
    otherwise leave a synthesized Alt key held down.
    """
    global _console_handler

    def handler(ctrl_type: int) -> bool:
        if ctrl_type in (2, 5, 6):  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT
            try:
                callback()
            except Exception:
                pass
        return False  # let the default handling continue

    _console_handler = _HandlerRoutine(handler)
    kernel32.SetConsoleCtrlHandler(_console_handler, True)
