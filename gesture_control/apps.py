"""The apps the app picker offers: everything in the Start menu, plus favourites.

Apps are listed with PowerShell's Get-StartApps, which covers both ordinary
programs and Microsoft Store apps, and are launched through the shell's
AppsFolder, the same way the Start menu launches them. Icons come from the
shell too. Loading happens once, on a background thread, at start-up.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import logging
import os
import subprocess
import threading
import uuid
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

ICON_SIZE = 96
MAX_FAVORITES = 18
# Used to fill up the favourites after the apps pinned to the taskbar.
DEFAULT_FAVORITES = (
    "File Explorer", "Microsoft Edge", "Google Chrome", "Firefox", "Brave", "Settings",
    "Microsoft Store", "Calculator", "Notepad", "Photos", "Spotify", "Discord", "Steam",
    "Visual Studio Code", "Word", "Excel", "Outlook", "Teams",
)
# Start-menu entries that aren't apps you'd want to open this way.
_JUNK_WORDS = ("uninstall", "readme", "read me", "release notes", "license", "documentation", "manual",
               "website")
_JUNK_SUFFIXES = (".url", ".txt", ".chm", ".pdf", ".htm", ".html", ".rtf", ".md", ".hlp")


@dataclass
class App:
    name: str
    app_id: str  # the Start menu's AppUserModelID or path
    icon: np.ndarray | None = field(default=None, repr=False, compare=False)  # RGBA, ICON_SIZE square


def is_launchable(name: str, app_id: str) -> bool:
    target = app_id.casefold()
    if target.startswith(("http:", "https:")) or target.endswith(_JUNK_SUFFIXES):
        return False
    return not any(word in name.casefold() for word in _JUNK_WORDS)


def pick_favorites(apps: list[App], wanted: list[str] | tuple[str, ...]) -> list[App]:
    """The apps named in `wanted`, in that order. An exact name wins; otherwise
    the shortest app name containing it ("Chrome" finds "Google Chrome")."""
    chosen: list[App] = []
    for want in wanted:
        key = want.casefold()
        match = next((a for a in apps if a.name.casefold() == key), None)
        if match is None:
            candidates = [a for a in apps if key in a.name.casefold()]
            match = min(candidates, key=lambda a: len(a.name)) if candidates else None
        if match is not None and match not in chosen:
            chosen.append(match)
    return chosen[:MAX_FAVORITES]


def taskbar_pins() -> list[str]:
    """Names of the apps pinned to the taskbar (the shortcuts Explorer keeps for them)."""
    folder = Path(os.environ.get("APPDATA", "")) / "Microsoft/Internet Explorer/Quick Launch/User Pinned/TaskBar"
    return [p.stem for p in sorted(folder.glob("*.lnk"))] if folder.is_dir() else []


def start_menu_apps() -> list[App]:
    command = ("[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
               "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
    data = json.loads(result.stdout.decode("utf-8-sig").strip() or "[]")
    if isinstance(data, dict):  # a single app comes back as an object, not a list
        data = [data]
    apps, seen = [], set()
    for item in data:
        name, app_id = (item.get("Name") or "").strip(), (item.get("AppID") or "").strip()
        if name and app_id and is_launchable(name, app_id) and name.casefold() not in seen:
            seen.add(name.casefold())
            apps.append(App(name, app_id))
    return sorted(apps, key=lambda a: a.name.casefold())


def launch(app: App) -> None:
    os.startfile(f"shell:AppsFolder\\{app.app_id}")


class AppCatalog:
    """Loads the app list in the background, then their icons.

    `ready` is set as soon as the list is known (about a second); icons keep
    arriving for a few seconds more, and `version` goes up as they do, so the
    picker knows when to redraw.
    """

    def __init__(self, favorites: list[str]) -> None:
        self._wanted = favorites
        self.apps: list[App] = []
        self.favorites: list[App] = []
        self.ready = threading.Event()
        self.version = 0
        self.error: str | None = None

    def load_in_background(self) -> None:
        threading.Thread(target=self._load, name="apps", daemon=True).start()

    def _load(self) -> None:
        try:
            apps = start_menu_apps()
            self.favorites = pick_favorites(apps, self._wanted or [*taskbar_pins(), *DEFAULT_FAVORITES])
            self.apps = apps
            self.ready.set()
            with _com():
                for i, app in enumerate(self.favorites + apps, 1):  # favourites first: they are shown first
                    if app.icon is None:
                        app.icon = app_icon(app.app_id)
                    if i % 8 == 0:
                        self.version += 1
            self.version += 1
            log.info("App picker: found %d apps, %d favourites.", len(apps), len(self.favorites))
        except Exception as exc:  # the picker shows the problem instead of the apps
            self.error = f"Couldn't list your apps: {exc}"
            log.warning("%s", self.error)
        finally:
            self.ready.set()


# ---------------------------------------------------------------------------
# App icons, from the shell (IShellItemImageFactory)
# ---------------------------------------------------------------------------

class _GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]


class _SIZE(ctypes.Structure):
    _fields_ = [("cx", wintypes.LONG), ("cy", wintypes.LONG)]


class _BITMAP(ctypes.Structure):
    _fields_ = [("bmType", wintypes.LONG), ("bmWidth", wintypes.LONG), ("bmHeight", wintypes.LONG),
                ("bmWidthBytes", wintypes.LONG), ("bmPlanes", wintypes.WORD), ("bmBitsPixel", wintypes.WORD),
                ("bmBits", ctypes.c_void_p)]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


_shell32 = ctypes.WinDLL("shell32")
_ole32 = ctypes.WinDLL("ole32")
_gdi32 = ctypes.WinDLL("gdi32")
_user32 = ctypes.WinDLL("user32")
_shell32.SHCreateItemFromParsingName.argtypes = (wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(_GUID),
                                                 ctypes.POINTER(ctypes.c_void_p))
_shell32.SHCreateItemFromParsingName.restype = ctypes.c_long
_ole32.CoInitializeEx.argtypes = (ctypes.c_void_p, wintypes.DWORD)
_ole32.CoInitializeEx.restype = ctypes.c_long
_gdi32.GetObjectW.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p)
_gdi32.GetDIBits.argtypes = (wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p,
                             ctypes.POINTER(_BITMAPINFOHEADER), wintypes.UINT)
_gdi32.DeleteObject.argtypes = (wintypes.HGDIOBJ,)
_user32.GetDC.argtypes = (wintypes.HWND,)
_user32.GetDC.restype = wintypes.HDC
_user32.ReleaseDC.argtypes = (wintypes.HWND, wintypes.HDC)

_IID_IShellItemImageFactory = _GUID.from_buffer_copy(uuid.UUID("bcc18b79-ba16-442f-80c4-8a59c30c463b").bytes_le)
_GetImage = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, _SIZE, ctypes.c_int, ctypes.POINTER(wintypes.HBITMAP))
_Release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
SIIGBF_BIGGERSIZEOK, SIIGBF_ICONONLY = 0x1, 0x4


@contextlib.contextmanager
def _com():
    hr = _ole32.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED, as the shell expects
    try:
        yield
    finally:
        if hr >= 0:
            _ole32.CoUninitialize()


def app_icon(app_id: str, size: int = ICON_SIZE) -> np.ndarray | None:
    """The app's icon as a size x size RGBA array, or None if the shell has none."""
    factory = ctypes.c_void_p()
    hr = _shell32.SHCreateItemFromParsingName(f"shell:AppsFolder\\{app_id}", None,
                                              ctypes.byref(_IID_IShellItemImageFactory), ctypes.byref(factory))
    if hr != 0 or not factory:
        return None
    vtable = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    try:
        bitmap = wintypes.HBITMAP()
        hr = _GetImage(vtable[3])(factory, _SIZE(size, size), SIIGBF_ICONONLY | SIIGBF_BIGGERSIZEOK,
                                  ctypes.byref(bitmap))
        if hr != 0 or not bitmap:
            return None
        try:
            return _bitmap_to_rgba(bitmap, size)
        finally:
            _gdi32.DeleteObject(bitmap)
    finally:
        _Release(vtable[2])(factory)


def _bitmap_to_rgba(bitmap: wintypes.HBITMAP, size: int) -> np.ndarray | None:
    info = _BITMAP()
    _gdi32.GetObjectW(bitmap, ctypes.sizeof(info), ctypes.byref(info))
    w, h = info.bmWidth, abs(info.bmHeight)
    if w <= 0 or h <= 0:
        return None
    header = _BITMAPINFOHEADER(biSize=ctypes.sizeof(_BITMAPINFOHEADER), biWidth=w, biHeight=-h,  # top-down
                               biPlanes=1, biBitCount=32)
    pixels = (ctypes.c_ubyte * (w * h * 4))()
    dc = _user32.GetDC(None)
    try:
        lines = _gdi32.GetDIBits(dc, bitmap, 0, h, pixels, ctypes.byref(header), 0)
    finally:
        _user32.ReleaseDC(None, dc)
    if lines != h:
        return None
    bgra = np.frombuffer(pixels, np.uint8).reshape(h, w, 4).astype(np.float32)
    alpha = bgra[..., 3:]
    if alpha.max() == 0:
        bgra[..., 3] = 255  # no transparency information: fully opaque
    elif not (bgra[..., :3] > alpha + 1).any():
        # Colour never exceeds alpha: the shell gave premultiplied alpha. Undo it.
        bgra[..., :3] = np.where(alpha > 0, bgra[..., :3] * 255 / np.maximum(alpha, 1), 0)
    rgba = np.clip(bgra[..., [2, 1, 0, 3]], 0, 255).astype(np.uint8)
    if (w, h) != (size, size):
        rgba = cv2.resize(rgba, (size, size), interpolation=cv2.INTER_AREA)
    return rgba
