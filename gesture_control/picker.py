"""The app picker: a full-screen grid of big app tiles that is easy to hit
with the finger mouse, and to page through with swipes."""

from __future__ import annotations

import logging
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import win32
from .apps import App, AppCatalog

log = logging.getLogger(__name__)

WINDOW = "Gesture Control - apps"
FONTS = Path(r"C:\Windows\Fonts")

# Layout at 1920x1080; everything scales with the screen.
REF_W, REF_H = 1920, 1080
SIDE, TOP, BOTTOM, GAP = 150, 150, 170, 22
GRIDS = {"favorites": (6, 3), "all": (7, 4)}  # columns, rows

BACKGROUND = (15, 17, 22)
TILE = (34, 38, 47)
TILE_HOVER = (52, 59, 73)
TILE_PRESSED = (40, 86, 170)
ACCENT = (86, 152, 255)
TEXT = (238, 240, 244)
MUTED = (150, 156, 168)
DISABLED = (70, 75, 86)


@dataclass(frozen=True)
class Target:
    """Something clickable on the page."""

    kind: str  # "app", "tab", "page", "letter" or "close"
    rect: tuple[int, int, int, int]  # x0, y0, x1, y1
    app: App | None = None
    value: object = None  # tab name, page step (+1/-1) or page number
    label: str = ""
    enabled: bool = True
    selected: bool = False

    def contains(self, x: int, y: int) -> bool:
        x0, y0, x1, y1 = self.rect
        return x0 <= x < x1 and y0 <= y < y1


def letter_of(app: App) -> str:
    first = app.name[:1].upper()
    return first if first.isalpha() else "#"


def page_count(apps: list[App], tab: str) -> int:
    cols, rows = GRIDS[tab]
    return max(1, -(-len(apps) // (cols * rows)))


def layout(width: int, height: int, tab: str, page: int, favorites: list[App],
           apps: list[App]) -> list[Target]:
    """Everything clickable on one page of the picker, for a screen of the given size."""
    s = min(width / REF_W, height / REF_H)
    shown = favorites if tab == "favorites" else apps
    pages = page_count(shown, tab)
    page = max(0, min(page, pages - 1))
    targets: list[Target] = []

    # Header: tabs in the middle, close button on the right.
    tab_w, tab_h, top = round(210 * s), round(58 * s), round(44 * s)
    x = width // 2 - tab_w - round(8 * s)
    for name, label in (("favorites", "Favorites"), ("all", "All apps")):
        targets.append(Target("tab", (x, top, x + tab_w, top + tab_h), value=name, label=label,
                              selected=name == tab))
        x += tab_w + round(16 * s)
    targets.append(Target("close", (width - round(108 * s), round(38 * s), width - round(38 * s), round(108 * s)),
                          label="\u00d7"))

    # The grid of app tiles.
    cols, rows = GRIDS[tab]
    x0, y0 = round(SIDE * s), round(TOP * s)
    tile_w = (width - 2 * x0 - (cols - 1) * GAP * s) / cols
    tile_h = (height - y0 - BOTTOM * s - (rows - 1) * GAP * s) / rows
    per_page = cols * rows
    for i, app in enumerate(shown[page * per_page:(page + 1) * per_page]):
        c, r = i % cols, i // cols
        tx, ty = x0 + c * (tile_w + GAP * s), y0 + r * (tile_h + GAP * s)
        targets.append(Target("app", (round(tx), round(ty), round(tx + tile_w), round(ty + tile_h)), app=app,
                              label=app.name))

    # Big page arrows on both sides.
    if pages > 1:
        mid, half, w = height // 2, round(120 * s), round(86 * s)
        edge = round(32 * s)
        targets.append(Target("page", (edge, mid - half, edge + w, mid + half), value=-1, label="\u2039",
                              enabled=page > 0))
        targets.append(Target("page", (width - edge - w, mid - half, width - edge, mid + half), value=+1,
                              label="\u203a", enabled=page < pages - 1))

    # A strip of letters to jump through the full list.
    if tab == "all" and apps:
        first_page: dict[str, int] = {}
        for i, app in enumerate(apps):
            first_page.setdefault(letter_of(app), i // per_page)
        on_page = {letter_of(a) for a in apps[page * per_page:(page + 1) * per_page]}
        size, gap = round(50 * s), round(8 * s)
        x = (width - len(first_page) * (size + gap) + gap) // 2
        y = height - round(128 * s)
        for letter, target_page in first_page.items():
            targets.append(Target("letter", (x, y, x + size, y + size), value=target_page, label=letter,
                                  selected=letter in on_page))
            x += size + gap
    return targets


class AppPicker:
    """The picker window. Driven by the main loop: open(), close(), refresh()
    every frame, next_page()/previous_page() for swipes, handle_key() for keys."""

    def __init__(self, catalog: AppCatalog, on_launch: Callable[[App], None]) -> None:
        self.catalog = catalog
        self.on_launch = on_launch
        self.is_open = False
        self.hwnd: int | None = None
        self.tab = "favorites"
        self.page = 0
        self.size = (REF_W, REF_H)
        self._targets: list[Target] = []
        self._base: Image.Image | None = None
        self._hover: Target | None = None
        self._pressed: Target | None = None
        self._drawn_state: tuple[int, bool] | None = None  # catalog (version, ready) when last drawn
        self._dirty = False

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        if self.is_open:
            return
        _, _, width, height = win32.screen_rect("primary")
        self.size = (width, height)
        self.tab = "favorites" if self.catalog.favorites or not self.catalog.ready.is_set() else "all"
        self.page = 0
        self._hover = self._pressed = None
        self._render()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        cv2.imshow(WINDOW, self._compose())
        cv2.waitKey(1)
        self.hwnd = win32.find_window(WINDOW)
        if self.hwnd:
            win32.style_overlay_window(self.hwnd, alpha=245)
            win32.set_foreground(self.hwnd)
        self.is_open = True

    def close(self) -> None:
        if not self.is_open:
            return
        self.is_open = False
        self.hwnd = None
        cv2.destroyWindow(WINDOW)

    def refresh(self) -> None:
        """Redraw if anything changed (hover, page, newly loaded icons)."""
        if not self.is_open:
            return
        try:
            visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            visible = False
        if not visible:  # closed some other way, e.g. Alt+F4
            self.is_open = False
            self.hwnd = None
            return
        if (self.catalog.version, self.catalog.ready.is_set()) != self._drawn_state:
            self._render()  # the app list arrived, or more icons did
        if self._dirty:
            cv2.imshow(WINDOW, self._compose())
            self._dirty = False

    @property
    def focused(self) -> bool:
        return self.is_open and self.hwnd is not None and self.hwnd == win32.user32.GetForegroundWindow()

    # -- navigation ----------------------------------------------------------

    def next_page(self) -> None:
        self._go(self.page + 1)

    def previous_page(self) -> None:
        self._go(self.page - 1)

    def show_tab(self, tab: str) -> None:
        if tab != self.tab:
            self.tab, self.page = tab, 0
            self._render()

    def handle_key(self, key: int) -> None:
        if key == 27:  # Esc
            self.close()
        elif key in (ord("a"), ord("d")):  # also usable from the keyboard
            self._go(self.page + (1 if key == ord("d") else -1))

    def _go(self, page: int) -> None:
        shown = self.catalog.favorites if self.tab == "favorites" else self.catalog.apps
        page = max(0, min(page, page_count(shown, self.tab) - 1))
        if page != self.page:
            self.page = page
            self._render()

    def _activate(self, target: Target) -> None:
        if target.kind == "app":
            self.close()
            self.on_launch(target.app)
        elif target.kind == "tab":
            self.show_tab(target.value)
        elif target.kind == "page":
            self._go(self.page + target.value)
        elif target.kind == "letter":
            self._go(target.value)
        elif target.kind == "close":
            self.close()

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        target = next((t for t in self._targets if t.enabled and t.contains(x, y)), None)
        if event == cv2.EVENT_MOUSEMOVE:
            if target != self._hover:
                self._hover = target
                self._dirty = True
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._pressed = target
            self._dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            pressed, self._pressed = self._pressed, None
            self._dirty = True
            if target is not None and target == pressed:  # like a real button: press and release on it
                self._activate(target)

    # -- drawing -------------------------------------------------------------

    def _render(self) -> None:
        """Draw the current page with nothing hovered into self._base."""
        width, height = self.size
        s = min(width / REF_W, height / REF_H)
        self._drawn_state = (self.catalog.version, self.catalog.ready.is_set())
        image = Image.new("RGB", self.size, BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.text((round(60 * s), round(50 * s)), "Open an app", font=_font(round(40 * s), bold=True), fill=TEXT)

        if not self.catalog.ready.is_set() or self.catalog.error:
            self._targets = [t for t in layout(width, height, self.tab, 0, [], []) if t.kind == "close"]
            message = self.catalog.error or "Finding your apps..."
            _centered(draw, (width // 2, height // 2), message, _font(round(30 * s)), MUTED)
        else:
            favorites, apps = self.catalog.favorites, self.catalog.apps
            shown = favorites if self.tab == "favorites" else apps
            self.page = max(0, min(self.page, page_count(shown, self.tab) - 1))
            self._targets = layout(width, height, self.tab, self.page, favorites, apps)
            if not shown:
                _centered(draw, (width // 2, height // 2), "No favourites found - add some in config.toml",
                          _font(round(30 * s)), MUTED)
            pages = page_count(shown, self.tab)
            if pages > 1:
                label = f"Page {self.page + 1} of {pages}"
                font = _font(round(24 * s))
                draw.text((width - round(140 * s) - draw.textlength(label, font=font), round(62 * s)), label,
                          font=font, fill=MUTED)
        for target in self._targets:
            self._draw_target(draw, image, target, s, hover=False, pressed=False)
        hint = ("Point at an app and press your thumb in to open it   \u00b7   Swipe left or right for more   "
                "\u00b7   Swipe down or show a victory sign to close")
        _centered(draw, (width // 2, height - round(38 * s)), hint, _font(round(22 * s)), MUTED)
        self._base = image
        self._hover = next((t for t in self._targets if t == self._hover), None)
        self._dirty = True

    def _compose(self) -> np.ndarray:
        """The rendered page with the hovered and pressed targets drawn on top."""
        image = self._base.copy()
        draw = ImageDraw.Draw(image)
        s = min(self.size[0] / REF_W, self.size[1] / REF_H)
        highlighted: list[Target] = []
        for target in (self._hover, self._pressed):
            if target is not None and target not in highlighted:
                highlighted.append(target)
        for target in highlighted:
            self._draw_target(draw, image, target, s, hover=target == self._hover, pressed=target == self._pressed)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def _draw_target(self, draw: ImageDraw.ImageDraw, image: Image.Image, t: Target, s: float,
                     hover: bool, pressed: bool) -> None:
        x0, y0, x1, y1 = t.rect
        if t.kind == "app":
            fill = TILE_PRESSED if pressed else TILE_HOVER if hover else TILE
            draw.rounded_rectangle(t.rect, radius=round(20 * s), fill=fill,
                                   outline=ACCENT if hover else None, width=max(2, round(3 * s)))
            big = self.tab == "favorites"
            icon_size = round((96 if big else 72) * s)
            name_font = _font(round((23 if big else 19) * s))
            lines = _wrap(draw, t.label, name_font, x1 - x0 - round(20 * s), max_lines=2)
            line_h = round(name_font.size * 1.25)
            block = icon_size + round(14 * s) + line_h * len(lines)
            top = y0 + (y1 - y0 - block) // 2
            cx = (x0 + x1) // 2
            _paste_icon(image, draw, t.app, (cx - icon_size // 2, top), icon_size)
            for i, line in enumerate(lines):
                _centered(draw, (cx, top + icon_size + round(14 * s) + line_h * i + line_h // 2), line,
                          name_font, TEXT)
        elif t.kind in ("tab", "letter"):
            selected_fill = ACCENT if t.kind == "tab" else (58, 66, 82)
            fill = TILE_PRESSED if pressed else selected_fill if t.selected else TILE_HOVER if hover else TILE
            draw.rounded_rectangle(t.rect, radius=(y1 - y0) // 2 if t.kind == "tab" else round(12 * s),
                                   fill=fill, outline=ACCENT if hover else None, width=max(2, round(2 * s)))
            font = _font(round((24 if t.kind == "tab" else 22) * s), bold=t.kind == "tab")
            _centered(draw, ((x0 + x1) // 2, (y0 + y1) // 2), t.label, font,
                      TEXT if t.selected or hover else MUTED)
        elif t.kind in ("page", "close"):
            if not t.enabled:
                colour, fill = DISABLED, None
            else:
                colour, fill = TEXT, TILE_PRESSED if pressed else TILE_HOVER if hover else TILE
            draw.rounded_rectangle(t.rect, radius=round(22 * s), fill=fill,
                                   outline=ACCENT if hover and t.enabled else None, width=max(2, round(3 * s)))
            size = round((96 if t.kind == "page" else 56) * s)
            _centered(draw, ((x0 + x1) // 2, (y0 + y1) // 2), t.label, _font(size), colour)


# -- drawing helpers ----------------------------------------------------------

@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in (("seguisb.ttf", "segoeuib.ttf") if bold else ("segoeui.ttf",)):
        try:
            return ImageFont.truetype(str(FONTS / name), size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _centered(draw: ImageDraw.ImageDraw, centre: tuple[int, int], text: str, font, fill) -> None:
    draw.text(centre, text, font=font, fill=fill, anchor="mm")


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int) -> list[str]:
    """Word-wrap into at most max_lines lines, ending with "..." if it doesn't fit."""
    lines: list[str] = []
    words = text.split()
    while words and len(lines) < max_lines:
        line = words.pop(0)
        while words and draw.textlength(f"{line} {words[0]}", font=font) <= max_width:
            line = f"{line} {words.pop(0)}"
        lines.append(line)
    if words or any(draw.textlength(line, font=font) > max_width for line in lines):
        last = lines[-1] if lines else text
        while last and draw.textlength(last + "...", font=font) > max_width:
            last = last[:-1]
        lines[-1:] = [last.rstrip() + "..."]
    return lines


def _paste_icon(image: Image.Image, draw: ImageDraw.ImageDraw, app: App, corner: tuple[int, int],
                size: int) -> None:
    if app.icon is not None:
        icon = Image.fromarray(app.icon, "RGBA")
        if icon.size != (size, size):
            icon = icon.resize((size, size), Image.LANCZOS)
        image.paste(icon, corner, icon)
        return
    # No icon (yet): a coloured circle with the app's first letter.
    hue = zlib.crc32(app.name.encode()) % 360
    colour = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue // 2, 150, 200]]]), cv2.COLOR_HSV2RGB)[0, 0])
    x, y = corner
    draw.ellipse((x, y, x + size, y + size), fill=colour)
    _centered(draw, (x + size // 2, y + size // 2), letter_of(app), _font(size // 2, bold=True), TEXT)
