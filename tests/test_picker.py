import string
import unittest

import cv2

from gesture_control.apps import App, AppCatalog
from gesture_control.picker import GRIDS, AppPicker, layout, page_count

W, H = 1920, 1080
APPS = [App(f"{letter} app {i}", f"id-{letter}-{i}") for letter in string.ascii_uppercase for i in range(5)]
FAVORITES = APPS[:15]


def of_kind(targets, kind):
    return [t for t in targets if t.kind == kind]


class LayoutTests(unittest.TestCase):
    def test_favorites_page(self):
        targets = layout(W, H, "favorites", 0, FAVORITES, APPS)
        self.assertEqual(len(of_kind(targets, "app")), 15)
        self.assertEqual([t.value for t in of_kind(targets, "tab")], ["favorites", "all"])
        self.assertEqual(len(of_kind(targets, "close")), 1)
        self.assertEqual(of_kind(targets, "page"), [])  # all on one page: no arrows
        self.assertEqual(of_kind(targets, "letter"), [])

    def test_all_apps_are_split_into_pages(self):
        cols, rows = GRIDS["all"]
        self.assertEqual(page_count(APPS, "all"), -(-len(APPS) // (cols * rows)))
        first = layout(W, H, "all", 0, FAVORITES, APPS)
        self.assertEqual(len(of_kind(first, "app")), cols * rows)
        prev, nxt = of_kind(first, "page")
        self.assertFalse(prev.enabled)
        self.assertTrue(nxt.enabled)
        last = layout(W, H, "all", 99, FAVORITES, APPS)  # past the end: clamped to the last page
        self.assertFalse(of_kind(last, "page")[1].enabled)
        self.assertEqual(len(of_kind(last, "app")), len(APPS) - (page_count(APPS, "all") - 1) * cols * rows)

    def test_tiles_fit_on_screen_without_overlapping(self):
        for tab in ("favorites", "all"):
            targets = layout(W, H, tab, 0, FAVORITES, APPS)
            for t in targets:
                x0, y0, x1, y1 = t.rect
                self.assertTrue(0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H, t)
            rects = [t.rect for t in targets]
            for i, a in enumerate(rects):
                for b in rects[i + 1:]:
                    overlap = a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
                    self.assertFalse(overlap, (tab, a, b))

    def test_tiles_are_big_enough_to_hit(self):
        for tab in ("favorites", "all"):
            tile = of_kind(layout(W, H, tab, 0, FAVORITES, APPS), "app")[0]
            x0, y0, x1, y1 = tile.rect
            self.assertGreaterEqual(min(x1 - x0, y1 - y0), 170)

    def test_letters_jump_to_the_page_with_that_letter(self):
        targets = layout(W, H, "all", 0, FAVORITES, APPS)
        letters = {t.label: t.value for t in of_kind(targets, "letter")}
        per_page = GRIDS["all"][0] * GRIDS["all"][1]
        self.assertEqual(letters["A"], 0)
        self.assertEqual(letters["M"], APPS.index(next(a for a in APPS if a.name[0] == "M")) // per_page)
        on_page = {t.label for t in of_kind(targets, "letter") if t.selected}
        self.assertEqual(on_page, {a.name[0] for a in APPS[:per_page]})

    def test_layout_scales_with_the_screen(self):
        small = of_kind(layout(W, H, "all", 0, FAVORITES, APPS), "app")[0].rect
        big = of_kind(layout(2 * W, 2 * H, "all", 0, FAVORITES, APPS), "app")[0].rect
        self.assertAlmostEqual((big[2] - big[0]) / (small[2] - small[0]), 2.0, delta=0.02)


class PickerTests(unittest.TestCase):
    """Navigation and clicking, without opening the window."""

    def setUp(self):
        catalog = AppCatalog([])
        catalog.apps, catalog.favorites = APPS, FAVORITES
        catalog.ready.set()
        self.launched = []
        self.picker = AppPicker(catalog, self.launched.append)
        self.picker._render()

    def click(self, target, release_on=None):
        (x0, y0, x1, y1), other = target.rect, release_on or target
        self.picker._on_mouse(cv2.EVENT_LBUTTONDOWN, (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
        ox0, oy0, ox1, oy1 = other.rect
        self.picker._on_mouse(cv2.EVENT_LBUTTONUP, (ox0 + ox1) // 2, (oy0 + oy1) // 2, 0, None)

    def targets(self, kind):
        return of_kind(self.picker._targets, kind)

    def test_clicking_an_app_launches_it(self):
        tile = self.targets("app")[2]
        self.click(tile)
        self.assertEqual(self.launched, [tile.app])

    def test_releasing_somewhere_else_does_not_launch(self):
        first, second = self.targets("app")[:2]
        self.click(first, release_on=second)
        self.assertEqual(self.launched, [])

    def test_swipes_turn_pages_and_stop_at_the_ends(self):
        self.picker.show_tab("all")
        self.picker.previous_page()
        self.assertEqual(self.picker.page, 0)
        for _ in range(20):
            self.picker.next_page()
        self.assertEqual(self.picker.page, page_count(APPS, "all") - 1)

    def test_tabs_page_arrows_and_letters(self):
        self.click(next(t for t in self.targets("tab") if t.value == "all"))
        self.assertEqual(self.picker.tab, "all")
        self.click(next(t for t in self.targets("page") if t.value == +1))
        self.assertEqual(self.picker.page, 1)
        z = next(t for t in self.targets("letter") if t.label == "Z")
        self.click(z)
        self.assertEqual(self.picker.page, z.value)

    def test_hover_follows_the_mouse(self):
        tile = self.targets("app")[5]
        x0, y0, x1, y1 = tile.rect
        self.picker._on_mouse(cv2.EVENT_MOUSEMOVE, (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
        self.assertEqual(self.picker._hover, tile)
        self.picker._on_mouse(cv2.EVENT_MOUSEMOVE, 2, 2, 0, None)
        self.assertIsNone(self.picker._hover)


if __name__ == "__main__":
    unittest.main()
