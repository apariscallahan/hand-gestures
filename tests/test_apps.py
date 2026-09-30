import unittest

from gesture_control.apps import MAX_FAVORITES, App, is_launchable, pick_favorites


class LaunchableTests(unittest.TestCase):
    def test_real_apps_are_kept(self):
        self.assertTrue(is_launchable("Calculator", "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"))
        self.assertTrue(is_launchable("Brave", "Brave"))
        self.assertTrue(is_launchable("Photoshop", r"{6D809377-6AF0-444B-8957-A3773F02200E}\Adobe\Photoshop.exe"))

    def test_links_documents_and_uninstallers_are_dropped(self):
        self.assertFalse(is_launchable("Buy Pro Version!", r"{6D809377}\CS Browser\Buy Pro Version.url"))
        self.assertFalse(is_launchable("Project website", "https://example.com"))
        self.assertFalse(is_launchable("Read me", r"C:\Program Files\App\readme.txt"))
        self.assertFalse(is_launchable("Uninstall Foo", r"C:\Program Files\Foo\unins000.exe"))


class FavoriteTests(unittest.TestCase):
    APPS = [App("Google Chrome", "Chrome"), App("Chrome Remote Desktop", "crd"), App("Word", "word"),
            App("WordPad", "wordpad"), App("Spotify", "spotify")]

    def names(self, wanted):
        return [a.name for a in pick_favorites(self.APPS, wanted)]

    def test_keeps_the_requested_order(self):
        self.assertEqual(self.names(["Spotify", "Word"]), ["Spotify", "Word"])

    def test_exact_names_win(self):
        self.assertEqual(self.names(["word"]), ["Word"])

    def test_partial_names_pick_the_closest_app(self):
        self.assertEqual(self.names(["chrome"]), ["Google Chrome"])

    def test_unknown_and_repeated_names_are_skipped(self):
        self.assertEqual(self.names(["Photoshop", "Spotify", "spotify"]), ["Spotify"])

    def test_number_of_favourites_is_capped(self):
        apps = [App(f"App {i:02}", str(i)) for i in range(40)]
        self.assertEqual(len(pick_favorites(apps, [a.name for a in apps])), MAX_FAVORITES)


if __name__ == "__main__":
    unittest.main()
