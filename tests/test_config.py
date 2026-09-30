import tempfile
import unittest
from pathlib import Path

from gesture_control.app import GestureApp
from gesture_control.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config


def load_text(text: str) -> Config:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.toml"
        path.write_text(text, encoding="utf-8")
        return load_config(path)


class ConfigTests(unittest.TestCase):
    def test_shipped_config_matches_the_built_in_defaults(self):
        self.assertEqual(load_config(DEFAULT_CONFIG_PATH), Config())

    def test_values_override_defaults(self):
        cfg = load_text('[actions]\nswipe_up = "show_desktop"\n[tuning]\nswipe_distance = 3\n')
        self.assertEqual(cfg.actions.swipe_up, "show_desktop")
        self.assertEqual(cfg.tuning.swipe_distance, 3.0)
        self.assertEqual(cfg.actions.swipe_left, "next_desktop")

    def test_typo_in_a_setting_name_is_reported(self):
        with self.assertRaisesRegex(ConfigError, "swipe_lef"):
            load_text('[actions]\nswipe_lef = "none"\n')

    def test_wrong_type_is_reported(self):
        with self.assertRaisesRegex(ConfigError, "true/false"):
            load_text("[camera]\nmirror = 1\n")

    def test_invalid_choice_is_reported(self):
        with self.assertRaisesRegex(ConfigError, "any, left, right"):
            load_text('[tracking]\nhand = "both"\n')

    def test_missing_explicit_file_is_reported(self):
        with self.assertRaises(ConfigError):
            load_config(Path(tempfile.gettempdir()) / "no-such-gesture-config.toml")

    def test_unknown_action_is_reported(self):
        with self.assertRaisesRegex(ConfigError, "launch_rockets"):
            GestureApp(load_text('[actions]\npinch_hold = "launch_rockets"\n'))

    def test_bad_hotkey_is_reported(self):
        with self.assertRaisesRegex(ConfigError, "banana"):
            GestureApp(load_text('[actions]\nswipe_up = "hotkey:ctrl+banana"\n'))
        with self.assertRaises(ConfigError):
            GestureApp(load_text('[general]\npause_hotkey = "ctrl+alt"\n'))


if __name__ == "__main__":
    unittest.main()
