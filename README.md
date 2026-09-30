# Gesture Control

Control Windows with your hands. Your webcam watches your hand, Google's
MediaPipe hand-tracking model (running locally on your PC) finds 21 points on
it 30 times a second, and Gesture Control turns movements into actions:
switching virtual desktops, flicking through apps with Alt+Tab, and closing
windows.

## Start it

Double-click **`run.bat`**.

The first time on a new computer it installs what it needs, which takes a
minute or two. A small preview window then appears in the bottom-right corner
showing what the camera sees, with your hand's skeleton drawn on top. It stays
on top of other windows and follows you across virtual desktops.

- **Pause / resume:** `Ctrl+Alt+G` from any window. This also switches the
  camera (and its light) off.
- **Quit:** press `Q` in the preview window, close it, or press `Ctrl+C` in the
  console.

To try the gestures without anything actually happening, run
`run.bat --dry-run`. The preview then shows what *would* have happened.

## Gestures

Left and right are as you see yourself in the preview, which works like a
mirror.

| Gesture | How | Does |
|---|---|---|
| **Swipe left** | Open hand, quick sweep to your left | Next desktop (`Ctrl+Win+Right`) |
| **Swipe right** | Open hand, quick sweep to your right | Previous desktop (`Ctrl+Win+Left`) |
| **Swipe up** | Open hand, quick sweep upward | Task View (`Win+Tab`) |
| **Pinch and hold** | From an open hand, touch thumb and index fingertips together and hold still until the ring fills (about 0.8 s) | Close the active window |
| **Grab** | From an open hand, make a fist and hold it for a moment | Opens the Alt+Tab switcher |
| ...then move the fist | Left / right | Moves the Alt+Tab selection |
| ...then open your hand | | Switches to the selected app |

Tips:

- **Start from an open hand.** Pinch and grab only count when you've just shown
  an open hand, palm toward the camera. This stops pens, mugs and resting your
  chin on your fist from closing windows or opening the switcher. If you forget,
  the preview shows a grey ring saying *start from an open hand*.
- **Swipes need a moment of stillness first.** Hold your hand up for about half
  a second, then swipe. Bringing your hand back afterwards is ignored, and the
  opposite direction is blocked for one second.
- **The ring shows what's coming.** A held gesture draws a filling ring labelled
  with its action, such as *Close window*. Let go before it fills to cancel.
- **Alt+Tab cancels if your hand leaves the view** for almost a second, and you
  stay on the current app.
- Good light and your hand 40–80 cm from the camera work best.

**You need at least two virtual desktops for swiping to do anything.** Press
`Win+Ctrl+D` to add one (or open Task View and click *New desktop*). The
preview shows `desktop 2/3` so you know where you are.

## The preview window

- A green dot means active. The centre shows the recognised hand shape; the
  right side shows which desktop you are on.
- With no hand in view, a legend lists every gesture and its action.
- Press `D` for live numbers (finger curl, pinch distance, fps), useful for
  tuning.
- Press `P` to pause and `Q` to quit.

## Customizing

Everything is in **`config.toml`**, with comments. Restart after editing.

Each gesture can do any of these:

```
next_desktop  previous_desktop  new_desktop  close_desktop  task_view  show_desktop
close_window  minimize_window  maximize_window  none
```

It can also press any key combination, for example `"hotkey:ctrl+shift+esc"`,
`"hotkey:volumeup"` or `"hotkey:playpause"`. Extra gestures are recognised but
off by default, because people make them naturally on video calls:
`thumb_up_hold`, `thumb_down_hold`, `victory_hold`, `pointing_up_hold` and
`love_you_hold`. For example, to pause music with a thumbs up:

```toml
[actions]
thumb_up_hold = "hotkey:playpause"
```

If gestures trigger too easily (or not easily enough), adjust `[tuning]`.
For example, raise `swipe_distance` to require longer swipes, or raise
`pinch_hold_time` to require a longer pinch. `hand = "right"` makes only your
right hand count.

### Command-line options

```
run.bat --dry-run        show gestures without acting on them
run.bat --details        start with the tuning numbers visible
run.bat --no-preview     no preview window (pause with Ctrl+Alt+G, quit with Ctrl+C)
run.bat --camera 1       use a different camera
run.bat --list-cameras   list the cameras that can be opened
run.bat --config other.toml
```

### Start without a console window, or with Windows

Make a shortcut with this target, "Start in" set to this folder:

```
<this folder>\.venv\Scripts\pythonw.exe -m gesture_control
```

To start it with Windows, put the shortcut in the folder that opens when you
press `Win+R` and enter `shell:startup`. Don't combine `pythonw` with
`--no-preview`: you would have no way to quit except Task Manager.

## Troubleshooting

- **"NO CAMERA"**: another app (Teams, Zoom, the Camera app) may be using the
  webcam. Gesture Control keeps retrying. Also check *Settings > Privacy &
  security > Camera*. Pausing releases the camera for other apps.
- **The wrong camera** (such as a virtual OBS camera): run
  `run.bat --list-cameras`, then set `index` under `[camera]`.
- **Swipes do nothing**: you probably have only one virtual desktop; the
  preview tells you so.
- **A window won't close**: windows of programs running as administrator can't
  be controlled by a normal program. Windows blocks this on purpose. Run
  Gesture Control as administrator if you need it.
- **Accidental triggers**: pause with `Ctrl+Alt+G` during video calls. Increase
  the relevant `[tuning]` values, or set rarely used gestures to `"none"`.
- **Alt seems stuck** (only possible if Gesture Control was killed while the
  switcher was open): press and release `Alt`.

## Privacy

Everything runs on your computer. Camera frames are never saved or sent
anywhere. The only network access is a one-time download of the hand-tracking
model (about 8 MB, from Google's MediaPipe model storage) into `models/`.

## How it works

```
camera.py    webcam on a background thread; frames stamped with the camera's capture time
tracker.py   MediaPipe gesture recognizer: 21 landmarks per hand + a gesture label
features.py  finger curl, pinch distance, palm position -> hand shape (open, fist, pinch...)
engine.py    shapes and movement over time -> swipes, held gestures, the Alt+Tab grab
actions.py   gesture events -> keyboard shortcuts and window commands
win32.py     Windows APIs: SendInput, window messages, global hotkey, desktop info
overlay.py   the preview window
app.py       the main loop and command line
```

The engine is pure logic, so it is tested with simulated hand movements.
Hand-shape rules are tested against landmarks MediaPipe produced for Google's
sample gesture photos. To run the tests:

```
.venv\Scripts\python.exe -m unittest discover -s tests -t .
```
