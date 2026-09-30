# Gesture Control

Control Windows with your hands. Your webcam watches your hand, Google's
MediaPipe hand-tracking model (running locally on your PC) finds 21 points on
it 30 times a second, and Gesture Control turns those movements into actions:
switching virtual desktops, flicking through apps with Alt+Tab, closing
windows, steering the mouse with your finger, clicking, and opening apps from
a big, easy-to-click app picker.

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

To try gestures without anything actually happening, run `run.bat --dry-run`.
The preview then shows what *would* have happened.

## Gestures

Left and right are as you see yourself in the preview, which works like a
mirror.

| Gesture | How | Does |
|---|---|---|
| **Swipe left / right** | Open hand, quick sweep sideways | Next / previous desktop |
| **Swipe up** | Open hand, quick sweep upward | Task View |
| **Pinch** | From an open hand, touch thumb and index fingertips and hold for a moment | Close the active window |
| **Grab** | From an open hand, make a fist | Opens Alt+Tab; move the fist to choose, open your hand to switch |
| **Point** | Index finger out, other fingers curled | The cursor follows your fingertip |
| **Thumb press** | While pointing, press your thumb in against your hand | Click (keep it pressed to drag) |
| **Victory sign** ✌ | From an open hand, hold up two fingers for a moment | Opens the app picker |

Everything reacts quickly:

- **Swipes** fire partway through the movement, and quick flicks need less
  travel.
- **Holds** are short: pinch ¼ second, grab about 0.15 seconds, victory sign
  0.3 seconds.
- **Clicks** register as your thumb goes in.

Tips:

- **Start held gestures from an open hand.** Pinch, grab and the victory sign
  only count if you showed an open hand just before. This stops pens, mugs and
  resting your chin on your fist from triggering things. If you forget, the
  preview shows a grey ring saying *start from an open hand*.
- **Swipes need a brief pause first.** A hand has to be still for a moment
  before it can swipe, so raising or reaching isn't mistaken for a swipe.
  Upward swipes need a slightly longer pause. Bringing your hand back after a
  swipe is ignored, and the opposite direction is blocked for one second.
- **The ring shows what's coming.** A held gesture draws a filling ring
  labelled with its action. Let go before it fills to cancel.
- **Alt+Tab cancels if your hand leaves the view** for almost a second.
- Good light and your hand 40–80 cm from the camera work best.

**Swiping between desktops needs at least two virtual desktops.** Press
`Win+Ctrl+D` to add one. The preview shows `desktop 2/3` so you know where you
are.

## The finger mouse

Point with your index finger (other fingers curled). While you point, the
preview outlines the **steering area**: that part of the camera view is mapped
onto your whole screen, so your fingertip's position in it is where the
cursor goes.

- **Click:** press your thumb in against the side of your hand, then let it
  spring back out. The click lands where the cursor was *before* your thumb
  moved, and the cursor holds still during the click, so the small jolt of
  pressing doesn't make you miss. Two quick presses make a double-click.
- **Drag:** press your thumb in and keep it in while you move, then let go.
- **Stop steering:** open your hand or curl your finger. The cursor stays where
  it is.
- Pointing with your thumb already tucked in doesn't click. A click needs a
  real press, so stick your thumb out while aiming.

The cursor is smoothed so it doesn't shake. If it feels jumpy or sluggish,
change `smoothing` under `[mouse]` in `config.toml`. If reaching the screen
edges takes too much arm movement, make `area_width` smaller.

## The app picker

Hold up a **victory sign** ✌ and a full-screen app picker appears with big
tiles:

- **Favorites** shows the apps pinned to your taskbar, plus common ones like
  Settings and Calculator. List your own in `config.toml` under
  `[app_picker] favorites`.
- **All apps** has everything in your Start menu, a few pages of big tiles.
  The letter strip at the bottom jumps straight to a letter.

Point at an app and do a thumb press to open it. Turn pages by swiping left or
right, or with the big arrows on either side. Close the picker with the ×
button, a swipe up or down, another victory sign, or `Esc`. While the picker is
open, the desktop, Alt+Tab and close-window gestures are paused so they can't
fire by accident.

## Customizing

Everything is in **`config.toml`**, with comments. Restart after editing.

Each gesture can do any of these:

```
next_desktop  previous_desktop  new_desktop  close_desktop  task_view  show_desktop
close_window  minimize_window  maximize_window  app_picker  none
```

It can also press any key combination, for example `"hotkey:ctrl+shift+esc"`,
`"hotkey:volumeup"` or `"hotkey:playpause"`. `thumb_up_hold`,
`thumb_down_hold` and `love_you_hold` are recognised but off by default,
because people make them naturally on video calls. For example, to pause music
with a thumbs up:

```toml
[actions]
thumb_up_hold = "hotkey:playpause"
```

If gestures trigger too easily (or not easily enough), adjust `[tuning]`.
Larger `swipe_distance` means longer swipes; larger `pinch_hold_time` means a
longer pinch before a window closes. `hand = "right"` under `[tracking]` makes
only your right hand count. To switch the finger mouse off, set
`enabled = false` under `[mouse]`.

### Command-line options

```
run.bat --dry-run        show gestures without acting on them
run.bat --details        start with the tuning numbers visible (or press D)
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
- **Clicks don't register**: press D in the preview and watch `thumb gap`
  while you point. It should drop clearly (by 0.2 or more) when you press your
  thumb in. Start with your thumb sticking out, then press.
- **A window won't close, or ignores clicks**: programs running as
  administrator can't be controlled by a normal program. Windows blocks this on
  purpose. Run Gesture Control as administrator if you need it.
- **An app is missing from the picker**: the picker lists what's in your Start
  menu. Apps can also be added to `favorites` by name.
- **Accidental triggers**: pause with `Ctrl+Alt+G` during video calls. Increase
  the relevant `[tuning]` values, or set rarely used gestures to `"none"`.
- **Alt or the mouse button seems stuck** (only possible if Gesture Control was
  killed mid-gesture): press and release `Alt`, or click once.

## Privacy

Everything runs on your computer. Camera frames are never saved or sent
anywhere. The only network access is a one-time download of the hand-tracking
model (about 8 MB, from Google's MediaPipe model storage) into `models/`.

## How it works

```
camera.py    webcam on a background thread; frames stamped with the camera's capture time
tracker.py   MediaPipe gesture recognizer: 21 landmarks per hand + a gesture label
features.py  finger curl, pinch and thumb distances, palm position -> hand shape
engine.py    shapes and movement over time -> swipes, held gestures, the Alt+Tab grab
pointer.py   the finger mouse: fingertip -> cursor position, thumb press -> clicks
actions.py   gesture events -> keyboard shortcuts, mouse input and window commands
apps.py      the Start menu's apps and their icons, for the picker
picker.py    the app picker window
win32.py     Windows APIs: SendInput, window messages, global hotkey, desktop info
overlay.py   the preview window
app.py       the main loop and command line
```

The engine and the finger mouse are pure logic, so they are tested with
simulated hand movements. Hand-shape rules are tested against landmarks
MediaPipe produced for Google's sample gesture photos. To run the tests:

```
.venv\Scripts\python.exe -m unittest discover -s tests -t .
```
