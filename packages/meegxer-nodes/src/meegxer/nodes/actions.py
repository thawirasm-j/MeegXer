"""Action sinks (rule 9: plugins emit events; only trusted host sinks act).

The keyboard sink presses a key for each event whose label it maps
(``pred_left=left``): a classifier's decisions become key presses, e.g. to
drive a game or a speller. Safety comes first:

- it is off until armed by hand, for one live run (the GUI's switch, or
  ``meegxer arm NODE``); a pipeline or a session never arms it, and a batch
  run never presses anything;
- it presses only the keys of its fixed map, at most ``max_rate`` keys per
  second (the rest are dropped and counted);
- Esc, the GUI's switch and *Stop* disarm it.

Key presses come from pynput, imported only when it is armed (it needs a
desktop session: on a headless machine arming says so). MeegXer is not a
medical device.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from meegxer.core.block import Event
from meegxer.core.manifest import Manifest
from meegxer.core.node import ActionSink
from meegxer.nodes.references import credits

KEY_NAMES = ("left", "right", "up", "down", "space", "enter", "tab", "backspace",
             "home", "end", "page_up", "page_down",
             *(f"f{i}" for i in range(1, 13)))  # fmt: skip


def parse_keys(text: str) -> dict[str, str]:
    """``label=key`` pairs, comma-separated: an event label and the key it
    presses (one letter or digit, or a key name: ``KEY_NAMES``)."""
    keys: dict[str, str] = {}
    for part in str(text).split(","):
        if not part.strip():
            continue
        label, sep, key = (p.strip() for p in part.partition("="))
        if not sep or not label or not key:
            raise ValueError(f"{part.strip()!r}: write label=key, e.g. pred_left=left")
        if not (len(key) == 1 and key.isalnum()) and key.lower() not in KEY_NAMES:
            names = ", ".join(KEY_NAMES[:8])
            raise ValueError(
                f"{key!r} is not a key it presses: one letter or digit, or {names}, …"
            )
        keys[label] = key if len(key) == 1 else key.lower()
    if not keys:
        raise ValueError("map at least one event label to a key, e.g. pred_left=left")
    return keys


# a keyboard: (press and release one key, stop listening for Esc)
type Keyboard = tuple[Callable[[str], None], Callable[[], None]]


def pynput_keyboard(on_escape: Callable[[], None]) -> Keyboard:
    """The desktop's keyboard (pynput), and a listener that calls
    ``on_escape`` when Esc is pressed."""
    try:
        from pynput import keyboard
    except Exception as exc:  # no desktop session (e.g. a headless Pi)
        reason = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
        raise RuntimeError(f"no keyboard to press here ({reason})") from None
    controller = keyboard.Controller()

    def press(key: str) -> None:
        target = key if len(key) == 1 else getattr(keyboard.Key, key)
        controller.press(target)
        controller.release(target)

    def pressed(key: object) -> None:
        if key == keyboard.Key.esc:
            on_escape()

    listener = keyboard.Listener(on_press=pressed)
    listener.daemon = True
    listener.start()
    return press, listener.stop


class KeyboardSink(ActionSink):
    """Presses a key for each mapped event, only while armed by hand."""

    manifest: ClassVar[Manifest] = Manifest.from_dict({
        "spec_version": "0.2.23",
        "name": "meegxer.nodes.keyboard",
        "version": "0.1.0",
        "type": "stream",
        "role": "sink",
        "description": "Presses a key for each event it maps (e.g. LDA's "
                       "pred_left), only while armed by hand.",
        "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [],
        "params": [
            {"name": "keys", "type": "string",
             "default": "pred_left=left, pred_right=right", "primary": True,
             "description": "Event label = key, comma-separated (a letter or "
                            "digit, or left, right, up, down, space, enter, …)"},
            {"name": "max_rate", "type": "float", "default": 2.0, "min": 0.1,
             "max": 20.0, "unit": "/s", "primary": True,
             "description": "Keys per second at most; the rest are dropped "
                            "(MeegXer's setting)"},
        ],
        "availability": "both",
        **credits("meegxer.nodes.keyboard"),
    })  # fmt: skip

    # how it reaches a keyboard (tests put a fake one here)
    keyboard: ClassVar[Callable[[Callable[[], None]], Keyboard]] = staticmethod(
        pynput_keyboard
    )

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        self.keys = parse_keys(str(self.params["keys"]))
        self.pressed = 0
        self.dropped = 0
        self._last = -float("inf")  # when it last pressed a key
        self._press: Callable[[str], None] | None = None
        self._stop: Callable[[], None] | None = None
        self._lock = threading.Lock()  # Esc disarms from the listener's thread

    def arm(self, on: bool) -> None:
        with self._lock:
            if on and not self.armed:
                self._press, self._stop = type(self).keyboard(self._escape)
                self.armed = True
            elif not on and self.armed:
                self._disarm()

    def _escape(self) -> None:
        with self._lock:
            if self.armed:
                self._disarm()
                self.notes.append("disarmed by Esc")

    def _disarm(self) -> None:
        self.armed = False
        stop, self._stop, self._press = self._stop, None, None
        if stop is not None:
            stop()

    def act(self, events: Sequence[Event]) -> None:
        gap = 1.0 / float(self.params["max_rate"])  # seconds between presses
        for event in events:
            key = self.keys.get(event.label)
            with self._lock:
                if key is None or not self.armed or self._press is None:
                    continue
                now = time.monotonic()
                if now - self._last < gap:  # faster than max_rate: dropped
                    self.dropped += 1
                    continue
                self._last = now
                self._press(key)
                self.pressed += 1

    def action_status(self) -> dict[str, Any]:
        return {"armed": self.armed, "pressed": self.pressed, "dropped": self.dropped}
