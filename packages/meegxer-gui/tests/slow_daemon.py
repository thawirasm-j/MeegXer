"""The daemon, slow to answer the first window read of each probe once an
offline run is done: longer than the GUI waits (3 s), as a busy CI runner can
be. Started by the ``daemon`` fixture with ``"slow"``; the arguments are the
daemon's."""

from __future__ import annotations

import sys
import time
from typing import Any

from meegxer.core import control
from meegxer.core.daemon import main

DELAY_S = 3.5  # the GUI's control client times out after 3 s
_slowed: set[tuple[int, str]] = set()  # (the offline run's id, probe) slowed
_window = control.Daemon.offline_window


def _slow_window(self: control.Daemon, probe: str, *args: Any, **kwargs: Any) -> Any:
    run = self._offline
    if run is not None and run.state == "done" and (id(run), probe) not in _slowed:
        _slowed.add((id(run), probe))
        time.sleep(DELAY_S)
    return _window(self, probe, *args, **kwargs)


control.Daemon.offline_window = _slow_window  # type: ignore[method-assign]

if __name__ == "__main__":
    sys.exit(main())
