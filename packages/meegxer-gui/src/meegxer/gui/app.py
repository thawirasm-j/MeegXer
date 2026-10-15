"""``meegxer-gui``: connect to the daemon (starting one if none answers) and open
the mixer window."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from meegxer.core.control import DEFAULT_CONTROL, DEFAULT_DISPLAY
from meegxer.gui.connection import DaemonLink, daemon_running, spawn_daemon
from meegxer.gui.theme import load_theme


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meegxer-gui", description=__doc__)
    parser.add_argument(
        "--connect", default=DEFAULT_CONTROL, help="daemon control endpoint"
    )
    parser.add_argument(
        "--display",
        default=DEFAULT_DISPLAY,
        help="display endpoint for a daemon started by the GUI",
    )
    parser.add_argument(
        "--theme", default="dark", help="dark, light, or a theme JSON file"
    )
    parser.add_argument("--pipeline", type=Path, help="pipeline JSON to open")
    parser.add_argument(
        "--start", action="store_true", help="start (or run offline) right away"
    )
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("MeegXer")
    owns = False
    if not daemon_running(args.connect):
        print(f"meegxer-gui: starting a daemon on {args.connect}", flush=True)
        spawn_daemon(args.connect, args.display)
        owns = True

    from meegxer.gui.main_window import MainWindow

    window = MainWindow(
        DaemonLink(args.connect), load_theme(args.theme), owns_daemon=owns
    )
    if args.pipeline is not None:
        window.doc.load_pipeline(json.loads(args.pipeline.read_text(encoding="utf-8")))
        window._default_taps()
        window.refresh_chain()
    window.show()
    if args.start:
        window.start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
