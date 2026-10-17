"""``meegxer-gui``: connect to the daemon (starting one if none answers) and open
the mixer window. Flags win over the settings (``meegxer.core.settings``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from meegxer.core import settings as core_settings
from meegxer.core.control import DEFAULT_DISPLAY
from meegxer.gui.connection import DaemonLink, daemon_running, spawn_daemon


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meegxer-gui", description=__doc__)
    parser.add_argument(
        "--connect", help="daemon control endpoint (default: the setting)"
    )
    parser.add_argument(
        "--display",
        default=DEFAULT_DISPLAY,
        help="display endpoint for a daemon started by the GUI",
    )
    parser.add_argument(
        "--theme",
        help="dark, light, system, or a theme JSON file (default: the setting)",
    )
    parser.add_argument("--pipeline", type=Path, help="pipeline JSON to open")
    parser.add_argument(
        "--start", action="store_true", help="start (or run offline) right away"
    )
    args = parser.parse_args(argv)

    settings = core_settings.load_settings()
    connect = args.connect or settings["general"]["daemon"]
    app = QApplication(sys.argv[:1])
    app.setApplicationName("MeegXer")
    owns = False
    if not daemon_running(connect):
        print(f"meegxer-gui: starting a daemon on {connect}", flush=True)
        spawn_daemon(connect, args.display)
        owns = True

    from meegxer.gui.main_window import MainWindow, resolve_theme

    window = MainWindow(
        DaemonLink(connect),
        resolve_theme(args.theme) if args.theme else None,
        owns_daemon=owns,
        settings=settings,
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
