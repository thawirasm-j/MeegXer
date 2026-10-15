"""``meegxer-daemon``: the headless core process (rule 1)."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from meegxer.core.control import DEFAULT_CONTROL, DEFAULT_DISPLAY, Daemon


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meegxer-daemon", description=__doc__)
    parser.add_argument("--control", default=DEFAULT_CONTROL, help="JSON-RPC endpoint")
    parser.add_argument("--display", default=DEFAULT_DISPLAY, help="display endpoint")
    parser.add_argument("--pipeline", type=Path, help="pipeline JSON to load")
    parser.add_argument("--start", action="store_true", help="start it immediately")
    parser.add_argument(
        "--no-warm-up", action="store_true",
        help="do not preload slow node dependencies in the background",
    )  # fmt: skip
    args = parser.parse_args(argv)

    daemon = Daemon(args.control, args.display)
    if not args.no_warm_up:
        daemon.warm_up_nodes()
    if args.pipeline is not None:
        daemon.load_pipeline(json.loads(args.pipeline.read_text(encoding="utf-8")))
        if args.start:
            daemon.start()

    def _shutdown(signum: int, frame: object) -> None:
        daemon.request_shutdown()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    print(f"meegxer-daemon control={daemon.control_endpoint} "
          f"display={daemon.display_endpoint}", flush=True)  # fmt: skip
    daemon.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
