"""``meegxer-plugin``: install, list, remove and check plugins (research builds)."""

from __future__ import annotations

import argparse
import json
import sys

from meegxer.runtime import index
from meegxer.runtime.conformance import format_report, run_conformance
from meegxer.runtime.worker import ContainerLauncher, LocalLauncher


def _params(items: list[str]) -> dict[str, object]:
    params: dict[str, object] = {}
    for item in items:
        name, _, value = item.partition("=")
        try:
            params[name] = json.loads(value)
        except ValueError:
            params[name] = value
    return params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meegxer-plugin", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="run the conformance suite")
    check.add_argument("image", nargs="?", help="container image")
    check.add_argument("--engine", help="docker or podman")
    check.add_argument(
        "--local",
        nargs=argparse.REMAINDER,
        help="run a local command instead (UNSANDBOXED, development)",
    )
    check.add_argument("--param", action="append", default=[], help="name=value")

    add = sub.add_parser("add", help="install a plugin image")
    add.add_argument("image")
    add.add_argument("--engine")

    sub.add_parser("list", help="list installed plugins")
    rm = sub.add_parser("remove", help="uninstall a plugin by name")
    rm.add_argument("name")
    args = parser.parse_args(argv)

    if args.command == "check":
        if args.local:
            launcher: ContainerLauncher | LocalLauncher = LocalLauncher(
                tuple(args.local), allow=True
            )
        elif args.image:
            launcher = ContainerLauncher(args.image, args.engine)
        else:
            parser.error("give an image or --local COMMAND")
        checks = run_conformance(launcher, _params(args.param))
        print(format_report(checks))
        return 0 if all(c.passed for c in checks) else 1
    if args.command == "add":
        entry = index.add_image(args.image, args.engine)
        print(
            f"installed {entry['manifest']['name']} {entry['manifest']['version']} "
            f"({entry['image_id'][:19]}) into {index.index_path()}"
        )
        return 0
    if args.command == "list":
        for entry in index.load_index():
            where = entry.get("image") or " ".join(entry.get("command", []))
            flag = "  UNSANDBOXED" if entry.get("unsandboxed") else ""
            name, version = entry["manifest"]["name"], entry["manifest"]["version"]
            print(f"{name:40} {version:8} {where}{flag}")
        return 0
    if args.command == "remove":
        if not index.remove(args.name):
            print(f"no plugin named {args.name}", file=sys.stderr)
            return 1
        print(f"removed {args.name}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
