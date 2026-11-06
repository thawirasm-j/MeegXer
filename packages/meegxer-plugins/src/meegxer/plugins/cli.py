"""``meegxer-plugin``: install, list, remove and check plugins."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from meegxer.core.notices import LOCAL_CODE_NOTICE
from meegxer.plugins import index
from meegxer.plugins.conformance import format_report, run_conformance
from meegxer.plugins.scaffold import KINDS, new_script
from meegxer.plugins.worker import ContainerLauncher, LocalLauncher


def _params(items: list[str]) -> dict[str, object]:
    params: dict[str, object] = {}
    for item in items:
        name, _, value = item.partition("=")
        try:
            params[name] = json.loads(value)
        except ValueError:
            params[name] = value
    return params


def _missing(script: Path) -> str:
    """Why ``script`` cannot run here (its ``REQUIRES`` not installed, or
    written wrong), or empty."""
    from meegxer.plugins.requires import installed
    from meegxer.plugins.script import ScriptError

    try:
        installed(script, also=index.declared_requires())
    except ScriptError as exc:
        return str(exc)
    return ""


def _accepted(args: argparse.Namespace) -> bool:
    """Show the local-code notice; accepted with --accept-risk or a typed yes."""
    print(LOCAL_CODE_NOTICE, file=sys.stderr)
    if args.accept_risk:
        return True
    if not sys.stdin.isatty():
        print("\nRead the notice, then run again with --accept-risk.", file=sys.stderr)
        return False
    answer = input("\nType yes to add this plugin: ")
    return answer.strip().lower() == "yes"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meegxer-plugin", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="run the conformance suite")
    check.add_argument("image", nargs="?", help="container image")
    check.add_argument("--engine", help="docker or podman")
    check.add_argument(
        "--script",
        type=Path,
        help="a custom script: a .py file, or a folder (its helpers and data "
        "files with it)",
    )
    check.add_argument("--param", action="append", default=[], help="name=value")
    check.add_argument(
        "--local",
        nargs=argparse.REMAINDER,
        help="run a local command instead (unsandboxed)",
    )

    add = sub.add_parser(
        "add", help="install a plugin: an image, a custom script or a local command"
    )
    add.add_argument("image", nargs="?", help="container image")
    add.add_argument("--engine")
    add.add_argument(
        "--script",
        type=Path,
        help="a custom script: a .py file, or a folder (its helpers and data "
        "files with it)",
    )
    add.add_argument(
        "--accept-risk",
        action="store_true",
        help="accept the local-code notice (local plugins run unsandboxed)",
    )
    add.add_argument(
        "--local",
        nargs=argparse.REMAINDER,
        help="a local command speaking the plugin protocol (unsandboxed)",
    )

    new = sub.add_parser(
        "new", help="write a custom script from a template (docs/PORTING.md)"
    )
    new.add_argument("name", help="the script's name, e.g. my_filter")
    new.add_argument(
        "--kind",
        choices=KINDS,
        default="process",
        help="process: the whole recording (MNE Raw), offline; transform: block "
        "by block, live and offline; calibrated: fit, then transform; "
        "features: values per window; epochs: the epochs an Epochs strip cut "
        "(MNE Epochs), offline",
    )
    new.add_argument("--folder", type=Path, default=Path(), help="where to write it")

    trial = sub.add_parser(
        "run", help="try a custom script on a recording (nothing is added)"
    )
    trial.add_argument("--script", type=Path, required=True)
    trial.add_argument("-i", "--input", type=Path, required=True, help="a recording")
    trial.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write the result (.set/.edf, .tsv for features, -epo.fif/.set "
        "for epochs)",
    )
    trial.add_argument("--param", action="append", default=[], help="name=value")
    trial.add_argument(
        "--epochs",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="an epochs script: a setting of the Epochs strip before it "
        "(tmin, tmax, events, baseline)",
    )
    trial.add_argument("--overwrite", action="store_true")

    listing = sub.add_parser("list", help="list installed plugins")
    listing.add_argument(
        "--json", action="store_true", help="one JSON array (name, version, ...)"
    )
    rm = sub.add_parser("remove", help="uninstall a plugin by name")
    rm.add_argument("name")
    args = parser.parse_args(argv)

    if args.command == "new":
        try:
            path = new_script(args.name, args.kind, args.folder)
        except (ValueError, FileExistsError) as exc:
            parser.error(str(exc))
        article = "an" if args.kind[0] in "aeiou" else "a"
        print(f"wrote {path} ({article} {args.kind} script): put your code where it "
              "says, then\n"
              f"  meegxer-plugin run --script {path} -i RECORDING\n"
              f"  meegxer-plugin add --script {path}")  # fmt: skip
        return 0
    if args.command == "run":
        if not args.script.exists():
            parser.error(f"no such script: {args.script}")
        if reason := _missing(args.script):
            print(f"error: {reason}", file=sys.stderr)
            return 1
        from meegxer.plugins.trial import try_script

        try:
            done = try_script(args.script, args.input, out=args.output,
                              params=_params(args.param),
                              overwrite=args.overwrite,
                              epochs=_params(args.epochs))  # fmt: skip
        except Exception as exc:  # the script's own error, in its words
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(done.summary())
        return 0
    if args.command == "check":
        if args.script and (reason := _missing(args.script)):
            print(f"error: {reason}", file=sys.stderr)
            return 1
        if args.script:
            launcher: ContainerLauncher | LocalLauncher = LocalLauncher(
                tuple(index.script_command(args.script)), allow=True
            )
        elif args.local:
            launcher = LocalLauncher(tuple(args.local), allow=True)
        elif args.image:
            launcher = ContainerLauncher(args.image, args.engine)
        else:
            parser.error("give an image, --script FILE or --local COMMAND")
        checks = run_conformance(launcher, _params(args.param))
        print(format_report(checks))
        return 0 if all(c.passed for c in checks) else 1
    if args.command == "add" and (args.script or args.local):
        if args.script and not args.script.exists():
            parser.error(f"no such script: {args.script}")
        if args.script and (reason := _missing(args.script)):
            print(f"error: {reason}", file=sys.stderr)
            return 1
        if not _accepted(args):
            return 1
        command = index.script_command(args.script) if args.script else args.local
        if args.script:  # a script must pass the conformance suite to be added
            checks = run_conformance(LocalLauncher(tuple(command), allow=True))
            if not all(c.passed for c in checks):
                print(format_report(checks), file=sys.stderr)
                print(
                    f"\n{args.script.name} was not added: fix what failed above "
                    "(docs/CUSTOM_SCRIPTS.md)",
                    file=sys.stderr,
                )
                return 1
        entry = index.add_local(
            command, accept_risk=True, script=args.script if args.script else None
        )
        print(
            f"installed {entry['manifest']['name']} {entry['manifest']['version']} "
            f"(local, unsandboxed) into {index.index_path()}"
        )
        return 0
    if args.command == "add":
        if not args.image:
            parser.error("give an image, --script FILE or --local COMMAND")
        entry = index.add_image(args.image, args.engine)
        print(
            f"installed {entry['manifest']['name']} {entry['manifest']['version']} "
            f"({entry['image_id'][:19]}) into {index.index_path()}"
        )
        return 0
    if args.command == "list":
        if args.json:
            print(json.dumps([{
                "name": e["manifest"]["name"],
                "version": e["manifest"]["version"],
                "where": e.get("image") or " ".join(e.get("command", [])),
                "unsandboxed": bool(e.get("unsandboxed")),
                "description": e["manifest"].get("description", ""),
            } for e in index.load_index()]))  # fmt: skip
            return 0
        for entry in index.load_index():
            where = entry.get("image") or " ".join(entry.get("command", []))
            flag = "  local, unsandboxed" if entry.get("unsandboxed") else ""
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
