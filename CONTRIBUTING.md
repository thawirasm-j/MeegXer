# Contributing to MeegXer

Thank you for helping. Bug reports, questions, documentation fixes and code are
all welcome. By taking part you agree to follow the
[code of conduct](CODE_OF_CONDUCT.md).

## Reporting bugs and asking for features

Open an issue with the *Bug report* or *Feature request* form. For a bug, the
version, your operating system, the pipeline (or session file) and the steps
that show the problem help most. Security problems go privately, see
[`SECURITY.md`](SECURITY.md); please do not attach recordings of real people.

## Setting up

MeegXer is a [uv](https://docs.astral.sh/uv/) workspace (Python 3.13 or newer;
development uses 3.14):

```
git clone https://github.com/thawirasm-j/MeegXer
cd MeegXer
uv sync --all-packages
uv run meegxer-gui
```

## Before you open a pull request

Run what CI runs:

```
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run mypy --platform win32
uv run lint-imports
uv run pytest
```

- **One change per pull request**, with tests for it (a fix comes with a test
  that fails without it). Roadmap items keep their IDs (`M9.8`, …) in the title.
- **Fix the cause.** Describe what was wrong and why the change fixes it; keep
  the change to what the fix needs.
- **Match the code around you**: naming, comments and the existing style.
- **Docs and specs move with the code.** A change to a format or protocol
  updates `spec/` (with its version) and the release notes
  (`docs/RELEASE_NOTES.md`).

## The architecture rules

`CLAUDE.md` lists the rules the design rests on (a headless core that never
waits for the GUI, one node implementation for live and offline runs, plugins as
untrusted code that only emit events, Sterile mode as isolation, a
language-neutral plugin spec). Changing one of them needs a discussion in an
issue first; `docs/DESIGN.md` has the reasons behind each.

## New nodes and plugins

Built-in nodes register in the `meegxer.nodes` entry-point group and implement
the fit/transform contract (`docs/DESIGN.md` §3.3); one that runs live must pass
`meegxer.core.conformance.check_blockwise_equivalence`. Plugins follow
`spec/plugin-protocol.md` and pass `meegxer-plugin check IMAGE`; custom scripts
are described in `docs/CUSTOM_SCRIPTS.md`.

## Licence

MeegXer is licensed under the GNU General Public License v3.0 or later
([`LICENSE`](LICENSE)). Contributions are accepted under the same licence.
