# MeegXer

Explore brain signals with a user-friendly EEG mixer.

Pre-MVP. Design: [`docs/DESIGN.md`](docs/DESIGN.md). Plan: [`docs/WORKPLAN.md`](docs/WORKPLAN.md).

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.13+ (developed on 3.14).

```bash
uv sync --all-packages          # create .venv with every workspace package
uv run pytest                   # tests, including architecture guard self-tests
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run lint-imports             # architecture rules (CLAUDE.md)
uv run python tools/check_sterile.py   # sterile build has no plugin runtime
uv run pre-commit install       # optional: run the checks on commit
```

Run a pipeline offline (see `spec/pipeline.schema.json`):

```bash
uv run meegxer nodes            # installed nodes
uv run meegxer run pipeline.json --input sub-01.set --output sub-01_clean.set
```

| Package | Role |
|---|---|
| `packages/meegxer-core` | headless engine (no GUI) |
| `packages/meegxer-sterile` | audited built-in nodes |
| `packages/meegxer-runtime` | containerized plugin runtime (research builds only) |
| `packages/meegxer-gui` | PySide6 + pyqtgraph client |
| `packages/meegxer-cli` | command-line interface (`meegxer`) |

Licensed under GPL-3.0 (see `LICENSE`).
