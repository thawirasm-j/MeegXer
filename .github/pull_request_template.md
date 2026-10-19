## What and why

<!-- What this changes and why. For a fix: the cause, and why the change fixes it.
Roadmap item (e.g. M9.8) or issue it closes. -->

## How it was tested

<!-- The tests added or changed; manual checks if any. -->

## Checklist

- [ ] `ruff check`, `ruff format --check`, `mypy` (also `--platform win32`),
      `lint-imports` and `pytest` pass locally
- [ ] Tests cover the change (a fix has a test that fails without it)
- [ ] Docs, `spec/` (with its version) and `docs/RELEASE_NOTES.md` updated where
      the change is visible to users or changes a format
- [ ] No architecture rule in `CLAUDE.md` changes (or it was discussed in an issue first)
