# MeegXer

Explore brain signals with a user-friendly EEG mixer.

Pre-MVP. Design: [`docs/DESIGN.md`](docs/DESIGN.md). Plan: [`docs/WORKPLAN.md`](docs/WORKPLAN.md).

## Install

MeegXer is not on PyPI yet, so it runs from a copy of this repository. The
[uv](https://docs.astral.sh/uv/) tool installs the right Python (3.14, from
`.python-version`) and every dependency into a private `.venv` folder; nothing is
installed system-wide.

### Windows 10/11

1. Open **PowerShell** (Start menu, type "PowerShell").
2. Install uv and Git (skip either if you already have it):

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   winget install --id Git.Git -e
   ```

   Close and reopen PowerShell so both commands are found.
3. Get MeegXer and install it (the first run downloads Python and the packages,
   a few hundred MB):

   ```powershell
   cd $HOME
   git clone https://github.com/thawirasm-j/MeegXer.git
   cd MeegXer
   uv sync --all-packages
   ```

4. Start the mixer:

   ```powershell
   uv run meegxer-gui
   ```

   The GUI starts the processing engine (the daemon) in the background if none is
   running. Closing the GUI while a pipeline runs leaves the engine running; stop
   it with `uv run meegxer stop`, or `uv run meegxer shutdown` to end the daemon.

- **Windows Firewall:** the first time an LSL stream is opened, Windows may ask
  whether Python may use the network. Allow **private networks** so LSL can find
  amplifiers on your lab network. The control and display channels themselves only
  listen on this computer (`127.0.0.1`).
- **Update:** `cd $HOME\MeegXer`, then `git pull` and `uv sync --all-packages`.
- **Ports in use:** the daemon uses TCP ports 5555 (control) and 5556 (display). If
  another program holds them, start with other ports:
  `uv run meegxer-gui --connect tcp://127.0.0.1:5565 --display tcp://127.0.0.1:5566`.
  Then give `meegxer` commands the same control port, e.g.
  `uv run meegxer stop --control tcp://127.0.0.1:5565`.
- **Without `uv run`:** the programs are also in `.venv\Scripts\` (for example
  `.venv\Scripts\meegxer-gui.exe`), which you can pin to the taskbar or use for a
  desktop shortcut.
- **Plugins (optional, research builds):** containerized plugins need
  [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)
  with the WSL 2 backend. MeegXer itself, the built-in nodes and LSL do not need
  Docker. Then `uv run meegxer-plugin check IMAGE` and
  `uv run meegxer-plugin add IMAGE`.

### macOS and Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/thawirasm-j/MeegXer.git && cd MeegXer
uv sync --all-packages
uv run meegxer-gui
```

On a minimal Linux install the GUI may need the Qt system libraries
(Debian/Ubuntu: `sudo apt install libegl1 libgl1 libxkbcommon0 libfontconfig1`).

## Development

Python 3.13+ (developed on 3.14); set up as in [Install](#install).

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

Run live in the headless daemon and control it (see `spec/control-api.md`):

```bash
uv run meegxer daemon --pipeline live.json --start   # terminal 1
uv run meegxer status                                # terminal 2
uv run meegxer set hp low=0.5
```

| Package | Role |
|---|---|
| `packages/meegxer-core` | headless engine (no GUI) |
| `packages/meegxer-sterile` | audited built-in nodes |
| `packages/meegxer-runtime` | containerized plugin runtime (research builds only) |
| `packages/meegxer-gui` | PySide6 + pyqtgraph client |
| `packages/meegxer-cli` | command-line interface (`meegxer`) |

Licensed under GPL-3.0 (see `LICENSE`).
