# MeegXer

Explore brain signals with a user-friendly EEG mixer.

Alpha (0.1.0a). Design: [`docs/DESIGN.md`](docs/DESIGN.md). Plan: [`docs/WORKPLAN.md`](docs/WORKPLAN.md).

## Install

MeegXer is on PyPI as `meegxer`. The [uv](https://docs.astral.sh/uv/) tool installs
it with its own Python (3.14) into a private folder and puts the commands on your
PATH; nothing else on the system changes.

### Windows 10/11

1. Open **PowerShell** (Start menu, type "PowerShell").
2. Install uv:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Close and reopen PowerShell so `uv` is found.
3. Install MeegXer (the first run downloads Python and the packages, a few hundred
   MB; `--prerelease allow` is needed while MeegXer is in alpha):

   ```powershell
   uv tool install --python 3.14 --prerelease allow meegxer
   ```

4. Start the mixer:

   ```powershell
   meegxer-gui
   ```

   The GUI starts the processing engine (the daemon) in the background if none is
   running. Closing the GUI while a pipeline runs leaves the engine running; stop
   it with `meegxer stop`, or `meegxer shutdown` to end the daemon.

- **Network drives:** Python cannot load its libraries from a network share
  ("DLL load failed ... The parameter is incorrect"). If your user folder is on a
  network drive (common on university PCs), keep uv's Python and tools on a local
  folder instead; run this before step 3 (or set both permanently in *Edit
  environment variables for your account*):

  ```powershell
  $env:UV_PYTHON_INSTALL_DIR = "$env:LOCALAPPDATA\uv\python"
  $env:UV_TOOL_DIR = "$env:LOCALAPPDATA\uv\tools"
  ```
- **Windows Firewall:** the first time an LSL stream is opened, Windows may ask
  whether Python may use the network. Allow **private networks** so LSL can find
  amplifiers on your lab network. The control and display channels themselves only
  listen on this computer (`127.0.0.1`).
- **Update:** `uv tool upgrade meegxer`.
- **Ports in use:** the daemon uses TCP ports 5555 (control) and 5556 (display). If
  another program holds them, start with other ports:
  `meegxer-gui --connect tcp://127.0.0.1:5565 --display tcp://127.0.0.1:5566`.
  Then give `meegxer` commands the same control port, e.g.
  `meegxer stop --control tcp://127.0.0.1:5565`.

### macOS and Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install --python 3.14 --prerelease allow meegxer
meegxer-gui
```

On a minimal Linux install the GUI may need the Qt system libraries
(Debian/Ubuntu: `sudo apt install libegl1 libgl1 libxkbcommon0 libfontconfig1`).

### Research build (containerized plugins)

The `meegxer` install is the sterile build: the audited built-in nodes, no plugin
runtime. Research builds add the runtime by name, which also installs the
`meegxer-plugin` command:

```bash
uv tool install --python 3.14 --prerelease allow meegxer --with meegxer-runtime --with-executables-from meegxer-runtime
```

Plugins run in containers, so they need Docker (on Windows,
[Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)
with the WSL 2 backend). MeegXer itself, the built-in nodes and LSL do not. Then
`meegxer-plugin check IMAGE` and `meegxer-plugin add IMAGE`.

## Development

Python 3.13+ (developed on 3.14). Install uv as in [Install](#install), plus Git, then:

```bash
git clone https://github.com/thawirasm-j/MeegXer.git && cd MeegXer
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
