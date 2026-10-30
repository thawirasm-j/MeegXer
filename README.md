# MeegXer

Explore brain signals with a user-friendly EEG mixer.

Beta: 0.3.0b1 is the first beta; formats change only in compatible ways until
1.0.0 ([`docs/TESTING.md`](docs/TESTING.md) lists the release checks). What is new:
[`docs/RELEASE_NOTES.md`](docs/RELEASE_NOTES.md). Design: [`docs/DESIGN.md`](docs/DESIGN.md).
Plan: [`docs/WORKPLAN.md`](docs/WORKPLAN.md); the road to 1.0: [`docs/ROADMAP.md`](docs/ROADMAP.md).

MeegXer runs EEG processing as a chain of strips, like a mixing desk: live from
an LSL stream or a recording played back, or offline over whole recordings and
BIDS datasets, with the same nodes either way. Tap the chain anywhere to see the
signal, its spectrum and head plot, the features a strip computes, or the ICA
components with their ICLabel classes.

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
   MB; `--prerelease allow` is needed while MeegXer is a pre-release):

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

## Quickstart

A first look in the GUI, about five minutes, no EEG hardware needed:

1. **Start it:** `meegxer-gui`. The window opens with a synthetic EEG input, a
   high-pass and a notch in the chain, and the input set to *live*.
2. **Run live:** click **▶ Start**. The signal view shows the newest seconds;
   each ◇ between strips is a tap. Click one to show it (◆ Main), click another
   to compare; right-click makes a tap Main. *Difference* shows what the strips
   between two taps removed, *Spectrum* the power spectra with a head plot.
3. **Change it while it runs:** click a strip and edit its parameters in the
   inspector; processing goes on. **+** between strips adds one (filters,
   re-referencing, ICA + ICLabel, features, outputs).
4. **Look at features:** add *Power spectrum* or *Band power*. Its rail appears
   above the chain; click the rail's name to see a spectrum per channel, or each
   channel's band power over time.
5. **Process a recording offline:** on the Input strip pick *Recording file*
   (EEGLAB `.set`, EDF/BDF, BrainVision, FIF) or *BIDS recording*, check
   channels and positions under *Recording…*, then **▶ Run offline**. Strips that learn (ICA) fit on the
   whole recording; scroll through it with the time bar. Click the ICs rail's
   name to see the components coloured by ICLabel class. With *BIDS recording*,
   leave *Subject* empty to run every subject into the dataset's derivatives
   (the views then show the last one).
6. **Keep it:** *Save…* writes the chain as a pipeline file; *Save session…* also
   keeps the fitted models (ICA) so a rerun gives the same output.

**Start from a template:** *New…* offers an ERP study, resting-state spectra
and ICA cleaning, each reading a BIDS dataset into its BIDS derivatives. Tick
*Use the sample data* to run it on a public dataset: ERP CORE's flanker task
(one subject) for the ERP study, the EEGBCI eyes open and eyes closed baseline
runs (109 subjects) for the others. Each subject is downloaded (with MNE's
fetchers, then written as BIDS) the first time a run reads it, so a run over
three subjects fetches three. The datasets go to `meegxer-samples` in MNE's data
folder (Settings → Data & storage, or `MEEGXER_SAMPLE_DIR`).

```bash
meegxer templates                        # the templates and sample datasets
meegxer new rest rest.json --sample      # a pipeline over the EEGBCI sample
meegxer run rest.json --bids ~/mne_data/meegxer-samples/eegbci --subjects 001,002
```

The same pipeline runs without the GUI:

```bash
meegxer run pipeline.json -i sub-01.set -o sub-01_clean.set   # one recording
meegxer run pipeline.json --bids ds --task rest              # a BIDS dataset
meegxer run pipeline.json --bids ds --jobs 4                # 4 at once, QC in ds/derivatives/meegxer/qc.html
```

More in [Development](#development) below and in `spec/` (pipeline, session,
control API and plugin formats).

## Plugins and Sterile mode

The `meegxer` install includes everything: the built-in nodes, the plugin runtime
and the `meegxer-plugin` command. Processing runs on your machine.

**Custom scripts:** your own MNE-Python function becomes a strip:
`process(raw, ...)` for offline runs, `transform(data, state, ...)` for live and
offline, or `features(data, ...)` for windowed feature extraction; see [docs/CUSTOM_SCRIPTS.md](docs/CUSTOM_SCRIPTS.md). In the
GUI: *+ Add strip → Add a custom script…*. It runs on your computer with your
permissions, so MeegXer asks you to accept that first.

Plugin images run in containers, so they need Docker (on Windows,
[Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)
with the WSL 2 backend) or Podman. MeegXer itself, the built-in nodes, custom
scripts and LSL do not. Then `meegxer-plugin check IMAGE` and
`meegxer-plugin add IMAGE`.

Planned: **Sterile mode**, one switch that runs all processing in a container cut
off from the network and the rest of the machine. Sterile means isolation only;
auditing the pipeline for compliance (HIPAA, GDPR, IRB) stays your responsibility.

Upgrading from 0.2.0 or earlier: `meegxer-sterile` is now `meegxer-nodes` and
`meegxer-runtime` is now `meegxer-plugins`. Old pipelines and sessions still load.
The old packages can be removed (`pip uninstall meegxer-sterile meegxer-runtime`;
a `uv tool install` reinstall drops them by itself).

## Development

Python 3.13+ (developed on 3.14). Install uv as in [Install](#install), plus Git, then:

```bash
git clone https://github.com/thawirasm-j/MeegXer.git && cd MeegXer
uv sync --all-packages          # create .venv with every workspace package
uv run pytest                   # tests, including architecture guard self-tests
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run lint-imports             # architecture rules (CLAUDE.md)
uv run pre-commit install       # optional: run the checks on commit
```

Run a pipeline offline (see `spec/pipeline.schema.json`):

```bash
uv run meegxer nodes            # installed nodes
uv run meegxer run pipeline.json --input sub-01.set --output sub-01_clean.set
```

Save what a run used (pipeline, node versions, fitted models such as ICA, file
hashes, software versions) as a session file, and rerun it later or elsewhere with
identical output (format: `spec/session.md`):

```bash
uv run meegxer run pipeline.json -i sub-01.set -o sub-01_clean.set --session sub-01.meegxer.json
uv run meegxer session check sub-01.meegxer.json   # what would be reused or refitted
uv run meegxer run sub-01.meegxer.json             # same output, byte for byte
```

To send a session to someone else, share a copy that names no files or folders
(`docs/DEIDENTIFICATION.md`); they rerun it on their own recording:

```bash
uv run meegxer session share sub-01.meegxer.json shared.meegxer.json
uv run meegxer run shared.meegxer.json -i their.set -o their_clean.set
```

Write processed recordings into a BIDS derivatives dataset (`<dataset>/derivatives/meegxer`,
layout in `spec/bids-derivatives.md`), one recording or a whole dataset; finished
recordings are skipped on a rerun, so an interrupted run continues:

```bash
uv run meegxer run pipeline.json -i ds/sub-01/eeg/sub-01_task-rest_eeg.set --bids-out ds/derivatives/meegxer
uv run meegxer run pipeline.json --bids ds --task rest            # every subject
uv run meegxer run pipeline.json --bids ds --subjects 01,02,05    # some of them
```

In the GUI, *Save session…* saves the chain with the models fitted by the last
offline run (or calibrated live), and *Open…* reads sessions as well as pipelines.

Run live in the headless daemon and control it (see `spec/control-api.md`):

```bash
uv run meegxer daemon --pipeline live.json --start   # terminal 1
uv run meegxer status                                # terminal 2
uv run meegxer set hp low=0.5
```

A live recording to .set or EDF is spooled to disk as it comes (`rec.set.part`)
and written when the run ends. If the run crashes, `uv run meegxer recover rec.set`
writes the file from the spool.

| Package | Role |
|---|---|
| `packages/meegxer-core` | headless engine (no GUI) |
| `packages/meegxer-nodes` | built-in nodes |
| `packages/meegxer-plugins` | plugin runtime and SDKs |
| `packages/meegxer-gui` | PySide6 + pyqtgraph client |
| `packages/meegxer-cli` | command-line interface (`meegxer`) |

Contributions are welcome: see `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.
Report security problems privately, as `SECURITY.md` describes.

Licensed under GPL-3.0 (see `LICENSE`).

## Citing

If MeegXer helps your work, please cite it (`CITATION.cff`, or GitHub's *Cite this
repository*) together with the methods your pipeline used: each strip lists its
references under *Credits*, and BIDS derivatives collect them in
`dataset_description.json` (`ReferencesAndLinks`).
