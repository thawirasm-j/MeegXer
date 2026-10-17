# Manual release checks

Two checks that need a real machine and real data, done before tagging the first
beta, 0.3.0b1, and again before 1.0.0 (`docs/DESIGN.md` §6, `docs/WORKPLAN.md` M7). Tick each box; write down what you
see in the *Notes* column of the result tables, especially anything unexpected.

---

## A. Acceptance test 5: the full suite on Windows, Docker plugins included

**Pass criterion:** every step below passes on Windows 10/11 with Docker Desktop
(WSL 2 backend). CI already runs the test suite on Windows without Docker; this
adds the containerized plugins, the stability run and the GUI with a plugin.

Use PowerShell. Work on a **local disk**: a project or uv folder on a network share
fails with "DLL load failed … The parameter is incorrect".

### A1. Set up (once)

- [ ] **Docker Desktop** installed with the WSL 2 backend and running
      (<https://docs.docker.com/desktop/setup/install/windows-install/>). Check:
      `docker run --rm hello-world` prints "Hello from Docker!".
- [ ] **uv** and **Git**:
      ```powershell
      powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
      winget install --id Git.Git -e
      ```
      Close and reopen PowerShell.
- [ ] If your user folder is on a network drive:
      ```powershell
      $env:UV_PYTHON_INSTALL_DIR = "$env:LOCALAPPDATA\uv\python"
      $env:UV_CACHE_DIR = "$env:LOCALAPPDATA\uv\cache"
      ```
- [ ] Get the code on a local disk and install every package:
      ```powershell
      mkdir C:\dev -Force; cd C:\dev
      git clone https://github.com/thawirasm-j/MeegXer.git
      cd MeegXer
      uv sync --all-packages --locked
      ```
      Note the commit for the result table: `git log --oneline -1`.

### A2. Static checks

- [ ] `uv run ruff check .` prints "All checks passed!"
- [ ] `uv run ruff format --check .` reports all files formatted
- [ ] `uv run mypy` prints "Success: no issues found"
- [ ] `uv run lint-imports` prints "Contracts: 3 kept, 0 broken."

### A3. Test suite without Docker

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
uv run pytest
```

- [ ] Ends with `N passed, M skipped` and **no failures or errors**.
- [ ] The skips are only these (`uv run pytest -rs` lists the reasons):
      "set MEEGXER_TEST_IMAGES …", "needs the meegxer/eeglab-asr image …",
      "GNU Octave not installed". Any other skip reason: note it.

### A4. Build the reference plugin images

```powershell
cd packages\meegxer-plugins
docker build -f plugins/average_reference/Dockerfile -t meegxer/example-average-reference:0.1.0 .
docker build -f plugins/octave_iir_bandpass/Dockerfile -t meegxer/example-octave-iir-bandpass:0.1.0 .
docker build -f plugins/eeglab_asr/Dockerfile -t meegxer/eeglab-asr:0.1.0 .
cd ..\..
```

- [ ] All three builds finish. The ASR image downloads EEGLAB and takes several
      minutes; it needs internet.
- [ ] `docker images meegxer/*` lists the three images.

### A5. Plugin conformance inside Docker

```powershell
uv run meegxer-plugin check meegxer/example-average-reference:0.1.0
uv run meegxer-plugin check meegxer/example-octave-iir-bandpass:0.1.0
uv run meegxer-plugin check meegxer/eeglab-asr:0.1.0
```

- [ ] Each one passes (protocol, acceptance test 1, restart).

### A6. Plugin tests and ASR concordance in the sandbox

```powershell
$env:MEEGXER_TEST_IMAGES = "meegxer/example-average-reference:0.1.0,meegxer/example-octave-iir-bandpass:0.1.0,meegxer/eeglab-asr:0.1.0"
uv run pytest packages/meegxer-plugins -rs
```

- [ ] No failures. The "set MEEGXER_TEST_IMAGES" and "needs the meegxer/eeglab-asr
      image" skips are gone (ASR concordance against EEGLAB ran).

### A7. Stability and latency with a plugin (acceptance tests 2 and 3 on Windows)

Start with 2 minutes; if that passes, run the full 30.

```powershell
uv run python tools/stability_test.py --minutes 2 --plugin-image meegxer/example-average-reference:0.1.0
uv run python tools/stability_test.py --minutes 30 --plugin-image meegxer/example-average-reference:0.1.0 --report stability-windows.json
```

- [ ] The report ends with `"passed": true` (zero dropped or duplicated samples at
      64 ch / 1 kHz).
- [ ] Note the latency p95 (target < 150 ms). Docker Desktop's file and pipe
      overhead on Windows is a known risk (`docs/WORKPLAN.md`); if it is close
      to the target, note the number.

### A8. GUI with a plugin

```powershell
uv run meegxer-plugin add meegxer/example-average-reference:0.1.0
uv run meegxer-gui
```

- [ ] The GUI opens; Start shows the synthetic signal.
- [ ] *+* (Add strip) lists the plugin (`examples.average_reference`) under
      Processing; insert it while running: the chain restarts and the signal keeps
      flowing through the plugin.
- [ ] Click its strip, change nothing, Stop, Start: no errors in the log.
- [ ] Close the GUI, then `uv run meegxer shutdown`; afterwards
      `uv run meegxer-plugin remove examples.average_reference`.

### A9. Result

| Item | Value / result | Notes |
|---|---|---|
| Windows version (`winver`) | | |
| Docker Desktop version | | |
| Commit (`git log --oneline -1`) | | |
| A2 static checks | pass / fail | |
| A3 tests (passed / skipped) | | |
| A5 conformance ×3 | pass / fail | |
| A6 plugin tests | pass / fail | |
| A7 30 min stability, latency p95 | pass / fail, ___ ms | |
| A8 GUI with plugin | pass / fail | |

---

## B. BIDS data: real recordings through the derivatives output

**Goal:** a real BIDS EEG dataset goes through a cleaning pipeline into
`derivatives/meegxer`, the outputs open in EEGLAB and MNE, and an interrupted run
continues. Use the checkout from part A (`git pull` first), so commands run as
`uv run meegxer …`; with v0.1.0a6 or later from PyPI, drop the `uv run`.

### B1. Choose and check the dataset

- [ ] A BIDS EEG dataset on a local disk: your own, or a small one from
      [OpenNeuro](https://openneuro.org) (search "EEG"; a few subjects is enough
      for the first pass). Write its path below as `DS`.
- [ ] The raw dataset validates: <https://bids-standard.github.io/bids-validator/>
      (runs in your browser, nothing is uploaded). Note warnings; errors in the raw
      data are not MeegXer's, but note them.
- [ ] Note: task name(s), number of subjects, file format (.set/.edf/.vhdr/.bdf),
      whether there is an `electrodes.tsv` (positions; ICLabel needs them).

### B2. The pipeline

Save as `clean.json` (set the notch to your mains frequency, 50 or 60 Hz):

```json
{"spec_version": "0.2.10", "nodes": [
  {"id": "input", "node": "meegxer.core.synthetic"},
  {"id": "hp", "node": "meegxer.nodes.iir_filter", "params": {"kind": "highpass", "low": 1.0}},
  {"id": "notch", "node": "meegxer.nodes.notch", "params": {"freq": 50.0}},
  {"id": "car", "node": "meegxer.nodes.rereference"},
  {"id": "out", "node": "meegxer.core.bids_writer"},
  {"id": "bp", "node": "meegxer.nodes.band_power"},
  {"id": "feat", "node": "meegxer.core.bids_features"}
]}
```

The input is replaced by each recording of the dataset; its *Recording* settings
(channel renames, types, montage, bad channels) apply to all of them. A dataset
without `electrodes.tsv` needs a montage there before ICA (step B6).

### B3. One recording

```powershell
uv run meegxer run clean.json -i DS\sub-01\eeg\sub-01_task-TASK_eeg.EXT --bids-out DS\derivatives\meegxer
```

- [ ] Ends with "computed: …" and no error.
- [ ] `DS\derivatives\meegxer\sub-01\eeg\` holds `…_desc-clean_eeg.set`,
      `_eeg.json`, `_channels.tsv`, `_events.tsv` (if the recording has events),
      `_electrodes.tsv` + `_coordsystem.json` (if positions are known),
      `_bp_features.tsv` + `.json` and `_meegxer.json`.
- [ ] `dataset_description.json` and `.bidsignore` exist in `derivatives\meegxer`.

### B4. Open the outputs elsewhere

- [ ] **EEGLAB:** `EEG = pop_loadset('…_desc-clean_eeg.set');` loads; channel
      names and count match; `pop_eegplot(EEG)` looks filtered (no slow drift,
      no mains line); events are present.
- [ ] **MNE-BIDS** (Python):
      ```python
      import mne_bids

      p = mne_bids.BIDSPath(
          root=r"DS\derivatives\meegxer",
          subject="01",
          task="TASK",
          description="clean",
          datatype="eeg",
          suffix="eeg",
          extension=".set",
      )
      raw = mne_bids.read_raw_bids(p)
      print(raw, raw.info["bads"], raw.annotations)
      raw.compute_psd().plot()  # mains peak gone, below 1 Hz attenuated
      ```
- [ ] `_eeg.json`: `Sources` points at the raw file; `Description` lists the
      chain; `PowerLineFrequency` and other fields came over from the raw sidecar.
- [ ] `_channels.tsv`: types (EEG/EOG/…) and `status` as expected.
- [ ] `_bp_features.tsv`: one row per window, plausible band-power values.
- [ ] Validate the derivatives folder with the BIDS validator: note errors and
      warnings (the EEG-derivatives extension is still a draft, so some warnings
      may be expected; send them to me).

### B5. The whole dataset, then an interrupted run

```powershell
uv run meegxer run clean.json --bids DS --task TASK
```

- [ ] Each recording prints `[i/N] sub-XX_task-…: wrote …`; it ends with
      `done N, skipped 0, failed 0` (sub-01 shows as skipped if B3's output is
      still there).
- [ ] Note the time per recording.
- [ ] Run it again: `done 0, skipped N`.
- [ ] Delete one subject's derivative folder, start the loop and press **Ctrl+C**
      while that subject is being written; run the same command again: it redoes
      the interrupted subject and skips the rest.
- [ ] If any recording failed, the summary names it and the exit status is 1
      (`echo $LASTEXITCODE`); note the error message.

### B6. ICA on real data

Insert after `car` (positions needed: from `electrodes.tsv`, or a montage in the
input's `recording`, e.g. `"recording": {"montage": "colin27_1005"}`):

```json
  {"id": "ica", "node": "meegxer.nodes.ica"},
```

```powershell
uv run meegxer run clean.json --bids DS --subjects 01 --task TASK --overwrite
```

- [ ] Finishes with `fitted: ica`.
- [ ] `_meegxer.json` holds `models.ica` (readable JSON arrays).
- [ ] Eye blinks are visibly reduced in the cleaned data (EEGLAB/MNE plot).
- [ ] `uv run meegxer session check DS\derivatives\meegxer\sub-01\eeg\sub-01_task-TASK_desc-clean_meegxer.json`
      reports `model ica: reused`.

### B7. In the GUI

- [ ] `uv run meegxer-gui`; Input → *File*; set the path to a raw recording of the
      dataset; mode shows OFFLINE.
- [ ] *+* → Outputs → *BIDS writer*; Run offline: the derivative appears in
      `DS\derivatives\meegxer` with its `_meegxer.json`.

### B8. Result

| Item | Value / result | Notes |
|---|---|---|
| Dataset (name / source), subjects, task, format | | |
| B3 single recording | pass / fail | |
| B4 EEGLAB / MNE-BIDS read-back | pass / fail | |
| B4 validator on derivatives (errors / warnings) | | |
| B5 whole dataset: done / failed, time per recording | | |
| B5 rerun skip, interrupt and resume | pass / fail | |
| B6 ICA (blinks reduced, session check) | pass / fail | |
| B7 GUI offline BIDS writer | pass / fail | |
