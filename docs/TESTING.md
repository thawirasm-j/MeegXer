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

- [x] **Docker Desktop** installed with the WSL 2 backend and running
      (<https://docs.docker.com/desktop/setup/install/windows-install/>). Check:
      `docker run --rm hello-world` prints "Hello from Docker!".
- [x] **uv** and **Git**:
      ```powershell
      powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
      winget install --id Git.Git -e
      ```
      Close and reopen PowerShell.
- [x] If your user folder is on a network drive:
      ```powershell
      $env:UV_PYTHON_INSTALL_DIR = "$env:LOCALAPPDATA\uv\python"
      $env:UV_CACHE_DIR = "$env:LOCALAPPDATA\uv\cache"
      ```
- [x] Get the code on a local disk and install every package:
      ```powershell
      mkdir C:\dev -Force; cd C:\dev
      git clone https://github.com/thawirasm-j/MeegXer.git
      cd MeegXer
      uv sync --all-packages --locked
      ```
      Note the commit for the result table: `git log --oneline -1`.

### A2. Static checks

- [x] `uv run ruff check .` prints "All checks passed!"
- [x] `uv run ruff format --check .` reports all files formatted
- [x] `uv run mypy` prints "Success: no issues found"
- [x] `uv run lint-imports` prints "Contracts: 3 kept, 0 broken."

### A3. Test suite without Docker

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
uv run pytest
```

- [x] Ends with `N passed, M skipped` and **no failures or errors**.
- [x] The skips are only these (`uv run pytest -rs` lists the reasons):
      "set MEEGXER_TEST_IMAGES …", "needs the meegxer/eeglab-asr image …",
      "GNU Octave not installed", and on Windows "POSIX signals" (a Linux/macOS
      test that hard-kills the GUI; CI runs it there). Any other skip reason:
      note it.

### A4. Build the reference plugin images

```powershell
cd packages\meegxer-plugins
docker build -f plugins/average_reference/Dockerfile -t meegxer/example-average-reference:0.1.0 .
docker build -f plugins/octave_iir_bandpass/Dockerfile -t meegxer/example-octave-iir-bandpass:0.1.0 .
docker build -f plugins/eeglab_asr/Dockerfile -t meegxer/eeglab-asr:0.1.0 .
cd ..\..
```

- [x] All three builds finish. The ASR image downloads EEGLAB and takes several
      minutes; it needs internet.
- [x] `docker images meegxer/*` lists the three images.

### A5. Plugin conformance inside Docker

```powershell
uv run meegxer-plugin check meegxer/example-average-reference:0.1.0
uv run meegxer-plugin check meegxer/example-octave-iir-bandpass:0.1.0
uv run meegxer-plugin check meegxer/eeglab-asr:0.1.0
```

- [x] Each one passes (protocol, acceptance test 1, restart).

### A6. Plugin tests and ASR concordance in the sandbox

```powershell
$env:MEEGXER_TEST_IMAGES = "meegxer/example-average-reference:0.1.0,meegxer/example-octave-iir-bandpass:0.1.0,meegxer/eeglab-asr:0.1.0"
uv run pytest packages/meegxer-plugins -rs
```

- [x] No failures. The "set MEEGXER_TEST_IMAGES" and "needs the meegxer/eeglab-asr
      image" skips are gone (ASR concordance against EEGLAB ran). "GNU Octave not
      installed" may remain: that test runs the Octave plugin with Octave on the
      host, and the same plugin was just checked inside its image.

### A7. Stability and latency with a plugin (acceptance tests 2 and 3 on Windows)

Start with 2 minutes; if that passes, run the full 30.

```powershell
uv run python tools/stability_test.py --minutes 2 --plugin-image meegxer/example-average-reference:0.1.0
uv run python tools/stability_test.py --minutes 30 --plugin-image meegxer/example-average-reference:0.1.0 --report stability-windows.json
```

- [x] The report ends with `"passed": true` (zero dropped or duplicated samples at
      64 ch / 1 kHz).
- [x] Note the latency p95 (target < 150 ms). Docker Desktop's file and pipe
      overhead on Windows is a known risk (`docs/WORKPLAN.md`); if it is close
      to the target, note the number.

### A8. GUI with a plugin

```powershell
uv run meegxer-plugin add meegxer/example-average-reference:0.1.0
uv run meegxer-gui
```

- [x] The GUI opens; Start shows the synthetic signal.
- [x] *+* (Add strip) lists the plugin (`examples.average_reference`) under
      Processing; insert it while running: the chain restarts and the signal keeps
      flowing through the plugin.
- [x] Click its strip, change nothing, Stop, Start: no errors in the log.
- [x] Close the GUI, then `uv run meegxer shutdown`; afterwards
      `uv run meegxer-plugin remove examples.average_reference`.

### A9. Result

| Item | Value / result | Notes |
|---|---|---|
| Windows version (`winver`) | OK | |
| Docker Desktop version | OK | |
| Commit (`git log --oneline -1`) | OK | |
| A2 static checks | pass | |
| A3 tests (passed / skipped) | pass | |
| A5 conformance ×3 | pass | |
| A6 plugin tests | pass | |
| A7 30 min stability, latency p95 | pass | daemon 5.6549 ms, end-to-end 15.4378 ms |
| A8 GUI with plugin | pass | |

---

## B. BIDS data: real recordings through the derivatives output

**Goal:** a real BIDS EEG dataset goes through a cleaning pipeline into
`derivatives/meegxer`, the outputs open in EEGLAB and MNE, and an interrupted run
continues. Use the checkout from part A (`git pull` first), so commands run as
`uv run meegxer …`; with v0.1.0a6 or later from PyPI, drop the `uv run`.

### B1. Choose and check the dataset

- [x] A BIDS EEG dataset on a local disk: your own, or a small one from
      [OpenNeuro](https://openneuro.org) (search "EEG"; a few subjects is enough
      for the first pass). Write its path below as `DS`.
- [x] The raw dataset validates: <https://bids-standard.github.io/bids-validator/>
      (runs in your browser, nothing is uploaded). Note warnings; errors in the raw
      data are not MeegXer's, but note them.
- [x] Note: task name(s), number of subjects, file format (.set/.edf/.vhdr/.bdf),
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

- [x] Ends with "computed: …" and no error.
- [x] `DS\derivatives\meegxer\sub-01\eeg\` holds `…_desc-clean_eeg.set`,
      `_eeg.json`, `_channels.tsv`, `_events.tsv` (if the recording has events),
      `_electrodes.tsv` + `_coordsystem.json` (if positions are known),
      `_bp_features.tsv` + `.json` and `_meegxer.json`.
- [x] `dataset_description.json` and `.bidsignore` exist in `derivatives\meegxer`.

### B4. Open the outputs elsewhere

- [x] **EEGLAB:** `EEG = pop_loadset('…_desc-clean_eeg.set');` loads; channel
      names and count match; `pop_eegplot(EEG)` looks filtered (no slow drift,
      no mains line); events are present.
- [x] **MNE-BIDS** (Python):
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
- [x] `_eeg.json`: `Sources` points at the raw file; `Description` lists the
      chain; `PowerLineFrequency` and other fields came over from the raw sidecar.
- [x] `_channels.tsv`: types (EEG/EOG/…) and `status` as expected.
- [x] `_bp_features.tsv`: one row per window, plausible band-power values.
- [x] Validate the derivatives folder with the BIDS validator: no errors and no
      warnings when the raw dataset lists its `Authors` (they are inherited;
      otherwise TOO_FEW_AUTHORS remains). Send me anything else.

### B5. The whole dataset, then an interrupted run

```powershell
uv run meegxer run clean.json --bids DS --task TASK
```

- [x] Each recording prints `[i/N] sub-XX_task-…: wrote …`; it ends with
      `done N, skipped 0, failed 0` (sub-01 shows as skipped if B3's output is
      still there).
- [x] Note the time per recording.
- [x] Run it again: `done 0, skipped N`.
- [x] Delete one subject's derivative folder, start the loop and press **Ctrl+C**
      while that subject is being written; run the same command again: it redoes
      the interrupted subject and skips the rest.
- [x] If any recording failed, the summary names it and the exit status is 1
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

- [x] Finishes with `fitted: ica`.
- [x] `_meegxer.json` holds `models.ica` (readable JSON arrays).
- [x] Eye blinks are visibly reduced in the cleaned data (EEGLAB/MNE plot).
- [x] `uv run meegxer session check DS\derivatives\meegxer\sub-01\eeg\sub-01_task-TASK_desc-clean_meegxer.json`
      reports `model ica: reused`.

### B7. In the GUI

- [x] `uv run meegxer-gui`; Input → *File*; set the path to a raw recording of the
      dataset; mode shows OFFLINE.
- [x] *+* → Outputs → *BIDS writer*; Run offline: the derivative appears in
      `DS\derivatives\meegxer` with its `_meegxer.json`.

### B8. Result

| Item | Value / result | Notes |
|---|---|---|
| Dataset (name / source), subjects, task, format | ds004504 (OpenNeuro), sub-001 to sub-009, eyesclosed, EEGLAB .set | |
| B3 single recording | pass | |
| B4 EEGLAB / MNE-BIDS read-back | pass | 0.3.1 (main after M9.8), Linux, sub-001: EEGLAB (Octave, `pop_loadset` + `eeg_checkset`) loads 19 ch, labels in order, 500 Hz, 599.8 s; it notes "xmax adjusted" (eeglabio 0.1.4 writes xmax = n/srate, one sample late; harmless). MNE-BIDS reads it, 19 ch; PSD below 0.6 Hz 3e-9 → 1e-14 V²/Hz, 49–51 Hz at its neighbours' level |
| B4 validator on derivatives (errors / warnings) | pass | no errors; TOO_FEW_AUTHORS only (after PR #54) |
| B5 whole dataset: done / failed, time per recording | done | not measured |
| B5 rerun skip, interrupt and resume | pass | |
| B6 ICA (blinks reduced, session check) | pass | Same run: ICA removed 3 components, all ICLabel eye (≥ 0.92); frontal (Fp1/Fp2, 0.5–8 Hz) events over 75 µV 15 → 0, p99.9 86 → 17 µV. Alpha (8–12 Hz, all channels) 35 % lower than without ICA: eye-component content in frontal channels |
| B7 GUI offline BIDS writer | pass | |

## C. Templates and sample data (needs the internet)

The tests use fake fetchers; this checks the real downloads (ERP CORE from
osf.io, EEGBCI from physionet.org). Use an empty sample folder:
`set MEEGXER_SAMPLE_DIR=C:\meegxer-samples` (or `export` on macOS/Linux).

1. `uv run meegxer templates` lists three templates and both samples, with
   0 of 1 and 0 of 109 subjects fetched.
2. `uv run meegxer new erp erp.json --sample`, then the `meegxer run …` line it
   prints without `--subjects`. Expect sub-001 downloaded once, then
   `derivatives\meegxer\sub-001\eeg\sub-001_task-flanker_desc-clean_epo.fif` and
   `_ave.fif`. In MNE: `mne.read_evokeds(…)` gives four ERPs (compatible and
   incompatible flankers, target left and right); no response epochs.
3. `uv run meegxer new rest rest.json --sample`, then run it with
   `--subjects 001,002`: two subjects downloaded, four recordings done
   (`resteyesopen`, `resteyesclosed`), each with `_psd_features.tsv` and
   `_bands_features.tsv`. Run it again with `--subjects 001,002,003`: only 003
   is downloaded; 001 and 002 are skipped as done.
4. GUI: *New…* → *ICA cleaning*, tick *Use the sample data*: a progress box
   while subject 001 downloads, then the chain (Input on the sample, high-pass,
   reference, ICA, BIDS writer). *Recording…* shows 64 channels with positions.
   *▶ Run offline* with *Subject* `001` runs it; with *Subject* empty it runs
   every subject, downloading each in turn (stop it after a few).
5. The BIDS validator on the sample folder: no errors.

| Item | Value / result | Notes |
|---|---|---|
| C1 listing | pass | |
| C2 ERP CORE ERPs (labels, count) | pass | 98/100/99/100 |
| C3 EEGBCI per-subject fetch, rerun | pass | |
| C4 GUI New from template with the sample | pass | no button to terminate the run, no downloading progress bar |
| C5 validator on the sample dataset | pass | with warnings |

## D. Batch workers and QC reports

1. On a BIDS dataset of at least four subjects (e.g. the EEGBCI sample after
   §C): `uv run meegxer run rest.json --bids DS --jobs 4 --overwrite`. Expect
   the recordings to finish out of order, faster than with `--jobs 1`, and
   `QC summary: …\qc.html` at the end.
2. Open `qc.html`: one row per recording, each linking to its report. A
   report shows the MeegXer table, the input and processed power spectra, and
   (ERP template) the epochs and ERPs with topographies.
3. GUI: *Settings → General → Workers* 4, then an all-subjects offline run:
   the log ends with `QC summary: …`; the views show the last recording.

| Item | Value / result | Notes |
|---|---|---|
| D1 time with --jobs 1 / --jobs 4 (recordings, CPU) | 17.54s, 8.85s | |
| D2 summary and reports | pass | |
| D3 GUI all-subjects run with workers | pass | |

## E. Live ERP view (M10.3)

1. `uv run python tools/live_erp_demo.py C:\meegxer-erp` writes `live_erp.json`
   (a synthetic input played live with `go` and `stop` markers every 1.2 s, and
   an Epochs strip) and its `events.tsv`.
2. `uv run meegxer-gui`, open `live_erp.json`, *Start*. Click the Epochs
   strip's `epochs` rail name: the ERP view replaces the signal view. Within a
   few seconds it shows `go (n=…)` and `stop (n=…)` on Cz, the counts
   growing by one each 2.4 s, the curves flattening toward 0 µV (the
   synthetic input has no evoked response).
3. *Show* `go`: the channels overlaid (butterfly); clicks on the head under
   *Channels* hide some. *Show* `All conditions`: *Channel* opens the head,
   and a click picks the one channel every condition is drawn on.
4. *Reset*: the legend reads `0 epochs`, then the counts start again at 1.
   Change the Epochs strip's `tmax` while it runs: it starts over too.
5. Offline: the ERP template on the ERP CORE sample (§C), *Start*, then the
   epochs rail: four conditions with the counts of C2.
6. The time axis reads ms (-200 … 800). Tick *SEM*: each curve gets a shaded
   band (mean ± SEM) in its colour, narrowing as epochs add up; a condition
   with a single epoch has none.
7. Set the Epochs strip's *baseline* to `custom`, *baseline start* 0.1 and
   *baseline end* 0.3: the live ERP starts over, and each curve now averages
   to zero between 100 and 300 ms. An end before the start, or a period
   outside the epoch, is refused at once with the reason (in the log).

| Item | Value / result | Notes |
|---|---|---|
| E2 live ERP view, counts growing | pass | |
| E3 butterfly and channel choice | pass | |
| E4 reset, and a change upstream | pass | |
| E5 offline ERP view (counts as C2) | pass | |
| E3b head picker: one channel, then the butterfly's channels | pass | |
| E6 ms axis and SEM band | pass | |
| E7 custom baseline (zero over 100–300 ms; refusals) | pass | |
| E8 head picker in the feature view and both review windows | pass | |
| E9 epoch times in ms (50 ms steps); long labels whole; picker buttons readable | passed | |
| E10 amplitude range: zero-centred by default, data range in Settings; y = 0 line | passed | |

## F. XDF input (M10.5)

With the XDF example files (MIT): `git clone
https://github.com/xdf-modules/example-files`, then `clock_resets.xdf`. It
holds a BioSemi EEG stream (8 channels, 100 Hz, 278 s of samples, no units
stated) and a marker stream, recorded across a clock reset: no samples for
274 s at 128.8 s. Your own LabRecorder file works too.

Before you start: `uv run meegxer shutdown`. The GUI uses a daemon that is
already running, so one started before `git pull` would still read XDF files
with the old code; after the shutdown the GUI starts a new one.

1. GUI: *Input → File*, choose `clock_resets.xdf`. The log says `Input: XDF:
   signal stream 'BioSemi', markers from 'MyMarkerStream'`, `8 of 8 channels
   state no unit: taken to be in uV`, `273.869 s of samples lost at 128.760 s
   (marked BAD_gap)` and `10 markers left out` (sent while nothing was
   recorded). The Recording window lists Ch1–Ch8 and the events: Test-1-2-3
   38, XXX 28, Testtest 28, Marker 27, Blah 24, Test 20, BAD_gap 1.
2. *Start* (offline): 278 s of signal (values up to about 1 µV: the file's
   test data), the markers as event lines, a BAD_gap mark at 128.76 s.
3. An Epochs strip on `Marker` (-200 to 800 ms): 27 epochs; the ERP view
   shows them.
4. and 5. CLI (`tools/xdf_check.py`), from the MeegXer folder, with
   `XDF` the path to `clock_resets.xdf` and `OUT` a new folder:
   - `uv run python tools/xdf_check.py write XDF OUT` writes
     `OUT/xdf_epochs.json` (the file as the input, an Epochs strip on
     `Marker`, the recording to `out.set`, the epochs to `out-epo.fif`).
   - `uv run meegxer run OUT/xdf_epochs.json` runs it (F5).
   - `uv run python tools/xdf_check.py check XDF OUT` prints
     `F4: 27 epochs; gaps at 128.76 s; epochs across a gap: 0 (PASS)` and
     `F5: out.set holds 166 events (166 in the XDF file, BAD_gap included)`.

| Item | Value / result | Notes |
|---|---|---|
| F1 streams and notes in the log; channels and events listed | passed | |
| F2 278 s of signal, events in place, BAD_gap at 128.76 s | passed | |
| F3 27 epochs on Marker; ERP view | passed | |
| F4 no epoch across the gap | passed | |
| F5 CLI run with -i clock_resets.xdf | passed | |

## G. EEGLAB decompositions (M10.6)

`uv run python tools/eeglab_ica_demo.py OUT` writes `OUT/ica_demo.set` (the
ICA test recording, 19 channels, with EEGLAB's decomposition and ICLabel
classes) and `OUT/ica_demo_unclassified.set` (without the classes). An EEGLAB
`.set` of your own with ICA works too. First `uv run meegxer shutdown`.

1. GUI: *Input → File*, `ica_demo.set`. At once (before *Start*), the log says
   `EEGLAB decomposition: 18 components on 19 channels start the ICs rail;
   ICLabel classes from the file`; the chain shows an ICs rail starting at the
   Input strip.
2. *Start* (offline), then click the ICs rail's name: 18 components IC01 to
   IC18, coloured by class, each named with its class and probability in
   *Channels*. *Spectrum*: the head plot maps them back on the electrodes.
3. Add an ICA strip: the file's ICs rail ends at its left edge (a bar) and
   ICA's ICs rail goes on from it along the same row.
4. `ica_demo_unclassified.set`: at once, the log says `ICLabel classes
   computed when a run first reads them`; after *Start*, the components have classes too.
5. Writing: `ica_demo.set`, an ICA strip, then *Output → Save as .set*
   `OUT/clean.set`; *Start*. Open `clean.set` in EEGLAB if you have it
   (*Tools → Inspect/label components*: N components with their classes, N =
   18 less the components the ICA strip removed: the eye and muscle
   components at or above its removal threshold, 0.9),
   or `uv run python -c "import mne; print(mne.preprocessing.read_ica_eeglab(r'OUT/clean.set'))"`
   (`N ICA components`). Add a re-reference strip before the writer and run
   again: the log says `ICA decomposition not stored: … transforms the
   channels after it`.

| Item | Value / result | Notes |
|---|---|---|
| G1 the log note; the ICs rail from the Input strip | passed | |
| G2 components with classes; the head plot | passed | |
| G3 an ICA strip ends the file's rail | passed | |
| G4 classes computed for the unclassified file | passed | |
| G5 the written `.set` keeps the ICA (EEGLAB or MNE); none after a re-reference | passed | |

## H. Python API (M10.7)

In the MeegXer folder after `git pull` and `uv sync --all-packages`:

```bash
uv run python tools/api_check.py --plot
```

With no recording named, it uses the ERP CORE flanker sample (downloaded the
first time, unless §C already did) and its most frequent event. For your own recording
with events (a `.set`, `.edf`, `.fif`, …), name it, and optionally one of its
event labels (the script lists them):
`uv run python tools/api_check.py C:\data\sub-01.set [LABEL] --plot`. The
script band-passes (1 to 30 Hz) and cuts epochs on the event (-200 to 800 ms)
with `meegxer.api.run`, from an `mne.io.Raw` and from the file.

1. It prints `H1 Raw: … annotations (the input: …)`: as many annotations as
   the input has; MNE's browser shows the cleaned recording with them.
2. `H2 Epochs: N on 'EVENT', -200 to 800 ms`; MNE's ERP figure looks like
   an ERP of that event.
3. `H3 from the file: N epochs (the same)`.
4. In your own Python (IPython or Jupyter, in the same environment):
   `from meegxer.api import run` and a pipeline of yours saved from the GUI
   (*Save*), e.g. `run("my_pipeline.json", raw)`: it returns an `mne.io.Raw`.

MNE ICA files: `uv run python tools/mne_ica_demo.py OUT` writes
`OUT/ica_demo_raw.fif` (the ICA test recording) and `OUT/ica_demo-ica.fif`
(MNE's own ICA of it: 15 components, component 1 marked excluded). An MNE ICA
of your own recording works too. First `uv run meegxer shutdown`.

5. GUI: *Input → File*; in the input's settings, *File* `ica_demo_raw.fif`
   (the recording), then *MNE ICA file* `ica_demo-ica.fif` (the field below
   it; both take `.fif`, and an ICA file chosen as *File* says so). At once the log says `MNE
   ICA decomposition: 15 components on 19 channels start the ICs rail; ICLabel
   classes computed when a run first reads them` and that the file marks
   component 1 for removal; the chain shows the ICs rail from the Input strip.
   After *Start*, its tap shows IC01 to IC15 with classes, one readable row
   each in *Channels*. Add a *Power spectrum* strip and set its *Reads* to
   *ICs (Recording file)*: the log says `Power spectrum reads ICs (Recording
   file).`; remove it again. A re-reference or the MNE ICA writer has no
   *Reads* box: the signal is all they read (spec 0.2.22).
6. Add an ICA strip after the input and *Output → MNE ICA file (.fif)*; in
   its *File*, *Browse…* and type `clean`: it becomes `clean-ica.fif` (MNE's
   convention). *Start*. In Python:
   `mne.preprocessing.read_ica(r"OUT/clean-ica.fif")` shows the ICA strip's
   18 components with the removed ones marked for exclusion (`ica.exclude`:
   those `ica.labels_` names eye blink or muscle artifact at or above the
   strip's removal threshold, 0.9) and ICLabel's classes in `ica.labels_`. Without an ICA strip (and no ICA on the input) the log
   says `no ICA decomposition before it`.
7. Python: `run(PIPELINE, raw, ica=r"OUT/ica_demo-ica.fif",
   output="input:ics")` returns the 15 components as an `mne.io.Raw`.

| Item | Value / result | Notes |
|---|---|---|
| H1 Raw back, with the input's annotations | passed | |
| H2 Epochs on the event; the ERP | passed | |
| H3 the same from the file | passed | |
| H4 a GUI pipeline from your own Python | passed | |
| H5 an MNE ICA file starts the ICs rail (GUI) | passed | |
| H6 the MNE ICA writer; `read_ica` reads it | passed | |
| H7 `ica=` in Python | passed | run by Claude (script; the maintainer ran H1–H6) |

## I. Undo and redo (M10.10)

Already run by Claude, offscreen through the real window
(`test_gui_undo.py` and a driven window): Undo and Redo of an added strip, three quick
steps of a value as one undo, the shortcuts, *Open…* starting a new history,
and a live run where an undone setting reaches the pipeline without a
restart. On your machine (Windows), in the GUI:

1. Add a strip, then *↶ Undo*: it goes; *↷ Redo*: it is back. The log says
   `Undo: added …` and `Redo: added …`.
2. Click the high-pass's *Low cut-off* and scroll it a few steps; Ctrl+Z
   once: it is back to its first value. Ctrl+Y (or Ctrl+Shift+Z) redoes it.
3. Type in a text field (e.g. an Epochs strip's events) and press Ctrl+Z
   while it has the focus: the text's own undo, not the chain's.
4. *Start* live, change the high-pass, Ctrl+Z: the pipeline keeps running
   (no "Restarting" in the log) with the old value.

| Item | Value / result | Notes |
|---|---|---|
| I1 Undo and Redo of a strip | passed | |
| I2 a scrolled value undoes in one step; Ctrl+Y | passed | |
| I3 Ctrl+Z in a text field undoes its text | passed | |
| I4 live: a setting undone without a restart | passed | |

## J. Review report (review step 6a)

Already run by Claude, offscreen through the real window, with the data
below: both subjects reviewed, `qc.tsv` with the reviewer, marks, verdicts
and viewed %, and *QC summary* shown. On your machine (Windows):

`uv run python tools/review_demo.py OUT` writes two small BIDS datasets (2
subjects, 60 s, an EOG channel with 4 blinks each) and a pipeline next to
each: `OUT/j/review.json` (blink detector → Review asking about its blinks →
BIDS writer, every subject) and `OUT/k/two_reviewers.json` (§K). First
`uv run meegxer shutdown`.

1. GUI: *Open…* `OUT/j/review.json`; *Run offline*; give your name. Each
   subject's review window opens in turn: answer its blink suggestions (Y or
   N), mark a span or two (drag, then *Reject* or *Annotate*), page through
   to the end, *Save marks and continue*.
2. When the run is done, *QC summary* opens `qc.html`: a row per subject with
   `review1_reviewer`, `review1_marks`, `review1_verdicts`,
   `review1_viewed_percent`. Its report link (or
   `OUT/j/ds/derivatives/meegxer/sub-01/eeg/sub-01_task-rest_desc-clean_report.html`)
   has a *Review (review1)* section with the same, and the marks per label.

| Item | Value / result | Notes |
|---|---|---|
| J1 the report's Review section | passed | run by Claude offscreen (tools/review_demo.py data); how it looks on Windows unchecked |
| J2 the review columns in the QC summary | passed | run by Claude offscreen; qc.html rendered |

## K. Two reviewers (review step 6b)

Already run by Claude, offscreen through the real window, with the data of
§J's tool: A's run reviewed both subjects as reviewer 1 and stopped ("needs a
second reviewer"); B's run asked "Reviewer 2 of 2" (A's marks hidden), then
"Resolve" with only the disagreements; `qc.tsv` has `review1_raters` "A, B"
and `review1_kappa`. On your machine (Windows), one person can play both
reviewers by giving a different name each run:

1. *Open…* `OUT/k/two_reviewers.json`; *Run offline*; name **A**. Each
   subject's window says *Reviewer 1 of 2*: answer the blinks, mark a BAD
   span (e.g. 25–27 s), save. The run then ends with "needs a second
   reviewer: A reviewed as reviewer 1" for each subject (expected: listed as
   failed).
2. *Run offline* again; name **A** again: the same message (A cannot be
   reviewer 2).
3. *Run offline*; name **B**. *Reviewer 2 of 2*: none of A's marks are shown.
   Answer the blinks, mark BAD at 25–27 s (as A) and at 44–46 s (A did not),
   save. Then *Resolve*: the heading gives Cohen's kappa; the suggestions list
   only the 44–46 s span, *From* reviewer 2 (B). Answer it, save.
4. *QC summary*: `review1_raters` "A, B" and `review1_kappa` per label.

| Item | Value / result | Notes |
|---|---|---|
| K1 reviewer 1, then refused as reviewer 2 | passed | run by Claude offscreen, A twice |
| K2 reviewer 2 blind | passed | run by Claude offscreen |
| K3 the resolve window: disagreements, kappa | passed | run by Claude offscreen; raters A, B, kappa BAD 0.65 in qc.tsv |


## L. LDA classifier (M11.3)

Already run by Claude: the CLI run, the GUI offline run (offscreen, through
the real window) and the live run through the daemon. Steps 1 and 2 need
nothing but the repository. Step 3 plays the recording in real time (about
2 minutes).

1. `uv run python tools/lda_demo.py OUT`, then
   `uv run meegxer run OUT/csp_lda.json`: "fitted: csp1, lda1".
   `OUT/mi_lda.set` has the cues (`left`, `right`) and LDA's `pred_left` and
   `pred_right` events, one per CSP window (every 0.25 s).
2. GUI: *Open…* `OUT/csp_lda.json`.
   - The LDA strip draws from the CSP rail and adds to Events.
   - Its *Reads* says *Nearest (CSP)*.
   - *Run offline*, then click the LDA strip's tap. The feature view shows
     two probabilities over time, swapping at the trials.
3. Live: switch the input to live and *Run*. Then:
   - Calibrate *CSP* for 40 s. Once it is ready, calibrate *LDA* for 40 s.
   - Both turn ready, and LDA's tap shows the probabilities as they come.
   - *Stop*: the live `.set` has the cues and, from LDA's calibration on,
     the `pred_` events.
4. Real data (optional, needs the download): the EEGBCI sample's motor
   imagery runs, with *Classes* `T1, T2` on CSP and LDA.

| Item | Value / result | Notes |
|---|---|---|
| L1 CLI run, events in the .set | passed | run by Claude; 583 `pred_left`, 374 `pred_right`; every in-trial window matched its cue (training data) |
| L2 GUI offline: chain, *Reads*, the LDA tap | passed | run by Claude offscreen |
| L3 live: CSP then LDA calibrated, events in the live .set | passed | run by Claude through the daemon and the CLI; both ready, p95 latency 3.6 ms; events in the live .set after the fix |
| L4 real data | | |

## M. Keyboard sink (M11.4)

Already run by Claude on a virtual X display (Xvfb): the live CSP → LDA
pipeline with a keyboard strip. Before arming, nothing was pressed; once
armed, the presses arrived at ≤ 2 per second; *Disarm*, Esc and *Stop* each
stopped them. On your machine (Windows), where the keys really land:

1. `uv run python tools/lda_demo.py OUT`, *Open…* `OUT/csp_lda.json`, add
   *Keyboard (key presses)* at the end (Actions). Its keys default to
   `pred_left=left, pred_right=right`.
2. Live: *Run*, calibrate CSP (40 s) then LDA (40 s). Open Notepad (or any
   text field) beside MeegXer.
3. Select the keyboard strip: *Arm*. The strip turns red ("ARMED · Esc
   disarms"). Notepad's cursor moves left and right, at most twice a second.
4. Press Esc: it disarms (the log says "disarmed by Esc"). Arm again, then
   *Stop*: disarmed. Run again: it starts disarmed.
5. *?* (top right): the About box says MeegXer is not a medical device.

| Item | Value / result | Notes |
|---|---|---|
| M1 nothing pressed until armed | passed | run by Claude (Xvfb, daemon + CLI): 0 presses in 5 s disarmed; arming before *Run* refused |
| M2 presses at most 2/s once armed | passed | run by Claude (Xvfb): 14 presses in 10 s, left and right, never closer than 0.5 s; the rest counted as dropped |
| M3 Esc, *Disarm*, *Stop* disarm | passed | run by Claude (Xvfb): 0 presses after each; the GUI's *Arm* turns the strip red, *Stop* clears it (offscreen) |
| M4 on Windows: keys land in another app | | |
| M5 About box statement | | |

## N. SSVEP → keypress demo (M10.9, M11's exit criterion)

Already run by Claude, on a virtual X display (Xvfb) with real key presses.
- Offline: 156 of 156 detections in the targets' looks were correct.
- Live, armed: 74 presses in 60 s (at most 2/s); all 67 in-look detections
  correct; p95 latency 2.6 ms.
- The GUI armed it offscreen (screenshot checked).

On your machine:

1. `uv run python tools/ssvep_demo.py OUT`. *Open…* `OUT/ssvep_keys.json`.
   It is the *SSVEP BCI (keys)* template reading the demo recording instead
   of an amplifier.
2. Live, *Run*. Tap the SSVEP strip: four scores (8, 10, 12, 15 Hz), the
   target's highest during each 5 s look.
3. Arm the keyboard strip with Notepad beside MeegXer. The arrow keys follow
   the targets: 8 Hz left, 10 right, 12 up, 15 down. At most 2 per second.
4. *New…* → *SSVEP BCI (keys)*: no sample checkbox; the input is LSL (your
   amplifier's stream name).

| Item | Value / result | Notes |
|---|---|---|
| N1 detections offline | passed | run by Claude; 156/156 in-look windows correct |
| N2 live keys follow the targets | passed | run by Claude (Xvfb); 74 presses in 60 s; 67/67 in-look detections |
| N3 the template (New…) | passed | run by Claude (CLI `meegxer new ssvep_bci`; the dialog hides the sample) |
| N4 on Windows, keys in another app | | |
