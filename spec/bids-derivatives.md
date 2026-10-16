# BIDS-derivatives output (spec 0.2.10, unstable)

How MeegXer writes processed recordings into a BIDS **derivatives** dataset
([BIDS derivatives](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html)),
so BIDS tools (EEGLAB's BIDS plugin, MNE-BIDS, the BIDS validator) find them, and
how every output records how it was made. The EEG-derivatives extension of BIDS
(BEP021) is still a draft; this layout follows the general derivatives rules with
the `desc-` entity, as BEP021 does.

## Nodes

| Node | Input | Writes |
|---|---|---|
| `meegxer.core.bids_writer` | Signal | the processed recording and its sidecars |
| `meegxer.core.bids_features` | Features | one feature table and its sidecar |

Both are offline only (they need the whole recording) and take these parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `root` | `<raw dataset>/derivatives/meegxer` | the derivatives dataset folder; required when the input is not inside a BIDS dataset |
| `desc` | `clean` | the `desc-` label of the outputs (letters and digits) |
| `subject`, `session`, `task`, `run` | from the input | override the entities (for inputs whose names carry none) |
| `overwrite` | `false` | replace this recording's files if they exist |

Entities come from the input: the parameters of a `meegxer.core.bids` source, or
the BIDS-style file name of a `meegxer.core.file` source
(`sub-01_ses-a_task-rest_run-1_eeg.set`). `subject` and `task` are required.

## Layout

For `sub-01_task-rest_eeg.set` and `desc` `clean`:

```
<root>/
├── dataset_description.json
├── .bidsignore
└── sub-01/eeg/
    ├── sub-01_task-rest_desc-clean_eeg.set
    ├── sub-01_task-rest_desc-clean_eeg.json
    ├── sub-01_task-rest_desc-clean_channels.tsv
    ├── sub-01_task-rest_desc-clean_events.tsv          (when the input has events)
    ├── sub-01_task-rest_desc-clean_electrodes.tsv      (when positions are known)
    ├── sub-01_task-rest_desc-clean_coordsystem.json    (with electrodes.tsv)
    ├── sub-01_task-rest_desc-clean_<node>_features.tsv (bids_features, per node)
    ├── sub-01_task-rest_desc-clean_<node>_features.json
    └── sub-01_task-rest_desc-clean_meegxer.json        (the run's session file)
```

With `ses-` the files go under `sub-01/ses-a/eeg/` and every name carries `ses-a`.

- **Signal:** EEGLAB `.set`, float32, in volts, written deterministically (the same
  data give the same bytes). EDF is not used: its 16-bit samples would lose detail
  in cleaned data.
- **`_eeg.json`:** the raw recording's sidecar when there is one (task name,
  manufacturer, power line frequency, …), with `SamplingFrequency`, the channel
  counts per type and `RecordingDuration` of the output, `Sources` (the raw file
  as a BIDS URI, `bids:raw:sub-01/eeg/sub-01_task-rest_eeg.set`), `Description`
  (the chain, node by node) and `MeegXerSession` (the session file's name).
- **`_channels.tsv`:** `name`, `type` (BIDS upper case: `EEG`, `EOG`, `ECG`, `EMG`,
  `MISC`, `RESP`, `TRIG`), `units` (`V`), `status` (`good`/`bad`), after the edits
  of the Recording window.
- **`_events.tsv`:** `onset`, `duration` (seconds), `trial_type` (the label after
  renaming and dropping).
- **`_electrodes.tsv`:** `name`, `x`, `y`, `z` in metres (`n/a` where unknown);
  **`_coordsystem.json`:** `EEGCoordinateSystem` `CapTrak`, `EEGCoordinateUnits`
  `m` (MNE head coordinates).
- **`_features.tsv`:** one row per window: `window_start`, `window_stop` (sample
  indices), `onset` (seconds), then one column per feature value named by its
  coordinates (`Fz|alpha`). Its sidecar describes every column, the node and its
  parameters.
- **`_meegxer.json`:** the session of the run (`session.md`): pipeline, node
  versions and plugin images, fitted models, input and output hashes. Listed in
  `.bidsignore`, so the validator skips it.

## `dataset_description.json`

Created on the first write and updated after:

```json
{
  "Name": "MeegXer derivatives",
  "BIDSVersion": "1.10.0",
  "DatasetType": "derivative",
  "GeneratedBy": [{"Name": "MeegXer", "Version": "0.1.0a6",
                   "CodeURL": "https://github.com/thawirasm-j/MeegXer"}],
  "SourceDatasets": [{"URL": "bids:raw:"}],
  "DatasetLinks": {"raw": "../.."}
}
```

`DatasetLinks.raw` is the raw dataset relative to the derivatives folder (absent
when the input is not in a BIDS dataset). Existing fields are kept; MeegXer's
`GeneratedBy` entry is replaced with the current version.

## Many recordings

`meegxer run PIPELINE --bids RAW_ROOT [--subjects 01 02 …] [--task T] [--session S]
[--run R]` runs the pipeline on every matching EEG recording, one after another,
each with a fresh graph (models are fitted per recording). A recording whose
derivative already exists is skipped unless `--overwrite`, so rerunning the same
command continues an interrupted run. A failing recording is reported and the
others still run; the command exits with status 1 if any failed.
