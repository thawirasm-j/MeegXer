# MeegXer session file (format 1.0, spec 0.2.9, unstable)

A session file is one JSON document holding everything needed to repeat a run: the
pipeline, the exact node versions and where each node came from (built-in, or the
container image of an external plugin), the fitted models (for example an ICA
unmixing matrix), the hashes of the files read and written, and the software it ran
with. It is the unit of reproducibility (`docs/DESIGN.md` §6, acceptance test 4).

MeegXer writes it, but it is meant to be read, edited and written by people too, in
Python, MATLAB (`jsondecode`) or a text editor. It is data only: a session never
contains code. Schema: [`session.schema.json`](session.schema.json).

Recommended file name: `NAME.meegxer.json` (plain JSON; any `.json` file with a
`meegxer_session` key is read as a session).

## Principles

1. **The pipeline is embedded unchanged.** `pipeline` is a pipeline document
   ([`pipeline.schema.json`](pipeline.schema.json)). A session is that pipeline plus
   optional sections, so writing one by hand starts from a pipeline you already have.
2. **Only `meegxer_session` and `pipeline` are required.** A missing section is not
   pinned: no `models` means nodes calibrate as usual, no `inputs` means input files
   are not checked, and so on.
3. **Readable arrays.** Models hold plain JSON numbers (below), not binary blobs.
   Numbers are written so that they read back bit-exact.
4. **Hashes verify, they do not hide.** Model and file entries may carry a SHA-256 so
   a reload can check them. A hand-written entry may leave the hash out.
5. **Fitted models are never used silently on other data** (CLAUDE.md rule 7). A
   loaded model is used only if its `fingerprint` matches the pipeline and input it
   is loaded into; otherwise the node needs refitting.

## Sections

```json
{
  "meegxer_session": "1.0",
  "spec_version": "0.2.9",
  "name": "sub-01 rest, ICA cleaned",
  "description": "Highpass 1 Hz, notch, ICA/ICLabel.",

  "pipeline": {"spec_version": "0.2.9", "nodes": [
    {"id": "input", "node": "meegxer.core.file", "params": {"path": "sub-01_rest.set"}},
    {"id": "hp", "node": "meegxer.nodes.iir_filter",
     "params": {"kind": "highpass", "low": 1.0}},
    {"id": "ica", "node": "meegxer.nodes.ica", "params": {"threshold": 0.8}},
    {"id": "asr", "node": "lab.asr", "params": {"cutoff": 20.0}}
  ]},

  "nodes": {
    "input": {"node": "meegxer.core.file", "version": "0.1.0"},
    "hp":    {"node": "meegxer.nodes.iir_filter", "version": "0.1.0"},
    "ica":   {"node": "meegxer.nodes.ica", "version": "0.1.0"},
    "asr":   {"node": "lab.asr", "version": "1.2.0", "provenance": {
               "image": "ghcr.io/lab/asr:1.2.0", "image_id": "sha256:…",
               "repo_digests": ["ghcr.io/lab/asr@sha256:…"]}}
  },

  "models": {
    "ica": {
      "fitted_on": {"channels": ["Fp1", "Fp2", "…"], "sfreq": 500.0, "bads": [],
                    "types": ["eeg", "eeg", "…"], "positions": [[-0.03, 0.08, 0.02], "…"]},
      "fingerprint": "…",
      "content_hash": "sha256:…",
      "data": {
        "unmixing": {"__array__": "float64", "shape": [19, 19],
                     "data": [[0.12, -0.03, "…"], "…"]},
        "excluded": [0, 3, 7]
      }
    }
  },

  "inputs": {"input": [{"path": "sub-01_rest.set", "sha256": "sha256:…"},
                       {"path": "sub-01_rest.fdt", "sha256": "sha256:…"}]},
  "outputs": [{"path": "sub-01_rest_clean.set", "sha256": "sha256:…"}],

  "environment": {"created": "2026-10-16T09:12:00+07:00", "meegxer": "0.1.0a5",
                  "python": "3.14.0", "platform": "Windows-11-10.0.26100",
                  "packages": {"numpy": "…", "scipy": "…", "mne": "…"}}
}
```

| Section | Holds |
|---|---|
| `meegxer_session` | Format version, `major.minor`. Readers accept their own major version. |
| `pipeline` | The pipeline document, unchanged. |
| `nodes` | Per node id: manifest `node` name, `version`, and `provenance` for external plugins: `image`, `image_id` (the local image ID the host recorded at `meegxer-plugin add`), `repo_digests` (registry digests, when the engine reports them), or, for a local plugin (unsandboxed: a custom script or a local command), `command`, `unsandboxed`, `sha256` (one hash over its code files, in name order) and `script` (the script's file name). A review strip or a Review epochs strip (`docs/REVIEW.md`) also has `review`: `reviewer` (the name given when the run started), `reviewed` (when, ISO 8601 UTC) and `upstream` (the fingerprint of the input the review was made on); the marks stay in the review's sidecar. |
| `models` | Per node id, for nodes that need calibration: `data` (the model exactly as the node returned it), `fitted_on` (the channels it was fitted on), `fingerprint`, `content_hash`. External plugins' models use the same form: the host owns them (plugin protocol), so they are stored like any other. |
| `inputs` | Per source node id: the files read, with hashes. |
| `outputs` | The files the run wrote, with hashes. |
| `environment` | When and with what it was made: MeegXer, Python, platform and library versions. |

### Arrays and numbers in `models`

A model's `data` is a JSON tree of numbers, strings, booleans, lists, objects and
arrays. An array is

```json
{"__array__": "float64", "shape": [2, 3], "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
```

with `__array__` one of the NumPy dtype names (`float64`, `float32`, `int64`,
`bool`, …). Complex arrays have `"real"` and `"imag"` instead of `"data"`. Inside an
array, non-finite values are the strings `"NaN"`, `"Infinity"` and `"-Infinity"`;
elsewhere a non-finite float is `{"__float__": "NaN"}`. MATLAB reads `data` with
`jsondecode` (nested lists become matrices); NumPy with
`np.array(x["data"], dtype=x["__array__"])`.

`content_hash` is computed over the decoded tree (the same hash the core keeps for
its models), so an edited model without an updated hash is reported, not used. Leave
`content_hash` out of a hand-written model.

### Paths

Paths in `pipeline` parameters named `path`, `root` or `ica`, in a source's `recording`
(montage and events files), and in `inputs`/`outputs` are relative to the session
file's folder unless absolute. MeegXer writes paths inside that folder as relative
paths, so a session folder can be moved or zipped as a whole.

### Sharing

A session can hold identifying names: paths with a user's folders, and file names
that name a subject (`JaneDoe_2024-05-06.edf`). `meegxer session share IN OUT`
(or *Share session…* in the GUI, or `meegxer session save --share-safe`) writes a
copy that names no file or folder:

| Field | Becomes |
|---|---|
| The source's `path` / `root` | `<input>.ext` / `<input-root>` |
| The File input's `ica` (an MNE ICA file) | `""` (none: it belongs to its recording) |
| A writer's `path` | `<output-N>.ext` (the same file keeps the same N) |
| A BIDS writer's `root` | `""` (its default: the raw dataset's `derivatives/meegxer`) |
| `recording.montage` (a file), `recording.events.file` | `<montage>.ext`, `<events>.ext` |
| `inputs` / `outputs` paths | the same placeholders; `sha256` kept |
| A model's `fitted_on.stream_id` (a file source's file name) | `<input>.ext` (reloading compares the `fingerprint`, a one-way hash) |
| A local plugin's `command` | file names only (no folders) |
| A review strip's or Review epochs strip's `file` | `""` (its default: next to each recording) |
| A review's `reviewer` (`nodes`) | left out (it names a person) |

A placeholder is `<word>` or `<word-N>` with the original extension; it is not a
path, so reading and writing leave it as it is. Kept: hashes (they confirm a file
someone already has and reveal nothing else), models, node versions, the
environment, and `name` and `description`, which are the user's own words: the
GUI and CLI show them and offer to remove them (`--drop-text`).

Rerunning a shared session needs the recording: `meegxer run OUT -i FILE -o
RESULT` (or `--bids ROOT`). Without `-i`, the run stops and says so; writers whose
path is a placeholder write nothing (`-o` gives the output). A placeholder montage
or events file is left out on loading, with a warning.

## Reloading

| Item | If it differs from what is installed or found |
|---|---|
| `meegxer_session` major version, `spec_version` | Refused. |
| A node is not installed | Refused, naming the node (for a plugin: `meegxer-plugin add IMAGE` with the recorded image). |
| A built-in recorded under its pre-0.2.1 name (`meegxer.sterile.*`) | Loads as the same node under its current name (`meegxer.nodes.*`); models are reused. |
| Node `version`, plugin `image_id`, local plugin `sha256` (the script changed) | Refused by default. With the override (`--allow-version-mismatch`), that node's model is dropped, so it needs refitting. |
| Model `content_hash` | Refused: the model was changed after it was saved. |
| Model `fingerprint` | The model is kept but not used: the node needs refitting (offline runs refit it on their input, live runs wait for calibration). |
| Input file `sha256` | Warning; refused with `--strict`. |
| Library versions | Warning only. They are recorded because they can change results in the last bits. |

A model fitted on a file stays valid when the file moves: a file source is
identified by its content (file names and bytes), not by its path. A model applied to
a different input (for example `-i` another subject) needs refitting, by design.

## Acceptance test 4

Save a session after an offline run (its `outputs` hold the written files' hashes),
reload it on another machine and run it again: the output files are byte-identical.
The file writers are deterministic for that reason (no write-time timestamps).
