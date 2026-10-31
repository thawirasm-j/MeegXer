# De-identification

What MeegXer writes about the people recorded, and how to share results without
names. This is a checklist, not legal advice: what counts as identifying depends on
your consent forms, IRB/ethics approval and the law that applies (HIPAA, GDPR).

## What MeegXer writes

| Output | Identifying data | What to do |
|---|---|---|
| EDF / BDF (`edf_writer`) | None from the input: patient and recording fields are written as `X`, the start date as EDF's default (1985-01-01). | Nothing. (The recording date is not kept either.) |
| EEGLAB `.set` (`set_writer`) | None from the input: no subject information or recording date; the MAT-file header names MeegXer only. An ICA decomposition, when one unmixes the data, holds only its matrices and ICLabel classes. | Nothing. |
| BIDS derivatives | `sub-`/`ses-` labels and the raw dataset's sidecar fields and credits (`dataset_description.json`), as in the raw dataset. | Share derivatives of a raw dataset that is already de-identified. |
| Feature tables (`.tsv`) | None. | Nothing. |
| Session files (`.meegxer.json`) | File and folder names (they often name the subject), and the name and description you typed. | Share a **share-safe copy** (below). |
| Channel names, event labels you typed | Whatever you typed. | Avoid names in labels. |

## Share-safe sessions

```
meegxer session share study/sub-01.meegxer.json shared.meegxer.json
```

or *Share session…* in the GUI (`meegxer session save FILE --share-safe` from a
running daemon). The copy names no file or folder: paths become placeholders such
as `<input>.set` and `<output-1>.edf`, keeping only the extension. It keeps what
makes the run reproducible: the pipeline, node versions, fitted models (e.g. the ICA
unmixing matrix) and file hashes. Hashes reveal nothing about the content; they only
let someone who already has the same file confirm it.

The session's name and description are your own words, so MeegXer keeps them but
shows them first and offers to remove them (`--drop-text` removes them without
asking).

The recipient reruns it on their recording:

```
meegxer run shared.meegxer.json -i their_recording.set -o result.set
```

A model fitted on your recording is not used on theirs (its fingerprint differs):
it is refitted on their data, by design (rule 7). Details: `spec/session.md`,
*Sharing*.

## What MeegXer cannot clean

- Names inside the recording you load (e.g. an EDF patient field) never reach
  MeegXer's outputs, but the **input file itself** still has them: de-identify
  recordings before sharing them (e.g. `mne.io.Raw.anonymize()`, or the BIDS
  conversion's de-identification options).
- File names you choose for outputs: pick neutral ones (`sub-01_clean.set`).
- Free text you type into labels, events or descriptions.
- Electrode positions digitized per subject (head shape) can be identifying in
  rare cases; standard montages are not.
