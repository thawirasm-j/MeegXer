# Manual review (design and status)

A **Review** strip (`meegxer.core.review`, *Event detection*) lets a person mark
a recording by hand: bad segments (`BAD_muscle`, `BAD_eyes`), states
(`eyes_closed`, `drowsy`), anything with a label and a span. The marks go on
the Events rail, where later strips see them like the recording's own events.

Built (PR 1, continuous marks):

- **The strip** passes the signal on unchanged and puts the recording's marks on
  the Events rail, at the strip's rate (rule 8). Offline only: a person reviews
  a whole recording. Parameters:
  - `labels`: the labels offered (default `BAD`). The input's event types are
    offered too, and a new label can be typed in.
  - `file` (Advanced): where the marks are saved. If you leave it empty, they
    go next to each recording (see below).
- **One sidecar per recording**, so the pipeline stays a template for every
  subject:
  - Where:
    - An input in a BIDS dataset: in its BIDS derivatives, as
      `derivatives/meegxer/sub-01/eeg/sub-01_task-rest_desc-<strip>_events.tsv`.
    - Any other file input: next to it, as `<file>_desc-<strip>_events.tsv`.
    - An input with no file (synthetic, LSL): set `file`.
  - What: a BIDS events file (`onset`, `duration` in seconds from the
    recording's start, `trial_type` the label), which MNE and EEGLAB read too.
    Its JSON sidecar records under `MeegXer` the strip, when it was reviewed,
    and `upstream`: the fingerprint of the strip's input.
- **A batch run at the strip:**
  - If the sidecar was made on the same input, it uses the marks.
  - If there is no sidecar yet, it asks for a review.
  - If something upstream changed since (a filter setting, the recording's
    channels), it asks again and offers the old marks to confirm or change.
    They are never reused silently (rule 7).
- **GUI offline run: it pauses.**
  - The status `state` is `review`. Batch runs go node by node, so every tap
    before the strip is filled; the main window keeps showing them, with the
    probes chosen as Main and aux, and Features too.
  - The review window opens:
    1. Drag the band on the signal view over a span.
    2. Pick or type a label, then *Mark selection*. Marks show as shaded spans,
       and double-clicking a row shows that mark.
    3. *Save marks and continue* saves the sidecar and the run goes on. *Cancel
       run*, or closing the window, stops the run without saving.
  - In an all-subjects BIDS run, it pauses at every recording not yet
    reviewed.
  - The core never waits for the GUI (rule 2). Only the offline job's own
    thread waits on the answer (`submit_review` / `cancel_review`); the live
    pipeline, the status and the views carry on.
- **CLI (headless):** a recording that is not reviewed stops at the strip, so
  nothing past it is processed or written. The CLI makes sure this is not
  missed:
  - `meegxer run` prints a warning and exits with code 2.
  - `meegxer run … --bids` goes on with the other recordings, flags each one
    `NOT REVIEWED`, ends with a warning listing them, and exits with code 2
    (1 if a recording failed).

Next:

1. **Epochs side rail** (M10): epochs cut around events, as a rail.
2. **Epoch review:** the same strip pulls epochs from that rail. Mark each one
   good or rejected, or give it a label (new or an existing event type).
   Rejected epochs go on the rail as marks.
3. **Live marks:** hotkeys add marks to the Events rail while recording.

Automated rejection needs no new machinery. A strip or a custom script emits
events labelled `BAD_…` (amplitude, flatline, …); later strips choose which
labels they honour, by a keyword parameter.
