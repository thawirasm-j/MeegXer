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

Built (PR 2, cuts):

- **Rejected spans are cut out.** A mark whose label starts with one of the
  strip's `reject` words (default `BAD`; empty: nothing is cut) is removed
  from the signal. Overlapping spans are merged first.
- **Samples keep their index (rule 8).** After a cut the stream simply has a
  gap: taps before and after the strip still line up, and every event keeps
  its index. A `boundary` event (EEGLAB's name) sits at the first sample after
  each cut; its duration is the number of samples cut.
- **Nothing leaks across a cut.** At each gap the batch scheduler restarts
  every strip after the review strip (`Stage.restart`):
  - A delaying node hands over its tail (`flush`).
  - Streaming state is dropped, and the next piece starts fresh, exactly as
    if it were a recording of its own.
  - Offline nodes (zero-phase filters) run on each piece separately.
  - Fitted nodes (ICA, CSP) fit on the pieces joined, with the events in them.
  This needs no change in any node, plugin or custom script.
- **Not yet:** a strip that reads rails cannot fit on a cut stream (it says
  so), and a cut output is not cached.
- **Writers** save the pieces joined:
  - `.set` and EDF get a `boundary` event at each cut (also after a crash
    recovery).
  - The BIDS writer's `events.tsv` gives onsets on the processed recording's
    time: events inside a cut are left out, and a `boundary` row marks each
    cut.
- **Offline views** show a cut as a gap.

Next:

1. **Stacked review window:** the stacked view with tools below it to
   annotate and reject spans, and the reviewed and not-yet-reviewed parts
   shown in different colours.
2. **Epochs side rail** (M10): epochs cut around events, as a rail.
3. **Epoch review:** the same strip pulls epochs from that rail. Mark each one
   good or rejected, or give it a label (new or an existing event type).
   A rejected epoch drops from the Epochs rail only; the continuous
   signal stays whole.
4. **Live marks:** hotkeys add marks to the Events rail while recording.

Automated rejection needs no new machinery. A strip or a custom script emits
events labelled `BAD_…` (amplitude, flatline, …); later strips choose which
labels they honour, by a keyword parameter.
