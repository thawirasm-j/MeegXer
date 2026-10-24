# Manual review (design and status)

A **Review** strip (`meegxer.core.review`, *Event detection*) lets a person mark
a recording by hand: bad segments (`BAD_muscle`, `BAD_eyes`), states
(`eyes_closed`, `drowsy`), anything with a label and a span. The marks go on
the Events rail, where later strips see them like the recording's own events.

## Design v2 (agreed 2026-10-24; not built yet)

Who reviews, and what helps them:

- **Researchers** (ERP and resting-state studies, many subjects) want clean
  data, the same criteria across subjects and raters, speed, and a record of
  who decided what.
- **Clinicians** read the recording, label findings (spikes, seizures,
  slowing, state changes), tell artifact from abnormality, and confirm or
  overrule a detector's proposals.
- **Both** need the same tools:
  - clinical display conventions: montages, display filters, sensitivity,
    page length
  - keyboard-first marking with a label palette, and undo
  - an overview of the whole recording, and the ability to stop and resume
  - their name on the result

### Principles

1. **A review produces events, and channel status; nothing else.** Marks are
   events with a label, onset and duration. Marking a channel bad sets the
   stream's bad channels, so interpolation and reference strips act on it.
2. **Cutting and epoching are separate strips**, and both pick events by
   keyword:
   - **Reject segments** (new) cuts out spans whose label starts with its
     `reject` words (default `BAD`), with boundaries and a restart per piece
     (the cut machinery below).
   - **Epochs** (built) cuts epochs around chosen events.

   The Review strip's `reject` parameter moves to the Reject segments strip.
   The window's *Reject* key still labels a span `BAD…`; the strip downstream
   cuts it.
3. **The pipeline is what the researcher designed for the reviewer.**
   - The Review strip's settings fix what the reviewer may edit and how the
     traces look.
   - Settings (`docs/SETTINGS.md`) give the defaults where the pipeline does
     not fix them: units (µV/mm and mm/s, or µV and seconds per page) and
     montage.
   - The default montage is monopolar against the average reference; bipolar
     and Laplacian montages are chosen in Settings or in the strip.
4. **The reviewer is asked for their name once per run**, when a run reaches
   its first review. It is stored with every review (sidecar) and in the
   session file.
5. **Headless runs stop at an unreviewed strip** (as built). A pending
   suggestion counts as unreviewed.
6. **Detectors:** only well-known, published methods are built in. They emit
   events like any strip:
   - amplitude (peak-to-peak) and flat-signal spans: MNE `annotate_amplitude`
   - muscle: MNE `annotate_muscle_zscore`
   - blinks from EOG: MNE `find_eog_events`

   Anything else (spike, seizure, sleep-stage detectors) comes as a plugin.
   No detector of MeegXer's own invention.

### Two review modes

1. **Review at import**: no priors, everything editable.
   - *Where:* the Input strip's Recording window gets a *Review* action. It
     opens the review window on the raw recording, before any pipeline.
   - *Editable:*
     - channel names, types and bad status (as the Recording window does)
     - events: add, move, rename, delete
     - labelled spans
     - crop at the start and end
   - *Output:* a per-recording sidecar, independent of any pipeline. Every
     pipeline sees it as the input's own events and bad channels. The raw
     file is never touched.
2. **Review in the pipeline**: the Review strip at a point in the chain.
   - *Settings* (what the researcher designed):
     - **edit**: what the reviewer may change (spans, events, channels; later
       epochs and ICs)
     - **priors**: none, or the event labels to propose as suggestions (a
       detector strip upstream)
     - **decision**: *free* (suggestions are a starting point) or *verify*
       (yes/no per suggestion only)
     - **display presets**: montage, display filters, sensitivity, page
       length; empty means the reviewer's Settings
   - *Verdicts:* each suggestion's verdict (accepted, rejected or edited) is
     kept with its origin. Only accepted ones go on as events; rejected ones
     stay recorded, so a detector's misses can be measured.
   - *After an upstream change:* only new or changed suggestions need a
     verdict.

### One window, tools by mode

```
┌ toolbar: montage · display filters · sensitivity · page length · channels ┐
│ traces (stacked)                                    │ side panel (tabs):   │
│  shaded marks, dashed suggestions, events, cursor   │  Marks · Suggestions │
│                                                     │  Channels · Spectrum │
├ overview bar: coverage · marks · suggestion density ┤  / topomap           │
└ palette: 1 BAD  2 eyes  3 spike … · Y/N (verify) · undo · save/continue   ┘
```

- **Import mode:** the Channels and events tools are editable, and there is
  no Suggestions tab.
- **Verify mode:** the Suggestions queue (Y accept, N reject, E edit, next)
  comes first, and free marking is off.

### Order of work (one PR each)

1. **Events only, and the Reject segments strip.** The cutting moves out of
   the Review strip into its own strip. The reviewer's name is asked once per
   run.
2. **Display essentials:**
   - montage (average reference by default; bipolar and Laplacian presets)
   - display filters
   - sensitivity and page length, in Settings units
   - label palette (keys 1–9) and undo
   - bad-channel marking
   - display presets in the Review strip
3. **Review at import:** the Recording window's *Review* action and the
   per-recording sidecar.
4. **Suggestions:**
   - detector strips (amplitude/flat, muscle, blinks from EOG)
   - priors and the yes/no queue in the Review strip
   - verdict provenance
5. **Epoch review** in the same window (accept, reject or label each epoch),
   and the ERP view.
6. **Later:** a second reviewer with compare and resolve; a review report.

## Built so far (v1)

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

Built (PR 3, the stacked review window): the review window is the stacked
view (the probes chosen as Main and aux, every channel by default, its own
span, spacing and panner) with the review tools below:

- **Coverage bar:** the whole recording. Parts already shown are green,
  rejected spans red, annotations purple, and the shown page is outlined.
  Click to go there. *Next unreviewed* (N) jumps to the first part never
  shown.
- **Tools:** drag the band on the traces over a span, then:
  - *Annotate* (A) labels it with the chosen label (the strip's, an input
    event type, or a new one typed in).
  - *Reject* (R) marks it to be cut out, with a label starting with a reject
    word (the chosen label if it starts with one, else the first word,
    `BAD`).
  - Pages move with ←/→ or PgUp/PgDn.
- **Marks:** shaded on the traces with their label at the top, and listed
  below. Double-click a row to show that mark; Delete removes it. The
  progress line reads `Reviewed 67% · 2 marks · 1.2 s rejected`.
- **Save marks and continue** asks first if part of the recording was never
  shown. The spans you looked at are saved with the marks (sidecar JSON,
  `viewed`) and come back as green when that recording is reviewed again.
  Control API 0.2.9: `submit_review` takes `viewed`, and the request carries
  `reject` and `viewed`.

Built: the **Epochs** strip writes the epochs side rail (`docs/RAILS.md`).

Next: the order of work in *Design v2* above. Live marks (hotkeys that add
marks to the Events rail while recording) come after it.

Automated rejection needs no new machinery. A strip or a custom script emits
events labelled `BAD_…` (amplitude, flatline, …); later strips choose which
labels they honour, by a keyword parameter.
