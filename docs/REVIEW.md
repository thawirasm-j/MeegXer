# Manual review (design and status)

A **Review** strip (`meegxer.core.review`, *Event detection*) lets a person mark
a recording by hand: bad segments (`BAD_muscle`, `BAD_eyes`), states
(`eyes_closed`, `drowsy`), anything with a label and a span. The marks go on
the Events rail, where later strips see them like the recording's own events.

## Design v2 (agreed 2026-10-24; steps 1 and 2 built)

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
   run. *Built* (see *Built: step 1* below).
2. **Display essentials:**
   - montage (average reference by default; bipolar and Laplacian presets)
   - display filters
   - sensitivity and page length, in Settings units
   - label palette (keys 1–9) and undo
   - bad-channel marking
   - display presets in the Review strip

   *Built* (see *Built: step 2* below).
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

Built (PR 2, cuts; since step 1 of Design v2 a Reject segments strip cuts,
not the Review strip):

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

Built (step 1 of Design v2):

- **The Review strip produces events only.** Its `reject` setting now names the
  labels the window's *Reject* key gives a span; nothing is cut at the strip.
- **Reject segments** (`meegxer.core.reject_segments`, *Artifacts*, offline)
  does the cutting described in *PR 2* above. It reads the Events rail, so it
  cuts every span whose label starts with one of its `reject` words (default
  `BAD`): the recording's own events, a review's marks, a detector's. Its
  `boundary` events carry its id as their origin. A pipeline made with 0.3.1b3
  needs one added after its Review strip to keep cutting.
- **The reviewer's name** is asked once when a GUI offline run with a Review
  strip starts (remembered for next time in `state.json`; *Cancel* does not
  start the run). It goes to `run_offline(reviewer)` (control API 0.2.11) and
  is saved with each review made in that run: in the sidecar JSON
  (`MeegXer.reviewer`) and in the session (`nodes.<id>.review`: `reviewer`,
  `reviewed`, `upstream`). A shared session leaves the name out.

Built (step 2 of Design v2, display essentials):

- **The display bar** above the traces: montage, display filters (high-pass,
  low-pass, notch; 0 is off), the scale, and *Bad channels*.
  - Montages (`meegxer.core.montage`): *as recorded*; *average*, each EEG
    channel minus the mean of the good EEG channels (the default); the
    *longitudinal* ("double banana") and *transverse bipolar* montages, from
    the 10-20 pairs the recording has (`T3`/`T4`/`T5`/`T6` count as
    `T7`/`T8`/`P7`/`P8`); *Laplacian*, each EEG channel minus the mean of its
    four nearest good neighbours, by electrode position. Other channels (EOG,
    ECG) are shown as recorded.
  - The core computes the page through the montage and filters from the
    probe's raw samples (`offline_window` with `display`, control API 0.2.12),
    reading a little either side so the zero-phase filters settle, and each
    stretch between cuts on its own. A page over 120 s is the plain envelope.
  - The scale: clinical by default, a sensitivity in µV/mm and a speed in
    mm/s on the screen's real size (from its physical DPI), so 7 µV/mm and
    30 mm/s look as they do on paper; or (Settings) a channel spacing in µV
    and seconds per page.
- **Settings, Review page:** the units, sensitivity, speed, spacing, page,
  montage and filters a reviewer starts with (`docs/SETTINGS.md`).
- **Display presets** in the Review strip (Advanced): `montage`, `highpass`,
  `lowpass`, `notch` (-1: the reviewer's Settings), `sensitivity`, `page` (0:
  the reviewer's). What the strip sets is fixed in the window (greyed out).
- **Bad channels:** *Bad channels (n)* lists the recording's channels to tick.
  A bad channel is grey and dashed, left out of the average and the
  Laplacian at once, saved with the review (sidecar JSON, `MeegXer.bads`),
  and added to the stream's bad channels after the strip; the strip's output
  fingerprint includes them, so later strips refit when they change.
- **Palette and undo:** keys 1 to 9 mark the band with the first nine labels
  (listed under the tools); *Undo* (Ctrl+Z) takes back the last change to
  the marks or bad channels.

Built (after the first look at step 2):

- **Spans are drawn, not placed.** There is no band until you drag on the
  traces from the start of a span to its end (either way); a click picks an
  instant. Each edge has a tab at the top to drag it. Dragging an edge (or
  the drag itself) to the border of the page scrolls on, so a span can be
  longer than the page; then a tab at the border shows where it ends
  (`14.0 s ▶`), and a click there goes to that end.
- **Marks are edited in place.** Marking leaves the span on the new mark, to
  adjust its edges. A click on a mark (or double-clicking its row) selects
  it: drag its edges, press a label key or Annotate/Reject to relabel it,
  Delete to remove it. Escape lets go of the span.
- **Channel names are pills** in the margin, outlined in the channel's
  colour: a click marks the channel bad (dashed grey, struck through) or good
  again; a bipolar pair asks which electrode. A right-click opens the
  channel's menu: bad or good per electrode, hide the channel, show all
  channels, mark all channels good. *Bad channels (n)* stays as a list.

Built (review points, and the display as a starting point):

- **The display is never locked.** The Review strip's display settings
  (Advanced: montage, high-pass, low-pass, notch, sensitivity, page length)
  are what the window opens with; each is *Reviewer's choice* by default
  (the reviewer's Settings), and the reviewer can change all of them.
- **Review points.** The strip's `points` names event labels to step through,
  e.g. those of a beta detector upstream (120 candidates for a physician to
  confirm or reject):
  - ◀/▶ and ←/→ go to the previous or next point (PgUp/PgDn still turn
    pages). The point is centred and selected; if it was answered, its
    answer is selected instead.
  - The reviewer labels it (keys 1 to 9, Annotate) and retimes it by
    dragging its edges. The mark keeps the point it answers (`point` on the
    wire, a `review_point` column in the sidecar), so a detector's points can
    be compared with the reviewer's answers.
  - *Next point after marking* (on by default) moves on after each answer.
  - Points show as dashed outlines (green once answered), as ticks on the
    coverage bar, and in the counter `Point 12/120 · 45 answered`. Saving
    asks first if some points have no answer (with points, the coverage of
    the whole recording is not asked about).

Fixed: the marks reach the output files and the views. Every file writer
(`.set`, EDF, BIDS) saves the events on the Events rail where it sits (the
input's, a review's marks, a detector's), shifted onto the processed
recording's time; the offline views mark the run's events (control API
`offline_events`).

Next: the order of work in *Design v2* above. Live marks (hotkeys that add
marks to the Events rail while recording) come after it.

Automated rejection needs no new machinery. A strip or a custom script emits
events labelled `BAD_…` (amplitude, flatline, …); later strips choose which
labels they honour, by a keyword parameter.
