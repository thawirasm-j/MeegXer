# Manual review (design and status)

A **Review** strip (`meegxer.core.review`, *Event detection*) lets a person mark
a recording by hand: bad segments (`BAD_muscle`, `BAD_eyes`), states
(`eyes_closed`, `drowsy`), anything with a label and a span. The marks go on
the Events rail, where later strips see them like the recording's own events.

## Design v2 (agreed 2026-10-24; steps 1 to 5 built)

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
   per-recording sidecar. *Built* (see *Built: step 3* below).
4. **Suggestions**, in two PRs:
   - 4a. detector strips (amplitude/flat, muscle, blinks from EOG). *Built*
     (see *Built: step 4a* below).
   - 4b. priors and the yes/no queue in the Review strip, and verdict
     provenance. *Built* (see *Built: step 4b* below).
5. **Epoch review** (accept, reject or label each epoch), and the ERP view.
   *Built* (see *Built: step 5* below), as its own strip and window.
6. **Later:** a second reviewer with compare and resolve; a review report.

### Step 4b: decisions (2026-10-30; built)

- **Priors are the Review strip's `points`** (the labels to propose), with a
  new `decision` setting: *free* (suggestions are a starting point) or
  *verify* (the Y/N/E queue).
- **A Review strip with priors takes those labels over.** Later strips do not
  see the prior-labelled events of the strips upstream of it; they see the
  Review strip's verdicts instead: accepted and edited suggestions (as
  edited), never rejected ones. This is a new Events rail rule
  (`docs/RAILS.md`, written with 4b).
- **Verdicts are columns of the marks file** (the strip's
  `_desc-<strip>_events.tsv`): `suggestion_origin` (the detector's strip
  id), `suggestion_onset` and `suggestion_duration` (as proposed), and
  `verdict` (`accepted`, `rejected` or `edited`). Rejected rows stay in the
  file, so a detector's misses can be measured, but are not emitted.
- **Proposed bad channels** (a detector's `bad_channel` events) are asked
  about in the queue like the spans; only confirmed ones become bad.
- **Headless runs stop** while a suggestion has no verdict (principle 5).

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

Built (step 3 of Design v2, review at import):

- **Recording window → *Review…*** (file and BIDS inputs) opens the review
  window on the raw recording, before any pipeline: an offline run of the
  input alone (without the pipeline's recording settings) fills the traces.
- **What it edits:** the recording's events (they open as marks: move their
  edges, relabel them with a key or Annotate, delete them, add new ones),
  labelled spans, bad channels (the channel pills), and a **crop**: *Keep
  from … to … s*, from the selection or typed in; the parts left out are
  shaded. Channel names and types stay in the Recording window's tabs.
- **Where it goes:** one review per recording, independent of any pipeline
  (`meegxer.core.import_review`): next to the file as
  `<file>_desc-import_events.tsv`, or in its BIDS derivatives
  (`…_desc-import_events.tsv`). The events file is BIDS (onset and duration
  in seconds from the raw file's start); its JSON sidecar records under
  `MeegXer` the bad channels, the crop, who reviewed and when, and the spans
  looked at. The raw file is never touched. Opening *Review…* again starts
  from the saved review.
- **How pipelines use it:** file and BIDS inputs apply it when they load
  (their *Use the review made at import* setting, Advanced, on by default):
  the recording is cropped (its first sample is the crop's start, so every
  index and time downstream counts from there), its events are the reviewed
  ones (shifted onto the cropped time; those outside are left out), and its
  bad channels include the reviewed ones. The Recording window's Events tab
  says *Events of the review at import*. The review is part of the input's
  content id: changing it marks fitted models and later reviews for a refit
  or a new look (rule 7). Control API 0.2.14: `import_review_open`,
  `save_import_review`.

Built (step 5 of Design v2, epoch review):

- **A Review epochs strip** (`meegxer.core.epoch_review`, *Event detection*,
  offline) sits after an Epochs strip. It passes the signal on unchanged,
  reads the `epochs` rail and writes its own `epochs` rail: the epochs kept,
  with the labels the reviewer gave. Later strips read the nearest `epochs`
  rail (`docs/RAILS.md` rule 7), so they get the reviewed epochs; the Epochs
  strip's rail stays as it was cut (rule 8: one producer per rail). Without an
  Epochs strip before it, the pipeline is refused ("reads the epochs rail,
  which no earlier node writes"). Parameters: `labels` (offered for relabelling,
  with the epochs' own labels) and `file` (Advanced, as the Review strip's).
- **Why a strip of its own:** a strip's ports are fixed by its manifest, and a
  rail input needs a producer before it. Making the Review strip join the
  `epochs` rail only when there is one would need optional rail ports (a
  plugin spec change). The two strips share the sidecar layout, the reviewer's
  name and the pause in the offline run.
- **The verdicts** are saved per recording, as the Review strip's marks are
  (`<file>_desc-<strip id>_events.tsv`, or in its BIDS derivatives): one row
  per epoch, `onset` and `duration` its span in seconds from the recording's
  start, `trial_type` the label given, `event` the label it was cut around,
  `epoch_status` `good` or `bad`; the JSON sidecar names the input (signal and
  epochs) they were made on, who reviewed and when. The same input reuses them;
  any change upstream asks again, with the verdicts made before filled in for
  the epochs at the same place (rule 7). Every epoch is reviewed: one left
  untouched is kept as it was cut.
- **Rejected epochs on the Events rail** (decided 2026-10-30): each rejected
  epoch is also a `rejected_epoch` span over its window (payload: the label
  given and the event it was cut around), so the file writers record it (an
  annotation in `.set` and EDF, a row in the BIDS `events.tsv`). It is not a
  `BAD` label: a Reject segments strip cuts it only when its `reject` words
  name it. Strip version 0.2.0.
- **The epoch review window** opens while the run waits there: a page of
  epochs side by side (5 by default), every channel stacked, the event at the
  dashed line. Click an epoch to select it; ←/→ move the selection, PgUp/PgDn
  turn pages. *Accept* (A), *Reject* (R; Space toggles) and *Label* (L, with
  the label chosen or typed in); keys 1 to 9 give the first nine labels;
  Ctrl+Z undoes. Rejected epochs are shaded red.
- **The ERP view** below: the mean of the kept epochs of each label, on the
  channel chosen (Cz when there is one), redrawn as the verdicts change.
- **Control API 0.2.15:** a waiting review's request has `kind` (`marks` or
  `epochs`) and, for an epoch review, `epochs`; `offline_review_epochs` sends
  the epochs' data, or their mean; `submit_review` takes `epochs`.

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

Built (step 4a of Design v2, detectors): three strips under *Event
detection* (`meegxer.nodes.detectors`, offline, on the whole recording; each
piece on its own after a cut). Each passes the signal on unchanged, wraps one
MNE function (principle 6) and puts what it finds on the Events rail, where a
Reject segments strip cuts the `BAD_…` spans and (4b) a Review strip proposes
them. The input's bad channels are left out (but an EOG channel named in
*Detect blinks* is used).

- **Detect amplitude & flat** (`detect_amplitude`, MNE `annotate_amplitude`).
  MNE's peak-to-peak is the change between two consecutive samples:
  - `peak` (default 100 µV): a larger jump is a `BAD_peak` span (from
    `min_duration`, Advanced, 0.005 s).
  - `flat` (default 0.1 µV): smaller changes lasting `flat_duration` (0.1 s)
    are a `BAD_flat` span. MNE is called once per limit, so each has its own
    shortest span.
  - A channel past a limit for more than `bad_percent` (5 %) of the
    recording gets no spans: it is proposed as bad, as a `bad_channel` event
    at the recording's first sample (payload `channel`, and `reason` `peak`
    or `flat`). The strip does not mark it bad; 4b asks the reviewer.
- **Detect muscle** (`detect_muscle`, MNE `annotate_muscle_zscore`):
  `BAD_muscle` spans where the z-scored envelope of the `low`–`high` band
  (default 110–140 Hz) passes `zscore` (4). The band must lie below half
  the sampling rate: at 250 Hz, lower it (e.g. 70–110 Hz); the strip says so.
- **Detect blinks** (`detect_blinks`, MNE `find_eog_events`, on the strongest
  of the `eog` channels: `auto` takes those typed or named EOG, or name them,
  e.g. `Fp1`): a span of `half_width` (0.25 s) either side of each blink's
  peak, labelled `mark_label` (default `blink`; `BAD_blink` to have them cut), with
  the peak's sample in the payload (`peak`). Advanced: the filter band (1–10
  Hz) and `blink_threshold` (MNE's, a peak's prominence on the filtered EOG; 0: a
  quarter of its range). A piece shorter than 10 s (MNE's EOG filter) is
  skipped.

Built (step 4b of Design v2, suggestions):

- **Suggestions** are the events the Review strip's `points` names: a
  detector's upstream, or the recording's own (`origin` `input`); add
  `bad_channel` for a detector's proposed bad channels. Its `decision`
  (*free* by default, or *verify*) says how they are answered.
- **The strip takes those labels over** (`docs/RAILS.md` rule 6): strips
  after it, file writers and the views see its verdicts, never the
  suggestions. A Reject segments strip after it cuts only the accepted and
  edited `BAD_…` spans.
- **Verdicts are worked out in the core** (`meegxer.core.review.settle`) from
  the answer: a mark answering a suggestion (its `suggestion`) is
  `accepted` when its span and label are the suggestion's, else `edited`; a
  `bad_channel` suggestion is accepted when its channel is marked bad; one
  answered no is `rejected`. With *verify* a suggestion without an answer is
  refused; with *free* it is rejected.
- **The marks file** gets the columns `channel`, `suggestion_origin`,
  `suggestion_label`, `suggestion_onset`, `suggestion_duration` and `verdict`
  (described in its JSON sidecar). Rejected rows stay in the file, so a
  detector's misses can be measured, but are not emitted; `bad_channel` rows
  record the channel's verdict (the channel itself is in `bads`). An older
  file's `review_point` column still loads.
- **After an upstream change** the window opens again with the earlier
  answers to the suggestions still made filled in (rejections too), so only
  new or changed suggestions are left to answer.
- **The window** (the review points' navigation, ◀/▶ and ←/→): a
  *Suggestions* list under the marks (onset, label, the strip it came from,
  the verdict; double-click to go there), and the counter `Suggestion 3/40 ·
  12 answered`. On the traces a suggestion is dashed while as proposed (the
  accent colour until answered, green once accepted), solid once edited, red
  when answered no.
  - *Free:* each suggestion opens as a mark (and a proposed bad channel as
    bad), to keep, change or delete; deleting it answers no.
  - *Verify:* nothing is decided at the start. **Y** yes (as proposed),
    **N** no, **E** edit (kept, its span selected to retime and relabel);
    Y and N move on (*Next suggestion after answering*). Free marking is
    off, and *Save* waits until every suggestion has an answer.
- **Control API 0.2.17:** the request's `points` carry `origin` (and
  `channel`) and `decision`; marks carry `suggestion` (no longer `point`),
  and a rejection is a mark with `verdict` `rejected`.

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
