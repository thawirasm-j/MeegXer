# Rails (design)

Status: the chain view draws rails for what the pipeline has today (feature rails,
the input's events), and the core carries the Events rail in both schedulers
(step 2). The rest is a proposal; points marked **[rule]** change an
architecture rule and need approval before they are built.

## 1. The idea

The pipeline is **one chain of strips** on one thick line, the **signal**. Every
strip sits on it in order. Above it, thinner **parallel rails** carry what a
strip, or the input, produces for later strips: events, ICs, source-region
activity, features. A strip *pulls* from the signal or from a rail and *pushes*
to the signal or to a rail; the drawing shows only those pulls and pushes.

There are no side rails. A feature strip always starts its own named rail, and an
output strip chooses which line it draws from.

| Line | What it is | Who writes it | Who reads it | Drawn as |
|---|---|---|---|---|
| **Signal** | The sensor signal, changed in place by each processor | Every processor | The next strip; any output | Thick line through every strip |
| **Parallel rail** | Something produced at one point that later strips may use | Its producer; any strip may add to Events | Strips and outputs to the right of its origin | Thin line above the strips from its producer (●), solid to its last reader, then dashed to the end of the chain |

## 2. Rails

| Rail | Carries | Port type | Started by | Read by (examples) |
|---|---|---|---|---|
| **Events** | Markers and commands at sample indices | Events | **The input**, when the file has events (annotations, an events file) or LSL markers; else the first strip that emits events | CSP and other `fit_events` nodes, epoching, action sinks, events writer |
| **ICs** | Component activations `IC01…` with classes | Signal, space `component` | ICA, or **the input** when the file carries a decomposition (an EEGLAB `.set` with `icaweights`) | IC rejection, eye/muscle detectors, IC features |
| **Sources** | Region time courses (atlas labels) | Signal, space `source` | An inverse strip | ROI features, ROI connectivity |
| **Features · name** | Windowed features, one rail per feature strip (`PSD`, `Band power`, `Band power·2`, `CSP`) | Features | That feature strip | Outputs (tables, LSL), classifiers (LDA) |

## 3. Rules

1. **One chain.** Every strip, outputs included, sits on the signal line in one
   order. Nothing is placed on a rail.
2. **A rail starts at its producer** (a strip, or the Input strip for what the file
   carries) and exists only to its right.
3. **A rail lasts to the end of the chain.** Past its last reader it is drawn
   dashed: still available, read by nothing yet.
4. **A rail is fixed at its origin.** "ICs from ICA after High-pass" stay the
   components of the high-passed signal, even if later strips change the signal.
5. **Staleness follows the origin.** A change upstream of a rail's origin, or a
   refit of its producer, marks the rail and its readers as needs-refit (rule 7).
6. **Events: one rail per chain, many contributors**, merged in sample-index order
   and tagged with their origin. ICs and Sources: one rail per producer, so
   nothing is silently replaced. A strip may **take labels over** (a Review
   strip with suggestions, `docs/REVIEW.md` step 4b): strips after it no longer
   see the events with those labels from before it (the input's or an earlier
   strip's), only what it emits for them, its verdicts.
7. **Feature rails are named** after their strip, numbered when repeated, so a
   reader knows what it draws from. A reader takes the nearest rail of its input
   type unless it names one (the inspector's *Reads*; the node's `input` in the
   pipeline file).
8. **Only the producer writes a rail**, except Events, which any strip may add to.

## Events rail in the core (built)

- **Where it starts:** the input.
  - Offline, and for a recording played live: the recording's events (annotations
    or an events file, after the recording overrides).
  - Live LSL: the input's marker stream (`marker_stream`, with `marker_type` under
    Advanced). Each marker is placed once, at the inlet, on the signal's sample
    index from its LSL timestamp (rule 8); both inlets use clock sync and
    dejitter, and the signal's recent chunks anchor timestamps to indices.
  - An XDF file (LabRecorder, M10.5): its marker streams, each marker on the
    signal sample nearest its timestamp (pyxdf's clock sync and dejitter, the
    same rule as live), and a `BAD_gap` mark where samples were lost
    (`meegxer.core.xdf`).
- **Who adds to it:** any strip, through the events its transform emits.
- **Who sees what:** a strip sees the input's events and those of the strips
  upstream of it, in sample-index order (`meegxer.core.events.EventsRail`),
  except the labels a strip between takes over (rule 6; a processor's
  `takes_over(params)`). An output (a file writer, an action sink) sees
  every strip's events to its left, as the rail is drawn, even one that hangs
  off the signal: the LDA strip's class events reach a writer reading the
  signal (M11.3). A processor sees its ancestors' only, because its input
  fingerprint names them (rule 7). The offline views see what a writer at the
  chain's end would see. Live, the writers get the rail every 2 s (a crash
  recovery keeps most of it) and in full before they close; the live scheduler
  keeps an untrimmed copy for them.
- **Across a rate change** (a Resample strip): the rail counts the input's
  samples; a strip at another rate sees, and adds, events at its own rate,
  mapped by the exact ratio of the rates to the nearest sample (rule 8). A
  Signal rail is not mapped: reading one across a rate change stops with a
  message (read it before the Resample strip). Feature rails and Features taps
  after a rate change still count windows in the resampled samples.
- **Live labelled calibration:** `calibrate NODE --seconds N` collects N s, then
  waits `marker_wait` (default 0.5 s) for markers still on their way, while
  processing goes on, and fits on the events inside the stretch.
- **Status:** `input_events`, `events_late` (markers that landed in a stretch
  already fitted on: missed trials), and the LSL input's `markers` and
  `markers_dropped` (before the first sample, or without a label).

## ICs rail (built, steps 4a and 4b)

- **ICA + ICLabel writes it:** an extra output (`ics`, space `component`) carries
  the component activations (`weights @ x`, in volts), one channel per component
  (`IC01`...). Each channel has ICLabel's most likely class and its probability
  as attributes (`SignalInfo.ch_classes`, `ch_scores`).
- **A strip can take it as its main input:** `input: "ica1:ics"` (spec 0.2.18).
  Only a strip that makes sense on components: the ICs rail is in `component`
  space, and strips that only make sense on electrodes take `sensor` space
  (re-reference, interpolation, CSD, EOG regression, ICA, and the MNE ICA
  writer, which stores the decomposition of electrodes). Spaces that differ
  do not connect (spec 0.2.22): *Reads* does not offer the ICs there, and a
  pipeline file wiring them is refused.
  In the inspector, a strip on the signal gets *Reads*: the signal, or a rail
  output such as *ICs (ICA + ICLabel)*; e.g. PSD or band power per component.
  The chain draws it as a reader of the ICs rail, without a pull from the signal.
- **Refits:** a refit of ICA changes its components, so the strips reading them
  need a refit too (their input fingerprint names the rail).
- **From an EEGLAB file (4b, M10.6):** a `.set` with `icaweights` starts the ICs
  rail at the Input strip (`meegxer.core.eeglab_ica`; the File input's rail
  output `ics`, spec 0.2.21: a source's extra outputs name the rails its input
  carries).
  - The decomposition: MNE's `read_ica_eeglab` (`icaweights @ icasphere`, or
    `pinv(icawinv)` where components were removed in EEGLAB), on the channels
    of `icachansind`; activations `weights @ x` as ICA's.
  - ICLabel classes: the file's (`EEG.etc.ic_classification.ICLabel`) when it
    has them, else ICLabel's, computed by the ICA strip (reached through the
    node registry: the core imports no node package) when a run first reads
    them; without positions or the ICA strip they stay unclassified, and the
    notes say why.
  - `describe_source` says what the input carries (`carries`, control API
    0.2.22) and the notes how (`EEGLAB decomposition: 18 components on 19
    channels start the ICs rail; ICLabel classes from the file`); `get_model`
    on the input gives the decomposition, for the IC head plot. The GUI asks as
    soon as the input's file is chosen (the log note and the rail, before any
    run).
  - The chain: an ICA strip after the input ends the file's rail at its left
    edge (a bar), and its own ICs rail goes on along the same row.
- **Into the `.set` written (M10.6):** the *Save as .set* writer stores the
  decomposition that unmixes the data it writes, as EEGLAB keeps one
  (`icaweights`, an identity `icasphere`, `icawinv`, `icachansind`, and the
  ICLabel classes as `etc.ic_classification.ICLabel`): the ICs rail's nearest
  writer before it (the input's file, or an ICA strip), less the components
  the strip removed from the data, as EEGLAB's `pop_subcomp` (rows of
  `icaweights` and columns of `icawinv` kept; MNE and EEGLAB then read
  `icawinv`).
  - None after a strip that mixes the channels (re-reference, interpolation,
    CSD, EOG regression): the decomposition no longer unmixes the data, and
    the notes say so. Filters, resampling and segment rejection keep it, as in
    EEGLAB. A decomposition on a channel the writer no longer has is left out,
    with a note.
  - Live and offline alike (writers are bound to the graph in both
    schedulers). A crashed run's recording (`meegxer recover`) is written from
    its spool alone, without the decomposition.
- **MNE ICA files (M10.7b):** the File input's `ica` (an `-ica.fif` of the
  recording; `meegxer.api.run(..., ica=)` takes the ICA object too) starts
  the ICs rail instead of a `.set`'s own (`meegxer.core.mne_ica`).
  - The decomposition is measured with MNE's own `ICA.get_sources` on a probe
    recording (a zero column, then one unit column per channel), so the
    ICA's projectors (an average reference), pre-whitener and PCA mean are
    MNE's: `get_sources(x) = W x + b`. MeegXer's weights take µV, as
    EEGLAB's, so the rail carries MNE's sources read as µV: `W / 1e6` and
    `b / 1e6` (`Decomposition.offset`).
  - ICLabel classes are computed: an MNE ICA keeps none (`labels_` names a
    class, without its probabilities). Components the file marks `exclude`
    stay on the rail (the data is not cleaned), and the notes say so.
  - The *MNE ICA (.fif)* writer (`meegxer.core.ica_writer`, `…-ica.fif`)
    stores the decomposition of the signal it reads, chosen as the `.set`
    writer's: every component, with those an ICA strip removed marked
    `exclude` (MNE keeps them all), and ICLabel's classes as `labels_` (as
    mne-icalabel's `label_components` sets them). It goes into MNE through
    MNE's own EEGLAB import (`read_ica_eeglab` on a small `.set` holding it),
    then `ICA.save`; `get_sources` on volts then gives EEGLAB's activations
    in µV. With no decomposition before it, it writes nothing and says why.
  - Sessions resolve `ica` as a path; a shared session leaves it out (`""`:
    an ICA belongs to its recording).
  - An ICA strip after the input rewrites the rail: later readers get ICA's
    (the nearest writer, rule 7). The chain draws the file's rail ending at
    the ICA strip, with a bar, and ICA's starting there.

## Epochs rail (built, live and offline)

- **Who writes it:** the **Epochs** strip (`meegxer.nodes.epochs`, Epochs).
  It passes the signal on unchanged and writes its `epochs` output (a Features
  rail).
- **What it carries:** one feature window per epoch.
  - `dims` are `(epoch, channel, time)`.
  - The `epoch` coordinate is the event's label; `time` is in seconds from the
    event; the window is the epoch's sample span; the unit is volts.
  - So no new port type is needed (rule 6): the schedulers, the plugin
    protocol, Features taps and rail readers carry it as it is.
- **What it reads:** the Events rail (the input's events, review marks, and
  those of earlier strips).
- **Parameters:**
  - `events`: the labels to cut around (empty: every event).
  - `tmin`/`tmax`: the epoch, in seconds from the event (default -0.2 to 0.8).
  - `baseline`: `pre-stimulus` subtracts each channel's mean over tmin..0
    (the default, MNE's `(None, 0)`); `custom` its mean over
    `baseline_start`..`baseline_end` (seconds from the event, inside the
    epoch; MNE's `(a, b)`); `none` keeps the signal as is. MNE's `rescale`
    subtracts it, so the epochs equal `mne.Epochs(…, baseline=…)`'s.
  - `reject` (default `BAD`): an epoch overlapping a mark whose label starts
    with one of these words is left out.
- **Left out:** an epoch across a cut (each piece between cuts is epoched on
  its own), or one running off either end of the recording.
- **Live:** between blocks the strip keeps the newest samples an epoch can
  still need (the longer of the epoch and `-tmin`), the events whose epochs
  end after the block, and the marks to reject that reach into those samples.
  An epoch is written in the block that brings its last sample, so live and
  offline write the same epochs whatever the block sizes (acceptance test 1).
  A gap in the sample index (a cut) starts it over.
- **Offline store:** the offline feature store keeps each epoch's label
  (`offline_features` returns `labels`, control API 0.2.10).
- **ERP tap (M10.3):** the GUI taps an epochs rail with an **ERP tap**
  (`meegxer.core.erp_probe`), not a Features tap. It keeps each label's
  running mean, sum of squared deviations and count, in the core (Welford's
  update), so the mean (MNE's `Epochs.average` over the same epochs) and its
  standard error are exact however many display frames are lost (rule 2).
  - **SEM:** the sample standard deviation (n − 1) over √n, as
    `scipy.stats.sem` (the maintainer's choice; MNE's `Epochs.standard_error`
    divides by n). None for a single epoch.
  - **What it covers:** every epoch since the run started, its last *Reset*
    (`reset_probe`, control API 0.2.20), or a change upstream of the tap (its
    input fingerprint changes; rule 7: no mean across a change). Another epoch
    span or montage starts it over too.
  - **Live:** at each display flush where it changed it sends its whole ERP as
    one features frame: `dims` `(stat, condition, channel, time)`, the mean
    and its SEM, with each condition's epoch count (`counts`, display
    protocol 0.4).
  - **Offline:** the same tap runs in the offline run; `offline_erp` reads it.
  - **The ERP view:** times in ms from the event. *Show* `All conditions`
    draws each condition on one channel (Cz first; a click on the head
    picks another) with its n in the legend; one condition draws the
    channels picked on the head overlaid (butterfly). *SEM* (off by
    default) shades each curve's mean ± SEM. *Reset* (live) starts the
    averages over.
- **Reviewed epochs:** a **Review epochs** strip after it
  (`meegxer.core.epoch_review`, `docs/REVIEW.md` step 5) reads the rail and
  starts its own `epochs` rail with the epochs a person kept and relabelled,
  and puts each rejected one on the Events rail as a `rejected_epoch` span.
  Later readers take the nearest one (rule 7), so they get the reviewed
  epochs; the Epochs strip's rail is not rewritten (rule 8). Its window has
  the ERP view (the mean of the kept epochs of each label).
- **Into files:** an **Epochs and ERPs file** strip
  (`meegxer.core.epochs_writer`) or a **BIDS epochs and ERPs** strip
  (`meegxer.core.bids_epochs`, `spec/bids-derivatives.md`) reads the nearest
  epochs rail (its input names it, e.g. `ep:epochs`) and writes the epochs as
  an MNE `EpochsArray` (`-epo.fif`, or an epoched EEGLAB `.set`) and their
  average per label, the ERPs, as `-ave.fif` evoked files (MNE
  `Epochs.average`, `write_evokeds`). After a Review epochs strip that is the
  kept epochs with their labels; the rejected ones stay in the review's file.
  Each epoch's event is at its sample on the source's clock; channel types
  come from the recording.

## 4. Display (built)

- The signal is the thick grey line; rails are thin coloured lines above the
  strips (Events violet, the ICs blue, feature rails orange).
- Preprocessing strips sit on the signal. Feature and output strips hang below it.
- Each rail has its own row: feature rails in chain order, the earliest highest,
  Events on the top row. A rail is solid to its last reader, then continues
  dashed, thinner and fainter (whiter on dark themes, darker on light ones) to
  the end of the chain, with an arrowhead, half transparent.
- **Arrows:**
  - ↓ from the signal into each strip that reads the signal, near its left edge;
  - ↓ from a rail into each strip that reads it (an output reads one line, a
    labelled fit also reads Events);
  - ↑ from a feature strip into its rail, near its right edge, with ● where it
    starts the rail, and ○ where a strip adds to a rail it did not start
    (Events: a review, a detector, Reject segments);
  - outputs only pull; they push nothing;
  - a hop where an arrow crosses another rail or the signal;
  - the rail's name above its producer.
- **The Events rail** starts at the input when the recording (or the LSL
  marker stream) has events, else at the first strip that writes events (an
  Event detection strip: a review, a detector; or Reject segments). Every
  later strip writing events adds to it (○).
- **A strip that needs a rail the chain lacks** shows a red "needs Events" at the
  top row (e.g. CSP on a recording without events, before any review or
  detector).
- Hovering a rail names it, its producer, the strips adding to it and its
  readers.
- **Clicking a rail's name shows it** (◇, ◆ while shown): an epochs rail in the
  ERP view, another feature rail in the
  feature view (a spectrum, or values over time), the ICs rail as component
  traces coloured by ICLabel class. It is shown alone, in place of the signal
  taps, until clicked again or a signal tap is clicked.

## 5. Strip and plugin inputs and outputs (built, spec 0.2.17)

Strips and plugins say which lines they read and write through their ports.
Built: manifest ports, the transform and fit contract, both schedulers (live:
wait and align, a reader's input held at most 30 s for a lagging rail), the
conformance check, the plugin protocol and the Python SDK, and the chain view.
Still to come: sources declaring extra outputs, custom scripts, the Octave SDK.

- **Manifest ports name lines [rule 6].**
  - Inputs: `[{"name": "in", "type": "signal"}, {"name": "labels", "type": "events", "rail": "events"}]`.
  - Outputs: `[{"name": "out", "type": "signal"}, {"name": "ics", "type": "signal", "rail": "ics", "space": "component"}]`.
  - The first port stays the main one. Port types stay three; Signal ports gain
    `space` (`sensor`, `component`, `source`).
- **Transform [rule 7].**
  - `transform(block, model, state, params, rails)`: `rails` holds the current
    blocks of the extra input ports.
  - It returns `TransformResult(block, state, events, rails={...})` for the extra
    output ports.
  - `events` is already such an extra output today. Fit gets the same `rails` for
    labelled or rail-based fits; `fit_events` becomes the Events case of it.
- **Plugin protocol.**
  - `transform` and `fit` requests carry a `rails` object (headers plus arrays per
    port).
  - Replies carry `rails` the same way.
  - SDKs pass them as a dict, the way `info["events"]` works for fit now.
- **Sources.** A source may declare extra outputs too: the file's events, an
  EEGLAB decomposition as ICs. That is how the Input strip starts rails.
- **Custom scripts.** Extra parameters named after rails (`events`, `ics`) receive
  those rails. Returning a dict with `rails` writes them.

## 6. Build order (one PR each)

1. Rail drawing in the chain view, named feature rails, *Reads* choice (this PR).
2. Events rail in the core, and an LSL marker inlet on the input (done: live
   labelled calibration, e.g. CSP).
3. Rail ports and `TransformResult.rails` [rules 6, 7]; plugin protocol and the
   Python SDK (done; scripts, sources and the Octave SDK follow).
4. ICs rail: from ICA, with ICLabel classes, read as a main input (done, 4a);
   from EEGLAB files at the Input strip, rewritten by ICA (4b).
5. LDA classifier reading a feature rail and adding events (done, M11.3: its main
   input is the feature strip, chosen under *Reads*, spec 0.2.23).
6. Sources rail: inverse on a template head (cached ahead for Sterile mode).
7. Event outputs: keyboard sink (done, M11.4: an action sink, armed by hand,
   acting on each new event it sees as it reaches the rail), events writer,
   epoching (done, M10.2).
8. An events strip (e.g. "LSL markers") for an external event source: it adds to
   the Events rail (○) rather than starting it. Its markers are still placed on
   the input's sample index; its place in the chain only decides which strips
   see them.
