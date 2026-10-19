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
   nothing is silently replaced.
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
- **Who adds to it:** any strip, through the events its transform emits.
- **Who sees what:** a strip sees the input's events and those of the strips
  upstream of it, in sample-index order (`meegxer.core.events.EventsRail`).
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

## ICs rail (built, step 4a)

- **ICA + ICLabel writes it:** an extra output (`ics`, space `component`) carries
  the component activations (`weights @ x`, in volts), one channel per component
  (`IC01`...). Each channel has ICLabel's most likely class and its probability
  as attributes (`SignalInfo.ch_classes`, `ch_scores`).
- **A strip can take it as its main input:** `input: "ica1:ics"` (spec 0.2.18).
  In the inspector, a strip on the signal gets *Reads*: the signal, or a rail
  output such as *ICs (ICA + ICLabel)*; e.g. PSD or band power per component.
  The chain draws it as a reader of the ICs rail, without a pull from the signal.
- **Refits:** a refit of ICA changes its components, so the strips reading them
  need a refit too (their input fingerprint names the rail).
- **Next (4b):** an EEGLAB file's decomposition starts the ICs rail at the Input
  strip; an ICA strip after it rewrites it, drawn as the file's rail ending at the
  ICA strip and a new ICs rail starting there.

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
    starts the rail; ○ will mark a strip that adds to an existing rail (Events);
  - outputs only pull; they push nothing;
  - a hop where an arrow crosses another rail or the signal;
  - the rail's name above its producer.
- **A strip that needs a rail the chain lacks** shows a red "needs Events" at the
  top row (e.g. CSP on a recording without events).
- Hovering a rail names it, its producer and its readers.
- **Clicking a rail's name shows it** (◇, ◆ while shown): a feature rail in the
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
5. LDA classifier reading a feature rail and adding events.
6. Sources rail: inverse on a template head (cached ahead for Sterile mode).
7. Event outputs: keyboard sink, events writer, epoching.
8. An events strip (e.g. "LSL markers") for an external event source: it adds to
   the Events rail (○) rather than starting it. Its markers are still placed on
   the input's sample index; its place in the chain only decides which strips
   see them.
