# Rails (design)

Status: the chain view draws rails for what the pipeline has today (feature rails,
the input's events). The rest is a proposal; points marked **[rule]** change an
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

## 4. Display (built)

- The signal is the thick line; rails are thin lines above the strips.
- Preprocessing strips sit on the signal. Feature and output strips hang below it.
- Each rail has its own row: feature rails in chain order, the earliest highest,
  Events on the top row. A rail is solid to its last reader, then continues
  dashed, thinner and fainter (whiter on dark themes, darker on light ones) to
  the end of the chain, with an arrowhead.
- **Arrows:**
  - ↓ from the signal into each strip that reads the signal, near its left edge;
  - ↓ from a rail into each strip that reads it (an output reads one line, a
    labelled fit also reads Events);
  - ↑ from a feature strip into its rail, near its right edge, with ● where it
    starts the rail; ○ will mark a strip that adds to an existing rail (Events);
  - outputs only pull; they push nothing;
  - a hop where an arrow crosses another rail;
  - the rail's name above its producer.
- **A strip that needs a rail the chain lacks** shows a red "needs Events" at the
  top row (e.g. CSP on a recording without events).
- Hovering a rail names it, its producer and its readers.

## 5. Strip and plugin inputs and outputs (to design next)

Strips and plugins need a way to say which lines they read and write.

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
2. Events rail in the core, and an LSL marker inlet (live CSP calibration follows).
3. Rail ports and `TransformResult.rails` [rules 6, 7]; plugin protocol and SDKs.
4. ICs rail from ICA and from EEGLAB files; IC rejection and IC features read it.
5. LDA classifier reading a feature rail and adding events.
6. Sources rail: inverse on a template head (cached ahead for Sterile mode).
7. Event outputs: keyboard sink, events writer, epoching.
