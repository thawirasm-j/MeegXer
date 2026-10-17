# MeegXer plugin specification

Language-neutral definition of how MeegXer talks to processing nodes and clients
(CLAUDE.md rule 11). The Python and Octave SDKs implement this spec; they do not
define it.

- **Versioning:** semver, recorded in `VERSION`, independent of package versions.
  `0.x` means unstable: any minor bump may break compatibility.
- **Contents:**
  - `manifest.schema.json` (0.2.0): node manifest: identity (`name`, `version`),
    `type` (stream | analysis), typed `inputs`/`outputs`, `params` (float, int, bool,
    enum, string with defaults and bounds), `latency_samples`, `requires_fit`, and
    `image` for external plugins. The core ships an identical copy for validation.
  - `pipeline.schema.json` (0.2.1: per-node `pass_uncalibrated`; 0.2.2: per-node
    `strip_params`, display only; 0.2.3: `recording` on the source: channel
    names, types, bad flags and positions, a montage, and events; 0.2.4:
    `channel_types` on processors, default EEG only; 0.2.7: an omitted `input`
    takes the nearest earlier node whose output type matches, so Features nodes
    branch off a linear chain): a processing graph as an
    ordered node list;
    each node's `input` names an earlier node (acyclic by construction) and defaults
    to the nearest earlier source or processor, so a plain list is a linear chain.
  - 0.2.0 also added manifest `role` (source | processor | probe | sink),
    `live_params` (parameters that change without resetting streaming state; on a
    node that needs fitting, since 0.2.6, also without invalidating its model),
    `availability` (online | offline | both) and the per-parameter `advanced` flag
    (hidden in the UI's friendly mode). 0.2.2 added the per-parameter `primary` flag
    (crucial parameters, shown on the node's strip in friendly mode).
  - 0.2.8: the host never sends a node a block with zero samples; the empty
    result of a delaying node is not handed on (`plugin-protocol.md`).
  - `session.schema.json` and `session.md` (0.2.9, session format 1.0): the
    session file, the unit of reproducibility: a pipeline plus node versions and
    provenance (external plugins: image, image ID, registry digests), fitted
    models as readable JSON, input/output file hashes and the environment.
  - `bids-derivatives.md` (0.2.10): the BIDS-derivatives layout MeegXer writes
    (`meegxer.core.bids_writer`, `meegxer.core.bids_features`): `desc-` outputs
    with sidecars, `dataset_description.json` with `GeneratedBy`, and the run's
    session file next to each recording.
  - 0.2.17: rail ports. A port may carry `rail` (the rail an extra port, not the
    first, reads or writes; `events` is the Events rail) and a Signal port
    `space` (`sensor`, `component`, `source`). `fit` and `transform` requests
    carry `rails` for the extra inputs; results carry `rails` for the extra
    outputs (`plugin-protocol.md`).
  - 0.2.16: optional manifest `fit_events`: the node learns from labelled
    trials; its `fit` request carries the calibration's `events` (absolute sample
    indices), and hosts refuse to fit it without any. Needs `requires_fit`.
  - 0.2.15: optional manifest `category`: where a node is listed within its
    group (e.g. `filters`, `spectral`, `files`); hosts group their node menus by
    it, built-ins and plugins alike. Optional `channel_types`: the channel types
    the node processes unless the pipeline names others (default `["eeg"]`).
  - 0.2.14: feature results: an analysis plugin's `result` carries `features`
    (window, dims, coords, optional timestamp and an array per completed window)
    instead of `data`.
  - 0.2.13: optional parameter `step` (float and int parameters): how much one
    step (arrow key, wheel) changes the value in a UI. Hosts choose one when it
    is absent.
  - 0.2.12: manifest `authors`, `references` (each `citation` with an optional
    `doi` or `url`) and `license`, for credit; hosts show them and gather the
    references into session and BIDS provenance. Optional; nothing is refused for
    leaving them out.
  - 0.2.11: manifest `requires_positions`: the node needs an electrode position
    for every good channel it processes (ICA's ICLabel). Hosts check the input
    (after its `recording` overrides) before running and refuse to start without
    them, naming the node and channels; a live stream is checked once it is open.
  - `control-api.md`: JSON-RPC 2.0 methods between the daemon and clients.
  - `display-protocol.md`: decimated min/max envelope frames on the display stream.
  - `plugin-protocol.md` (`meegxer.plugin/0.2`): framed stdin/stdout protocol for
    containerized plugins (handshake, fit, load_model, transform, flush, shutdown,
    errors), with host-owned model and state. 0.2 (spec 0.2.5): results carry the
    `sample_index` of their first sample, so output may lag its input (look-ahead),
    plus the end-of-stream `flush` message and `MEEGXER_PROTOCOL_FD`.

Still 0.x: expect breaking changes until v1.
