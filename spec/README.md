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
    `channel_types` on processors, default EEG only): a processing graph as an
    ordered node list;
    each node's `input` names an earlier node (acyclic by construction) and defaults
    to the nearest earlier source or processor, so a plain list is a linear chain.
  - 0.2.0 also added manifest `role` (source | processor | probe | sink),
    `live_params` (parameters that change without resetting streaming state; on a
    node that needs fitting, since 0.2.6, also without invalidating its model),
    `availability` (online | offline | both) and the per-parameter `advanced` flag
    (hidden in the UI's friendly mode). 0.2.2 added the per-parameter `primary` flag
    (crucial parameters, shown on the node's strip in friendly mode).
  - `control-api.md`: JSON-RPC 2.0 methods between the daemon and clients.
  - `display-protocol.md`: decimated min/max envelope frames on the display stream.
  - `plugin-protocol.md` (`meegxer.plugin/0.2`): framed stdin/stdout protocol for
    containerized plugins (handshake, fit, load_model, transform, flush, shutdown,
    errors), with host-owned model and state. 0.2 (spec 0.2.5): results carry the
    `sample_index` of their first sample, so output may lag its input (look-ahead),
    plus the end-of-stream `flush` message and `MEEGXER_PROTOCOL_FD`.

Still 0.x: expect breaking changes until v1.
