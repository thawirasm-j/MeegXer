# MeegXer plugin specification

Language-neutral definition of how MeegXer talks to processing nodes and clients
(CLAUDE.md rule 11). The Python and Octave SDKs implement this spec; they do not
define it.

- **Versioning:** semver, recorded in `VERSION`, independent of package versions.
  `0.x` means unstable: any minor bump may break compatibility.
- **Contents:**
  - `manifest.schema.json` (0.1.0): node manifest: identity (`name`, `version`),
    `type` (stream | analysis), typed `inputs`/`outputs`, `params` (float, int, bool,
    enum, string with defaults and bounds), `latency_samples`, `requires_fit`, and
    `image` for external plugins. The core ships an identical copy for validation.
- **Planned:**
  - `control-api.md`: JSON-RPC methods between core and clients. Milestone M3.
  - `plugin-protocol.md`: framed stdin/stdout protocol for containerized plugins.
    Milestone M4.

Still 0.x: expect breaking changes until the M4 plugin protocol lands.
