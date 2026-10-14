# MeegXer plugin specification

Language-neutral definition of how MeegXer talks to processing nodes and clients
(CLAUDE.md rule 11). The Python and Octave SDKs implement this spec; they do not
define it.

- **Versioning:** semver, recorded in `VERSION`, independent of package versions.
  `0.x` means unstable: any minor bump may break compatibility.
- **Planned contents:**
  - `manifest.schema.json`: node manifest (name, version, type, params, latency,
    `requires_fit`). Milestone M1.
  - `control-api.md`: JSON-RPC methods between core and clients. Milestone M3.
  - `plugin-protocol.md`: framed stdin/stdout protocol for containerized plugins.
    Milestone M4.

Nothing here is normative yet.
