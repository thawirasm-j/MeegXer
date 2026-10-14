# meegxer-runtime

Containerized plugin runtime and SDKs (research builds only, never in the sterile
build). Protocol: `spec/plugin-protocol.md` at the repository root.

- `src/meegxer/runtime/`: host side (wire format, worker lifecycle, adapter, plugin
  index, conformance suite, `meegxer-plugin` CLI).
- `sdk/python/meegxer_plugin.py`: single-file Python SDK (numpy only); copy it into
  your image.
- `sdk/octave/meegxer_plugin_run.m`: GNU Octave SDK.
- `plugins/`: reference plugins with Dockerfiles. Build from this directory, e.g.

```bash
docker build -f plugins/average_reference/Dockerfile -t meegxer/example-average-reference:0.1.0 .
uv run meegxer-plugin check meegxer/example-average-reference:0.1.0
uv run meegxer-plugin add meegxer/example-average-reference:0.1.0
```

After `add`, the plugin is a node like any other (`meegxer nodes` lists it) and can be
used in pipeline files by its manifest name.
