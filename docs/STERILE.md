# Sterile mode

Sterile mode (CLAUDE.md rule 10, ROADMAP M11.1) runs a pipeline's processing
isolated from the computer and the network: in a container with no network,
no devices, no capabilities and a read-only filesystem. It is a runtime
switch, not a package or a build. It means isolation only. Which code runs,
and whether it is fit for patient data (HIPAA, GDPR, IRB), stays the user's
responsibility.

Status: in progress. Done: the worker, the host proxies and the image
(`meegxer sterile build`); in the container, every built-in strip gives the
local run's output exactly (offline; CI checks it on every change). Next: the
*Sterilize* switch in the CLI, the daemon and the GUI, live runs with the
delay measured, and custom scripts inside the worker.

## The image

```bash
meegxer sterile build     # once, and again after an upgrade
meegxer sterile check     # is it built from this MeegXer?
```

`meegxer sterile build` (Docker or Podman) builds `meegxer-sterile:VERSION`
from the MeegXer installed here: it copies the installed `meegxer.core`,
`meegxer.nodes` and `meegxer.plugins` into the image, whatever they are (a
release or an edited checkout), and installs their third-party dependencies
at the versions installed here (`python:3.X-slim`, the same Python minor
version). It needs the network once, for the base image and the
dependencies; a run never does. The image is labelled with a SHA-256 of the
code it holds; a Sterile run checks the label first and refuses an image
built from other code (after an upgrade or an edit) with what to do, and the
worker checks it again when it starts. It holds no keyboard library (the
keyboard is a host sink).

The container runs as `docker run --rm -i --network none --read-only
--tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges --pids-limit 256
--memory 8g`, as a non-root user, with no folder mounted.

## How it works (decided 2026-11-04)

- **One worker per run.** One container runs every built-in strip of the
  pipeline (`python -m meegxer.plugins.sterile_worker`). Data reaches it on
  stdin and leaves on stdout, as a plugin's does (`spec/plugin-protocol.md`
  → *Sterile worker*). No folder is mounted: it reads and writes no files.
- **A proxy per strip.** On the host, each built-in processor is replaced by
  a proxy (`meegxer.plugins.sterile.SterileProcessor`) that hands each call
  (fit, transform, flush, …) to the worker and returns its answer. The
  schedulers, stages, rails, calibration, refits and sessions are unchanged:
  they call the proxy as they would call the node (rule 4: one node
  implementation). The host keeps model and state, so a worker that dies is
  started again and goes on where it stopped; models are sent to it once
  each, by content hash.
- **One hop per strip per block.** Each strip's call crosses the container
  boundary once per block (a few milliseconds per strip on a typical chain;
  measured with `tools/stability_test.py` before the switch ships). The
  alternative, the whole chain per hop with its own scheduler in the
  container, would have needed calibration, live settings, pass-through,
  refits and session models forwarded across the boundary.
- **Exact output.** Both ends are the same MeegXer: the image is built from
  the installed packages, and the worker refuses to start against another
  version. Arrays keep their dtype (float64 stays float64), so the output is
  the local run's, bit for bit
  (`packages/meegxer-plugins/tests/test_plugins_sterile.py` runs every
  built-in processor both ways). A node's error is raised on the host as the
  same kind of error with its words.

## What stays on the host

- Sources (files, BIDS, LSL), probes (what the GUI shows), sinks (file
  writers, LSL out, the keyboard): trusted I/O.
- The review strips (*Review*, *Epoch review*, *Reject segments*): they
  apply a reviewer's own marks and verdicts.
- Container plugins (EEGLAB ASR, Octave plugins): already in their own
  containers with no network and a read-only filesystem.

A custom script or local plugin runs as a local process, without isolation:
Sterile mode refuses it for now. It will run inside the Sterile worker
instead (M11.1c).

## Using it (from code, for now)

```python
from meegxer.core.sterile import sterilize

with sterilize(graph):  # the image meegxer-sterile:VERSION
    BatchScheduler().run(graph)
```

The core reaches the Sterile code only through the `meegxer.isolation`
entry point (rule 9's import contract); `meegxer-plugins` provides it.
For development, `sterilize(graph, local=True)` runs the worker as a local
process (needs `MEEGXER_ALLOW_UNSANDBOXED=1`, as unsandboxed plugins do).
