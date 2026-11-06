# Sterile mode

Sterile mode (CLAUDE.md rule 10, ROADMAP M11.1) runs a pipeline's processing
isolated from the computer and the network: in a container with no network,
no devices, no capabilities and a read-only filesystem. It is a runtime
switch, not a package or a build. It means isolation only. Which code runs,
and whether it is fit for patient data (HIPAA, GDPR, IRB), stays the user's
responsibility.

Status: in 0.5.0b2: the worker, the host proxies, the image (`meegxer
sterile build`) and the switch (the setting, `--sterile`, the GUI's *Sterile*
button), offline and live, checked on Linux and on Windows with Docker
Desktop; in the container, every built-in strip gives the local run's output
exactly (CI checks it on every change). Since then: custom scripts run in
Sterile mode too (below). Next: the delay brought down.

## Switching it on

Sterile mode is a choice per computer, like the theme, never written into a
pipeline or a session:

- **The GUI:** *Sterile* on the top bar (the `general.sterile` setting). A
  live run's status says `● running (Sterile)`; the log says `Offline run
  started (Sterile).` Switched on with no image, or one built from other code
  (after an upgrade), the GUI offers to build it, shows the build in the log,
  and runs once you press *Start* again.
- **The CLI:** `meegxer run … --sterile` (or `--no-sterile`); `meegxer daemon
  --sterile`. Without the image the run is refused at once, with what to do.
- **The setting:** `general.sterile` in `settings.json` (`docs/SETTINGS.md`),
  or `MEEGXER_STERILE=1` / `0`. A flag wins over the variable, which wins over
  the setting.
- **Python:** `meegxer.api.run(…, sterile=True)`;
  `BatchScheduler(sterile=True)`, `RealtimeScheduler(graph, sterile=True)`.

Every run in Sterile mode (offline, live, an all-subjects BIDS run and its
worker processes) runs its built-in strips in one Sterile worker, which ends
with the run; the stages have their local processors again after it.

## The delay (measured 2026-11-05)

Live, each strip's call crosses the container boundary once per block. On
Linux (Docker 29, `docker run -i`), 16 channels at 500 Hz in blocks of about
10 ms, through band-pass, notch and re-reference: about 2–3 ms per strip per
block in the container against 0.1–0.2 ms locally; the run kept up with its
input (the same samples in 25 s), with 62 of 2209 blocks late (3 %; 0
locally). Offline the cost is the container's start (a few seconds) and one
hop per strip per block. On Windows with Docker Desktop (the maintainer's headset over LSL,
`docs/TESTING.md` §X4), Sterile on: a delay of 4 ms (95 % of blocks) and 1
late block. Bringing it down (one hop per block for a chain of built-in
strips, a socket instead of Docker's stdin relay) is open.

## The image

```bash
meegxer sterile build     # once, and again after an upgrade
meegxer sterile check     # is it built from this MeegXer?
```

Docker (on Windows and macOS: Docker Desktop, started, *Engine running*) or
Podman must be running: an engine that is installed but not started is said as
such ("docker is installed but not running: start Docker Desktop…"), not as a
missing image.

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

## Custom scripts

A custom script (`docs/CUSTOM_SCRIPTS.md`) runs in a container of the
Sterile image of its own, with the same isolation, started like a plugin
(`python -m meegxer.plugins.sterile_script`). Nothing is mounted: the host
sends the script's code as the first frame on stdin (`spec/plugin-protocol.md`
→ *Sterile worker*): the file, or every file of a script folder (its helper
modules and data files), each checked by its SHA-256, and the container
serves it as
`meegxer.plugins.script` does outside Sterile mode, so the output is the
same. The code is read when the run starts, so an edited script needs no
image rebuild. After the run the strip is a local process again. A local
plugin that is a program of its own (`meegxer-plugin add --local`) cannot run
in the image: Sterile mode refuses it, saying so.

## For developers

The core reaches the Sterile code only through the `meegxer.isolation`
entry point (rule 9's import contract; `sterile`, `build` and `check`), which
`meegxer-plugins` provides: `meegxer.core.sterile.running(graph)` wraps a run,
`sterilize(graph)` gives the isolation to close. `MEEGXER_STERILE_IMAGE`
names another image tag (CI's `meegxer-sterile:ci`); `MEEGXER_STERILE_LOCAL=1`
runs the worker as a local process with no container, for tests (it needs
`MEEGXER_ALLOW_UNSANDBOXED=1`, as unsandboxed plugins do).
