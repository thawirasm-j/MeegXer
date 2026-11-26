# Security: the red team's findings

MeegXer treats plugins, custom scripts (Python/MNE and Octave), shared
pipelines and shared sessions as untrusted (rule 9). This page is the record
of the red team run against Sterile mode and custom scripts on 2026-11-26
(decided that day: fix every finding, one PR each; the live-run decisions
below), and of each finding's fix. The attacks are kept as regression tests
(`packages/meegxer-plugins/tests/test_plugins_hostile.py`, with the
`hostile` fixture plugin).

| Finding | Status |
|---|---|
| M1 stderr can exhaust host memory | Fixed: lines cut at 4 KB (PR #177) |
| M2 output length and position unchecked | Fixed: output, rails and windows within the input given (PR #178) |
| M3 event contents unchecked | Fixed: labels, positions, counts and payloads checked (PR #179; `spec/plugin-protocol.md`) |
| M4 a hung plugin freezes a live run ~40 s | Open (decided: live, a block budget, no retry, the strip passes through) |
| M5 symlinks reach into the Sterile container | Fixed: a file linking outside its script folder is refused (this PR) |
| L1 no CPU cap | Open (decided: live runs, cores − 1, a setting) |
| L2 plugin text as rich text, links of any scheme | Open |
| L3 keyboard sink maps from a shared pipeline | Open (to decide) |
| L4 per-script EEGLAB plugins read as a boundary | Open (docs) |
| I1 `run`/`check --script` without the notice | Open (decided: the notice once per script version) |
| I2 declared packages join the shared Sterile image | Open (docs) |
| I3 Sterile does not isolate file parsing | Noted (architecture) |

2026-11-26 · main at 1b5787e (0.5.0b2 + M11.5) · Linux, Docker 29, the real
`meegxer-sterile` and `meegxer-octave` images · every result below was run,
not inferred.

**Threat model.** A custom script, a plugin image or an Octave script from
someone else (shared folder, template, paper's supplement) is hostile. It
wants to (1) escape the container or reach the network, (2) read the user's
files, (3) crash, freeze or exhaust the host (daemon/GUI), (4) corrupt
results silently, or (5) make a host sink act for it. Shared pipelines and
sessions are untrusted too.

**Verdict.** No escape and no network: the sandbox held against everything
tried. The weak side is the host's trust in what a plugin *sends back*
(sizes, positions, events, stderr) and two ways data reaches a container it
should not (symlinks). Nothing found is High.

## What held

| Attack | Result |
|---|---|
| Network from Sterile worker, Sterile script, Octave containers (TCP 1.1.1.1, docker gateway :2375, UDP, DNS, `/dev/tcp`) | only `lo`; all unreachable |
| Write to `/`, `/usr`, `/etc`, `/app`, `/opt/eeglab`, `/proc/sys`, `/sys` | read-only |
| Privileges: capabilities, setuid (`su`), `mount`, `/proc/1/mem` | CapEff 0, NoNewPrivs 1, seccomp on, uid 10001; all refused |
| Docker socket, host folders, host environment | none mounted, none passed |
| 400 processes; 3 GiB in a 2 GiB container; 200 MB in `/tmp`, `/dev/shm` | stopped at 255; OOM-killed; 64 MB caps |
| Host wire decoder: dtypes, pickle, header/payload sizes, bad descriptors | allow-list, no pickle, 16 MB / 256 MB caps, checked |
| Reply with wrong channel count, sfreq 0 or < 0, duplicate labels, negative event index | refused |
| Unsolicited reply flood; reply with another id | refused at the first |
| Plugin that never answers | killed after the timeout |
| Code smuggled by a shared pipeline or session | not possible: launchers come only from the user's own `plugins.json`; sessions record, never launch |
| Sterile with a missing or stale image, or a non-script local plugin | refused (fails closed) |
| `../`, absolute or `\` paths in a Sterile script's files | refused |
| Shell injection through `% Requires:` | refused (fixed in #175) |
| Control API from another user | token in a 0600 file |

## Findings

### M1 · A plugin's stderr can exhaust host memory · Medium
`PluginWorker._read_stderr` uses `readline()` with no limit. A plugin wrote
600 MB to stderr without a newline: the host held it as one line and put it
whole into the error (310 MB of text; in the GUI it would go to the log).
Repeatable at will. **Fix:** read stderr in chunks, cut each line at ~4 KB.

### M2 · Output length and position are not checked · Medium
A plugin returned 100× the samples it was given (accepted; the host's
memory grew 4× on one block), and one put its output at sample 10¹⁵
(accepted). Repeated per block this amplifies memory without bound and
breaks rule 8 (alignment) silently. **Fix:** refuse a result whose end lies
past the input block's end (mapped by the exact rate ratio) or before 0.

### M3 · Event contents are not checked · Medium
Accepted from a plugin: a label that is a dict, a 15 MB label, 300,000 events
in one 1 s block, an event at sample 10¹². An unknown key crashed with a raw
`TypeError` (not a plugin error). Events reach writers (events.tsv, .set),
the GUI and the keyboard sink. **Fix:** label a string (≤ 256 chars),
duration ≥ 0, payload JSON scalars, index within the block (plus what a
lagging plugin may hold), at most a few events per sample; anything else is
a `PluginError` naming the plugin.

### M4 · A hung plugin freezes a live run for ~40 s · Medium
The timeout (10 s) is retried with restarts: a plugin that stops answering
froze the whole chain for 44 s, every strip and output included. For a BCI
this is a 40 s outage from one bad script. **Fix (decision):** live runs
give a script its block budget (e.g. 2 s) and no retry; the strip errors
and the run says so (or passes through).

### M5 · Symlinks in a script folder reach into the Sterile container · Medium
`folder_files` follows symlinked files: a folder with
`notes.txt -> ~/.ssh/id_rsa` sent the key into the "isolated" container (and
hashed it). A shared folder (a zip with relative links) can pull a known
secret path into code that can then encode it in its outputs (events,
derivatives the user shares). Symlinked folders were not followed. **Fix:**
refuse symlinks that resolve outside the folder.

### L1 · No CPU cap: one script takes every core · Low
A Sterile script spun 16 processes: 400% CPU (all 4 cores). The host's 2 ms
timer: p99 0.23 → 1.33 ms, worst 0.27 → 4 ms. Live timing degrades; on a Pi
it would be worse. **Fix (decision):** `--cpus` default (cores − 1) for
containers in live runs, a setting to change it.

### L2 · Plugin text shown as rich text; links of any scheme · Low
A plugin's manifest description is put in a `QLabel` (auto rich text), and
reference URLs become links opened with `setOpenExternalLinks` without a
scheme check (`file:`, custom protocol handlers open with the OS's
default). Error dialogs show plugin stderr the same way. Spoofing, or one
click to launch a local handler. **Fix:** plain text for plugin text; links
`https:`/`http:` (and DOIs) only.

### L3 · Keyboard sink maps from a shared pipeline · Low
The map allows letters, digits, space and enter (no modifiers), ≤ 20/s, and
only after arming by hand. A shared pipeline mapping all of them lets any
plugin, even a Sterile one, type words and Enter into the focused window
once armed. **Fix (decision):** show the map when arming; ask again when
it includes enter with letters.

### L4 · "Each script sees only its own EEGLAB plugins" is not a boundary · Low (docs)
An Octave script that declares nothing ran `addpath('/opt/eeglab/plugins/clean_rawdata')`
and `clean_drifts`; `system('id')` works (inside the sandbox). Harmless —
every plugin is equally sandboxed — but `docs/OCTAVE.md` reads like a
security property. **Fix:** say it is to keep scripts from clashing.

### I1 · `meegxer-plugin run|check --script` run Python without the notice · Info
`add` asks for the local-code notice; `run` and `check` start the script as a
local process with full user rights without it. Explicit commands, but
"try it first" is the advertised first step for a downloaded script.
**Fix (decision):** the notice once per script hash, or `run --sterile`.

### I2 · Declared packages join the shared Sterile image · Info
`REQUIRES` packages are installed into the one Sterile image (a package's
`.pth` runs in every Sterile container, the built-in worker included);
pip may build sdists (code runs at build time, with network). They must be
installed on the host first, so they are already trusted there.
**Fix:** say so in `docs/STERILE.md`; optionally `--only-binary :all:`.

### I3 · Sterile does not isolate file parsing · Info (architecture)
Recordings are read on the host (rule 10: sources are trusted I/O), so a
crafted EDF/.set/XDF targets MNE/pymatreader/pyxdf on the host, outside
Sterile. **Fix (decision, architecture):** document; later, an option to
parse files in the worker.

## Reproduce
The host-side attacks are the `hostile` fixture plugin's modes, run by
`test_plugins_hostile.py`. The container probes were run with the exact
flags `ContainerLauncher.command` builds (`--network none --read-only
--tmpfs /tmp:rw,size=64m --cap-drop ALL --security-opt no-new-privileges
--pids-limit 256 --memory …`).
