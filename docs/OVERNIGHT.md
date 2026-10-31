# Overnight log (2026-10-31 to 11-01)

What Claude did and decided while the maintainer slept, for review in the
morning. Every default taken without asking is here, with the alternative it
passed over. This is a one-time note: delete it once reviewed (anything worth
keeping moves into `docs/ROADMAP.md`, `docs/DESIGN.md` or `CLAUDE.md`).

## In short

- **The BCI path is done** (your order): LDA classifier (#143), keyboard sink
  (#145), SSVEP BCI template with an SSVEP → keypress demo (#146), plus M11.6
  shortcuts (#147) and M11.2 plugin hardening (the last PR). A pre-existing
  bug was found and fixed
  on the way: live recordings had no events (#144).
- **M11's exit criterion is half met.** The SSVEP → keypress demo runs (§N).
  Sterile mode (M11.1) needs Docker or Podman, and the Octave tier (M11.5)
  needs GNU Octave; neither is in this container, so I stopped there.
- **For you:**
  - tag `v0.4.0b3` (#142);
  - the manual checks only your machine can do: L4, M4, M5, N4, O2 (each
    listed below and in `docs/TESTING.md`);
  - review the 27 defaults below.
  - Once those checks pass, I'd propose 0.5.0b1 ("BCI") with its release
    notes in a PR.

## Your decisions (asked before you left)

| Question | Your answer |
|---|---|
| M11 order | BCI path first: LDA classifier, keyboard sink, the SSVEP template and demo; Sterile mode later (it needs Docker, which this container lacks) |
| LDA method | Shrinkage LDA: scikit-learn `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` (Ledoit-Wolf; Blankertz et al. 2011), with a setting to turn shrinkage off |
| Keyboard sink | Off until armed by hand each session (GUI switch or CLI `--arm`), at most N keys per second (default 2), a visible stop (Esc, the GUI's stop), a fixed map of event label to key, never armed in a batch run; pynput |
| Overnight | Take the recommended default for small choices, list each here and in its PR, auto-merge as usual; big ones (new dependencies beyond these, architecture rules) wait for you |

## Done

1. **§J and §K run offscreen** (the real window, `QT_QPA_PLATFORM=offscreen`,
   `tools/review_demo.py`): all passed. J: viewed 100 %, marks
   "BAD: 1 (1.5 s); blink: 2 (1.0 s)" in the QC report's Review section. K:
   reviewer A refused as the second reviewer, B reviewed blind, the resolve
   showed 1 disagreement; raters "A, B", kappa "BAD: 0.65; blink: 1.00".
   What only you can check: how the two-reviewer header and the report look
   on your machine. PR #141 (merged).
2. **0.4.0b3 release notes** ("Editing and review": undo, the review report,
   two reviewers), the roadmap status and §J/§K marked passed. PR #142
   (merged). **Your decision:** tag `v0.4.0b3` (I can't push tags).
3. **M11.3 LDA classifier**: the *LDA* strip reads a feature strip (CSP, band
   power…) and gives class probabilities plus `pred_<class>` events. Offline
   and live. §L run by me: the CLI run, the GUI through the real window
   (offscreen), and live through the daemon (`calibrate csp1`, then
   `calibrate lda1`; both ready, p95 latency 3.6 ms). Left for you: L4 (real
   EEGBCI data). PR #143.
4. **Fix: live recordings keep their events.** The live scheduler keeps an
   untrimmed copy of the Events rail and hands each writer what it sees,
   every 2 s while running (so a crash recovery keeps most of it) and in
   full before closing. Default taken: **2 s** between header rewrites (the
   spool header is rewritten whole, with an fsync). PR (queued after #143).
5. **M11.4 keyboard sink**, as you decided: off until armed by hand for one
   live run (*Arm* in its card, or `meegxer arm NODE`); a fixed map of event
   label → key (`pred_left=left, pred_right=right`); at most 2 keys/s; Esc,
   *Disarm* and *Stop* disarm it; never armed in a batch run, nor from a
   pipeline or session; pynput. "Not a medical device" in the README and a
   new About box (*?*, top right). §M run by me on a virtual X display
   (Xvfb) with real pynput. Left for you: M4 (keys landing in another app on
   Windows) and M5 (how the About box looks). PR #145.
6. **SSVEP BCI template and keypress demo** (M10.9's last part; half of
   M11's exit criterion): the SSVEP detection strip can put each window's
   detected target on the Events rail (`ssvep_12`), a live *SSVEP BCI (keys)*
   template (LSL → SSVEP detection → keyboard), and `tools/ssvep_demo.py`, a
   synthetic SSVEP recording played live. §N run by me on Xvfb with real key
   presses (detections 100 % correct on the synthetic looks). Also fixed on
   the way: the feature view drew SSVEP's unitless scores in dB. PR #146.
7. **M11.6 shortcuts**: `meegxer-gui --create-shortcut` / `--remove-shortcut`
   (Windows Start menu and desktop; Linux application menu; macOS
   `~/Applications`). Run by me on Linux; Windows and macOS only with the
   system calls faked. Left for you: O2 (Windows, no console window). PR
   #147.
8. **M11.2 plugin hardening**: the container limits and digest pinning
   were already in place. Added: the plugin wire reader checks every size
   and array descriptor a plugin sends before reading (payload cap 256 MiB,
   read in 1 MiB pieces; a bad frame is a protocol error, not a memory
   blow-up), and the GUI drops display frames it cannot decode instead of
   losing its display thread. Fuzzed with seeded mutations (no new
   dependency). This PR (the last one tonight).

## Defaults taken (small choices)

| # | Where | Chosen | Passed over |
|---|---|---|---|
| 1 | How LDA gets its features | Its **main input is the feature strip** (*Reads*), as DESIGN.md and RAILS.md planned ("Features → Processor later"; "a reader takes the nearest rail of its input type"). Spec 0.2.23. The schedulers hand it `Windows`: the windows that ended in a stretch of the producer's input, so no second rate anywhere. | LDA on the signal with an extra Features *rail port*: it reuses the rail machinery, but needs a new way to name which feature strip, and doesn't match the docs |
| 2 | Which windows train LDA | Windows lying **wholly** from *Trial start* to *Trial end* after a class cue (defaults 0.5–4 s); windows on two classes' trials are left out | Window *end* in the trial: more windows, but they would include pre-cue data |
| 3 | When LDA emits a class event | One per window, at its **last sample** (when the decision exists), named **`pred_<class>`** so it never mixes with the cues on the rail; *Event only from probability* (default 0: every window) and the prefix are MeegXer's settings, live (no refit) | Events named exactly as the class: a keyboard sink mapping `left` would also fire on the cue markers |
| 4 | Parameter names | `min_probability` and `event_prefix`, not `threshold`/`prefix`: the GUI's rule is "the same name means the same thing on every node", and `threshold` is already "Removal threshold" | |
| 5 | Who sees LDA's events | **An output strip (sink) sees every event on the rail to its left**, as the chain draws it, so a `.set` writer after LDA records `pred_` events. A **processor still sees only its ancestors'** events: its fingerprint names them, so a changed event marks it for a refit (rule 7) | Every strip sees everything to its left: would break rule 7 (a CSP after LDA could fit on predicted events without its fingerprint knowing) |
| 6 | Plugins with a Features input | Refused with a clear message (built-in only for now) | Extending the plugin protocol: a bigger change, not needed for M11 |
| 7 | The demo data | `tools/lda_demo.py`: synthetic motor imagery (mu ERD over the hemisphere opposite the hand), so §L needs no download | The EEGBCI sample: real, but needs the download (left to you as L4) |
| 9 | How the keyboard gets events | A new core `ActionSink` base: the live scheduler hands it each new event it sees (the rail to its left) right after each block; file writers keep the 2 s cadence | Giving it the writers' 2 s batches: too slow for a BCI |
| 10 | A played recording's own cues | Not acted on (they all reach the rail at the start, ahead of time); only events strips emit live, and live LSL markers, are | Acting on them when their sample is reached: more code, and pressing keys for cue markers is not the use |
| 11 | Arming | Control API `arm` (0.2.24) and CLI `meegxer arm NODE [--off]`; refused unless a live run is running; the run's end disarms | `meegxer start --arm NODE`: arming before data flows |
| 12 | Rate limit | A minimum gap of 1/rate between presses (default 2/s → 0.5 s); extra events dropped and counted (`dropped` in the status and the card) | A 1 s sliding window: wrong below 1 key/s |
| 13 | Keys it may press | One letter or digit, or a named key (arrows, space, enter, tab, backspace, home, end, page up/down, F1–F12); no modifiers (Ctrl, Alt, Win) | Any key: a mapped Ctrl or Win could do real harm |
| 14 | Armed state on screen | The strip turns red with "ARMED · Esc disarms"; the card shows pressed/dropped | A modal warning on arming |
| 15 | About box | A *?* button beside ⚙ (there was no About box) | A menu bar |
| 16 | pynput's install | It builds `evdev` from source on Linux (needs a C compiler and kernel headers; fine on CI and here; may matter on a Pi). Left out of `tools/smoke_deps.py`, because importing it needs a desktop session | |
| 17 | SSVEP decisions as events | An option on the SSVEP detection strip, **off by default** (*Detections as events*): the window's best target, the published decision rule (Lin 2006; Chen 2015), at its last sample, `ssvep_<Hz>`; *Event only from score* (default 0) and the prefix are MeegXer's | An LDA after the scores: needs calibration, and CCA's point is that it needs none |
| 18 | Parameter name | `detections` (the Epochs strip already uses `events` for something else) | |
| 19 | SSVEP template | **Live from LSL**, no sample dataset (templates may now have none; the dialog hides the sample checkbox, `--sample` is refused) | MNE's SSVEP dataset as a sample: I could not confirm its citation and licence, which the BIDS sample needs |
| 20 | Template settings | Targets 8, 10, 12, 15 Hz, filter-bank CCA, 2 s windows every 0.5 s; keys arrows (8 left, 10 right, 12 up, 15 down); no channel picking (the description says to add Pick channels for the occipital ones) | |
| 21 | SSVEP node version | 0.1.0 → 0.2.0 (new settings; same output with them off) | |
| 22 | Shortcut target | This Python with `-m meegxer.gui.app` (`pythonw.exe` on Windows: no console), so it starts the MeegXer it was made from | The `meegxer-gui` script: a console window opens behind it on Windows |
| 23 | Windows `.lnk` | Written by PowerShell's `WScript.Shell` (built into Windows) | pywin32: a new dependency |
| 24 | Icon | None yet (the system's default); worth designing one | A placeholder icon |
| 25 | Doing M11.2 tonight | Only the parts that need no Docker: the decoders and their fuzzing (the limits and pinning existed) | Waiting for Sterile mode |
| 26 | Payload cap | 256 MiB per frame (a 64 ch × 10 s float32 block is 2.5 MB; models are far smaller) | 1 GiB |
| 27 | Fuzzing | Seeded random mutations in plain pytest (thousands per run, < 1 s) | Hypothesis: a new dev dependency |
| 8 | References | Fisher 1936, Blankertz et al. 2011, Ledoit & Wolf 2004, scikit-learn (Pedregosa et al. 2011). DOIs checked by web search (doi.org is blocked from here) | |

## Found on the way

- **Live file writers recorded no events at all** (not the cues, not
  markers, not LDA's): the live scheduler never gave sinks the Events rail.
  This predates tonight; offline runs were fine. Fixed in its own PR (4
  below).

## Waiting for you

- Tag `v0.4.0b3` (#142 is merged).
- L4: the LDA strip on real motor-imagery data (EEGBCI sample, *Classes*
  `T1, T2`).
- M4/M5: the keyboard on Windows (keys landing in Notepad), the About box.
- N4: the SSVEP demo's keys on Windows.
- O2: the shortcuts on Windows.
- An icon for MeegXer (shortcuts and the window use the system default).
