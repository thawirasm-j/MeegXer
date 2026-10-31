# Overnight log (2026-10-31 to 11-01)

What Claude did and decided while the maintainer slept, for review in the
morning. Every default taken without asking is here, with the alternative it
passed over. This is a one-time note: delete it once reviewed (anything worth
keeping moves into `docs/ROADMAP.md`, `docs/DESIGN.md` or `CLAUDE.md`).

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
   spool header is rewritten whole, with an fsync). This PR.

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
