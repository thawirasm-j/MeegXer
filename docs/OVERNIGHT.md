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
   two reviewers), the roadmap status and §J/§K marked passed: this PR.
   **Your decision:** tag `v0.4.0b3` (I can't push tags).

## Defaults taken (small choices)

| # | Where | Chosen | Passed over |
|---|---|---|---|

## Waiting for you

- Tag `v0.4.0b3` once this PR is merged.
