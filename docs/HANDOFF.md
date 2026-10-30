# Handoff (for the next Claude session, on any account)

What a new session needs and cannot get from the code: decisions the
maintainer made in conversation, how the work is done, what is open, and why
the last designs came out as they did. Read this after `CLAUDE.md`. Keep it
current: whoever finishes a PR that changes any of this updates it in the same
PR.

State as of main `34ad514` (2026-10-30), after PR #97 (review step 4a,
detectors). Plugin spec 0.2.20, control API 0.2.16 (from the recovery fix in
flight). In flight: the recovery timeout fix, then review step 4b.

## 1. Who works on what

Two Claude accounts may work on this repo. To keep them from colliding:

| Area | Owner | Branch |
|---|---|---|
| Main development (review track and the rest of `docs/ROADMAP.md`) | Lab account | its own `claude/…` branch |
| Recovery timeout fix, then review step 4b (suggestions queue) | Lab account | `claude/dreamy-hopper-wwm0sc` |
| PR review (`/code-review` on the other account's PRs), quick questions | Personal account | none (reviews only) |

Rules while both are active:

- **One branch per session.** Never push to another session's branch. A
  session that resets its branch to main after a merge must own that branch.
- **One area at a time.** Before starting an item, put it in the table above
  (or the *Open* list below) in the same PR. Do not start an item that touches
  the same files as one in flight (`batch.py`, `control.py`, `main_window.py`,
  `review_window.py`, `docs/REVIEW.md` change in most review-track PRs).
- **Version bumps** (plugin spec, control API, a node's manifest `version`):
  only on a branch based on the latest main, and rebase (merge main in) before
  the PR merges. If main took the number you used, take the next one.
- **Decisions live in the repo.** A decision the maintainer makes in one
  session goes into section 3 below (or the relevant design doc), or the other
  session will not know it.

## 2. How the work is done

- **One PR per item**, from the session's own branch, after the full local
  checks pass (`CLAUDE.md` → Commands): `uv run pytest` (GUI tests need
  `QT_QPA_PLATFORM=offscreen`), `ruff check`, `ruff format --check`, `mypy`,
  `mypy --platform win32`, `lint-imports`. Run `uv sync --all-packages` after
  adding an entry point (a new node), or the registry will not see it.
- **The maintainer merges.** Never merge, approve, or push to main.
- **Commits and PRs:** end with the attribution lines the session's system
  prompt gives. Never put a model name or ID in a commit, PR, code or doc.
- **After opening a PR:** watch it (PR activity subscription) and set one
  check-in about 50 minutes later; fix red CI and review comments on it.
- **After a merge:** start the next item from the latest main
  (`git fetch origin main && git checkout -B <branch> origin/main`).
- **Ask first** before changing an architecture rule (`CLAUDE.md`), before a
  method or architecture choice (give the alternatives briefly), and when the
  maintainer's intent is unclear. Do not refactor code the item does not need.
- **Docs in the same PR:** the design doc of the area (`docs/REVIEW.md`,
  `docs/RAILS.md`, …), `docs/RELEASE_NOTES.md` under *Next*, `spec/control-api.md`
  for a new or changed RPC (and its version line), `spec/session.md` when a
  session field changes.
- **Style:** match the surrounding code. Lines up to 88 characters; the code
  keeps hand-aligned literals with `# fmt: skip`. Comments and docstrings say
  what something is, in plain words.
- **GUI changes:** render a screenshot offscreen (`tools/gui_screenshot.py`, or
  a small script that builds the widget with fake data) and look at it in both
  themes before the PR.

## 3. The maintainer's standing decisions

Review window (`docs/REVIEW.md`):

- Display filters are **off** by default.
- The display is **never locked**. The Review strip's display settings are
  starting values, each *Reviewer's choice* by default; the reviewer can change
  everything.
- Clinical scale (7 µV/mm, 30 mm/s) as the default unit setting is fine.
- Channel names are rounded pills in the left margin: **click toggles bad**,
  right-click opens the channel menu.
- Marks are drawn by **dragging from start to end**, with edge tabs to adjust;
  a tab at the page border when the span runs past it.
- **Review points** (◀/▶ between a detector's candidates, each selected to
  label and retime) instead of page stepping when points are set.
- The review window "looks good for now": no layout changes unless asked.
- A review **creates events** (they reach the writers and the views).

Method choices:

- Montages and display filters are computed **in the core, from the raw
  samples** (`meegxer.core.montage`), never in the GUI.
- **Detectors:** built in only for well-known published methods (MNE
  `annotate_amplitude`, `annotate_muscle_zscore`, `find_eog_events`); anything
  else (spike, seizure, sleep stage) is a plugin. No detector of MeegXer's own
  invention.
- **Suggestions (review step 4b, decided 2026-10-30;** `docs/REVIEW.md` →
  *Step 4b: decisions*): a Review strip with priors takes those labels over
  (later strips see only its verdicts, never rejected suggestions: a new
  Events rail rule); verdicts are columns of the marks file
  (`suggestion_origin`, `suggestion_onset`, `suggestion_duration`,
  `verdict`), rejected rows kept but not emitted; priors are the existing
  `points` plus a `decision` setting (*free*/*verify*); a detector's proposed
  bad channels are asked about, not applied; headless runs stop while a
  suggestion has no verdict.
- **Detectors' outputs:** blinks are short spans (±0.25 s) labelled `blink`
  (not cut unless relabelled `BAD_blink`); the amplitude detector proposes
  bad channels as `bad_channel` events.
- Dependabot PRs: the maintainer merges them; check open Dependabot alerts now
  and then.

## 4. Open work (in order)

1. **Review step 4: suggestions** (`docs/REVIEW.md` → *Design v2*, *Order of
   work* 4), two PRs:
   - 4a, detector strips (amplitude/flat, muscle, blinks): built (PR #97).
   - 4b, the Review strip's priors (`points`) and `decision` mode (*free* or
     *verify*: a Y/N/E queue), verdicts kept with their origin, the Events
     rail rule that the Review strip takes its prior labels over, and the
     detector's proposed bad channels in the queue. Decided (section 3); not
     built. Review points (built) are its navigation half. Window (agreed
     2026-10-30): a *Suggestions* list under the marks list (label, time,
     origin, verdict); *verify*: Y accept, N reject, E edit (edited once its
     edges change), each moving on, free marking off; *free*: suggestions open
     as dashed marks (left alone: accepted; deleted: rejected; changed:
     edited).
2. **Small follow-up to epoch review:** rejected epochs are only in the
   sidecar and the reviewed epochs rail. Optionally put them on the Events rail
   (e.g. `BAD_epoch` spans) so file writers record them. Ask first.
3. **Review step 6:** a second reviewer, compare and resolve; a review report.
4. Then the rest of `docs/ROADMAP.md` (M10 research workflows onward).

## 5. Why recent designs came out as they did

- **Epoch review is its own strip** (`meegxer.core.epoch_review`, PR #93), not
  a mode of the Review strip. A strip's ports are fixed by its manifest, and a
  rail input with no earlier producer makes the pipeline invalid
  (`Graph._rail_source`). An optional epochs port on the Review strip would
  need optional rail ports, a plugin spec change. The Review epochs strip
  writes its **own** `epochs` rail; later readers take the nearest one
  (`docs/RAILS.md` rule 7), so the Epochs strip's rail is never rewritten
  (rule 8). Its epoch data stays in the core; the window fetches pages and
  ERP means with `offline_review_epochs`.
- **A review produces events only** (step 1). Cutting is the Reject segments
  strip's job, by label keyword; epoching is the Epochs strip's.
- **Review at import** (step 3) is saved per recording next to the file (or in
  its BIDS derivatives), never in the raw file, and is part of the input's
  identity, so changing it asks for refits downstream (rule 7).
- **Sidecars are BIDS events files** with a JSON sidecar holding the input
  fingerprint (`upstream`), reviewer and time. Same input: reused silently;
  changed input: asked again with the earlier answers pre-filled, never reused
  silently.
- **The review window keeps its own display cache slot**
  (`f"{probe}#review"` in `main_window.py`), so the main view and the review
  window do not evict each other's pages.
- **File writers get the Events rail** through `Sink.write_events`, mapped
  through cuts, so marks reach `.set`, EDF and BIDS outputs (PR #90).

## 6. Things that bit before

- GUI tests with the daemon: wait for the offline state to leave both
  `running` and `review` after submitting a review (the state stays `review`
  until the run thread picks the answer up).
- A GUI test pipeline gets the default notch; a 100 Hz recording fails it
  (50 Hz is Nyquist). Use 250 Hz in GUI test recordings.
- `QWidget` already has `move()`; do not name a method `move` on a window.
- EDF export adds a `BAD_ACQ_SKIP` annotation; `.set` export does not keep
  bad channels. Tests account for both.
- MNE `annotate_amplitude`'s "peak-to-peak" is the change between two
  consecutive samples, and it takes one `min_duration` for peak and flat
  (so the amplitude detector calls it once per limit). `find_eog_events`'
  `thresh` is a peak's prominence: around a lone synthetic bump, the filter's
  ringing counts as blinks at the default, so tests set it.
- A control call must answer within the client's timeout (the GUI waits
  3 s). Work that can take longer (writing a recovered recording) runs on a
  daemon thread and is reported by another call (`recoveries`), as offline
  runs are by `status`.
- ruff can reformat a hand-aligned block; fix the line or keep `# fmt: skip`
  rather than accepting a reflow of untouched code.

## 7. Starting a new session

Paste this as the first message of a new session on this repo:

```text
You're picking up MeegXer development. Before doing anything else, read
CLAUDE.md, then docs/HANDOFF.md (the maintainer's decisions, the work routine,
the open items and why recent designs came out as they did), then the design
doc for the item you take (docs/REVIEW.md for the review track).

Then, without changing any files yet:
1. Confirm main builds: uv sync --all-packages, then the checks in CLAUDE.md
   (pytest with QT_QPA_PLATFORM=offscreen, ruff, ruff format --check, mypy,
   mypy --platform win32, lint-imports). Report anything red.
2. Tell me in a few lines what you understand the next item to be (HANDOFF.md
   section 4, item 1 unless I say otherwise), how you would split it into
   PRs, and the questions you need answered first (HANDOFF.md says which
   decisions to ask about).

Wait for my answers before building. Follow HANDOFF.md section 2 for the
routine (one PR per item, I merge, checks before every push, docs in the same
PR) and section 1 if another session is working on the repo at the same time.
When a decision I make changes anything in HANDOFF.md, update it in that PR.
```
