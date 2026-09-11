# DebtPayoffCalculator — single-file offline debt planner

One self-contained `index.html` (HTML/CSS/JS inline), offline, localStorage-backed. Avalanche/
snowball payoff simulation, savings drawdown, promo-rate handling, projection + consolidation
tabs (Chart.js).

## Durability: never lose work to a token cutoff

**Standing rule from Eric, 2026-09-11: "Be sure agents don't lose their work if they run out
of tokens. This should always be true everywhere, every project. Make things as non-volatile
as possible using c:\claude\ for scratch pad access if needed."**

DURABILITY: NEVER LOSE WORK — for any agent or workflow working in this repo:

- **Write findings down as you form them, not at the end.** Append each finding as one JSON
  line to `C:\Claude\_scratch\agent-work\DebtPayoffCalculator\<date>-<topic>\journal\<yourname>.jsonl`
  the moment you have it, before looking for the next. Your structured return value is then a
  second copy rather than the only one. This is not busywork: two review runs on another
  project died on a usage limit after ~2.7M tokens of reading and left nothing behind,
  because every agent was holding its results to report at the end.
- **If you are editing code, the repo is already durable** — but append a one-line note per
  completed sub-task to that journal, so an agent killed mid-batch leaves a trail of what it
  changed and what it was about to do. Afterwards `git diff` is ground truth; the journal
  supplies the intent.
- **Keep artifacts under `C:\Claude`, never `%TEMP%`.** The session scratchpad under
  `%TEMP%\claude\...` is volatile and has already lost probe sources between sessions.
  Durable home is `C:\Claude\_scratch\agent-work\<project>\<date>-<topic>\` with
  `baseline/ regress/ reports/ probes/ journal/`; see the README there. DevClean reclaims
  `bin`/`obj`/`build`/`.gradle` and named data dirs only, so sources and results survive —
  build output does not.
- **Probe and harness SOURCES belong in `C:\Claude\_scratch\agent-work\DebtPayoffCalculator\probes\`.**
  A probe you cannot re-run next session is a probe you will rewrite.
- **Prefer many small sequential batches to one long agent**, so a cutoff costs one batch.
  Workflows are resumable: relaunch with `{scriptPath, resumeFromRunId}` and read the run's
  `journal.jsonl` before assuming a cached result was non-empty.

## Critical invariants

- **`index-original.html` is the preserved original — NEVER modify it.** All work goes in
  `index.html`.
- Storage key is **`dfc6`**, with a migration path from the older **`df5`** schema. Any schema
  change MUST include a lossless migration and MUST NOT drop stored user data. `normalize()` is a
  never-throws repair pass (rewritten 2026-07-19) — keep it total; import is validate-then-commit
  with rollback so a corrupt import can't poison saved state. IDs are sanitized to `[A-Za-z0-9_-]`,
  deduped, and expense `paidWith` refs are remapped (they're interpolated into HTML attributes).
- The payoff **engine math is verified correct to the cent** against a reference amortization
  (zero-APR, minimum-greater-than-balance, promo changes, card-charge routing, %-of-balance
  minimums all covered) — be careful not to regress it. The 50-year cap is exactly **600 payment
  periods** (601 opening snapshots).
- All typed numeric input flows through `clampNum`/`num()` (clamps to the input's own min/max
  attributes) — HTML attributes alone don't stop typed negatives.
- Date-only strings (`YYYY-MM-DD` rate changes) MUST go through `parseLocalDate` — `new Date()`
  parses them as UTC and shows the previous day in US time zones. APR-dependent displays/sorts
  MUST use `effectiveApr(d)`, never raw `d.apr` (promo changes go stale otherwise).
- Cash honesty: `CALC.freeCash` is the cash surplus; `CALC.sustainable` subtracts card-financed
  living expenses. Headline "left over" figures use `sustainable` — a green number while the
  household borrows for groceries is the bug that motivated it.
- Consolidation compares **whole-plan engine sims at equal monthly outlay** (selected debts
  replaced by the loan, freed minimums redirected into extra, fee added to principal, card
  charges paid in full post-swap). "Savings" is only reported when both plans repay everything.

## Build & test

No build step. Tests are node-free and drive the REAL inline engine in headless Edge:

```
powershell -File tests\run-tests.ps1        # extracts + runs the engine, 32 tests as of 2026-08-02
```

`tests\tests.js` holds the cases. Verify online AND with the CDN blocked (Chart.js is
CDN-loaded — the app must degrade gracefully offline; vendoring Chart.js is the top deferred item).

## Gotchas / recent fixes (see AUDIT-2026-07-19.md)

- The "Sample" button overwrites real data — it now confirms first.
- Money is floating-point; an integer-cents engine is a deferred hardening item.
- Month arithmetic must not skip a month when opened on the 29th–31st.
- Most recent audit + deferred design sketches: **AUDIT-2026-07-19.md**.
