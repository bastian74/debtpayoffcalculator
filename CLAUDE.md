# DebtPayoffCalculator — single-file offline debt planner

One self-contained `index.html` (HTML/CSS/JS inline), offline, localStorage-backed. Avalanche/
snowball payoff simulation, savings drawdown, promo-rate handling, projection + consolidation
tabs (Chart.js).

## Critical invariants

- **`index-original.html` is the preserved original — NEVER modify it.** All work goes in
  `index.html`.
- Storage key is **`dfc6`**, with a migration path from the older **`df5`** schema. Any schema
  change MUST include a lossless migration and MUST NOT drop stored user data. `normalize()` is a
  never-throws repair pass (rewritten 2026-07-19) — keep it total; import is validate-then-commit
  with rollback so a corrupt import can't poison saved state.
- The payoff **engine math is verified correct to the cent** against a reference amortization
  (zero-APR, minimum-greater-than-balance, promo changes, card-charge routing all covered) — be
  careful not to regress it.

## Build & test

No build step. Tests are node-free and drive the REAL inline engine in headless Edge:

```
powershell -File tests\run-tests.ps1        # extracts + runs the engine, 20 tests as of 2026-07-19
```

`tests\tests.js` holds the cases. Verify online AND with the CDN blocked (Chart.js is
CDN-loaded — the app must degrade gracefully offline; vendoring Chart.js is the top deferred item).

## Gotchas / recent fixes (see AUDIT-2026-07-19.md)

- The "Sample" button overwrites real data — it now confirms first.
- Money is floating-point; an integer-cents engine is a deferred hardening item.
- Month arithmetic must not skip a month when opened on the 29th–31st.
- Most recent audit + deferred design sketches: **AUDIT-2026-07-19.md**.
