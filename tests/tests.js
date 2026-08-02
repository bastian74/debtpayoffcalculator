// ============================================================
// DebtPayoffCalculator math-engine tests
// Engine-agnostic: expects the app's inline script to already be
// evaluated in the same scope (S, simulate, simConsolidated,
// migrateOld, normalize, defaultState, MAX_MONTHS available).
// Collects results into global __TEST_RESULTS (array of {name,pass,msg}).
// Run via tests/run-tests.ps1 (headless Edge) or node (see runner).
// ============================================================
var __TEST_RESULTS = [];
(function () {
  'use strict';
  function t(name, fn) {
    try { fn(); __TEST_RESULTS.push({ name: name, pass: true, msg: '' }); }
    catch (e) { __TEST_RESULTS.push({ name: name, pass: false, msg: String(e && e.message || e) }); }
  }
  function assert(cond, msg) { if (!cond) throw new Error(msg || 'assertion failed'); }
  function close(a, b, eps, msg) { if (Math.abs(a - b) > eps) throw new Error((msg || 'not close') + ': ' + a + ' vs ' + b); }

  // Reset global state to a clean baseline for each test
  function base() {
    S = defaultState();
    S.incomeMode = 'takehome';
    S.payFreq = 'monthly';
    S.incomeAmt = 10000;          // $10k/mo take-home: plenty of budget
    S.savings = { balance: 0, monthly: 0, efMonths: 3 };
    S.expenses = [];
    S.extra = 0;
    S.strategy = 'avalanche';
    return S;
  }
  function debt(o) {
    return Object.assign({ id: 'd' + Math.random().toString(36).slice(2, 8), name: 'D', type: 'credit', balance: 0, apr: 0, minPay: 0, color: '#ff6b6b', rateChanges: [] }, o);
  }

  // Independent reference amortization: month = interest accrual then payment
  function refAmort(balance, aprPct, pay) {
    var r = aprPct / 100 / 12, bal = balance, interest = 0, m = 0;
    while (bal > 0.005 && m < 2000) {
      var i = bal * r; interest += i; bal += i;
      bal -= Math.min(bal, pay); m++;
    }
    return { months: m, interest: interest };
  }

  // ---- T1: single-debt amortization matches independent reference ----
  t('T1 amortization 10000 @12% min 300', function () {
    base();
    S.debts = [debt({ balance: 10000, apr: 12, minPay: 300 })];
    var sim = simulate();
    var ref = refAmort(10000, 12, 300);
    assert(sim.allPaid, 'should pay off');
    close(sim.allPaidMonth, ref.months, 0, 'payoff month');
    close(sim.totalInterest, ref.interest, 0.01, 'total interest');
  });

  // ---- T2: zero-APR exact ----
  t('T2 zero APR 1000 min 100 -> 10 months, $0 interest', function () {
    base();
    S.debts = [debt({ balance: 1000, apr: 0, minPay: 100 })];
    var sim = simulate();
    assert(sim.allPaid, 'paid');
    assert(sim.allPaidMonth === 10, 'expected month 10, got ' + sim.allPaidMonth);
    assert(sim.totalInterest === 0, 'interest must be exactly 0, got ' + sim.totalInterest);
  });

  // ---- T3: minimum larger than balance ----
  t('T3 min > balance pays off first month', function () {
    base();
    S.debts = [debt({ balance: 500, apr: 12, minPay: 1000 })];
    var sim = simulate();
    assert(sim.allPaidMonth === 1, 'month 1, got ' + sim.allPaidMonth);
    close(sim.totalInterest, 5, 0.001, 'one month interest on 500 @1%/mo');
  });

  // ---- T4: strategy targeting ----
  t('T4 avalanche targets highest APR / snowball targets smallest balance', function () {
    base();
    S.extra = 100;
    S.debts = [
      debt({ id: 'big-lowapr', balance: 9000, apr: 5, minPay: 200 }),
      debt({ id: 'small-hiapr', balance: 1000, apr: 25, minPay: 50 })
    ];
    var av = simulate({ strategy: 'avalanche' });
    assert(av.snaps[0].target === 'small-hiapr', 'avalanche target');
    S.debts = [
      debt({ id: 'small-lowapr', balance: 1000, apr: 5, minPay: 50 }),
      debt({ id: 'big-hiapr', balance: 9000, apr: 25, minPay: 200 })
    ];
    var sn = simulate({ strategy: 'snowball' });
    assert(sn.snaps[0].target === 'small-lowapr', 'snowball target');
  });

  // ---- T5: rollover of freed minimums ----
  t('T5 freed minimum rolls into extra', function () {
    base();
    S.extra = 500;
    S.debts = [
      debt({ id: 'a', balance: 1000, apr: 0, minPay: 100 }),
      debt({ id: 'b', balance: 20000, apr: 0, minPay: 200 })
    ];
    var sim = simulate();
    var ev = sim.payoffEvents.find(function (e) { return e.id === 'a'; });
    assert(ev, 'debt a pays off');
    assert(ev.extraNow === 600, 'extra after rollover should be 500+100, got ' + ev.extraNow);
    assert(sim.allPaid, 'all paid');
  });

  // ---- T6: immediate breakdown when nothing can be paid ----
  t('T6 zero income + zero savings -> breakdown at month 0', function () {
    base();
    S.incomeAmt = 0;
    S.debts = [debt({ balance: 5000, apr: 10, minPay: 100 })];
    var sim = simulate();
    assert(sim.breakdownMonth === 0, 'breakdown month 0, got ' + sim.breakdownMonth);
  });

  // ---- T7: savings drawdown timing ----
  t('T7 savings absorb shortfall then plan breaks', function () {
    base();
    S.incomeAmt = 0;
    S.savings.balance = 500;   // covers 50/mo shortfall for 10 months
    S.debts = [debt({ balance: 100000, apr: 0, minPay: 50 })];
    var sim = simulate();
    assert(sim.breakdownMonth === 10, 'breakdown month 10, got ' + sim.breakdownMonth);
  });

  // ---- T8: pathological input terminates fast at MAX_MONTHS ----
  t('T8 giant balance terminates at cap quickly', function () {
    base();
    S.debts = [debt({ balance: 1e9, apr: 24, minPay: 10 })];
    var t0 = Date.now();
    var sim = simulate();
    var ms = Date.now() - t0;
    assert(!sim.allPaid, 'not paid');
    assert(sim.snaps.length <= MAX_MONTHS + 2, 'snaps bounded');
    assert(ms < 2000, 'simulate too slow: ' + ms + 'ms');
  });

  // ---- T9: no negative balances ever ----
  t('T9 snapshots never negative', function () {
    base();
    S.extra = 137.53;
    S.debts = [
      debt({ balance: 3333.33, apr: 19.99, minPay: 66.67 }),
      debt({ balance: 12345.67, apr: 6.5, minPay: 250 })
    ];
    var sim = simulate();
    sim.snaps.forEach(function (s) {
      Object.keys(s.bal).forEach(function (k) { assert(s.bal[k] >= 0, 'negative balance month ' + s.m); });
    });
    assert(sim.allPaid, 'paid');
  });

  // ---- T10: consolidation math ----
  t('T10 simConsolidated basics', function () {
    var a = simConsolidated(10000, 0, 500, 0);
    assert(a.paidOff && a.months === 20 && a.totalInterest === 0, '0% case: ' + JSON.stringify(a));
    var b = simConsolidated(10000, 12, 300, 0);
    var ref = refAmort(10000, 12, 300);
    assert(b.paidOff, '12% paid');
    close(b.months, ref.months, 1, 'months near ref');
    close(b.totalInterest, ref.interest, 1, 'interest near ref');
    var c = simConsolidated(10000, 12, 90, 0); // payment < monthly interest (100)
    assert(c.months === null && !c.paidOff, 'payment too low detected');
    var d = simConsolidated(10000, 12, 300, 24); // needs ~41mo, capped at 24
    assert(!d.paidOff, 'term cap prevents payoff');
  });

  // ---- T11: df5 migration fidelity ----
  t('T11 migrateOld maps old schema', function () {
    var old = {
      interval: 'biweekly', income: 2000, tax: 18, extra: 250, strategy: 'snowball',
      has401k: true, contribType: 'percent', contribAmt: 6, matchPct: 50, matchCap: 6, returnRate: 8,
      checkingBalance: 3000, emergencyFund: 100, otherSavings: 50, efMonths: 6,
      debts: [
        { id: 'x1', name: 'Card', type: 'credit', balance: 5000, apr: 22, minPayment: 100, color: '#123456', aprChanges: [{ date: '2027-01-01', apr: 28, retroactive: false }] },
        { id: 'x2', name: 'Loan', balance: 8000, apr: 6, minPayment: 150 }
      ],
      expenses: [
        { id: 'e1', cat: 'housing', name: 'Rent', amount: 1500, payAccount: 'checking' },
        { id: 'e2', cat: 'streaming', name: 'Netflix', amount: 15, payAccount: 'x1' },
        { id: 'e3', name: 'Mystery', amount: 20, payAccount: 'deleted-card' }
      ]
    };
    var n = migrateOld(JSON.parse(JSON.stringify(old)));
    assert(n.payFreq === 'biweekly', 'freq');
    assert(n.incomeMode === 'gross' && n.incomeAmt === 2000 && n.taxPct === 18, 'gross mode');
    assert(n.extra === 250 && n.strategy === 'snowball', 'plan');
    assert(n.k401.on === true && n.k401.amt === 6 && n.k401.returnPct === 8, '401k');
    assert(n.savings.balance === 3000 && n.savings.monthly === 150 && n.savings.efMonths === 6, 'savings');
    assert(n.debts.length === 2 && n.debts[0].minPay === 100 && n.debts[0].rateChanges.length === 1, 'debts');
    assert(n.debts[0].rateChanges[0].apr === 28, 'rate change apr');
    assert(n.expenses.length === 3, 'expenses');
    assert(n.expenses[0].paidWith === 'cash', 'checking->cash');
    assert(n.expenses[1].paidWith === 'x1', 'card link kept');
  });

  // ---- T12: migration of 0% tax must not become 22% ----
  t('T12 migrateOld preserves explicit 0% tax', function () {
    var n = migrateOld({ interval: 'monthly', income: 4000, tax: 0, debts: [], expenses: [] });
    assert(n.taxPct === 0, 'tax 0 preserved, got ' + n.taxPct);
  });

  // ---- T13: paycheck-stub migration ----
  t('T13 migrateOld stub-deduction mode -> takehome', function () {
    var n = migrateOld({ interval: 'weekly', income: 1000, fedTax: 120, stateTax: 40, fica: 62, medicareTax: 15, otherDeductions: 0, debts: [], expenses: [] });
    assert(n.incomeMode === 'takehome', 'takehome mode');
    close(n.incomeAmt, 1000 - 237, 0.001, 'net per check');
    assert(n.grossAnnual === 52000, 'gross annual');
  });

  // ---- T14: normalize survives corrupt shapes without throwing ----
  t('T14 normalize hardens corrupt state', function () {
    S = Object.assign(defaultState(), { debts: 'garbage', expenses: 42, k401: null, savings: 'x' });
    normalize();
    assert(Array.isArray(S.debts) && S.debts.length === 0, 'debts coerced to []');
    assert(Array.isArray(S.expenses) && S.expenses.length === 0, 'expenses coerced to []');
    assert(S.k401 && typeof S.k401.on === 'boolean', 'k401 rebuilt');
    assert(S.savings && typeof S.savings.balance === 'number', 'savings rebuilt');
    var sim = simulate();
    assert(sim.snaps.length >= 1, 'simulate runs on repaired state');
  });

  // ---- T15: normalize coerces bad field types & orphan paidWith ----
  t('T15 normalize coerces debt/expense fields', function () {
    base();
    S.debts = [{ id: 1, name: 5, balance: '1000', apr: '<img src=x onerror=alert(1)>', minPay: -50, color: '"><scr' + 'ipt>x</scr' + 'ipt>', rateChanges: 'nope' }, null, 'junk'];
    S.expenses = [{ id: 'e', name: 'Gym', amount: '40', paidWith: 'ghost-card' }, 7];
    normalize();
    assert(S.debts.length === 1, 'junk debts dropped, got ' + S.debts.length);
    var d = S.debts[0];
    assert(typeof d.name === 'string', 'name string');
    assert(d.balance === 1000, 'balance number');
    assert(typeof d.apr === 'number' && d.apr === 0, 'apr sanitized to number');
    assert(d.minPay === 0, 'negative minPay clamped');
    assert(/^#[0-9a-fA-F]{3,8}$/.test(d.color), 'color sanitized: ' + d.color);
    assert(Array.isArray(d.rateChanges), 'rateChanges array');
    assert(S.expenses.length === 1, 'junk expenses dropped');
    assert(S.expenses[0].paidWith === 'cash', 'orphan paidWith -> cash');
    assert(S.expenses[0].amount === 40, 'amount coerced');
    var sim = simulate();
    assert(sim.snaps.length >= 1, 'simulate runs');
  });

  // ---- T16: rate change in effect from start ----
  t('T16 past rate change applies immediately', function () {
    base();
    S.debts = [debt({ balance: 1000, apr: 0, minPay: 1010, rateChanges: [{ date: '2000-01-01', apr: 12 }] })];
    var sim = simulate();
    close(sim.totalInterest, 10, 0.001, 'one month at 1% despite base apr 0');
  });

  // ---- T17: card charges accrue on the card and flip to cash after payoff ----
  t('T17 expense charged to card increases balance until payoff', function () {
    base();
    S.debts = [debt({ id: 'cc', balance: 1000, apr: 0, minPay: 100 })];
    S.expenses = [{ id: 'e1', cat: 'other', name: 'Sub', amount: 50, paidWith: 'cc' }];
    var sim = simulate();
    // effective paydown 100-50=50/mo on 1000 -> 20 months
    assert(sim.allPaidMonth === 20, 'expected 20 months, got ' + sim.allPaidMonth);
    assert(sim.totalInterest === 0, 'no interest at 0%');
  });

  // ---- T18: extra respects mortgage exclusion toggle ----
  t('T18 extra skips mortgage unless enabled', function () {
    base();
    S.extra = 500;
    S.debts = [
      debt({ id: 'mort', type: 'mortgage', balance: 200000, apr: 6, minPay: 1200 }),
      debt({ id: 'cc', type: 'credit', balance: 2000, apr: 20, minPay: 40 })
    ];
    var off = simulate({ extraToMortgage: false });
    assert(off.snaps[0].target === 'cc', 'targets card when mortgage excluded');
    S.debts = [debt({ id: 'mort', type: 'mortgage', balance: 200000, apr: 6, minPay: 1200 })];
    var on = simulate({ extraToMortgage: true });
    assert(on.snaps[0].target === 'mort', 'targets mortgage when enabled');
  });

  // ---- T19: money never drifts below zero / interest is finite over 50yr ----
  t('T19 long simulation stays finite', function () {
    base();
    S.debts = [debt({ balance: 500000, apr: 199, minPay: 100 })]; // absurd APR, grows forever
    var sim = simulate();
    var last = sim.snaps[sim.snaps.length - 1];
    assert(isFinite(sim.totalInterest) || !sim.allPaid, 'interest finite or flagged');
    assert(isFinite(last.total), 'final total finite, got ' + last.total);
  });

  // ---- T20: import-style assign of hostile JSON must not corrupt S ----
  t('T20 defaultState+hostile assign then normalize is usable', function () {
    S = Object.assign(defaultState(), { incomeAmt: 'NaNny', payFreq: 'bogus', extra: -100, debts: [{ balance: 1e308, apr: 1e308, minPay: 0 }] });
    normalize();
    assert(typeof S.incomeAmt === 'number' && isFinite(S.incomeAmt), 'incomeAmt numeric');
    assert(FREQ[S.payFreq] !== undefined, 'payFreq valid, got ' + S.payFreq);
    assert(S.extra >= 0, 'extra clamped');
    var sim = simulate();
    var last = sim.snaps[sim.snaps.length - 1];
    assert(isFinite(last.total), 'no Infinity leak');
  });

  // ---- T21: clampNum guards every typed numeric input ----
  t('T21 clampNum clamps negatives, NaN, and absurd magnitudes', function () {
    assert(clampNum(-500, 0, 1e12) === 0, 'negative clamped to min');
    assert(clampNum('abc', 0, 100) === 0, 'NaN -> 0');
    assert(clampNum('abc', 5, 100) === 5, 'NaN with positive min -> min');
    assert(clampNum(1e15, 0, 1e12) === 1e12, 'huge value capped');
    assert(clampNum('12.5', 0, 100) === 12.5, 'normal parse intact');
    assert(clampNum('-3', 0, 60) === 0, 'typed negative string clamped');
  });

  // ---- T22: hostile imported debt id is sanitized and refs remapped ----
  t('T22 normalize sanitizes markup ids and remaps paidWith', function () {
    base();
    S.debts = [{ id: '"><img src=x onerror=alert(1)>', name: 'Evil', balance: 100, apr: 0, minPay: 10 }];
    S.expenses = [{ id: 'e1', name: 'Sub', amount: 10, paidWith: '"><img src=x onerror=alert(1)>' }];
    normalize();
    assert(/^[A-Za-z0-9_-]+$/.test(S.debts[0].id), 'id restricted to safe charset: ' + S.debts[0].id);
    assert(S.expenses[0].paidWith === S.debts[0].id, 'expense follows the replaced id');
  });

  // ---- T23: duplicate imported ids are deduped, refs follow the first ----
  t('T23 normalize dedupes ids', function () {
    base();
    S.debts = [
      { id: 'dup', name: 'A', balance: 100, apr: 0, minPay: 10 },
      { id: 'dup', name: 'B', balance: 200, apr: 0, minPay: 20 }
    ];
    S.expenses = [{ id: 'e1', name: 'x', amount: 5, paidWith: 'dup' }];
    normalize();
    assert(S.debts[0].id === 'dup', 'first keeps its id');
    assert(S.debts[1].id !== 'dup', 'second regenerated');
    assert(S.expenses[0].paidWith === 'dup', 'ref stays with the first');
  });

  // ---- T24: local date parsing + shared effective APR ----
  t('T24 parseLocalDate is local / effectiveApr applies past changes', function () {
    var dt = parseLocalDate('2026-03-01');
    assert(dt.getFullYear() === 2026 && dt.getMonth() === 2 && dt.getDate() === 1, 'YYYY-MM-DD parses as LOCAL Mar 1 (no UTC previous-day shift)');
    var d = { apr: 5, rateChanges: [{ date: '2000-01-01', apr: 20 }] };
    assert(effectiveApr(d) === 20, 'past rate change reflected in current APR');
    assert(effectiveApr(d, new Date(1999, 0, 1).getTime()) === 5, 'base APR before the change');
  });

  // ---- T25: 50-year cap is exactly 600 payment periods ----
  t('T25 boundary: 600 payments fit, 601 do not', function () {
    base();
    S.debts = [debt({ balance: 600, apr: 0, minPay: 1 })];
    var sim = simulate();
    assert(sim.allPaid, 'a plan needing exactly 600 payments must finish');
    assert(sim.allPaidMonth === 600, 'paid at month 600, got ' + sim.allPaidMonth);
    S.debts = [debt({ balance: 601, apr: 0, minPay: 1 })];
    sim = simulate();
    assert(!sim.allPaid, 'a 601st payment must NOT be processed');
  });

  // ---- T26: %-of-balance minimum matches independent reference ----
  t('T26 pct-mode declining minimum amortization', function () {
    base();
    S.debts = [debt({ balance: 1000, apr: 18, payMode: 'pct', payPct: 5, payFloor: 50, minPay: 0 })];
    var sim = simulate();
    var bal = 1000, interest = 0, m = 0;
    while (bal > 0.005 && m < 2000) {
      var i = bal * 0.18 / 12; interest += i; bal += i;
      bal -= Math.min(bal, Math.max(50, bal * 0.05)); m++;
    }
    assert(sim.allPaid, 'paid');
    assert(sim.allPaidMonth === m, 'months match ref: ' + sim.allPaidMonth + ' vs ' + m);
    close(sim.totalInterest, interest, 0.01, 'interest matches ref');
  });

  // ---- T27: pct-mode payoff frees only the floor ----
  t('T27 pct-mode rollover uses the floor', function () {
    base();
    S.extra = 100;
    S.debts = [
      debt({ id: 'p', balance: 200, apr: 0, payMode: 'pct', payPct: 10, payFloor: 40, minPay: 0 }),
      debt({ id: 'q', balance: 10000, apr: 0, minPay: 100 })
    ];
    var sim = simulate();
    var ev = sim.payoffEvents.find(function (e) { return e.id === 'p'; });
    assert(ev, 'p pays off');
    assert(ev.freedMin === 40, 'freed minimum is the floor, got ' + ev.freedMin);
  });

  // ---- T28: contractual term payment ----
  t('T28 requiredPayment annuity formula', function () {
    close(requiredPayment(10000, 0, 60), 10000 / 60, 0.001, '0% term payment');
    var p = requiredPayment(10000, 12, 60);
    close(p, 222.44, 0.05, '12%/60mo annuity payment');
    var r = simConsolidated(10000, 12, p + 0.01, 60);
    assert(r.paidOff, 'required payment repays within the term');
  });

  // ---- T29: sustainable surplus vs cash surplus ----
  t('T29 card-financed expenses reduce sustainable surplus', function () {
    base(); // $10k/mo take-home
    S.debts = [debt({ id: 'cc', balance: 1000, apr: 0, minPay: 100 })];
    S.expenses = [
      { id: 'e1', cat: 'other', name: 'Rent', amount: 2000, paidWith: 'cash' },
      { id: 'e2', cat: 'other', name: 'Food', amount: 500, paidWith: 'cc' }
    ];
    var C = calculate();
    close(C.freeCash, 7900, 0.001, 'cash surplus ignores card charges');
    close(C.cardExp, 500, 0.001, 'card expenses tracked');
    close(C.sustainable, 7400, 0.001, 'sustainable = cash surplus - card-financed spending');
    close(C.sustainableBase, 7400, 0.001, 'base equals sustainable when extra=0');
    // extra payment reduces `sustainable` but not `sustainableBase` — the base
    // decides whether the household is genuinely borrowing to live
    S.extra = 300;
    C = calculate();
    close(C.freeCash, 7600, 0.001, 'extra reduces cash surplus');
    close(C.sustainable, 7100, 0.001, 'extra reduces sustainable');
    close(C.sustainableBase, 7400, 0.001, 'base unaffected by voluntary extra');
  });

  // ---- T30: 401(k) advice compares against the match CAP, not match dollars ----
  t('T30 matchCapAmt fixes the hardship-advice comparison', function () {
    S = defaultState();
    S.incomeMode = 'gross'; S.payFreq = 'monthly'; S.incomeAmt = 6000; S.taxPct = 20;
    S.k401 = { on: true, type: 'percent', amt: 5, matchPct: 50, matchCap: 6, returnPct: 7 };
    S.debts = []; S.expenses = [];
    var C = calculate();
    close(C.m401, 300, 0.001, '5% of $6k gross');
    close(C.matchCapAmt, 360, 0.001, 'cap threshold = 6% of gross');
    assert(C.m401 > C.mMatch, 'sanity: the OLD buggy condition fires here');
    assert(!(C.m401 > C.matchCapAmt), 'new condition must NOT fire — every dollar is still match-eligible');
    S.k401.amt = 10;
    C = calculate();
    close(C.m401, 600, 0.001, '10% of $6k gross');
    assert(C.m401 > C.matchCapAmt, 'genuinely over the cap IS flagged');
    close(C.mMatch, 180, 0.001, 'match = 50% of the capped $360');
  });

  // ---- T31: consolidation comparison is engine-vs-engine, apples to apples ----
  t('T31 identical consolidated loan reproduces the current plan exactly', function () {
    base();
    S.debts = [debt({ id: 'orig', balance: 5000, apr: 10, minPay: 150 })];
    var cur = simulate();
    var swapped = [
      Object.assign({}, S.debts[0], { balance: 0 }),
      { id: 'consol-loan', name: 'Loan', type: 'other', balance: 5000, apr: 10, minPay: 150, color: '#fff', rateChanges: [] }
    ];
    var plan = simulate({ debts: swapped, extra: 0 });
    assert(plan.allPaid === cur.allPaid, 'same payoff outcome');
    assert(plan.allPaidMonth === cur.allPaidMonth, 'same payoff month: ' + plan.allPaidMonth + ' vs ' + cur.allPaidMonth);
    close(plan.totalInterest, cur.totalInterest, 0.01, 'same interest');
  });

  // ---- T32: payMode fields survive and are clamped by normalize ----
  t('T32 normalize coerces payMode/payPct/payFloor', function () {
    base();
    S.debts = [{ id: 'a', name: 'A', balance: 1000, apr: 5, minPay: 20, payMode: 'pct', payPct: 250, payFloor: -5 }];
    normalize();
    assert(S.debts[0].payMode === 'pct', 'pct mode kept');
    assert(S.debts[0].payPct === 100, 'payPct clamped to 100, got ' + S.debts[0].payPct);
    assert(S.debts[0].payFloor === 0, 'negative floor clamped');
    S.debts = [{ id: 'b', name: 'B', balance: 1000, apr: 5, minPay: 20, payMode: 'weird' }];
    normalize();
    assert(S.debts[0].payMode === 'fixed', 'unknown mode -> fixed');
  });
})();
