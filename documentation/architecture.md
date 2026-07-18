# Vega — Architecture

> Stub, grows one section per shipped work item. Source of truth for product intent: [STRATEGY.md](../STRATEGY.md).

## The one structural rule

**Deterministic quant core; LLM at the edges.** Signals, backtests, sizing, exits, and
portfolio math live in deterministic Python under `src/vega/`. The LLM synthesizes prose
(briefings, theses) exclusively from numbers the engine hands it — it never produces a
price, percentage, or statistic from memory.

## Module map (grows per work item)

| Module | Work item | Status |
|---|---|---|
| `src/vega/` package root, tooling, verify gate | WI-057 | shipped |
| `src/vega/data/` — sources, snapshots, validation, universe | WI-058 | shipped |
| `src/vega/regime/` — regime state + macro/earnings calendar | WI-059 | shipped |
| `src/vega/ledger/` — append-only ledger + override log | WI-060 | shipped |
| `src/vega/execution/` — paper executor + slippage-haircut P&L | WI-061 | shipped |
| `src/vega/briefing/` — deterministic pre-market briefing v1 | WI-062 | shipped (5-day gate running) |
| `src/vega/backtest/` — walk-forward engine + backtest registry | WI-063 | shipped |
| `src/vega/risk/` — sizing, portfolio heat, exit-spec writer | WI-064 | shipped |
| `src/vega/lifecycle/` — signal promotion state machine | WI-065 | shipped |
| `src/vega/signals/` — first 3 candidate signal families | WI-066 | shipped (1 paper-live, 1 held, 1 retired) |
| `src/vega/briefing/calls.py` — ranked calls, entries only | WI-067 | shipped |
| `src/vega/execution/exits.py`, `lifecycle/live_trades.py` — exit monitor + auto-demotion | WI-087 | shipped |

## Data layer (WI-058)

- **Adapters** (`src/vega/data/sources/`): yfinance = canonical equity bars (consolidated
  volume); Alpaca IEX = equity close cross-check only (its volume is never consumed);
  Binance = canonical crypto bars; CoinGecko = crypto close cross-check. Class-share
  notation is normalized to Yahoo style (`BF-B`) at the adapter boundary.
- **Snapshots** (`snapshot.py`): raw payloads are append-only under `data/snapshots/`;
  validated output is write-once per data date under `data/clean/` (identical rewrite =
  no-op, drifted rewrite = `SnapshotConflictError`). DuckDB views `bars` / `quarantine`
  over the clean tree are the ONLY read path for downstream consumers.
- **Validation** (`validate.py`): per (symbol, date) close reconciliation, default
  tolerance 0.5%; breaches and cross-check gaps are quarantined with a reason.
- **Universe** (`universe.py` + `data/universe/universe-v1.csv`, committed): S&P 500 +
  Nasdaq-100 + 30 ETFs + top-20 crypto, $20M median-dollar-volume filter, versions
  append-only via `scripts/refresh_universe.py`.
- **Incremental ingest** (`ingest.py`, fixed during WI-063's backfill): vendors revise
  historical values retroactively — yfinance restates past adjusted-close when a new
  dividend is declared — so a wider re-ingest routinely sees different content for dates
  already in the clean store. That's expected drift, not corruption; `has_clean()` lets
  `_write_result` skip dates already written instead of raising `SnapshotConflictError`,
  so the frozen store stays frozen and only genuinely new dates are added.

## Regime & calendar (WI-059)

- `regime.py` is a pure function over stored inputs → `RegimeState` (trend via SPY vs
  200DMA, VIX bands, universe breadth vs own 200DMA, crypto fear/greed). Composite is
  conservative: any red component degrades to caution; broken trend or crisis VIX = risk_off.
- `inputs.py` fetches+snapshots ^VIX (yfinance, single-source, labeled) and alternative.me
  fear/greed before any computation — regime only ever reads stored data.
- `calendar.py`: committed versioned macro artifact (`data/calendar/macro-v1.csv`, FOMC +
  CPI 2026 from official sources) + on-demand per-symbol earnings via yfinance.
  `in_macro_window()` implements the no-entries-before-FOMC/CPI gate.
- Zero-signup decision (Jose, 2026-07-05): no FRED/Finnhub keys; keyless equivalents.

## Ledger (WI-060)

- `types.py`: `Recommendation` enforces the full contract at construction — the four-part
  exit spec (stop, time stop, profit rule, invalidation) is mandatory; a long without
  signal attribution cannot be instantiated.
- `store.py`: append-only JSONL with fsync per write (Caral audit-log pattern). No
  update/delete API exists; corrections append with `supersedes`, human deviations are
  `override` records linked to the original call. Runtime state under gitignored `data/ledger/`.

## Execution & briefing (WI-061, WI-062)

- `execution/executor.py`: pending ledger longs → Alpaca paper market orders behind an
  injectable `TradingBackend` protocol (tests run offline against a fake). Fills append
  to the ledger linked by `ref_id` — the recommendation is never mutated. Failures go to
  an append-only log the briefing surfaces; one bad order never stops the batch.
  Sizing = fixed $1,000 notional until WI-064's risk engine replaces the caller.
- `execution/pnl.py`: all paper P&L reported through a slippage haircut
  (10 bps/side equities, 30 bps/side crypto) — paper fills are never taken at face value.
- `briefing/`: pure deterministic template (no LLM in v1 — evidence integrity by
  construction). Assembles regime + movers + macro events + execution failures from the
  clean store, renders write-once markdown to `data/briefings/{date}.md` with a
  data-provenance footer. Daily run: `uv run python -m vega.briefing` (after the ingest).

## Backtest engine (WI-063) — the highest-stakes module in the codebase

A subtle bug here silently invalidates every accuracy claim Vega will ever make. Every
anti-self-deception mechanism is structural, not disciplinary:

- `market_view.py`: a `MarketView` filters to `date <= as_of` on every read — a signal
  physically cannot see a future row. Signals receive ONLY a `MarketView`, never a frame.
- `simulate.py`: decisions at the close of session T fill at T+1's **open** (no same-bar-close
  fills exist). Stops are gap-aware (open-through-stop fills at the open, never the stop
  price). Signal trend logic uses `adj_close`; stop distance (ATR) and all fills use raw
  OHLC — the space fills actually happen in. Costs (`costs.py`) are applied inside the one
  fill function every trade passes through, deliberately calibrated at or above the live
  paper-execution haircuts (`execution/pnl.py`) so a backtest is always the pessimistic estimate.
- `folds.py`: a locked 20%-recent holdout plus expanding-window walk-forward folds on the
  remaining dev segment (~quarterly test slices).
- `registry.py`: append-only (built on `vega.common.appendlog`, shared with the ledger).
  An unregistered backtest cannot promote. The promotion Sharpe bar rises with cumulative
  param-grid points tried per signal family (`0.8 + 0.1·log10(trials)`) — a crude,
  auditable stand-in for proper multiple-testing correction. Holdout touches are counted
  per family and flagged if a family burns its holdout more than once.
- `engine.py`: orchestrates dev walk-forward → verdict (`insufficient_sample` below 30
  closed trades, `pass`/`fail` against the promotion bar and a 1.5× benchmark-drawdown
  cap, or `non_promotable_placeholder` for fixture signals) → holdout run **only** when
  the dev verdict is `pass` for a promotable signal. Live smoke test (`SmaCrossSignal`,
  `promotable=False`): 6 dev folds, 3,446 trades, verdict `non_promotable_placeholder`,
  holdout never touched — proves the pipeline without ever burning the holdout on a fixture.
- `vega.common.appendlog`: the shared append-only+fsync primitive `ledger/store.py` was
  refactored onto (behavior-preserving) so the registry and the ledger share one audited
  I/O path instead of two copies of the same durability logic.

## Risk engine (WI-064) — the second highest-stakes module

Sizing and exits determine P&L more than entries do. Pure math over caller-supplied
data (no network, no clock — enforced, not aspirational); the single writer of exit
specs consumed by the ledger, the backtester, and briefing v2.

- **`vega/common/doctrine.py` — the ONE definition of the exit doctrine** (stop/gap
  multiples, profit rule, time-stop sessions). Both the live engine and
  `backtest/signals.py`'s `EntryProposal` defaults import it; a test asserts the
  identity so the two authorities structurally cannot drift (review finding: they
  previously matched by hand-copied coincidence). `vega.common.atr` likewise supplies
  the one ATR implementation.
- `sizing.py`: gap-stressed sizing — the `min(base, gap)` formula algebraically
  guarantees nominal risk ≤1R and gap-stressed worst case ≤2R for any positive k/G
  (live smoke: exactly 2.00R for both sleeves). There is deliberately NO extra clamp:
  a review found the prior 1.5R clamp was unreachable dead code with a misleading
  test — false safety in a safety module. **Price space: entry_ref_price and ATR are
  RAW prices — the space fills happen in** (signals decide on adj_close; risk doesn't).
- `clusters.py`: dumb bucket assignment; the one real correlation (90-day crypto-vs-SPY,
  >0.5 → 50% of R contaminates `us_equity_beta`) is computed on **merged-then-tailed
  shared sessions** (crypto's 7-day calendar vs equities' 5-day would otherwise
  under-fire it), and a **SPY-less frame raises loudly** — a review found the rule was
  dead on arrival because the only caller source-filtered SPY away. Cluster frozensets
  are guarded by a test against the committed universe (cluster column on the
  artifact = parked universe-v2 migration).
- `heat.py`: heat floored at 0 (a trailed stop frees heat); caps 6R total (3R in
  `caution` regime), 4R equity-beta, 2.5R crypto-beta, 3R rates/commodities.
  `SizedProposal.heat_after_r` reports **R multiples** (directly comparable to the
  caps); dollar heat is engine-internal.
- `gates.py`: regime `risk_off` blocks all entries; macro T-1/T FOMC-CPI blocks both
  sleeves; earnings consume a **caller-supplied `EarningsFact`** (the network lookup
  runs outside the engine, crypto never hits the vendor) and **fail CLOSED** — an
  unavailable earnings lookup rejects, "vendor down" never means "permission granted".
- `engine.py`: `propose()` → gates → sizing → heat → `SizedProposal | Rejection`.
  **Time stops are trading SESSIONS (canonical, identical to the backtester's
  semantics)**; the ledger's `time_stop_date` is a derived calendar display value
  (`ceil(sessions·7/5)` days) and `exit_params.time_stop_sessions` carries the real
  deadline. `open_position_heat()` lets batch callers accumulate heat across
  proposals. Executor's fixed-notional placeholder is replaced when WI-067 wires this in.
- Live smoke (`uv run python -m vega.risk`): real paper equity, real regime (caution →
  3R cap active), heat genuinely accumulating across candidates (AAPL 0.8R + BTC 1.0R
  = 1.8R total), BTC contamination measured (not silently skipped), both sleeves at
  exactly 2.00R worst case, round-tripping into valid ledger entries.

## Signal lifecycle (WI-065)

`candidate → backtested → paper-live → trusted → retired`. Only `paper-live`/`trusted`
signals are eligible for recommendations (`is_eligible_state`) — the filter WI-067's
briefing v2 will apply. Retirement is reachable from any state; retired is terminal (a
reconsidered signal registers as a new family/version, preserving the audit trail).

- `rationale.py`: `RationaleRegistry` (append-only economic rationales) + a `RationaleSource`
  protocol and `NullRationaleRegistry` for **explicit** opt-out. The rationale-first gate
  fires at the **top of `run_backtest`, before any data load, simulation, or holdout** — a
  review found placing it inside `record_run` (the last step) let an ungated run compute
  everything, see the results, and silently burn the holdout without the touch counter
  recording it, so the anti-HARKing audit was itself bypassable. The gate is **mandatory**
  (the param is required); tests opt out visibly via `NullRationaleRegistry`, never by
  omission. `record_run` keeps a defense-in-depth check.
- `lifecycle.py`: `LifecycleRegistry` — an explicit transition table. `backtested → backtested`
  is a legal **re-justification** self-transition: after a demotion, a fresh backtest run
  attaches a new `justifying_run_id` so the demotion band reflects post-demotion evidence
  instead of the stale band the signal already breached. Trust-granting transitions
  (`paper-live`, `trusted`, `retire`) **require a `human:`-prefixed actor** (an unattended
  agent cannot promote to live — a prefix contract, the same solo-scale posture as Caral's
  role tokens). Justifying-run selection uses `is None`, not `or`, so a legitimate 0.0-Sharpe
  run isn't scored as −∞. Every read-validate-append holds a cross-process lock
  (`AppendLog.exclusive_lock`) so two writers can't both validate against stale state (e.g.
  racing a retire against a promote). **Version policy is stated, not implicit:** lifecycle
  state is a family-level decision; the justifying run's version is recorded for audit; a
  materially different algorithm registers as a new family.
- `demotion.py`: auto-demotion when live performance falls below the **backtest confidence
  band = [worst, best] dev-fold Sharpe of the justifying run**. Live Sharpe comes from
  `backtest.live_metrics.live_sharpe` — a backtest-**owned** service (governance no longer
  reaches into engine internals; a review flagged the inverted dependency) that enforces two
  comparability rules the review found broken: it computes over the **full trading-session
  grid** (flat days included, exactly as backtest folds do — sampling only trade-event days
  inflated live Sharpe and made demotion under-fire), and it **raises on a mixed-asset-class
  batch** rather than annualizing a blended series by whichever trade sorted first. Evaluated
  only once ≥30 live trades exist. Demotion always lands on `backtested`.
  **Scope note:** the live ledger doesn't record exit fills yet (WI-067's job), so there is no
  real live-trade source today — fully wired and tested against synthetic data so it
  activates the moment WI-067 lands.

## First signal families (WI-066)

Three families, rationales recorded in the `RationaleRegistry` before their first backtest
(the WI-065-enforced ordering — registry timestamps prove it), price/volume-only,
equities/ETFs, 545-symbol universe, 6-point total grid:

All three families were re-run at **version 1.1** after the WI-066 strongest-model review
(10 findings fixed @ 83a646f: mixed price spaces, NaN fall-through, grid double-counting,
trough-depth rule mismatch, `is_new_high` semantics, `signal_params` recorded on run records,
holdout-divergence machine flag, MarketView pre-grouping perf fix — the 1.5–2h batch became
~25 min). Final dispositions (Jose's human decisions, 2026-07-14):

- `trend_pullback.py` — buy the first up-close after a 3–5% pullback (measured at the trough
  since v1.1) inside a rising-SMA50 uptrend. v1.1 holdout recovered from negative to
  +0.17/+0.28 but dev 1.9–2.5 = ~90% dev→holdout degradation. **HELD at `candidate`** —
  legal to promote under the gate, judged unwise; revisit with more live history or a v2 rule.
- `breakout_volume.py` — new N-session closing high (N=40, 55) on ≥1.5× median 60-session
  *consolidated* volume (never IEX). **Failed both grid points twice (v1.0 and v1.1,
  Sharpe ≈0.0–0.02)** — its own registered falsification condition was met. **RETIRED**
  (terminal; transition `0567a1ae`, actor `human:jose`). The rationale stays on file as an
  honest negative result.
- `oversold_reversion.py` — 3-session drop ≥2.0–2.5×ATR14 (fully in adjusted price space
  since v1.1) while still above SMA100, 7-session time stop (doctrine override), half off at
  +1.5R. v1.1: **passed both grid points, holdout Sharpe 3.75/3.67 vs dev 1.31/1.30** —
  non-degrading out-of-sample. **Promoted to `paper-live`** by Jose 2026-07-14 (transition
  `2305b7bf`, justifying run `29469e7e`, `justifying_params {"k": 2.0}`). Calibration note
  on record: holdout > dev likely reflects a reversion-friendly recent window; the realistic
  live expectation is dev-level (~1.3), and the demotion band polices it once WI-087 delivers
  exit fills. Verdicts were computed under the executor's fixed-$1k sizing; WI-067 replaces
  that with risk-engine qty.

`helpers.py` centralizes the price/volume math (SMA, N-session-high, median volume,
3-session change) as pure functions over an already-PIT-truncated bars frame; ATR reuses
the one shared `vega.common.atr` implementation via a thin adapter, never reimplemented.

## Ranked calls (WI-067) — entries only; exits are WI-087

The first output a manager could act on: `briefing/calls.py` wires signals (WI-066) →
lifecycle eligibility (WI-065) → risk engine (WI-064) → ledger (WI-060) → paper executor
(WI-061) into one deterministic daily pass. Never mutates or closes a position — that's
WI-087's job (sell orders, exit monitoring, exit fills, and therefore live demotion).

- **Eligibility is family-agnostic and evidence-locked**: for every family in
  `FAMILY_SIGNALS` currently `paper-live`/`trusted`, `build_calls` resolves the exact
  `justifying_params` from the backtest run that earned its promotion and instantiates
  the signal with THOSE parameters — never defaults. A family that's eligible but missing
  that evidence is a bookkeeping bug, not a "guess and proceed" situation: `CallsError`
  raises rather than run unvalidated parameters live (closes WI-066 review finding #2 —
  a promoted family's live behavior can no longer silently drift from what was validated).
- **Risk-sizing runs in rank order** (confidence DESC, family dev-Sharpe DESC, symbol ASC
  — a full deterministic tiebreak) and heat accumulates via `open_position_heat()` as each
  proposal is accepted, so higher-ranked calls claim heat first and the caps alone bound
  the day's count (no separate max-calls knob). Open positions are reconstructed from the
  ledger as every FILLED long with no exit fill (the ledger has no exit concept until
  WI-087) — the ORIGINAL stop price is used for heat (no trailing-stop tracking yet),
  which overstates rather than understates heat.
- **Exit-spec fidelity (`risk/engine.py`, hardened by the WI-067 review)**: `propose()`
  honors ALL FOUR per-family exit params the backtester honors (`time_stop_sessions`,
  `profit_take_half_at_r`, `stop_atr_mult`, `profit_trail_atr_mult`) — the build initially
  plumbed only two, and the review showed that "honoring a subset" silently re-creates the
  live/backtest exit divergence the single-writer doctrine exists to prevent (simulate.py
  honors all four; the first family to override stop or trail would have diverged). Every
  override is validated against a doctrine band (`common/doctrine.py` `*_BAND` constants):
  a typo'd exit spec is a `Rejection`, never a binding ledger contract.
- **Same-day auto-execution (`execution/executor.py`, hardened by the review)**:
  `execute_pending(as_of, equity)` submits ONLY calls decided at the current session — a
  pending rec from an earlier session missed its T+1 open (the only fill the backtest
  models) and EXPIRES with a one-time failure record, never late-fills. qty is exclusively
  the risk engine's: a qty-less or non-positive rec is refused (the $1,000 notional
  fallback is gone — a sizing decision the risk engine never made must not enter the track
  record), and any order breaching the notional ceiling (25% of equity) is refused, not
  clamped. `reconcile_fills` re-polls orders whose ledger record is still an acceptance
  (price=None — the normal pre-market case) and appends the real fill or terminal cancel
  once the venue knows it; `LedgerStore.latest_with_fills` resolves fills through
  supersede chains and prefers priced records.
- **Rendering**: a `## Ranked calls` section appears only once ≥1 family is eligible (empty
  `eligible_families` renders the v1 sections byte-identically — no regression for the
  streak gate). An empty call list still renders an explicit **"No trade today"** line with
  a reason (`regime risk_off` > `macro T-1/T window` > `no qualifying setups` > `N
  candidates considered, none cleared gates/heat` — checked in that priority order since
  regime/macro are blanket, symbol-independent gates). A rejections table always shows what
  was considered and refused — evidence integrity means the reader sees the denominator,
  not just the winners.
- **Live smoke (2026-07-15, real store/regime/Alpaca account)**: 2 real calls
  (`oversold_reversion_v1`, CDW & CSCO), correctly risk-sized (qty from the engine, not
  $1,000 notional), correct 7-session time stop and +1.5R profit-take override, heat
  accumulating 0.80R → 1.60R across the ranked pair, both landed on the ledger and
  auto-submitted to Alpaca paper (`status: accepted`). 2 real rejections
  (`earnings_in_horizon`, network-resolved `EarningsFact` outside the engine, correctly
  gating both ELV and EW). 208 tests total (was 182), verify.sh green.

### WI-067 strongest-model review (10/10 findings confirmed, all fixed same session)

The review's top finding was empirically live: the scheduled launchd run was already
crashing on the write-once briefing check (regime inputs drift between same-day runs),
and under the initial WI-067 build that crash landed AFTER ledger appends and BEFORE
execution — same-day re-runs would have stacked duplicate positions. The fix set:

- **Idempotency by position, not by file**: `build_calls` never proposes a symbol with an
  active position — filled (chain-resolved) or same-session pending — rejecting it as
  `already_held`. Same-day re-runs and multi-family overlap structurally cannot stack
  entries; the write-once conflict is now tolerated (first briefing of the day wins) and
  the run continues to reconciliation/execution.
- **Failure domains split in `briefing/__main__.py`**: stale store (>4 days) and an
  unreachable Alpaca account are hard gates for the whole calls path; a calls-generation
  failure fails CLOSED for execution (partial appends never execute unpublished) and is
  PUBLISHED via `BriefingData.calls_error` — a day the call engine failed is
  distinguishable on the record from "no eligible families".
- **Position reconstruction (`_active_positions`)**: filled longs resolve fills through
  supersede chains (a filled-then-corrected position keeps its heat and is never
  re-bought); THIS session's pending calls carry heat (they execute later in the same
  run); stale pending calls don't (they expire); terminally dead orders don't.
- **Honest no-trade reasons**: derived from the rejections actually collected (reason
  counts), never re-derived gate conditions — zero-proposal days say "no qualifying
  setups", not "blocked by FOMC".
- **Loud family bookkeeping**: eligibility iterates `lifecycle.families()` (new API), so
  a paper-live family missing a `FAMILY_SIGNALS` class registration raises `CallsError`
  instead of being silently untradeable.
- **Efficiency/reuse**: per-symbol frames into `propose()` (no full-frame masks per
  candidate), `view.bars()` reused instead of a parallel groupby, date-bounded
  `load_signal_frame(as_of)` (frame stays O(lookback) as the store grows), registry runs
  read once, `live_account_equity()` deduplicated into `execution/executor.py`, shared
  test fixtures in `tests/conftest.py`, rank computed at construction, render tables
  typed. Live re-run proof: the fixed pipeline ran against the production ledger the same
  night — 0 duplicate appends, conflict tolerated, and `reconcile_fills` healed both
  broken acceptance records into real priced fills (CDW @ 132.55, CSCO @ 110.66).
  221 tests (was 208), verify.sh green.

## Exit monitor + auto-demotion (WI-087)

Closes the position loop WI-067 deliberately left open. `src/vega/execution/exits.py`
mirrors `backtest/simulate.py`'s exit mechanics against the live ledger — same triggers
(gap-stop, stop, profit-partial, time-stop), same priority order, same session-counting
semantics — so a position closes live the same way the backtest that justified its
promotion said it would.

- **State is derived, never persisted**: `reconstruct_positions(ledger, frame, as_of)` is a
  pure function of (entry fill, exit fills, `exit_params`, the pooled equity/ETF session
  calendar) — remaining qty, trailed stop, and sessions-held are recomputed fresh every
  run, safe to re-run any number of times. `OpenPosition.is_pending` distinguishes a
  same-session submitted-not-yet-filled call (original stop, no trail) from a filled,
  still-open position (trailed stop, qty net of partials) — the ONE reconstruction both
  heat accounting (`briefing/calls.py`, via `to_heat()`) and exit evaluation read, so the
  two can never drift the way two independent reconstructions would.
- **The trail is windowed, not replayed**: a partial exit is tagged with the deciding
  `session` (a new additive field on ledger fills — see below); the trail's high-water
  close is `max(close)` over `[partial_session, as_of]` from the store, which is
  mathematically identical to simulate.py's day-by-day running max without needing to
  replay every intervening session.
- **Documented timing divergence (accepted by Jose at enrichment time)**: simulate.py
  triggers stop/profit intraday on the same session bar (full history, already known); the
  live monitor runs pre-market and can only see the last COMPLETED session, so a triggered
  stop/profit submits a market sell that fills one session later than the backtest's
  intraday fill. Time stops already had this lag in simulate.py itself (queue-and-fill-at-
  next-open), so they match exactly; only stop/profit fills carry the one-session gap.
- **Ledger schema, additive**: `LedgerStore.append_fill` gains `side` ("buy"/"sell"),
  `reason` (simulate.py's own vocabulary: gap_stop/stop/profit_partial/time_stop), and
  `session` (the deciding store session — never a wall-clock timestamp). `Recommendation`
  already had `as_of` from WI-067. `latest_with_all_fills()` is the new chain-aware join
  (every fill, buy and sell, resolved through supersede chains) that both `exits.py` and
  `lifecycle/live_trades.py` build on; the existing `latest_with_fills()` (buy-only, one
  fill per chain) is untouched — it answers "was this entered", not "is this still open".
- **Daily flow order** (`briefing/__main__.py`): reconcile fills → evaluate + execute exits
  → check auto-demotions → `build_calls` (heat and idempotency are now exit-aware) →
  entries. A symbol just exited is excluded from re-entry the same run
  (`exited_today`/`same_day_exit`) — simulate.py never re-enters a symbol the session it
  exits, so neither does this.
- **Auto-demotion activates for real** (`lifecycle/live_trades.py`): `closed_round_trips`
  turns every realized exit lot into a `backtest.live_metrics.LiveTrade` row (one row per
  lot — a partial followed by a time-stop produces two), grouped by family (parsed from
  `signal_attribution`). `check_and_apply_demotions` evaluates every `paper-live`/`trusted`
  family per asset-class sleeve against its justifying backtest band and calls
  `LifecycleRegistry.demote(..., automatic=True)` — agent-legal, unlike promotion — the
  moment the evidence says so; demotes at most once per family per run even when multiple
  sleeves qualify (a second `demote()` on an already-demoted family would raise).
- **Real production bug caught by the live smoke test, not by review**: the two real open
  positions from WI-067's first live smoke (CDW, CSCO) predate the `as_of` field entirely
  (it was added during WI-067's *review fixes*, after those positions were already
  ledgered) — `rec.get("as_of")` is `None` for both. Without a fallback, `reconstruct_positions`
  would derive no `entry_session` for them and they would sit outside every exit check
  forever — a live position with a 7-session time stop that could never fire. Fixed with a
  fallback to the ORIGINAL (first) buy fill's timestamp date. A second bug surfaced
  immediately after: the fallback initially read the LATEST buy fill's timestamp, which is
  the reconciliation event's wall-clock time (whenever that run happened to execute) —
  for CDW/CSCO this landed on a date one full day past the store's most recent session,
  producing an empty candidate list and silently reproducing the same bug. Fixed by reading
  `buy_fills[0]` (the original submission) instead of `buy_fills[-1]`; both bugs are now
  regression-tested (`test_legacy_position_without_as_of_falls_back_to_fill_timestamp`,
  `test_legacy_fallback_uses_first_buy_fill_not_a_later_reconciliation`).
- **Briefing gains two sections**: "Exits" (symbol, reason, qty, trigger detail) and
  "Signal health" (per family/sleeve: live trade count, live Sharpe when computable, band,
  verdict) — both empty-default and absent from rendering until there's something to show,
  preserving the v1/v2 byte-identical-when-empty contract.
- **Live smoke (2026-07-17, real store/regime/ledger/lifecycle)**: reconstructed all 4 real
  open positions correctly (CDW/CSCO via the legacy fallback, entry_session on the current
  session; ALL/C — two fresh signals this run — correctly submitted, sized, and pending);
  zero exit triggers fired (none of the four is near its stop, profit target, or time
  stop yet — an honest, verified negative, not a skipped check); a same-day re-run
  produced zero duplicate appends (`executed 0 call(s)` on the second pass, proving
  idempotency against real state); `check_and_apply_demotions` ran cleanly with
  `insufficient_sample` (fewer than 30 real live trades exist yet — correct, no false
  demotion). 253 tests (was 221), verify.sh green.

### WI-087 strongest-model review fixes (10 findings, all shipped)

The review's headline: the exit loop had never completed a full round trip in production,
and that unexercised path hid a chain that would have killed the feature's purpose. All
fixed at this commit:

- **Reconciliation identity (#1)**: `reconcile_fills` now propagates `side`/`reason`/
  `session` onto the records it appends — previously every pre-market SELL's
  reconciliation was re-labeled a buy, corrupting reconstruction (half-open positions read
  fully closed; canceled sells erased whole positions and re-enabled double-buys) and
  starving `closed_round_trips` of every round trip, leaving auto-demotion permanently at
  `insufficient_sample`.
- **Per-order fill resolution (#9)**: `latest_with_all_fills` resolves records per
  order_id (priced wins; otherwise latest knowledge; identity fields merged; `at` stays
  anchored to submission) so an acceptance plus its reconciliation is ONE fill — qty sums
  can never double-count — and pre-fix mislabeled records are healed on read.
- **In-flight sells (#2)**: an accepted-but-unpriced sell no longer reduces the position.
  `remaining_qty` counts only PRICED sells (heat stays conservative); a new
  `in_flight_sell_qty` keeps exit decisions from re-selling qty already covered by a
  working order.
- **Session-clock origin (#3)**: `entry_session_for` anchors to the chain-ORIGIN's
  `as_of` (`LedgerStore` now annotates recs with `origin_as_of`), so a supersede
  correction can no longer reset a position's time-stop clock — while the correction's
  own stop still governs.
- **Confirmed entries only (#4)**: `entry_confirmed` distinguishes a priced buy from a
  presumed acceptance — presumed positions reserve heat (conservative) but can never arm
  the SELL path (selling never-bought shares was the unsafe direction).
- **Track-record calendar (#5)**: `closed_round_trips` takes the FULL store session
  calendar (`full_session_calendar()`), never the 220-day signal frame — old entries no
  longer clamp to a window floor and old exits no longer vanish from PnL exactly when the
  30-trade demotion gate opens; legacy `as_of=None` recs (CDW/CSCO) now enter the track
  record via the same shared session fallback.
- **Exits publish unconditionally (#6)**: `__main__` attaches executed exits to the
  briefing THE MOMENT they run, before demotions or entry generation — a later failure
  can no longer place sells the published record omits.
- **Stale-store semantics (#7)**: exit monitoring now RUNS on a stale store (risk-reducing;
  a late stop beats an unmanaged one) — only new entries require fresh data; an
  unreachable Alpaca publishes how many open positions went unmanaged that run.
- **Structural data gaps fail loudly (#8)**: a held, confirmed position whose symbol has
  no bars in the monitoring frame raises `ExitMonitorGapError` (the first crypto
  promotion would otherwise have run with stops silently never evaluated).
- **R basis documented (#10)**: live R = fill price minus the PUBLISHED ledger stop — the
  binding contract — which diverges from simulate.py's fill-derived stop with the
  overnight gap; now a stated assumption in the module docstring, with the structural fix
  (one shared trigger function) parked on WI-084.

Live re-run proof after the fixes: legacy CDW/CSCO fills resolve to single priced
records, ALL/C remain honest unpriced acceptances (`entry_confirmed=False`, unsellable),
zero duplicate appends, demotion clean over the full calendar. 259 tests (was 253),
verify.sh green.

## Verification gate

`scripts/verify.sh` — executed by Caral's daily-build runner; non-zero exit = failed build.
Steps: frozen sync → mypy (strict) → ruff check + format (incl. bandit `S` security rules) →
pytest → pip-audit → secret scan.
