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

## Verification gate

`scripts/verify.sh` — executed by Caral's daily-build runner; non-zero exit = failed build.
Steps: frozen sync → mypy (strict) → ruff check + format (incl. bandit `S` security rules) →
pytest → pip-audit → secret scan.
