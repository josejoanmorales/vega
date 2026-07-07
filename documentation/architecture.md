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

Sizing and exits determine P&L more than entries do. Pure math over stored data; the
single writer of exit specs consumed by the ledger, the backtester, and briefing v2.

- `sizing.py`: gap-stressed sizing (`STOP_ATR_MULT` k=2.0 equity/2.5 crypto,
  `GAP_STRESS_MULT` G=2.5 equity/2.0 crypto). The `min(base, gap)` formula
  algebraically guarantees nominal risk ≤1R and gap-stressed worst case ≤2R for
  *any* positive k/G — proven by the live smoke test landing exactly at 2.00R for
  both an equity and a crypto proposal. A 1.5R clamp sits on top as a defensive
  invariant against a future caller bypassing the formula (`vega.common.atr` supplies
  the shared ATR — extracted from `backtest/simulate.py` so live sizing and backtest
  simulation can never silently diverge on stop-distance math).
- `clusters.py`: deliberately dumb bucket assignment (`us_equity_beta`, `rates`
  — TLT/IEF, `commodities` — GLD/SLV/USO/XME, `crypto_beta`), not covariance. The one
  real correlation computed: a 90-day trailing Pearson of a crypto symbol's returns
  against SPY; above 0.5 it counts 50% of that position's R into `us_equity_beta`
  too — the reason "separate sleeves" isn't actually safe. Unmeasurable correlation
  (insufficient history) is treated as non-contaminating, never assumed.
- `heat.py`: position heat = `qty * max(entry - current_stop, 0)`, floored at 0 so a
  stop trailed past breakeven frees heat for new risk. Caps: 6R total (3R when regime
  is `caution`), 4R `us_equity_beta`, 2.5R `crypto_beta`, 3R `rates`/`commodities`.
- `gates.py`: regime `risk_off` blocks all new entries; `in_macro_window` blocks T-1/T
  of FOMC/CPI (both sleeves); `next_earnings` within the horizon rejects (v1 has no
  earnings-play exceptions).
- `engine.py`: `propose()` orchestrates gates → ATR/sizing → cluster/heat-cap check →
  `SizedProposal | Rejection`, emitting the full exit spec (stop, calendar-approximate
  time stop, structured profit rule, invalidation). `to_recommendation()` bridges a
  proposal straight into a valid `ledger.types.Recommendation` (which gained optional
  `exit_params`/`qty` fields — additive, append-only-compatible schema growth).
  Executor's fixed-notional placeholder is replaced when WI-067 wires this in.
- Live smoke (`uv run python -m vega.risk`): real Alpaca paper equity ($99,900.90),
  real regime, real store — AAPL and BTC proposals both size to exactly 2.00R worst
  case, correctly clustered, round-trip into valid ledger entries.

## Verification gate

`scripts/verify.sh` — executed by Caral's daily-build runner; non-zero exit = failed build.
Steps: frozen sync → mypy (strict) → ruff check + format (incl. bandit `S` security rules) →
pytest → pip-audit → secret scan.
