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

## Verification gate

`scripts/verify.sh` — executed by Caral's daily-build runner; non-zero exit = failed build.
Steps: frozen sync → mypy (strict) → ruff check + format (incl. bandit `S` security rules) →
pytest → pip-audit → secret scan.
