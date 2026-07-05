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
| regime & event calendar | WI-059 | planned |
| ledger + override log | WI-060 | planned |
| paper executor | WI-061 | planned |
| briefing v1 | WI-062 | planned |

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

## Verification gate

`scripts/verify.sh` — executed by Caral's daily-build runner; non-zero exit = failed build.
Steps: frozen sync → mypy (strict) → ruff check + format (incl. bandit `S` security rules) →
pytest → pip-audit → secret scan.
