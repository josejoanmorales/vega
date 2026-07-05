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
| `src/vega/data/` — sources, snapshots, validation, universe | WI-058 | planned |
| regime & event calendar | WI-059 | planned |
| ledger + override log | WI-060 | planned |
| paper executor | WI-061 | planned |
| briefing v1 | WI-062 | planned |

## Verification gate

`scripts/verify.sh` — executed by Caral's daily-build runner; non-zero exit = failed build.
Steps: frozen sync → mypy (strict) → ruff check + format (incl. bandit `S` security rules) →
pytest → pip-audit → secret scan.
