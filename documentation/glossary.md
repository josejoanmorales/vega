# Vega — Glossary

- **R** — one unit of risk: the fraction of portfolio equity risked per trade (0.5–1%). A trade that hits its stop loses ~1R.
- **Portfolio heat** — total open risk across all positions, in R; capped at 6R, correlation-aware across the equities and crypto sleeves.
- **Ledger** — the append-only record of every recommendation (thesis, confidence, horizon, full exit spec, attribution). Corrections are new entries, never edits.
- **Exit spec** — the mandatory quartet on every call: hard stop, time stop, profit rule, invalidation condition.
- **Signal lifecycle** — candidate → backtested → paper-live → trusted → retired. Only paper-live+ influences recommendations.
- **Regime gate** — deliberately simple market-state filter (200DMA, VIX bands, breadth, crypto fear/greed) that scales exposure down to and including "no trade".
- **Quarantine** — a symbol/day excluded from all downstream use because its two data sources disagreed beyond tolerance.
- **Backtest registry** — append-only record of every backtest ever run, so the hypothesis count is auditable (anti-data-mining).
