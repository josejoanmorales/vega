"""Vega backtest engine — the highest-stakes module in the codebase.

A subtle bug here silently invalidates every accuracy claim Vega makes.
The four backtest sins (lookahead, survivorship, ignored costs, overfitting)
are engineered out structurally (STRATEGY.md §6, Pillar 3), not merely
avoided by convention:

- lookahead: signals only ever see a MarketView truncated to `as_of` (market_view.py)
- survivorship: bounded by the fixed universe artifact; every registry
  record and report states the bound explicitly (known v1 limitation)
- costs: no zero-cost code path exists — costs.py is called inside the
  one fill function every trade passes through (simulate.py)
- overfitting: a locked holdout with touch-counting, plus a promotion bar
  that rises with the number of hypotheses tried (folds.py, registry.py)
"""
