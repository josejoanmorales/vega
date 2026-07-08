"""Signal promotion lifecycle (STRATEGY.md §6, Pillar 2).

candidate -> backtested -> paper-live -> trusted -> retired. Only paper-live+
signals influence recommendations. A signal cannot enter testing without a
written economic rationale recorded first. Auto-demotion when live
performance falls below the signal's backtest confidence band.
"""
