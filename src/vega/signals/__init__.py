"""Vega's first candidate signal families (WI-066).

Each family's economic rationale is recorded in the RationaleRegistry BEFORE
its first backtest (enforced structurally by run_backtest's gate — WI-065).
All three are price/volume-only, PIT-safe by construction (data comes only
from MarketView.bars), and equities/ETFs-only in v1.
"""
