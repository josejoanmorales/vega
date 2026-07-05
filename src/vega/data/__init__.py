"""Vega data layer — dual-source prices, write-once snapshots, versioned universe.

Backtests and all downstream consumers read ONLY the clean DuckDB views built
from immutable local snapshots — never live APIs (STRATEGY.md §6, Pillar 5).
"""
