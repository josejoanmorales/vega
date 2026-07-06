"""Vega ledger — the append-only, unfakeable record of every recommendation.

This IS the product thesis (STRATEGY.md §2): corrections are new entries,
never edits; human deviations are logged as overrides linked to the original
call; nothing here can be silently rewritten.
"""
