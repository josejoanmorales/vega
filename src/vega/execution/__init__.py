"""Vega paper execution — every recommendation forward-tests itself.

Fills are appended to the ledger as linked records (the recommendation is
never mutated); failures are logged append-only and surfaced by the briefing.
"""
