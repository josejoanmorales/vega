"""Vega risk engine — sizing, exits, portfolio heat (STRATEGY.md §5).

Pure math over stored data: no network, no clock inside the math itself.
The single writer of exit specs — the ledger, the backtester, and briefing
v2 all consume the same structured output this package produces.
"""
