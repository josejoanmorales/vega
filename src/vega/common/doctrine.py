"""Exit-spec doctrine constants — the ONE definition (STRATEGY.md §5).

Both authorities consume these: the live risk engine (vega.risk) sizes and
writes exit specs from them, and the backtester (vega.backtest.signals'
EntryProposal defaults) simulates the same mechanics. They must never be
redefined locally — a hand-copied constant that drifts silently invalidates
every walk-forward result against what live trading actually does.

Units: time stops are TRADING SESSIONS (the backtest's semantics — canonical).
Calendar-date representations are derived display values, never the deadline.
"""

STOP_ATR_MULT = {"equity": 2.0, "etf": 2.0, "crypto": 2.5}
GAP_STRESS_MULT = {"equity": 2.5, "etf": 2.5, "crypto": 2.0}
PROFIT_TAKE_HALF_AT_R = 2.0
PROFIT_TRAIL_ATR_MULT = 2.5
DEFAULT_TIME_STOP_SESSIONS = 15

# Approximate trading-sessions -> calendar-days conversion for derived display
# dates (7 calendar days per 5 sessions; crypto trades 7/7 but we keep one
# conservative conversion for the human-readable ledger field).
CALENDAR_DAYS_PER_SESSION = 7.0 / 5.0
