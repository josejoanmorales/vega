"""MarketView — the only way a signal ever touches price data.

Lookahead bias is not a discipline problem here, it is an API impossibility:
`bars()` filters to `date <= as_of` inside the class. A signal is handed a
MarketView and NOTHING else — it never receives a raw DataFrame it could
slice incorrectly.
"""

from __future__ import annotations

import pandas as pd


class MarketView:
    def __init__(self, frame: pd.DataFrame, as_of: str) -> None:
        # frame: all sessions for the run's universe/date-span, kept private.
        # Truncation happens on every read, not once at construction, so a
        # single MarketView can be cheaply re-used across as_of advances by
        # the engine (see simulate.py) without ever leaking future rows.
        self._frame = frame
        self._as_of = as_of

    @property
    def as_of(self) -> str:
        return self._as_of

    def with_as_of(self, as_of: str) -> MarketView:
        """New view over the same frame at a different (necessarily later) cutoff."""
        return MarketView(self._frame, as_of)

    def bars(self, symbol: str, lookback: int | None = None) -> pd.DataFrame:
        sub = self._frame[
            (self._frame["symbol"] == symbol) & (self._frame["date"] <= self._as_of)
        ].sort_values("date")
        if lookback is not None:
            sub = sub.tail(lookback)
        return sub.reset_index(drop=True)

    def symbols(self) -> list[str]:
        visible = self._frame[self._frame["date"] <= self._as_of]
        return sorted(visible["symbol"].unique())
