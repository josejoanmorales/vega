"""MarketView — the only way a signal ever touches price data.

Lookahead bias is not a discipline problem here, it is an API impossibility:
`bars()` filters to `date <= as_of` inside the class. A signal is handed a
MarketView and NOTHING else — it never receives a raw DataFrame it could
slice incorrectly.

Performance: the frame is grouped by symbol ONCE at construction (each group
pre-sorted by date), so bars() is a dict lookup + binary search instead of a
full-frame boolean mask + sort per call. Three reviews flagged the old O(n)
scan; with 3 signal families × 545 symbols × ~440 test days × folds it was
~10M full-frame passes and the dominant cost of a backtest batch. The PIT
contract is unchanged and covered by the same tests.
"""

from __future__ import annotations

import pandas as pd


class MarketView:
    def __init__(self, frame: pd.DataFrame, as_of: str) -> None:
        # Truncation happens on every read, not once at construction, so a
        # single MarketView can be cheaply re-used across as_of advances
        # without ever leaking future rows.
        self._as_of = as_of
        self._by_symbol: dict[str, pd.DataFrame] = {
            str(symbol): group.sort_values("date").reset_index(drop=True)
            for symbol, group in frame.groupby("symbol")
        }

    @property
    def as_of(self) -> str:
        return self._as_of

    def bars(self, symbol: str, lookback: int | None = None) -> pd.DataFrame:
        group = self._by_symbol.get(symbol)
        if group is None:
            return pd.DataFrame(columns=["symbol", "date"])
        # group is date-sorted; searchsorted gives the PIT cutoff in O(log n)
        cutoff = int(group["date"].searchsorted(self._as_of, side="right"))
        sub = group.iloc[:cutoff]
        if lookback is not None:
            sub = sub.tail(lookback)
        return sub.reset_index(drop=True)

    def symbols(self) -> list[str]:
        return sorted(
            symbol
            for symbol, group in self._by_symbol.items()
            if str(group["date"].iloc[0]) <= self._as_of
        )
