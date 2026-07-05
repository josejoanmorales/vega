"""Cross-source reconciliation: primary bars vs check-source closes.

Canonical rule (STRATEGY.md §6): equities — yfinance is primary (consolidated
volume), Alpaca IEX cross-checks the close only; crypto — Binance is primary,
CoinGecko cross-checks. A (symbol, date) whose sources disagree beyond tolerance,
or that the check source cannot see at all, is quarantined for that day and
excluded from every downstream consumer.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_TOLERANCE = 0.005  # 0.5% on close — stated assumption in WI-058, configurable

QUARANTINE_COLUMNS = ("symbol", "date", "primary_close", "check_close", "rel_diff", "reason")


@dataclass(frozen=True)
class CrossCheckResult:
    clean: pd.DataFrame
    quarantine: pd.DataFrame


def cross_check(
    primary: pd.DataFrame, check: pd.DataFrame, tolerance: float = DEFAULT_TOLERANCE
) -> CrossCheckResult:
    if primary.empty:
        return CrossCheckResult(
            clean=primary.copy(), quarantine=pd.DataFrame(columns=list(QUARANTINE_COLUMNS))
        )
    merged = primary.merge(
        check[["symbol", "date", "close"]].rename(columns={"close": "check_close"}),
        on=["symbol", "date"],
        how="left",
    )
    merged["rel_diff"] = (merged["close"] - merged["check_close"]).abs() / merged["check_close"]
    missing = merged["check_close"].isna()
    breach = ~missing & (merged["rel_diff"] > tolerance)

    bad = merged.loc[missing | breach, ["symbol", "date", "close", "check_close", "rel_diff"]]
    bad = bad.rename(columns={"close": "primary_close"}).copy()
    bad["reason"] = [
        "missing from cross-check source"
        if pd.isna(c)
        else f"close diverges beyond {tolerance:.3%}"
        for c in bad["check_close"]
    ]
    clean = merged.loc[~(missing | breach), list(primary.columns)].reset_index(drop=True)
    return CrossCheckResult(
        clean=clean, quarantine=bad[list(QUARANTINE_COLUMNS)].reset_index(drop=True)
    )
