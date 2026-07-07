"""Cluster assignment + the crypto/equity cross-contamination rule (STRATEGY.md §5.B).

Deliberately dumb (cluster buckets, not covariance) — the doctrine explicitly
rejects "clever" correlation math for the heat cap itself. The ONE place real
correlation is computed is the crypto-contamination check, and even that is a
simple trailing-window Pearson correlation on daily returns, not a model.
"""

from __future__ import annotations

import pandas as pd

from vega.risk.types import CLUSTERS

RATES = frozenset({"TLT", "IEF"})
COMMODITIES = frozenset({"GLD", "SLV", "USO", "XME"})
CORRELATION_WINDOW = 90
CONTAMINATION_THRESHOLD = 0.5
CONTAMINATION_FRACTION = 0.5


def classify(symbol: str, asset_class: str) -> str:
    if asset_class == "crypto":
        return "crypto_beta"
    if symbol in RATES:
        return "rates"
    if symbol in COMMODITIES:
        return "commodities"
    return "us_equity_beta"  # stated default: every other equity/ETF, incl. credit ETFs


def spy_correlation(
    frame: pd.DataFrame, symbol: str, as_of: str, window: int = CORRELATION_WINDOW
) -> float | None:
    """Trailing daily-return Pearson correlation to SPY through `as_of`.

    None if either series has insufficient overlapping history — callers must
    treat that as "unmeasurable", not "zero correlation" (see contaminates_equity_beta).
    """
    sub = frame[frame["date"] <= as_of]
    target = sub[sub["symbol"] == symbol].sort_values("date").tail(window + 1)
    bench = sub[sub["symbol"] == "SPY"].sort_values("date").tail(window + 1)
    if len(target) < window + 1 or len(bench) < window + 1:
        return None
    merged = target[["date", "adj_close"]].merge(
        bench[["date", "adj_close"]], on="date", suffixes=("_t", "_b")
    )
    if len(merged) < window:
        return None
    t_ret = merged["adj_close_t"].pct_change().dropna()
    b_ret = merged["adj_close_b"].pct_change().dropna()
    if t_ret.std() == 0 or b_ret.std() == 0:
        return None
    return float(t_ret.corr(b_ret))


def contaminates_equity_beta(correlation: float | None) -> bool:
    """Stated assumption: unmeasurable correlation does NOT contaminate — we
    never invent risk exposure the data can't support (never assume silently)."""
    return correlation is not None and correlation > CONTAMINATION_THRESHOLD


assert set(CLUSTERS) == {"us_equity_beta", "rates", "commodities", "crypto_beta"}  # noqa: S101
