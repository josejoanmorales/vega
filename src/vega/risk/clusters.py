"""Cluster assignment + the crypto/equity cross-contamination rule (STRATEGY.md §5.B).

Deliberately dumb (cluster buckets, not covariance) — the doctrine explicitly
rejects "clever" correlation math for the heat cap itself. The ONE place real
correlation is computed is the crypto-contamination check, and even that is a
simple trailing-window Pearson correlation on daily returns, not a model.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from vega.risk.types import CLUSTERS

CORRELATION_WINDOW = 90
CONTAMINATION_THRESHOLD = 0.5
CONTAMINATION_FRACTION = 0.5


@lru_cache(maxsize=1)
def _cluster_by_symbol() -> dict[str, str]:
    """symbol -> cluster, from the committed universe artifact's `cluster`
    column (universe-v2, WI-084 item 8 — cluster membership used to be a
    hardcoded RATES/COMMODITIES frozenset here; it is now symbol metadata on
    the artifact, like every other UniverseEntry field). Cached module-level:
    the universe is a committed, versioned file that never changes within a
    process lifetime."""
    from vega.data.universe import load_universe

    return {e.symbol: e.cluster for e in load_universe()}


def classify(symbol: str, asset_class: str) -> str:
    if asset_class == "crypto":
        return "crypto_beta"
    cluster = _cluster_by_symbol().get(symbol)
    if cluster in ("rates", "commodities"):
        return cluster
    return "us_equity_beta"  # stated default: every other equity/ETF, incl. credit ETFs and
    # symbols absent from the committed universe entirely — never guess rates/commodities


def spy_correlation(
    frame: pd.DataFrame, symbol: str, as_of: str, window: int = CORRELATION_WINDOW
) -> float | None:
    """Trailing shared-session return correlation to SPY through `as_of`.

    Raises if `frame` contains no SPY rows AT ALL — that is a broken data
    contract at the call site (review finding: a source-filtered frame made
    this rule silently dead), not an "unmeasurable" market fact. Returns None
    only for genuinely thin overlapping history — callers must treat None as
    "unmeasurable", never "zero correlation" (see contaminates_equity_beta).

    The two series are merged on date FIRST, then the tail window is taken on
    the shared sessions — crypto's 7-day calendar vs equities' 5-day calendar
    would otherwise shrink the intersection below the window and under-fire
    the contamination rule even on well-formed frames (review finding).
    """
    sub = frame[frame["date"] <= as_of]
    target = sub[sub["symbol"] == symbol].sort_values("date")
    bench = sub[sub["symbol"] == "SPY"].sort_values("date")
    if bench.empty:
        raise ValueError(
            "spy_correlation: frame contains no SPY rows — the caller must supply "
            "SPY history alongside the crypto symbol (do not source-filter SPY away)"
        )
    merged = (
        target[["date", "adj_close"]]
        .merge(bench[["date", "adj_close"]], on="date", suffixes=("_t", "_b"))
        .sort_values("date")
        .tail(window + 1)
    )
    if len(merged) < window + 1:
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
