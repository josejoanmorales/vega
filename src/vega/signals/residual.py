"""Idiosyncratic residual — separating a stock's own move from its cluster's.

THE FAILURE THIS REMOVES. The dislocation trigger cannot tell the difference
between a name that fell on its own and a name that fell because everything
fell. A stock down 6% inside a cluster down 5.5% has roughly zero
idiosyncratic move: there is no forced seller of THAT name to provide
liquidity to, only market beta, and the reversion thesis has nothing to
harvest. It is the most common false opportunity in the whole strategy and it
costs a full position each time.

DELIBERATELY DUMB (STRATEGY.md §5). Beta against ONE declared proxy per
cluster, OLS on daily returns, estimated on a window that ENDS BEFORE the
shock. No factor model, no rolling covariance, no data-driven proxy
selection — the doctrine says the correlation layer is where retail systems
overfit, and picking each symbol's "best-correlated ETF" would be exactly
that.

STATED LIMITATION, not an oversight: universe-v2 carries a *risk* cluster
(us_equity_beta / rates / commodities / crypto_beta), not a sector. So the
proxy for an equity is the broad market, and sector-specific contagion — an
energy name dragged down by XLE while SPY is flat — still reads as
idiosyncratic. Closing that needs a sector column on a universe-v3 and a real
GICS source; it is recorded as an open question on WI-228 rather than guessed
at here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vega.backtest.market_view import MarketView
from vega.backtest.signals import EntryProposal, Signal
from vega.signals.helpers import adjusted_atr14

# One declared proxy per risk cluster. `commodities` deliberately has none:
# GLD, SLV, USO and XME do not share a factor any single ETF represents, and
# inventing one would be worse than declaring the measurement unavailable.
CLUSTER_PROXY: dict[str, str] = {
    "us_equity_beta": "SPY",
    "crypto_beta": "BTC",
    "rates": "IEF",
}

BETA_WINDOW = 60  # sessions of daily returns used to estimate beta
SHOCK_SESSIONS = 3  # matches oversold_reversion's 3-session change
MIN_RESIDUAL_ATR = 1.5  # residual must itself be this many ATRs to count as a dislocation


@dataclass(frozen=True)
class ResidualDecomposition:
    """A move split into the part the cluster explains and the part it does not."""

    symbol: str
    proxy: str
    beta: float
    raw_change: float  # symbol's N-session change, adjusted price space
    explained_change: float  # beta * proxy's N-session change, same space
    residual_change: float  # raw - explained: the idiosyncratic part
    atr: float

    @property
    def residual_atr(self) -> float:
        return self.residual_change / self.atr

    @property
    def raw_atr(self) -> float:
        return self.raw_change / self.atr

    def describe(self) -> str:
        return (
            f"raw {self.raw_atr:+.1f}x ATR, {self.proxy} explains "
            f"{self.explained_change / self.atr:+.1f}x (beta {self.beta:.2f}), "
            f"residual {self.residual_atr:+.1f}x"
        )


def proxy_for(symbol: str, cluster: str) -> str | None:
    """The proxy this symbol is measured against, or None if unmeasurable.

    A symbol that IS its own cluster's proxy has no residual by construction
    (SPY against SPY is always exactly zero), and must return None rather than
    a meaningless 0.0 that a threshold would silently reject.
    """
    proxy = CLUSTER_PROXY.get(cluster)
    if proxy is None or proxy == symbol:
        return None
    return proxy


def _log_returns(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars[["date", "adj_close"]].dropna().sort_values("date")
    out = out.assign(ret=out["adj_close"].pct_change())
    return out.dropna(subset=["ret"])


def decompose(
    view: MarketView,
    symbol: str,
    proxy: str,
    sessions: int = SHOCK_SESSIONS,
    beta_window: int = BETA_WINDOW,
) -> ResidualDecomposition | None:
    """Split `symbol`'s N-session change into cluster-explained and residual parts.

    Returns None — "unmeasurable", never "zero" — when the proxy has no
    overlapping history, the beta window is too thin, or the proxy's returns
    have no variance. Callers must treat None as its own outcome; see
    `ResidualGate` for the stated policy.

    The beta window ENDS `sessions` bars before the current bar. Estimating
    beta on a window that contains the shock would let the shock set the very
    coefficient used to explain it away, which biases every residual toward
    zero exactly when it matters.
    """
    if sessions <= 0 or beta_window <= 1:
        raise ValueError("sessions must be >= 1 and beta_window >= 2")
    lookback = beta_window + sessions + 5
    target = _log_returns(view.bars(symbol, lookback=lookback))
    bench = _log_returns(view.bars(proxy, lookback=lookback))
    if target.empty or bench.empty:
        return None

    merged = target.merge(bench, on="date", suffixes=("_t", "_b")).sort_values("date")
    # Merge on date FIRST, then window — a 7-day crypto calendar against a
    # 5-day equity one would otherwise shrink the overlap below the window
    # (the same defect risk.clusters.spy_correlation was fixed for).
    if len(merged) < beta_window + sessions:
        return None

    estimation = merged.iloc[-(beta_window + sessions) : -sessions]
    bench_var = float(estimation["ret_b"].var())
    if bench_var <= 0 or pd.isna(bench_var):
        return None
    covariance = float(estimation["ret_t"].cov(estimation["ret_b"]))
    if pd.isna(covariance):
        return None
    beta = covariance / bench_var

    window = merged.tail(sessions)
    if len(window) < sessions:
        return None
    # Compound the shock-window returns back into a price change in the
    # symbol's own space, so the result stays comparable to ATR.
    target_bars = view.bars(symbol, lookback=lookback)
    atr = adjusted_atr14(target_bars, symbol, view.as_of)
    if atr is None or atr <= 0:
        return None
    start_price = float(target_bars["adj_close"].iloc[-(sessions + 1)])
    raw_change = float(target_bars["adj_close"].iloc[-1]) - start_price
    # Compounded explicitly rather than via Series.prod(): the shock window is
    # three elements, and this keeps the arithmetic in plain floats.
    proxy_growth = 1.0
    for daily in window["ret_b"].astype(float).tolist():
        proxy_growth *= 1.0 + float(daily)
    proxy_return = proxy_growth - 1.0
    explained_change = beta * proxy_return * start_price

    return ResidualDecomposition(
        symbol=symbol,
        proxy=proxy,
        beta=beta,
        raw_change=raw_change,
        explained_change=explained_change,
        residual_change=raw_change - explained_change,
        atr=atr,
    )


class ResidualGate:
    """Wraps any signal and keeps only proposals whose move is genuinely its own.

    Composition rather than modification: a promoted family keeps its exact
    behaviour and its registry history, and the gated variant is a SEPARATE
    family that must earn its own rationale and its own walk-forward verdict.

    UNMEASURABLE IS A REJECTION, stated explicitly. Where risk.clusters refuses
    to invent exposure it cannot measure, this gate refuses to confirm an
    opportunity it cannot measure — the two point opposite ways because one
    adds risk and the other adds a position. Flat is a position (STRATEGY.md
    §3), so the missing-data direction is "no trade", surfaced under its own
    reason so it never blurs into a real beta rejection.
    """

    promotable = True

    def __init__(
        self,
        inner: Signal,
        cluster_by_symbol: dict[str, str],
        min_residual_atr: float = MIN_RESIDUAL_ATR,
        beta_window: int = BETA_WINDOW,
        sessions: int = SHOCK_SESSIONS,
    ) -> None:
        self.inner = inner
        self.cluster_by_symbol = cluster_by_symbol
        self.min_residual_atr = min_residual_atr
        self.beta_window = beta_window
        self.sessions = sessions
        self.family = f"{inner.family}_residual"
        self.version = inner.version
        self.params = {
            **getattr(inner, "params", {}),
            "min_residual_atr": min_residual_atr,
            "beta_window": beta_window,
        }
        self.last_rejections: list[tuple[str, str, str]] = []  # (symbol, reason, detail)

    def scan(self, view: MarketView, universe: list[str]) -> list[EntryProposal]:
        self.last_rejections = []
        kept: list[EntryProposal] = []
        for proposal in self.inner.scan(view, universe):
            cluster = self.cluster_by_symbol.get(proposal.symbol, "us_equity_beta")
            proxy = proxy_for(proposal.symbol, cluster)
            if proxy is None:
                self.last_rejections.append(
                    (proposal.symbol, "residual_unmeasurable", f"no proxy for cluster {cluster}")
                )
                continue
            split = decompose(view, proposal.symbol, proxy, self.sessions, self.beta_window)
            if split is None:
                self.last_rejections.append(
                    (proposal.symbol, "residual_unmeasurable", f"insufficient overlap with {proxy}")
                )
                continue
            if abs(split.residual_atr) < self.min_residual_atr:
                self.last_rejections.append((proposal.symbol, "beta_move", split.describe()))
                continue
            kept.append(
                EntryProposal(
                    **{
                        **proposal.__dict__,
                        "signal_family": self.family,
                        "thesis": f"{proposal.thesis}; {split.describe()}",
                    }
                )
            )
        return kept
