"""Idiosyncratic residual gate — the beta/own-move split and its refusals."""

from __future__ import annotations

import pandas as pd
import pytest

from vega.backtest.market_view import MarketView
from vega.backtest.signals import EntryProposal
from vega.signals.residual import (
    BETA_WINDOW,
    ResidualGate,
    decompose,
    proxy_for,
)

BARS = 130
CLUSTERS = {"AAA": "us_equity_beta", "SPY": "us_equity_beta"}


def _path(start: float, returns: list[float]) -> list[float]:
    prices = [start]
    for r in returns:
        prices.append(prices[-1] * (1.0 + r))
    return prices


def _frame(paths: dict[str, list[float]], spread: float = 1.0) -> pd.DataFrame:
    n = len(next(iter(paths.values())))
    dates = pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    rows = []
    for symbol, prices in paths.items():
        for date, close in zip(dates, prices, strict=True):
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": close,
                    "close": close,
                    "adj_close": close,
                    "high": close + spread,
                    "low": close - spread,
                    "volume": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def _wiggle(n: int) -> list[float]:
    """Deterministic alternating returns — non-zero variance, zero drift."""
    return [0.005 if i % 2 == 0 else -0.005 for i in range(n)]


def _build(beta: float, symbol_shock: list[float], proxy_shock: list[float]) -> MarketView:
    base = _wiggle(BARS - 4)
    proxy = _path(400.0, base + proxy_shock)
    target = _path(100.0, [beta * r for r in base] + symbol_shock)
    frame = _frame({"AAA": target, "SPY": proxy})
    return MarketView(frame, as_of=str(frame["date"].max()))


class _FiresAlways:
    family = "fake_v1"
    version = "1.0"
    promotable = True
    params = {"k": 2.0}

    def scan(self, view: MarketView, universe: list[str]) -> list[EntryProposal]:
        return [
            EntryProposal(
                symbol=s,
                signal_family=self.family,
                signal_version=self.version,
                thesis="fired",
                confidence=0.5,
                invalidation="none",
            )
            for s in universe
        ]


# --------------------------------------------------------------- the split itself


def test_a_move_the_cluster_explains_leaves_almost_no_residual() -> None:
    """Down with everything is beta, not opportunity."""
    view = _build(beta=1.0, symbol_shock=[-0.02, -0.02, -0.02], proxy_shock=[-0.02, -0.02, -0.02])
    split = decompose(view, "AAA", "SPY")
    assert split is not None
    assert split.beta == pytest.approx(1.0, abs=0.05)
    assert split.raw_atr < -1.5  # the raw move looks like a real dislocation
    assert abs(split.residual_atr) < 0.4  # ...and almost all of it was the market


def test_a_move_the_cluster_does_not_explain_survives_as_residual() -> None:
    view = _build(beta=1.0, symbol_shock=[-0.02, -0.02, -0.02], proxy_shock=[0.0, 0.0, 0.0])
    split = decompose(view, "AAA", "SPY")
    assert split is not None
    assert split.residual_atr == pytest.approx(split.raw_atr, abs=0.05)


def test_beta_scales_the_explained_part() -> None:
    """A high-beta name falling twice as far as the market has not moved on its own."""
    view = _build(beta=2.0, symbol_shock=[-0.04, -0.04, -0.04], proxy_shock=[-0.02, -0.02, -0.02])
    split = decompose(view, "AAA", "SPY")
    assert split is not None
    assert split.beta == pytest.approx(2.0, abs=0.1)
    assert abs(split.residual_atr) < 0.4


def test_beta_is_estimated_before_the_shock_not_through_it() -> None:
    """A shock inside the estimation window would set the coefficient used to explain it.

    History says beta is 1.0. The shock has the symbol falling three times as
    far as the proxy. If those three bars leaked into estimation, beta would be
    dragged upward — and the inflated coefficient would then explain the shock
    away as ordinary market beta, which is the exact self-fulfilling error this
    ordering exists to prevent.
    """
    view = _build(beta=1.0, symbol_shock=[-0.06, -0.06, -0.06], proxy_shock=[-0.02, -0.02, -0.02])
    split = decompose(view, "AAA", "SPY", sessions=3, beta_window=BETA_WINDOW)
    assert split is not None
    assert split.beta == pytest.approx(1.0, abs=0.05)

    # Teeth: the same estimator over a window that DOES include the shock.
    bars = view.bars("AAA")
    proxy = view.bars("SPY")
    merged = (
        bars[["date", "adj_close"]]
        .assign(ret_t=bars["adj_close"].pct_change())
        .merge(
            proxy[["date", "adj_close"]].assign(ret_b=proxy["adj_close"].pct_change()),
            on="date",
        )
        .dropna()
        .tail(BETA_WINDOW)
    )
    leaked_beta = merged["ret_t"].cov(merged["ret_b"]) / merged["ret_b"].var()
    assert leaked_beta > split.beta + 0.2  # contamination is real, and it is excluded


def test_a_zero_session_window_is_refused_not_silently_empty() -> None:
    view = _build(beta=1.0, symbol_shock=[-0.02] * 3, proxy_shock=[0.0] * 3)
    with pytest.raises(ValueError, match="sessions must be"):
        decompose(view, "AAA", "SPY", sessions=0)


# --------------------------------------------------------------------- refusals


def test_a_symbol_that_is_its_own_proxy_is_unmeasurable_not_zero() -> None:
    assert proxy_for("SPY", "us_equity_beta") is None
    assert proxy_for("AAA", "us_equity_beta") == "SPY"


def test_a_cluster_with_no_declared_proxy_is_unmeasurable() -> None:
    """commodities has no single ETF that represents it — declared, not guessed."""
    assert proxy_for("GLD", "commodities") is None


def test_thin_overlapping_history_returns_none_never_zero() -> None:
    short = _frame({"AAA": _path(100.0, _wiggle(10)), "SPY": _path(400.0, _wiggle(10))})
    view = MarketView(short, as_of=str(short["date"].max()))
    assert decompose(view, "AAA", "SPY") is None


def test_a_flat_proxy_has_no_variance_and_is_unmeasurable() -> None:
    flat = _frame({"AAA": _path(100.0, _wiggle(BARS)), "SPY": [400.0] * (BARS + 1)})
    view = MarketView(flat, as_of=str(flat["date"].max()))
    assert decompose(view, "AAA", "SPY") is None


# ------------------------------------------------------------------- the gate


def test_gate_rejects_a_beta_move_under_its_own_reason() -> None:
    view = _build(beta=1.0, symbol_shock=[-0.02, -0.02, -0.02], proxy_shock=[-0.02, -0.02, -0.02])
    gate = ResidualGate(_FiresAlways(), CLUSTERS)
    assert gate.scan(view, ["AAA"]) == []
    symbol, reason, detail = gate.last_rejections[0]
    assert (symbol, reason) == ("AAA", "beta_move")
    assert "SPY explains" in detail and "residual" in detail


def test_gate_keeps_an_idiosyncratic_move_and_renames_the_family() -> None:
    view = _build(beta=1.0, symbol_shock=[-0.04, -0.04, -0.04], proxy_shock=[0.0, 0.0, 0.0])
    inner = _FiresAlways()
    gate = ResidualGate(inner, CLUSTERS)
    kept = gate.scan(view, ["AAA"])
    assert len(kept) == 1
    assert kept[0].signal_family == "fake_v1_residual" != inner.family
    assert "residual" in kept[0].thesis
    assert gate.params["min_residual_atr"] > 0


def test_unmeasurable_is_a_rejection_and_is_named_separately() -> None:
    """Missing data must not blur into a real beta rejection, and must not pass."""
    view = _build(beta=1.0, symbol_shock=[-0.04, -0.04, -0.04], proxy_shock=[0.0, 0.0, 0.0])
    gate = ResidualGate(_FiresAlways(), {"AAA": "commodities"})
    assert gate.scan(view, ["AAA"]) == []
    assert gate.last_rejections[0][1] == "residual_unmeasurable"


def test_gate_does_not_mutate_the_inner_family() -> None:
    """A promoted family keeps its exact behaviour and its registry identity."""
    inner = _FiresAlways()
    ResidualGate(inner, CLUSTERS)
    assert inner.family == "fake_v1"
    assert inner.params == {"k": 2.0}
