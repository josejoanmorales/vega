"""Ranked calls (WI-067): eligible signals -> risk-sized proposals -> ledger.

Entries only. Exit management (sell orders, time-stop/stop/profit-take
monitoring, exit fills, and therefore live demotion) is WI-087's job — this
module never mutates or closes a position, only opens one.

A live scan only ever runs the EXACT parameterization that justified a
family's promotion (`justifying_params` on the backtest run that earned
`promote_to_backtested`) — running different parameters live would decouple
the paper track record from the evidence that licensed it (WI-066 review
finding #2: the registry recorded no grid params, so a promoted family's
live behavior could silently drift from what was actually validated). A
family that is eligible but missing that evidence is a bookkeeping bug, not
a "just use defaults" situation — `build_calls` raises rather than guess.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from vega.backtest.market_view import MarketView
from vega.backtest.registry import BacktestRegistry
from vega.backtest.signals import EntryProposal, Signal
from vega.data import snapshot
from vega.data.types import UniverseEntry
from vega.data.universe import load_universe
from vega.data.universe import symbols as universe_symbols
from vega.ledger.store import LedgerStore
from vega.lifecycle.lifecycle import LifecycleRegistry, is_eligible_state
from vega.regime.calendar import in_macro_window
from vega.regime.regime import RegimeState
from vega.risk.clusters import contaminates_equity_beta
from vega.risk.engine import open_position_heat, propose, to_recommendation
from vega.risk.gates import EarningsFact
from vega.risk.heat import OpenPositionHeat
from vega.risk.types import Rejection, SizedProposal
from vega.signals.breakout_volume import BreakoutVolumeSignal
from vega.signals.oversold_reversion import OversoldReversionSignal
from vega.signals.trend_pullback import TrendPullbackSignal

FAMILY_SIGNALS: dict[str, type[Signal]] = {
    TrendPullbackSignal.family: TrendPullbackSignal,
    BreakoutVolumeSignal.family: BreakoutVolumeSignal,
    OversoldReversionSignal.family: OversoldReversionSignal,
}


class CallsError(RuntimeError):
    """Eligibility bookkeeping is broken — an eligible family has no auditable
    justifying evidence. Never silently fall back to unvalidated parameters."""


@dataclass(frozen=True)
class EligibleFamily:
    family: str
    state: str
    justifying_run_id: str
    justifying_params: dict[str, Any]
    dev_sharpe: float  # trade-weighted dev Sharpe of the justifying run; -inf if unrecorded


@dataclass(frozen=True)
class RenderedCall:
    rank: int
    symbol: str
    family: str
    version: str
    thesis: str
    qty: float
    entry_ref_price: float
    stop_price: float
    worst_case_r_multiple: float
    time_stop_sessions: int
    time_stop_date: str
    profit_rule: str
    invalidation: str
    heat_after_r: dict[str, float]
    ref_id: str


@dataclass(frozen=True)
class RenderedRejection:
    symbol: str
    family: str
    reason: str
    detail: str


@dataclass(frozen=True)
class CallsResult:
    as_of: str
    eligible_families: tuple[EligibleFamily, ...]
    calls: tuple[RenderedCall, ...]
    rejections: tuple[RenderedRejection, ...]
    no_trade_reason: str | None  # set only when eligible_families is non-empty and calls is empty


def load_signal_frame(root: Path = snapshot.DATA_ROOT) -> pd.DataFrame:
    """Raw OHLCV + adj_close for every yfinance-sourced (equity/ETF) symbol —
    the one frame both signal decisions (adj_close) and risk sizing (raw
    OHLC) read from, matching WI-064's price-space contract."""
    con = duckdb.connect(str(root / "vega.duckdb"), read_only=True)
    try:
        return con.execute(
            "SELECT symbol, date, close, high, low, adj_close, volume "
            "FROM bars WHERE source = 'yfinance'"
        ).df()
    finally:
        con.close()


def _eligible_families(
    lifecycle: LifecycleRegistry, backtest_registry: BacktestRegistry
) -> list[EligibleFamily]:
    out = []
    for family in FAMILY_SIGNALS:
        state = lifecycle.current_state(family)
        if not is_eligible_state(state):
            continue
        run_id = lifecycle.justifying_run_id(family)
        if run_id is None:
            raise CallsError(
                f"{family} is eligible ({state}) but has no justifying_run_id on "
                "record — refusing to scan on unvalidated parameters"
            )
        run = next((r for r in backtest_registry.runs(family) if r["run_id"] == run_id), None)
        if run is None:
            raise CallsError(f"{family}'s justifying run {run_id} is not in the backtest registry")
        params = run.get("signal_params")
        if not params:
            raise CallsError(
                f"{family}'s justifying run {run_id} has no signal_params recorded — "
                "cannot instantiate the validated parameterization"
            )
        sharpe = run.get("aggregate_metrics", {}).get("sharpe")
        out.append(
            EligibleFamily(
                family=family,
                state=state,
                justifying_run_id=run_id,
                justifying_params=dict(params),
                dev_sharpe=float("-inf") if sharpe is None else float(sharpe),
            )
        )
    return out


def _open_positions_from_ledger(ledger: LedgerStore) -> list[OpenPositionHeat]:
    """Every FILLED long is treated as an open position — the ledger has no
    concept of an exit fill yet (WI-087's job), so a filled long stays "open"
    until then. Uses the ORIGINAL stop price (no trailing-stop tracking until
    WI-087 either); the original stop is always >= a trailed stop, so this
    overstates heat rather than understates it — the conservative direction."""
    fills_by_ref = {f["ref_id"]: f for f in ledger.fills()}
    positions = []
    for rec in ledger.latest():
        if rec["direction"] != "long":
            continue
        fill = fills_by_ref.get(rec["id"])
        qty = (fill or {}).get("qty") or rec.get("qty")
        if fill is None or not qty:
            continue
        exit_params = rec.get("exit_params") or {}
        positions.append(
            OpenPositionHeat(
                symbol=rec["symbol"],
                asset_class=rec["asset_class"],
                qty=float(qty),
                entry_price=float(rec["entry_ref_price"]),
                current_stop_price=float(rec["stop_price"]),
                contaminates_equity_beta=contaminates_equity_beta(
                    exit_params.get("spy_correlation")
                ),
            )
        )
    return positions


def _rank_key(item: tuple[EntryProposal, float]) -> tuple[float, float, str]:
    proposal, family_dev_sharpe = item
    # confidence DESC, then family dev-Sharpe DESC (moot with one live family,
    # honest with >1 — holdout is never used for ranking), then symbol ASC as
    # the total-order tiebreak so rendering is fully deterministic.
    return (-proposal.confidence, -family_dev_sharpe, proposal.symbol)


def _no_trade_reason(
    regime: RegimeState,
    on_date: date,
    n_proposals: int,
    rejections: list[RenderedRejection],
) -> str:
    if regime.composite == "risk_off":
        return f"regime composite is risk_off as of {regime.as_of} — no entries permitted"
    if in_macro_window(on_date, days_before=1):
        return "a scheduled FOMC/CPI event falls within T-1..T — entries blocked"
    if n_proposals == 0:
        return "no qualifying setups found today across the eligible signal family(ies)"
    return (
        f"{n_proposals} candidate(s) considered; none cleared risk gates/heat caps — "
        "see rejections below"
    )


def build_calls(
    frame: pd.DataFrame,
    as_of: str,
    equity: float,
    regime: RegimeState,
    ledger: LedgerStore,
    lifecycle: LifecycleRegistry | None = None,
    backtest_registry: BacktestRegistry | None = None,
    universe_entries: list[UniverseEntry] | None = None,
    earnings_lookup: Callable[[str, str], EarningsFact] = EarningsFact.lookup,
) -> CallsResult:
    """One pre-market pass: scan every paper-live+ family (at its justified
    parameterization) over `frame` as of `as_of`, risk-size each proposal in
    rank order (heat accumulates across accepted calls, so the caps bound the
    day's count), and append every accepted call to `ledger`. Same-day paper
    execution is the caller's job (`execution.executor.execute_pending`).

    `universe_entries` defaults to the committed universe artifact; injectable
    for tests (a synthetic frame's symbols are rarely in the real universe).
    """
    lifecycle = lifecycle or LifecycleRegistry()
    backtest_registry = backtest_registry or BacktestRegistry()

    eligible = _eligible_families(lifecycle, backtest_registry)
    if not eligible:
        return CallsResult(
            as_of=as_of, eligible_families=(), calls=(), rejections=(), no_trade_reason=None
        )

    universe_entries = universe_entries if universe_entries is not None else load_universe()
    universe = universe_symbols(universe_entries, "equity", "etf")
    asset_class_by_symbol = {e.symbol: e.asset_class for e in universe_entries}
    view = MarketView(frame, as_of)

    scored_proposals: list[tuple[EntryProposal, float]] = []
    for fam in eligible:
        signal = FAMILY_SIGNALS[fam.family](**fam.justifying_params)
        for proposal in signal.scan(view, universe):
            scored_proposals.append((proposal, fam.dev_sharpe))
    scored_proposals.sort(key=_rank_key)

    open_positions = _open_positions_from_ledger(ledger)
    calls: list[RenderedCall] = []
    rejections: list[RenderedRejection] = []
    raw_by_symbol = {
        symbol: sub.sort_values("date")
        for symbol, sub in frame[frame["date"] <= as_of].groupby("symbol")
    }

    for proposal, _dev_sharpe in scored_proposals:
        symbol = proposal.symbol
        sub = raw_by_symbol.get(symbol)
        if sub is None or str(sub["date"].iloc[-1]) != as_of:
            rejections.append(
                RenderedRejection(
                    symbol,
                    proposal.signal_family,
                    "stale_price",
                    f"no raw close for {symbol} on {as_of}",
                )
            )
            continue
        entry_ref_price = float(sub["close"].iloc[-1])
        asset_class = asset_class_by_symbol.get(symbol, "equity")

        result = propose(
            symbol=symbol,
            asset_class=asset_class,
            entry_ref_price=entry_ref_price,
            frame=frame,
            as_of=as_of,
            equity=equity,
            regime=regime,
            open_positions=open_positions,
            earnings=earnings_lookup(symbol, asset_class),  # network OUTSIDE this loop's math
            invalidation=proposal.invalidation,
            time_stop_sessions=proposal.time_stop_days,
            profit_take_half_at_r=proposal.profit_take_half_at_r,
        )
        if isinstance(result, Rejection):
            rejections.append(
                RenderedRejection(symbol, proposal.signal_family, result.reason, result.detail)
            )
            continue
        assert isinstance(result, SizedProposal)  # noqa: S101 — Rejection handled above

        rec = to_recommendation(
            result,
            thesis=proposal.thesis,
            confidence=proposal.confidence,
            signal_attribution=(f"{proposal.signal_family}:{proposal.signal_version}",),
            as_of=as_of,
        )
        ref_id = ledger.append(rec)
        open_positions.append(open_position_heat(result))  # higher-ranked calls claim heat first

        calls.append(
            RenderedCall(
                rank=0,  # recomputed below over accepted calls only
                symbol=symbol,
                family=proposal.signal_family,
                version=proposal.signal_version,
                thesis=proposal.thesis,
                qty=result.qty,
                entry_ref_price=result.entry_ref_price,
                stop_price=result.stop_price,
                worst_case_r_multiple=result.worst_case_r_multiple,
                time_stop_sessions=result.time_stop_sessions,
                time_stop_date=rec.time_stop_date,
                profit_rule=result.profit_rule_text,
                invalidation=result.invalidation,
                heat_after_r=result.heat_after_r,
                ref_id=ref_id,
            )
        )

    calls = [replace(c, rank=i) for i, c in enumerate(calls, start=1)]

    reason = None
    if not calls:
        n = len(scored_proposals)
        reason = _no_trade_reason(regime, date.fromisoformat(as_of), n, rejections)

    return CallsResult(
        as_of=as_of,
        eligible_families=tuple(eligible),
        calls=tuple(calls),
        rejections=tuple(rejections),
        no_trade_reason=reason,
    )
