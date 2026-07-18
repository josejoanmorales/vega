"""Ranked calls (WI-067): eligible signals -> risk-sized proposals -> ledger.

Entries only. Exit management (sell orders, time-stop/stop/profit-take
monitoring, exit fills, and therefore live demotion) is WI-087's job — this
module never mutates or closes a position, only opens one.

A live scan only ever runs the EXACT parameterization that justified a
family's promotion (`justifying_params` on the backtest run that earned
`promote_to_backtested`) — running different parameters live would decouple
the paper track record from the evidence that licensed it. A family that is
eligible but missing that evidence — or missing a registered signal class —
is a bookkeeping bug, not a "just use defaults" situation: `build_calls`
raises rather than guess (WI-067 review: iteration is driven by the lifecycle
registry, never by the class map, so an unmapped eligible family is loud).

Idempotency (WI-067 review): a symbol with an active position — filled, or
pending from THIS session — is never re-proposed (`already_held` rejection),
so same-day re-runs and multi-family overlap cannot stack entries.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
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
from vega.execution.exits import reconstruct_positions, to_heat
from vega.ledger.store import LedgerStore
from vega.lifecycle.lifecycle import LifecycleRegistry, is_eligible_state
from vega.regime.regime import RegimeState
from vega.risk.engine import open_position_heat, propose, to_recommendation
from vega.risk.gates import EarningsFact
from vega.risk.types import Rejection, SizedProposal
from vega.signals.breakout_volume import BreakoutVolumeSignal
from vega.signals.oversold_reversion import OversoldReversionSignal
from vega.signals.trend_pullback import TrendPullbackSignal

FAMILY_SIGNALS: dict[str, type[Signal]] = {
    TrendPullbackSignal.family: TrendPullbackSignal,
    BreakoutVolumeSignal.family: BreakoutVolumeSignal,
    OversoldReversionSignal.family: OversoldReversionSignal,
}

# Widest lookback any consumer needs: signal max (oversold LOOKBACK=115) and
# the crypto/SPY correlation window (90+1) both sit under this, converted to
# calendar days with margin for holidays.
FRAME_LOOKBACK_CALENDAR_DAYS = 220


class CallsError(RuntimeError):
    """Eligibility bookkeeping is broken — an eligible family has no auditable
    justifying evidence (or no registered signal class). Never silently fall
    back to unvalidated parameters."""


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


def load_signal_frame(as_of: str, root: Path = snapshot.DATA_ROOT) -> pd.DataFrame:
    """Raw OHLCV + adj_close for every yfinance-sourced (equity/ETF) symbol —
    the one frame signal decisions (adj_close), risk sizing (raw OHLC), and
    the exit monitor (`open`, for gap-stops — WI-087) all read from, loaded
    once per run. Date-bounded at the source: `<= as_of` enforces the PIT
    contract before pandas ever sees a row, and the lookback floor keeps the
    frame O(lookback) instead of O(store history) as the store grows."""
    floor = (date.fromisoformat(as_of) - timedelta(days=FRAME_LOOKBACK_CALENDAR_DAYS)).isoformat()
    con = duckdb.connect(str(root / "vega.duckdb"), read_only=True)
    try:
        return con.execute(
            "SELECT symbol, date, open, close, high, low, adj_close, volume "
            "FROM bars WHERE source = 'yfinance' AND date <= ? AND date >= ?",
            [as_of, floor],
        ).df()
    finally:
        con.close()


def _eligible_families(
    lifecycle: LifecycleRegistry, backtest_registry: BacktestRegistry
) -> list[EligibleFamily]:
    runs_by_id = {r["run_id"]: r for r in backtest_registry.runs()}
    out = []
    for family in lifecycle.families():
        state = lifecycle.current_state(family)
        if not is_eligible_state(state):
            continue
        if family not in FAMILY_SIGNALS:
            raise CallsError(
                f"{family} is eligible ({state}) but has no signal class registered in "
                "FAMILY_SIGNALS — the promoted family would be silently untradeable"
            )
        run_id = lifecycle.justifying_run_id(family)
        if run_id is None:
            raise CallsError(
                f"{family} is eligible ({state}) but has no justifying_run_id on "
                "record — refusing to scan on unvalidated parameters"
            )
        run = runs_by_id.get(run_id)
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


def _rank_key(item: tuple[EntryProposal, float]) -> tuple[float, float, str]:
    proposal, family_dev_sharpe = item
    # confidence DESC, then family dev-Sharpe DESC (moot with one live family,
    # honest with >1 — holdout is never used for ranking), then symbol ASC as
    # the total-order tiebreak so rendering is fully deterministic.
    return (-proposal.confidence, -family_dev_sharpe, proposal.symbol)


def _no_trade_reason(n_proposals: int, rejections: list[RenderedRejection]) -> str:
    """Derived from what actually happened — never re-derives gate conditions
    (WI-067 review: claiming 'entries blocked by FOMC' on a day with zero
    proposals asserted a gate that never fired)."""
    if n_proposals == 0:
        return "no qualifying setups found today across the eligible signal family(ies)"
    counts = Counter(r.reason for r in rejections)
    dominant = ", ".join(f"{reason} ({n})" for reason, n in counts.most_common())
    return f"{n_proposals} candidate(s) considered, none cleared: {dominant} — see rejections below"


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
    exited_today: frozenset[str] = frozenset(),
) -> CallsResult:
    """One pre-market pass: scan every paper-live+ family (at its justified
    parameterization) over `frame` as of `as_of`, risk-size each proposal in
    rank order (heat accumulates across accepted calls, so the caps bound the
    day's count), and append every accepted call to `ledger`. Same-day paper
    execution is the caller's job (`execution.executor.execute_pending`).

    `universe_entries` defaults to the committed universe artifact; injectable
    for tests (a synthetic frame's symbols are rarely in the real universe).
    `exited_today` (WI-087): symbols the exit monitor just submitted a sell
    for THIS run — simulate.py never re-enters a symbol the same session it
    exits, so neither does this.
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

    open_positions = [to_heat(p) for p in reconstruct_positions(ledger, frame, as_of)]
    held = {p.symbol for p in open_positions}
    calls: list[RenderedCall] = []
    rejections: list[RenderedRejection] = []

    for proposal, _dev_sharpe in scored_proposals:
        symbol = proposal.symbol
        if symbol in held:
            rejections.append(
                RenderedRejection(
                    symbol,
                    proposal.signal_family,
                    "already_held",
                    "an active position or same-session pending call exists — entries never stack",
                )
            )
            continue
        if symbol in exited_today:
            rejections.append(
                RenderedRejection(
                    symbol,
                    proposal.signal_family,
                    "same_day_exit",
                    "this symbol was just exited this run — no same-session re-entry "
                    "(matches simulate.py)",
                )
            )
            continue
        bars = view.bars(symbol)
        if bars.empty or str(bars["date"].iloc[-1]) != as_of:
            rejections.append(
                RenderedRejection(
                    symbol,
                    proposal.signal_family,
                    "stale_price",
                    f"no raw close for {symbol} on {as_of}",
                )
            )
            continue
        entry_ref_price = float(bars["close"].iloc[-1])
        asset_class = asset_class_by_symbol.get(symbol, "equity")
        # per-symbol frame: propose()/ATR must never full-frame-scan per candidate;
        # crypto sizing additionally needs SPY history for the contamination check
        candidate_frame = (
            pd.concat([bars, view.bars("SPY")], ignore_index=True)
            if asset_class == "crypto"
            else bars
        )

        result = propose(
            symbol=symbol,
            asset_class=asset_class,
            entry_ref_price=entry_ref_price,
            frame=candidate_frame,
            as_of=as_of,
            equity=equity,
            regime=regime,
            open_positions=open_positions,
            earnings=earnings_lookup(symbol, asset_class),  # network OUTSIDE this loop's math
            invalidation=proposal.invalidation,
            time_stop_sessions=proposal.time_stop_days,
            profit_take_half_at_r=proposal.profit_take_half_at_r,
            stop_atr_mult=proposal.stop_atr_mult,
            profit_trail_atr_mult=proposal.profit_trail_atr_mult,
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
        held.add(symbol)

        calls.append(
            RenderedCall(
                rank=len(calls) + 1,
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

    reason = None
    if not calls:
        reason = _no_trade_reason(len(scored_proposals), rejections)

    return CallsResult(
        as_of=as_of,
        eligible_families=tuple(eligible),
        calls=tuple(calls),
        rejections=tuple(rejections),
        no_trade_reason=reason,
    )
