"""Risk engine orchestrator — the single writer of exit specs (STRATEGY.md §5.C).

propose() ties sizing + clustering + heat + hard gates together into one
SizedProposal or a specific Rejection. Pure math over a supplied frame/regime/
open-positions/earnings snapshot — no network, no clock (the caller supplies
`as_of` and resolves EarningsFact outside the engine).

Price space: entry_ref_price and ATR are RAW prices — the space orders fill in
(matching backtest/simulate.py). Signals decide on adj_close; risk lives where
fills happen.

Time stops are TRADING SESSIONS (canonical — identical to the backtester's
semantics). The ledger's time_stop_date string is a derived calendar-day
display value, never the deadline; exit_params carries the canonical count.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd

from vega.common.atr import compute_atr
from vega.common.doctrine import (
    CALENDAR_DAYS_PER_SESSION,
    DEFAULT_TIME_STOP_SESSIONS,
    GAP_STRESS_MULT,
    PROFIT_TAKE_HALF_AT_R,
    PROFIT_TRAIL_ATR_MULT,
    STOP_ATR_MULT,
)
from vega.ledger.types import Recommendation
from vega.regime.regime import RegimeState
from vega.risk.clusters import classify, contaminates_equity_beta, spy_correlation
from vega.risk.gates import EarningsFact, check_all_gates
from vega.risk.heat import CAPS_R, CAUTION_TOTAL_CAP_R, OpenPositionHeat, cluster_heat, first_breach
from vega.risk.sizing import DEFAULT_RISK_FRACTION, SizingError, compute_qty, compute_stop
from vega.risk.types import Rejection, SizedProposal


def propose(
    symbol: str,
    asset_class: str,
    entry_ref_price: float,
    frame: pd.DataFrame,
    as_of: str,
    equity: float,
    regime: RegimeState,
    open_positions: list[OpenPositionHeat],
    earnings: EarningsFact,
    invalidation: str,
    time_stop_sessions: int = DEFAULT_TIME_STOP_SESSIONS,
    profit_take_half_at_r: float = PROFIT_TAKE_HALF_AT_R,
    risk_fraction: float = DEFAULT_RISK_FRACTION,
) -> SizedProposal | Rejection:
    """`frame` must contain the symbol's raw OHLC history AND SPY's history when
    asset_class is crypto (spy_correlation raises loudly on a SPY-less frame —
    that is a caller bug, not an unmeasurable market fact)."""
    on_date = date.fromisoformat(as_of)

    gate_rejection = check_all_gates(symbol, on_date, time_stop_sessions, regime, earnings)
    if gate_rejection is not None:
        return gate_rejection

    atr = compute_atr(frame, symbol, as_of)
    if atr is None:
        return Rejection(symbol, "insufficient_history", "not enough sessions to compute ATR14")

    try:
        stop = compute_stop(entry_ref_price, atr, asset_class)
        sizing = compute_qty(entry_ref_price, stop, equity, asset_class, risk_fraction)
    except SizingError as exc:
        return Rejection(symbol, "sizing_error", str(exc))

    correlation = spy_correlation(frame, symbol, as_of) if asset_class == "crypto" else None
    contaminates = contaminates_equity_beta(correlation)
    this_position = OpenPositionHeat(
        symbol=symbol,
        asset_class=asset_class,
        qty=sizing.qty,
        entry_price=entry_ref_price,
        current_stop_price=sizing.stop_price,
        contaminates_equity_beta=contaminates,
    )
    r_dollar_unit = risk_fraction * equity
    heat_dollars = cluster_heat([*open_positions, this_position])
    heat_after_r = {k: round(v / r_dollar_unit, 4) for k, v in heat_dollars.items()}
    regime_caution = regime.composite == "caution"
    breach = first_breach(heat_dollars, r_dollar_unit, regime_caution)
    if breach is not None:
        cap_r = CAUTION_TOTAL_CAP_R if (breach == "total" and regime_caution) else CAPS_R[breach]
        return Rejection(
            symbol,
            f"heat_cap:{breach}",
            f"proposal pushes {breach} heat to {heat_after_r[breach]:.2f}R (cap {cap_r}R)",
        )

    exit_params = {
        "stop_atr_mult": STOP_ATR_MULT[asset_class],
        "gap_stress_mult": GAP_STRESS_MULT[asset_class],
        "take_half_at_r": profit_take_half_at_r,
        "trail_atr_mult": PROFIT_TRAIL_ATR_MULT,
        "time_stop_sessions": time_stop_sessions,  # CANONICAL deadline (backtest semantics)
        "worst_case_r_multiple": sizing.worst_case_r_multiple,
        "atr_at_proposal": round(atr, 6),
        "spy_correlation": round(correlation, 4) if correlation is not None else None,
    }

    return SizedProposal(
        symbol=symbol,
        asset_class=asset_class,
        entry_ref_price=entry_ref_price,
        stop_price=sizing.stop_price,
        qty=sizing.qty,
        initial_r_dollars=sizing.initial_r_dollars,
        worst_case_r_multiple=sizing.worst_case_r_multiple,
        time_stop_sessions=time_stop_sessions,
        exit_params=exit_params,
        profit_rule_text=(
            f"half at +{profit_take_half_at_r:g}R, trail remainder via "
            f"{PROFIT_TRAIL_ATR_MULT:g}xATR chandelier stop"
        ),
        invalidation=invalidation,
        cluster=classify(symbol, asset_class),
        contaminates_equity_beta=contaminates,
        heat_after_r=heat_after_r,
    )


def time_stop_date_iso(as_of: str, time_stop_sessions: int) -> str:
    """DERIVED display date for the ledger's human-readable field: sessions
    converted to calendar days (~7/5). The canonical deadline is
    exit_params['time_stop_sessions'] — session count, backtest semantics."""
    calendar_days = math.ceil(time_stop_sessions * CALENDAR_DAYS_PER_SESSION)
    return (date.fromisoformat(as_of) + timedelta(days=calendar_days)).isoformat()


def to_recommendation(
    proposal: SizedProposal,
    thesis: str,
    confidence: float,
    signal_attribution: tuple[str, ...],
    as_of: str,
) -> Recommendation:
    """Bridges a SizedProposal into a valid ledger.types.Recommendation — the
    proposal supplies every risk-owned field; the caller supplies only the idea."""
    return Recommendation(
        symbol=proposal.symbol,
        asset_class=proposal.asset_class,
        direction="long",
        thesis=thesis,
        confidence=confidence,
        horizon_days=proposal.time_stop_sessions,
        entry_ref_price=proposal.entry_ref_price,
        stop_price=proposal.stop_price,
        time_stop_date=time_stop_date_iso(as_of, proposal.time_stop_sessions),
        profit_rule=proposal.profit_rule_text,
        invalidation=proposal.invalidation,
        signal_attribution=signal_attribution,
        exit_params=proposal.exit_params,
        qty=proposal.qty,
    )


def open_position_heat(proposal: SizedProposal) -> OpenPositionHeat:
    """The heat record a caller appends after acting on a proposal — so batch
    callers accumulate heat across proposals instead of sizing each in isolation
    (review finding: the smoke test never exercised multi-position heat)."""
    return OpenPositionHeat(
        symbol=proposal.symbol,
        asset_class=proposal.asset_class,
        qty=proposal.qty,
        entry_price=proposal.entry_ref_price,
        current_stop_price=proposal.stop_price,
        contaminates_equity_beta=proposal.contaminates_equity_beta,
    )
