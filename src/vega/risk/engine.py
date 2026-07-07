"""Risk engine orchestrator — the single writer of exit specs (STRATEGY.md §5.C).

propose() ties sizing + clustering + heat + hard gates together into one
SizedProposal or a specific Rejection. Pure math over a supplied frame/regime/
open-positions snapshot — no network, no clock (the caller supplies `as_of`).
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from vega.common.atr import compute_atr
from vega.ledger.types import Recommendation
from vega.regime.regime import RegimeState
from vega.risk.clusters import classify, contaminates_equity_beta, spy_correlation
from vega.risk.gates import check_all_gates
from vega.risk.heat import CAPS_R, CAUTION_TOTAL_CAP_R, OpenPositionHeat, cluster_heat, first_breach
from vega.risk.sizing import (
    DEFAULT_RISK_FRACTION,
    GAP_STRESS_MULT,
    STOP_ATR_MULT,
    SizingError,
    compute_qty,
    compute_stop,
)
from vega.risk.types import Rejection, SizedProposal

PROFIT_TAKE_HALF_AT_R = 2.0
PROFIT_TRAIL_ATR_MULT = 2.5
DEFAULT_TIME_STOP_DAYS = 15


def propose(
    symbol: str,
    asset_class: str,
    entry_ref_price: float,
    frame: pd.DataFrame,
    as_of: str,
    equity: float,
    regime: RegimeState,
    open_positions: list[OpenPositionHeat],
    invalidation: str,
    time_stop_days: int = DEFAULT_TIME_STOP_DAYS,
    risk_fraction: float = DEFAULT_RISK_FRACTION,
) -> SizedProposal | Rejection:
    on_date = date.fromisoformat(as_of)

    gate_rejection = check_all_gates(symbol, on_date, time_stop_days, regime)
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
    heat_after = cluster_heat([*open_positions, this_position])
    r_dollar_unit = risk_fraction * equity
    regime_caution = regime.composite == "caution"
    breach = first_breach(heat_after, r_dollar_unit, regime_caution)
    if breach is not None:
        cap_r = CAUTION_TOTAL_CAP_R if (breach == "total" and regime_caution) else CAPS_R[breach]
        return Rejection(
            symbol,
            f"heat_cap:{breach}",
            f"proposal pushes {breach} heat to {heat_after[breach]:.2f} "
            f"(cap {cap_r}R = {cap_r * r_dollar_unit:.2f})",
        )

    exit_params = {
        "stop_atr_mult": STOP_ATR_MULT[asset_class],
        "gap_stress_mult": GAP_STRESS_MULT[asset_class],
        "take_half_at_r": PROFIT_TAKE_HALF_AT_R,
        "trail_atr_mult": PROFIT_TRAIL_ATR_MULT,
        "worst_case_r_multiple": sizing.worst_case_r_multiple,
        "atr_at_proposal": round(atr, 6),
        # Calendar-day approximation for the human-readable summary field only;
        # live session-count monitoring is WI-067's concern, not this engine's.
        "time_stop_date_is_calendar_day_approx": True,
    }

    return SizedProposal(
        symbol=symbol,
        asset_class=asset_class,
        entry_ref_price=entry_ref_price,
        stop_price=sizing.stop_price,
        qty=sizing.qty,
        initial_r_dollars=sizing.initial_r_dollars,
        worst_case_r_multiple=sizing.worst_case_r_multiple,
        time_stop_days=time_stop_days,
        exit_params=exit_params,
        profit_rule_text=(
            f"half at +{PROFIT_TAKE_HALF_AT_R:g}R, trail remainder via "
            f"{PROFIT_TRAIL_ATR_MULT:g}xATR chandelier stop"
        ),
        invalidation=invalidation,
        cluster=classify(symbol, asset_class),
        heat_after=heat_after,
    )


def time_stop_date_iso(as_of: str, time_stop_days: int) -> str:
    """Calendar-day approximation for the ledger's human-readable summary field."""
    return (date.fromisoformat(as_of) + timedelta(days=time_stop_days)).isoformat()


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
        horizon_days=proposal.time_stop_days,
        entry_ref_price=proposal.entry_ref_price,
        stop_price=proposal.stop_price,
        time_stop_date=time_stop_date_iso(as_of, proposal.time_stop_days),
        profit_rule=proposal.profit_rule_text,
        invalidation=proposal.invalidation,
        signal_attribution=signal_attribution,
        exit_params=proposal.exit_params,
        qty=proposal.qty,
    )
