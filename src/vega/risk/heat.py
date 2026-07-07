"""Portfolio heat: total open risk across positions, in units of R (STRATEGY.md §5.B).

A position's heat is floored at 0 — a stop trailed past breakeven frees heat,
letting winners run while new risk stays capped. Correlation-based crypto
contamination is applied here using a flag the caller precomputes (heat.py
itself stays a pure function of dollar amounts, no DataFrame dependency).
"""

from __future__ import annotations

from dataclasses import dataclass

from vega.risk.clusters import CONTAMINATION_FRACTION, classify
from vega.risk.types import CLUSTERS

CAPS_R = {"total": 6.0, "us_equity_beta": 4.0, "crypto_beta": 2.5, "rates": 3.0, "commodities": 3.0}
CAUTION_TOTAL_CAP_R = 3.0  # STRATEGY.md §5: regime=caution halves the total cap
EPSILON = 1e-9


@dataclass(frozen=True)
class OpenPositionHeat:
    symbol: str
    asset_class: str
    qty: float
    entry_price: float
    current_stop_price: float
    contaminates_equity_beta: bool = False  # precomputed via clusters.spy_correlation


def position_r_dollars(pos: OpenPositionHeat) -> float:
    return max(pos.qty * (pos.entry_price - pos.current_stop_price), 0.0)


def cluster_heat(positions: list[OpenPositionHeat]) -> dict[str, float]:
    """Cluster -> open R in dollars, plus 'total'. Crypto positions correlated to
    SPY beyond the threshold count CONTAMINATION_FRACTION of their R into
    us_equity_beta too (the reason 'separate sleeves' isn't actually safe)."""
    totals: dict[str, float] = dict.fromkeys(CLUSTERS, 0.0)
    totals["total"] = 0.0
    for pos in positions:
        r = position_r_dollars(pos)
        cluster = classify(pos.symbol, pos.asset_class)
        totals[cluster] += r
        totals["total"] += r
        if cluster == "crypto_beta" and pos.contaminates_equity_beta:
            totals["us_equity_beta"] += r * CONTAMINATION_FRACTION
    return {k: round(v, 6) for k, v in totals.items()}


def first_breach(heat: dict[str, float], r_dollar_unit: float, regime_caution: bool) -> str | None:
    """The name of the first cap this heat state breaches, or None."""
    total_cap_r = CAUTION_TOTAL_CAP_R if regime_caution else CAPS_R["total"]
    caps = {**CAPS_R, "total": total_cap_r}
    for cluster, cap_r in caps.items():
        if heat.get(cluster, 0.0) > cap_r * r_dollar_unit + EPSILON:
            return cluster
    return None
