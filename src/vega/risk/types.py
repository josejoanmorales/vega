"""Risk-engine output contract: a proposal is sized or it is rejected — never
partially either. Both carry enough detail to audit the decision after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CLUSTERS = ("us_equity_beta", "rates", "commodities", "crypto_beta")


@dataclass(frozen=True)
class SizedProposal:
    symbol: str
    asset_class: str
    entry_ref_price: float  # RAW price space (the space fills happen in)
    stop_price: float
    qty: float
    initial_r_dollars: float  # qty * (entry - stop): the risk actually taken, in dollars
    worst_case_r_multiple: float  # gap-stressed worst case, in R multiples (<= 2.0)
    time_stop_sessions: int  # TRADING sessions — canonical, matches backtest semantics
    exit_params: dict[str, Any]
    profit_rule_text: str
    invalidation: str
    cluster: str
    contaminates_equity_beta: bool  # crypto correlated >0.5 to SPY (always False otherwise)
    heat_after_r: dict[str, float]  # cluster -> open risk in R MULTIPLES incl. "total"
    # (directly comparable to heat.CAPS_R — dollar heat is an engine-internal detail)


@dataclass(frozen=True)
class Rejection:
    symbol: str
    reason: str  # short code, e.g. "regime_risk_off", "heat_cap:crypto_beta", "earnings_unknown"
    detail: str
