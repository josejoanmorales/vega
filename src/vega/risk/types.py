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
    entry_ref_price: float
    stop_price: float
    qty: float
    initial_r_dollars: float  # qty * (entry - stop): the risk actually taken
    worst_case_r_multiple: float  # gap-stressed worst case, in units of initial_r_dollars
    time_stop_days: int
    exit_params: dict[str, Any]
    profit_rule_text: str
    invalidation: str
    cluster: str
    heat_after: dict[str, float]  # cluster -> R after this proposal, incl. "total"


@dataclass(frozen=True)
class Rejection:
    symbol: str
    reason: str  # short code, e.g. "regime_risk_off", "heat_cap:crypto_beta"
    detail: str
