"""Recommendation schema — the full contract, enforced at construction time.

An entry that lacks any part of its exit spec cannot even be instantiated
(WI-058's ledger consumer, WI-064's risk engine, and WI-067's briefing all
rely on this invariant instead of re-validating).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

ASSET_CLASSES = ("equity", "etf", "crypto")
DIRECTIONS = ("long", "exit")
OVERRIDE_ACTIONS = ("skip", "resize", "early_exit")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Recommendation:
    symbol: str
    asset_class: str
    direction: str
    thesis: str
    confidence: float
    horizon_days: int
    entry_ref_price: float
    # exit spec — all four mandatory (STRATEGY.md §5)
    stop_price: float
    time_stop_date: str
    profit_rule: str
    invalidation: str
    signal_attribution: tuple[str, ...] = ()
    supersedes: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_now_iso)
    # Structured exit-spec params + sized qty (WI-064's risk engine is the sole writer of
    # these; profit_rule/stop_price/time_stop_date above stay the human-auditable summary
    # every recommendation already required — this is additive, append-only-compatible
    # schema growth, not a contract change).
    exit_params: dict[str, Any] | None = None
    qty: float | None = None
    # Decision session (WI-067, additive): the store date whose close produced this
    # call. Execution honors the backtest's T+1-open fill model — a rec whose as_of
    # is no longer the current session must EXPIRE (surfaced, never late-filled).
    as_of: str | None = None

    def __post_init__(self) -> None:
        _require(bool(self.symbol), "symbol is required")
        _require(self.asset_class in ASSET_CLASSES, f"asset_class must be one of {ASSET_CLASSES}")
        _require(self.direction in DIRECTIONS, f"direction must be one of {DIRECTIONS}")
        _require(bool(self.thesis.strip()), "thesis is required")
        _require(0.0 <= self.confidence <= 1.0, "confidence must be within [0, 1]")
        _require(self.horizon_days > 0, "horizon_days must be positive")
        _require(self.entry_ref_price > 0, "entry_ref_price must be positive")
        _require(self.stop_price > 0, "exit spec: stop_price is required and positive")
        date.fromisoformat(self.time_stop_date)  # raises on malformed exit spec dates
        _require(bool(self.profit_rule.strip()), "exit spec: profit_rule is required")
        _require(bool(self.invalidation.strip()), "exit spec: invalidation is required")
        if self.direction == "long":
            _require(
                len(self.signal_attribution) > 0,
                "signal_attribution is required for long recommendations",
            )
        if self.qty is not None:
            _require(self.qty > 0, "qty must be positive when provided")
        if self.as_of is not None:
            date.fromisoformat(self.as_of)  # raises on a malformed decision session
