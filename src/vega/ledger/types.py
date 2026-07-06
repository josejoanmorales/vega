"""Recommendation schema — the full contract, enforced at construction time.

An entry that lacks any part of its exit spec cannot even be instantiated
(WI-058's ledger consumer, WI-064's risk engine, and WI-067's briefing all
rely on this invariant instead of re-validating).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

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
