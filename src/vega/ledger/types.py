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
# WI-229. `reduce` and `avoid` are how a bearish view is expressed without shorting.
# Shorting was refused on payoff shape, not squeamishness: unbounded loss, a variable
# borrow cost, recall at the worst possible moment, dividend liability — and the worst
# asymmetry available, since the name you are short on a rumour is also the one that can
# open acquired at a premium. STRATEGY.md §3's long/flat-only rule stands. Most of the
# value survives anyway: being able to say "do not open this" and "cut this one down"
# turns negative evidence into a recorded, gradeable decision instead of silence — which
# also means the system can finally be scored on the calls it DECLINED to make.
DIRECTIONS = ("long", "exit", "reduce", "avoid")

# The only directions that may acquire inventory. Every path that places a BUY, reserves
# portfolio heat, or enters the realised track record keys off THIS set rather than off a
# bare `== "long"` literal, so a direction added later cannot quietly inherit those powers
# by being spelled differently. `avoid` and `reduce` are deliberately absent, and `exit`
# and `reduce` only ever reduce an existing position — nothing here can open a short.
POSITION_DIRECTIONS = frozenset({"long"})

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
    # Exit spec — mandatory for any direction that HOLDS (STRATEGY.md §5); enforced per
    # direction in __post_init__ rather than by field-level requiredness, so `avoid` and
    # `reduce` are not forced to invent numbers that would then read as real levels.
    stop_price: float = 0.0
    time_stop_date: str = ""
    profit_rule: str = ""
    invalidation: str = ""
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
    # WI-229, additive (append-only: old records parse with these as None).
    target_fraction: float | None = None  # reduce: the fraction of the position to close
    expires_on: str | None = None  # avoid: the date after which the decision no longer applies

    def __post_init__(self) -> None:
        _require(bool(self.symbol), "symbol is required")
        _require(self.asset_class in ASSET_CLASSES, f"asset_class must be one of {ASSET_CLASSES}")
        _require(self.direction in DIRECTIONS, f"direction must be one of {DIRECTIONS}")
        _require(bool(self.thesis.strip()), "thesis is required")
        _require(0.0 <= self.confidence <= 1.0, "confidence must be within [0, 1]")
        _require(self.horizon_days > 0, "horizon_days must be positive")
        _require(self.entry_ref_price > 0, "entry_ref_price must be positive")
        if self.qty is not None:
            _require(self.qty > 0, "qty must be positive when provided")
        if self.as_of is not None:
            date.fromisoformat(self.as_of)  # raises on a malformed decision session
        # Requirements are per-direction, not one blanket rule (WI-229). The four-part exit
        # spec is mandatory for a position that will be HELD; demanding a stop price from a
        # record whose entire content is "we did not take this" would force a meaningless
        # number into the ledger, and a meaningless number is worse than an absent one.
        if self.direction in ("long", "exit"):
            self._require_exit_spec()
        elif self.direction == "reduce":
            self._require_reduce_spec()
        elif self.direction == "avoid":
            self._require_avoid_spec()
        if self.direction == "long":
            _require(
                len(self.signal_attribution) > 0,
                "signal_attribution is required for long recommendations",
            )

    def _require_exit_spec(self) -> None:
        _require(self.stop_price > 0, "exit spec: stop_price is required and positive")
        date.fromisoformat(self.time_stop_date)  # raises on malformed exit spec dates
        _require(bool(self.profit_rule.strip()), "exit spec: profit_rule is required")
        _require(bool(self.invalidation.strip()), "exit spec: invalidation is required")

    def _require_reduce_spec(self) -> None:
        """A reduce cuts an existing position down; the remainder keeps the stop it already
        has, so it needs a SIZE and a reason, not a fresh exit spec."""
        _require(
            self.target_fraction is not None and 0.0 < self.target_fraction <= 1.0,
            "reduce: target_fraction is required and must be within (0, 1]",
        )
        _require(bool(self.invalidation.strip()), "reduce: invalidation is required")

    def _require_avoid_spec(self) -> None:
        """An avoid is a decision NOT to hold, so it must be structurally unable to look
        like something tradeable — the absences are asserted, not merely permitted."""
        _require(bool(self.expires_on), "avoid: expires_on is required")
        date.fromisoformat(str(self.expires_on))
        _require(self.qty is None, "avoid: qty must be absent — nothing is being sized")
        _require(self.stop_price == 0.0, "avoid: stop_price must be absent — nothing is held")
        _require(not self.profit_rule.strip(), "avoid: profit_rule must be absent")
        _require(bool(self.invalidation.strip()), "avoid: invalidation is required")
