"""Grading the calls Vega DECLINED to make (WI-229).

An `avoid` is the only recommendation that leaves no trace in the paper account:
nothing is bought, nothing is sold, no fill is ever written. Left ungraded it would be
the one kind of call the system could make for free — and a call that cannot be wrong
is not a call. This module scores it against what the position would have done.

TWO BOUNDARIES, both structural rather than advisory:

1. **A counterfactual return is never realised P&L.** `closed_round_trips` filters on
   `POSITION_DIRECTIONS`, so an avoid cannot reach the live Sharpe, the demotion bands,
   or the paper portfolio's equity curve. Every outcome here carries
   `counterfactual=True` so the separation survives being passed around.

2. **This is not alpha.** It measures what holding would have returned, gross, over a
   fixed window. It does NOT model where the capital went instead, so a "correct" avoid
   is evidence the system read the setup, not evidence it made money. Reporting it as
   performance would be exactly the sort of flattering arithmetic STRATEGY.md §4 exists
   to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vega.ledger.store import LedgerStore

AVOID = "avoid"


@dataclass(frozen=True)
class AvoidedOutcome:
    """What a declined position would have done. `counterfactual` is always True and is
    carried explicitly so no consumer can mistake this for a realised trade."""

    symbol: str
    decided_on: str
    expires_on: str
    ref_price: float
    outcome_price: float | None
    outcome_session: str | None
    return_pct: float | None
    resolved: bool
    counterfactual: bool = True

    @property
    def was_right(self) -> bool | None:
        """True when declining avoided a loss. None while the window is still open —
        never False, because "not yet resolved" and "wrong" are different facts."""
        if not self.resolved or self.return_pct is None:
            return None
        return self.return_pct < 0.0


@dataclass(frozen=True)
class AvoidScorecard:
    n_resolved: int
    n_open: int
    n_right: int
    avg_avoided_return_pct: float | None  # mean of resolved windows; negative = declining paid

    @property
    def hit_rate(self) -> float | None:
        """Reported, never optimized — same rule the rest of the system lives under."""
        return self.n_right / self.n_resolved if self.n_resolved else None


def _close_on_or_before(frame: pd.DataFrame, symbol: str, session: str) -> tuple[str, float] | None:
    bars = frame[(frame["symbol"] == symbol) & (frame["date"] <= session)]
    if bars.empty:
        return None
    row = bars.sort_values("date").iloc[-1]
    close = row["adj_close"] if "adj_close" in bars.columns else row["close"]
    if pd.isna(close):
        return None
    return str(row["date"]), float(close)


def score_avoids(ledger: LedgerStore, frame: pd.DataFrame, as_of: str) -> list[AvoidedOutcome]:
    """One outcome per live `avoid`, resolved only once its window has closed.

    An avoid whose `expires_on` is still in the future is returned UNRESOLVED rather
    than scored against a partial window — grading a decision before its own stated
    horizon is the cheapest way to manufacture a flattering number.
    """
    outcomes: list[AvoidedOutcome] = []
    for rec in ledger.latest():
        if rec.get("direction") != AVOID:
            continue
        expires_on = str(rec.get("expires_on") or "")
        decided_on = str(rec.get("as_of") or rec.get("created_at", ""))[:10]
        ref_price = float(rec["entry_ref_price"])
        if not expires_on or expires_on > as_of:
            outcomes.append(
                AvoidedOutcome(
                    symbol=rec["symbol"],
                    decided_on=decided_on,
                    expires_on=expires_on,
                    ref_price=ref_price,
                    outcome_price=None,
                    outcome_session=None,
                    return_pct=None,
                    resolved=False,
                )
            )
            continue
        found = _close_on_or_before(frame, rec["symbol"], expires_on)
        if found is None:
            # No bar anywhere in the window: unmeasurable, and unmeasurable is not zero.
            outcomes.append(
                AvoidedOutcome(
                    symbol=rec["symbol"],
                    decided_on=decided_on,
                    expires_on=expires_on,
                    ref_price=ref_price,
                    outcome_price=None,
                    outcome_session=None,
                    return_pct=None,
                    resolved=False,
                )
            )
            continue
        session, close = found
        outcomes.append(
            AvoidedOutcome(
                symbol=rec["symbol"],
                decided_on=decided_on,
                expires_on=expires_on,
                ref_price=ref_price,
                outcome_price=close,
                outcome_session=session,
                return_pct=(close - ref_price) / ref_price * 100.0,
                resolved=True,
            )
        )
    return outcomes


def scorecard(outcomes: list[AvoidedOutcome]) -> AvoidScorecard:
    resolved = [o for o in outcomes if o.resolved and o.return_pct is not None]
    returns = [o.return_pct for o in resolved if o.return_pct is not None]
    return AvoidScorecard(
        n_resolved=len(resolved),
        n_open=len(outcomes) - len(resolved),
        n_right=sum(1 for o in resolved if o.was_right),
        avg_avoided_return_pct=(sum(returns) / len(returns)) if returns else None,
    )
