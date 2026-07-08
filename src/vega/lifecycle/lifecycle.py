"""Signal state machine: candidate -> backtested -> paper-live -> trusted -> retired.

Only paper-live and trusted signals may influence recommendations. Retirement
is reachable from any state (a bad signal is pulled immediately). Retired is
terminal — a reconsidered signal registers as a new family/version, preserving
the append-only audit trail rather than reanimating a dead record.

Auto-demotion always lands on `backtested`, never `candidate`: the backtest
history stays valid — it's the live evidence that failed, so the signal
re-earns paper-live through a fresh human decision (promote_to_paper_live is
always a deliberate human act, never automatic).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from vega.common.appendlog import AppendLog
from vega.lifecycle.rationale import RationaleRegistry

DEFAULT_PATH = Path("data/lifecycle/transitions.jsonl")

STATES = ("candidate", "backtested", "paper-live", "trusted", "retired")
ELIGIBLE_STATES = ("paper-live", "trusted")

TRANSITIONS: dict[str, frozenset[str]] = {
    "candidate": frozenset({"backtested", "retired"}),
    "backtested": frozenset({"paper-live", "retired"}),
    "paper-live": frozenset({"trusted", "backtested", "retired"}),  # ->backtested = demotion
    "trusted": frozenset({"backtested", "retired"}),  # ->backtested = demotion
    "retired": frozenset(),  # terminal
}


class LifecycleError(ValueError):
    pass


class SupportsRuns(Protocol):
    def runs(self, family: str | None = None) -> list[dict[str, Any]]: ...


def is_eligible_state(state: str) -> bool:
    return state in ELIGIBLE_STATES


@dataclass(frozen=True)
class TransitionRecord:
    id: str
    at: str
    signal_family: str
    from_state: str
    to_state: str
    actor: str
    reason: str
    automatic: bool
    justifying_run_id: str | None = None


class LifecycleRegistry:
    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self._log = AppendLog(path)

    def current_state(self, family: str) -> str:
        events = [r for r in self._log.records() if r["signal_family"] == family]
        return events[-1]["to_state"] if events else "candidate"

    def eligible_for_recommendations(self, family: str) -> bool:
        return is_eligible_state(self.current_state(family))

    def history(self, family: str) -> list[dict[str, Any]]:
        return [r for r in self._log.records() if r["signal_family"] == family]

    def _transition(
        self,
        family: str,
        to_state: str,
        actor: str,
        reason: str,
        automatic: bool = False,
        justifying_run_id: str | None = None,
    ) -> TransitionRecord:
        current = self.current_state(family)
        if to_state not in TRANSITIONS.get(current, frozenset()):
            raise LifecycleError(f"illegal transition for {family}: {current} -> {to_state}")
        record = TransitionRecord(
            id=str(uuid.uuid4()),
            at=datetime.now(UTC).isoformat(),
            signal_family=family,
            from_state=current,
            to_state=to_state,
            actor=actor,
            reason=reason,
            automatic=automatic,
            justifying_run_id=justifying_run_id,
        )
        self._log.append({"type": "transition", **record.__dict__})
        return record

    def promote_to_backtested(
        self,
        family: str,
        rationale_registry: RationaleRegistry,
        backtest_registry: SupportsRuns,
        actor: str,
    ) -> TransitionRecord:
        if not rationale_registry.has_rationale(family):
            raise LifecycleError(
                f"{family} has no recorded economic rationale — cannot enter testing"
            )
        passing = [r for r in backtest_registry.runs(family) if r["verdict"] == "pass"]
        if not passing:
            raise LifecycleError(f"{family} has no passing backtest run recorded")
        best = max(passing, key=lambda r: r["aggregate_metrics"].get("sharpe") or float("-inf"))
        return self._transition(
            family,
            "backtested",
            actor,
            f"rationale on file + passing run {best['run_id']}",
            justifying_run_id=best["run_id"],
        )

    def promote_to_paper_live(self, family: str, actor: str) -> TransitionRecord:
        # always a deliberate human act — an agent may propose, only a human promotes
        return self._transition(family, "paper-live", actor, "human promotion to paper-live")

    def promote_to_trusted(self, family: str, actor: str) -> TransitionRecord:
        return self._transition(family, "trusted", actor, "human promotion to trusted")

    def demote(
        self, family: str, actor: str, reason: str, automatic: bool = False
    ) -> TransitionRecord:
        current = self.current_state(family)
        if current not in ("paper-live", "trusted"):
            raise LifecycleError(f"cannot demote {family} from {current}")
        return self._transition(family, "backtested", actor, reason, automatic=automatic)

    def retire(self, family: str, actor: str, reason: str) -> TransitionRecord:
        return self._transition(family, "retired", actor, reason)

    def justifying_run_id(self, family: str) -> str | None:
        """The run that earned this family's current (or most recent) backtested->
        paper-live promotion — used by demotion.py to fetch the confidence band."""
        for record in reversed(self.history(family)):
            if record["to_state"] == "backtested" and record.get("justifying_run_id"):
                return str(record["justifying_run_id"])
        return None
