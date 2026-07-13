"""Signal state machine: candidate -> backtested -> paper-live -> trusted -> retired.

Only paper-live and trusted signals may influence recommendations. Retirement
is reachable from any state (a bad signal is pulled immediately). Retired is
terminal — a reconsidered signal registers as a new family/version, preserving
the append-only audit trail rather than reanimating a dead record.

Auto-demotion always lands on `backtested`, never `candidate`: the backtest
history stays valid — it's the live evidence that failed. Re-promotion to
paper-live is a fresh human decision, and a demoted family can (and should)
RE-JUSTIFY: `backtested -> backtested` is a legal self-transition that attaches
a NEW justifying run, so the demotion band reflects post-demotion evidence
instead of the stale band the family already breached (review finding).

VERSION POLICY (stated, not implicit — review finding): lifecycle state is a
FAMILY-level decision by design. Versions of a family share its state; the
version whose run justified a promotion is recorded on the transition for
audit. A materially different algorithm must register as a NEW family — the
same doctrine that makes `retired` terminal.

Human-act enforcement: transitions that grant or extend trust (paper-live,
trusted, retire) require an actor with the `human:` prefix. This is a prefix
contract, not cryptographic identity — the same solo-scale posture as Caral's
WI-042 (role tokens are the Phase-2 hardening); it converts a silent
convention into an explicit, auditable contract that an unattended agent
cannot satisfy by accident.

Concurrency: every transition holds a cross-process exclusive lock around its
read-validate-append sequence, so two writers cannot both validate against the
same stale state (e.g. racing a retire against a promote).
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
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
    # backtested -> backtested is RE-JUSTIFICATION (attach a new justifying run)
    "backtested": frozenset({"backtested", "paper-live", "retired"}),
    "paper-live": frozenset({"trusted", "backtested", "retired"}),  # ->backtested = demotion
    "trusted": frozenset({"backtested", "retired"}),  # ->backtested = demotion
    "retired": frozenset(),  # terminal
}

HUMAN_GATED_TARGETS = frozenset({"paper-live", "trusted", "retired"})


class LifecycleError(ValueError):
    pass


class SupportsRuns(Protocol):
    # A Protocol rather than importing BacktestRegistry: backtest.registry imports
    # vega.lifecycle.rationale for its gate, so a direct import here would cycle.
    def runs(self, family: str | None = None) -> list[dict[str, Any]]: ...


def is_eligible_state(state: str) -> bool:
    return state in ELIGIBLE_STATES


def _require_human(actor: str, action: str) -> None:
    if not actor.startswith("human:"):
        raise LifecycleError(
            f"{action} is a human-only act — actor must carry the 'human:' prefix, got {actor!r}"
        )


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
    justifying_version: str | None = None  # which version's evidence backed this (audit)


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
        justifying_version: str | None = None,
    ) -> TransitionRecord:
        # lock around read-validate-append: two writers must never both validate
        # against the same stale state (could e.g. un-retire a terminal state)
        with self._log.exclusive_lock():
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
                justifying_version=justifying_version,
            )
            self._log.append({"type": "transition", **asdict(record)})
            return record

    def _best_passing_run(self, family: str, backtest_registry: SupportsRuns) -> dict[str, Any]:
        passing = [r for r in backtest_registry.runs(family) if r["verdict"] == "pass"]
        if not passing:
            raise LifecycleError(f"{family} has no passing backtest run recorded")

        def _sharpe(run: dict[str, Any]) -> float:
            value = run["aggregate_metrics"].get("sharpe")
            # `is None`, never `or`: a legitimate 0.0 Sharpe must not score as -inf
            return float("-inf") if value is None else float(value)

        return max(passing, key=_sharpe)

    def promote_to_backtested(
        self,
        family: str,
        rationale_registry: RationaleRegistry,
        backtest_registry: SupportsRuns,
        actor: str,
    ) -> TransitionRecord:
        """From `candidate` (first promotion) or `backtested` (RE-JUSTIFICATION
        after a demotion — attaches a fresh justifying run so the demotion band
        reflects current evidence, not the band the family already breached)."""
        if not rationale_registry.has_rationale(family):
            raise LifecycleError(
                f"{family} has no recorded economic rationale — cannot enter testing"
            )
        best = self._best_passing_run(family, backtest_registry)
        return self._transition(
            family,
            "backtested",
            actor,
            f"rationale on file + passing run {best['run_id']}",
            justifying_run_id=best["run_id"],
            justifying_version=best.get("signal_version"),
        )

    def promote_to_paper_live(self, family: str, actor: str) -> TransitionRecord:
        _require_human(actor, "promote_to_paper_live")
        return self._transition(family, "paper-live", actor, "human promotion to paper-live")

    def promote_to_trusted(self, family: str, actor: str) -> TransitionRecord:
        _require_human(actor, "promote_to_trusted")
        return self._transition(family, "trusted", actor, "human promotion to trusted")

    def demote(
        self, family: str, actor: str, reason: str, automatic: bool = False
    ) -> TransitionRecord:
        current = self.current_state(family)
        if current not in ("paper-live", "trusted"):
            raise LifecycleError(f"cannot demote {family} from {current}")
        return self._transition(family, "backtested", actor, reason, automatic=automatic)

    def retire(self, family: str, actor: str, reason: str) -> TransitionRecord:
        _require_human(actor, "retire")
        return self._transition(family, "retired", actor, reason)

    def justifying_run_id(self, family: str) -> str | None:
        """The run backing the family's most recent (re-)justification — used by
        demotion to fetch the confidence band. Re-justification transitions update
        this; demotions (justifying_run_id=None) are skipped."""
        for record in reversed(self.history(family)):
            if record["to_state"] == "backtested" and record.get("justifying_run_id"):
                return str(record["justifying_run_id"])
        return None
