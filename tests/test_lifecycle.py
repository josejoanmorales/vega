from pathlib import Path

import pytest

from vega.lifecycle.lifecycle import LifecycleError, LifecycleRegistry, is_eligible_state
from vega.lifecycle.rationale import RationaleRegistry

HUMAN = "human:jose"


class _FakeBacktestRegistry:
    def __init__(self, runs: list[dict[str, object]]) -> None:
        self._runs = runs

    def runs(self, family: str | None = None) -> list[dict[str, object]]:
        return [r for r in self._runs if family is None or r["signal_family"] == family]


def _passing_run(
    run_id: str = "run-1", sharpe: float = 1.2, version: str = "1"
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "signal_family": "fam",
        "signal_version": version,
        "verdict": "pass",
        "aggregate_metrics": {"sharpe": sharpe},
        "fold_metrics": [{"sharpe": 1.0}, {"sharpe": 1.5}, {"sharpe": 1.2, "is_holdout": True}],
    }


@pytest.fixture
def rationale(tmp_path: Path) -> RationaleRegistry:
    reg = RationaleRegistry(tmp_path / "rationale.jsonl")
    reg.record("fam", "a real economic rationale", author=HUMAN)
    return reg


def _to_backtested(reg: LifecycleRegistry, rationale: RationaleRegistry, **runs: object) -> None:
    reg.promote_to_backtested(
        "fam", rationale, _FakeBacktestRegistry([_passing_run()]), actor="agent:sonnet"
    )


def test_new_family_starts_at_candidate(tmp_path: Path) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    assert reg.current_state("fam") == "candidate"
    assert reg.eligible_for_recommendations("fam") is False


def test_is_eligible_state_truth_table() -> None:
    assert is_eligible_state("paper-live") is True
    assert is_eligible_state("trusted") is True
    for s in ("candidate", "backtested", "retired"):
        assert is_eligible_state(s) is False


def test_full_happy_path_promotion(tmp_path: Path, rationale: RationaleRegistry) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    _to_backtested(reg, rationale)
    assert reg.current_state("fam") == "backtested"
    reg.promote_to_paper_live("fam", actor=HUMAN)
    assert reg.current_state("fam") == "paper-live"
    assert reg.eligible_for_recommendations("fam") is True
    reg.promote_to_trusted("fam", actor=HUMAN)
    assert reg.current_state("fam") == "trusted"


def test_cannot_skip_candidate_straight_to_paper_live(tmp_path: Path) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    with pytest.raises(LifecycleError):
        reg.promote_to_paper_live("fam", actor=HUMAN)


def test_promote_to_backtested_blocked_without_rationale(tmp_path: Path) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    empty = RationaleRegistry(tmp_path / "empty.jsonl")
    with pytest.raises(LifecycleError, match="no recorded economic rationale"):
        reg.promote_to_backtested("fam", empty, _FakeBacktestRegistry([_passing_run()]), "agent:x")


def test_promote_to_backtested_blocked_without_a_passing_run(
    tmp_path: Path, rationale: RationaleRegistry
) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    failing = _FakeBacktestRegistry([{**_passing_run(), "verdict": "fail"}])
    with pytest.raises(LifecycleError, match="no passing backtest run"):
        reg.promote_to_backtested("fam", rationale, failing, actor="agent:x")


def test_paper_live_and_trusted_and_retire_require_a_human_actor(
    tmp_path: Path, rationale: RationaleRegistry
) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    _to_backtested(reg, rationale)
    with pytest.raises(LifecycleError, match="human-only"):
        reg.promote_to_paper_live("fam", actor="agent:sonnet")  # agent cannot go live
    reg.promote_to_paper_live("fam", actor=HUMAN)
    with pytest.raises(LifecycleError, match="human-only"):
        reg.promote_to_trusted("fam", actor="agent:sonnet")
    with pytest.raises(LifecycleError, match="human-only"):
        reg.retire("fam", actor="agent:sonnet", reason="x")


def test_retirement_reachable_from_any_state(tmp_path: Path, rationale: RationaleRegistry) -> None:
    for n, actions in enumerate(([], ["bt"], ["bt", "pl"])):
        reg = LifecycleRegistry(tmp_path / f"l_{n}.jsonl")
        for a in actions:
            if a == "bt":
                _to_backtested(reg, rationale)
            elif a == "pl":
                reg.promote_to_paper_live("fam", actor=HUMAN)
        reg.retire("fam", actor=HUMAN, reason="deprecated")
        assert reg.current_state("fam") == "retired"


def test_retired_is_terminal(tmp_path: Path, rationale: RationaleRegistry) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    reg.retire("fam", actor=HUMAN, reason="bad idea")
    with pytest.raises(LifecycleError):
        _to_backtested(reg, rationale)


def test_auto_demotion_lands_on_backtested_not_candidate(
    tmp_path: Path, rationale: RationaleRegistry
) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    with pytest.raises(LifecycleError):
        reg.demote("fam", actor="system", reason="auto")  # can't demote from candidate
    _to_backtested(reg, rationale)
    reg.promote_to_paper_live("fam", actor=HUMAN)
    reg.demote("fam", actor="system", reason="Sharpe below band", automatic=True)
    assert reg.current_state("fam") == "backtested"


def test_demotion_from_trusted_also_lands_on_backtested(
    tmp_path: Path, rationale: RationaleRegistry
) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    _to_backtested(reg, rationale)
    reg.promote_to_paper_live("fam", actor=HUMAN)
    reg.promote_to_trusted("fam", actor=HUMAN)
    reg.demote("fam", actor="system", reason="decayed", automatic=True)
    assert reg.current_state("fam") == "backtested"


def test_rejustification_attaches_a_fresh_run_after_demotion(
    tmp_path: Path, rationale: RationaleRegistry
) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    old = _FakeBacktestRegistry([_passing_run(run_id="old", sharpe=1.5)])
    reg.promote_to_backtested("fam", rationale, old, actor="agent:x")
    reg.promote_to_paper_live("fam", actor=HUMAN)
    reg.demote("fam", actor="system", reason="decayed", automatic=True)
    assert reg.justifying_run_id("fam") == "old"  # stale band still on record
    # re-justify with a NEW backtest run (backtested -> backtested is legal)
    fresh = _FakeBacktestRegistry([_passing_run(run_id="fresh", sharpe=1.1)])
    reg.promote_to_backtested("fam", rationale, fresh, actor="agent:x")
    assert reg.current_state("fam") == "backtested"
    assert reg.justifying_run_id("fam") == "fresh"  # band now reflects post-demotion evidence
    reg.promote_to_paper_live("fam", actor=HUMAN)
    assert reg.eligible_for_recommendations("fam") is True


def test_justifying_run_picks_best_sharpe_and_records_version(
    tmp_path: Path, rationale: RationaleRegistry
) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    runs = _FakeBacktestRegistry(
        [
            _passing_run(run_id="weak", sharpe=0.9, version="1"),
            _passing_run(run_id="strong", sharpe=2.0, version="2"),
        ]
    )
    reg.promote_to_backtested("fam", rationale, runs, actor="agent:x")
    assert reg.justifying_run_id("fam") == "strong"
    justifying = [h for h in reg.history("fam") if h["to_state"] == "backtested"][-1]
    assert justifying["justifying_version"] == "2"  # version recorded for audit


def test_zero_sharpe_run_is_not_scored_as_negative_infinity(
    tmp_path: Path, rationale: RationaleRegistry
) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    # a legitimate 0.0-Sharpe pass must beat a negative-Sharpe pass (falsy-zero guard)
    runs = _FakeBacktestRegistry(
        [
            _passing_run(run_id="zero", sharpe=0.0),
            _passing_run(run_id="neg", sharpe=-0.3),
        ]
    )
    reg.promote_to_backtested("fam", rationale, runs, actor="agent:x")
    assert reg.justifying_run_id("fam") == "zero"
