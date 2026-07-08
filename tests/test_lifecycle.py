from pathlib import Path

import pytest

from vega.lifecycle.lifecycle import LifecycleError, LifecycleRegistry, is_eligible_state
from vega.lifecycle.rationale import RationaleRegistry


class _FakeBacktestRegistry:
    def __init__(self, runs: list[dict[str, object]]) -> None:
        self._runs = runs

    def runs(self, family: str | None = None) -> list[dict[str, object]]:
        return [r for r in self._runs if family is None or r["signal_family"] == family]


def _passing_run(run_id: str = "run-1", sharpe: float = 1.2) -> dict[str, object]:
    return {
        "run_id": run_id,
        "signal_family": "fam",
        "verdict": "pass",
        "aggregate_metrics": {"sharpe": sharpe},
        "fold_metrics": [
            {"sharpe": 1.0},
            {"sharpe": 1.5},
            {"sharpe": 1.2, "is_holdout": True},
        ],
    }


@pytest.fixture
def rationale(tmp_path: Path) -> RationaleRegistry:
    reg = RationaleRegistry(tmp_path / "rationale.jsonl")
    reg.record("fam", "a real economic rationale", author="human:jose")
    return reg


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
    backtests = _FakeBacktestRegistry([_passing_run()])
    reg.promote_to_backtested("fam", rationale, backtests, actor="agent:sonnet")
    assert reg.current_state("fam") == "backtested"
    reg.promote_to_paper_live("fam", actor="human:jose")
    assert reg.current_state("fam") == "paper-live"
    assert reg.eligible_for_recommendations("fam") is True
    reg.promote_to_trusted("fam", actor="human:jose")
    assert reg.current_state("fam") == "trusted"
    assert reg.eligible_for_recommendations("fam") is True


def test_cannot_skip_candidate_straight_to_paper_live(tmp_path: Path) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    with pytest.raises(LifecycleError):
        reg.promote_to_paper_live("fam", actor="human:jose")


def test_promote_to_backtested_blocked_without_rationale(tmp_path: Path) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    empty_rationale = RationaleRegistry(tmp_path / "empty_rationale.jsonl")
    backtests = _FakeBacktestRegistry([_passing_run()])
    with pytest.raises(LifecycleError, match="no recorded economic rationale"):
        reg.promote_to_backtested("fam", empty_rationale, backtests, actor="agent:sonnet")


def test_promote_to_backtested_blocked_without_a_passing_run(
    tmp_path: Path, rationale: RationaleRegistry
) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    failing_run = {**_passing_run(), "verdict": "fail"}
    backtests = _FakeBacktestRegistry([failing_run])
    with pytest.raises(LifecycleError, match="no passing backtest run"):
        reg.promote_to_backtested("fam", rationale, backtests, actor="agent:sonnet")


def test_retirement_reachable_from_any_state(tmp_path: Path, rationale: RationaleRegistry) -> None:
    for start_actions in (
        [],
        [("promote_to_backtested",)],
        [("promote_to_backtested",), ("promote_to_paper_live",)],
    ):
        reg = LifecycleRegistry(tmp_path / f"l_{len(start_actions)}.jsonl")
        backtests = _FakeBacktestRegistry([_passing_run()])
        for action in start_actions:
            if action[0] == "promote_to_backtested":
                reg.promote_to_backtested("fam", rationale, backtests, actor="a")
            elif action[0] == "promote_to_paper_live":
                reg.promote_to_paper_live("fam", actor="a")
        reg.retire("fam", actor="human:jose", reason="deprecated")
        assert reg.current_state("fam") == "retired"


def test_retired_is_terminal(tmp_path: Path, rationale: RationaleRegistry) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    reg.retire("fam", actor="human:jose", reason="bad idea")
    with pytest.raises(LifecycleError):
        reg.promote_to_backtested("fam", rationale, _FakeBacktestRegistry([_passing_run()]), "a")


def test_demote_only_valid_from_paper_live_or_trusted(
    tmp_path: Path, rationale: RationaleRegistry
) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    with pytest.raises(LifecycleError):
        reg.demote("fam", actor="system", reason="auto")
    backtests = _FakeBacktestRegistry([_passing_run()])
    reg.promote_to_backtested("fam", rationale, backtests, actor="a")
    reg.promote_to_paper_live("fam", actor="a")
    reg.demote("fam", actor="system", reason="Sharpe below band", automatic=True)
    assert reg.current_state("fam") == "backtested"  # demotion lands on backtested, not candidate


def test_demotion_from_trusted_also_lands_on_backtested(
    tmp_path: Path, rationale: RationaleRegistry
) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    backtests = _FakeBacktestRegistry([_passing_run()])
    reg.promote_to_backtested("fam", rationale, backtests, actor="a")
    reg.promote_to_paper_live("fam", actor="a")
    reg.promote_to_trusted("fam", actor="a")
    reg.demote("fam", actor="system", reason="decayed", automatic=True)
    assert reg.current_state("fam") == "backtested"


def test_justifying_run_id_tracks_the_best_sharpe_passing_run(
    tmp_path: Path, rationale: RationaleRegistry
) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    weak = _passing_run(run_id="weak", sharpe=0.9)
    strong = _passing_run(run_id="strong", sharpe=2.0)
    backtests = _FakeBacktestRegistry([weak, strong])
    reg.promote_to_backtested("fam", rationale, backtests, actor="a")
    assert reg.justifying_run_id("fam") == "strong"
