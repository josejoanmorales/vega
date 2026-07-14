"""Pins for the WI-066 review fixes — each test names the finding it guards."""

from pathlib import Path

import pandas as pd
import pytest

from vega.backtest.market_view import MarketView
from vega.lifecycle.lifecycle import LifecycleError, LifecycleRegistry
from vega.lifecycle.rationale import RationaleRegistry
from vega.signals.helpers import is_new_high, three_session_change
from vega.signals.trend_pullback import TrendPullbackSignal

HUMAN = "human:jose"


class _FakeRegistry:
    def __init__(self, runs: list[dict[str, object]]) -> None:
        self._runs = runs

    def runs(self, family: str | None = None) -> list[dict[str, object]]:
        return [r for r in self._runs if family is None or r["signal_family"] == family]


def _run(run_id: str, sharpe: float, holdout_sharpe: float | None) -> dict[str, object]:
    return {
        "run_id": run_id,
        "signal_family": "fam",
        "signal_version": "1.1",
        "signal_params": {"k": 2.0},
        "verdict": "pass",
        "holdout_sharpe": holdout_sharpe,
        "aggregate_metrics": {"sharpe": sharpe},
        "fold_metrics": [{"sharpe": sharpe}],
    }


@pytest.fixture
def rationale(tmp_path: Path) -> RationaleRegistry:
    reg = RationaleRegistry(tmp_path / "r.jsonl")
    reg.record("fam", "a real rationale", author=HUMAN)
    return reg


# --- Finding 1: promotion gate must not be blind to the holdout ---


def test_negative_holdout_pass_is_refused_by_default(
    tmp_path: Path, rationale: RationaleRegistry
) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    overfit = _FakeRegistry([_run("r1", sharpe=2.4, holdout_sharpe=-0.9)])
    with pytest.raises(LifecycleError, match="negative or missing holdout"):
        reg.promote_to_backtested("fam", rationale, overfit, actor="agent:x")


def test_negative_holdout_override_is_human_only(
    tmp_path: Path, rationale: RationaleRegistry
) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    overfit = _FakeRegistry([_run("r1", sharpe=2.4, holdout_sharpe=-0.9)])
    with pytest.raises(LifecycleError, match="human-only"):
        reg.promote_to_backtested(
            "fam", rationale, overfit, actor="agent:x", allow_negative_holdout=True
        )
    record = reg.promote_to_backtested(
        "fam", rationale, overfit, actor=HUMAN, allow_negative_holdout=True
    )
    assert "human override" in record.reason


def test_healthy_holdout_beats_a_higher_sharpe_overfit_run(
    tmp_path: Path, rationale: RationaleRegistry
) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    mixed = _FakeRegistry(
        [
            _run("overfit", sharpe=3.0, holdout_sharpe=-0.5),
            _run("healthy", sharpe=1.3, holdout_sharpe=3.6),
        ]
    )
    record = reg.promote_to_backtested("fam", rationale, mixed, actor="agent:x")
    assert record.justifying_run_id == "healthy"  # the overfit run is excluded, not selected


# --- Finding 2: param identity recorded on the transition ---


def test_justifying_params_recorded_on_promotion(
    tmp_path: Path, rationale: RationaleRegistry
) -> None:
    reg = LifecycleRegistry(tmp_path / "l.jsonl")
    record = reg.promote_to_backtested(
        "fam", rationale, _FakeRegistry([_run("r1", 1.3, 3.6)]), actor="agent:x"
    )
    assert record.justifying_params == {"k": 2.0}


# --- Finding 4: NaN must not slip through signal guards ---


def test_three_session_change_returns_none_on_nan() -> None:
    closes = pd.Series([100.0, 101.0, 102.0, float("nan")])
    assert three_session_change(closes) is None
    closes = pd.Series([float("nan"), 101.0, 102.0, 103.0])
    assert three_session_change(closes) is None


def test_is_new_high_false_on_nan_today() -> None:
    closes = pd.Series([100.0] * 41)
    assert is_new_high(pd.concat([closes, pd.Series([float("nan")])]), 40) is False


# --- Finding 7: strict Donchian semantics ---


def test_flat_series_is_not_a_breakout() -> None:
    assert is_new_high(pd.Series([100.0] * 60), 40) is False  # ties never count


def test_new_high_compares_against_n_prior_sessions() -> None:
    # exactly N prior flat sessions + a higher today = breakout; N-1 prior = insufficient
    assert is_new_high(pd.Series([100.0] * 40 + [110.0]), 40) is True
    assert is_new_high(pd.Series([100.0] * 39 + [110.0]), 40) is False


# --- Finding 6: pullback depth measured at the trough, not the recovered close ---


def test_depth_is_high_to_trough_not_high_to_recovered_close() -> None:
    base = [100.0 + i * 0.5 for i in range(60)]  # rising trend, peak 129.5
    # deep trough (-6.2%) but strong recovery: today closes only 1.5% below the peak.
    # Old (recovered-close) rule: depth 1.5% -> no fire. New (trough) rule: 6.2% -> fires.
    closes = base + [129.5 - 8.0, 129.5 - 2.0]
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D").strftime("%Y-%m-%d")
    frame = pd.DataFrame({"symbol": "AAA", "date": dates, "adj_close": closes})
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    assert len(TrendPullbackSignal(depth=0.05).scan(view, ["AAA"])) == 1


def test_trough_before_the_high_does_not_count_as_pullback() -> None:
    # low happened BEFORE the window high (ascent, not pullback) -> depth ~0 -> no fire
    base = [100.0 + i * 0.5 for i in range(55)]
    closes = base + [120.0, 126.0, 127.0, 128.0, 129.0, 129.5, 129.6]  # dip then new highs
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D").strftime("%Y-%m-%d")
    frame = pd.DataFrame({"symbol": "AAA", "date": dates, "adj_close": closes})
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    assert TrendPullbackSignal(depth=0.05).scan(view, ["AAA"]) == []


# --- Finding 8: pending entries retry instead of vanishing ---


def test_pending_entry_survives_a_missing_bar_and_fills_later() -> None:
    from vega.backtest.signals import EntryProposal
    from vega.backtest.simulate import simulate_signal

    dates = [f"2026-02-{d:02d}" for d in range(1, 25)]
    rows = [
        {
            "symbol": "TEST",
            "date": d,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adj_close": 100.0,
            "volume": 1e6,
        }
        for d in dates
        if d != "2026-02-21"  # the symbol has NO bar the day after the signal fires
    ]
    frame = pd.DataFrame(rows)

    class _FireOnce:
        family = "test_retry"
        version = "1"
        promotable = False

        def scan(self, view: MarketView, universe: list[str]) -> list[EntryProposal]:
            if view.as_of != "2026-02-20":
                return []
            return [EntryProposal("TEST", self.family, "1", "t", 0.5, "i")]

    trades = simulate_signal(frame, dates, _FireOnce(), ["TEST"], "equity")
    assert len(trades) == 1
    assert trades[0].entry_date == "2026-02-22"  # filled on the NEXT available bar, not dropped


# --- Finding 9: default paths are project-anchored, not CWD-relative ---


def test_default_registry_paths_are_absolute_and_project_anchored() -> None:
    from vega.backtest.registry import DEFAULT_PATH as reg_path
    from vega.common.paths import PROJECT_ROOT
    from vega.lifecycle.rationale import DEFAULT_PATH as rat_path

    assert reg_path.is_absolute() and rat_path.is_absolute()
    assert PROJECT_ROOT in reg_path.parents and PROJECT_ROOT in rat_path.parents
    assert (PROJECT_ROOT / "pyproject.toml").exists()  # the anchor really is the repo root
