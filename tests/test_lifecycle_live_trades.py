from pathlib import Path

import pandas as pd

from vega.backtest.registry import BacktestRegistry
from vega.ledger.store import LedgerStore
from vega.ledger.types import Recommendation
from vega.lifecycle.lifecycle import LifecycleRegistry
from vega.lifecycle.live_trades import check_and_apply_demotions, closed_round_trips
from vega.lifecycle.rationale import RationaleRegistry

DATES = list(pd.date_range("2026-03-01", periods=150, freq="D").strftime("%Y-%m-%d"))


def _calendar_frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"symbol": "SPY", "date": d, "close": 100.0, "adj_close": 100.0} for d in dates]
    )


def _rec(**overrides: object) -> Recommendation:
    base: dict[str, object] = {
        "symbol": "AAA",
        "asset_class": "equity",
        "direction": "long",
        "thesis": "fixture",
        "confidence": 0.5,
        "horizon_days": 7,
        "entry_ref_price": 100.0,
        "stop_price": 90.0,
        "time_stop_date": "2026-05-01",
        "profit_rule": "half at +1.5R",
        "invalidation": "fixture",
        "signal_attribution": ("oversold_reversion_v1:1.1",),
        "as_of": DATES[0],
    }
    base.update(overrides)
    return Recommendation(**base)  # type: ignore[arg-type]


def _seed_paper_live(
    tmp_path: Path, family: str = "oversold_reversion_v1", sharpes: tuple[float, ...] = (1.0, 1.5)
) -> tuple[LifecycleRegistry, BacktestRegistry]:
    rationale = RationaleRegistry(tmp_path / "rationale.jsonl")
    rationale.record(family, "a real economic rationale", author="human:jose")
    registry = BacktestRegistry(tmp_path / "registry.jsonl")
    registry.record_run(
        signal_family=family,
        signal_version="1.1",
        param_grid_size=1,
        universe_version="v1",
        data_span=("2025-01-01", DATES[0]),
        n_folds=len(sharpes),
        fold_metrics=[{"sharpe": s} for s in sharpes],
        aggregate_metrics={"sharpe": sum(sharpes) / len(sharpes), "n_trades": 40},
        verdict="pass",
        holdout_evaluated=True,
        promotion_bar=0.8,
        notes=[],
        holdout_sharpe=2.0,
        signal_params={"k": 2.0},
    )
    lifecycle = LifecycleRegistry(tmp_path / "lifecycle.jsonl")
    lifecycle.promote_to_backtested(family, rationale, registry, actor="agent:sonnet")
    lifecycle.promote_to_paper_live(family, actor="human:jose")
    return lifecycle, registry


def _seed_closed_trade(
    ledger: LedgerStore,
    i: int,
    entry_price: float,
    exit_price: float,
    family: str = "oversold_reversion_v1",
    asset_class: str = "equity",
) -> None:
    rec = _rec(
        symbol=f"SYM{i}",
        asset_class=asset_class,
        as_of=DATES[i],
        entry_ref_price=entry_price,
        stop_price=entry_price - 10.0,
        signal_attribution=(f"{family}:1.1",),
    )
    ledger.append(rec)
    ledger.append_fill(rec.id, f"buy-{i}", 10.0, entry_price, "filled")
    ledger.append_fill(
        rec.id,
        f"sell-{i}",
        10.0,
        exit_price,
        "filled",
        side="sell",
        reason="stop",
        session=DATES[i + 1],
    )


# ---- closed_round_trips ------------------------------------------------------


def test_groups_trades_by_family_parsed_from_attribution(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    _seed_closed_trade(ledger, 0, 100.0, 105.0, family="oversold_reversion_v1")
    _seed_closed_trade(ledger, 1, 100.0, 95.0, family="trend_pullback_v1")
    frame = _calendar_frame(DATES[:5])
    by_family = closed_round_trips(ledger, frame)
    assert set(by_family) == {"oversold_reversion_v1", "trend_pullback_v1"}
    assert len(by_family["oversold_reversion_v1"]) == 1


def test_entry_and_exit_dates_are_store_sessions_not_timestamps(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    _seed_closed_trade(ledger, 0, 100.0, 105.0)
    frame = _calendar_frame(DATES[:5])
    (trade,) = closed_round_trips(ledger, frame)["oversold_reversion_v1"]
    assert trade.entry_date == DATES[1]  # first session after rec.as_of (DATES[0])
    assert trade.exit_date == DATES[1]  # the sell fill's tagged session


def test_partial_then_time_stop_yields_two_trade_rows(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec(as_of=DATES[0])
    ledger.append(rec)
    ledger.append_fill(rec.id, "buy-1", 10.0, 100.0, "filled")
    ledger.append_fill(
        rec.id,
        "sell-1",
        5.0,
        115.0,
        "filled",
        side="sell",
        reason="profit_partial",
        session=DATES[2],
    )
    ledger.append_fill(
        rec.id, "sell-2", 5.0, 98.0, "filled", side="sell", reason="time_stop", session=DATES[5]
    )
    frame = _calendar_frame(DATES[:7])
    (trades,) = closed_round_trips(ledger, frame).values()
    assert len(trades) == 2
    assert {t.qty for t in trades} == {5.0}
    assert {t.exit_price for t in trades} == {115.0, 98.0}


def test_pending_and_unpriced_positions_produce_no_trades(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    ledger.append(_rec(as_of=DATES[3]))  # never filled
    unpriced = _rec(symbol="BBB", as_of=DATES[0])
    ledger.append(unpriced)
    ledger.append_fill(unpriced.id, "buy-2", 10.0, None, "accepted")  # entry not yet reconciled
    frame = _calendar_frame(DATES[:5])
    assert closed_round_trips(ledger, frame) == {}


# ---- check_and_apply_demotions ----------------------------------------------


def test_insufficient_sample_takes_no_action(tmp_path: Path) -> None:
    lifecycle, registry = _seed_paper_live(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    for i in range(5):  # below MIN_TRADES_FOR_VERDICT
        _seed_closed_trade(ledger, i, 100.0, 95.0)
    frame = _calendar_frame(DATES[:40])
    outcomes = check_and_apply_demotions(ledger, frame, lifecycle, registry, DATES[10])
    assert len(outcomes) == 1
    assert outcomes[0].verdict.should_demote is False
    assert "insufficient_sample" in outcomes[0].verdict.reason
    assert lifecycle.current_state("oversold_reversion_v1") == "paper-live"


def test_consistent_losses_trigger_automatic_demotion(tmp_path: Path) -> None:
    lifecycle, registry = _seed_paper_live(tmp_path, sharpes=(1.5, 2.0))
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    for i in range(35):
        # alternating losses -> consistently negative, breaches a positive band
        exit_price = 100.0 - (1.0 if i % 2 == 0 else 0.5)
        _seed_closed_trade(ledger, i, 100.0, exit_price)
    frame = _calendar_frame(DATES[:40])
    outcomes = check_and_apply_demotions(
        ledger, frame, lifecycle, registry, DATES[36], actor="agent:exit-monitor"
    )
    assert len(outcomes) == 1
    assert outcomes[0].verdict.should_demote is True
    assert lifecycle.current_state("oversold_reversion_v1") == "backtested"
    last = lifecycle.history("oversold_reversion_v1")[-1]
    assert last["actor"] == "agent:exit-monitor" and last["automatic"] is True


def test_non_eligible_families_are_never_evaluated(tmp_path: Path) -> None:
    rationale = RationaleRegistry(tmp_path / "rationale.jsonl")
    rationale.record("candidate_family_v1", "text", author="human:jose")
    lifecycle = LifecycleRegistry(tmp_path / "lifecycle.jsonl")
    registry = BacktestRegistry(tmp_path / "registry.jsonl")
    # never promoted past candidate -- has a rationale but no lifecycle transition
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    frame = _calendar_frame(DATES[:5])
    assert check_and_apply_demotions(ledger, frame, lifecycle, registry, DATES[3]) == []


def test_sleeve_split_demotes_the_family_only_once(tmp_path: Path) -> None:
    lifecycle, registry = _seed_paper_live(tmp_path, sharpes=(1.5, 2.0))
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    for i in range(0, 35):
        exit_price = 100.0 - (1.0 if i % 2 == 0 else 0.5)
        _seed_closed_trade(ledger, i, 100.0, exit_price, asset_class="equity")
    for i in range(35, 70):
        exit_price = 100.0 - (1.0 if i % 2 == 0 else 0.5)
        _seed_closed_trade(ledger, i, 100.0, exit_price, asset_class="crypto")
    frame = _calendar_frame(DATES[:75])
    # both sleeves breach -> exactly one demote() call, no LifecycleError from a
    # second demote on an already-demoted family
    outcomes = check_and_apply_demotions(ledger, frame, lifecycle, registry, DATES[72])
    assert {o.asset_class for o in outcomes} == {"equity", "crypto"}
    assert all(o.verdict.should_demote for o in outcomes)
    assert lifecycle.current_state("oversold_reversion_v1") == "backtested"
    demotions = [
        r
        for r in lifecycle.history("oversold_reversion_v1")
        if r["to_state"] == "backtested" and r["from_state"] == "paper-live"
    ]
    assert len(demotions) == 1
