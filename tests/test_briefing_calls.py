from pathlib import Path

import pandas as pd
import pytest

from conftest import make_ohlc_frame, steep_uptrend_then_shock
from vega.backtest.registry import BacktestRegistry
from vega.backtest.signals import EntryProposal
from vega.briefing.calls import (
    CallsError,
    RenderedRejection,
    _no_trade_reason,
    _rank_key,
    build_calls,
)
from vega.data.types import UniverseEntry
from vega.ledger.store import LedgerStore
from vega.ledger.types import Recommendation
from vega.lifecycle.lifecycle import LifecycleRegistry
from vega.lifecycle.rationale import RationaleRegistry
from vega.regime.regime import RegimeState
from vega.risk.gates import EarningsFact

NO_EARNINGS = EarningsFact("none")
AS_OF = "2026-04-13"  # last date of the 103-bar conftest fixture


def _shocked_frame(symbol: str = "AAA", drop_total: float = 39.0) -> pd.DataFrame:
    closes = steep_uptrend_then_shock(drop_total)
    return make_ohlc_frame(closes, shocked={100, 101, 102}, symbol=symbol, volume=1_000_000.0)


def _regime(composite: str = "risk_on") -> RegimeState:
    return RegimeState(
        as_of=AS_OF,
        trend="risk_on",
        vix=15.0,
        vix_band="calm",
        breadth_pct=60.0,
        crypto_fg=50,
        composite=composite,
    )


def _universe(symbols: list[str]) -> list[UniverseEntry]:
    return [UniverseEntry(symbol=s, asset_class="equity", name=s) for s in symbols]


def _seed_paper_live(
    tmp_path: Path,
    family: str = "oversold_reversion_v1",
    params: dict[str, object] | None = None,
    dev_sharpe: float = 1.3,
) -> tuple[LifecycleRegistry, BacktestRegistry]:
    params = params if params is not None else {"k": 2.0}
    rationale = RationaleRegistry(tmp_path / "rationale.jsonl")
    rationale.record(family, "a real economic rationale", author="human:jose")
    registry = BacktestRegistry(tmp_path / "registry.jsonl")
    registry.record_run(
        signal_family=family,
        signal_version="1.1",
        param_grid_size=1,
        universe_version="v1",
        data_span=("2025-01-01", AS_OF),
        n_folds=2,
        fold_metrics=[{"sharpe": 1.0}, {"sharpe": 1.5}],
        aggregate_metrics={"sharpe": dev_sharpe, "n_trades": 40},
        verdict="pass",
        holdout_evaluated=True,
        promotion_bar=0.8,
        notes=[],
        holdout_sharpe=2.0,
        signal_params=params,
    )
    lifecycle = LifecycleRegistry(tmp_path / "lifecycle.jsonl")
    lifecycle.promote_to_backtested(family, rationale, registry, actor="agent:sonnet")
    lifecycle.promote_to_paper_live(family, actor="human:jose")
    return lifecycle, registry


def _build(tmp_path: Path, lifecycle, registry, **overrides):  # type: ignore[no-untyped-def]
    kwargs: dict[str, object] = {
        "frame": _shocked_frame(),
        "as_of": AS_OF,
        "equity": 100_000.0,
        "regime": _regime(),
        "ledger": LedgerStore(tmp_path / "ledger.jsonl"),
        "lifecycle": lifecycle,
        "backtest_registry": registry,
        "universe_entries": _universe(["AAA"]),
        "earnings_lookup": lambda *_: NO_EARNINGS,
    }
    kwargs.update(overrides)
    return build_calls(**kwargs)  # type: ignore[arg-type]


def _rec(**overrides: object) -> Recommendation:
    base: dict[str, object] = {
        "symbol": "P1",
        "asset_class": "equity",
        "direction": "long",
        "thesis": "pre-existing open position",
        "confidence": 0.5,
        "horizon_days": 10,
        "entry_ref_price": 100.0,
        "stop_price": 92.5,  # 7.5 spread * qty 100 = 750 R-dollars = 1R at equity 100k/0.75%
        "time_stop_date": "2026-05-01",
        "profit_rule": "half at +2R",
        "invalidation": "fixture",
        "signal_attribution": ("fixture_family",),
        "qty": 100.0,
        "as_of": AS_OF,
    }
    base.update(overrides)
    return Recommendation(**base)  # type: ignore[arg-type]


# ---- eligibility gating ----------------------------------------------------


def test_no_eligible_families_returns_empty_result(tmp_path: Path) -> None:
    lifecycle = LifecycleRegistry(tmp_path / "lifecycle.jsonl")
    registry = BacktestRegistry(tmp_path / "registry.jsonl")
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    result = _build(tmp_path, lifecycle, registry, ledger=ledger)
    assert result.eligible_families == ()
    assert result.calls == ()
    assert result.no_trade_reason is None
    assert ledger.entries() == []


def test_retired_family_never_produces_calls(tmp_path: Path) -> None:
    lifecycle, registry = _seed_paper_live(tmp_path)
    lifecycle.retire("oversold_reversion_v1", actor="human:jose", reason="falsified")
    result = _build(tmp_path, lifecycle, registry)
    assert result.eligible_families == ()
    assert result.calls == ()


def test_eligible_family_without_justifying_run_id_raises(tmp_path: Path) -> None:
    # Forge an unreachable-via-public-API state: backtested with no justifying
    # run recorded (promote_to_backtested always sets one; bypass it directly
    # to prove build_calls refuses to guess at unvalidated parameters).
    lifecycle = LifecycleRegistry(tmp_path / "lifecycle.jsonl")
    lifecycle._transition(
        "oversold_reversion_v1", "backtested", "agent:sonnet", "forged, no evidence"
    )
    lifecycle.promote_to_paper_live("oversold_reversion_v1", actor="human:jose")
    registry = BacktestRegistry(tmp_path / "registry.jsonl")
    with pytest.raises(CallsError, match="no justifying_run_id"):
        _build(tmp_path, lifecycle, registry)


def test_eligible_family_without_registered_class_raises(tmp_path: Path) -> None:
    # WI-067 review: iteration is driven by the lifecycle registry, so a
    # paper-live family missing from FAMILY_SIGNALS must be LOUD, not invisible.
    lifecycle = LifecycleRegistry(tmp_path / "lifecycle.jsonl")
    lifecycle._transition("mystery_family_v1", "backtested", "agent:sonnet", "forged")
    lifecycle.promote_to_paper_live("mystery_family_v1", actor="human:jose")
    registry = BacktestRegistry(tmp_path / "registry.jsonl")
    with pytest.raises(CallsError, match="no signal class registered"):
        _build(tmp_path, lifecycle, registry)


# ---- justified parameters actually drive the live scan --------------------


def test_signal_instantiated_with_justifying_params(tmp_path: Path) -> None:
    # k=100 is an impossibly strict threshold — if params didn't flow through,
    # the signal would use some other default and (with this fixture) still fire.
    lifecycle, registry = _seed_paper_live(tmp_path, params={"k": 100.0})
    result = _build(tmp_path, lifecycle, registry)
    assert result.eligible_families[0].justifying_params == {"k": 100.0}
    assert result.calls == ()
    assert result.rejections == ()  # scan itself found nothing — never reached risk sizing


# ---- accepted call round-trips into the ledger with the family's exit spec -


def test_accepted_call_lands_on_ledger_with_qty_and_family_exit_override(tmp_path: Path) -> None:
    lifecycle, registry = _seed_paper_live(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    result = _build(tmp_path, lifecycle, registry, ledger=ledger)
    assert len(result.calls) == 1
    call = result.calls[0]
    assert call.rank == 1
    assert call.family == "oversold_reversion_v1"
    assert call.time_stop_sessions == 7  # family's doctrine override, not the 15-session default
    assert "1.5" in call.profit_rule  # family's +1.5R half-take override

    entries = ledger.entries()
    assert len(entries) == 1
    assert entries[0]["id"] == call.ref_id
    assert entries[0]["qty"] == call.qty
    assert entries[0]["as_of"] == AS_OF  # decision session recorded for expiry semantics
    assert entries[0]["exit_params"]["time_stop_sessions"] == 7
    assert entries[0]["exit_params"]["take_half_at_r"] == 1.5


# ---- idempotency: active positions are never re-proposed --------------------


def test_already_held_symbol_is_rejected_not_stacked(tmp_path: Path) -> None:
    lifecycle, registry = _seed_paper_live(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    first = _build(tmp_path, lifecycle, registry, ledger=ledger)
    assert len(first.calls) == 1  # AAA called and appended

    # same-day re-run: AAA is now a same-session pending call — never re-proposed
    second = _build(tmp_path, lifecycle, registry, ledger=ledger)
    assert second.calls == ()
    assert [r.reason for r in second.rejections] == ["already_held"]
    assert len(ledger.entries()) == 1  # NO duplicate append

    # and once filled, still blocked
    ledger.append_fill(first.calls[0].ref_id, "ord-1", first.calls[0].qty, 130.0, "filled")
    third = _build(tmp_path, lifecycle, registry, ledger=ledger)
    assert third.calls == ()
    assert [r.reason for r in third.rejections] == ["already_held"]
    assert len(ledger.entries()) == 1


# ---- existing ledger heat can reject a new, otherwise-valid candidate -----


def test_existing_ledger_heat_rejects_new_candidate(tmp_path: Path) -> None:
    lifecycle, registry = _seed_paper_live(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    # 5 pre-existing filled equity positions x 1R each = 5R, over the 4R
    # us_equity_beta cluster cap — a fresh candidate must be rejected on sight.
    for i in range(5):
        rec = _rec(symbol=f"P{i}")
        ledger.append(rec)
        ledger.append_fill(rec.id, f"ord-{i}", 100.0, 100.0, "filled")

    result = _build(tmp_path, lifecycle, registry, ledger=ledger)
    assert result.calls == ()
    assert len(result.rejections) == 1
    assert result.rejections[0].reason.startswith("heat_cap:")
    assert result.no_trade_reason is not None
    assert "heat_cap" in result.no_trade_reason


# ---- c3: regime risk_off yields an explicit, honest no-trade result -------


def test_risk_off_regime_yields_no_trade_reason(tmp_path: Path) -> None:
    lifecycle, registry = _seed_paper_live(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    result = _build(tmp_path, lifecycle, registry, ledger=ledger, regime=_regime("risk_off"))
    assert result.calls == ()
    assert result.no_trade_reason is not None
    assert "regime_risk_off" in result.no_trade_reason
    assert ledger.entries() == []  # a rejected proposal must never reach the ledger


def test_no_trade_reason_is_honest_about_zero_proposals() -> None:
    # WI-067 review: zero proposals must never be reported as "gates blocked
    # entries" — the gates never fired.
    assert "no qualifying setups" in _no_trade_reason(0, [])
    gated = _no_trade_reason(2, [RenderedRejection("A", "f", "regime_risk_off", "d")] * 2)
    assert "regime_risk_off (2)" in gated and "2 candidate(s)" in gated


# ---- WI-087: a just-exited symbol is never re-entered the same run --------


def test_exited_today_symbol_is_rejected_not_reentered(tmp_path: Path) -> None:
    lifecycle, registry = _seed_paper_live(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    result = build_calls(
        frame=_shocked_frame(),
        as_of=AS_OF,
        equity=100_000.0,
        regime=_regime(),
        ledger=ledger,
        lifecycle=lifecycle,
        backtest_registry=registry,
        universe_entries=_universe(["AAA"]),
        earnings_lookup=lambda *_: NO_EARNINGS,
        exited_today=frozenset({"AAA"}),
    )
    assert result.calls == ()
    assert [r.reason for r in result.rejections] == ["same_day_exit"]
    assert ledger.entries() == []


# ---- deterministic ranking --------------------------------------------------


def test_rank_key_orders_by_confidence_then_dev_sharpe_then_symbol() -> None:
    def _entry(symbol: str, confidence: float) -> EntryProposal:
        return EntryProposal(
            symbol=symbol,
            signal_family="fam",
            signal_version="1",
            thesis="t",
            confidence=confidence,
            invalidation="i",
        )

    items = [
        (_entry("ZZZ", 0.6), 1.0),
        (_entry("AAA", 0.6), 2.0),
        (_entry("BBB", 0.9), 0.5),
        (_entry("CCC", 0.6), 2.0),
    ]
    ordered = sorted(items, key=_rank_key)
    assert [p.symbol for p, _ in ordered] == ["BBB", "AAA", "CCC", "ZZZ"]

    # order independent of input shuffling
    import random

    shuffled = items[:]
    random.shuffle(shuffled)
    assert [p.symbol for p, _ in sorted(shuffled, key=_rank_key)] == ["BBB", "AAA", "CCC", "ZZZ"]
