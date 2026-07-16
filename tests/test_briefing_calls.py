from pathlib import Path

import pandas as pd
import pytest

from vega.backtest.registry import BacktestRegistry
from vega.backtest.signals import EntryProposal
from vega.briefing.calls import (
    CallsError,
    _open_positions_from_ledger,
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


def _ohlc_frame(closes: list[float], shocked: set[int], symbol: str = "AAA") -> pd.DataFrame:
    """Raw OHLC + volume around each close; wider range on shocked indices
    (adapted from test_signals_oversold_reversion.py's fixture, +volume so
    the frame satisfies the general bars schema)."""
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D").strftime("%Y-%m-%d")
    rows = []
    for i, (d, c) in enumerate(zip(dates, closes, strict=True)):
        spread = 5.0 if i in shocked else 2.0
        rows.append(
            {
                "symbol": symbol,
                "date": d,
                "adj_close": c,
                "close": c,
                "high": c + spread,
                "low": c - spread,
                "volume": 1_000_000.0,
            }
        )
    return pd.DataFrame(rows)


def _steep_uptrend_then_shock(drop_total: float) -> list[float]:
    base = [100.0 + i * 1.0 for i in range(100)]
    peak = base[-1]
    return base + [peak - drop_total / 3, peak - 2 * drop_total / 3, peak - drop_total]


def _shocked_frame(symbol: str = "AAA", drop_total: float = 39.0) -> pd.DataFrame:
    closes = _steep_uptrend_then_shock(drop_total)
    return _ohlc_frame(closes, shocked={100, 101, 102}, symbol=symbol)


def _multi_symbol_frame(symbols: list[str], drop_total: float = 39.0) -> pd.DataFrame:
    return pd.concat([_shocked_frame(s, drop_total) for s in symbols], ignore_index=True)


def _regime(composite: str = "risk_on") -> RegimeState:
    return RegimeState(
        as_of="2026-04-13",
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
        data_span=("2025-01-01", "2026-04-13"),
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
    }
    base.update(overrides)
    return Recommendation(**base)  # type: ignore[arg-type]


# ---- eligibility gating ----------------------------------------------------


def test_no_eligible_families_returns_empty_result(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    lifecycle = LifecycleRegistry(tmp_path / "lifecycle.jsonl")
    registry = BacktestRegistry(tmp_path / "registry.jsonl")
    result = build_calls(
        frame=_shocked_frame(),
        as_of="2026-04-13",
        equity=100_000.0,
        regime=_regime(),
        ledger=ledger,
        lifecycle=lifecycle,
        backtest_registry=registry,
        universe_entries=_universe(["AAA"]),
    )
    assert result.eligible_families == ()
    assert result.calls == ()
    assert result.no_trade_reason is None
    assert ledger.entries() == []


def test_candidate_family_never_produces_calls(tmp_path: Path) -> None:
    # rationale recorded but never promoted past candidate
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    lifecycle = LifecycleRegistry(tmp_path / "lifecycle.jsonl")
    registry = BacktestRegistry(tmp_path / "registry.jsonl")
    result = build_calls(
        frame=_shocked_frame(),
        as_of="2026-04-13",
        equity=100_000.0,
        regime=_regime(),
        ledger=ledger,
        lifecycle=lifecycle,
        backtest_registry=registry,
        universe_entries=_universe(["AAA"]),
    )
    assert result.eligible_families == ()


def test_retired_family_never_produces_calls(tmp_path: Path) -> None:
    lifecycle, registry = _seed_paper_live(tmp_path)
    lifecycle.retire("oversold_reversion_v1", actor="human:jose", reason="falsified")
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    result = build_calls(
        frame=_shocked_frame(),
        as_of="2026-04-13",
        equity=100_000.0,
        regime=_regime(),
        ledger=ledger,
        lifecycle=lifecycle,
        backtest_registry=registry,
        universe_entries=_universe(["AAA"]),
    )
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
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    registry = BacktestRegistry(tmp_path / "registry.jsonl")
    with pytest.raises(CallsError, match="no justifying_run_id"):
        build_calls(
            frame=_shocked_frame(),
            as_of="2026-04-13",
            equity=100_000.0,
            regime=_regime(),
            ledger=ledger,
            lifecycle=lifecycle,
            backtest_registry=registry,
            universe_entries=_universe(["AAA"]),
        )


# ---- justified parameters actually drive the live scan --------------------


def test_signal_instantiated_with_justifying_params(tmp_path: Path) -> None:
    # k=100 is an impossibly strict threshold — if params didn't flow through,
    # the signal would use some other default and (with this fixture) still fire.
    lifecycle, registry = _seed_paper_live(tmp_path, params={"k": 100.0})
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    result = build_calls(
        frame=_shocked_frame(),
        as_of="2026-04-13",
        equity=100_000.0,
        regime=_regime(),
        ledger=ledger,
        lifecycle=lifecycle,
        backtest_registry=registry,
        universe_entries=_universe(["AAA"]),
    )
    assert result.eligible_families[0].justifying_params == {"k": 100.0}
    assert result.calls == ()
    assert result.rejections == ()  # scan itself found nothing — never reached risk sizing


# ---- accepted call round-trips into the ledger with the family's exit spec -


def test_accepted_call_lands_on_ledger_with_qty_and_family_exit_override(tmp_path: Path) -> None:
    lifecycle, registry = _seed_paper_live(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    result = build_calls(
        frame=_shocked_frame(),
        as_of="2026-04-13",
        equity=100_000.0,
        regime=_regime(),
        ledger=ledger,
        lifecycle=lifecycle,
        backtest_registry=registry,
        universe_entries=_universe(["AAA"]),
        earnings_lookup=lambda *_: NO_EARNINGS,
    )
    assert len(result.calls) == 1
    call = result.calls[0]
    assert call.rank == 1
    assert call.family == "oversold_reversion_v1"
    assert call.time_stop_sessions == 7  # family's doctrine override, not the 15-session default
    assert "1.5" in call.profit_rule  # family's +1.5R half-take override (risk/engine.py fix)

    entries = ledger.entries()
    assert len(entries) == 1
    assert entries[0]["id"] == call.ref_id
    assert entries[0]["qty"] == call.qty
    assert entries[0]["exit_params"]["time_stop_sessions"] == 7
    assert entries[0]["exit_params"]["take_half_at_r"] == 1.5


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

    result = build_calls(
        frame=_shocked_frame(),
        as_of="2026-04-13",
        equity=100_000.0,
        regime=_regime(),
        ledger=ledger,
        lifecycle=lifecycle,
        backtest_registry=registry,
        universe_entries=_universe(["AAA"]),
        earnings_lookup=lambda *_: NO_EARNINGS,
    )
    assert result.calls == ()
    assert len(result.rejections) == 1
    assert result.rejections[0].reason.startswith("heat_cap:")
    assert result.no_trade_reason is not None
    assert "candidate" in result.no_trade_reason


# ---- c3: regime risk_off yields an explicit, honest no-trade result -------


def test_risk_off_regime_yields_no_trade_reason(tmp_path: Path) -> None:
    lifecycle, registry = _seed_paper_live(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    result = build_calls(
        frame=_shocked_frame(),
        as_of="2026-04-13",
        equity=100_000.0,
        regime=_regime("risk_off"),
        ledger=ledger,
        lifecycle=lifecycle,
        backtest_registry=registry,
        universe_entries=_universe(["AAA"]),
        earnings_lookup=lambda *_: NO_EARNINGS,
    )
    assert result.calls == ()
    assert result.no_trade_reason is not None
    assert "risk_off" in result.no_trade_reason
    assert ledger.entries() == []  # a rejected proposal must never reach the ledger


# ---- open-position reconstruction ------------------------------------------


def test_open_positions_from_ledger_skips_unfilled_and_uses_original_stop(
    tmp_path: Path,
) -> None:
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    filled = _rec(symbol="FILLED")
    ledger.append(filled)
    ledger.append_fill(filled.id, "ord-1", 100.0, 100.0, "filled")
    unfilled = _rec(symbol="UNFILLED")
    ledger.append(unfilled)

    positions = _open_positions_from_ledger(ledger)
    assert [p.symbol for p in positions] == ["FILLED"]
    assert positions[0].current_stop_price == 92.5  # original stop, no trailing-stop tracking yet


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
