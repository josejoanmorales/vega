from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import make_ohlc_frame, steep_uptrend_then_shock
from vega.backtest.registry import BacktestRegistry
from vega.data.types import UniverseEntry
from vega.ledger.store import LedgerStore
from vega.ledger.types import Recommendation
from vega.lifecycle.lifecycle import LifecycleRegistry
from vega.lifecycle.rationale import RationaleRegistry
from vega.regime.regime import RegimeState
from vega.risk.gates import EarningsFact
from vega.web import dashboard

AS_OF = "2026-04-13"  # last date of the 103-bar conftest fixture


def _shocked_frame(symbol: str = "AAA", drop_total: float = 39.0):
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
        "as_of": AS_OF,
    }
    base.update(overrides)
    return Recommendation(**base)  # type: ignore[arg-type]


def _seed_paper_live(tmp_path: Path, params: dict[str, object] | None = None):
    params = params if params is not None else {"k": 2.0}
    rationale = RationaleRegistry(tmp_path / "rationale.jsonl")
    rationale.record("oversold_reversion_v1", "a real economic rationale", author="human:jose")
    registry = BacktestRegistry(tmp_path / "registry.jsonl")
    registry.record_run(
        signal_family="oversold_reversion_v1",
        signal_version="1.1",
        param_grid_size=1,
        universe_version="v1",
        data_span=("2025-01-01", AS_OF),
        n_folds=2,
        fold_metrics=[{"sharpe": 1.0}, {"sharpe": 1.5}],
        aggregate_metrics={"sharpe": 1.3, "n_trades": 40},
        verdict="pass",
        holdout_evaluated=True,
        promotion_bar=0.8,
        notes=[],
        holdout_sharpe=2.0,
        signal_params=params,
    )
    lifecycle = LifecycleRegistry(tmp_path / "lifecycle.jsonl")
    lifecycle.promote_to_backtested(
        "oversold_reversion_v1", rationale, registry, actor="agent:sonnet"
    )
    lifecycle.promote_to_paper_live("oversold_reversion_v1", actor="human:jose")
    return lifecycle, registry


# ---- positions ---------------------------------------------------------


def test_positions_matches_reconstruction(tmp_path: Path, monkeypatch) -> None:
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    rec = _rec()
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, 101.0, "filled")

    monkeypatch.setattr(dashboard, "_max_session", lambda: AS_OF)
    monkeypatch.setattr(dashboard, "load_signal_frame", lambda as_of: _shocked_frame())

    rows = dashboard.positions(ledger)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAA"
    assert rows[0]["qty"] == 10.0
    assert rows[0]["entry_price"] == 101.0
    assert rows[0]["is_pending"] is False


def test_positions_empty_when_no_store_data(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "_max_session", lambda: "")
    assert dashboard.positions(LedgerStore(tmp_path / "ledger.jsonl")) == []


# ---- signal_health -------------------------------------------------------


def test_signal_health_never_demotes(tmp_path: Path, monkeypatch) -> None:
    lifecycle, registry = _seed_paper_live(tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    monkeypatch.setattr(dashboard, "_max_session", lambda: AS_OF)
    monkeypatch.setattr(dashboard, "LifecycleRegistry", lambda: lifecycle)
    monkeypatch.setattr(dashboard, "BacktestRegistry", lambda: registry)

    lifecycle_events_before = len(lifecycle.history("oversold_reversion_v1"))
    rows = dashboard.signal_health(ledger)
    assert len(rows) == 1
    assert rows[0]["family"] == "oversold_reversion_v1"
    assert rows[0]["state"] == "paper-live"
    assert "insufficient_sample" in rows[0]["reason"]
    # the regression that matters: a page load must never append a transition
    assert len(lifecycle.history("oversold_reversion_v1")) == lifecycle_events_before


# ---- failures -------------------------------------------------------------


def test_failures_returns_newest_first(tmp_path: Path, monkeypatch) -> None:
    from vega.execution import executor
    from vega.execution.executor import record_failure

    path = tmp_path / "f.jsonl"
    record_failure("ref-1", "AAA", "first", path)
    record_failure("ref-2", "BBB", "second", path)
    monkeypatch.setattr(dashboard, "read_failures", lambda: executor.read_failures(path))

    rows = dashboard.failures()
    assert [r["symbol"] for r in rows] == ["BBB", "AAA"]


# ---- inspect_symbol -------------------------------------------------------


def _patch_inspector(monkeypatch, tmp_path: Path, params: dict[str, object] | None = None):
    lifecycle, registry = _seed_paper_live(tmp_path, params=params)
    monkeypatch.setattr(dashboard, "LifecycleRegistry", lambda: lifecycle)
    monkeypatch.setattr(dashboard, "BacktestRegistry", lambda: registry)
    monkeypatch.setattr(
        dashboard, "load_universe", lambda: [UniverseEntry("AAA", "equity", "AAA Inc")]
    )
    monkeypatch.setattr(
        dashboard, "assemble", lambda: SimpleNamespace(as_of=AS_OF, regime=_regime())
    )
    monkeypatch.setattr(dashboard, "_max_session", lambda: AS_OF)
    monkeypatch.setattr(dashboard, "load_signal_frame", lambda as_of: _shocked_frame())
    monkeypatch.setattr(
        dashboard.EarningsFact, "lookup", classmethod(lambda cls, s, a: EarningsFact("none"))
    )
    return lifecycle, registry


def test_inspect_unknown_symbol_raises(tmp_path: Path, monkeypatch) -> None:
    _patch_inspector(monkeypatch, tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    with pytest.raises(dashboard.SymbolNotInUniverse):
        dashboard.inspect_symbol("ZZZZ", ledger)


def test_inspect_firing_symbol_reports_thesis(tmp_path: Path, monkeypatch) -> None:
    _patch_inspector(monkeypatch, tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    result = dashboard.inspect_symbol("AAA", ledger)
    assert result["symbol"] == "AAA"
    assert result["position"] is None
    assert len(result["signals"]) == 1
    assert result["signals"][0]["fires_today"] is True
    assert result["signals"][0]["thesis"]
    assert result["gates"]["blocked"] is False
    assert "read-only" in result["read_only_notice"]


def test_inspect_non_firing_symbol_reports_false(tmp_path: Path, monkeypatch) -> None:
    # k=100 is impossibly strict -- proves justifying_params actually drive the scan
    _patch_inspector(monkeypatch, tmp_path, params={"k": 100.0})
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    result = dashboard.inspect_symbol("AAA", ledger)
    assert result["signals"][0]["fires_today"] is False
    assert result["signals"][0]["thesis"] is None


def test_inspect_reports_held_position(tmp_path: Path, monkeypatch) -> None:
    _patch_inspector(monkeypatch, tmp_path)
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    rec = _rec()
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, 101.0, "filled")
    result = dashboard.inspect_symbol("AAA", ledger)
    assert result["position"] is not None
    assert result["position"]["held"] is True
    assert result["position"]["qty"] == 10.0


def test_inspect_never_writes_to_ledger_or_lifecycle(tmp_path: Path, monkeypatch) -> None:
    lifecycle, registry = _patch_inspector(monkeypatch, tmp_path)
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = LedgerStore(ledger_path)
    rec = _rec()
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, 101.0, "filled")

    ledger_before = ledger_path.read_text()
    lifecycle_before = (tmp_path / "lifecycle.jsonl").read_text()

    dashboard.inspect_symbol("AAA", ledger)
    with pytest.raises(dashboard.SymbolNotInUniverse):
        dashboard.inspect_symbol("ZZZZ", ledger)  # 404 path too

    assert ledger_path.read_text() == ledger_before
    assert (tmp_path / "lifecycle.jsonl").read_text() == lifecycle_before
