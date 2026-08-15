from pathlib import Path

import duckdb
import pandas as pd
import pytest

from vega.data.types import UniverseEntry
from vega.execution import cost_audit
from vega.ledger.store import LedgerStore
from vega.ledger.types import Recommendation

_BARS_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]


def _seed_store(root: Path, bars: list[dict]) -> None:
    root.mkdir(exist_ok=True)
    frame = pd.DataFrame(bars, columns=_BARS_COLUMNS)  # noqa: F841 — duckdb resolves it by name in SQL
    con = duckdb.connect(str(root / "vega.duckdb"))
    con.execute("CREATE TABLE bars AS SELECT * FROM frame")
    con.close()


def _bar(symbol: str, date: str, open_: float, close: float, volume: float = 1_000_000.0) -> dict:
    return {
        "symbol": symbol,
        "date": date,
        "open": open_,
        "high": max(open_, close),
        "low": min(open_, close),
        "close": close,
        "adj_close": close,
        "volume": volume,
        "source": "yfinance",
    }


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
        "signal_attribution": ("test:1",),
    }
    base.update(overrides)
    return Recommendation(**base)  # type: ignore[arg-type]


def _patch_universe(monkeypatch, asset_class: str = "equity") -> None:
    monkeypatch.setattr(
        cost_audit, "load_universe", lambda: [UniverseEntry("AAA", asset_class, "AAA Inc")]
    )


# ---- implementation shortfall (entries) ------------------------------------


def test_implementation_shortfall_sign_convention(tmp_path: Path, monkeypatch) -> None:
    """Positive = unfavorable. A buy filling ABOVE its reference price cost
    money; a buy filling below it was favorable — verified against the exact
    real-CDW numbers this module was built to explain."""
    _patch_universe(monkeypatch)
    _seed_store(tmp_path, [])
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    rec = _rec(entry_ref_price=130.86)
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, 132.548345, "filled")  # paid MORE

    report = cost_audit.audit(ledger, root=tmp_path)

    assert report.implementation_shortfall.n == 1
    expected = (132.548345 - 130.86) / 130.86 * 10_000.0
    assert report.implementation_shortfall.mean_bps == pytest.approx(expected)
    assert report.implementation_shortfall.mean_bps > 0  # unfavorable, correctly positive


def test_favorable_fill_is_negative_bps(tmp_path: Path, monkeypatch) -> None:
    _patch_universe(monkeypatch)
    _seed_store(tmp_path, [])
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    rec = _rec(entry_ref_price=111.77)
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, 110.66, "filled")  # paid LESS

    report = cost_audit.audit(ledger, root=tmp_path)
    assert report.implementation_shortfall.mean_bps < 0


def test_shortfall_is_entry_only_by_construction(tmp_path: Path, monkeypatch) -> None:
    """A time-stop exit has no target price to be 'short' against — this must
    not be silently filled in with something misleading."""
    _patch_universe(monkeypatch)
    _seed_store(tmp_path, [])
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    rec = _rec()
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, 100.0, "filled")
    ledger.append_fill(
        rec.id,
        "ord-2",
        10.0,
        105.0,
        "filled",
        side="sell",
        reason="time_stop",
        session="2026-04-20",
    )

    report = cost_audit.audit(ledger, root=tmp_path)
    exit_fill = next(f for f in report.fills if f.side == "sell")
    assert exit_fill.entry_ref_price is None
    assert exit_fill.implementation_shortfall_bps is None


# ---- execution slippage (requires filled_at) -------------------------------


def test_execution_slippage_uses_filled_at_not_wall_clock_at(tmp_path: Path, monkeypatch) -> None:
    """THE property this module exists to get right: the real execution
    session comes from Alpaca's filled_at, never from when reconciliation
    happened to observe it."""
    _patch_universe(monkeypatch)
    _seed_store(tmp_path, [_bar("AAA", "2026-04-15", open_=100.0, close=101.0)])
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    rec = _rec(entry_ref_price=99.0)
    ledger.append(rec)
    # Reconciliation observed this fill a full day later than it happened.
    ledger.append_fill(
        rec.id, "ord-1", 10.0, 100.5, "filled", filled_at="2026-04-15T09:30:00-04:00"
    )

    report = cost_audit.audit(ledger, root=tmp_path)
    fill = report.fills[0]
    assert fill.fill_session == "2026-04-15"
    assert fill.session_open == 100.0
    expected = (100.5 - 100.0) / 100.0 * 10_000.0
    assert fill.execution_slippage_bps == pytest.approx(expected)
    assert report.execution_slippage_entries.n == 1


def test_fills_without_filled_at_are_excluded_never_guessed(tmp_path: Path, monkeypatch) -> None:
    """Historical fills predate this item's filled_at capture. They must be
    excluded from execution slippage and flagged in caveats — never silently
    dropped, and never estimated from the `at` reconciliation timestamp."""
    _patch_universe(monkeypatch)
    _seed_store(tmp_path, [])
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    rec = _rec()
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, 100.0, "filled")  # no filled_at

    report = cost_audit.audit(ledger, root=tmp_path)
    assert report.fills[0].fill_session is None
    assert report.fills[0].execution_slippage_bps is None
    assert report.execution_slippage_entries.n == 0
    assert any("no recorded execution session" in c for c in report.caveats)


def test_exit_execution_slippage_is_tracked_separately_from_entries(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_universe(monkeypatch)
    _seed_store(
        tmp_path,
        [
            _bar("AAA", "2026-04-15", open_=100.0, close=101.0),
            _bar("AAA", "2026-04-20", open_=110.0, close=111.0),
        ],
    )
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    rec = _rec()
    ledger.append(rec)
    ledger.append_fill(
        rec.id, "ord-1", 10.0, 100.5, "filled", filled_at="2026-04-15T09:30:00-04:00"
    )
    ledger.append_fill(
        rec.id,
        "ord-2",
        10.0,
        109.0,
        "filled",
        side="sell",
        reason="time_stop",
        session="2026-04-19",
        filled_at="2026-04-20T09:30:00-04:00",
    )

    report = cost_audit.audit(ledger, root=tmp_path)
    assert report.execution_slippage_entries.n == 1
    assert report.execution_slippage_exits.n == 1
    # sell filling BELOW the open (109 < 110) is unfavorable -> positive
    assert report.execution_slippage_exits.mean_bps > 0


# ---- doctrine check ----------------------------------------------------------


def test_doctrine_check_is_none_not_a_guess_with_zero_evidence(tmp_path: Path, monkeypatch) -> None:
    _patch_universe(monkeypatch)
    _seed_store(tmp_path, [])
    report = cost_audit.audit(LedgerStore(tmp_path / "ledger.jsonl"), root=tmp_path)
    assert report.doctrine.n == 0
    assert report.doctrine.doctrine_holds is None


def test_doctrine_check_flags_a_real_violation(tmp_path: Path, monkeypatch) -> None:
    """If realized slippage exceeds what the backtest would have charged, the
    doctrine is inverted -- this must be reported as a violation, not
    smoothed over."""
    _patch_universe(monkeypatch)
    # Thin volume -> "standard" tier -> backtest would charge 20bps.
    bars = [_bar("AAA", "2026-04-15", open_=100.0, close=100.0, volume=1_000.0)]
    _seed_store(tmp_path, bars)
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    rec = _rec()
    ledger.append(rec)
    # Filled 5% (500bps) above the open -- vastly more than any tier cost.
    ledger.append_fill(
        rec.id, "ord-1", 10.0, 105.0, "filled", filled_at="2026-04-15T09:30:00-04:00"
    )

    report = cost_audit.audit(ledger, root=tmp_path)
    assert report.doctrine.doctrine_holds is False
    assert "VIOLATED" in report.doctrine.detail


def test_doctrine_check_confirms_when_realized_cost_is_low(tmp_path: Path, monkeypatch) -> None:
    _patch_universe(monkeypatch)
    bars = [_bar("AAA", "2026-04-15", open_=100.0, close=100.0, volume=1_000_000_000.0)]
    _seed_store(tmp_path, bars)
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    rec = _rec()
    ledger.append(rec)
    # 1bp of slippage -- comfortably under any backtest tier.
    ledger.append_fill(
        rec.id, "ord-1", 10.0, 100.01, "filled", filled_at="2026-04-15T09:30:00-04:00"
    )

    report = cost_audit.audit(ledger, root=tmp_path)
    assert report.doctrine.doctrine_holds is True
    assert "HOLDS" in report.doctrine.detail


# ---- honesty contract: sample size, no automatic action --------------------


def test_small_sample_is_flagged_in_caveats(tmp_path: Path, monkeypatch) -> None:
    _patch_universe(monkeypatch)
    _seed_store(tmp_path, [])
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    rec = _rec()
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, 100.0, "filled")

    report = cost_audit.audit(ledger, root=tmp_path)
    assert report.implementation_shortfall.n == 1
    assert any("below" in c and "directional only" in c for c in report.caveats)


def test_cancelled_orders_are_excluded_not_counted_as_zero_cost(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_universe(monkeypatch)
    _seed_store(tmp_path, [])
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    rec = _rec()
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 0.0, None, "canceled")  # never traded

    report = cost_audit.audit(ledger, root=tmp_path)
    assert report.fills == ()
    assert report.implementation_shortfall.n == 0


# ---- read-only proof ---------------------------------------------------------


def test_audit_never_writes_to_the_ledger(tmp_path: Path, monkeypatch) -> None:
    _patch_universe(monkeypatch)
    bars = [_bar("AAA", "2026-04-15", open_=100.0, close=101.0)]
    _seed_store(tmp_path, bars)
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = LedgerStore(ledger_path)
    rec = _rec()
    ledger.append(rec)
    ledger.append_fill(
        rec.id, "ord-1", 10.0, 100.5, "filled", filled_at="2026-04-15T09:30:00-04:00"
    )

    before = ledger_path.read_text()
    cost_audit.audit(ledger, root=tmp_path)
    cost_audit.render_text(cost_audit.audit(ledger, root=tmp_path))
    assert ledger_path.read_text() == before


def test_audit_opens_the_store_read_only(tmp_path: Path, monkeypatch) -> None:
    """A read_only=False connection would hold a lock that blocks the live
    pipeline's own writer -- the audit must never do that."""
    _patch_universe(monkeypatch)
    _seed_store(tmp_path, [])
    LedgerStore(tmp_path / "ledger.jsonl")
    cost_audit.audit(LedgerStore(tmp_path / "ledger.jsonl"), root=tmp_path)
    # A second, genuinely write-mode connection must still be obtainable --
    # proves the audit closed its (read-only) connection and held no lock.
    con = duckdb.connect(str(tmp_path / "vega.duckdb"))
    con.execute("SELECT 1")
    con.close()


# ---- crypto tiering ----------------------------------------------------------


def test_crypto_fills_use_the_flat_crypto_tier_not_dollar_volume(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        cost_audit, "load_universe", lambda: [UniverseEntry("BTC", "crypto", "Bitcoin")]
    )
    _seed_store(tmp_path, [_bar("BTC", "2026-04-15", open_=50_000.0, close=50_000.0)])
    ledger = LedgerStore(tmp_path / "ledger.jsonl")
    rec = _rec(symbol="BTC", asset_class="crypto", entry_ref_price=50_000.0)
    ledger.append(rec)
    ledger.append_fill(
        rec.id, "ord-1", 0.1, 50_050.0, "filled", filled_at="2026-04-15T00:00:00+00:00"
    )

    report = cost_audit.audit(ledger, root=tmp_path)
    assert report.doctrine.n == 1
    assert report.doctrine.mean_backtest_tier_bps == 35.0  # BTC = CRYPTO_MAJOR_BPS
