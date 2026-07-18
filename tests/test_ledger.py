from pathlib import Path

import pytest

from vega.ledger.store import LedgerStore
from vega.ledger.types import Recommendation


def _rec(**overrides: object) -> Recommendation:
    base: dict[str, object] = {
        "symbol": "AAPL",
        "asset_class": "equity",
        "direction": "long",
        "thesis": "trend pullback to rising 20DMA in a risk-on regime",
        "confidence": 0.62,
        "horizon_days": 10,
        "entry_ref_price": 230.0,
        "stop_price": 221.0,
        "time_stop_date": "2026-07-20",
        "profit_rule": "half at +2R, trail rest at 10DMA",
        "invalidation": "close below the June swing low on consolidated volume",
        "signal_attribution": ("trend_pullback_v1",),
    }
    base.update(overrides)
    return Recommendation(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stop_price", 0.0),
        ("time_stop_date", "not-a-date"),
        ("profit_rule", "  "),
        ("invalidation", ""),
    ],
)
def test_missing_exit_spec_component_cannot_be_instantiated(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _rec(**{field: value})


def test_long_without_attribution_rejected() -> None:
    with pytest.raises(ValueError, match="signal_attribution"):
        _rec(signal_attribution=())


def test_append_persists_and_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    rid = LedgerStore(path).append(_rec())
    reopened = LedgerStore(path)
    assert [r["id"] for r in reopened.entries()] == [rid]


def test_supersede_chain_resolves_to_latest(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path / "ledger.jsonl")
    first = _rec()
    store.append(first)
    corrected = _rec(stop_price=222.5, supersedes=first.id)
    store.append(corrected)
    latest = store.latest()
    assert len(latest) == 1 and latest[0]["id"] == corrected.id


def test_override_links_and_rejects_unknown_ref(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path / "ledger.jsonl")
    rid = store.append(_rec())
    store.append_override(rid, "resize", "half size into CPI week", actor="human:jose")
    assert store.overrides()[0]["ref_id"] == rid
    with pytest.raises(ValueError, match="unknown recommendation"):
        store.append_override("nope", "skip", "", actor="human:jose")


def test_append_fill_defaults_side_to_buy(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path / "l.jsonl")
    rid = store.append(_rec())
    store.append_fill(rid, "ord-1", 4.0, 230.0, "filled")
    fill = store.fills()[0]
    assert fill["side"] == "buy" and fill["reason"] is None and fill["session"] is None


def test_latest_with_all_fills_includes_buy_and_sell(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path / "l.jsonl")
    rid = store.append(_rec())
    store.append_fill(rid, "ord-1", 4.0, 230.0, "filled")
    store.append_fill(
        rid,
        "ord-2",
        2.0,
        250.0,
        "filled",
        side="sell",
        reason="profit_partial",
        session="2026-07-10",
    )
    (rec, fills) = store.latest_with_all_fills()[0]
    assert rec["id"] == rid
    assert {f["side"] for f in fills} == {"buy", "sell"}
    sell = next(f for f in fills if f["side"] == "sell")
    assert sell["reason"] == "profit_partial" and sell["session"] == "2026-07-10"


def test_latest_with_all_fills_resolves_through_supersede_chains(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path / "l.jsonl")
    first = _rec()
    store.append(first)
    store.append_fill(first.id, "ord-1", 4.0, 230.0, "filled")
    corrected = _rec(stop_price=222.0, supersedes=first.id)
    store.append(corrected)
    store.append_fill(first.id, "ord-2", 4.0, 231.0, "filled", side="sell", reason="stop")
    (rec, fills) = store.latest_with_all_fills()[0]
    assert rec["id"] == corrected.id  # the SURVIVING rec in the chain
    assert len(fills) == 2  # both fills, filed against the superseded id, still resolve


def test_latest_with_fills_ignores_sell_fills(tmp_path: Path) -> None:
    # a sold-out position must still resolve as "entered" for pending_longs'
    # purposes — latest_with_fills answers "was it entered", not "is it open"
    store = LedgerStore(tmp_path / "l.jsonl")
    rid = store.append(_rec())
    store.append_fill(rid, "ord-1", 4.0, 230.0, "filled")
    store.append_fill(rid, "ord-2", 4.0, 240.0, "filled", side="sell", reason="stop")
    (rec, fill) = store.latest_with_fills()[0]
    assert fill is not None and fill["side"] == "buy" and fill["price"] == 230.0


def test_file_only_ever_grows(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    store = LedgerStore(path)
    sizes = []
    first = _rec()
    store.append(first)
    sizes.append(path.stat().st_size)
    store.append(_rec(symbol="MSFT"))
    sizes.append(path.stat().st_size)
    store.append_override(first.id, "skip", "regime turned", actor="human:jose")
    sizes.append(path.stat().st_size)
    assert sizes == sorted(sizes) and len(set(sizes)) == 3
