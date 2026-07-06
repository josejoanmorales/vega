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
