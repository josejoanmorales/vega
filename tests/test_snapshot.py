from pathlib import Path

import pandas as pd
import pytest

from vega.data.snapshot import merge_clean, write_clean
from vega.data.types import SnapshotConflictError

FRAME = pd.DataFrame({"symbol": ["AAPL"], "date": ["2026-07-02"], "close": [230.5]})


def _bars(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"symbol": s, "date": "2026-07-02", "close": c} for s, c in rows],
        columns=["symbol", "date", "close"],
    )


def _quar(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"symbol": s, "date": "2026-07-02", "reason": r} for s, r in rows],
        columns=["symbol", "date", "reason"],
    )


def test_merge_adds_new_symbols_to_an_existing_date(tmp_path: Path) -> None:
    merge_clean(
        "2026-07-02",
        "bars_equity",
        "quarantine_equity",
        _bars([("MSFT", 500.0)]),
        _quar([]),
        tmp_path,
    )
    # a symbol previously quarantined-and-unstored (or newly listed) arrives later
    added, _, frozen, _ = merge_clean(
        "2026-07-02",
        "bars_equity",
        "quarantine_equity",
        _bars([("MSFT", 501.0), ("AAPL", 230.0)]),
        _quar([]),
        tmp_path,
    )
    assert added == 1 and frozen == 1  # AAPL added, MSFT frozen (drift ignored, not lost)
    stored = pd.read_parquet(tmp_path / "clean/2026-07-02/bars_equity.parquet")
    assert set(stored["symbol"]) == {"MSFT", "AAPL"}
    assert (
        float(stored.loc[stored["symbol"] == "MSFT", "close"].iloc[0]) == 500.0
    )  # frozen value kept


def test_merge_never_contradicts_a_frozen_quarantine(tmp_path: Path) -> None:
    merge_clean(
        "2026-07-02",
        "bars_equity",
        "quarantine_equity",
        _bars([]),
        _quar([("SPGI", "diverged")]),
        tmp_path,
    )
    # a later run thinks SPGI is clean for that date — the frozen quarantine verdict stands
    added, _, frozen, _ = merge_clean(
        "2026-07-02",
        "bars_equity",
        "quarantine_equity",
        _bars([("SPGI", 400.0)]),
        _quar([]),
        tmp_path,
    )
    assert added == 0 and frozen == 1
    assert not (tmp_path / "clean/2026-07-02/bars_equity.parquet").exists()


def test_merge_counts_vendor_drift_on_frozen_rows(tmp_path: Path) -> None:
    merge_clean(
        "2026-07-02",
        "bars_equity",
        "quarantine_equity",
        _bars([("AAPL", 230.0)]),
        _quar([]),
        tmp_path,
    )
    _, _, _, drift = merge_clean(
        "2026-07-02",
        "bars_equity",
        "quarantine_equity",
        _bars([("AAPL", 231.5)]),
        _quar([]),
        tmp_path,
    )
    assert drift == 1


def test_write_once_then_identical_rewrite_is_noop(tmp_path: Path) -> None:
    p1 = write_clean("2026-07-02", "bars_equity", FRAME, root=tmp_path)
    p2 = write_clean("2026-07-02", "bars_equity", FRAME.copy(), root=tmp_path)
    assert p1 == p2 and p1.exists()


def test_rewrite_with_different_content_raises(tmp_path: Path) -> None:
    write_clean("2026-07-02", "bars_equity", FRAME, root=tmp_path)
    drifted = FRAME.assign(close=[231.0])
    with pytest.raises(SnapshotConflictError):
        write_clean("2026-07-02", "bars_equity", drifted, root=tmp_path)
