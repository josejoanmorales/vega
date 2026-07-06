from pathlib import Path

import pandas as pd
import pytest

from vega.briefing.engine import BriefingData, top_movers
from vega.briefing.render import render, write_briefing
from vega.data.types import SnapshotConflictError
from vega.regime.calendar import MacroEvent
from vega.regime.regime import RegimeState


def _bars(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"symbol": s, "date": d, "adj_close": c} for s, d, c in rows])


def _data(failures: list[dict[str, str]] | None = None) -> BriefingData:
    movers = top_movers(
        _bars(
            [
                ("AAPL", "2026-07-01", 100.0),
                ("AAPL", "2026-07-02", 103.0),
                ("MSFT", "2026-07-01", 500.0),
                ("MSFT", "2026-07-02", 490.0),
            ]
        )
    )
    return BriefingData(
        as_of="2026-07-02",
        regime=RegimeState("2026-07-02", "risk_on", 16.15, "normal", 67.6, 24, "caution"),
        movers_equity=movers,
        movers_crypto=top_movers(_bars([])),
        events=[MacroEvent("2026-07-14", "CPI release")],
        failures=failures or [],
        store_range=("2025-06-02", "2026-07-02"),
        quarantined_today=3,
    )


def test_top_movers_needs_both_sessions_and_sorts() -> None:
    movers = _data().movers_equity
    assert list(movers["symbol"]) == ["AAPL", "MSFT"]
    assert movers.iloc[0]["pct"] == 3.0 and movers.iloc[1]["pct"] == -2.0


def test_render_is_deterministic_and_carries_provenance() -> None:
    a, b = render(_data()), render(_data())
    assert a == b
    assert "Composite: CAUTION" in a and "2026-07-14" in a
    assert "validated local store" in a and "3 symbol-days quarantined" in a
    assert "Execution failures" not in a


def test_failures_section_appears_when_present() -> None:
    out = render(
        _data(failures=[{"at": "t", "symbol": "GRAM", "ref_id": "abcdefgh", "error": "boom"}])
    )
    assert "Execution failures" in out and "GRAM" in out


def test_briefing_write_once(tmp_path: Path) -> None:
    p1 = write_briefing(_data(), root=tmp_path)
    p2 = write_briefing(_data(), root=tmp_path)
    assert p1 == p2
    with pytest.raises(SnapshotConflictError):
        write_briefing(
            _data(failures=[{"at": "t", "symbol": "X", "ref_id": "y" * 8, "error": "e"}]),
            root=tmp_path,
        )
