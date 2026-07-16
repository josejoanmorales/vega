from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from vega.briefing.calls import EligibleFamily, RenderedCall, RenderedRejection
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


def test_no_eligible_families_renders_v1_sections_unchanged() -> None:
    # byte-identical to the pre-WI-067 render — the ranked-calls block must not
    # appear at all until a family is actually eligible for recommendations
    assert "Ranked calls" not in render(_data())


def test_no_trade_renders_explicit_line_with_reason() -> None:
    data = replace(
        _data(),
        eligible_families=(
            EligibleFamily("oversold_reversion_v1", "paper-live", "run-1", {"k": 2.0}, 1.3),
        ),
        no_trade_reason="regime composite is risk_off as of 2026-07-02 — no entries permitted",
    )
    out = render(data)
    assert "## Ranked calls" in out
    assert "**No trade today** — regime composite is risk_off" in out
    assert "oversold_reversion_v1` (paper-live)" in out


def test_ranked_calls_table_renders_with_rejections() -> None:
    call = RenderedCall(
        rank=1,
        symbol="AAPL",
        family="oversold_reversion_v1",
        version="1.1",
        thesis="3-session shock reversion",
        qty=1.234567,
        entry_ref_price=230.0,
        stop_price=221.0,
        worst_case_r_multiple=1.8,
        time_stop_sessions=7,
        time_stop_date="2026-07-11",
        profit_rule="half at +1.5R, trail remainder via 2.5xATR chandelier stop",
        invalidation="close below the 100-session SMA",
        heat_after_r={"total": 0.98, "us_equity_beta": 0.98},
        ref_id="rec-1",
    )
    rejection = RenderedRejection(
        "MSFT", "oversold_reversion_v1", "earnings_unknown", "vendor down"
    )
    data = replace(
        _data(),
        eligible_families=(
            EligibleFamily("oversold_reversion_v1", "paper-live", "run-1", {"k": 2.0}, 1.3),
        ),
        calls=(call,),
        rejections=(rejection,),
    )
    out = render(data)
    assert "AAPL" in out and "1.234567" in out and "7 sessions (2026-07-11)" in out
    assert "### Considered and rejected" in out and "MSFT" in out and "earnings_unknown" in out


def test_briefing_write_once(tmp_path: Path) -> None:
    p1 = write_briefing(_data(), root=tmp_path)
    p2 = write_briefing(_data(), root=tmp_path)
    assert p1 == p2
    with pytest.raises(SnapshotConflictError):
        write_briefing(
            _data(failures=[{"at": "t", "symbol": "X", "ref_id": "y" * 8, "error": "e"}]),
            root=tmp_path,
        )
