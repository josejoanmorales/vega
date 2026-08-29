"""WI-229 — expressing a bearish view without shorting.

`reduce` cuts an existing position; `avoid` declines to open one. The tests that matter
here are not the happy paths but the boundaries: that neither direction can acquire
inventory, that an `avoid` cannot be mistaken for something tradeable, and that a
counterfactual score can never leak into the realized track record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vega.briefing.engine import BriefingData
from vega.briefing.render import render
from vega.common.paths import DATA_ROOT
from vega.execution.executor import pending_longs
from vega.execution.exits import pending_reductions, reconstruct_positions
from vega.ledger.store import LedgerStore
from vega.ledger.types import POSITION_DIRECTIONS, Recommendation
from vega.lifecycle.avoided import score_avoids, scorecard
from vega.regime.regime import RegimeState

DATES = [f"2026-02-{d:02d}" for d in range(2, 20)]


def _frame(closes: dict[str, list[float]], dates: list[str]) -> pd.DataFrame:
    rows = []
    for symbol, series in closes.items():
        for d, c in zip(dates, series, strict=True):
            rows.append(
                {
                    "symbol": symbol,
                    "date": d,
                    "open": c,
                    "high": c + 1,
                    "low": c - 1,
                    "close": c,
                    "adj_close": c,
                }
            )
    return pd.DataFrame(rows)


def _long(**over: object) -> Recommendation:
    base: dict[str, object] = {
        "symbol": "AAA",
        "asset_class": "equity",
        "direction": "long",
        "thesis": "fixture",
        "confidence": 0.5,
        "horizon_days": 7,
        "entry_ref_price": 100.0,
        "stop_price": 90.0,
        "time_stop_date": "2026-03-01",
        "profit_rule": "half at +1.5R",
        "invalidation": "below SMA100",
        "signal_attribution": ("oversold_reversion_v1:1.1",),
        "as_of": DATES[0],
        "exit_params": {
            "atr_at_proposal": 5.0,
            "time_stop_sessions": 7,
            "take_half_at_r": 1.5,
            "trail_atr_mult": 2.5,
        },
    }
    base.update(over)
    return Recommendation(**base)  # type: ignore[arg-type]


def _avoid(**over: object) -> Recommendation:
    base: dict[str, object] = {
        "symbol": "BBB",
        "asset_class": "equity",
        "direction": "avoid",
        "thesis": "4.02 non-reliance filed — the drop is information",
        "confidence": 0.6,
        "horizon_days": 10,
        "entry_ref_price": 100.0,
        "invalidation": "restatement withdrawn",
        "expires_on": DATES[5],
        "as_of": DATES[0],
    }
    base.update(over)
    return Recommendation(**base)  # type: ignore[arg-type]


# ---- the contract, enforced per direction ----------------------------------


def test_avoid_needs_no_exit_spec_because_nothing_is_held() -> None:
    rec = _avoid()
    assert rec.stop_price == 0.0 and rec.qty is None and rec.profit_rule == ""


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("qty", 5.0, "qty must be absent"),
        ("stop_price", 90.0, "stop_price must be absent"),
        ("profit_rule", "half at +1.5R", "profit_rule must be absent"),
    ],
)
def test_an_avoid_carrying_tradeable_fields_is_refused(
    field: str, value: object, message: str
) -> None:
    """Absences are asserted, not merely permitted — an avoid must be structurally
    incapable of looking like a position."""
    with pytest.raises(ValueError, match=message):
        _avoid(**{field: value})


def test_avoid_without_an_expiry_is_refused() -> None:
    with pytest.raises(ValueError, match="expires_on is required"):
        _avoid(expires_on=None)


def test_reduce_needs_a_size_not_a_fresh_exit_spec() -> None:
    rec = Recommendation(
        symbol="AAA",
        asset_class="equity",
        direction="reduce",
        thesis="cut it",
        confidence=0.5,
        horizon_days=5,
        entry_ref_price=100.0,
        invalidation="thesis intact",
        target_fraction=0.5,
    )
    assert rec.target_fraction == 0.5 and rec.stop_price == 0.0


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5, None])
def test_reduce_rejects_a_meaningless_fraction(fraction: float | None) -> None:
    with pytest.raises(ValueError, match="target_fraction"):
        Recommendation(
            symbol="AAA",
            asset_class="equity",
            direction="reduce",
            thesis="cut it",
            confidence=0.5,
            horizon_days=5,
            entry_ref_price=100.0,
            invalidation="x",
            target_fraction=fraction,
        )


def test_a_long_still_requires_its_full_exit_spec() -> None:
    """Loosening the fields for avoid/reduce must not loosen them for a held position."""
    with pytest.raises(ValueError, match="stop_price is required"):
        _long(stop_price=0.0)


# ---- nothing here can acquire inventory -------------------------------------


def test_no_bearish_direction_can_open_a_position(tmp_path: Path) -> None:
    assert POSITION_DIRECTIONS == frozenset({"long"})
    ledger = LedgerStore(tmp_path / "l.jsonl")
    ledger.append(_avoid())
    ledger.append(
        Recommendation(
            symbol="CCC",
            asset_class="equity",
            direction="reduce",
            thesis="cut",
            confidence=0.5,
            horizon_days=5,
            entry_ref_price=50.0,
            invalidation="x",
            target_fraction=0.5,
            as_of=DATES[0],
        )
    )
    assert pending_longs(ledger) == [], "no bearish record may become an opening order"
    frame = _frame({"BBB": [100.0] * 6, "CCC": [50.0] * 6}, DATES[:6])
    assert reconstruct_positions(ledger, frame, DATES[5]) == []


# ---- reduce routes through the ONE existing sell path -----------------------


def test_reduce_sells_a_fraction_and_books_it_against_the_POSITION(tmp_path: Path) -> None:
    """The subtle one: a sell booked against the reduce record would land in a chain
    that owns no inventory, so the position would still read as fully held."""
    ledger = LedgerStore(tmp_path / "l.jsonl")
    held = _long(as_of=DATES[0])
    ledger.append(held)
    ledger.append_fill(held.id, "ord-1", 10.0, 100.0, "filled")
    cut = Recommendation(
        symbol="AAA",
        asset_class="equity",
        direction="reduce",
        thesis="peer 8-K",
        confidence=0.6,
        horizon_days=5,
        entry_ref_price=100.0,
        invalidation="peer risk clears",
        target_fraction=0.4,
        as_of=DATES[3],
    )
    ledger.append(cut)

    decisions = pending_reductions(ledger, _frame({"AAA": [100.0] * 4}, DATES[:4]), DATES[3])
    assert len(decisions) == 1
    d = decisions[0]
    assert d.qty == 4.0 and d.reason == "reduce"
    assert d.ref_id == held.id, "the sell must join the POSITION's chain, not the reduce's"
    assert d.ref_id != cut.id


def test_a_reduce_is_executed_once_not_on_every_run(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    held = _long(as_of=DATES[0])
    ledger.append(held)
    ledger.append_fill(held.id, "ord-1", 10.0, 100.0, "filled")
    ledger.append(
        Recommendation(
            symbol="AAA",
            asset_class="equity",
            direction="reduce",
            thesis="peer 8-K",
            confidence=0.6,
            horizon_days=5,
            entry_ref_price=100.0,
            invalidation="x",
            target_fraction=0.4,
            as_of=DATES[3],
        )
    )
    frame = _frame({"AAA": [100.0] * 4}, DATES[:4])
    assert len(pending_reductions(ledger, frame, DATES[3])) == 1
    ledger.append_fill(
        held.id, "ord-2", 4.0, 100.0, "filled", side="sell", reason="reduce", session=DATES[3]
    )
    assert pending_reductions(ledger, frame, DATES[3]) == []


def test_an_unconfirmed_entry_is_never_reduced(tmp_path: Path) -> None:
    """Presumed-filled buys reserve heat but must never sell — same rule as exits."""
    ledger = LedgerStore(tmp_path / "l.jsonl")
    held = _long(as_of=DATES[0])
    ledger.append(held)
    ledger.append_fill(held.id, "ord-1", 10.0, None, "accepted")  # never priced
    ledger.append(
        Recommendation(
            symbol="AAA",
            asset_class="equity",
            direction="reduce",
            thesis="cut",
            confidence=0.6,
            horizon_days=5,
            entry_ref_price=100.0,
            invalidation="x",
            target_fraction=0.5,
            as_of=DATES[3],
        )
    )
    assert pending_reductions(ledger, _frame({"AAA": [100.0] * 4}, DATES[:4]), DATES[3]) == []


# ---- counterfactual grading, quarantined from realized P&L ------------------


def test_an_avoid_is_scored_against_what_holding_would_have_done(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    ledger.append(_avoid())  # ref 100.0, expires DATES[5]
    frame = _frame({"BBB": [100.0, 98.0, 95.0, 92.0, 90.0, 88.0]}, DATES[:6])
    out = score_avoids(ledger, frame, DATES[5])
    assert len(out) == 1
    o = out[0]
    assert o.resolved is True and o.counterfactual is True
    assert o.return_pct == pytest.approx(-12.0)
    assert o.was_right is True


def test_an_open_window_is_unresolved_not_scored_early(tmp_path: Path) -> None:
    """Grading a decision before its own stated horizon is the cheapest way to
    manufacture a flattering number."""
    ledger = LedgerStore(tmp_path / "l.jsonl")
    ledger.append(_avoid())
    frame = _frame({"BBB": [100.0, 80.0, 70.0]}, DATES[:3])
    o = score_avoids(ledger, frame, DATES[2])[0]
    assert o.resolved is False and o.return_pct is None
    assert o.was_right is None, "unresolved is not 'wrong'"


def test_counterfactual_returns_never_enter_the_realized_track_record(tmp_path: Path) -> None:
    from vega.lifecycle.live_trades import closed_round_trips

    ledger = LedgerStore(tmp_path / "l.jsonl")
    avoid = _avoid()
    ledger.append(avoid)
    # Even a forged fill on an avoid must not produce a round trip.
    ledger.append_fill(avoid.id, "ord-x", 10.0, 100.0, "filled")
    ledger.append_fill(avoid.id, "ord-y", 10.0, 88.0, "filled", side="sell", session=DATES[5])
    assert closed_round_trips(ledger, DATES[:6]) == {}


def test_scorecard_reports_hit_rate_and_mean_forgone_move(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    ledger.append(_avoid(symbol="BBB"))
    ledger.append(_avoid(symbol="DDD"))
    frame = _frame({"BBB": [100.0] * 5 + [90.0], "DDD": [100.0] * 5 + [110.0]}, DATES[:6])
    sc = scorecard(score_avoids(ledger, frame, DATES[5]))
    assert (sc.n_resolved, sc.n_right, sc.n_open) == (2, 1, 0)
    assert sc.hit_rate == pytest.approx(0.5)
    assert sc.avg_avoided_return_pct == pytest.approx(0.0)


# ---- the briefing keeps the lanes visibly apart ------------------------------


def _briefing(**over: object) -> BriefingData:
    base: dict[str, object] = {
        "as_of": DATES[5],
        "regime": RegimeState(
            as_of=DATES[5],
            trend="risk_on",
            vix=15.0,
            vix_band="calm",
            breadth_pct=60.0,
            crypto_fg=50,
            composite="risk_on",
        ),
        "movers_equity": pd.DataFrame(columns=["symbol", "close", "pct"]),
        "movers_crypto": pd.DataFrame(columns=["symbol", "close", "pct"]),
        "events": [],
        "failures": [],
        "store_range": (DATES[0], DATES[5]),
        "quarantined_today": 0,
    }
    base.update(over)
    return BriefingData(**base)  # type: ignore[arg-type]


def test_bearish_decisions_render_in_their_own_section() -> None:
    text = render(
        _briefing(
            bearish=(
                {
                    "direction": "avoid",
                    "symbol": "BBB",
                    "thesis": "4.02 non-reliance",
                    "invalidation": "restatement withdrawn",
                    "expires_on": DATES[5],
                },
                {
                    "direction": "reduce",
                    "symbol": "AAA",
                    "thesis": "peer 8-K",
                    "invalidation": "peer risk clears",
                    "target_fraction": 0.4,
                },
            )
        )
    )
    assert "## Bearish decisions — reduce / avoid" in text
    assert "No shorting" in text
    assert "cut 40%" in text
    assert "## Ranked calls" not in text, "a bearish decision is not a ranked call"


def test_the_scorecard_is_labelled_counterfactual_wherever_it_appears() -> None:
    text = render(
        _briefing(
            avoided_scorecard={
                "n_resolved": 3,
                "n_open": 1,
                "n_right": 2,
                "avg_avoided_return_pct": -4.2,
            }
        )
    )
    assert "counterfactual" in text.lower()
    assert "not evidence of alpha" in text
    assert "-4.20%" in text


# ---- append-only: the real ledger still parses ------------------------------


def test_the_live_ledger_still_parses_under_the_widened_contract() -> None:
    """Criterion 1: widening DIRECTIONS must not orphan a single existing record."""
    path = DATA_ROOT / "ledger/ledger.jsonl"
    if not path.exists():
        pytest.skip("no live ledger in this checkout (data/ is gitignored)")
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    recs = [r for r in recs if r.get("type") == "recommendation"]
    assert recs, "the live ledger has no recommendations to check"
    for r in recs:
        payload = {k: v for k, v in r.items() if k != "type"}
        payload["signal_attribution"] = tuple(payload.get("signal_attribution") or ())
        Recommendation(**payload)  # raises if the widened contract orphaned it
