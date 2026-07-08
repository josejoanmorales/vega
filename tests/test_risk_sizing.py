import pytest

from vega.risk.sizing import (
    GAP_STRESS_MULT,
    SizingError,
    compute_qty,
    compute_stop,
)


def test_stop_below_entry_by_k_times_atr() -> None:
    stop = compute_stop(entry_price=100.0, atr=2.0, asset_class="equity")
    assert stop == pytest.approx(96.0)  # k=2.0 for equities


def test_stop_rejects_nonpositive_inputs() -> None:
    with pytest.raises(SizingError):
        compute_stop(entry_price=0.0, atr=2.0, asset_class="equity")
    with pytest.raises(SizingError):
        compute_stop(entry_price=100.0, atr=0.0, asset_class="equity")


def test_qty_gap_binds_at_default_equity_params_giving_point_eight_of_base() -> None:
    stop = compute_stop(100.0, atr=2.0, asset_class="equity")  # stop=96, distance=4
    result = compute_qty(100.0, stop, equity=100_000.0, asset_class="equity")
    r_dollars = 0.0075 * 100_000.0  # 750
    qty_base = r_dollars / 4.0
    assert result.qty == pytest.approx(0.8 * qty_base)
    assert result.initial_r_dollars == pytest.approx(0.8 * r_dollars)


def test_crypto_gap_tie_gives_nominal_one_r_and_worst_case_two_r() -> None:
    stop = compute_stop(100.0, atr=4.0, asset_class="crypto")  # k=2.5 -> stop=90, distance=10
    result = compute_qty(100.0, stop, equity=100_000.0, asset_class="crypto")
    r_dollars = 0.0075 * 100_000.0
    assert result.initial_r_dollars == pytest.approx(r_dollars)  # base==gap at G=2.0
    assert result.worst_case_r_multiple == pytest.approx(2.0)


@pytest.mark.parametrize("asset_class", ["equity", "crypto"])
def test_worst_case_never_exceeds_two_r_across_a_range_of_k(asset_class: str) -> None:
    equity = 250_000.0
    r_dollars = 0.0075 * equity
    for entry, atr in [(50.0, 1.0), (500.0, 30.0), (10.0, 0.05)]:
        stop = compute_stop(entry, atr, asset_class)
        result = compute_qty(entry, stop, equity, asset_class)
        assert result.worst_case_r_dollars <= 2.0 * r_dollars + 1e-6
        assert result.initial_r_dollars <= r_dollars + 1e-6


def test_qty_rejects_stop_at_or_above_entry() -> None:
    with pytest.raises(SizingError):
        compute_qty(100.0, 100.0, 100_000.0, "equity")
    with pytest.raises(SizingError):
        compute_qty(100.0, 105.0, 100_000.0, "equity")


def test_nominal_risk_never_exceeds_one_r_for_any_configured_asset_class() -> None:
    # the min(base, gap) formula itself is the invariant — there is deliberately
    # no clamp (a prior review found a dead clamp giving false safety confidence)
    r_dollars = 0.0075 * 100_000.0
    for asset_class in GAP_STRESS_MULT:
        stop = compute_stop(100.0, atr=1.0, asset_class=asset_class)
        result = compute_qty(100.0, stop, equity=100_000.0, asset_class=asset_class)
        assert result.initial_r_dollars <= r_dollars + 1e-6


def test_gap_stress_constants_within_the_contracted_two_to_three_range() -> None:
    assert 2.0 <= GAP_STRESS_MULT["equity"] <= 3.0
    assert 2.0 <= GAP_STRESS_MULT["crypto"] <= 3.0


def test_doctrine_constants_are_the_shared_module_not_local_copies() -> None:
    from vega.backtest import signals
    from vega.common import doctrine
    from vega.risk import sizing

    assert sizing.STOP_ATR_MULT is doctrine.STOP_ATR_MULT
    assert sizing.GAP_STRESS_MULT is doctrine.GAP_STRESS_MULT
    # backtest EntryProposal defaults come from the same doctrine module
    proposal_defaults = signals.EntryProposal(
        symbol="X",
        signal_family="f",
        signal_version="1",
        thesis="t",
        confidence=0.5,
        invalidation="i",
    )
    assert proposal_defaults.time_stop_days == doctrine.DEFAULT_TIME_STOP_SESSIONS
    assert proposal_defaults.stop_atr_mult == doctrine.STOP_ATR_MULT["equity"]
    assert proposal_defaults.profit_take_half_at_r == doctrine.PROFIT_TAKE_HALF_AT_R
    assert proposal_defaults.profit_trail_atr_mult == doctrine.PROFIT_TRAIL_ATR_MULT
