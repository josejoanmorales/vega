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


def test_single_position_cap_clamps_extreme_gap_multiples() -> None:
    # a pathological gap multiple below 1 would otherwise let qty_base dominate
    # and push nominal risk toward 2R; the 1.5R clamp must still hold.
    stop = compute_stop(100.0, atr=1.0, asset_class="equity")
    result = compute_qty(100.0, stop, equity=100_000.0, asset_class="equity", risk_fraction=0.0075)
    r_dollars = 0.0075 * 100_000.0
    assert result.initial_r_dollars <= 1.5 * r_dollars + 1e-6


def test_gap_stress_constants_within_the_contracted_two_to_three_range() -> None:
    assert 2.0 <= GAP_STRESS_MULT["equity"] <= 3.0
    assert 2.0 <= GAP_STRESS_MULT["crypto"] <= 3.0
