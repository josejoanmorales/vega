import pandas as pd

from conftest import flat_history_n
from vega.common.atr import compute_atr


def test_none_below_period_plus_one_bars() -> None:
    frame = pd.DataFrame(flat_history_n("AAA", 10))
    assert compute_atr(frame, "AAA", as_of="2026-05-10") is None


def test_exact_on_constant_true_range() -> None:
    frame = pd.DataFrame(flat_history_n("AAA", 20))
    # TR = max(101-99, |101-100|, |99-100|) = 2 every day
    assert compute_atr(frame, "AAA", as_of="2026-05-20") == 2.0


def test_shared_by_backtest_and_risk_modules() -> None:
    from vega.backtest.simulate import compute_atr as backtest_atr
    from vega.common.atr import compute_atr as common_atr

    assert backtest_atr is common_atr  # one implementation, not two copies to drift
