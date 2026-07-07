from vega.backtest.folds import split_dev_holdout, walk_forward_folds


def _dates(n: int) -> list[str]:
    return [f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)]


def test_split_is_a_locked_proportion_dev_before_holdout() -> None:
    dates = _dates(100)
    dev, holdout = split_dev_holdout(dates, holdout_frac=0.2)
    assert len(dev) == 80 and len(holdout) == 20
    assert dev[-1] < holdout[0]  # dev strictly precedes holdout, no overlap


def test_split_is_deterministic() -> None:
    dates = _dates(57)
    assert split_dev_holdout(dates) == split_dev_holdout(list(reversed(dates)))


def test_walk_forward_folds_are_expanding_and_non_overlapping() -> None:
    dev = _dates(200)
    folds = walk_forward_folds(dev, test_size=63)
    assert len(folds) == 3  # 200 // 63 = 3 full folds, remainder dropped
    assert folds[0].train_dates == ()
    assert len(folds[1].train_dates) == 63
    assert folds[1].test_dates[0] > folds[0].test_dates[-1]
    assert folds[2].test_dates[0] > folds[1].test_dates[-1]


def test_too_few_sessions_yields_no_folds() -> None:
    assert walk_forward_folds(_dates(10), test_size=63) == []
