"""Locked dev/holdout split + expanding-window walk-forward folds.

Pure functions of the date list — deterministic, no randomness. The holdout
is meant to be evaluated exactly once per signal family; touch-counting
against that rule lives in registry.py, which is the only place that can see
a family's full run history.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_HOLDOUT_FRAC = 0.2
DEFAULT_TEST_SESSIONS = 63  # ~one quarter


@dataclass(frozen=True)
class Fold:
    train_dates: tuple[str, ...]
    test_dates: tuple[str, ...]


def split_dev_holdout(
    dates: list[str], holdout_frac: float = DEFAULT_HOLDOUT_FRAC
) -> tuple[list[str], list[str]]:
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError("holdout_frac must be within (0, 1)")
    ordered = sorted(dates)
    split_at = int(len(ordered) * (1 - holdout_frac))
    return ordered[:split_at], ordered[split_at:]


def walk_forward_folds(dev_dates: list[str], test_size: int = DEFAULT_TEST_SESSIONS) -> list[Fold]:
    """Expanding-window folds: fold k trains on everything before its test slice."""
    ordered = sorted(dev_dates)
    folds: list[Fold] = []
    start = 0
    while start + test_size <= len(ordered):
        test = ordered[start : start + test_size]
        train = ordered[:start]
        folds.append(Fold(train_dates=tuple(train), test_dates=tuple(test)))
        start += test_size
    return folds
