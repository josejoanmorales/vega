"""Backtest registry — append-only, anti-data-mining.

An unregistered backtest cannot promote (WI-065's lifecycle reads this
registry, never ad-hoc results). The promotion bar rises with the number of
hypotheses already tried for a family — a crude but auditable stand-in for
proper multiple-testing correction (deflated Sharpe etc.), to be refined only
if real trial volume justifies the complexity.

Rationale-first gate (WI-065): pass `rationale_registry` to `record_run` to
enforce "a signal cannot enter testing without a written economic rationale
recorded first" — testing IS calling record_run. The param is optional and
defaults to no gate, so it stays backward-compatible with callers (WI-063's
smoke test, existing tests) that don't supply one.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vega.common.appendlog import AppendLog
from vega.lifecycle.rationale import RationaleRegistry

DEFAULT_PATH = Path("data/backtests/registry.jsonl")
BASE_SHARPE_BAR = 0.8
LOG_SLOPE = 0.1


def _git_commit() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        out = subprocess.run(  # noqa: S603 — fixed args, git resolved via shutil.which above
            [git, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — best-effort provenance, never blocks a run
        return "unknown"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    at: str
    git_commit: str
    signal_family: str
    signal_version: str
    param_grid_size: int
    universe_version: str
    data_span: tuple[str, str]
    n_folds: int
    fold_metrics: tuple[dict[str, Any], ...]
    aggregate_metrics: dict[str, Any]
    verdict: str
    holdout_evaluated: bool
    holdout_touch_count_after: int
    promotion_bar: float | None
    notes: tuple[str, ...]


class BacktestRegistry:
    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self._log = AppendLog(path)

    def cumulative_grid_points(self, family: str, before_run_id: str | None = None) -> int:
        return sum(
            r["param_grid_size"]
            for r in self._log.records()
            if r["signal_family"] == family and r["run_id"] != before_run_id
        )

    def holdout_touch_count(self, family: str) -> int:
        return sum(
            1
            for r in self._log.records()
            if r["signal_family"] == family and r["holdout_evaluated"]
        )

    def promotion_bar(self, cumulative_grid_points: int) -> float:
        # log10 of anything < 1 is undefined/negative; a single trial is the floor.
        trials = max(cumulative_grid_points, 1)
        return round(BASE_SHARPE_BAR + LOG_SLOPE * math.log10(trials), 4)

    def record_run(
        self,
        signal_family: str,
        signal_version: str,
        param_grid_size: int,
        universe_version: str,
        data_span: tuple[str, str],
        n_folds: int,
        fold_metrics: list[dict[str, Any]],
        aggregate_metrics: dict[str, Any],
        verdict: str,
        holdout_evaluated: bool,
        promotion_bar: float | None,
        notes: list[str],
        rationale_registry: RationaleRegistry | None = None,
    ) -> RunRecord:
        if rationale_registry is not None and not rationale_registry.has_rationale(signal_family):
            raise ValueError(
                f"{signal_family} has no recorded economic rationale — cannot enter testing "
                "(record one via RationaleRegistry.record before running a backtest)"
            )
        prior_holdout_touches = self.holdout_touch_count(signal_family)
        record = RunRecord(
            run_id=str(uuid.uuid4()),
            at=datetime.now(UTC).isoformat(),
            git_commit=_git_commit(),
            signal_family=signal_family,
            signal_version=signal_version,
            param_grid_size=param_grid_size,
            universe_version=universe_version,
            data_span=data_span,
            n_folds=n_folds,
            fold_metrics=tuple(fold_metrics),
            aggregate_metrics=aggregate_metrics,
            verdict=verdict,
            holdout_evaluated=holdout_evaluated,
            holdout_touch_count_after=prior_holdout_touches + (1 if holdout_evaluated else 0),
            promotion_bar=promotion_bar,
            notes=tuple(notes),
        )
        payload = asdict(record)
        payload["type"] = "run"
        self._log.append(payload)
        return record

    def runs(self, family: str | None = None) -> list[dict[str, Any]]:
        records = self._log.records_of_type("run")
        return [r for r in records if family is None or r["signal_family"] == family]
