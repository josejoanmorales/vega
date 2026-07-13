"""Economic rationale registry — the rationale-first gate (STRATEGY.md §6).

A signal cannot enter testing without a written rationale recorded first.
Corrections are new records, never edits — same append-only doctrine as the
ledger and the backtest registry.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from vega.common.appendlog import AppendLog

DEFAULT_PATH = Path("data/lifecycle/rationale.jsonl")


class RationaleSource(Protocol):
    """What the backtest engine's rationale-first gate consumes."""

    def has_rationale(self, signal_family: str) -> bool: ...


class NullRationaleRegistry:
    """EXPLICIT opt-out of the rationale-first gate — for test fixtures only.

    The gate parameter is mandatory everywhere (a review found the previous
    silent `None` default meant the codebase's own blessed example ran
    ungated — the 'safety that only fires when the caller opts in' pattern).
    Opting out must be visible at the call site, never a default.
    """

    def has_rationale(self, signal_family: str) -> bool:  # noqa: ARG002 — uniform interface
        return True


@dataclass(frozen=True)
class RationaleRecord:
    id: str
    at: str
    signal_family: str
    text: str
    author: str


class RationaleRegistry:
    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self._log = AppendLog(path)

    def record(self, signal_family: str, text: str, author: str) -> RationaleRecord:
        if not text.strip():
            raise ValueError("a rationale must have non-empty text — never assume silently")
        record = RationaleRecord(
            id=str(uuid.uuid4()),
            at=datetime.now(UTC).isoformat(),
            signal_family=signal_family,
            text=text,
            author=author,
        )
        self._log.append({"type": "rationale", **record.__dict__})
        return record

    def has_rationale(self, signal_family: str) -> bool:
        return any(r["signal_family"] == signal_family for r in self._log.records())

    def history(self, signal_family: str) -> list[dict[str, str]]:
        return [r for r in self._log.records() if r["signal_family"] == signal_family]
