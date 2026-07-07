"""Append-only JSONL log, fsync on every write (Caral audit-log pattern).

Shared by the ledger (WI-060) and the backtest registry (WI-063): both need
"write once, never rewrite, always durable" and nothing more. No update or
delete is provided by design — a caller that needs correction semantics
appends a new record referencing the old one (e.g. `supersedes`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class AppendLog:
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def records(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        with self._path.open() as fh:
            return [json.loads(line) for line in fh]

    def records_of_type(self, kind: str) -> list[dict[str, Any]]:
        return [r for r in self.records() if r.get("type") == kind]
