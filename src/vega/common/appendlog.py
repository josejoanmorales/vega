"""Append-only JSONL log, fsync on every write (Caral audit-log pattern).

Shared by the ledger (WI-060) and the backtest registry (WI-063): both need
"write once, never rewrite, always durable" and nothing more. No update or
delete is provided by design — a caller that needs correction semantics
appends a new record referencing the old one (e.g. `supersedes`).
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple


class _CacheEntry(NamedTuple):
    mtime_ns: int
    size: int
    records: list[dict[str, Any]]


# Keyed by resolved path, not per-instance: call sites routinely construct a
# fresh Store/Registry (and therefore a fresh AppendLog) per call, so an
# instance-level cache would never hit. (mtime_ns, size)-checked so a write
# from ANY process invalidates it automatically — no explicit invalidation
# hook needed on append().
_CACHE: dict[Path, _CacheEntry] = {}


class AppendLog:
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: dict[str, Any]) -> None:
        # No default=str: a non-JSON-native value (numpy scalar, Timestamp, Decimal)
        # must raise HERE, at the write site, not be silently stringified into an
        # append-only audit log and break a reader far from the bug.
        line = json.dumps(record, sort_keys=True)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def records(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        resolved = self._path.resolve()
        stat = self._path.stat()
        cached = _CACHE.get(resolved)
        if (
            cached is not None
            and cached.mtime_ns == stat.st_mtime_ns
            and cached.size == stat.st_size
        ):
            return list(cached.records)
        with self._path.open() as fh:
            records = [json.loads(line) for line in fh]
        _CACHE[resolved] = _CacheEntry(stat.st_mtime_ns, stat.st_size, records)
        return list(records)

    def records_of_type(self, kind: str) -> list[dict[str, Any]]:
        # r["type"], not r.get("type"): a record missing its type is corruption and
        # must raise, not silently vanish from an audit log that then looks clean.
        return [r for r in self.records() if r["type"] == kind]

    @contextmanager
    def exclusive_lock(self) -> Iterator[None]:
        """Cross-process exclusive lock for read-validate-append sequences.

        A bare append is atomic enough on its own; this exists for callers whose
        VALIDATION depends on the current log contents (e.g. a state machine's
        legal-transition check) — without it, two writers can both validate
        against the same stale state and append conflicting records (review
        finding: a race could effectively un-retire a terminal lifecycle state).
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        with lock_path.open("w") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
