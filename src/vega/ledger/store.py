"""Append-only ledger store — JSONL, fsync on every write (Caral audit-log pattern).

There is deliberately NO update or delete API. Corrections append a new
recommendation with `supersedes` set; `latest()` resolves the chains.
"""

from __future__ import annotations

import dataclasses
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vega.ledger.types import OVERRIDE_ACTIONS, Recommendation

DEFAULT_PATH = Path("data/ledger/ledger.jsonl")


class LedgerStore:
    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self._path = path

    def _append_line(self, record: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def append(self, rec: Recommendation) -> str:
        record = {"type": "recommendation", **dataclasses.asdict(rec)}
        record["signal_attribution"] = list(rec.signal_attribution)
        self._append_line(record)
        return rec.id

    def append_override(self, ref_id: str, action: str, detail: str, actor: str) -> str:
        if action not in OVERRIDE_ACTIONS:
            raise ValueError(f"action must be one of {OVERRIDE_ACTIONS}")
        if ref_id not in {r["id"] for r in self.entries()}:
            raise ValueError(f"override references unknown recommendation {ref_id}")
        oid = str(uuid.uuid4())
        self._append_line(
            {
                "type": "override",
                "id": oid,
                "at": datetime.now(UTC).isoformat(),
                "ref_id": ref_id,
                "action": action,
                "detail": detail,
                "actor": actor,
            }
        )
        return oid

    def _records(self, kind: str) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self._path.open() as fh:
            for line in fh:
                record = json.loads(line)
                if record["type"] == kind:
                    out.append(record)
        return out

    def entries(self) -> list[dict[str, Any]]:
        return self._records("recommendation")

    def overrides(self) -> list[dict[str, Any]]:
        return self._records("override")

    def latest(self) -> list[dict[str, Any]]:
        """Recommendations with supersede chains resolved to their newest version."""
        entries = self.entries()
        superseded = {r["supersedes"] for r in entries if r.get("supersedes")}
        return [r for r in entries if r["id"] not in superseded]
