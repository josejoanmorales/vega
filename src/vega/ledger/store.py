"""Append-only ledger store, built on vega.common.appendlog.

There is deliberately NO update or delete API. Corrections append a new
recommendation with `supersedes` set; `latest()` resolves the chains.
"""

from __future__ import annotations

import dataclasses
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vega.common.appendlog import AppendLog
from vega.common.paths import DATA_ROOT
from vega.ledger.types import OVERRIDE_ACTIONS, Recommendation

DEFAULT_PATH = DATA_ROOT / "ledger/ledger.jsonl"


class LedgerStore:
    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self._log = AppendLog(path)

    def append(self, rec: Recommendation) -> str:
        record = {"type": "recommendation", **dataclasses.asdict(rec)}
        record["signal_attribution"] = list(rec.signal_attribution)
        self._log.append(record)
        return rec.id

    def append_override(self, ref_id: str, action: str, detail: str, actor: str) -> str:
        if action not in OVERRIDE_ACTIONS:
            raise ValueError(f"action must be one of {OVERRIDE_ACTIONS}")
        if ref_id not in {r["id"] for r in self.entries()}:
            raise ValueError(f"override references unknown recommendation {ref_id}")
        oid = str(uuid.uuid4())
        self._log.append(
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

    def append_fill(
        self, ref_id: str, order_id: str, qty: float, price: float | None, status: str
    ) -> str:
        """Paper-fill record linked to a recommendation (added by WI-061; append-only)."""
        if ref_id not in {r["id"] for r in self.entries()}:
            raise ValueError(f"fill references unknown recommendation {ref_id}")
        fid = str(uuid.uuid4())
        self._log.append(
            {
                "type": "fill",
                "id": fid,
                "at": datetime.now(UTC).isoformat(),
                "ref_id": ref_id,
                "order_id": order_id,
                "qty": qty,
                "price": price,
                "status": status,
            }
        )
        return fid

    def fills(self) -> list[dict[str, Any]]:
        return self._log.records_of_type("fill")

    def entries(self) -> list[dict[str, Any]]:
        return self._log.records_of_type("recommendation")

    def overrides(self) -> list[dict[str, Any]]:
        return self._log.records_of_type("override")

    def latest(self) -> list[dict[str, Any]]:
        """Recommendations with supersede chains resolved to their newest version."""
        entries = self.entries()
        superseded = {r["supersedes"] for r in entries if r.get("supersedes")}
        return [r for r in entries if r["id"] not in superseded]
