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
        self,
        ref_id: str,
        order_id: str,
        qty: float,
        price: float | None,
        status: str,
        side: str = "buy",
        reason: str | None = None,
        session: str | None = None,
    ) -> str:
        """Paper-fill record linked to a recommendation (added by WI-061;
        append-only). `side`/`reason`/`session` are additive (WI-087): a sell
        fill closing (or partially closing) a position carries `side="sell"`,
        the trigger `reason` (mirrors `backtest/simulate.py`'s exit reasons —
        gap_stop/stop/profit_partial/time_stop), and `session` — the store
        session the exit was DECIDED against (never the wall-clock fill time),
        the one piece of state the trail's high-water-close window needs.
        Existing buy-fill callers are unaffected (defaults preserve old shape).
        """
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
                "side": side,
                "reason": reason,
                "session": session,
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

    def _chain_roots(self) -> dict[str, str]:
        """rec id -> the root id of its supersede chain (shared by every
        chain-aware fill join below)."""
        parent = {r["id"]: r.get("supersedes") for r in self.entries()}

        def _root(rec_id: str) -> str:
            seen = set()
            while parent.get(rec_id) and rec_id not in seen:
                seen.add(rec_id)
                rec_id = parent[rec_id]  # type: ignore[assignment]
            return rec_id

        return {rid: _root(rid) for rid in parent}

    def latest_with_fills(self) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
        """`latest()` recommendations paired with their ENTRY (buy) fill,
        resolved through supersede chains: a fill on ANY id in a rec's chain
        belongs to the surviving rec (WI-067 review — fills key on the id that
        was pending at execution time, so a filled-then-corrected position
        must neither vanish from heat accounting nor be re-executed as
        'unfilled'). When several buy-fill records exist for one chain (e.g. a
        later reconciliation adds the real price), the one carrying a price
        wins over price-less acceptances. A position that has since (partly or
        fully) SOLD still resolves to its buy fill here — this method answers
        "was it entered", not "is it still open"; use `latest_with_all_fills`
        for exit-aware reconstruction."""
        roots = self._chain_roots()
        fills_by_root: dict[str, dict[str, Any]] = {}
        for fill in self.fills():  # chronological — later records are newer knowledge
            if fill.get("side", "buy") != "buy":
                continue
            root = roots.get(fill["ref_id"], fill["ref_id"])
            current = fills_by_root.get(root)
            if current is not None and current.get("price") is not None:
                continue  # a priced (real) fill is final — never displaced
            fills_by_root[root] = fill
        return [(rec, fills_by_root.get(roots.get(rec["id"], rec["id"]))) for rec in self.latest()]

    def latest_with_all_fills(self) -> list[tuple[dict[str, Any], tuple[dict[str, Any], ...]]]:
        """`latest()` recommendations paired with EVERY fill (buy and sell)
        anywhere in their supersede chain, chronological — the one join exit
        reconstruction (WI-087) and entry reconstruction (WI-067) both build
        on, so there is exactly one definition of "the fills belonging to this
        position" instead of two independently-evolving joins."""
        roots = self._chain_roots()
        fills_by_root: dict[str, list[dict[str, Any]]] = {}
        for fill in self.fills():
            root = roots.get(fill["ref_id"], fill["ref_id"])
            fills_by_root.setdefault(root, []).append(fill)
        return [
            (rec, tuple(fills_by_root.get(roots.get(rec["id"], rec["id"]), [])))
            for rec in self.latest()
        ]
