"""Markdown rendering — same BriefingData in, byte-identical markdown out."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vega.briefing.engine import BriefingData
from vega.common.paths import DATA_ROOT
from vega.data.types import SnapshotConflictError

BRIEFINGS_DIR = DATA_ROOT / "briefings"
TOP_N = 5


def _movers_table(movers: pd.DataFrame) -> str:
    if movers.empty:
        return "_no data for the last two sessions_\n"
    picks = pd.concat([movers.head(TOP_N), movers.tail(TOP_N)]).drop_duplicates("symbol")
    lines = ["| symbol | close | Δ% |", "|---|---|---|"]
    lines += [f"| {r.symbol} | {r.close:,.2f} | {r.pct:+.2f}% |" for r in picks.itertuples()]
    return "\n".join(lines) + "\n"


def render(data: BriefingData) -> str:
    r = data.regime
    breadth = f"{r.breadth_pct}%" if r.breadth_pct is not None else "insufficient history"
    parts = [
        f"# Vega pre-market briefing — {data.as_of}",
        "",
        "## Regime",
        "",
        f"**Composite: {r.composite.upper()}** — trend {r.trend}, VIX {r.vix} ({r.vix_band}), "
        f"breadth {breadth}, crypto fear/greed {r.crypto_fg}.",
        "",
        "## Movers — equities & ETFs",
        "",
        _movers_table(data.movers_equity),
        "## Movers — crypto",
        "",
        _movers_table(data.movers_crypto),
        "## Event calendar (next 14 days)",
        "",
    ]
    if data.events:
        parts += [f"- **{e.date}** — {e.event}" for e in data.events]
    else:
        parts.append("_no scheduled macro events_")
    if data.failures:
        parts += ["", "## ⚠ Execution failures (unresolved)", ""]
        parts += [
            f"- {f['at']} `{f['symbol']}` (rec {f['ref_id'][:8]}): {f['error']}"
            for f in data.failures
        ]
    parts += [
        "",
        "---",
        f"_All figures from the validated local store ({data.store_range[0]} → "
        f"{data.store_range[1]}); {data.quarantined_today} symbol-days quarantined on "
        f"{data.as_of}. No live or recalled figures._",
        "",
    ]
    return "\n".join(parts)


def write_briefing(data: BriefingData, root: Path = BRIEFINGS_DIR) -> Path:
    """Write-once per date: identical rewrite is a no-op, drifted rewrite raises."""
    path = root / f"{data.as_of}.md"
    content = render(data)
    if path.exists():
        if path.read_text() == content:
            return path
        raise SnapshotConflictError(f"{path} already exists with different content")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
