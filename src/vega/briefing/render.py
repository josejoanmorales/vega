"""Markdown rendering — same BriefingData in, byte-identical markdown out."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vega.briefing.calls import RenderedCall, RenderedRejection
from vega.briefing.engine import BriefingData
from vega.common.paths import DATA_ROOT
from vega.data.types import SnapshotConflictError
from vega.execution.exits import ExitDecision
from vega.lifecycle.live_trades import DemotionOutcome

BRIEFINGS_DIR = DATA_ROOT / "briefings"
TOP_N = 5


def _movers_table(movers: pd.DataFrame) -> str:
    if movers.empty:
        return "_no data for the last two sessions_\n"
    picks = pd.concat([movers.head(TOP_N), movers.tail(TOP_N)]).drop_duplicates("symbol")
    lines = ["| symbol | close | Δ% |", "|---|---|---|"]
    lines += [f"| {r.symbol} | {r.close:,.2f} | {r.pct:+.2f}% |" for r in picks.itertuples()]
    return "\n".join(lines) + "\n"


def _calls_table(calls: tuple[RenderedCall, ...]) -> str:
    lines = [
        "| rank | symbol | family:version | thesis | qty | entry | stop | worst-case | "
        "time stop | profit rule | invalidation | heat (total) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in calls:
        lines.append(
            f"| {c.rank} | {c.symbol} | {c.family}:{c.version} | {c.thesis} | "
            f"{c.qty:.6f} | {c.entry_ref_price:,.2f} | {c.stop_price:,.2f} | "
            f"{c.worst_case_r_multiple:.2f}R | {c.time_stop_sessions} sessions "
            f"({c.time_stop_date}) | {c.profit_rule} | {c.invalidation} | "
            f"{c.heat_after_r.get('total', 0.0):.2f}R |"
        )
    return "\n".join(lines) + "\n"


def _rejections_table(rejections: tuple[RenderedRejection, ...]) -> str:
    lines = ["| symbol | family | reason | detail |", "|---|---|---|---|"]
    lines += [f"| {r.symbol} | {r.family} | {r.reason} | {r.detail} |" for r in rejections]
    return "\n".join(lines) + "\n"


def _exits_table(exits: tuple[ExitDecision, ...]) -> str:
    lines = ["| symbol | reason | qty | detail |", "|---|---|---|---|"]
    lines += [f"| {e.symbol} | {e.reason} | {e.qty:.6f} | {e.detail} |" for e in exits]
    return "\n".join(lines) + "\n"


def _signal_health_table(outcomes: tuple[DemotionOutcome, ...]) -> str:
    lines = [
        "| family | sleeve | n live trades | live Sharpe | band | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        v = o.verdict
        sharpe = f"{v.live_sharpe:.2f}" if v.live_sharpe is not None else "n/a"
        band = f"[{v.band[0]:.2f}, {v.band[1]:.2f}]" if v.band is not None else "n/a"
        verdict = "DEMOTED" if v.should_demote else v.reason
        lines.append(
            f"| {o.family} | {o.asset_class or '—'} | {v.n_trades} | {sharpe} | {band} | "
            f"{verdict} |"
        )
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
    if data.exits:
        parts += ["", "## Exits", "", _exits_table(data.exits)]
    if data.calls_error is not None:
        parts += [
            "",
            "## Ranked calls",
            "",
            f"⚠ **Ranked calls unavailable this run** — {data.calls_error}",
        ]
    elif data.eligible_families:
        parts += ["", "## Ranked calls", ""]
        if data.calls:
            parts.append(_calls_table(data.calls))
        else:
            parts += [f"**No trade today** — {data.no_trade_reason}", ""]
        if data.rejections:
            parts += ["### Considered and rejected", "", _rejections_table(data.rejections)]
        parts += ["_Eligible signal families:_"]
        parts += [
            f"- `{f.family}` ({f.state}) — justifying run `{f.justifying_run_id}`, "
            f"params {f.justifying_params}"
            for f in data.eligible_families
        ]
    if data.signal_health:
        parts += ["", "## Signal health", "", _signal_health_table(data.signal_health)]
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
