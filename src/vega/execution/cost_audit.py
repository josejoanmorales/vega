"""Read-only cost-model validation against real fills (WI-135).

Every promotion verdict rests on ONE doctrine, stated in `backtest/costs.py`'s
own module docstring: "a backtest must be the pessimistic estimate, never the
optimistic one." Today that is enforced only between two sets of CONSTANTS —
`backtest/costs.py`'s tier bps vs `execution/pnl.py`'s `SLIPPAGE_BPS` — via a
module-level assert. `SLIPPAGE_BPS` (10bps equity, 30bps crypto) was a STATED
ASSUMPTION from WI-061 and has never once been measured against a real fill.
If real slippage exceeds it, the doctrine is inverted and every `pass`
verdict the lifecycle registry has ever issued — including
`oversold_reversion_v1`'s paper-live promotion — is built on an optimistic
backtest. This module measures the gap. It NEVER writes to the ledger, any
registry, or any cost constant: it reports, a human decides, and re-tuning a
constant would invalidate every prior backtest run against it.

TWO DISTINCT QUANTITIES, deliberately never pooled (conflating them would
blame the broker for the market's overnight move):

- IMPLEMENTATION SHORTFALL (entries only): fill price vs `entry_ref_price`,
  the prior session's close the call was computed from. This INCLUDES the
  gap between decision and execution and is NOT execution cost — it answers
  "what did the decision cost", and it is computable today from data already
  in the ledger. There is no symmetric reference for an exit (a time-stop
  exit has no target price), so this metric is entry-only by construction,
  not by omission.

- EXECUTION SLIPPAGE (entries + exits): fill price vs the OPEN of the session
  the order actually executed in — the same T+1-open fill model
  `backtest/simulate.py` uses, so it is the like-for-like comparison against
  `backtest/costs.py`'s tier costs. This is the number that validates or
  breaks the pessimism doctrine.

  It requires knowing which session a fill actually executed in, which the
  ledger could not answer until this item added Alpaca's own `filled_at` to
  fill records (`ledger/store.py`, `execution/executor.py`,
  `execution/exits.py`). The wall-clock `at` timestamp records when
  reconciliation OBSERVED the fill, not when it happened — verified against
  production data to lag the true fill by a run cycle in multiple cases,
  worse across the 2026-07-27/2026-08-05 outages. Every fill from before this
  addition has no `filled_at` and is EXCLUDED from execution slippage, never
  guessed from `at`.

HONESTY CONTRACT: every statistic carries its sample size. Nothing here
triggers automatic action at any n. At small n (today: n=4 implementation
shortfall, n=0 execution slippage — the historical fills predate `filled_at`)
the report says so plainly rather than implying a verdict the data can't
support.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from vega.backtest.costs import cost_bps
from vega.common import db
from vega.data import snapshot
from vega.data.universe import load_universe
from vega.ledger.store import LedgerStore

# Mirrors backtest/simulate.py's MEDIAN_VOLUME_WINDOW — the tier lookup must
# use the same window the backtest itself used to classify liquidity, or the
# doctrine comparison would be checking two different definitions of "tier".
MEDIAN_VOLUME_WINDOW = 60

# Below this many fills, a mean/median is directional noise, not evidence —
# stated in every caveat rather than silently trusted.
MIN_SAMPLE_FOR_A_VERDICT = 30


@dataclass(frozen=True)
class FillCost:
    symbol: str
    asset_class: str
    side: str  # "buy" | "sell"
    reason: str | None  # exit trigger (time_stop, stop, ...) — None for entries
    ref_id: str
    order_id: str
    fill_price: float
    entry_ref_price: float | None  # entries only
    fill_session: str | None  # None when filled_at is unavailable (pre-WI-135 fill)
    session_open: float | None
    # Signed bps: POSITIVE = unfavorable (cost), NEGATIVE = favorable — a buy
    # filling above reference or a sell filling below reference is a cost,
    # matching backtest/costs.py's own apply_cost asymmetry.
    implementation_shortfall_bps: float | None
    execution_slippage_bps: float | None


@dataclass(frozen=True)
class SlippageStat:
    n: int
    mean_bps: float | None
    median_bps: float | None


_EMPTY_STAT = SlippageStat(n=0, mean_bps=None, median_bps=None)


@dataclass(frozen=True)
class DoctrineFinding:
    """Realized execution slippage vs what the backtest would have charged
    the same symbols. `doctrine_holds` is None, not a guess, when there is no
    priced-and-sessioned fill to compare — a verdict must never be implied
    from zero evidence."""

    n: int
    mean_execution_slippage_bps: float | None
    mean_backtest_tier_bps: float | None
    doctrine_holds: bool | None
    detail: str


@dataclass(frozen=True)
class CostAuditReport:
    fills: tuple[FillCost, ...]
    implementation_shortfall: SlippageStat  # entries only
    execution_slippage_entries: SlippageStat
    execution_slippage_exits: SlippageStat
    doctrine: DoctrineFinding
    caveats: tuple[str, ...]


def _stat(values: list[float]) -> SlippageStat:
    if not values:
        return _EMPTY_STAT
    return SlippageStat(n=len(values), mean_bps=mean(values), median_bps=median(values))


def _signed_bps(reference: float, fill: float, side: str) -> float:
    if side == "buy":
        return (fill - reference) / reference * 10_000.0
    if side == "sell":
        return (reference - fill) / reference * 10_000.0
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def _session_open(symbol: str, session: str, root: Path) -> float | None:
    with db.connect(root) as con:
        row = con.execute(
            "SELECT open FROM bars WHERE symbol = ? AND date = ? AND source = 'yfinance'",
            [symbol, session],
        ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _median_dollar_volume(symbol: str, as_of: str, root: Path) -> float:
    """Mirrors backtest/simulate.py's `_median_dollar_volume` (trailing
    window ending at `as_of`, close*volume) so the doctrine comparison tiers
    a symbol exactly as the backtest that justified it would have."""
    with db.connect(root) as con:
        rows = con.execute(
            "SELECT close, volume FROM bars WHERE symbol = ? AND date <= ? "
            "AND source = 'yfinance' ORDER BY date DESC LIMIT ?",
            [symbol, as_of, MEDIAN_VOLUME_WINDOW],
        ).fetchall()
    if not rows:
        return 0.0
    return float(median(c * v for c, v in rows))


def _doctrine_check(fills: list[FillCost], root: Path) -> DoctrineFinding:
    priced = [f for f in fills if f.execution_slippage_bps is not None]
    if not priced:
        return DoctrineFinding(
            n=0,
            mean_execution_slippage_bps=None,
            mean_backtest_tier_bps=None,
            doctrine_holds=None,
            detail=(
                "no fill has a known execution session yet — filled_at was added by this "
                "item, so historical fills cannot be compared; the next real fills will "
                "populate this"
            ),
        )
    tier_costs = []
    for f in priced:
        if f.asset_class == "crypto":
            tier_costs.append(cost_bps("crypto", f.symbol))
        else:
            assert f.fill_session is not None  # noqa: S101 — priced implies a known session
            mdv = _median_dollar_volume(f.symbol, f.fill_session, root)
            tier_costs.append(cost_bps(f.asset_class, f.symbol, mdv))
    slippages = [f.execution_slippage_bps for f in priced if f.execution_slippage_bps is not None]
    mean_slip = mean(slippages)
    mean_tier = mean(tier_costs)
    holds = mean_slip <= mean_tier
    small_n = f" (n={len(priced)}, well below {MIN_SAMPLE_FOR_A_VERDICT} — directional only)"
    detail = (
        f"mean realized execution slippage {mean_slip:.2f}bps vs mean backtest tier cost "
        f"{mean_tier:.2f}bps over n={len(priced)} fill(s) — "
        + ("doctrine HOLDS" if holds else "DOCTRINE VIOLATED: backtest is cheaper than live")
        + (small_n if len(priced) < MIN_SAMPLE_FOR_A_VERDICT else "")
    )
    return DoctrineFinding(
        n=len(priced),
        mean_execution_slippage_bps=mean_slip,
        mean_backtest_tier_bps=mean_tier,
        doctrine_holds=holds,
        detail=detail,
    )


def audit(ledger: LedgerStore | None = None, root: Path = snapshot.DATA_ROOT) -> CostAuditReport:
    """Read-only: touches the ledger and store only via read methods
    (`latest_with_all_fills`, `db.connect(..., read_only=True)` by default).
    Never appends a fill, override, or registry record; never changes a cost
    constant."""
    ledger = ledger or LedgerStore()
    asset_class_by_symbol = {e.symbol: e.asset_class for e in load_universe()}

    fills: list[FillCost] = []
    for rec, order_fills in ledger.latest_with_all_fills():
        for f in order_fills:
            if f.get("price") is None:
                continue  # unresolved acceptance or terminal cancel — no trade occurred
            side = f.get("side") or "buy"
            fill_price = float(f["price"])

            entry_ref = float(rec["entry_ref_price"]) if side == "buy" else None
            impl_shortfall = (
                _signed_bps(entry_ref, fill_price, side) if entry_ref is not None else None
            )

            filled_at = f.get("filled_at")
            fill_session = filled_at[:10] if filled_at else None
            session_open = (
                _session_open(rec["symbol"], fill_session, root) if fill_session else None
            )
            exec_slippage = (
                _signed_bps(session_open, fill_price, side) if session_open is not None else None
            )

            fills.append(
                FillCost(
                    symbol=rec["symbol"],
                    asset_class=asset_class_by_symbol.get(
                        rec["symbol"], rec.get("asset_class", "equity")
                    ),
                    side=side,
                    reason=f.get("reason"),
                    ref_id=rec["id"],
                    order_id=f["order_id"],
                    fill_price=fill_price,
                    entry_ref_price=entry_ref,
                    fill_session=fill_session,
                    session_open=session_open,
                    implementation_shortfall_bps=impl_shortfall,
                    execution_slippage_bps=exec_slippage,
                )
            )

    impl_stat = _stat(
        [
            f.implementation_shortfall_bps
            for f in fills
            if f.implementation_shortfall_bps is not None
        ]
    )
    exec_entries = _stat(
        [
            f.execution_slippage_bps
            for f in fills
            if f.side == "buy" and f.execution_slippage_bps is not None
        ]
    )
    exec_exits = _stat(
        [
            f.execution_slippage_bps
            for f in fills
            if f.side == "sell" and f.execution_slippage_bps is not None
        ]
    )
    doctrine = _doctrine_check(fills, root)

    caveats: list[str] = []
    n_no_session = sum(1 for f in fills if f.fill_session is None)
    if n_no_session:
        caveats.append(
            f"{n_no_session}/{len(fills)} fill(s) have no recorded execution session "
            "(filled_at was added by this item; earlier fills are excluded from execution "
            "slippage — never estimated from the wall-clock reconciliation timestamp)"
        )
    if 0 < impl_stat.n < MIN_SAMPLE_FOR_A_VERDICT:
        caveats.append(
            f"implementation shortfall n={impl_stat.n}, below {MIN_SAMPLE_FOR_A_VERDICT} — "
            "directional only, not a statistically meaningful sample"
        )
    if not fills:
        caveats.append("no priced fills in the ledger yet")

    return CostAuditReport(
        fills=tuple(fills),
        implementation_shortfall=impl_stat,
        execution_slippage_entries=exec_entries,
        execution_slippage_exits=exec_exits,
        doctrine=doctrine,
        caveats=tuple(caveats),
    )


def render_text(report: CostAuditReport) -> str:
    lines = ["Vega cost-model audit (read-only — reports only, never retunes a constant)", ""]

    def _fmt(stat: SlippageStat, label: str) -> str:
        if stat.n == 0:
            return f"{label}: n=0 (no data yet)"
        return (
            f"{label}: n={stat.n}  mean={stat.mean_bps:+.2f}bps  median={stat.median_bps:+.2f}bps"
        )

    lines.append(_fmt(report.implementation_shortfall, "Implementation shortfall (entries)"))
    lines.append(_fmt(report.execution_slippage_entries, "Execution slippage (entries)"))
    lines.append(_fmt(report.execution_slippage_exits, "Execution slippage (exits)"))
    lines.append("")
    lines.append(f"Pessimism doctrine: {report.doctrine.detail}")
    if report.caveats:
        lines.append("")
        lines.append("Caveats:")
        lines.extend(f"  - {c}" for c in report.caveats)
    return "\n".join(lines)


def main() -> None:
    print(render_text(audit()))


if __name__ == "__main__":
    main()
