"""Trade simulation: the one place every trade passes through a fill and a cost.

Timing (PIT doctrine, STRATEGY.md §6 Pillar 3): a signal decides at the CLOSE
of session T using a MarketView cut off at T; every fill — entry, stop, time
stop, or profit-take — executes at the OPEN of the next available session.
Same-bar-close fills do not exist in this module.

Price spaces: entries/exits are simulated in raw (unadjusted) price terms —
the space fills actually happen in — while signals decide using adj_close
(so split/dividend history doesn't create phantom trend signals). ATR / stop
distance is computed from raw OHLC so it matches the space the stop is
compared against. This is a deliberate refinement of the enriched spec:
adj_close drives WHEN to enter, raw prices drive WHERE the risk sits.

Every exit checks the stop before profit-taking (conservative bias, matches
the doctrine that a backtest must be the pessimistic estimate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from vega.backtest.costs import apply_cost, cost_bps
from vega.backtest.market_view import MarketView
from vega.backtest.signals import EntryProposal, Signal

ATR_PERIOD = 14
MEDIAN_VOLUME_WINDOW = 60


@dataclass
class _OpenPosition:
    symbol: str
    asset_class: str
    signal_family: str
    signal_version: str
    entry_date: str
    entry_price: float
    initial_qty: float
    remaining_qty: float
    stop_price: float
    initial_r: float  # entry_price - stop_price, in raw-price terms, > 0
    time_stop_days: int
    profit_take_half_at_r: float
    profit_trail_atr_mult: float
    atr_at_entry: float
    invalidation: str
    thesis: str
    confidence: float
    sessions_held: int = 0
    took_partial: bool = False
    high_water_close: float = 0.0
    exits: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TradeRecord:
    symbol: str
    asset_class: str
    signal_family: str
    signal_version: str
    entry_date: str
    entry_price: float
    initial_qty: float
    stop_price: float
    initial_r: float
    thesis: str
    confidence: float
    invalidation: str
    exits: tuple[dict[str, Any], ...]
    realized_pnl: float
    r_multiple: float
    unresolved_at_end: bool


def _true_range(prev_close: float, high: float, low: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def compute_atr(
    frame: pd.DataFrame, symbol: str, as_of: str, period: int = ATR_PERIOD
) -> float | None:
    """Raw-OHLC ATR through `as_of` inclusive. None if there isn't enough history."""
    bars = (
        frame[(frame["symbol"] == symbol) & (frame["date"] <= as_of)]
        .sort_values("date")
        .tail(period + 1)
    )
    if len(bars) < period + 1:
        return None
    trs = [
        _true_range(prev, high, low)
        for prev, high, low in zip(
            bars["close"].iloc[:-1], bars["high"].iloc[1:], bars["low"].iloc[1:], strict=True
        )
    ]
    return sum(trs) / len(trs)


def _median_dollar_volume(frame: pd.DataFrame, symbol: str, as_of: str) -> float:
    bars = (
        frame[(frame["symbol"] == symbol) & (frame["date"] <= as_of)]
        .sort_values("date")
        .tail(MEDIAN_VOLUME_WINDOW)
    )
    if bars.empty:
        return 0.0
    return float((bars["close"] * bars["volume"]).median())


def simulate_signal(
    frame: pd.DataFrame,
    dates: list[str],
    signal: Signal,
    universe: list[str],
    asset_class: str,
    notional_usd: float = 1_000.0,
) -> list[TradeRecord]:
    """Walk `dates` in order, simulating one signal against one asset class.

    `frame` must contain all sessions needed for lookback (including sessions
    before dates[0]) up through dates[-1] for `universe`'s symbols.
    """
    by_date_symbol: dict[str, dict[str, Any]] = {}
    for date, group in frame[frame["symbol"].isin(universe)].groupby("date"):
        by_date_symbol[str(date)] = {str(row.symbol): row for row in group.itertuples()}

    open_positions: dict[str, _OpenPosition] = {}
    pending_exits: dict[str, _OpenPosition] = {}
    pending_entries: dict[str, EntryProposal] = {}
    completed: list[TradeRecord] = []

    def _bps(symbol: str, as_of: str) -> float:
        mdv = _median_dollar_volume(frame, symbol, as_of) if asset_class != "crypto" else None
        return cost_bps(asset_class, symbol, mdv)

    def _finalize(pos: _OpenPosition, unresolved: bool) -> None:
        realized = (
            sum((e["price"] - pos.entry_price) * e["qty"] for e in pos.exits)
            if pos.initial_qty
            else 0.0
        )
        completed.append(
            TradeRecord(
                symbol=pos.symbol,
                asset_class=pos.asset_class,
                signal_family=pos.signal_family,
                signal_version=pos.signal_version,
                entry_date=pos.entry_date,
                entry_price=pos.entry_price,
                initial_qty=pos.initial_qty,
                stop_price=pos.stop_price,
                initial_r=pos.initial_r,
                thesis=pos.thesis,
                confidence=pos.confidence,
                invalidation=pos.invalidation,
                exits=tuple(pos.exits),
                realized_pnl=round(realized, 6),
                r_multiple=round(realized / (pos.initial_r * pos.initial_qty), 4)
                if pos.initial_r * pos.initial_qty
                else 0.0,
                unresolved_at_end=unresolved,
            )
        )

    for i, date in enumerate(dates):
        today = by_date_symbol.get(date, {})
        prior_date = dates[i - 1] if i > 0 else None
        # Cost tiering uses the PRIOR session's dollar-volume window everywhere —
        # today's own close*volume is unknowable at the open being filled.
        tier_asof = prior_date if prior_date is not None else date

        # 0. fill queued time-stop exits at today's open (retry next day if no bar)
        for symbol, pos in list(pending_exits.items()):
            row = today.get(symbol)
            if row is None:
                continue
            price = apply_cost(float(row.open), "sell", _bps(symbol, tier_asof))
            pos.exits.append(
                {"date": date, "qty": pos.remaining_qty, "price": price, "reason": "time_stop"}
            )
            _finalize(pos, unresolved=False)
            del pending_exits[symbol]

        # 1. fill queued entries at today's open (retry next day if no bar)
        for symbol, proposal in list(pending_entries.items()):
            row = today.get(symbol)
            if row is None or prior_date is None:
                continue
            atr = compute_atr(frame, symbol, as_of=prior_date)
            if atr is None or atr <= 0:
                del pending_entries[symbol]
                continue
            fill_price = apply_cost(float(row.open), "buy", _bps(symbol, tier_asof))
            stop_price = fill_price - proposal.stop_atr_mult * atr
            if stop_price <= 0:
                del pending_entries[symbol]
                continue
            qty = notional_usd / fill_price
            open_positions[symbol] = _OpenPosition(
                symbol=symbol,
                asset_class=asset_class,
                signal_family=proposal.signal_family,
                signal_version=proposal.signal_version,
                entry_date=date,
                entry_price=fill_price,
                initial_qty=qty,
                remaining_qty=qty,
                stop_price=stop_price,
                initial_r=fill_price - stop_price,
                time_stop_days=proposal.time_stop_days,
                profit_take_half_at_r=proposal.profit_take_half_at_r,
                profit_trail_atr_mult=proposal.profit_trail_atr_mult,
                atr_at_entry=atr,
                invalidation=proposal.invalidation,
                thesis=proposal.thesis,
                confidence=proposal.confidence,
                high_water_close=float(row.close),
            )
            del pending_entries[symbol]

        # 2. manage open positions using today's bar: stop (gap-aware) then profit-take/trail
        for symbol, pos in list(open_positions.items()):
            row = today.get(symbol)
            if row is None:
                continue
            bps = _bps(symbol, tier_asof)
            if float(row.open) <= pos.stop_price:
                price = apply_cost(float(row.open), "sell", bps)
                pos.exits.append(
                    {"date": date, "qty": pos.remaining_qty, "price": price, "reason": "gap_stop"}
                )
                _finalize(pos, unresolved=False)
                del open_positions[symbol]
                continue
            if float(row.low) <= pos.stop_price:
                price = apply_cost(pos.stop_price, "sell", bps)
                pos.exits.append(
                    {"date": date, "qty": pos.remaining_qty, "price": price, "reason": "stop"}
                )
                _finalize(pos, unresolved=False)
                del open_positions[symbol]
                continue

            # No profit-taking on the entry bar itself: a same-bar profit fill is the
            # optimistic sibling of the same-bar fills the doctrine forbids (same-day
            # STOP-outs above stay — those are the pessimistic direction).
            if pos.entry_date == date:
                continue

            if not pos.took_partial:
                target = pos.entry_price + pos.profit_take_half_at_r * pos.initial_r
                if float(row.high) >= target:
                    half_qty = pos.remaining_qty / 2
                    price = apply_cost(target, "sell", bps)
                    pos.exits.append(
                        {"date": date, "qty": half_qty, "price": price, "reason": "profit_partial"}
                    )
                    pos.remaining_qty -= half_qty
                    pos.took_partial = True
                    pos.high_water_close = float(row.close)

            if pos.took_partial:
                pos.high_water_close = max(pos.high_water_close, float(row.close))
                trail = pos.high_water_close - pos.profit_trail_atr_mult * pos.atr_at_entry
                pos.stop_price = max(pos.stop_price, trail)  # a trail only ever tightens

        # 3. time-stop trigger for positions still open after today's management.
        # The entry bar does not count as a held session — time_stop_days=N means
        # N full sessions AFTER entry.
        for symbol, pos in list(open_positions.items()):
            if pos.entry_date == date:
                continue
            pos.sessions_held += 1
            if pos.sessions_held >= pos.time_stop_days:
                pending_exits[symbol] = pos
                del open_positions[symbol]

        # 4. new entries decided at close of `date`, queued for the next session's open
        if i < len(dates) - 1:
            view = MarketView(frame, as_of=date)
            proposals = signal.scan(view, universe)
            pending_entries = {
                p.symbol: p
                for p in proposals
                if p.symbol not in open_positions and p.symbol not in pending_exits
            }
        else:
            pending_entries = {}

    # end of window: force-close everything still open or queued. These are accounting
    # closes, not real trades — they pay full sell costs (no zero-cost path exists, even
    # here) and are finalized unresolved, which excludes them from the sample-size gate.
    last_date = dates[-1]
    last_bars = by_date_symbol.get(last_date, {})
    tier_asof = dates[-2] if len(dates) > 1 else last_date
    for pos in list(open_positions.values()) + list(pending_exits.values()):
        row = last_bars.get(pos.symbol)
        raw_price = float(row.close) if row is not None else pos.stop_price
        price = apply_cost(raw_price, "sell", _bps(pos.symbol, tier_asof))
        pos.exits.append(
            {"date": last_date, "qty": pos.remaining_qty, "price": price, "reason": "end_of_data"}
        )
        _finalize(pos, unresolved=True)

    return completed
