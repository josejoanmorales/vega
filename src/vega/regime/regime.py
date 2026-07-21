"""Pure regime computation — same stored inputs, same output, always."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from vega.data import snapshot
from vega.regime.inputs import fetch_fear_greed, fetch_vix

# VIX bands — dumb on purpose, constants not parameters
VIX_CALM = 15.0
VIX_NORMAL = 20.0
VIX_STRESSED = 28.0

BREADTH_WEAK_PCT = 40.0
FG_FEAR = 25
MIN_BREADTH_COVERAGE = 0.3  # fraction of universe needing 200 sessions before breadth counts


@dataclass(frozen=True)
class RegimeState:
    as_of: str
    trend: str  # risk_on | neutral | risk_off
    vix: float
    vix_band: str  # calm | normal | stressed | crisis
    breadth_pct: float | None  # None => insufficient_history
    crypto_fg: int
    composite: str  # risk_on | caution | risk_off


def _trend(spy: pd.DataFrame) -> str:
    closes = spy.sort_values("date")["adj_close"]
    if len(closes) < 220:
        return "neutral"  # not enough history to judge — never guess
    sma200 = closes.rolling(200).mean()
    above = float(closes.iloc[-1]) > float(sma200.iloc[-1])
    rising = float(sma200.iloc[-1]) > float(sma200.iloc[-21])
    if above and rising:
        return "risk_on"
    if not above and not rising:
        return "risk_off"
    return "neutral"


def _vix_band(vix: float) -> str:
    if vix < VIX_CALM:
        return "calm"
    if vix < VIX_NORMAL:
        return "normal"
    if vix < VIX_STRESSED:
        return "stressed"
    return "crisis"


def _breadth(universe_bars: pd.DataFrame) -> float | None:
    """% of universe symbols whose latest adj_close is above their own 200DMA."""
    total_symbols = universe_bars["symbol"].nunique()
    if total_symbols == 0:
        return None
    above = 0
    judged = 0
    for _, group in universe_bars.groupby("symbol"):
        closes = group.sort_values("date")["adj_close"]
        if len(closes) < 200:
            continue
        judged += 1
        if float(closes.iloc[-1]) > float(closes.rolling(200).mean().iloc[-1]):
            above += 1
    if judged < total_symbols * MIN_BREADTH_COVERAGE:
        return None
    return round(100.0 * above / judged, 1)


def compute_regime(
    spy: pd.DataFrame,
    vix: pd.DataFrame,
    universe_bars: pd.DataFrame,
    crypto_fg: int,
) -> RegimeState:
    as_of = str(max(spy["date"].max(), vix["date"].max()))
    trend = _trend(spy)
    latest_vix = float(vix.sort_values("date")["close"].iloc[-1])
    band = _vix_band(latest_vix)
    breadth = _breadth(universe_bars)

    # conservative composite: any red component degrades; crisis or broken trend = risk_off
    if trend == "risk_off" or band == "crisis":
        composite = "risk_off"
    elif (
        trend == "risk_on"
        and band in ("calm", "normal")
        and (breadth is None or breadth >= BREADTH_WEAK_PCT)
        and crypto_fg >= FG_FEAR
    ):
        composite = "risk_on"
    else:
        composite = "caution"

    return RegimeState(
        as_of=as_of,
        trend=trend,
        vix=round(latest_vix, 2),
        vix_band=band,
        breadth_pct=breadth,
        crypto_fg=crypto_fg,
        composite=composite,
    )


def assemble_regime(
    spy: pd.DataFrame, universe_bars: pd.DataFrame, root: Path = snapshot.DATA_ROOT
) -> RegimeState:
    """Fetch the two live-only inputs (VIX, crypto fear/greed) and compute the
    regime from them plus already-loaded store data (WI-084: this fetch+fetch+
    compute triplet was hand-copied across briefing.engine.assemble,
    risk.__main__, and regime.__main__ — one assembly path now, callers only
    differ in how they load `spy`/`universe_bars` from the store)."""
    vix = fetch_vix(days=300, root=root)
    fng = fetch_fear_greed(limit=30, root=root)
    return compute_regime(spy, vix, universe_bars, crypto_fg=int(fng["value"].iloc[-1]))
