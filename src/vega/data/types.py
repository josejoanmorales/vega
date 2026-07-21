"""Shared data-layer types and the normalized bar schema."""

from __future__ import annotations

from dataclasses import dataclass

# Normalized daily-bar schema every source adapter must produce.
BAR_COLUMNS = ("symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source")

ASSET_CLASSES = ("equity", "etf", "crypto")


@dataclass(frozen=True)
class UniverseEntry:
    symbol: str
    asset_class: str
    name: str
    coingecko_id: str = ""
    binance_symbol: str = ""
    # Sleeve for risk.clusters heat-cap classification (universe-v2, WI-084 item
    # 8 -- previously hardcoded RATES/COMMODITIES frozensets in risk/clusters.py).
    # Optional/defaulted so pre-v2 artifacts and test fixtures without the column
    # still parse; vega.data.universe.load_universe fills a sensible default
    # (us_equity_beta / crypto_beta by asset_class) when the CSV omits it.
    cluster: str = ""


class SnapshotConflictError(RuntimeError):
    """A write-once key was rewritten with different content — data changed underneath us."""
