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


class SnapshotConflictError(RuntimeError):
    """A write-once key was rewritten with different content — data changed underneath us."""
