"""Versioned tradable-universe artifact (data/universe/universe-vN.csv, committed to git).

Versions are append-only: a refresh produces universe-v{N+1}.csv, never mutates v{N}.
"""

from __future__ import annotations

import csv
from pathlib import Path

from vega.data.types import ASSET_CLASSES, UniverseEntry

DEFAULT_ARTIFACT = Path("data/universe/universe-v1.csv")


def load_universe(path: Path = DEFAULT_ARTIFACT) -> list[UniverseEntry]:
    entries: list[UniverseEntry] = []
    with path.open(newline="") as fh:
        rows = [line for line in fh if not line.startswith("#")]
    for rec in csv.DictReader(rows):
        asset_class = rec["asset_class"].strip()
        if asset_class not in ASSET_CLASSES:
            raise ValueError(f"unknown asset_class {asset_class!r} for {rec['symbol']}")
        entry = UniverseEntry(
            symbol=rec["symbol"].strip(),
            asset_class=asset_class,
            name=rec["name"].strip(),
            coingecko_id=(rec.get("coingecko_id") or "").strip(),
            binance_symbol=(rec.get("binance_symbol") or "").strip(),
        )
        if entry.asset_class == "crypto" and not (entry.coingecko_id and entry.binance_symbol):
            raise ValueError(f"crypto entry {entry.symbol} is missing source mappings")
        entries.append(entry)
    if not entries:
        raise ValueError(f"universe artifact {path} is empty")
    return entries


def universe_version(path: Path = DEFAULT_ARTIFACT) -> str:
    """e.g. 'universe-v1.csv' -> 'v1' — recorded on every backtest run for provenance."""
    stem = path.stem
    return stem.split("-")[-1] if "-" in stem else stem


def symbols(entries: list[UniverseEntry], *classes: str) -> list[str]:
    wanted = classes or ASSET_CLASSES
    return [e.symbol for e in entries if e.asset_class in wanted]
