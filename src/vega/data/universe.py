"""Versioned tradable-universe artifact (data/universe/universe-vN.csv, committed to git).

Versions are append-only: a refresh produces universe-v{N+1}.csv, never mutates v{N}.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from vega.common.paths import DATA_ROOT
from vega.data.types import ASSET_CLASSES, UniverseEntry

# Anchored via common.paths (WI-089 live smoke: a long-running server process
# invoked outside the repo root — unlike every prior CLI caller, which always
# ran via `uv run python -m vega.X` from the repo root by convention — hit
# this the moment a CWD-relative default was used from anywhere else. Same
# fragility class common/paths.py was created to close for other modules.
DEFAULT_ARTIFACT = DATA_ROOT / "universe" / "universe-v1.csv"


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


def universe_version(directory: Path = DEFAULT_ARTIFACT.parent) -> str:
    """Highest version present in `directory` (e.g. 'v2'), or 'unknown' if none.

    Recorded on every backtest run for provenance — 'unknown' is honest;
    a hardcoded guess would let the append-only registry lie.
    """
    versions = sorted(
        int(m.group(1))
        for p in directory.glob("universe-v*.csv")
        if (m := re.search(r"universe-v(\d+)", p.stem))
    )
    return f"v{versions[-1]}" if versions else "unknown"


def symbols(entries: list[UniverseEntry], *classes: str) -> list[str]:
    wanted = classes or ASSET_CLASSES
    return [e.symbol for e in entries if e.asset_class in wanted]
