"""Write-once snapshot store + DuckDB catalog.

Raw API payloads are append-only under data/snapshots/{source}/{fetch_date}/
(keyed by fetch time — a payload is written once and never touched again).
Validated output is written once per data date under data/clean/{date}/:
rewriting a clean key with identical content is a no-op (re-runs are harmless),
different content raises SnapshotConflictError — that is the immutability
guarantee backtest reproducibility rests on.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from vega.data.types import SnapshotConflictError

DATA_ROOT = Path("data")


def _stamp() -> tuple[str, str]:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%d"), now.strftime("%H%M%S%f")


def snapshot_raw_json(source: str, name: str, payload: object, root: Path = DATA_ROOT) -> Path:
    day, ts = _stamp()
    path = root / "snapshots" / source / day / f"{name}_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(json.dumps(payload, default=str).encode())
    tmp.rename(path)
    return path


def snapshot_raw_frame(source: str, name: str, frame: pd.DataFrame, root: Path = DATA_ROOT) -> Path:
    day, ts = _stamp()
    path = root / "snapshots" / source / day / f"{name}_{ts}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.rename(path)
    return path


def clean_path(date: str, name: str, root: Path = DATA_ROOT) -> Path:
    return root / "clean" / date / f"{name}.parquet"


def has_clean(date: str, name: str, root: Path = DATA_ROOT) -> bool:
    return clean_path(date, name, root).exists()


def write_clean(date: str, name: str, frame: pd.DataFrame, root: Path = DATA_ROOT) -> Path:
    """Write-once per data date. Identical rewrite = no-op; different content raises.

    Vendors (yfinance in particular) retroactively revise historical adjusted-close
    values when new dividends are declared — so a wider re-ingest WILL see different
    content for already-written dates. That is not corruption; the immutability
    guarantee is doing its job. Callers should use has_clean() to only ingest dates
    not yet in the store, rather than let this raise on routine vendor drift.
    """
    path = clean_path(date, name, root)
    fresh = frame.reset_index(drop=True)
    if path.exists():
        existing = pd.read_parquet(path)
        if existing.equals(fresh):
            return path
        raise SnapshotConflictError(f"{path} already exists with different content")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    fresh.to_parquet(tmp, index=False)
    tmp.rename(path)
    return path


def refresh_catalog(root: Path = DATA_ROOT) -> None:
    """(Re)build DuckDB views over the clean parquet tree (skips views with no files yet)."""
    views = {
        "bars": "clean/*/bars_*.parquet",
        "quarantine": "clean/*/quarantine_*.parquet",
    }
    con = duckdb.connect(str(root / "vega.duckdb"))
    try:
        for view, pattern in views.items():
            if not list(root.glob(pattern)):
                continue
            glob = str(root / pattern)
            # S608: view names and globs come from the hardcoded dict above, never user input
            con.execute(
                f"CREATE OR REPLACE VIEW {view} AS "  # noqa: S608
                f"SELECT * FROM read_parquet('{glob}', union_by_name=true)"
            )
    finally:
        con.close()
