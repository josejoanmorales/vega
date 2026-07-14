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

from vega.common.paths import DATA_ROOT as DATA_ROOT  # project-anchored, never CWD-relative
from vega.data.types import SnapshotConflictError


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


def merge_clean(
    date: str,
    bars_name: str,
    quarantine_name: str,
    bars: pd.DataFrame,
    quarantine: pd.DataFrame,
    root: Path = DATA_ROOT,
) -> tuple[int, int, int, int]:
    """Per-(symbol, date) write-once across a bars/quarantine file pair.

    A symbol already present on EITHER side for this date is frozen — its rows
    are never modified or contradicted (no mixed-vintage partitions, no
    stale-quarantine flips, no silent rewrites). Symbols present in neither
    file are appended to whichever side this run assigns them — so a symbol
    missing from an old date (short listing history, later universe extension,
    or an originally quarantined day whose data was never stored) can still
    enter the store, instead of being dropped because the date's file exists.

    Returns (bars_added, quarantine_added, frozen_skipped, drift_rows).
    drift_rows counts frozen bars whose freshly fetched close differs — the
    detection signal the old SnapshotConflictError provided, kept as
    observability instead of a crash (vendor revisions are routine).
    """
    bars_path = clean_path(date, bars_name, root)
    quar_path = clean_path(date, quarantine_name, root)
    existing_bars = pd.read_parquet(bars_path) if bars_path.exists() else None
    existing_quar = pd.read_parquet(quar_path) if quar_path.exists() else None

    frozen: set[str] = set()
    if existing_bars is not None:
        frozen |= set(existing_bars["symbol"])
    if existing_quar is not None:
        frozen |= set(existing_quar["symbol"])

    drift = 0
    if existing_bars is not None and not bars.empty:
        overlap = existing_bars[["symbol", "close"]].merge(
            bars[["symbol", "close"]], on="symbol", suffixes=("_old", "_new")
        )
        drift = int((overlap["close_old"] != overlap["close_new"]).sum())

    new_bars = bars[~bars["symbol"].isin(frozen)]
    new_quar = quarantine[~quarantine["symbol"].isin(frozen)]
    frozen_skipped = (len(bars) - len(new_bars)) + (len(quarantine) - len(new_quar))

    def _extend(path: Path, existing: pd.DataFrame | None, new: pd.DataFrame) -> None:
        if new.empty:
            return
        combined = new if existing is None else pd.concat([existing, new], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.parquet")
        combined.reset_index(drop=True).to_parquet(tmp, index=False)
        tmp.rename(path)

    _extend(bars_path, existing_bars, new_bars)
    _extend(quar_path, existing_quar, new_quar)
    return len(new_bars), len(new_quar), frozen_skipped, drift


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
