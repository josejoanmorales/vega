"""One place every module opens the DuckDB store (WI-084: this connect+close
boilerplate was hand-copied across ~9 call sites — dashboard, snapshot,
briefing, backtest, risk, regime, lifecycle).

A context manager, not a bare `connect()` wrapper, so `close()` is never
forgotten at a call site the way a raw `duckdb.connect` invites.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb


@contextmanager
def connect(root: Path, read_only: bool = True) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open `<root>/vega.duckdb`, closing it on the way out (success or not).

    `read_only` defaults True since almost every caller only queries; the one
    writer (`data.snapshot.refresh_catalog`) passes `read_only=False`.
    """
    con = duckdb.connect(str(root / "vega.duckdb"), read_only=read_only)
    try:
        yield con
    finally:
        con.close()
