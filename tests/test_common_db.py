"""vega.common.db — the one place every module opens the DuckDB store
(WI-084: connect+close boilerplate was hand-copied across ~9 call sites)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from vega.common import db


def _make_store(root: Path) -> None:
    con = duckdb.connect(str(root / "vega.duckdb"))
    con.execute("CREATE TABLE bars (symbol VARCHAR, date VARCHAR)")
    con.execute("INSERT INTO bars VALUES ('AAA', '2026-01-01')")
    con.close()


def test_connect_yields_a_working_connection_and_closes_on_exit(tmp_path: Path) -> None:
    _make_store(tmp_path)
    with db.connect(tmp_path) as con:
        row = con.execute("SELECT count(*) FROM bars").fetchone()
        assert row is not None and row[0] == 1
    # closed: a query against the same connection object now raises
    with pytest.raises(Exception):  # noqa: B017, PT011 — duckdb's own post-close error
        con.execute("SELECT 1")


def test_default_is_read_only(tmp_path: Path) -> None:
    _make_store(tmp_path)
    with db.connect(tmp_path) as con, pytest.raises(Exception):  # noqa: B017, PT011
        con.execute("INSERT INTO bars VALUES ('BBB', '2026-01-02')")


def test_read_only_false_permits_writes(tmp_path: Path) -> None:
    _make_store(tmp_path)
    with db.connect(tmp_path, read_only=False) as con:
        con.execute("INSERT INTO bars VALUES ('BBB', '2026-01-02')")
        row = con.execute("SELECT count(*) FROM bars").fetchone()
        assert row is not None and row[0] == 2
