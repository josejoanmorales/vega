from pathlib import Path

from vega.common.appendlog import AppendLog


def test_records_reflects_appends_from_a_fresh_instance(tmp_path: Path) -> None:
    """WI-084-i4: AppendLog caches records() by (path, mtime, size). A second,
    freshly-constructed AppendLog over the same path must still see writes
    made through a first instance — this is the exact usage pattern (fresh
    Store/Registry per call site) the cache has to survive."""
    path = tmp_path / "log.jsonl"
    a = AppendLog(path)
    a.append({"type": "x", "n": 1})

    b = AppendLog(path)  # fresh instance, same underlying file
    assert b.records() == [{"type": "x", "n": 1}]

    a.append({"type": "x", "n": 2})
    assert b.records() == [{"type": "x", "n": 1}, {"type": "x", "n": 2}]


def test_records_returns_independent_lists(tmp_path: Path) -> None:
    """Caller mutation of a returned list must never corrupt the cache for
    the next caller (the cache stores dicts read-only across the codebase,
    but the outer list itself must always be a fresh copy)."""
    path = tmp_path / "log.jsonl"
    log = AppendLog(path)
    log.append({"type": "x", "n": 1})

    first = log.records()
    first.append({"type": "x", "n": 999})  # mutate the returned list

    second = log.records()
    assert second == [{"type": "x", "n": 1}]
