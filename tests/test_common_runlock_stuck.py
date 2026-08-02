"""WI-129: telling a healthy concurrent run apart from a wedged one."""

import os
import time
from pathlib import Path

from vega.common.runlock import (
    acquire_run_lock,
    held_for_seconds,
    holder,
    is_run_in_progress,
)


def test_hold_age_starts_at_zero_and_records_the_holder(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    with acquire_run_lock(lock):
        age = held_for_seconds(lock)
        assert age is not None and age < 5
        who = holder(lock)
        assert who is not None
        assert who["pid"] == os.getpid()  # the pid a human would go kill
        assert who["started_at"]


def test_probes_do_not_reset_the_hold_age(tmp_path: Path) -> None:
    """THE regression that makes stuck-detection possible at all.

    `acquire_run_lock` used to open with "w", truncating BEFORE flock — so a
    failed probe still stamped the mtime. `is_run_in_progress()` runs on every
    dashboard poll, so a wedged run would have looked permanently young and
    never crossed the stuck threshold.
    """
    lock = tmp_path / "run.lock"
    with acquire_run_lock(lock):
        time.sleep(1.1)
        before = held_for_seconds(lock)
        assert before is not None and before >= 1.0

        for _ in range(5):
            assert is_run_in_progress(lock) is True  # each probe fails to acquire

        after = held_for_seconds(lock)
        assert after is not None
        assert after >= before  # age kept growing; probes did not reset it


def test_age_is_none_when_the_lock_was_never_taken(tmp_path: Path) -> None:
    assert held_for_seconds(tmp_path / "never.lock") is None
    assert holder(tmp_path / "never.lock") is None


def test_a_fresh_acquisition_restamps_the_age(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    with acquire_run_lock(lock):
        pass
    time.sleep(1.1)
    with acquire_run_lock(lock):
        age = held_for_seconds(lock)
        assert age is not None and age < 1.0  # new hold, new clock


def test_probing_a_free_lock_does_not_stamp_it(tmp_path: Path) -> None:
    """A probe must be read-only. `/api/status` probes on every dashboard
    poll; stamping there would rewrite+fsync the lock every few seconds and
    replace the holder identity with the web server's pid."""
    lock = tmp_path / "run.lock"
    with acquire_run_lock(lock):
        real_holder = holder(lock)
    stamped_at = lock.stat().st_mtime

    time.sleep(1.1)
    for _ in range(3):
        assert is_run_in_progress(lock) is False  # lock is free; probe succeeds

    assert lock.stat().st_mtime == stamped_at  # untouched by probes
    assert holder(lock) == real_holder  # identity not overwritten
