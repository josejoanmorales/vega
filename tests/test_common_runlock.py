import subprocess
import sys
from pathlib import Path

import pytest

from vega.common.runlock import RunInProgress, acquire_run_lock, is_run_in_progress


def test_lock_acquired_then_released(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    assert is_run_in_progress(path) is False
    with acquire_run_lock(path):
        pass
    assert is_run_in_progress(path) is False


def test_second_acquire_raises_while_first_holds(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    with acquire_run_lock(path):
        assert is_run_in_progress(path) is True
        with pytest.raises(RunInProgress):
            with acquire_run_lock(path):
                pass


def test_lock_releases_when_holding_process_dies(tmp_path: Path) -> None:
    # flock is held only for the fd's lifetime -- a crashed/killed process
    # must never leave a stale lock behind.
    path = tmp_path / "run.lock"
    proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell, no user input
        [
            sys.executable,
            "-c",
            f"from vega.common.runlock import acquire_run_lock\n"
            f"from pathlib import Path\n"
            f"with acquire_run_lock(Path({str(path)!r})):\n"
            f"    import time; time.sleep(5)\n",
        ],
    )
    try:
        import time

        for _ in range(50):
            if is_run_in_progress(path):
                break
            time.sleep(0.05)
        else:
            pytest.fail("child process never acquired the lock")
        proc.kill()
        proc.wait()
        assert is_run_in_progress(path) is False
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
