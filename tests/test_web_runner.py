import subprocess
import sys
import time
from pathlib import Path

import pytest

from vega.common.runlock import acquire_run_lock
from vega.web.runner import RunAlreadyInProgress, Runner

_REAL_POPEN = subprocess.Popen  # captured before any test monkeypatches the module attribute


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("condition never became true")


def test_idle_status_before_any_run(tmp_path: Path) -> None:
    runner = Runner(runs_dir=tmp_path)
    assert runner.status() == {"state": "idle"}


def test_successful_run_transitions_to_succeeded(tmp_path: Path, monkeypatch) -> None:
    runner = Runner(runs_dir=tmp_path)
    monkeypatch.setattr(
        "vega.web.runner.subprocess.Popen",
        lambda *a, **kw: _REAL_POPEN(
            [sys.executable, "-c", "print('ok')"], stdout=kw["stdout"], stderr=kw["stderr"]
        ),
    )
    run_id = runner.start()
    _wait_for(lambda: runner.status()["state"] != "running")
    status = runner.status()
    assert status["state"] == "succeeded"
    assert status["run_id"] == run_id
    assert status["returncode"] == 0
    assert "ok" in status["log_tail"]


def test_failing_run_transitions_to_failed(tmp_path: Path, monkeypatch) -> None:
    runner = Runner(runs_dir=tmp_path)
    monkeypatch.setattr(
        "vega.web.runner.subprocess.Popen",
        lambda *a, **kw: _REAL_POPEN(
            [sys.executable, "-c", "import sys; sys.exit(1)"],
            stdout=kw["stdout"],
            stderr=kw["stderr"],
        ),
    )
    runner.start()
    _wait_for(lambda: runner.status()["state"] != "running")
    status = runner.status()
    assert status["state"] == "failed"
    assert status["returncode"] == 1


def test_second_start_while_running_is_refused(tmp_path: Path, monkeypatch) -> None:
    runner = Runner(runs_dir=tmp_path)
    monkeypatch.setattr(
        "vega.web.runner.subprocess.Popen",
        lambda *a, **kw: _REAL_POPEN(
            [sys.executable, "-c", "import time; time.sleep(1)"],
            stdout=kw["stdout"],
            stderr=kw["stderr"],
        ),
    )
    runner.start()
    with pytest.raises(RunAlreadyInProgress):
        runner.start()


def test_start_refused_when_external_lock_held(tmp_path: Path) -> None:
    runner = Runner(runs_dir=tmp_path)
    lock_path = tmp_path / "external.lock"
    with acquire_run_lock(lock_path):
        import vega.web.runner as runner_module

        original = runner_module.is_run_in_progress
        runner_module.is_run_in_progress = lambda path=None: original(lock_path)  # type: ignore[assignment]
        try:
            with pytest.raises(RunAlreadyInProgress):
                runner.start()
        finally:
            runner_module.is_run_in_progress = original


def test_skipped_exit_code_maps_to_skipped_not_failed(tmp_path: Path, monkeypatch) -> None:
    # WI-088 review: a lost lock race (EXIT_SKIPPED=3) is a correct no-op,
    # NOT a failure — the UI and launchd logs must not mistake it for one.
    runner = Runner(runs_dir=tmp_path)
    monkeypatch.setattr(
        "vega.web.runner.subprocess.Popen",
        lambda *a, **kw: _REAL_POPEN(
            [sys.executable, "-c", "import sys; sys.exit(3)"],
            stdout=kw["stdout"],
            stderr=kw["stderr"],
        ),
    )
    runner.start()
    _wait_for(lambda: runner.status()["state"] != "running")
    status = runner.status()
    assert status["state"] == "skipped"
    assert status["returncode"] == 3


def test_status_reports_external_when_lock_held_without_tracked_run(tmp_path: Path) -> None:
    # WI-088 review: a restarted server (no tracked run) must not claim "idle"
    # while an orphaned/launchd pipeline still holds the lock.
    runner = Runner(runs_dir=tmp_path)
    import vega.web.runner as runner_module

    original = runner_module.is_run_in_progress
    runner_module.is_run_in_progress = lambda path=None: True  # type: ignore[assignment]
    try:
        assert runner.status() == {"state": "external"}
    finally:
        runner_module.is_run_in_progress = original
