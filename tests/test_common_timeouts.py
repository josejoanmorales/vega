"""WI-129: a hang must become an ordinary exception."""

import threading
import time

import pytest

from vega.common.timeouts import HardTimeout, hard_timeout


def test_hang_raises_instead_of_blocking_forever() -> None:
    t0 = time.time()
    with pytest.raises(HardTimeout, match="hung, not failed"):
        with hard_timeout(0.3, "fake vendor fetch"):
            time.sleep(30)  # the 2026-07-27 shape: never returns, never raises
    assert time.time() - t0 < 5  # interrupted, not waited out


def test_fast_work_is_untouched() -> None:
    with hard_timeout(30, "quick"):
        result = 2 + 2
    assert result == 4


def test_alarm_is_disarmed_after_the_block() -> None:
    # A leaked itimer would fire into unrelated later work.
    with hard_timeout(0.3, "first"):
        pass
    time.sleep(0.5)  # would have fired by now if still armed


def test_nested_use_restores_the_outer_handler() -> None:
    import signal

    original = signal.getsignal(signal.SIGALRM)
    with hard_timeout(10, "outer"):
        with hard_timeout(5, "inner"):
            pass
    assert signal.getsignal(signal.SIGALRM) is original


def test_off_main_thread_degrades_to_a_no_op() -> None:
    # Cannot arm a signal off the main thread; the block must still RUN
    # rather than raising — a cap that can't be armed must not break callers.
    out = []

    def worker() -> None:
        with hard_timeout(0.2, "threaded"):
            time.sleep(0.4)
        out.append("completed")

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=10)
    assert out == ["completed"]
