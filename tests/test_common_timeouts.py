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


def test_an_inner_cap_does_not_cancel_the_outer_one() -> None:
    """There is ONE process itimer. A naive inner block disarms it on exit and
    the enclosing deadline is lost forever — the outer cap then never fires,
    silently. This is the regression that would have bitten the moment the
    briefing was capped with ingest nested inside it."""
    t0 = time.time()
    with pytest.raises(HardTimeout, match="OUTER"):
        with hard_timeout(1.0, "OUTER"):
            with hard_timeout(0.2, "inner"):
                pass
            time.sleep(5)  # outer must still fire at ~1.0s
    elapsed = time.time() - t0
    assert 0.8 < elapsed < 2.5, elapsed


def test_an_inner_cap_cannot_extend_the_outer_budget() -> None:
    """The tighter deadline always wins: a 30s inner block inside a 0.5s outer
    one must die at 0.5s, not buy itself another 30."""
    t0 = time.time()
    with pytest.raises(HardTimeout):
        with hard_timeout(0.5, "OUTER"):
            with hard_timeout(30, "greedy inner"):
                time.sleep(10)
    assert time.time() - t0 < 3


def test_an_outer_cap_already_overdue_fires_promptly() -> None:
    """If the outer deadline passed while inside the inner block, it must not
    be dropped — it fires as soon as control returns."""
    t0 = time.time()
    with pytest.raises(HardTimeout):
        with hard_timeout(0.4, "OUTER"):
            with hard_timeout(1.0, "inner"):
                pass
            time.sleep(3)
    assert time.time() - t0 < 3


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


def test_hard_timeout_is_not_swallowed_by_bare_except_exception() -> None:
    """THE property the briefing cap depends on.

    The briefing/execution path contains eight `except Exception` handlers
    ("unreachable venue", "one bad order must not stop the batch", "unknown
    earnings is acceptable"). If HardTimeout were an Exception, whichever one
    the hang happened inside would swallow the cap and turn a wedged,
    order-placing pipeline into a benign-looking vendor miss.
    """
    assert not issubclass(HardTimeout, Exception)

    swallowed = False
    with pytest.raises(HardTimeout):
        with hard_timeout(0.3, "guarded work"):
            try:
                time.sleep(30)
            except Exception:  # noqa: BLE001 — simulating the handlers downstream
                swallowed = True
    assert not swallowed


def test_cleanup_still_runs_when_the_cap_fires() -> None:
    """BaseException must still unwind finally blocks — locks released, files
    closed. Bypassing handlers must not mean bypassing cleanup."""
    cleaned = []
    with pytest.raises(HardTimeout):
        with hard_timeout(0.3, "work"):
            try:
                time.sleep(30)
            finally:
                cleaned.append("cleanup ran")
    assert cleaned == ["cleanup ran"]
