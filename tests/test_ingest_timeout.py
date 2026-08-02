"""WI-129: the wall-clock cap on ingest itself, end to end."""

import time

import pytest

from vega.common.timeouts import HardTimeout
from vega.data import ingest
from vega.data.sources import yfinance_src


def test_a_hung_vendor_raises_instead_of_wedging(monkeypatch, tmp_path) -> None:
    """The 2026-07-27 shape: the vendor call never returns and never raises.

    Before this cap, ingest.run inherited that hang and the process sat at 0%
    CPU for days holding the run lock. Now it raises — which is exactly what
    WI-103's retry-then-degrade already knows how to handle.
    """

    def _hangs(*args, **kwargs):
        time.sleep(60)

    monkeypatch.setattr(yfinance_src, "fetch_daily", _hangs)

    t0 = time.time()
    with pytest.raises(HardTimeout, match="ingest exceeded"):
        ingest.run(days=1, root=tmp_path, cap_s=0.4)
    assert time.time() - t0 < 10  # cut off, not waited out


def test_cap_is_disarmed_on_the_success_path(monkeypatch, tmp_path) -> None:
    """A leaked itimer would fire into whatever ran next (the briefing)."""
    import pandas as pd

    from vega.data import snapshot
    from vega.data.sources import alpaca_src, binance_src, coingecko_src
    from vega.data.types import BAR_COLUMNS

    empty = pd.DataFrame(columns=list(BAR_COLUMNS))
    monkeypatch.setattr(yfinance_src, "fetch_daily", lambda *a, **k: empty)
    monkeypatch.setattr(alpaca_src, "fetch_daily", lambda *a, **k: empty)
    monkeypatch.setattr(binance_src, "fetch_daily", lambda *a, **k: (empty, {}))
    monkeypatch.setattr(coingecko_src, "fetch_daily", lambda *a, **k: (empty, {}))
    monkeypatch.setattr(snapshot, "snapshot_raw_frame", lambda *a, **k: None)
    monkeypatch.setattr(snapshot, "snapshot_raw_json", lambda *a, **k: None)
    monkeypatch.setattr(snapshot, "refresh_catalog", lambda *a, **k: None)

    ingest.run(days=1, root=tmp_path, cap_s=0.5)
    time.sleep(0.8)  # would raise here if the alarm were still armed
