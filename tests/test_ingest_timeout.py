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


# ---- sleeve independence (production incident 2026-08-05..08) --------------


def _stub_equity(monkeypatch, bars) -> None:
    from vega.data import snapshot
    from vega.data.sources import alpaca_src, yfinance_src

    monkeypatch.setattr(yfinance_src, "fetch_daily", lambda *a, **k: bars)
    monkeypatch.setattr(alpaca_src, "fetch_daily", lambda *a, **k: bars)
    monkeypatch.setattr(snapshot, "snapshot_raw_frame", lambda *a, **k: None)
    monkeypatch.setattr(snapshot, "snapshot_raw_json", lambda *a, **k: None)
    monkeypatch.setattr(snapshot, "refresh_catalog", lambda *a, **k: None)


def _equity_bars():
    import pandas as pd

    from vega.data.types import BAR_COLUMNS

    return pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "date": "2020-01-02",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "adj_close": 1.0,
                "volume": 1.0,
                "source": "yfinance",
            }
        ]
    )[list(BAR_COLUMNS)]


def test_a_crypto_vendor_outage_no_longer_discards_equity_bars(monkeypatch, tmp_path) -> None:
    """THE production incident: CoinGecko — the crypto CROSS-CHECK source, for
    a system holding only equity positions — dropped a connection every
    morning. Because it was fetched last in one all-or-nothing sequence, the
    already-successful equity bars never reached the clean store and the store
    sat frozen for days while yfinance worked perfectly.
    """
    import requests

    from vega.data.sources import binance_src, coingecko_src

    _stub_equity(monkeypatch, _equity_bars())
    monkeypatch.setattr(binance_src, "fetch_daily", lambda *a, **k: (_equity_bars(), {}))

    def _coingecko_down(*a, **k):
        raise requests.exceptions.ConnectionError("Connection aborted, RemoteDisconnected")

    monkeypatch.setattr(coingecko_src, "fetch_daily", _coingecko_down)

    summary = ingest.run(days=1, root=tmp_path, cap_s=30)

    assert summary.failed_sleeves == ("crypto",)  # reported, not hidden
    assert summary.clean_rows > 0, "equity bars must still reach the clean store"


def test_every_sleeve_failing_still_raises(monkeypatch, tmp_path) -> None:
    """A partial ingest degrades gracefully; a TOTAL one must not return a
    summary that reads like success — it raises so WI-103's degrade path runs."""
    from vega.data.sources import alpaca_src, binance_src, coingecko_src, yfinance_src

    def _down(*a, **k):
        raise ConnectionError("vendor down")

    for mod in (yfinance_src, alpaca_src, binance_src, coingecko_src):
        monkeypatch.setattr(mod, "fetch_daily", _down)

    with pytest.raises(ingest.IngestError, match="every ingest sleeve failed"):
        ingest.run(days=1, root=tmp_path, cap_s=30)
