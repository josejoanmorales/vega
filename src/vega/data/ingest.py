"""Daily ingest: fetch → snapshot raw → cross-validate → write-once clean → catalog.

Run: uv run python -m vega.data.ingest [days]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from vega.data import snapshot
from vega.data.sources import alpaca_src, binance_src, coingecko_src, yfinance_src
from vega.data.universe import load_universe, symbols
from vega.data.validate import CrossCheckResult, cross_check


@dataclass(frozen=True)
class IngestSummary:
    clean_rows: int  # rows actually added to the store this run
    quarantined_rows: int  # quarantine rows actually added this run
    frozen_rows: int  # incoming rows skipped because their (symbol, date) is frozen
    drift_rows: int  # frozen rows whose freshly fetched close differed (vendor revision)
    dates: tuple[str, ...]


def _write_result(
    result: CrossCheckResult, sleeve: str, root: Path
) -> tuple[int, int, int, int, set[str]]:
    """Per-(symbol, date) write-once merge — the policy lives in snapshot.merge_clean.

    Vendors revise historical values retroactively (yfinance restates past
    adjusted-close when new dividends are declared), so a wider re-ingest
    routinely disagrees with frozen rows. Frozen rows stay frozen; genuinely
    new (symbol, date) rows — including symbols absent from an already-written
    date — are appended; drift against frozen rows is counted, not raised.
    """
    dates: set[str] = set()
    added = quarantined = frozen = drift = 0
    all_dates = sorted(set(result.clean["date"]) | set(result.quarantine["date"]))
    for date in all_dates:
        day_bars = result.clean[result.clean["date"] == date].reset_index(drop=True)
        day_quar = result.quarantine[result.quarantine["date"] == date].reset_index(drop=True)
        b, q, f, d = snapshot.merge_clean(
            str(date), f"bars_{sleeve}", f"quarantine_{sleeve}", day_bars, day_quar, root
        )
        added += b
        quarantined += q
        frozen += f
        drift += d
        if b or q:
            dates.add(str(date))
    return added, quarantined, frozen, drift, dates


def run(days: int = 7, root: Path = snapshot.DATA_ROOT) -> IngestSummary:
    load_dotenv()
    universe = load_universe()
    equities = symbols(universe, "equity", "etf")
    crypto = [e for e in universe if e.asset_class == "crypto"]

    today = datetime.now(UTC).date().isoformat()
    start = (datetime.now(UTC).date() - timedelta(days=days)).isoformat()

    yf_bars = yfinance_src.fetch_daily(equities, start, today)
    snapshot.snapshot_raw_frame("yfinance", "bars", yf_bars, root)
    alp_bars = alpaca_src.fetch_daily(equities, start, today)
    snapshot.snapshot_raw_frame("alpaca_iex", "bars", alp_bars, root)

    # CoinGecko keyless access is capped at 365 days of history — the crypto sleeve's
    # window is capped with it so primary bars never outrun their cross-check source.
    crypto_days = min(days, 364)
    bn_bars, bn_raw = binance_src.fetch_daily(
        {e.symbol: e.binance_symbol for e in crypto}, crypto_days
    )
    snapshot.snapshot_raw_json("binance", "klines", bn_raw, root)
    cg_bars, cg_raw = coingecko_src.fetch_daily(
        {e.symbol: e.coingecko_id for e in crypto}, crypto_days
    )
    snapshot.snapshot_raw_json("coingecko", "market_chart", cg_raw, root)

    # strictness: only fully completed sessions enter the clean store
    yf_bars = yf_bars[yf_bars["date"] < today]
    alp_bars = alp_bars[alp_bars["date"] < today]

    eq_result = cross_check(yf_bars, alp_bars)
    cr_result = cross_check(bn_bars, cg_bars)

    eq_clean, eq_bad, eq_frozen, eq_drift, eq_dates = _write_result(eq_result, "equity", root)
    cr_clean, cr_bad, cr_frozen, cr_drift, cr_dates = _write_result(cr_result, "crypto", root)
    snapshot.refresh_catalog(root)

    return IngestSummary(
        clean_rows=eq_clean + cr_clean,
        quarantined_rows=eq_bad + cr_bad,
        frozen_rows=eq_frozen + cr_frozen,
        drift_rows=eq_drift + cr_drift,
        dates=tuple(sorted(eq_dates | cr_dates)),
    )


def main() -> None:
    pd.set_option("display.width", 160)
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    s = run(days)
    print(
        f"ingest ok — added: {s.clean_rows} clean / {s.quarantined_rows} quarantined, "
        f"frozen (already stored): {s.frozen_rows}, vendor drift on frozen rows: {s.drift_rows}, "
        f"dates touched: {len(s.dates)}"
    )


if __name__ == "__main__":
    main()
