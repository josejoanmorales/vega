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
    clean_rows: int
    quarantined_rows: int
    dates: tuple[str, ...]


def _write_result(result: CrossCheckResult, sleeve: str, root: Path) -> tuple[int, int, set[str]]:
    dates: set[str] = set()
    for date, group in result.clean.groupby("date"):
        snapshot.write_clean(str(date), f"bars_{sleeve}", group.reset_index(drop=True), root)
        dates.add(str(date))
    for date, group in result.quarantine.groupby("date"):
        snapshot.write_clean(str(date), f"quarantine_{sleeve}", group.reset_index(drop=True), root)
        dates.add(str(date))
    return len(result.clean), len(result.quarantine), dates


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

    eq_clean, eq_bad, eq_dates = _write_result(eq_result, "equity", root)
    cr_clean, cr_bad, cr_dates = _write_result(cr_result, "crypto", root)
    snapshot.refresh_catalog(root)

    return IngestSummary(
        clean_rows=eq_clean + cr_clean,
        quarantined_rows=eq_bad + cr_bad,
        dates=tuple(sorted(eq_dates | cr_dates)),
    )


def main() -> None:
    pd.set_option("display.width", 160)
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    s = run(days)
    print(
        f"ingest ok — clean rows: {s.clean_rows}, quarantined: {s.quarantined_rows}, "
        f"dates: {', '.join(s.dates)}"
    )


if __name__ == "__main__":
    main()
