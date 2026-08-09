"""Daily ingest: fetch → snapshot raw → cross-validate → write-once clean → catalog.

Run: uv run python -m vega.data.ingest [days]
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from vega.common.timeouts import HardTimeout, hard_timeout
from vega.data import snapshot
from vega.data.sources import alpaca_src, binance_src, coingecko_src, yfinance_src
from vega.data.types import BAR_COLUMNS
from vega.data.universe import load_universe, symbols
from vega.data.validate import CrossCheckResult, cross_check

# Total wall-clock cap (WI-129). Every adapter has its own per-request
# timeout, but those only bound calls we control in libraries we do not: the
# 2026-07-27 wedge was inside yfinance's own machinery and never returned.
# This is the catch-all that converts ANY hang into an exception, which
# WI-103's retry-then-degrade already handles (exits still get evaluated).
# Generous vs a real run (~6 min observed on a 4-session catch-up, and
# CoinGecko alone paces 20 crypto symbols at 7s apart).
WALL_CLOCK_CAP_S = 900.0


class IngestError(RuntimeError):
    """No sleeve produced data — nothing can be written this run."""


@dataclass(frozen=True)
class IngestSummary:
    clean_rows: int  # rows actually added to the store this run
    quarantined_rows: int  # quarantine rows actually added this run
    frozen_rows: int  # incoming rows skipped because their (symbol, date) is frozen
    drift_rows: int  # frozen rows whose freshly fetched close differed (vendor revision)
    dates: tuple[str, ...]
    # Sleeves that failed while others succeeded — a PARTIAL ingest. Empty on a
    # fully clean run. Never silently empty on failure: if every sleeve fails,
    # run() raises IngestError instead of returning a summary that looks fine.
    failed_sleeves: tuple[str, ...] = ()


_EMPTY_RESULT = CrossCheckResult(
    clean=pd.DataFrame(columns=list(BAR_COLUMNS)),
    quarantine=pd.DataFrame(columns=list(BAR_COLUMNS)),
)


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


def run(
    days: int = 7, root: Path = snapshot.DATA_ROOT, cap_s: float = WALL_CLOCK_CAP_S
) -> IngestSummary:
    """Fetch → snapshot → cross-check → write-once, under a hard wall-clock
    cap so a hung vendor raises instead of wedging the pipeline (WI-129).
    The cap wraps the WHOLE body: a partial ingest is safe to abandon and
    retry because the clean store is per-(symbol, date) write-once."""
    with hard_timeout(cap_s, "ingest"):
        return _run(days, root)


def _equity_sleeve(equities: list[str], start: str, today: str, root: Path) -> CrossCheckResult:
    yf_bars = yfinance_src.fetch_daily(equities, start, today)
    snapshot.snapshot_raw_frame("yfinance", "bars", yf_bars, root)
    alp_bars = alpaca_src.fetch_daily(equities, start, today)
    snapshot.snapshot_raw_frame("alpaca_iex", "bars", alp_bars, root)
    # strictness: only fully completed sessions enter the clean store
    return cross_check(yf_bars[yf_bars["date"] < today], alp_bars[alp_bars["date"] < today])


def _crypto_sleeve(crypto: list[Any], days: int, root: Path) -> CrossCheckResult:
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
    return cross_check(bn_bars, cg_bars)


def _run(days: int, root: Path) -> IngestSummary:
    load_dotenv()
    universe = load_universe()
    equities = symbols(universe, "equity", "etf")
    crypto = [e for e in universe if e.asset_class == "crypto"]

    today = datetime.now(UTC).date().isoformat()
    start = (datetime.now(UTC).date() - timedelta(days=days)).isoformat()

    # THE SLEEVES ARE INDEPENDENT. Previously one sequence fetched all four
    # vendors and only then cross-checked and wrote, so ANY vendor failure
    # discarded the work of all the others. Observed in production
    # 2026-08-05..08: CoinGecko — the crypto CROSS-CHECK source, for a system
    # holding only equity positions — dropped a connection every morning, and
    # because it is fetched last the already-successful equity bars never
    # reached the clean store. The store sat frozen for days while yfinance
    # worked perfectly. A sleeve may now fail alone; the summary reports it so
    # a partial ingest is visible rather than silently "fine".
    failed: dict[str, str] = {}
    results: dict[str, CrossCheckResult] = {}
    sleeves: tuple[tuple[str, Callable[[], CrossCheckResult]], ...] = (
        ("equity", lambda: _equity_sleeve(equities, start, today, root)),
        ("crypto", lambda: _crypto_sleeve(crypto, days, root)),
    )
    for name, fetch in sleeves:
        try:
            results[name] = fetch()
        except HardTimeout:
            raise  # the wall-clock cap is not a per-sleeve error; abort the run
        except Exception as exc:  # noqa: BLE001 — one vendor must not sink the others
            failed[name] = f"{type(exc).__name__}: {exc}"
            print(f"INGEST SLEEVE FAILED [{name}]: {failed[name]}", file=sys.stderr)

    if not results:
        raise IngestError(f"every ingest sleeve failed: {failed}")

    eq_result = results.get("equity", _EMPTY_RESULT)
    cr_result = results.get("crypto", _EMPTY_RESULT)

    eq_clean, eq_bad, eq_frozen, eq_drift, eq_dates = _write_result(eq_result, "equity", root)
    cr_clean, cr_bad, cr_frozen, cr_drift, cr_dates = _write_result(cr_result, "crypto", root)
    snapshot.refresh_catalog(root)

    return IngestSummary(
        clean_rows=eq_clean + cr_clean,
        quarantined_rows=eq_bad + cr_bad,
        frozen_rows=eq_frozen + cr_frozen,
        drift_rows=eq_drift + cr_drift,
        dates=tuple(sorted(eq_dates | cr_dates)),
        failed_sleeves=tuple(sorted(failed)),
    )


def main() -> None:
    pd.set_option("display.width", 160)
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    s = run(days)
    print(
        f"ingest ok — added: {s.clean_rows} clean / {s.quarantined_rows} quarantined, "
        f"frozen (already stored): {s.frozen_rows}, vendor drift on frozen rows: {s.drift_rows}, "
        f"dates touched: {len(s.dates)}"
        + (
            f" | PARTIAL — sleeves failed: {', '.join(s.failed_sleeves)}"
            if s.failed_sleeves
            else ""
        )
    )


if __name__ == "__main__":
    main()
