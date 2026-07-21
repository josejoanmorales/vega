#!/usr/bin/env python3
"""Build a new versioned universe artifact: data/universe/universe-v{N}.csv.

Maintenance tooling, run manually (uv run python scripts/refresh_universe.py).
Versions are append-only — this script writes the next version, never mutates
an existing one. Sources: Wikipedia constituent lists (S&P 500, Nasdaq-100),
a curated liquid-ETF list, CoinGecko top crypto by market cap (stablecoins and
wrapped/staked duplicates excluded; Binance USDT pair required). Equities/ETFs
must pass the median-dollar-volume filter (>= $20M over the last 60 sessions).
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

TIMEOUT = 30
UA = {"User-Agent": "vega-universe-refresh/0.1"}
MIN_MEDIAN_DOLLAR_VOLUME = 20_000_000
CRYPTO_COUNT = 20

# Sleeve for risk.clusters heat-cap classification (universe-v2+, WI-084 item 8).
# Every equity/ETF not listed here defaults to us_equity_beta -- rates/commodities
# must be explicit, never guessed (mirrors risk.clusters.classify's fallback).
RATES_ETFS = {"TLT", "IEF"}
COMMODITIES_ETFS = {"GLD", "SLV", "USO", "XME"}


def _cluster_for(symbol: str, asset_class: str) -> str:
    if asset_class == "crypto":
        return "crypto_beta"
    if symbol in RATES_ETFS:
        return "rates"
    if symbol in COMMODITIES_ETFS:
        return "commodities"
    return "us_equity_beta"


ETFS = [
    ("SPY", "SPDR S&P 500"),
    ("QQQ", "Invesco Nasdaq-100"),
    ("IWM", "iShares Russell 2000"),
    ("DIA", "SPDR Dow Jones"),
    ("XLK", "Technology Select SPDR"),
    ("XLF", "Financial Select SPDR"),
    ("XLE", "Energy Select SPDR"),
    ("XLV", "Health Care Select SPDR"),
    ("XLI", "Industrial Select SPDR"),
    ("XLP", "Consumer Staples SPDR"),
    ("XLY", "Consumer Discretionary SPDR"),
    ("XLU", "Utilities Select SPDR"),
    ("XLB", "Materials Select SPDR"),
    ("XLRE", "Real Estate Select SPDR"),
    ("XLC", "Communication Services SPDR"),
    ("SMH", "VanEck Semiconductor"),
    ("XBI", "SPDR S&P Biotech"),
    ("KRE", "SPDR S&P Regional Banking"),
    ("GLD", "SPDR Gold Shares"),
    ("SLV", "iShares Silver"),
    ("USO", "United States Oil"),
    ("TLT", "iShares 20+ Year Treasury"),
    ("IEF", "iShares 7-10 Year Treasury"),
    ("HYG", "iShares High Yield Corp"),
    ("LQD", "iShares Investment Grade Corp"),
    ("EEM", "iShares MSCI Emerging Markets"),
    ("EFA", "iShares MSCI EAFE"),
    ("VNQ", "Vanguard Real Estate"),
    ("XME", "SPDR S&P Metals & Mining"),
    ("ITB", "iShares U.S. Home Construction"),
]

STABLE_OR_WRAPPED = {
    "usdt",
    "usdc",
    "dai",
    "fdusd",
    "usds",
    "tusd",
    "usde",
    "pyusd",
    "usdp",
    "usd1",
    "wbtc",
    "weth",
    "wsteth",
    "steth",
    "weeth",
    "cbbtc",
    "cbeth",
    "reth",
    "usdtb",
    "bsc-usd",
}


def _wiki_table(url: str, symbol_col: str, name_col: str) -> list[tuple[str, str]]:
    html = requests.get(url, timeout=TIMEOUT, headers=UA).text
    for table in pd.read_html(io.StringIO(html)):
        if symbol_col in table.columns and name_col in table.columns:
            return [
                (str(r[symbol_col]).strip(), str(r[name_col]).strip()) for _, r in table.iterrows()
            ]
    raise RuntimeError(f"no table with columns {symbol_col}/{name_col} at {url}")


def equities() -> dict[str, str]:
    sp500 = _wiki_table(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol", "Security"
    )
    ndx = _wiki_table("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker", "Company")
    merged: dict[str, str] = {}
    for sym, name in sp500 + ndx:
        merged[sym.replace(".", "-")] = name  # Yahoo notation (BRK.B -> BRK-B)
    return merged


def crypto() -> list[tuple[str, str, str, str]]:
    markets = requests.get(
        "https://api.coingecko.com/api/v3/coins/markets",
        params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": "60", "page": "1"},
        timeout=TIMEOUT,
        headers=UA,
    ).json()
    listed = {
        s["symbol"]
        for s in requests.get(
            "https://data-api.binance.vision/api/v3/exchangeInfo", timeout=TIMEOUT
        ).json()["symbols"]
        if s["status"] == "TRADING"
    }
    picked: list[tuple[str, str, str, str]] = []
    for coin in markets:
        sym = str(coin["symbol"]).lower()
        pair = f"{sym.upper()}USDT"
        if sym in STABLE_OR_WRAPPED or pair not in listed:
            continue
        picked.append((sym.upper(), str(coin["name"]), str(coin["id"]), pair))
        if len(picked) == CRYPTO_COUNT:
            break
    return picked


def liquid(symbols: dict[str, str]) -> dict[str, str]:
    """Keep equities/ETFs whose median daily dollar volume (60 sessions) clears the bar."""
    prices = yf.download(
        tickers=" ".join(symbols),
        period="6mo",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    kept: dict[str, str] = {}
    for sym, name in symbols.items():
        try:
            sub = prices[sym].dropna(subset=["Close"]).tail(60)
        except KeyError:
            continue
        if len(sub) < 40:
            continue  # too little history to judge liquidity honestly
        if float((sub["Close"] * sub["Volume"]).median()) >= MIN_MEDIAN_DOLLAR_VOLUME:
            kept[sym] = name
    return kept


def main() -> None:
    out_dir = Path("data/universe")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("universe-v*.csv"))
    version = 1 + max(
        (int(m.group(1)) for p in existing if (m := re.search(r"v(\d+)", p.name))), default=0
    )
    out = out_dir / f"universe-v{version}.csv"

    eq = liquid(equities())
    etf = liquid(dict(ETFS))
    cr = crypto()

    lines = [
        f"# vega universe v{version} — built {pd.Timestamp.now('UTC'):%Y-%m-%d}",
        "# equities: S&P 500 + Nasdaq-100 (Wikipedia constituent snapshot, Yahoo symbol notation)",
        "# etfs: curated liquid list",
        f"# filter (equity/etf): median 60-session dollar volume >= ${MIN_MEDIAN_DOLLAR_VOLUME:,}",
        f"# crypto: CoinGecko top-{CRYPTO_COUNT} mcap, ex stable/wrapped, Binance USDT pair",
        "# cluster: risk.clusters heat-cap sleeve (us_equity_beta|rates|commodities|crypto_beta)",
        "symbol,asset_class,name,coingecko_id,binance_symbol,cluster",
    ]
    for sym, name in sorted(eq.items()):
        lines.append(f'{sym},equity,"{name}",,,{_cluster_for(sym, "equity")}')
    for sym, name in sorted(etf.items()):
        lines.append(f'{sym},etf,"{name}",,,{_cluster_for(sym, "etf")}')
    for sym, name, cid, pair in cr:
        lines.append(f'{sym},crypto,"{name}",{cid},{pair},crypto_beta')
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(eq)} equities, {len(etf)} etfs, {len(cr)} crypto")


if __name__ == "__main__":
    sys.exit(main())
