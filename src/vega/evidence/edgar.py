"""SEC EDGAR evidence client — 8-K item codes and Form 4 open-market buys.

WHY THIS EXISTS. Vega's causal classifier is currently one comparison:
`close > SMA100`. That is the entire test separating "this drop is a liquidity
shock we can provide liquidity into" from "this drop is information and the
seller is right". This module fetches the cheapest data that can answer the
question directly.

WHY FILINGS ARE ADMISSIBLE WHERE NEWS IS NOT. STRATEGY.md Pillar 2 bans
news/sentiment as an entry signal (at free-API latency the move already
happened) and bans free fundamentals in backtests (they are current-state, so
they inject lookahead). Filings are neither: the SEC stamps an acceptance
datetime it never revises, so a backtest can ask "what was public at 16:00 on
2023-04-11?" and get a true answer. That is the exception, and it is an
exception to the letter of the rule in service of its spirit.

WHAT IT DOES NOT DO. Nothing here influences a recommendation. This is an
evidence source; WI-227 measures whether the evidence is worth anything, and
only WI-230 may gate a trade on it.

Network discipline: every response is written to the immutable snapshot store
BEFORE it is parsed, and a deterministic cache means re-running the same
(symbol, as_of) window makes zero requests — a labelling job over thousands of
historical trades must not re-hit a public service it does not pay for.
"""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET  # noqa: S405 — see _parse_form4 for the threat model
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from vega.common.paths import DATA_ROOT
from vega.data import snapshot
from vega.evidence.types import (
    OPEN_MARKET_PURCHASE,
    EdgarError,
    Filing8K,
    FilingEvidence,
    InsiderBuy,
)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"

TIMEOUT = 30
# SEC's published ceiling is 10 requests/second. We pace well under it: this is
# a public service we do not pay for, and a labelling run is not latency-bound.
MIN_INTERVAL_S = 0.15
# A Form 4 is a few KB and a submissions document a few MB. Anything wildly
# larger is not a document we understand, and we refuse to parse it.
MAX_BYTES = 32 * 1024 * 1024

_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_last_request_at = 0.0


def _user_agent() -> str:
    """SEC requires a declared, contactable User-Agent and blocks generic ones.

    Deliberately env-only with no baked-in default: a contact address is
    personal data that does not belong in a repository, and a placeholder
    would get the whole project rate-limited under someone else's name.
    """
    ua = os.environ.get("VEGA_SEC_USER_AGENT", "").strip()
    if not ua:
        raise EdgarError(
            "VEGA_SEC_USER_AGENT is not set. SEC requires a declared contact, e.g. "
            'VEGA_SEC_USER_AGENT="Vega research you@example.com" — see .env.example.'
        )
    return ua


def _cache_path(name: str, as_of: str, root: Path) -> Path:
    return root / "edgar" / as_of / f"{name}.json"


def _get(url: str, name: str, as_of: str, root: Path) -> Any:
    """Paced, cached, snapshotted GET. Cache hit ⇒ zero network.

    The cache is keyed by (as_of, name), not by fetch time: a historical
    labelling window is immutable, so its answer is cached forever, while a
    live run passing today's date naturally re-fetches tomorrow.
    """
    global _last_request_at
    cached = _cache_path(name, as_of, root)
    if cached.exists():
        return json.loads(cached.read_text())

    elapsed = time.monotonic() - _last_request_at
    if elapsed < MIN_INTERVAL_S:
        time.sleep(MIN_INTERVAL_S - elapsed)
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        raise EdgarError(f"EDGAR request failed for {url}: {exc}") from exc
    finally:
        _last_request_at = time.monotonic()

    if resp.status_code in (403, 429):
        raise EdgarError(
            f"EDGAR refused {url} with HTTP {resp.status_code} — throttled or User-Agent "
            "rejected. This is NOT 'no filings'; the symbol is quarantined for the day."
        )
    if resp.status_code != 200:
        raise EdgarError(f"EDGAR returned HTTP {resp.status_code} for {url}")
    if len(resp.content) > MAX_BYTES:
        raise EdgarError(f"EDGAR response for {url} exceeded {MAX_BYTES} bytes — refusing to parse")

    body: Any = resp.text if url.endswith(".xml") else resp.json()
    snapshot.snapshot_raw_json("sec_edgar", name.replace("/", "_"), body, root)
    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(".tmp")
    tmp.write_text(json.dumps(body))
    tmp.rename(cached)
    return body


def cik_for(symbol: str, as_of: str, root: Path = DATA_ROOT) -> int:
    """Resolve a universe ticker to its SEC CIK. An unmapped ticker raises.

    Silence here would be the worst possible failure: an unmapped symbol would
    return no filings, which reads downstream as "clean, nothing happened".
    """
    payload = _get(TICKER_MAP_URL, "company_tickers", as_of, root)
    if not isinstance(payload, dict):
        raise EdgarError("company_tickers.json is not an object — schema changed")
    wanted = symbol.upper().replace(".", "-")
    for row in payload.values():
        if not isinstance(row, dict) or "ticker" not in row or "cik_str" not in row:
            raise EdgarError("company_tickers.json rows lack ticker/cik_str — schema changed")
        if str(row["ticker"]).upper() == wanted:
            return int(row["cik_str"])
    raise EdgarError(
        f"{symbol} has no SEC CIK in company_tickers.json (ETF, foreign issuer, or delisted). "
        "Filing evidence is unavailable for it — that is not the same as 'no filings'."
    )


def _recent(payload: Any, symbol: str) -> list[dict[str, Any]]:
    """Flatten submissions' parallel-array layout into rows, or raise."""
    if not isinstance(payload, dict) or "filings" not in payload:
        raise EdgarError(f"submissions payload for {symbol} has no 'filings' — schema changed")
    recent = payload["filings"].get("recent")
    if not isinstance(recent, dict):
        raise EdgarError(f"submissions payload for {symbol} has no 'filings.recent'")
    required = (
        "accessionNumber",
        "form",
        "filingDate",
        "acceptanceDateTime",
        "items",
        "reportDate",
        "primaryDocument",
    )
    missing = [c for c in required if c not in recent]
    if missing:
        raise EdgarError(f"submissions for {symbol} missing columns {missing} — schema changed")
    n = len(recent["accessionNumber"])
    return [{c: recent[c][i] for c in required} for i in range(n)]


def _parse_form4(
    xml_text: str, symbol: str, cik: int, accession: str, filed_at: str
) -> list[InsiderBuy]:
    """Extract non-derivative open-market purchases from one Form 4 document.

    Only the NON-derivative table: a code-P row in the derivative table is an
    option/warrant purchase, a different instrument and a different claim.
    Both code P and an "acquired" disposition are required — belt and braces
    against a malformed cover.

    XML threat model for the S314 suppression below: the input is a document
    fetched over TLS from a single government host, already size-capped at
    MAX_BYTES, and Python's ElementTree does not resolve external entities.
    The residual risk is a hostile SEC archive, which is out of scope for a
    tool that also *trades on* what that archive says.
    """
    try:
        root_el = ET.fromstring(xml_text)  # noqa: S314 — threat model documented above
    except ET.ParseError as exc:
        raise EdgarError(f"Form 4 {accession} for {symbol} is not parseable XML: {exc}") from exc

    def _strip(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    def _value(node: ET.Element | None) -> str:
        if node is None:
            return ""
        inner = next((c for c in node if _strip(c.tag) == "value"), None)
        return ((inner if inner is not None else node).text or "").strip()

    def _child(node: ET.Element, name: str) -> ET.Element | None:
        return next((c for c in node if _strip(c.tag) == name), None)

    owner = ""
    is_officer = is_director = False
    for owner_el in root_el.iter():
        if _strip(owner_el.tag) != "reportingOwner":
            continue
        ident = _child(owner_el, "reportingOwnerId")
        if ident is not None:
            owner = _value(_child(ident, "rptOwnerName")) or owner
        rel = _child(owner_el, "reportingOwnerRelationship")
        if rel is not None:
            is_officer = is_officer or _value(_child(rel, "isOfficer")) in ("1", "true")
            is_director = is_director or _value(_child(rel, "isDirector")) in ("1", "true")

    buys: list[InsiderBuy] = []
    for table in root_el.iter():
        if _strip(table.tag) != "nonDerivativeTable":
            continue
        for txn in table:
            if _strip(txn.tag) != "nonDerivativeTransaction":
                continue
            coding = _child(txn, "transactionCoding")
            amounts = _child(txn, "transactionAmounts")
            if coding is None or amounts is None:
                continue
            if _value(_child(coding, "transactionCode")) != OPEN_MARKET_PURCHASE:
                continue
            if _value(_child(amounts, "transactionAcquiredDisposedCode")) != "A":
                continue
            shares = _value(_child(amounts, "transactionShares"))
            price = _value(_child(amounts, "transactionPricePerShare"))
            if not shares:
                continue
            buys.append(
                InsiderBuy(
                    symbol=symbol,
                    cik=cik,
                    accession=accession,
                    filed_at=filed_at,
                    transaction_date=_value(_child(txn, "transactionDate")),
                    owner=owner,
                    is_officer=is_officer,
                    is_director=is_director,
                    shares=float(shares),
                    price=float(price or 0.0),
                )
            )
    return buys


def evidence_for(
    symbol: str, start: str, end: str, as_of: str | None = None, root: Path = DATA_ROOT
) -> FilingEvidence:
    """Every 8-K and open-market insider buy whose FILING timestamp falls in [start, end].

    The window is on `filed_at`, never on the transaction or event date. That
    is the structural guarantee: a Form 4 covering a purchase made on T but
    accepted on T+2 belongs to T+2, because that is when a trader could have
    known about it.
    """
    if start > end:
        raise EdgarError(f"window start {start} is after end {end}")
    as_of = as_of or end
    cik = cik_for(symbol, as_of, root)
    payload = _get(SUBMISSIONS_URL.format(cik=cik), f"submissions_{cik:010d}", as_of, root)

    filings_8k: list[Filing8K] = []
    form4_rows: list[dict[str, Any]] = []
    for row in _recent(payload, symbol):
        filed_date = str(row["filingDate"])
        if not (start <= filed_date <= end):
            continue
        accession = str(row["accessionNumber"])
        if not _ACCESSION_RE.match(accession):
            raise EdgarError(f"malformed accession {accession!r} for {symbol} — schema changed")
        filed_at = str(row["acceptanceDateTime"]) or filed_date
        form = str(row["form"]).upper()
        if form.startswith("8-K"):
            items = tuple(c.strip() for c in str(row["items"]).split(",") if c.strip())
            filings_8k.append(
                Filing8K(
                    symbol=symbol,
                    cik=cik,
                    accession=accession,
                    filed_at=filed_at,
                    filed_date=filed_date,
                    report_date=str(row["reportDate"]),
                    items=items,
                )
            )
        elif form == "4":
            form4_rows.append({**row, "accession": accession, "filed_at": filed_at})

    insider_buys: list[InsiderBuy] = []
    for row in form4_rows:
        accession = str(row["accession"])
        document = str(row["primaryDocument"])
        if not document.endswith(".xml"):
            # The rendered XSL view, not the machine-readable original — skip it
            # rather than screen-scrape HTML that has no stable contract.
            continue
        url = ARCHIVE_URL.format(
            cik=cik, accession_nodash=accession.replace("-", ""), document=document
        )
        xml_text = _get(url, f"form4_{accession}", as_of, root)
        insider_buys.extend(
            _parse_form4(str(xml_text), symbol, cik, accession, str(row["filed_at"]))
        )

    return FilingEvidence(
        symbol=symbol,
        cik=cik,
        start=start,
        end=end,
        filings_8k=tuple(sorted(filings_8k, key=lambda f: f.filed_at)),
        insider_buys=tuple(sorted(insider_buys, key=lambda b: b.filed_at)),
    )


def main() -> None:
    """Manual probe: uv run python -m vega.evidence.edgar AAPL 2026-08-01 2026-08-28"""
    import sys

    if len(sys.argv) != 4:
        print(main.__doc__)
        raise SystemExit(2)
    symbol, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    ev = evidence_for(symbol, start, end)
    print(f"{ev.symbol} (CIK {ev.cik})  {ev.start} → {ev.end}   fetched {datetime.now(UTC):%H:%M}")
    for f in ev.filings_8k:
        flag = "VALUE-CHANGING" if f.is_value_changing else "routine"
        print(f"  8-K  {f.filed_at}  items={','.join(f.items) or '-':<16} {flag}")
    for b in ev.insider_buys:
        role = "officer" if b.is_officer else "director" if b.is_director else "10%"
        print(f"  BUY  {b.filed_at}  {b.owner:<28} {role:<9} ${b.value_usd:>12,.0f}")
    if not ev.filings_8k and not ev.insider_buys:
        print("  (no 8-K and no open-market insider purchases in the window)")


if __name__ == "__main__":
    main()
