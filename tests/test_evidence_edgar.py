"""EDGAR evidence client — the filter, the point-in-time bound, and the refusals.

Fully offline: every test either seeds the on-disk cache (so the client makes
no request at all) or installs a fake transport. A test that reached the SEC
would be both rude and non-deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vega.evidence import edgar
from vega.evidence.types import EdgarError, Filing8K

AS_OF = "2026-08-28"
CIK = 320193

TICKERS = {"0": {"cik_str": CIK, "ticker": "AAPL", "title": "Apple Inc."}}


def _submissions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cols = (
        "accessionNumber",
        "form",
        "filingDate",
        "acceptanceDateTime",
        "items",
        "reportDate",
        "primaryDocument",
    )
    return {"filings": {"recent": {c: [r.get(c, "") for r in rows] for c in cols}}}


def _row(form: str, accession: str, filed: str, **kw: Any) -> dict[str, Any]:
    return {
        "accessionNumber": accession,
        "form": form,
        "filingDate": filed,
        "acceptanceDateTime": f"{filed}T16:30:00.000Z",
        "items": kw.get("items", ""),
        "reportDate": kw.get("reportDate", filed),
        "primaryDocument": kw.get("primaryDocument", "doc.xml"),
    }


def _form4(transactions: str, owner: str = "COOK TIMOTHY D", derivative: str = "") -> str:
    return f"""<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerCik>{CIK}</issuerCik><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>{owner}</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isOfficer>1</isOfficer><isDirector>0</isDirector>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>{transactions}</nonDerivativeTable>
  <derivativeTable>{derivative}</derivativeTable>
</ownershipDocument>"""


def _txn(
    code: str,
    shares: str = "1000",
    price: str = "150.00",
    acquired: str = "A",
    date: str = "2026-08-20",
) -> str:
    return f"""
    <nonDerivativeTransaction>
      <transactionDate><value>{date}</value></transactionDate>
      <transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>{shares}</value></transactionShares>
        <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>{acquired}</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>"""


def _seed(root: Path, name: str, payload: Any, as_of: str = AS_OF) -> None:
    path = root / "edgar" / as_of / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture(autouse=True)
def _ua(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEGA_SEC_USER_AGENT", "Vega tests test@example.com")


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("network call attempted — the cache should have served this")

    monkeypatch.setattr(edgar.requests, "get", _boom)


class _Resp:
    def __init__(self, body: Any, status: int = 200) -> None:
        self.status_code = status
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)
        self.content = self.text.encode()

    def json(self) -> Any:
        return self._body


# --------------------------------------------------------------- the P filter


def test_only_open_market_purchases_survive(tmp_path: Path, no_network: None) -> None:
    """A, M, S and F are compensation mechanics or sales — only P is a decision to buy."""
    _seed(tmp_path, "company_tickers", TICKERS)
    _seed(
        tmp_path,
        f"submissions_{CIK:010d}",
        _submissions([_row("4", "0000320193-26-000101", "2026-08-22")]),
    )
    _seed(
        tmp_path,
        "form4_0000320193-26-000101",
        _form4(
            _txn("A")
            + _txn("M")
            + _txn("S", acquired="D")
            + _txn("F", acquired="D")
            + _txn("P", shares="500", price="200.00")
        ),
    )

    ev = edgar.evidence_for("AAPL", "2026-08-01", "2026-08-28", AS_OF, tmp_path)

    assert len(ev.insider_buys) == 1
    assert ev.insider_buys[0].shares == 500.0
    assert ev.insider_buy_value_usd == 100_000.0
    assert ev.insider_buys[0].is_officer is True


def test_derivative_purchase_is_not_an_open_market_buy(tmp_path: Path, no_network: None) -> None:
    """A code-P row in the derivative table is an option purchase — a different claim."""
    _seed(tmp_path, "company_tickers", TICKERS)
    _seed(
        tmp_path,
        f"submissions_{CIK:010d}",
        _submissions([_row("4", "0000320193-26-000102", "2026-08-22")]),
    )
    _seed(
        tmp_path,
        "form4_0000320193-26-000102",
        _form4(
            "", derivative=_txn("P").replace("nonDerivativeTransaction", "derivativeTransaction")
        ),
    )

    ev = edgar.evidence_for("AAPL", "2026-08-01", "2026-08-28", AS_OF, tmp_path)
    assert ev.insider_buys == ()


def test_purchase_marked_disposed_is_rejected(tmp_path: Path, no_network: None) -> None:
    _seed(tmp_path, "company_tickers", TICKERS)
    _seed(
        tmp_path,
        f"submissions_{CIK:010d}",
        _submissions([_row("4", "0000320193-26-000103", "2026-08-22")]),
    )
    _seed(tmp_path, "form4_0000320193-26-000103", _form4(_txn("P", acquired="D")))

    ev = edgar.evidence_for("AAPL", "2026-08-01", "2026-08-28", AS_OF, tmp_path)
    assert ev.insider_buys == ()


# ------------------------------------------------------- the point-in-time bound


def test_window_filters_on_filing_date_not_transaction_date(
    tmp_path: Path, no_network: None
) -> None:
    """A purchase made inside the window but ACCEPTED after it was not knowable in it.

    This is the whole reason the client exists: Form 4 is due two business days
    after the trade, so transaction date and knowability routinely disagree.
    """
    _seed(tmp_path, "company_tickers", TICKERS)
    _seed(
        tmp_path,
        f"submissions_{CIK:010d}",
        _submissions([_row("4", "0000320193-26-000104", "2026-08-31")]),
    )
    _seed(tmp_path, "form4_0000320193-26-000104", _form4(_txn("P", date="2026-08-20")))

    ev = edgar.evidence_for("AAPL", "2026-08-01", "2026-08-28", AS_OF, tmp_path)
    assert ev.insider_buys == ()


def test_filed_at_is_the_acceptance_timestamp(tmp_path: Path, no_network: None) -> None:
    _seed(tmp_path, "company_tickers", TICKERS)
    _seed(
        tmp_path,
        f"submissions_{CIK:010d}",
        _submissions([_row("4", "0000320193-26-000105", "2026-08-22")]),
    )
    _seed(tmp_path, "form4_0000320193-26-000105", _form4(_txn("P", date="2026-08-20")))

    buy = edgar.evidence_for("AAPL", "2026-08-01", "2026-08-28", AS_OF, tmp_path).insider_buys[0]
    assert buy.filed_at == "2026-08-22T16:30:00.000Z"
    assert buy.transaction_date == "2026-08-20"


# ----------------------------------------------------------------- 8-K item codes


def test_8k_item_codes_are_parsed_from_the_cover(tmp_path: Path, no_network: None) -> None:
    _seed(tmp_path, "company_tickers", TICKERS)
    _seed(
        tmp_path,
        f"submissions_{CIK:010d}",
        _submissions(
            [
                _row("8-K", "0000320193-26-000201", "2026-08-05", items="2.02,9.01"),
                _row("8-K", "0000320193-26-000202", "2026-08-06", items="5.07,9.01"),
            ]
        ),
    )

    ev = edgar.evidence_for("AAPL", "2026-08-01", "2026-08-28", AS_OF, tmp_path)
    assert [f.items for f in ev.filings_8k] == [("2.02", "9.01"), ("5.07", "9.01")]
    assert ev.filings_8k[0].is_value_changing is True  # 2.02 results
    assert ev.filings_8k[1].is_value_changing is False  # vote results + exhibits only
    assert ev.has_value_changing_8k is True


def test_8k_with_no_parseable_items_counts_as_value_changing() -> None:
    """An unreadable cover page is not permission to trade."""
    blank = Filing8K(
        "AAPL", CIK, "0000320193-26-000203", "2026-08-05T16:00:00Z", "2026-08-05", "2026-08-05", ()
    )
    assert blank.is_value_changing is True


def test_distinct_buyers_counts_the_cluster(tmp_path: Path, no_network: None) -> None:
    _seed(tmp_path, "company_tickers", TICKERS)
    _seed(
        tmp_path,
        f"submissions_{CIK:010d}",
        _submissions(
            [
                _row("4", "0000320193-26-000106", "2026-08-22"),
                _row("4", "0000320193-26-000107", "2026-08-23"),
            ]
        ),
    )
    _seed(tmp_path, "form4_0000320193-26-000106", _form4(_txn("P"), owner="COOK TIMOTHY D"))
    _seed(tmp_path, "form4_0000320193-26-000107", _form4(_txn("P"), owner="PAREKH LUCA"))

    ev = edgar.evidence_for("AAPL", "2026-08-01", "2026-08-28", AS_OF, tmp_path)
    assert ev.distinct_insider_buyers == 2


# --------------------------------------------------------------- network discipline


def test_first_run_fetches_then_second_run_makes_zero_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = {
        edgar.TICKER_MAP_URL: TICKERS,
        edgar.SUBMISSIONS_URL.format(cik=CIK): _submissions(
            [_row("8-K", "0000320193-26-000204", "2026-08-05", items="8.01")]
        ),
    }
    calls: list[str] = []

    def _fake(url: str, **_k: Any) -> _Resp:
        calls.append(url)
        return _Resp(bodies[url])

    monkeypatch.setattr(edgar.requests, "get", _fake)
    first = edgar.evidence_for("AAPL", "2026-08-01", "2026-08-28", AS_OF, tmp_path)
    assert len(calls) == 2

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("second run must be served entirely from cache")

    monkeypatch.setattr(edgar.requests, "get", _boom)
    second = edgar.evidence_for("AAPL", "2026-08-01", "2026-08-28", AS_OF, tmp_path)
    assert second == first


def test_response_is_snapshotted_before_it_is_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(edgar.requests, "get", lambda url, **_k: _Resp(TICKERS))
    edgar.cik_for("AAPL", AS_OF, tmp_path)
    snapshots = list((tmp_path / "snapshots" / "sec_edgar").rglob("company_tickers_*.json"))
    assert len(snapshots) == 1


# -------------------------------------------------------------------- refusals


@pytest.mark.parametrize("status", [403, 429])
def test_throttled_raises_instead_of_returning_no_filings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """The failure mode that would manufacture the signal it is meant to verify."""
    monkeypatch.setattr(edgar.requests, "get", lambda url, **_k: _Resp("", status))
    with pytest.raises(EdgarError, match="refused"):
        edgar.evidence_for("AAPL", "2026-08-01", "2026-08-28", AS_OF, tmp_path)


def test_schema_change_raises(tmp_path: Path, no_network: None) -> None:
    _seed(tmp_path, "company_tickers", TICKERS)
    broken = _submissions([_row("8-K", "0000320193-26-000205", "2026-08-05")])
    del broken["filings"]["recent"]["acceptanceDateTime"]
    _seed(tmp_path, f"submissions_{CIK:010d}", broken)
    with pytest.raises(EdgarError, match="schema changed"):
        edgar.evidence_for("AAPL", "2026-08-01", "2026-08-28", AS_OF, tmp_path)


def test_unmapped_ticker_raises_rather_than_returning_clean(
    tmp_path: Path, no_network: None
) -> None:
    _seed(tmp_path, "company_tickers", TICKERS)
    with pytest.raises(EdgarError, match="no SEC CIK"):
        edgar.evidence_for("XLK", "2026-08-01", "2026-08-28", AS_OF, tmp_path)


def test_missing_user_agent_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VEGA_SEC_USER_AGENT", raising=False)
    monkeypatch.setattr(edgar.requests, "get", lambda url, **_k: _Resp(TICKERS))
    with pytest.raises(EdgarError, match="VEGA_SEC_USER_AGENT"):
        edgar.cik_for("AAPL", AS_OF, tmp_path)


def test_inverted_window_raises(tmp_path: Path, no_network: None) -> None:
    with pytest.raises(EdgarError, match="after end"):
        edgar.evidence_for("AAPL", "2026-08-28", "2026-08-01", AS_OF, tmp_path)


def test_malformed_accession_raises(tmp_path: Path, no_network: None) -> None:
    _seed(tmp_path, "company_tickers", TICKERS)
    _seed(
        tmp_path,
        f"submissions_{CIK:010d}",
        _submissions([_row("8-K", "not-an-accession", "2026-08-05", items="8.01")]),
    )
    with pytest.raises(EdgarError, match="malformed accession"):
        edgar.evidence_for("AAPL", "2026-08-01", "2026-08-28", AS_OF, tmp_path)
