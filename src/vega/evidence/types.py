"""Filing-evidence types — the point-in-time contract.

Every record here carries `filed_at`: the timestamp the SEC itself assigned
when it accepted the document. That field is the whole reason filings are
admissible where free news and free fundamentals are not (STRATEGY.md
Pillar 2) — it is assigned by the regulator, never revised, and it makes the
"what was knowable at decision time" question answerable instead of assumed.

Nothing in this package may be consumed without filtering on `filed_at`.
"""

from __future__ import annotations

from dataclasses import dataclass

# 8-K item codes that can change expected future cash flows, and therefore mean
# a drop is INFORMATION rather than a liquidity shock. Deliberately a data
# constant rather than a rule buried in a signal: WI-230 gates on it, WI-227
# reports against it, and a disagreement about the list should be a diff here.
VALUE_CHANGING_ITEMS = frozenset(
    {
        "1.01",  # entry into a material definitive agreement
        "1.02",  # termination of a material definitive agreement
        "1.03",  # bankruptcy or receivership
        "2.01",  # completion of acquisition or disposition of assets
        "2.02",  # results of operations and financial condition
        "2.03",  # creation of a material direct financial obligation
        "2.04",  # triggering events that accelerate an obligation
        "2.05",  # costs associated with exit or disposal activities
        "2.06",  # material impairments
        "3.01",  # notice of delisting / failure to satisfy a listing rule
        "4.01",  # changes in registrant's certifying accountant
        "4.02",  # non-reliance on previously issued financial statements
        "5.02",  # departure/election of directors or principal officers
        "7.01",  # Regulation FD disclosure
        "8.01",  # other events (the catch-all issuers use for real news)
    }
)

# Filed constantly, and administrative: their presence must NOT veto a setup.
# Listed explicitly so "not value-changing" is an assertion we can point at.
ROUTINE_ITEMS = frozenset(
    {
        "5.03",  # amendments to articles/bylaws; change in fiscal year
        "5.07",  # submission of matters to a vote of security holders
        "9.01",  # financial statements and exhibits
    }
)

# Form 4 transaction codes. Only P is a decision to SPEND money on the open
# market; A (grant), M (option exercise) and F (shares withheld for tax) are
# compensation mechanics that carry no view, and S is a sale. The entire
# informational value of the feed is this filter.
OPEN_MARKET_PURCHASE = "P"


@dataclass(frozen=True)
class Filing8K:
    """One 8-K, with the item codes the issuer itself reported on the cover."""

    symbol: str
    cik: int
    accession: str
    filed_at: str  # ISO-8601 acceptance datetime — the point-in-time basis
    filed_date: str  # YYYY-MM-DD, for window arithmetic
    report_date: str  # the event date the issuer reported (may precede filed_date)
    items: tuple[str, ...]

    @property
    def is_value_changing(self) -> bool:
        """True when ANY reported item can move expected cash flows.

        An 8-K reporting only routine items is not evidence that a drop was
        informational. An 8-K reporting NO items at all is treated as
        value-changing: an unparseable cover page is not a safety signal.
        """
        if not self.items:
            return True
        return any(code in VALUE_CHANGING_ITEMS for code in self.items)


@dataclass(frozen=True)
class InsiderBuy:
    """One open-market purchase (Form 4, non-derivative, transaction code P)."""

    symbol: str
    cik: int
    accession: str
    filed_at: str  # when it became knowable — NOT transaction_date
    transaction_date: str  # when the insider actually bought (up to 2 business days earlier)
    owner: str
    is_officer: bool
    is_director: bool
    shares: float
    price: float

    @property
    def value_usd(self) -> float:
        return self.shares * self.price


@dataclass(frozen=True)
class FilingEvidence:
    """Everything EDGAR knew about a symbol inside one [start, end] filing window.

    Both tuples are filtered on `filed_at`, so a consumer cannot accidentally
    read a document that did not exist at its decision time.
    """

    symbol: str
    cik: int
    start: str
    end: str
    filings_8k: tuple[Filing8K, ...]
    insider_buys: tuple[InsiderBuy, ...]

    @property
    def has_value_changing_8k(self) -> bool:
        return any(f.is_value_changing for f in self.filings_8k)

    @property
    def insider_buy_value_usd(self) -> float:
        return sum(b.value_usd for b in self.insider_buys)

    @property
    def distinct_insider_buyers(self) -> int:
        """Cluster size. One officer topping up is weak; four buying is a signal."""
        return len({b.owner for b in self.insider_buys})


class EdgarError(RuntimeError):
    """EDGAR was unreachable, throttled, or returned something we do not understand.

    Raised rather than degraded ON PURPOSE. An empty result and a refused
    request are indistinguishable downstream, and "no filings" is exactly the
    condition WI-230 treats as permission to trade — so a swallowed error
    would manufacture the signal it was supposed to verify.
    """
