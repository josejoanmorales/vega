"""Event calendar: versioned macro artifact + on-demand per-symbol earnings.

Macro dates (FOMC/CPI) live in a committed, append-only versioned CSV — same
pattern as the universe artifact. Earnings are looked up per symbol on demand
(only for held/candidate symbols, never the whole universe).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import yfinance as yf

DEFAULT_ARTIFACT = Path("data/calendar/macro-v1.csv")


@dataclass(frozen=True)
class MacroEvent:
    date: str
    event: str


def load_macro_calendar(path: Path = DEFAULT_ARTIFACT) -> list[MacroEvent]:
    with path.open(newline="") as fh:
        rows = [line for line in fh if not line.startswith("#")]
    events = [MacroEvent(date=r["date"], event=r["event"]) for r in csv.DictReader(rows)]
    if not events:
        raise ValueError(f"macro calendar {path} is empty")
    for e in events:
        date.fromisoformat(e.date)  # malformed artifact must fail loudly
    return events


def macro_events_within(
    on: date, days_ahead: int, path: Path = DEFAULT_ARTIFACT
) -> list[MacroEvent]:
    horizon = on + timedelta(days=days_ahead)
    return [e for e in load_macro_calendar(path) if on <= date.fromisoformat(e.date) <= horizon]


def in_macro_window(on: date, days_before: int = 1, path: Path = DEFAULT_ARTIFACT) -> bool:
    """True when a scheduled macro event lands within the next `days_before` days —
    the entry-gating rule (STRATEGY.md §5: no new entries just ahead of FOMC/CPI)."""
    return bool(macro_events_within(on, days_before, path))


def next_earnings(symbol: str) -> str | None:
    """Next scheduled earnings date via yfinance, ISO string, None if unknown."""
    try:
        cal = yf.Ticker(symbol).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            return str(min(dates))
    except Exception:  # noqa: BLE001 — vendor call; unknown is an acceptable answer
        return None
    return None
