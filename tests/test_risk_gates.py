from datetime import date

from vega.regime.regime import RegimeState
from vega.risk import gates
from vega.risk.gates import EarningsFact


def _regime(composite: str) -> RegimeState:
    return RegimeState(
        as_of="2026-07-01",
        trend="risk_on",
        vix=15.0,
        vix_band="calm",
        breadth_pct=60.0,
        crypto_fg=50,
        composite=composite,
    )


def test_regime_gate_blocks_only_risk_off() -> None:
    assert gates.regime_gate("AAPL", _regime("risk_off")) is not None
    assert gates.regime_gate("AAPL", _regime("caution")) is None
    assert gates.regime_gate("AAPL", _regime("risk_on")) is None


def test_macro_gate_blocks_t_minus_one_and_t_of_a_real_committed_event() -> None:
    # 2026-07-14 CPI release is in the committed macro-v1.csv
    assert gates.macro_gate("AAPL", date(2026, 7, 13)) is not None  # T-1
    assert gates.macro_gate("AAPL", date(2026, 7, 14)) is not None  # T
    assert gates.macro_gate("AAPL", date(2026, 7, 12)) is None  # T-2, clear
    assert gates.macro_gate("AAPL", date(2026, 7, 15)) is None  # T+1, clear


def test_earnings_gate_rejects_when_earnings_falls_in_horizon() -> None:
    fact = EarningsFact("date", "2026-07-10")
    rejection = gates.earnings_gate("AAPL", date(2026, 7, 1), 15, fact)
    assert rejection is not None and rejection.reason == "earnings_in_horizon"


def test_earnings_gate_clear_when_earnings_is_beyond_horizon() -> None:
    fact = EarningsFact("date", "2026-09-01")
    assert gates.earnings_gate("AAPL", date(2026, 7, 1), 15, fact) is None


def test_earnings_gate_fails_closed_when_lookup_unavailable() -> None:
    rejection = gates.earnings_gate("AAPL", date(2026, 7, 1), 15, EarningsFact("unavailable"))
    assert rejection is not None and rejection.reason == "earnings_unknown"


def test_earnings_gate_clear_for_no_earnings_concept() -> None:
    assert gates.earnings_gate("BTC", date(2026, 7, 1), 15, EarningsFact("none")) is None


def test_earnings_fact_lookup_never_hits_network_for_crypto() -> None:
    # crypto has no earnings concept — lookup short-circuits before any vendor call
    assert EarningsFact.lookup("BTC", "crypto") == EarningsFact("none")


def test_check_all_gates_regime_wins_over_macro_and_earnings() -> None:
    rejection = gates.check_all_gates(
        "AAPL", date(2026, 7, 14), 15, _regime("risk_off"), EarningsFact("date", "2026-07-15")
    )
    assert rejection is not None and rejection.reason == "regime_risk_off"
