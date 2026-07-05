# Vega — Tech Stack

- **Python 3.12** pinned via `.python-version` (system Python 3.9 is EOL and incompatible with the modern quant stack).
- **uv** — interpreter, venv, and lockfile management in one tool. `uv sync --frozen` in the verify gate guarantees the environment matches `uv.lock` exactly. Decision locked with Jose 2026-07-05.
- **pytest** (tests, offline-only — no network in the test suite), **ruff** (lint + format + `S` security rules), **mypy strict** (on `src/`), **pip-audit** (dependency vulns).
- Runtime dependencies arrive per work item (WI-058 adds pandas, pyarrow, duckdb, yfinance, alpaca-py, requests). The scaffold is intentionally dependency-free.
- Secrets: gitignored `.env` (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`); template in `.env.example`.
