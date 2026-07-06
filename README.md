# Vega

Evidence-based swing-trading analyst assistant — deterministic quant core, verified
(paper-traded, self-graded) track record. Strategy: [STRATEGY.md](STRATEGY.md).

## Quickstart

```bash
brew install uv          # once
uv sync                  # creates .venv from uv.lock (Python 3.12 auto-provisioned)
./scripts/verify.sh      # the full gate: typecheck, lint, tests, security
```

Secrets: `cp .env.example .env` and fill in your Alpaca keys (free account, paper keys).

Daily data ingest (fetch → validate → write-once clean store):

```bash
uv run python -m vega.data.ingest 7   # last 7 days
```

Daily pre-market routine (ingest first, then briefing):

```bash
uv run python -m vega.data.ingest 7 && uv run python -m vega.briefing
```

## Layout

- `src/vega/` — the deterministic engine (LLM never computes numbers; see `documentation/architecture.md`)
- `tests/` — offline-only test suite
- `scripts/verify.sh` — the verification gate Caral's runner executes
- `documentation/` — architecture, tech stack, glossary
