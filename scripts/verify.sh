#!/usr/bin/env bash
#
# Local verification gate for Vega.
# Caral's daily-build runner executes this to decide a change is good.
# It MUST exit non-zero on any failure. No interactive steps.
#
# Security posture is layered (factory convention):
#   Layer 1 (here, every build): dependency audit + security lint (ruff S rules) + secret scan.
#   Layers 2 & 3: deeper review / threat modeling, activated on higher-risk stories
#   (broker-key handling in WI-061 qualifies).
#
set -euo pipefail

cd "$(dirname "$0")/.."

echo "▶ sync (frozen lockfile)"
uv sync --frozen --quiet

echo "▶ typecheck"
uv run mypy

echo "▶ lint (incl. security static analysis)"
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts

echo "▶ test"
uv run pytest -q

echo "▶ security: dependency audit"
uv run pip-audit --skip-editable --progress-spinner off

echo "▶ security: secret scan"
scripts/secret-scan.sh

echo "✅ verify passed"
