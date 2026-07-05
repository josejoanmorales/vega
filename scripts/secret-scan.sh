#!/usr/bin/env bash
#
# Security layer 1 — secret scan.
# Dependency-free tripwire for high-signal credential patterns committed to source.
# Fails (exit 1) on any match. Deeper scanning (gitleaks, full history) is a layer-2
# convention that activates on risk-tier stories.
#
set -euo pipefail
cd "$(dirname "$0")/.."

# High-signal patterns: private keys, cloud/provider tokens, Alpaca credentials.
PATTERNS='-----BEGIN( RSA| EC| OPENSSH| PGP)? PRIVATE KEY-----|AKIA[0-9A-Z]{16}|ghp_[0-9A-Za-z]{36}|github_pat_[0-9A-Za-z_]{22,}|xox[baprs]-[0-9A-Za-z-]{10,}|AIza[0-9A-Za-z_-]{35}|sk_live_[0-9A-Za-z]{24,}|APCA-API-(KEY|SECRET)'

matches=$(grep -rInE -e "$PATTERNS" . \
  --exclude-dir=.venv \
  --exclude-dir=.git \
  --exclude-dir=data \
  --exclude-dir=__pycache__ \
  --exclude=.env \
  --exclude=secret-scan.sh || true)

if [ -n "$matches" ]; then
  echo "❌ potential secrets detected:"
  echo "$matches"
  exit 1
fi

echo "  no secrets detected"
