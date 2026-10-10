#!/bin/sh
# Ingest the three sample documents (one-command seed). Idempotent: re-running yields
# duplicates (HTTP 200), never a second document. Usage: scripts/seed.sh [base_url]
set -e
BASE="${1:-http://localhost:8000}"
DIR="$(dirname "$0")/../Assessment_Documents"
for name in medical_certificate receipt referral_letter; do
  printf '%-22s ' "$name.json:"
  curl -sS -o /dev/null -w '%{http_code}\n' -F "file=@$DIR/$name.json" "$BASE/documents"
done
