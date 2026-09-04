#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/backend"
export DATABASE_URL="${DATABASE_URL:-sqlite+aiosqlite:///$ROOT/backend/razorguard.db}"
export REDIS_URL="${REDIS_URL:-}"
cd "$ROOT/backend"
exec uvicorn app.main:app --reload --port 8000
