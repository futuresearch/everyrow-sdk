#!/usr/bin/env bash
# Launch Claude Code with everyrow plugin pointed at local engine.
# Usage: ./everyrow-sdk/dev-claude.sh

set -euo pipefail
cd "$(dirname "$0")/.."

ANON_KEY=$(grep SUPABASE_ANON_KEY= cohort/engine/.env | head -1 | cut -d= -f2)
JWT=$(curl -s -X POST "http://localhost:53000/auth/v1/token?grant_type=password" \
  -H "apikey:$ANON_KEY" \
  -H "Content-Type: application/json" \
  -d '{"email":"example@example.com","password":"example@example.com"}' | jq -r '.access_token')

if [ -z "$JWT" ] || [ "$JWT" = "null" ]; then
  echo "Failed to get JWT. Is supabase running? (docker ps | grep kong)" >&2
  exit 1
fi

export EVERYROW_API_URL=http://localhost:8000/api/v0
export EVERYROW_API_KEY="$JWT"

exec claude --plugin-dir "$(dirname "$0")" "$@"
