#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)/scripts"
TASK_FILE="$HOME/.everyrow/task.json"

# Setup: ensure clean state
mkdir -p "$HOME/.everyrow"
rm -f "$TASK_FILE"

# Test 1: Cleans up task file and sends notification
echo '{"task_id":"abc-123","session_url":"https://example.com","total":50,"completed":49,"failed":1,"running":0,"status":"completed","started_at":'"$(( $(date +%s) - 120 ))"'}' > "$TASK_FILE"

echo '{"tool_name":"mcp__everyrow__everyrow_results","tool_input":{"task_id":"abc-123"},"tool_response":{"output_file":"/tmp/results.csv","rows":49},"tool_use_id":"toolu_03"}' \
  | bash "$SCRIPT_DIR/everyrow-track-results.sh"

[ ! -f "$TASK_FILE" ] || { echo "FAIL: task file should be deleted"; exit 1; }
echo "PASS: results hook cleanup"

# Test 2: No-op when no task file
bash "$SCRIPT_DIR/everyrow-track-results.sh" < /dev/null
echo "PASS: results hook (no task file)"

echo "ALL PASS: results hook"
