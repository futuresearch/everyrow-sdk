#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)/scripts"

# Set up initial task state
echo '{"task_id":"abc-123","session_url":"https://example.com","total":50,"completed":0,"failed":0,"running":0,"status":"running","started_at":1700000000}' > /tmp/everyrow-task.json

# Test 1: Flat JSON response format
echo '{"tool_name":"mcp__everyrow__everyrow_progress","tool_input":{"task_id":"abc-123"},"tool_response":{"status":"running","completed":35,"failed":1,"running":10,"total":50},"tool_use_id":"toolu_02"}' \
  | bash "$SCRIPT_DIR/everyrow-track-progress.sh"

jq -e '.completed == 35' /tmp/everyrow-task.json || { echo "FAIL: completed"; exit 1; }
jq -e '.failed == 1' /tmp/everyrow-task.json || { echo "FAIL: failed"; exit 1; }
jq -e '.task_id == "abc-123"' /tmp/everyrow-task.json || { echo "FAIL: task_id preserved"; exit 1; }
jq -e '.started_at == 1700000000' /tmp/everyrow-task.json || { echo "FAIL: started_at preserved"; exit 1; }
echo "PASS: progress hook (flat JSON)"

# Test 2: Content-wrapped format
echo '{"task_id":"abc-123","session_url":"https://example.com","total":50,"completed":35,"failed":1,"running":10,"status":"running","started_at":1700000000}' > /tmp/everyrow-task.json
echo '{"tool_name":"mcp__everyrow__everyrow_progress","tool_input":{"task_id":"abc-123"},"tool_response":{"content":[{"type":"text","text":"{\"status\":\"running\",\"completed\":42,\"failed\":2,\"running\":5,\"total\":50}"}]},"tool_use_id":"toolu_03"}' \
  | bash "$SCRIPT_DIR/everyrow-track-progress.sh"

jq -e '.completed == 42' /tmp/everyrow-task.json || { echo "FAIL: completed (content-wrapped)"; exit 1; }
jq -e '.failed == 2' /tmp/everyrow-task.json || { echo "FAIL: failed (content-wrapped)"; exit 1; }
echo "PASS: progress hook (content-wrapped)"

# Test 3: No task file → no-op
rm -f /tmp/everyrow-task.json
echo '{"tool_name":"mcp__everyrow__everyrow_progress","tool_input":{"task_id":"abc-123"},"tool_response":{"status":"running","completed":10,"failed":0,"running":5,"total":50},"tool_use_id":"toolu_04"}' \
  | bash "$SCRIPT_DIR/everyrow-track-progress.sh"
[ ! -f /tmp/everyrow-task.json ] || { echo "FAIL: should not create file when missing"; exit 1; }
echo "PASS: progress hook (no task file)"

rm -f /tmp/everyrow-task.json
echo "ALL PASS: progress hook"
