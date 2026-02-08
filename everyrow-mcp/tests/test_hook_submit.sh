#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)/scripts"
rm -f /tmp/everyrow-task.json

# Test 1: Flat JSON response format
echo '{"tool_name":"mcp__everyrow__everyrow_screen_submit","tool_input":{"task":"test"},"tool_response":{"task_id":"abc-123","session_url":"https://everyrow.io/sessions/xyz","total":50},"tool_use_id":"toolu_01"}' \
  | bash "$SCRIPT_DIR/everyrow-track-submit.sh"

jq -e '.task_id == "abc-123"' /tmp/everyrow-task.json || { echo "FAIL: task_id"; exit 1; }
jq -e '.total == 50' /tmp/everyrow-task.json || { echo "FAIL: total"; exit 1; }
jq -e '.status == "running"' /tmp/everyrow-task.json || { echo "FAIL: status"; exit 1; }
jq -e '.session_url == "https://everyrow.io/sessions/xyz"' /tmp/everyrow-task.json || { echo "FAIL: session_url"; exit 1; }
jq -e '.started_at > 0' /tmp/everyrow-task.json || { echo "FAIL: started_at"; exit 1; }
echo "PASS: submit hook (flat JSON)"

# Test 2: MCP content-wrapped format
rm -f /tmp/everyrow-task.json
echo '{"tool_name":"mcp__everyrow__everyrow_screen_submit","tool_input":{"task":"test"},"tool_response":{"content":[{"type":"text","text":"{\"task_id\":\"def-456\",\"session_url\":\"https://everyrow.io/sessions/abc\",\"total\":25}"}]},"tool_use_id":"toolu_02"}' \
  | bash "$SCRIPT_DIR/everyrow-track-submit.sh"

jq -e '.task_id == "def-456"' /tmp/everyrow-task.json || { echo "FAIL: task_id (content-wrapped)"; exit 1; }
jq -e '.total == 25' /tmp/everyrow-task.json || { echo "FAIL: total (content-wrapped)"; exit 1; }
echo "PASS: submit hook (content-wrapped)"

# Test 3: String format
rm -f /tmp/everyrow-task.json
echo '{"tool_name":"mcp__everyrow__everyrow_screen_submit","tool_input":{"task":"test"},"tool_response":"{\"task_id\":\"ghi-789\",\"session_url\":\"https://everyrow.io/sessions/def\",\"total\":10}","tool_use_id":"toolu_03"}' \
  | bash "$SCRIPT_DIR/everyrow-track-submit.sh"

jq -e '.task_id == "ghi-789"' /tmp/everyrow-task.json || { echo "FAIL: task_id (string)"; exit 1; }
jq -e '.total == 10' /tmp/everyrow-task.json || { echo "FAIL: total (string)"; exit 1; }
echo "PASS: submit hook (string format)"

rm -f /tmp/everyrow-task.json
echo "ALL PASS: submit hook"
