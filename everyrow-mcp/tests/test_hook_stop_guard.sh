#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)/scripts"

# Test 1: Blocks when task is running
echo '{"task_id":"abc-123","status":"running","total":50,"completed":10}' > /tmp/everyrow-task.json
RESULT=$(echo '{"stop_hook_active": false}' | bash "$SCRIPT_DIR/everyrow-stop-guard.sh")
echo "$RESULT" | jq -e '.decision == "block"' || { echo "FAIL: should block"; exit 1; }
echo "$RESULT" | jq -e '.reason | contains("abc-123")' || { echo "FAIL: reason should contain task_id"; exit 1; }
echo "PASS: blocks when running"

# Test 2: Allows when stop_hook_active (prevent infinite loop)
RESULT=$(echo '{"stop_hook_active": true}' | bash "$SCRIPT_DIR/everyrow-stop-guard.sh")
[ -z "$RESULT" ] || { echo "FAIL: should produce no output when allowing"; exit 1; }
echo "PASS: allows when stop_hook_active"

# Test 3: Allows when no task file
rm -f /tmp/everyrow-task.json
RESULT=$(echo '{"stop_hook_active": false}' | bash "$SCRIPT_DIR/everyrow-stop-guard.sh")
[ -z "$RESULT" ] || { echo "FAIL: should allow when no task"; exit 1; }
echo "PASS: allows when no task"

# Test 4: Allows when task is completed
echo '{"task_id":"abc-123","status":"completed","total":50,"completed":50}' > /tmp/everyrow-task.json
RESULT=$(echo '{"stop_hook_active": false}' | bash "$SCRIPT_DIR/everyrow-stop-guard.sh")
[ -z "$RESULT" ] || { echo "FAIL: should allow when completed"; exit 1; }
echo "PASS: allows when completed"

rm -f /tmp/everyrow-task.json
echo "ALL PASS: stop guard"
