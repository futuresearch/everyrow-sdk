#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)/scripts"
TASK_FILE="$HOME/.everyrow/task.json"

# Setup: ensure clean state
mkdir -p "$HOME/.everyrow"
rm -f "$TASK_FILE"

# Test 1: No active task — just model info
OUTPUT=$(echo '{"model":{"display_name":"Opus"},"context_window":{"used_percentage":12.5}}' | bash "$SCRIPT_DIR/everyrow-statusline.sh")
echo "$OUTPUT" | grep -q "Opus" || { echo "FAIL: should show model"; exit 1; }
echo "$OUTPUT" | grep -q "everyrow" && { echo "FAIL: should not show everyrow line"; exit 1; }
echo "PASS: no task"

# Test 2: Running task — shows progress bar
echo '{"task_id":"abc","session_url":"https://example.com","total":50,"completed":25,"failed":0,"running":10,"status":"running","started_at":'"$(( $(date +%s) - 30 ))"'}' > "$TASK_FILE"
OUTPUT=$(echo '{"model":{"display_name":"Opus"},"context_window":{"used_percentage":12.5}}' | bash "$SCRIPT_DIR/everyrow-statusline.sh")
echo "$OUTPUT" | grep -q "25/50" || { echo "FAIL: should show 25/50"; exit 1; }
echo "$OUTPUT" | grep -q "everyrow" || { echo "FAIL: should show everyrow label"; exit 1; }
echo "PASS: running task"

# Test 3: Running task with failures — shows failure count
echo '{"task_id":"abc","session_url":"https://example.com","total":50,"completed":25,"failed":3,"running":10,"status":"running","started_at":'"$(( $(date +%s) - 30 ))"'}' > "$TASK_FILE"
OUTPUT=$(echo '{"model":{"display_name":"Opus"},"context_window":{"used_percentage":12.5}}' | bash "$SCRIPT_DIR/everyrow-statusline.sh")
echo "$OUTPUT" | grep -q "3 failed" || { echo "FAIL: should show failure count"; exit 1; }
echo "PASS: running task with failures"

# Test 4: Completed task — shows done
echo '{"task_id":"abc","session_url":"https://example.com","total":50,"completed":50,"failed":0,"running":0,"status":"completed","started_at":'"$(( $(date +%s) - 120 ))"'}' > "$TASK_FILE"
OUTPUT=$(echo '{"model":{"display_name":"Opus"},"context_window":{"used_percentage":12.5}}' | bash "$SCRIPT_DIR/everyrow-statusline.sh")
echo "$OUTPUT" | grep -q "done" || { echo "FAIL: should show done"; exit 1; }
echo "PASS: completed task"

# Cleanup
rm -f "$TASK_FILE"
echo "ALL PASS: statusline"
