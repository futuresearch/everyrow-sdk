#!/bin/bash
set -e

# Check for jq dependency
command -v jq >/dev/null 2>&1 || { echo "jq required" >&2; exit 1; }

INPUT=$(cat)
TASK_FILE="$HOME/.everyrow/task.json"

if [ -f "$TASK_FILE" ]; then
  TASK=$(cat "$TASK_FILE")
  COMPLETED=$(echo "$TASK" | jq -r '.completed')
  TOTAL=$(echo "$TASK" | jq -r '.total')
  FAILED=$(echo "$TASK" | jq -r '.failed')
  STARTED=$(echo "$TASK" | jq -r '.started_at' | cut -d. -f1)
  ELAPSED=$(( $(date +%s) - STARTED ))

  # Desktop notification (macOS or Linux)
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"$COMPLETED/$TOTAL complete ($FAILED failed) in ${ELAPSED}s\" with title \"Everyrow\" sound name \"Glass\"" 2>/dev/null
  elif command -v notify-send >/dev/null 2>&1; then
    notify-send "Everyrow" "$COMPLETED/$TOTAL complete ($FAILED failed) in ${ELAPSED}s"
  fi

  rm -f "$TASK_FILE"
fi

exit 0
