#!/bin/bash
INPUT=$(cat)
TASK_FILE="$HOME/.everyrow/task.json"

if [ -f "$TASK_FILE" ]; then
  TASK=$(cat "$TASK_FILE")
  COMPLETED=$(echo "$TASK" | jq -r '.completed')
  TOTAL=$(echo "$TASK" | jq -r '.total')
  FAILED=$(echo "$TASK" | jq -r '.failed')
  ELAPSED=$(( $(date +%s) - $(echo "$TASK" | jq -r '.started_at') ))

  # Desktop notification (macOS)
  osascript -e "display notification \"$COMPLETED/$TOTAL complete ($FAILED failed) in ${ELAPSED}s\" with title \"Everyrow\" sound name \"Glass\"" 2>/dev/null

  rm -f "$TASK_FILE"
fi
exit 0
