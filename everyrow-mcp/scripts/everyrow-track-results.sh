#!/bin/bash
INPUT=$(cat)

if [ -f /tmp/everyrow-task.json ]; then
  TASK=$(cat /tmp/everyrow-task.json)
  COMPLETED=$(echo "$TASK" | jq -r '.completed')
  TOTAL=$(echo "$TASK" | jq -r '.total')
  FAILED=$(echo "$TASK" | jq -r '.failed')
  ELAPSED=$(( $(date +%s) - $(echo "$TASK" | jq -r '.started_at') ))

  # Desktop notification (macOS)
  osascript -e "display notification \"$COMPLETED/$TOTAL complete ($FAILED failed) in ${ELAPSED}s\" with title \"Everyrow\" sound name \"Glass\"" 2>/dev/null

  rm -f /tmp/everyrow-task.json
fi
exit 0
