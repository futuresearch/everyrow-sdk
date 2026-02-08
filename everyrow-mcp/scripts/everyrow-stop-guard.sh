#!/bin/bash
INPUT=$(cat)
STOP_HOOK_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')

# Don't block if already in a stop-hook continuation (prevent infinite loop)
if [ "$STOP_HOOK_ACTIVE" = "true" ]; then
  exit 0
fi

if [ -f /tmp/everyrow-task.json ]; then
  STATUS=$(jq -r '.status' /tmp/everyrow-task.json)
  TASK_ID=$(jq -r '.task_id' /tmp/everyrow-task.json)

  if [ "$STATUS" = "running" ]; then
    jq -n \
      --arg reason "[everyrow] Task $TASK_ID still running. Call everyrow_progress(task_id=\"$TASK_ID\") to check status." \
      '{decision: "block", reason: $reason}'
    exit 0
  fi
fi

exit 0
