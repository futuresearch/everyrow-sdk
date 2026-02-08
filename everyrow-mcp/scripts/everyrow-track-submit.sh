#!/bin/bash
INPUT=$(cat)

# Defensive parsing: handle flat JSON, MCP content-wrapped, or string formats
RESP=$(echo "$INPUT" | jq -r '
  if (.tool_response | type) == "object" and .tool_response.task_id then .tool_response
  elif (.tool_response | type) == "object" and .tool_response.content[0].text then (.tool_response.content[0].text | fromjson)
  elif (.tool_response | type) == "string" then (.tool_response | fromjson)
  else .tool_response end
')
TASK_ID=$(echo "$RESP" | jq -r '.task_id // empty')
SESSION_URL=$(echo "$RESP" | jq -r '.session_url // empty')
TOTAL=$(echo "$RESP" | jq -r '.total // 0')

if [ -n "$TASK_ID" ]; then
  jq -n \
    --arg tid "$TASK_ID" \
    --arg url "$SESSION_URL" \
    --argjson total "$TOTAL" \
    --argjson ts "$(date +%s)" \
    '{task_id: $tid, session_url: $url, total: $total, completed: 0, failed: 0, running: 0, status: "running", started_at: $ts}' \
    > /tmp/everyrow-task.json
fi
exit 0
