#!/bin/bash
INPUT=$(cat)

RESP=$(echo "$INPUT" | jq -r '
  if (.tool_response | type) == "object" and .tool_response.status then .tool_response
  elif (.tool_response | type) == "object" and .tool_response.content[0].text then (.tool_response.content[0].text | fromjson)
  elif (.tool_response | type) == "string" then (.tool_response | fromjson)
  else .tool_response end
')
STATUS=$(echo "$RESP" | jq -r '.status // empty')
COMPLETED=$(echo "$RESP" | jq -r '.completed // 0')
FAILED=$(echo "$RESP" | jq -r '.failed // 0')
RUNNING=$(echo "$RESP" | jq -r '.running // 0')
TOTAL=$(echo "$RESP" | jq -r '.total // 0')

if [ -f /tmp/everyrow-task.json ]; then
  EXISTING=$(cat /tmp/everyrow-task.json)
  echo "$EXISTING" | jq \
    --arg s "$STATUS" \
    --argjson c "$COMPLETED" \
    --argjson f "$FAILED" \
    --argjson r "$RUNNING" \
    --argjson t "$TOTAL" \
    '.status=$s | .completed=$c | .failed=$f | .running=$r | .total=$t' \
    > /tmp/everyrow-task.json
fi
exit 0
