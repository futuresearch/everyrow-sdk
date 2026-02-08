#!/bin/bash
input=$(cat)

MODEL=$(echo "$input" | jq -r '.model.display_name')
PCT=$(echo "$input" | jq -r '.context_window.used_percentage // 0' | cut -d. -f1)

GREEN='\033[32m'
YELLOW='\033[33m'
CYAN='\033[36m'
DIM='\033[2m'
RESET='\033[0m'

echo -e "${CYAN}[$MODEL]${RESET} ${PCT}% context"

if [ -f /tmp/everyrow-task.json ]; then
  TASK=$(cat /tmp/everyrow-task.json)
  STATUS=$(echo "$TASK" | jq -r '.status')
  COMPLETED=$(echo "$TASK" | jq -r '.completed // 0')
  TOTAL=$(echo "$TASK" | jq -r '.total // 0')
  FAILED=$(echo "$TASK" | jq -r '.failed // 0')
  URL=$(echo "$TASK" | jq -r '.session_url // empty')
  STARTED=$(echo "$TASK" | jq -r '.started_at // 0')
  ELAPSED=$(( $(date +%s) - STARTED ))

  if [ "$STATUS" = "running" ] && [ "$TOTAL" -gt 0 ]; then
    TASK_PCT=$((COMPLETED * 100 / TOTAL))
    BAR_WIDTH=15
    FILLED=$((TASK_PCT * BAR_WIDTH / 100))
    EMPTY=$((BAR_WIDTH - FILLED))
    BAR=$(printf "%${FILLED}s" | tr ' ' '█')$(printf "%${EMPTY}s" | tr ' ' '░')

    FAIL_STR=""
    [ "$FAILED" -gt 0 ] && FAIL_STR=" ${YELLOW}${FAILED} failed${RESET}"

    LINK=""
    if [ -n "$URL" ]; then
      LINK=" $(printf '%b' "\e]8;;${URL}\a⬡ view\e]8;;\a")"
    fi

    echo -e "${GREEN}everyrow${RESET} ${BAR} ${COMPLETED}/${TOTAL} ${DIM}${ELAPSED}s${RESET}${FAIL_STR}${LINK}"
  elif [ "$STATUS" = "completed" ]; then
    echo -e "${GREEN}everyrow${RESET} ✓ done (${COMPLETED}/${TOTAL})"
  fi
fi
