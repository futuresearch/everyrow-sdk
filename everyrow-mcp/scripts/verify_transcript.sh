#!/bin/bash
# Usage: verify_transcript.sh <session-jsonl-path>
# Verifies MCP tool call sequence and timing from a Claude Code session transcript.
set -euo pipefail
SESSION="$1"

TOOLS=$(cat "$SESSION" | jq -r '
  select(.type == "assistant")
  | .message.content[]?
  | select(.type == "tool_use")
  | .name
' 2>/dev/null)

echo "=== Tool sequence ==="
echo "$TOOLS"

echo "$TOOLS" | grep -q "everyrow_.*_submit" || { echo "FAIL: no submit call found"; exit 1; }
echo "PASS: submit called"

PROGRESS_COUNT=$(echo "$TOOLS" | grep -c "everyrow_progress" || true)
[ "$PROGRESS_COUNT" -ge 1 ] || { echo "FAIL: no progress calls"; exit 1; }
echo "PASS: progress called ($PROGRESS_COUNT times)"

echo "$TOOLS" | grep -q "everyrow_results" || { echo "FAIL: no results call found"; exit 1; }
echo "PASS: results called"

SUBMIT_LINE=$(echo "$TOOLS" | grep -n "everyrow_.*_submit" | head -1 | cut -d: -f1)
FIRST_PROGRESS=$(echo "$TOOLS" | grep -n "everyrow_progress" | head -1 | cut -d: -f1)
RESULTS_LINE=$(echo "$TOOLS" | grep -n "everyrow_results" | head -1 | cut -d: -f1)
[ "$SUBMIT_LINE" -lt "$FIRST_PROGRESS" ] || { echo "FAIL: submit should come before progress"; exit 1; }
[ "$FIRST_PROGRESS" -lt "$RESULTS_LINE" ] || { echo "FAIL: progress should come before results"; exit 1; }
echo "PASS: correct order (submit → progress → results)"

echo ""
echo "=== Timing analysis ==="
# Pair tool_use calls with tool_result responses to measure duration and cadence.
# assistant messages contain tool_use blocks; user messages contain tool_result blocks.
python3 -c "
import json, sys
from datetime import datetime

def parse_ts(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00'))

with open('$SESSION') as f:
    lines = f.readlines()

calls = {}   # tool_use_id -> {name, start_ts}
pairs = []

for line in lines:
    try:
        msg = json.loads(line)
    except:
        continue
    ts_str = msg.get('timestamp')
    if not ts_str:
        continue
    ts = parse_ts(ts_str)

    if msg.get('type') == 'assistant':
        for block in msg.get('message', {}).get('content', []):
            if isinstance(block, dict) and block.get('type') == 'tool_use':
                calls[block['id']] = {'name': block.get('name','?'), 'start': ts}

    elif msg.get('type') == 'user':
        content = msg.get('message', {}).get('content', [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'tool_result':
                    tid = block.get('tool_use_id')
                    if tid in calls:
                        dur = (ts - calls[tid]['start']).total_seconds()
                        pairs.append({
                            'name': calls[tid]['name'],
                            'duration': dur,
                            'start': calls[tid]['start'],
                        })

# Filter to everyrow tools only
er_pairs = [p for p in pairs if 'everyrow' in p['name']]
if not er_pairs:
    print('No everyrow tool calls found in transcript.')
    sys.exit(0)

print(f'Everyrow tool calls: {len(er_pairs)}')
print()
for i, p in enumerate(er_pairs):
    gap = ''
    if i > 0:
        g = (p['start'] - er_pairs[i-1]['start']).total_seconds()
        gap = f'  gap={g:.1f}s'
    print(f'  {p[\"name\"]:35s} dur={p[\"duration\"]:5.1f}s{gap}  ({p[\"start\"].strftime(\"%H:%M:%S\")})')

# Verify polling cadence
progress_pairs = [p for p in er_pairs if p['name'] == 'mcp__everyrow__everyrow_progress']
if len(progress_pairs) >= 2:
    gaps = []
    for i in range(1, len(progress_pairs)):
        g = (progress_pairs[i]['start'] - progress_pairs[i-1]['start']).total_seconds()
        gaps.append(g)
    avg_gap = sum(gaps) / len(gaps)
    print(f'')
    print(f'Polling cadence: avg={avg_gap:.1f}s  min={min(gaps):.1f}s  max={max(gaps):.1f}s')
    if 10 <= avg_gap <= 40:
        print('PASS: polling cadence in expected range (10-40s)')
    else:
        print(f'WARN: polling cadence {avg_gap:.1f}s outside expected 10-40s range')

# Verify progress tool blocking duration (should be 10-15s server-side)
if progress_pairs:
    durs = [p['duration'] for p in progress_pairs]
    avg_dur = sum(durs) / len(durs)
    print(f'Progress tool duration: avg={avg_dur:.1f}s  min={min(durs):.1f}s  max={max(durs):.1f}s')
    if 8 <= avg_dur <= 25:
        print('PASS: progress blocking duration in expected range (8-25s)')
    else:
        print(f'WARN: progress blocking duration {avg_dur:.1f}s outside expected 8-25s range')
"
