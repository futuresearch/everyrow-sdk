#!/usr/bin/env python3
"""Verify that SDK progress output was emitted incrementally, not buffered."""
import json, os, sys

log_path = sys.argv[1] if len(sys.argv) > 1 else "~/.everyrow/progress.jsonl"
log_path = os.path.expanduser(log_path)

with open(log_path) as f:
    entries = [json.loads(line) for line in f if line.strip()]

if len(entries) < 2:
    print("FAIL: fewer than 2 progress entries")
    sys.exit(1)

times = [e["ts"] for e in entries]
deltas = [times[i+1] - times[i] for i in range(len(times)-1)]

print(f"Entries: {len(entries)}")
print(f"Total span: {times[-1] - times[0]:.1f}s")
print(f"Inter-entry deltas: {', '.join(f'{d:.1f}s' for d in deltas)}")

# Verify: if lines were buffered, all deltas would be ~0.
# If streaming, deltas should be >= 1s (the polling interval).
buffered_count = sum(1 for d in deltas if d < 0.5)
if buffered_count > len(deltas) * 0.5:
    print(f"FAIL: {buffered_count}/{len(deltas)} deltas < 0.5s — output was likely buffered")
    sys.exit(1)
else:
    print(f"PASS: output was emitted incrementally ({buffered_count}/{len(deltas)} fast deltas)")
