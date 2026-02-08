#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)/scripts"
rm -f /tmp/everyrow-task.json

# Progress hook is now a no-op (server writes the file directly).
# Verify it exits cleanly and doesn't create the file.
echo '{}' | bash "$SCRIPT_DIR/everyrow-track-progress.sh"
[ ! -f /tmp/everyrow-task.json ] || { echo "FAIL: no-op hook should not create file"; exit 1; }
echo "PASS: progress hook (no-op)"
echo "ALL PASS: progress hook"
