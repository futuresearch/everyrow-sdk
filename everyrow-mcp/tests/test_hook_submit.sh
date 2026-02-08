#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)/scripts"
rm -f /tmp/everyrow-task.json

# Submit hook is now a no-op (server writes the file directly).
# Verify it exits cleanly and doesn't create the file.
echo '{}' | bash "$SCRIPT_DIR/everyrow-track-submit.sh"
[ ! -f /tmp/everyrow-task.json ] || { echo "FAIL: no-op hook should not create file"; exit 1; }
echo "PASS: submit hook (no-op)"
echo "ALL PASS: submit hook"
