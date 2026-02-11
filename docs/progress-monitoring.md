---
title: Progress Monitoring
description: How to track the progress of long-running everyrow operations in Claude Code, Codex CLI, and Python scripts.
---

# Progress Monitoring

everyrow operations typically take 1–10+ minutes depending on dataset size and operation type. Both the MCP tools and the Python SDK provide real-time progress updates. They also output a URL to the session, which has a web UI that streams updates to the tables of data.

## What to Expect

When you run an everyrow operation:

1. Submit returns immediately (~0.6s) with a session URL with a full UI
2. Progress updates appear every ~15 seconds during execution
3. Results are saved as a CSV file when the operation completes
4. A desktop notification (macOS) tells you when it's done

## MCP Progress (Claude Code / Codex CLI)

When using the MCP tools (the default path with the plugin), long-running operations use a submit/poll pattern:

```
everyrow_agent_submit  →  start the operation, get a task_id and session URL
everyrow_progress      →  check status (blocks ~12s, then returns progress)
everyrow_progress      →  check again (the agent loops automatically)
everyrow_results       →  download results when complete
```

The agents should handle the polling loop automatically. You'll see progress in the conversation like:

```
Running: 23/50 complete, 5 running (45s elapsed)
```

And when it finishes:

```
Completed: 49 succeeded, 1 failed (142s total)
Saved 50 rows to /path/to/output.csv
```

### Status Line (Progress Bar)

For a persistent progress bar in the Claude Code terminal footer, add this to your `.claude/settings.json`:

```json
{
  "statusLine": {
    "type": "command",
    "command": "<path-to-plugin>/everyrow-mcp/scripts/everyrow-statusline.sh",
    "padding": 1
  }
}
```

If the everyrow-sdk repo is in your project directory:

```json
{
  "statusLine": {
    "type": "command",
    "command": "\"$CLAUDE_PROJECT_DIR\"/everyrow-sdk/everyrow-mcp/scripts/everyrow-statusline.sh",
    "padding": 1
  }
}
```

This shows a live progress bar:

```
everyrow ████████░░░░░░░ 42/100 23s   view
```

The "view" link is clickable in terminals that support OSC 8 hyperlinks (iTerm2, kitty, WezTerm, Windows Terminal) and opens the session dashboard in your browser.

The status line and hook scripts require **jq** for JSON parsing:

```bash
# macOS
brew install jq

# Linux
apt install jq
```

Note: After adding the config, restart Claude Code. Status line settings are loaded at startup only.

### Stop Guard

The plugin includes a stop guard hook that prevents Claude from ending its turn while an operation is running. If you see:

```
Stop hook error: [everyrow] Task abc123 still running. Call everyrow_progress(task_id="abc123") to check status.
```

This is expected behavior, not an error. The "error:" prefix is a [known cosmetic issue](https://github.com/anthropics/claude-code/issues/12667) in Claude Code. The hook is working correctly — it's keeping the agent focused on your running operation.

### Session URL

Every operation creates a session visible at `everyrow.io/sessions/<id>`. The session URL is returned when the operation starts and is also shown in the status line. You can open it in your browser to see:

- Real-time progress for each row
- Individual agent results and research traces
- Error details for failed rows

## Python SDK Progress

When using the Python SDK directly, progress is printed to stderr:

```
[11:16:55] Session: https://everyrow.io/sessions/abc123
[11:16:55] Starting (50 agents)...
[11:16:57]   [5/50]  10% | 5 running, 0 failed
[11:17:03]   [12/50] 24% | 8 running, 0 failed | ~15s remaining
...
[11:18:20]   [50/50] 100% | Done (85.2s total)
[11:18:20] Results: 49 succeeded, 1 failed
```

### Custom Progress Callback

For programmatic progress handling, pass an `on_progress` callback to `await_result()`:

```python
from everyrow.generated.models import TaskProgressInfo

def my_progress_handler(progress: TaskProgressInfo):
    print(f"{progress.completed}/{progress.total} done, {progress.failed} failed")

result = await task.await_result(on_progress=my_progress_handler)
```

The callback receives a `TaskProgressInfo` object with fields: `pending`, `running`, `completed`, `failed`, `total`. It only fires when the progress snapshot actually changes (no duplicate calls). Exceptions in the callback are caught and logged, so a buggy callback won't break the polling loop.

### JSONL Progress Log

The SDK also writes a machine-readable log to `~/.everyrow/progress.jsonl`:

```jsonl
{"ts": 1707400000.0, "step": "start", "total": 50, "session_url": "https://..."}
{"ts": 1707400002.0, "completed": 5, "running": 8, "failed": 0, "total": 50}
{"ts": 1707400085.0, "step": "done", "elapsed": 85.2, "succeeded": 49, "failed": 1}
```

### Crash Recovery

If your script crashes after submitting a task, you can recover the results using the task ID:

```python
from everyrow import fetch_task_data

df = await fetch_task_data("12345678-1234-1234-1234-123456789abc")
df.to_csv("recovered_results.csv", index=False)
```

Tip: Always print the task ID right after submitting async operations. The session URL (visible in stderr output) also helps you find the task in the everyrow.io dashboard.
