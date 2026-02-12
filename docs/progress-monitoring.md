---
title: Progress Monitoring
description: How to track the progress of long-running everyrow operations in Claude Code, Codex CLI, and Python scripts.
---

# Progress Monitoring

everyrow operations typically take 1–10+ minutes depending on dataset size and operation type. Both the Python SDK and the MCP tools provide progress monitoring and a session URL where you can watch tables update in real-time.

## What to Expect

### Web UI

Every operation is part of a session, which you can view at `https://everyrow.io/sessions/<session_id>`. You can open it in your browser to see:

- Real-time progress for each row
- Web searches each agent ran and the pages it read
- Explanation for each result, including links to sources
- Data visualizations

### Python SDK

When using the SDK directly, progress is printed to stderr by default:

```
Processing 50 rows...
Session: https://everyrow.io/sessions/abc123
 (5s) [5/50]  10% | 5 running
(15s) [20/50] 40% | 8 running
(45s) [50/50] 100%
```

You can also provide a custom `on_progress` callback for programmatic progress handling (see below).

### MCP Tools

When you run an everyrow operation via MCP:

1. The operation returns immediately with a session URL
2. Progress updates appear every few seconds during execution
3. Results are saved as a CSV file when the operation completes
4. If you've installed the plugin, a desktop notification (macOS and Linux) tells you when it's done

The workflow:

```
everyrow_agent        →  start the operation, get a task_id and session URL
everyrow_progress     →  check status (blocks for a few seconds, then returns progress)
everyrow_progress     →  check again (the agent loops automatically)
everyrow_results      →  download results when complete
```

The agents handle the polling loop automatically. You'll see progress in the conversation like:

```
Running: 20/50 complete, 30 running (45s elapsed)
```

And when it finishes:

```
Completed: 50/50 (0 failed) in 100s
...
Saved 50 rows to /path/to/output.csv
```

## Claude Code Integration

### Status Line (Progress Bar)

For a persistent progress bar in the Claude Code terminal footer, add this to either your global settings (`.claude/settings.json`) or project settings (`.claude/project-settings.json`), depending on where you have the MCP server configured:

```json
{
  "statusLine": {
    "type": "command",
    "command": "<path-to-plugin>/everyrow-mcp/scripts/everyrow-statusline.sh",
    "padding": 1
  }
}
```

This shows a live progress bar:

```
everyrow ████████░░░░░░░ 42/100 23s   view
```

The "view" link is clickable in terminals that support hyperlinks (iTerm2, kitty, WezTerm, Windows Terminal) and opens the session dashboard in your browser.

The status line and hook scripts require [**jq**](https://jqlang.org/) for JSON parsing:

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

## Python SDK Progress

Progress is printed to stderr by default. You can customize this with the `on_progress` callback:

```python
from everyrow.generated.models import TaskProgressInfo

def my_progress_handler(progress: TaskProgressInfo):
    print(f"{progress.completed}/{progress.total} done, {progress.failed} failed")

result = await task.await_result(on_progress=my_progress_handler)
```

The callback receives a `TaskProgressInfo` object and only fires when the progress snapshot actually changes.
