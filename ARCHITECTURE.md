# Architecture

Technical reference for the everyrow SDK, MCP server, and Claude Code plugin.

## Overview

everyrow provides two interfaces:

1. **MCP server** — 9 tools exposed directly to Claude Code / Codex. Zero-code, conversational progress updates, results saved as CSV.
2. **Python SDK** — `from everyrow.ops import rank, agent_map, ...` for scripts, pipelines, and programmatic control.

Both share the same backend (Cohort engine) and session/task model. The Claude Code plugin ships both in a single package, and the Skill gives further instructions.

## MCP Server

### Tools

The MCP server (`everyrow-mcp/src/everyrow_mcp/server.py`) exposes 9 tools via [FastMCP](https://github.com/jlowin/fastmcp):

Blocking tools (run to completion, return results inline):
- `everyrow_screen` — Filter rows by natural language criteria
- `everyrow_rank` — Score and sort rows
- `everyrow_dedupe` — Remove semantic duplicates
- `everyrow_merge` — Join two CSVs by intelligent entity matching
- `everyrow_agent` — Run web research agents on each row

Submit/poll tools (for long-running operations):
- `everyrow_agent_submit` — Submit agent_map, return immediately with `task_id` and `session_url`
- `everyrow_rank_submit` — Submit rank, return immediately
- `everyrow_screen_submit` — Submit screen, return immediately
- `everyrow_dedupe_submit` — Submit dedupe, return immediately
- `everyrow_merge_submit` — Submit merge, return immediately
- `everyrow_progress` — Poll task status (blocks ~12s server-side, returns progress text)
- `everyrow_results` — Retrieve completed results, save to CSV

All tools use `@mcp.tool(structured_output=False)` to suppress FastMCP's `structuredContent` field. Without this, Claude Code displays raw JSON blobs instead of clean text (see [claude-code#9962](https://github.com/anthropics/claude-code/issues/9962)).

### Submit/Poll Pattern

Long-running operations (agent_map, rank) use a submit/poll pattern because:
- Operations take 1–10+ minutes
- LLMs cannot tell time and will hallucinate if asked to wait ([arXiv:2601.13206](https://arxiv.org/abs/2601.13206))
- Client-side timeouts (60s in Codex CLI) kill blocking calls

The flow:

```
1. everyrow_*_submit    → creates session, submits async task, returns task_id + session_url (0.6s)
2. everyrow_progress    → server blocks 12s, polls engine, returns status text (12-15s per call)
3. (repeat step 2)      → progress text says "call everyrow_progress again immediately"
4. everyrow_results     → on completion, fetches data, saves CSV, cleans up
```

Server-controlled pacing: The `PROGRESS_POLL_DELAY` constant (12s) controls how long `everyrow_progress` blocks before returning. This prevents the agent from burning inference tokens on rapid polling. Combined with ~3s inference overhead, users see updates every ~15s.

Chaining instructions: The progress tool's response text includes "Immediately call everyrow_progress again" to keep the agent in a tight poll loop. This is critical — without it, Claude tends to stop and ask the user if they want to check again.

### Task State Tracking

The MCP server maintains two forms of state for in-flight tasks:

In-process: `_active_tasks` dictionary keyed by `task_id`. The value is an `ActiveTask` dataclass with typed fields: `session`, `session_ctx`, `client: AuthenticatedClient`, `total: int`, `session_url: str`, `started_at: float`, `input_csv: str`, `prefix: str`. Cleaned up by `everyrow_results`.

On-disk: `~/.everyrow/task.json` written by `_write_task_state()` on submit and each progress poll. Contains:

```json
{
  "task_id": "abc123",
  "session_url": "https://everyrow.io/sessions/...",
  "total": 50,
  "completed": 23,
  "failed": 1,
  "running": 5,
  "status": "running",
  "started_at": 1707400000.0
}
```

This file is a singleton. Only one task is tracked at a time. It is read by the status line script and hook scripts (see below). The MCP server writes it directly rather than relying on hooks, which avoids the fragile double-escaped JSON parsing required to extract `tool_response` from plugin MCP tools. The path `~/.everyrow/task.json` is consistent with the SDK's `~/.everyrow/` directory used for other files like `progress.jsonl`.

## SDK Progress Output

The SDK's `await_task_completion()` function (`src/everyrow/task.py`) polls the engine every 2 seconds and provides progress through three channels:

stderr (default): Timestamped progress lines:
```
[11:16:55] Session: https://everyrow.io/sessions/abc123
[11:16:55] Starting (50 agents)...
[11:16:57]   [5/50]  10% | 5 running, 0 failed
[11:17:03]   [12/50] 24% | 8 running, 0 failed | ~15s remaining
...
[11:18:20]   [50/50] 100% | Done (85.2s total)
[11:18:20] Results: 49 succeeded, 1 failed
```

JSONL log (`~/.everyrow/progress.jsonl`): Machine-readable log appended on each progress change. Useful for post-hoc analysis and verification scripts.

`on_progress` callback: Optional parameter to `await_result()` or `await_task_completion()`. Receives a `TaskProgressInfo` object (from `everyrow.generated.models`) with `pending`, `running`, `completed`, `failed`, `total` counts. Only fires when the snapshot changes (deduplication prevents redundant calls). The callback is wrapped in a try/except so exceptions don't break the polling loop.

### Snapshot deduplication

Progress callbacks and output only trigger when the tuple `(pending, running, completed, failed)` changes from the last poll. This prevents flooding stderr or the callback with identical lines when the engine hasn't made progress between polls.

## Plugin System

### What the plugin bundles

The Claude Code plugin (`.claude-plugin/plugin.json`) ships:

1. **MCP server** — All 9 tools, auto-started by Claude Code
2. **Hooks** — Stop guard, results tracking, session cleanup
3. **Skill** (`skills/everyrow-sdk/SKILL.md`) — SDK code-generation guidance for the Skills path

### Hooks

PostToolUse (matcher: `mcp__plugin_everyrow_everyrow__everyrow_results`):
Runs `everyrow-track-results.sh`, sends a desktop notification (macOS via `osascript`, Linux via `notify-send`) with completion summary, then deletes `~/.everyrow/task.json`.

Stop:
Runs `everyrow-stop-guard.sh`, reads `~/.everyrow/task.json`. If a task is running, outputs `{"decision": "block", "reason": "..."}` which prevents Claude from ending its turn. The reason text instructs Claude to call `everyrow_progress` to check status.

Note: Claude Code displays stop hook blocks as "Stop hook error: ..." This is a cosmetic UI bug ([claude-code#12667](https://github.com/anthropics/claude-code/issues/12667)), not an actual error. The hook is working correctly.

**SessionEnd**:
Runs `rm -f ~/.everyrow/task.json` to clean up tracking state.

### Status Line

The status line script (`everyrow-mcp/scripts/everyrow-statusline.sh`) is not part of the plugin (the plugin format cannot write to `settings.json`). It must be manually configured:

```json
{
  "statusLine": {
    "type": "command",
    "command": "<path>/everyrow-mcp/scripts/everyrow-statusline.sh",
    "padding": 1
  }
}
```

The script reads `~/.everyrow/task.json` on each refresh and renders:
```
everyrow ████████░░░░░░░ 42/100 23s   view
```

Features: model name, context usage from Claude Code env vars, progress bar, elapsed time, failure count (yellow if >0), OSC 8 clickable link to session URL (works in iTerm2, kitty, WezTerm, Windows Terminal; degrades gracefully elsewhere).

## Known Issues

| Issue | Impact | Workaround |
|-------|--------|------------|
| Plugin hooks don't fire with `isLocal: true` ([#14410](https://github.com/anthropics/claude-code/issues/14410)) | Hooks from `--plugin-dir` installs are silently ignored | Install from git URL or duplicate hooks into `.claude/settings.json` |
| Hook matchers don't support mid-pattern wildcards | `foo_*_bar` won't match | Use exact match or trailing `*` only |
| "Stop hook error:" cosmetic UI ([#12667](https://github.com/anthropics/claude-code/issues/12667)) | Misleading error display when stop guard blocks | No workaround — craft self-documenting reason text |
| Hooks stop after ~2.5h ([#16047](https://github.com/anthropics/claude-code/issues/16047)) | Stop guard and notifications stop in long sessions | Restart session for operations in sessions >2h |
| `type: "prompt"` hooks broken in plugins ([#13155](https://github.com/anthropics/claude-code/issues/13155)) | Must use `type: "command"` for all hooks | Already handled in plugin config |
| `~/.everyrow/task.json` is singleton | Only one operation tracked at a time | Sequential operations only (matches current behavior) |
| MCP/plugins/hooks can't reload mid-session | Any config change requires restart | Restart Claude Code after changes |
| `jq` required | Status line and hook scripts need `jq` installed | `brew install jq` (macOS) or `apt install jq` (Linux) |

## Rejected Alternatives

MCP Tasks (SEP-1686): The official MCP long-running task protocol. Near-zero client adoption as of Feb 2026. Custom submit/poll tools work today across all MCP clients.

MCP Notifications / Progress tokens: Not displayed by Claude Code or any major client. Would be invisible to users even if implemented.

Server-Sent Events (SSE): Would replace polling with push notifications. Adds complexity (persistent connections, reconnection logic) for marginal gain. The 12s polling cadence already provides smooth UX. MCP's stdio transport doesn't support SSE natively.

Hook-based state tracking: The original design had PostToolUse hooks on `_submit` and `_progress` tools parse `tool_response` JSON and write the task state file. This was fragile because plugin MCP tool responses are double-escaped JSON strings (`{"result": "<escaped>"}`) that required careful parsing. Moving state writes into the MCP server itself (`_write_task_state()`) was simpler and more reliable.
