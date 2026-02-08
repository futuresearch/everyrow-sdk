---
title: MCP Server
description: Reference for all 9 everyrow MCP server tools — blocking operations, submit/poll for long-running tasks, and result retrieval.
---

# MCP Server Reference

The everyrow MCP server exposes 9 tools for AI-powered data processing. These tools are called directly by Claude Code, Codex CLI, and other MCP clients — no Python code is needed.

## Blocking Tools

These tools run the operation to completion and return results inline. Use them for small datasets or quick operations.

### everyrow_screen

Filter rows by natural language criteria.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task` | string | Yes | Screening criteria. Rows that meet the criteria pass. |
| `input_csv` | string | Yes | Absolute path to input CSV. |
| `output_path` | string | Yes | Directory or full .csv path for output. |
| `response_schema` | object | No | JSON schema for custom fields. Default: `{passes: bool}`. |

### everyrow_rank

Score and sort rows by qualitative criteria.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task` | string | Yes | What makes a row score higher or lower. |
| `input_csv` | string | Yes | Absolute path to input CSV. |
| `output_path` | string | Yes | Directory or full .csv path for output. |
| `field_name` | string | Yes | Column name for the score. |
| `field_type` | string | No | Score type: `float` (default), `int`, `str`, `bool`. |
| `ascending_order` | bool | No | `true` = lowest first (default). |
| `response_schema` | object | No | JSON schema for additional fields. |

### everyrow_dedupe

Remove semantic duplicates.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `equivalence_relation` | string | Yes | What makes two rows duplicates. |
| `input_csv` | string | Yes | Absolute path to input CSV. |
| `output_path` | string | Yes | Directory or full .csv path for output. |
| `select_representative` | bool | No | `true` (default) = keep one per group. `false` = mark all with class info. |

### everyrow_merge

Join two CSVs using intelligent entity matching.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task` | string | Yes | How to match rows between tables. |
| `left_csv` | string | Yes | Absolute path to primary CSV. |
| `right_csv` | string | Yes | Absolute path to secondary CSV. |
| `output_path` | string | Yes | Directory or full .csv path for output. |
| `merge_on_left` | string | No | Column in left table to match on. |
| `merge_on_right` | string | No | Column in right table to match on. |
| `use_web_search` | string | No | `auto` (default), `yes`, or `no`. |

### everyrow_agent

Run web research agents on each row (blocking).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task` | string | Yes | Task for the agent to perform on each row. |
| `input_csv` | string | Yes | Absolute path to input CSV. |
| `output_path` | string | Yes | Directory or full .csv path for output. |
| `response_schema` | object | No | JSON schema for structured output. Default: `{answer: str}`. |

## Submit/Poll Tools

For long-running operations (agent_map, rank), use the submit/poll pattern. This returns immediately and lets the agent poll for progress.

### everyrow_agent_submit

Submit an agent_map operation (returns immediately).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task` | string | Yes | Task for the agent to perform on each row. |
| `input_csv` | string | Yes | Absolute path to input CSV. |
| `response_schema` | object | No | JSON schema for structured output. |

Returns `task_id` and `session_url`. The agent should immediately call `everyrow_progress` with the `task_id`.

### everyrow_rank_submit

Submit a rank operation (returns immediately).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task` | string | Yes | Ranking criteria. |
| `input_csv` | string | Yes | Absolute path to input CSV. |
| `field_name` | string | Yes | Column name for the score. |
| `field_type` | string | No | Score type (default: `float`). |
| `ascending_order` | bool | No | `true` = lowest first (default). |
| `response_schema` | object | No | JSON schema for additional fields. |

Returns `task_id` and `session_url`.

### everyrow_progress

Check progress of a running task. **Blocks ~12 seconds** before returning.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task_id` | string | Yes | The task ID from a `_submit` tool. |

Returns status text with completion counts and elapsed time. Instructs the agent to call again immediately until the task completes or fails.

### everyrow_results

Retrieve results from a completed task and save to CSV.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task_id` | string | Yes | The task ID of the completed task. |
| `output_path` | string | Yes | Directory or full .csv path for output. |

Returns confirmation with row count and file path.

## Submit/Poll Workflow

```
1. everyrow_agent_submit(task, input_csv)
   → Returns task_id + session_url (~0.6s)

2. everyrow_progress(task_id)
   → Blocks 12s, returns "Running: 5/50 complete, 8 running (15s elapsed)"
   → Response says "call everyrow_progress again immediately"

3. everyrow_progress(task_id)  (repeat)
   → "Running: 23/50 complete, 5 running (45s elapsed)"

4. everyrow_progress(task_id)  (final)
   → "Completed: 49 succeeded, 1 failed (142s total)"

5. everyrow_results(task_id, output_path)
   → "Saved 50 rows to /path/to/agent_companies.csv"
```

The agent handles this loop automatically. You don't need to intervene.

## Custom Response Schemas

All tools that accept `response_schema` take a JSON schema object:

```json
{
  "properties": {
    "annual_revenue": {
      "type": "integer",
      "description": "Annual revenue in USD"
    },
    "employee_count": {
      "type": "integer",
      "description": "Number of employees"
    }
  },
  "required": ["annual_revenue"]
}
```

Supported types: `string`, `integer`, `number`, `boolean`, `array`, `object`.

## Plugin

The Claude Code plugin (`.claude-plugin/plugin.json`) bundles:

1. MCP server, with all 9 tools above
2. Hooks, such as stop guard (prevents ending turn during operations), results notification (macOS), session cleanup
3. Skill, to guide agents with quick SDK code generation for the Python SDK path

Install with:
```bash
claude plugin add futuresearch/everyrow-sdk
```

See [Progress Monitoring](/docs/progress-monitoring) for status line setup and hook details.
