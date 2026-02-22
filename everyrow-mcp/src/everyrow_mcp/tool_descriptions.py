from __future__ import annotations

from everyrow_mcp.app import mcp

# ── everyrow_progress ──────────────────────────────────────────────────

_PROGRESS_DESC = """\
Check progress of a running task. Blocks briefly to limit the polling rate.

After receiving a status update, immediately call everyrow_progress again
unless the task is completed or failed. The tool handles pacing internally.
Do not add commentary between progress calls, just call again immediately."""

# ── everyrow_results ───────────────────────────────────────────────────

_RESULTS_STDIO = """\
Retrieve results from a completed everyrow task.

Only call this after everyrow_progress reports status 'completed'.
Pass output_path (ending in .csv) to save results as a local CSV file."""

_RESULTS_HTTP = """\
Retrieve results from a completed everyrow task.

Only call this after everyrow_progress reports status 'completed'.
Results are returned as a paginated preview with a download link.
Do NOT pass output_path — it has no effect in this mode."""

# ── Registry ───────────────────────────────────────────────────────────

_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "everyrow_progress": {"stdio": _PROGRESS_DESC, "http": _PROGRESS_DESC},
    "everyrow_results": {"stdio": _RESULTS_STDIO, "http": _RESULTS_HTTP},
}


def set_tool_descriptions(transport: str) -> None:
    """Patch registered tool descriptions to match *transport* ('stdio' or 'http').

    Call once from ``main()`` after determining the transport mode.
    """
    mode = "stdio" if transport == "stdio" else "http"
    for tool_name, descs in _DESCRIPTIONS.items():
        tool = mcp._tool_manager.get_tool(tool_name)
        if tool is not None:
            tool.description = descs[mode]
