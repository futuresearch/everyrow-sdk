"""Helper functions for MCP tool implementations.

Includes transport-aware UI helpers, client construction, task state
persistence, result fetching, and pagination logic.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from datetime import datetime
from typing import Any
from uuid import UUID

import pandas as pd
from everyrow.api_utils import handle_response
from everyrow.generated.api.tasks import (
    get_task_result_tasks_task_id_result_get,
    get_task_status_tasks_task_id_status_get,
)
from everyrow.generated.client import AuthenticatedClient
from everyrow.generated.models.public_task_type import PublicTaskType
from everyrow.generated.models.task_result_response_data_type_1 import (
    TaskResultResponseDataType1,
)
from everyrow.generated.models.task_status import TaskStatus
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.types import TextContent

from everyrow_mcp.models import ResultsInput
from everyrow_mcp.state import TASK_STATE_FILE, state


def _get_client():
    """Get an EveryRow API client for the current request.

    In stdio mode, returns the singleton client initialized at startup.
    In HTTP mode, creates a per-request client using the authenticated
    user's API key from the OAuth access token.
    """
    if state.is_stdio:
        if state.client is None:
            raise RuntimeError("MCP server not initialized")
        return state.client
    # HTTP mode: get JWT from authenticated request
    access_token = get_access_token()
    if access_token is None:
        raise RuntimeError("Not authenticated")
    if state.settings is None:
        raise RuntimeError("MCP server not initialized")
    return AuthenticatedClient(
        base_url=state.settings.everyrow_api_url,
        token=access_token.token,
        raise_on_unexpected_status=True,
        follow_redirects=True,
    )


def _with_ui(ui_text: str, *human: TextContent) -> list[TextContent]:
    """Prepend a widget JSON TextContent in HTTP mode; skip it in stdio to save tokens."""
    if state.is_http:
        return [TextContent(type="text", text=ui_text), *human]
    return list(human)


def _submission_text(label: str, session_url: str, task_id: str) -> str:
    """Build human-readable text for submission tool results."""
    if state.is_stdio:
        return (
            f"{label}\n"
            f"Session: {session_url}\n"
            f"Task ID: {task_id}\n\n"
            f"Share the session_url with the user, then immediately call everyrow_progress(task_id='{task_id}')."
        )
    return (
        f"{label}\n"
        f"Task ID: {task_id}\n\n"
        f"Immediately call everyrow_progress(task_id='{task_id}')."
    )


async def _submission_ui_json(
    session_url: str,
    task_id: str,
    total: int,
    token: str,
) -> str:
    """Build JSON for the session MCP App widget, and store the token for polling."""
    await state.store_task_token(task_id, token)
    poll_token = secrets.token_urlsafe(32)
    await state.store_poll_token(task_id, poll_token)
    data: dict[str, Any] = {
        "session_url": session_url,
        "task_id": task_id,
        "total": total,
        "status": "submitted",
    }
    if state.mcp_server_url:
        data["progress_url"] = (
            f"{state.mcp_server_url}/api/progress/{task_id}?token={poll_token}"
        )
    return json.dumps(data)


def _write_task_state(
    task_id: str,
    task_type: PublicTaskType,
    session_url: str,
    total: int,
    completed: int,
    failed: int,
    running: int,
    status: TaskStatus,
    started_at: datetime,
) -> None:
    """Write task tracking state for hooks/status line to read.

    Note: Only one task is tracked at a time. If multiple tasks run concurrently,
    only the most recent one's progress is shown.
    """
    if state.is_http:
        return
    try:
        TASK_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        task_state = {
            "task_id": task_id,
            "task_type": task_type.value,
            "session_url": session_url,
            "total": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "status": status.value,
            "started_at": started_at.timestamp(),
        }
        with open(TASK_STATE_FILE, "w") as f:
            json.dump(task_state, f)
    except Exception as e:
        logging.getLogger(__name__).debug(f"Failed to write task state: {e!r}")


class TaskNotReady(Exception):
    """Raised when a task is not in a terminal state."""

    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__(status)


async def _fetch_task_result(client: Any, task_id: str) -> tuple[pd.DataFrame, str]:
    """Fetch a task's result DataFrame and session ID from the API.

    Checks task status first, then retrieves and parses the result data.

    Returns:
        Tuple of (DataFrame, session_id).

    Raises:
        TaskNotReady: If the task is not in a terminal state.
        ValueError: If the result has no table data.
        Exception: On API errors.
    """
    status_response = handle_response(
        await get_task_status_tasks_task_id_status_get.asyncio(
            task_id=UUID(task_id),
            client=client,
        )
    )
    if status_response.status not in (
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.REVOKED,
    ):
        raise TaskNotReady(status_response.status.value)

    session_id = str(status_response.session_id) if status_response.session_id else ""

    result_response = handle_response(
        await get_task_result_tasks_task_id_result_get.asyncio(
            task_id=UUID(task_id),
            client=client,
        )
    )

    if isinstance(result_response.data, list):
        records = [item.additional_properties for item in result_response.data]
        return pd.DataFrame(records), session_id
    if isinstance(result_response.data, TaskResultResponseDataType1):
        return pd.DataFrame([result_response.data.additional_properties]), session_id
    raise ValueError("Task result has no table data.")


_TOKEN_BUDGET = int(os.environ.get("EVERYROW_TOKEN_BUDGET", "20000"))
_CHARS_PER_TOKEN = 4  # conservative estimate for JSON-heavy text


def _recommend_page_size(
    page_json: str, rows_shown: int, total: int, current_page_size: int
) -> int | None:
    """Estimate an optimal page_size based on the current page's token cost.

    Returns a recommended page_size, or None if the current one is fine.
    """
    if rows_shown == 0:
        return None
    avg_chars_per_row = len(page_json) / rows_shown
    avg_tokens_per_row = avg_chars_per_row / _CHARS_PER_TOKEN
    if avg_tokens_per_row <= 0:
        return None
    recommended = max(1, min(int(_TOKEN_BUDGET / avg_tokens_per_row), 100))
    # Only recommend if meaningfully different (>25% change) and there are more rows
    if (
        recommended >= total
        or abs(recommended - current_page_size) / current_page_size < 0.25
    ):
        return None
    return recommended


def _build_inline_response(
    task_id: str,
    df: pd.DataFrame,
    download_token: str,
    params: ResultsInput,
    session_url: str = "",
) -> list[TextContent]:
    """Build paginated inline response for in-memory results."""
    total = len(df)
    col_names = ", ".join(df.columns[:10])
    if len(df.columns) > 10:
        col_names += f", ... (+{len(df.columns) - 10} more)"

    offset = min(params.offset, total)
    page_size = params.page_size
    page_df = df.iloc[offset : offset + page_size]
    page_records = page_df.where(page_df.notna(), None).to_dict(orient="records")
    has_more = offset + page_size < total
    next_offset = offset + page_size if has_more else None

    # Widget JSON: HTTP mode includes results_url, stdio mode is plain records
    if state.is_http and state.mcp_server_url:
        results_url = f"{state.mcp_server_url}/api/results/{task_id}?format=json"
        csv_download_url = f"{state.mcp_server_url}/api/results/{task_id}?token={download_token}&format=csv"
        widget_data: dict[str, Any] = {
            "results_url": results_url,
            "download_token": download_token,
            "csv_url": csv_download_url,
            "preview": page_records,
            "total": total,
        }
        if session_url:
            widget_data["session_url"] = session_url
        widget_json = json.dumps(widget_data)
    else:
        widget_json = json.dumps(page_records)

    # Token-based page_size recommendation
    recommended = _recommend_page_size(widget_json, len(page_df), total, page_size)

    # Summary text for the LLM
    csv_url = ""
    if state.mcp_server_url:
        csv_url = f"{state.mcp_server_url}/api/results/{task_id}?token={download_token}&format=csv"

    default_page_size = ResultsInput.model_fields["page_size"].default
    if has_more:
        # Use recommendation in the call hint; fall back to current page_size
        hint_page_size = recommended if recommended is not None else page_size
        page_size_arg = (
            f", page_size={hint_page_size}"
            if hint_page_size != default_page_size
            else ""
        )
        summary = (
            f"Results: {total} rows, {len(df.columns)} columns ({col_names}). "
            f"Showing rows {offset + 1}-{offset + len(page_df)} of {total}.\n"
        )
        if recommended is not None:
            summary += f"Recommended page_size={recommended} for these results.\n"
        summary += f"Call everyrow_results(task_id='{task_id}', offset={next_offset}{page_size_arg}) for the next page."
        if csv_url and offset == 0:
            summary += f"\nFull CSV download: {csv_url}\nShare this download link with the user."
    elif offset == 0:
        summary = f"Results: {total} rows, {len(df.columns)} columns ({col_names}). All rows shown."
        if csv_url:
            summary += f"\nFull CSV download: {csv_url}\nShare this download link with the user."
    else:
        summary = f"Results: showing rows {offset + 1}-{offset + len(page_df)} of {total} (final page)."

    return _with_ui(
        widget_json,
        TextContent(type="text", text=summary),
    )
