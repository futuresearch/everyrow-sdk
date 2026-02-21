"""Helper functions for MCP tool implementations.

Includes transport-aware UI helpers, client construction, task state
persistence, and result fetching.
"""

from __future__ import annotations

import json
import logging
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

from everyrow_mcp.state import TASK_STATE_FILE, state


def _get_client():
    """Get an EveryRow API client for the current request.

    In stdio mode or --no-auth HTTP mode, returns the singleton client
    initialized at startup. In HTTP mode with auth, creates a per-request
    client using the authenticated user's API key from the OAuth access token.
    """
    # Singleton client (stdio mode or --no-auth HTTP mode)
    if state.client is not None:
        return state.client
    # HTTP mode: get JWT from authenticated request
    access_token = get_access_token()
    if access_token is None:
        raise RuntimeError("Not authenticated")
    return AuthenticatedClient(
        base_url=state.everyrow_api_url,
        token=access_token.token,
        raise_on_unexpected_status=True,
        follow_redirects=True,
    )


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
    poll_token = secrets.token_urlsafe(32)
    if state.store is not None:
        await state.store.store_task_token(task_id, token)
        await state.store.store_poll_token(task_id, poll_token)
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
