"""REST endpoints for the everyrow MCP server (progress polling)."""

from __future__ import annotations

import logging
import secrets
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from everyrow.api_utils import handle_response
from everyrow.generated.api.tasks import get_task_status_tasks_task_id_status_get
from everyrow.generated.client import AuthenticatedClient
from everyrow.generated.models.task_status import TaskStatus
from everyrow.session import get_session_url
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from everyrow_mcp.state import state

logger = logging.getLogger(__name__)


async def api_progress(request: Request) -> Any:
    """REST endpoint for the session widget to poll task progress."""
    cors = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET"}

    if request.method == "OPTIONS":
        return Response(status_code=204, headers=cors)

    task_id = request.path_params["task_id"]

    # Validate poll token
    expected_poll = await state.get_poll_token(task_id)
    request_poll = request.query_params.get("token", "")
    if not expected_poll or not secrets.compare_digest(request_poll, expected_poll):
        return JSONResponse({"error": "Unauthorized"}, status_code=403, headers=cors)

    api_key = await state.get_task_token(task_id)
    if not api_key:
        return JSONResponse({"error": "Unknown task"}, status_code=404, headers=cors)

    try:
        if state.settings is None:
            raise RuntimeError("MCP server not initialized")
        client = AuthenticatedClient(
            base_url=state.settings.everyrow_api_url,
            token=api_key,
            raise_on_unexpected_status=True,
            follow_redirects=True,
        )
        status_response = handle_response(
            await get_task_status_tasks_task_id_status_get.asyncio(
                task_id=UUID(task_id),
                client=client,
            )
        )

        status = status_response.status
        progress = status_response.progress
        session_url = get_session_url(status_response.session_id)

        completed = progress.completed if progress else 0
        failed = progress.failed if progress else 0
        running = progress.running if progress else 0
        total = progress.total if progress else 0

        elapsed_s = 0
        if status_response.created_at:
            created_at = status_response.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            if (
                status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.REVOKED)
                and status_response.updated_at
            ):
                updated_at = status_response.updated_at
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=UTC)
                elapsed_s = round((updated_at - created_at).total_seconds())
            else:
                elapsed_s = round((datetime.now(UTC) - created_at).total_seconds())

        data = {
            "status": status.value,
            "completed": completed,
            "total": total,
            "failed": failed,
            "running": running,
            "elapsed_s": elapsed_s,
            "session_url": session_url,
        }

        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.REVOKED):
            await state.pop_task_token(task_id)

        return JSONResponse(data, headers=cors)
    except Exception:
        logger.exception("Progress poll failed for task %s", task_id)
        return JSONResponse(
            {"error": "Internal server error"}, status_code=500, headers=cors
        )


async def api_download(request: Request) -> Any:
    """REST endpoint to download task results as CSV."""
    cors = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET"}

    if request.method == "OPTIONS":
        return Response(status_code=204, headers=cors)

    task_id = request.path_params["task_id"]

    # Validate poll token
    expected_poll = await state.get_poll_token(task_id)
    request_poll = request.query_params.get("token", "")
    if not expected_poll or not secrets.compare_digest(request_poll, expected_poll):
        return JSONResponse({"error": "Unauthorized"}, status_code=403, headers=cors)

    csv_text = await state.get_result_csv(task_id)
    if csv_text is None:
        return JSONResponse(
            {"error": "Results not found or expired"}, status_code=404, headers=cors
        )

    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={
            **cors,
            "Content-Disposition": f'attachment; filename="results_{task_id[:8]}.csv"',
        },
    )
