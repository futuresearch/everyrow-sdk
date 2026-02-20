"""Tests for REST endpoints in routes.py (api_progress).

Uses mock Starlette Request objects and fakeredis to test handler logic
without running a real ASGI server.
"""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import fakeredis.aioredis
import pytest
from everyrow.generated.models.public_task_type import PublicTaskType
from everyrow.generated.models.task_progress_info import TaskProgressInfo
from everyrow.generated.models.task_status import TaskStatus
from everyrow.generated.models.task_status_response import TaskStatusResponse

from everyrow_mcp.routes import api_progress
from everyrow_mcp.state import state

# ── Helpers ────────────────────────────────────────────────────


class FakeRequest:
    """Minimal Starlette Request stand-in for handler tests."""

    def __init__(
        self,
        *,
        method: str = "GET",
        path_params: dict | None = None,
        query_params: dict | None = None,
        headers: dict | None = None,
    ):
        self.method = method
        self.path_params = path_params or {}
        self.query_params = query_params or {}
        self.headers = headers or {}


def _make_status_response(
    *,
    task_id=None,
    session_id=None,
    status="running",
    completed=3,
    total=10,
    failed=0,
    running=2,
) -> TaskStatusResponse:
    return TaskStatusResponse(
        task_id=task_id or uuid4(),
        session_id=session_id or uuid4(),
        status=TaskStatus(status),
        task_type=PublicTaskType.AGENT,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        progress=TaskProgressInfo(
            pending=total - completed - failed - running,
            running=running,
            completed=completed,
            failed=failed,
            total=total,
        ),
    )


# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
async def setup_state(fake_redis):
    """Temporarily wire state to fakeredis and restore after test."""
    original_redis = state.redis
    state.redis = fake_redis
    yield
    state.redis = original_redis


# ── api_progress tests ─────────────────────────────────────────


class TestApiProgress:
    @pytest.mark.asyncio
    async def test_options_returns_204(self, setup_state):
        req = FakeRequest(method="OPTIONS", path_params={"task_id": "abc"})
        resp = await api_progress(req)
        assert resp.status_code == 204
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    @pytest.mark.asyncio
    async def test_invalid_poll_token_returns_403(self, setup_state):
        task_id = str(uuid4())
        await state.store_poll_token(task_id, "correct-token")
        await state.store_task_token(task_id, "api-key")

        req = FakeRequest(
            path_params={"task_id": task_id},
            query_params={"token": "wrong-token"},
        )
        resp = await api_progress(req)
        assert resp.status_code == 403
        body = json.loads(resp.body.decode())
        assert body["error"] == "Unauthorized"

    @pytest.mark.asyncio
    async def test_missing_poll_token_returns_403(self, setup_state):
        task_id = str(uuid4())
        # No poll token stored
        req = FakeRequest(
            path_params={"task_id": task_id},
            query_params={},
        )
        resp = await api_progress(req)
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_missing_task_token_returns_404(self, setup_state):
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        await state.store_poll_token(task_id, poll_token)
        # No task token stored

        req = FakeRequest(
            path_params={"task_id": task_id},
            query_params={"token": poll_token},
        )
        resp = await api_progress(req)
        assert resp.status_code == 404
        body = json.loads(resp.body.decode())
        assert body["error"] == "Unknown task"

    @pytest.mark.asyncio
    async def test_valid_progress_returns_status(self, setup_state):
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        await state.store_poll_token(task_id, poll_token)
        await state.store_task_token(task_id, "api-key-123")

        status_resp = _make_status_response(
            status="running", completed=3, total=10, failed=1, running=2
        )

        req = FakeRequest(
            path_params={"task_id": task_id},
            query_params={"token": poll_token},
        )

        with patch(
            "everyrow_mcp.routes.get_task_status_tasks_task_id_status_get.asyncio",
            new_callable=AsyncMock,
            return_value=status_resp,
        ):
            resp = await api_progress(req)

        assert resp.status_code == 200
        body = json.loads(resp.body.decode())
        assert body["status"] == "running"
        assert body["completed"] == 3
        assert body["total"] == 10
        assert body["failed"] == 1
        assert body["running"] == 2
        assert "elapsed_s" in body
        assert "session_url" in body
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    @pytest.mark.asyncio
    async def test_completed_task_pops_tokens(self, setup_state):
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        await state.store_poll_token(task_id, poll_token)
        await state.store_task_token(task_id, "api-key")

        status_resp = _make_status_response(status="completed", completed=10, total=10)

        req = FakeRequest(
            path_params={"task_id": task_id},
            query_params={"token": poll_token},
        )

        with patch(
            "everyrow_mcp.routes.get_task_status_tasks_task_id_status_get.asyncio",
            new_callable=AsyncMock,
            return_value=status_resp,
        ):
            resp = await api_progress(req)

        assert resp.status_code == 200
        body = json.loads(resp.body.decode())
        assert body["status"] == "completed"

        # Tokens should be cleaned up
        assert await state.get_task_token(task_id) is None
        assert await state.get_poll_token(task_id) is None

    @pytest.mark.asyncio
    async def test_api_error_returns_500(self, setup_state):
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        await state.store_poll_token(task_id, poll_token)
        await state.store_task_token(task_id, "api-key")

        req = FakeRequest(
            path_params={"task_id": task_id},
            query_params={"token": poll_token},
        )

        with patch(
            "everyrow_mcp.routes.get_task_status_tasks_task_id_status_get.asyncio",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API down"),
        ):
            resp = await api_progress(req)

        assert resp.status_code == 500
        body = json.loads(resp.body.decode())
        assert body["error"] == "Internal server error"
