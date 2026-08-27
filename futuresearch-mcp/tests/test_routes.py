"""Tests for REST endpoints in routes.py (api_progress)."""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from http import HTTPStatus
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from futuresearch.errors import FuturesearchClientError
from futuresearch.generated.models.public_task_type import PublicTaskType
from futuresearch.generated.models.task_progress_info import TaskProgressInfo
from futuresearch.generated.models.task_status import TaskStatus
from futuresearch.generated.models.task_status_response import TaskStatusResponse
from futuresearch.generated.types import Response

from futuresearch_mcp import redis_store
from futuresearch_mcp.http_config import glama_well_known
from futuresearch_mcp.routes import (
    SESSION_EXPIRED,
    _cors_headers,
    api_progress,
)
from tests.conftest import override_settings

# ── Helpers ────────────────────────────────────────────────────


class FakeRequest:
    """Minimal Starlette Request stand-in for handler tests."""

    def __init__(
        self,
        *,
        method: str = "GET",
        path_params: dict[str, str] | None = None,
        query_params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
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
) -> Response[TaskStatusResponse]:
    body = TaskStatusResponse(
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
    return Response(status_code=HTTPStatus.OK, content=b"", headers={}, parsed=body)


# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _use_fake_redis(fake_redis):
    """Patch get_redis_client to return the test Redis instance."""
    with patch.object(redis_store, "get_redis_client", return_value=fake_redis):
        yield


# ── api_progress tests ─────────────────────────────────────────


class TestApiProgress:
    @pytest.mark.asyncio
    async def test_options_returns_204(self):
        req = FakeRequest(method="OPTIONS", path_params={"task_id": "abc"})
        resp = await api_progress(req)  # pyright: ignore[reportArgumentType]
        assert resp.status_code == 204
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    @pytest.mark.asyncio
    async def test_invalid_poll_token_via_header_returns_403(self):
        task_id = str(uuid4())
        await redis_store.store_poll_token(task_id, "correct-token")
        await redis_store.store_task_token(task_id, "api-key")

        req = FakeRequest(
            path_params={"task_id": task_id},
            headers={"authorization": "Bearer wrong-token"},
        )
        resp = await api_progress(req)  # pyright: ignore[reportArgumentType]
        assert resp.status_code == 403
        body = json.loads(bytes(resp.body).decode())
        assert body["error"] == "Unauthorized"

    @pytest.mark.asyncio
    async def test_missing_poll_token_returns_403(self):
        task_id = str(uuid4())
        # No poll token stored
        req = FakeRequest(
            path_params={"task_id": task_id},
        )
        resp = await api_progress(req)  # pyright: ignore[reportArgumentType]
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_missing_task_token_returns_404(self):
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        await redis_store.store_poll_token(task_id, poll_token)

        # No task token stored

        req = FakeRequest(
            path_params={"task_id": task_id},
            headers={"authorization": f"Bearer {poll_token}"},
        )
        resp = await api_progress(req)  # pyright: ignore[reportArgumentType]
        assert resp.status_code == 404
        body = json.loads(bytes(resp.body).decode())
        assert body["error"] == "Unknown task"

    @pytest.mark.asyncio
    async def test_valid_progress_via_auth_header(self):
        """Poll token sent via Authorization: Bearer header works."""
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        await redis_store.store_poll_token(task_id, poll_token)
        await redis_store.store_task_token(task_id, "api-key-123")

        status_resp = _make_status_response(
            status="running", completed=3, total=10, failed=1, running=2
        )

        req = FakeRequest(
            path_params={"task_id": task_id},
            headers={"authorization": f"Bearer {poll_token}"},
        )

        with patch(
            "futuresearch_mcp.progress.get_task_status_tasks_task_id_status_get.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=status_resp,
        ):
            resp = await api_progress(req)  # pyright: ignore[reportArgumentType]

        assert resp.status_code == 200
        body = json.loads(resp.body.decode())  # pyright: ignore[reportAttributeAccessIssue]
        assert body["status"] == "running"
        assert body["completed"] == 3
        assert body["total"] == 10
        assert body["failed"] == 1
        assert body["running"] == 2
        assert "elapsed_s" in body
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    @pytest.mark.asyncio
    async def test_backward_compat_query_param_for_download(self):
        """Poll token via ?token= query param still works (for download links)."""
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        await redis_store.store_poll_token(task_id, poll_token)
        await redis_store.store_task_token(task_id, "api-key-123")

        status_resp = _make_status_response(
            status="running", completed=3, total=10, failed=1, running=2
        )

        req = FakeRequest(
            path_params={"task_id": task_id},
            query_params={"token": poll_token},
        )

        with patch(
            "futuresearch_mcp.progress.get_task_status_tasks_task_id_status_get.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=status_resp,
        ):
            resp = await api_progress(req)  # pyright: ignore[reportArgumentType]

        assert resp.status_code == 200
        body = json.loads(bytes(resp.body).decode())
        assert body["status"] == "running"
        assert body["completed"] == 3
        assert body["total"] == 10
        assert body["failed"] == 1
        assert body["running"] == 2
        assert "elapsed_s" in body
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    @pytest.mark.asyncio
    async def test_completed_task_pops_tokens(self):
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        await redis_store.store_poll_token(task_id, poll_token)
        await redis_store.store_task_token(task_id, "api-key")

        status_resp = _make_status_response(status="completed", completed=10, total=10)

        req = FakeRequest(
            path_params={"task_id": task_id},
            headers={"authorization": f"Bearer {poll_token}"},
        )

        with patch(
            "futuresearch_mcp.progress.get_task_status_tasks_task_id_status_get.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=status_resp,
        ):
            resp = await api_progress(req)  # pyright: ignore[reportArgumentType]

        assert resp.status_code == 200
        body = json.loads(bytes(resp.body).decode())
        assert body["status"] == "completed"

        # Both tokens kept — task token needed for CSV download, TTL expires them
        assert await redis_store.get_task_token(task_id) is not None
        assert await redis_store.get_poll_token(task_id) is not None

    @pytest.mark.asyncio
    async def test_api_error_returns_500(self):
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        await redis_store.store_poll_token(task_id, poll_token)
        await redis_store.store_task_token(task_id, "api-key")

        req = FakeRequest(
            path_params={"task_id": task_id},
            headers={"authorization": f"Bearer {poll_token}"},
        )

        with patch(
            "futuresearch_mcp.progress.get_task_status_tasks_task_id_status_get.asyncio_detailed",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API down"),
        ):
            resp = await api_progress(req)  # pyright: ignore[reportArgumentType]

        assert resp.status_code == 500
        body = json.loads(bytes(resp.body).decode())
        assert body["error"] == "Internal server error"

    @pytest.mark.asyncio
    async def test_expired_credential_reports_session_expired(self):
        """The incident case: an expired JWT must not surface as a 500.

        A 500 reads as retryable, so the widget polled the same dead
        credential every 10s indefinitely.
        """
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        await redis_store.store_poll_token(task_id, poll_token)
        await redis_store.store_task_token(task_id, "expired-jwt")

        req = FakeRequest(
            path_params={"task_id": task_id},
            headers={"authorization": f"Bearer {poll_token}"},
        )

        with patch(
            "futuresearch_mcp.progress.get_task_status_tasks_task_id_status_get.asyncio_detailed",
            new_callable=AsyncMock,
            side_effect=FuturesearchClientError("JWT has expired", status_code=401),
        ):
            resp = await api_progress(req)  # pyright: ignore[reportArgumentType]

        assert resp.status_code == 401
        body = json.loads(bytes(resp.body).decode())
        assert body["code"] == SESSION_EXPIRED

    @pytest.mark.asyncio
    async def test_owner_without_live_credential_reports_session_expired(self):
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        await redis_store.store_poll_token(task_id, poll_token)
        await redis_store.store_task_owner(task_id, "user-1")

        req = FakeRequest(
            path_params={"task_id": task_id},
            headers={"authorization": f"Bearer {poll_token}"},
        )

        resp = await api_progress(req)  # pyright: ignore[reportArgumentType]

        assert resp.status_code == 401
        assert json.loads(bytes(resp.body).decode())["code"] == SESSION_EXPIRED

    @pytest.mark.asyncio
    async def test_unknown_task_still_404s(self):
        """An unrecorded task is not the same as a lapsed session."""
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        await redis_store.store_poll_token(task_id, poll_token)

        req = FakeRequest(
            path_params={"task_id": task_id},
            headers={"authorization": f"Bearer {poll_token}"},
        )

        resp = await api_progress(req)  # pyright: ignore[reportArgumentType]

        assert resp.status_code == 404
        assert json.loads(bytes(resp.body).decode())["error"] == "Unknown task"

    @pytest.mark.asyncio
    async def test_poll_uses_owners_refreshed_token(self):
        """A task submitted with one JWT polls with whatever is current."""
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        await redis_store.store_poll_token(task_id, poll_token)
        await redis_store.store_task_owner(task_id, "user-1")
        await redis_store.store_user_token("user-1", "jwt-fresh", 3600)

        req = FakeRequest(
            path_params={"task_id": task_id},
            headers={"authorization": f"Bearer {poll_token}"},
        )

        status_resp = _make_status_response(status="running")
        with patch(
            "futuresearch_mcp.progress.get_task_status_tasks_task_id_status_get.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=status_resp,
        ) as mock_status:
            resp = await api_progress(req)  # pyright: ignore[reportArgumentType]

        assert resp.status_code == 200
        assert mock_status.await_args is not None
        assert mock_status.await_args.kwargs["client"].token == "jwt-fresh"

    @pytest.mark.asyncio
    async def test_poll_acts_as_the_submitting_account(self):
        """A team task polls as the team, not as the owner's personal account."""
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        await redis_store.store_poll_token(task_id, poll_token)
        await redis_store.store_task_owner(task_id, "user-1")
        await redis_store.store_user_token("user-1", "jwt-fresh", 3600)
        await redis_store.store_task_account(task_id, "team-acct")

        req = FakeRequest(
            path_params={"task_id": task_id},
            headers={"authorization": f"Bearer {poll_token}"},
        )

        with patch(
            "futuresearch_mcp.progress.get_task_status_tasks_task_id_status_get.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=_make_status_response(status="running"),
        ) as mock_status:
            await api_progress(req)  # pyright: ignore[reportArgumentType]

        assert mock_status.await_args is not None
        client = mock_status.await_args.kwargs["client"]
        headers = client.get_async_httpx_client().headers
        assert headers["x-cohort-account-id"] == "team-acct"

    @pytest.mark.asyncio
    async def test_poll_sends_no_account_when_none_was_selected(self):
        """Personal-only submissions stay header-free rather than guessing one."""
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        await redis_store.store_poll_token(task_id, poll_token)
        await redis_store.store_task_owner(task_id, "user-1")
        await redis_store.store_user_token("user-1", "jwt-fresh", 3600)

        req = FakeRequest(
            path_params={"task_id": task_id},
            headers={"authorization": f"Bearer {poll_token}"},
        )

        with patch(
            "futuresearch_mcp.progress.get_task_status_tasks_task_id_status_get.asyncio_detailed",
            new_callable=AsyncMock,
            return_value=_make_status_response(status="running"),
        ) as mock_status:
            await api_progress(req)  # pyright: ignore[reportArgumentType]

        assert mock_status.await_args is not None
        client = mock_status.await_args.kwargs["client"]
        assert "x-cohort-account-id" not in client.get_async_httpx_client().headers


class TestCorsHeaders:
    """Tests for CORS headers on widget endpoints."""

    def test_returns_wildcard_origin(self):
        headers = _cors_headers()
        assert headers["Access-Control-Allow-Origin"] == "*"
        assert headers["Access-Control-Allow-Methods"] == "GET"
        assert headers["Access-Control-Allow-Headers"] == "Authorization"


class TestGlamaWellKnown:
    """Tests for the Glama.ai connector claim endpoint."""

    @pytest.mark.asyncio
    async def test_returns_claim_body_when_email_configured(self):
        req = FakeRequest()
        with override_settings(glama_maintainer_email="jack@futuresearch.ai"):
            resp = await glama_well_known(req)  # pyright: ignore[reportArgumentType]
        assert resp.status_code == 200
        body = json.loads(bytes(resp.body).decode())
        assert body == {
            "$schema": "https://glama.ai/mcp/schemas/connector.json",
            "maintainers": [{"email": "jack@futuresearch.ai"}],
        }

    @pytest.mark.asyncio
    async def test_returns_404_when_email_empty(self):
        req = FakeRequest()
        with override_settings(glama_maintainer_email=""):
            resp = await glama_well_known(req)  # pyright: ignore[reportArgumentType]
        assert resp.status_code == 404
