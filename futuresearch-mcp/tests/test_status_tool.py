from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from futuresearch.generated.models import PublicTaskType, TaskStatus

from futuresearch_mcp.models import ProgressInput
from futuresearch_mcp.tools import futuresearch_status
from tests.conftest import make_test_context, override_settings

TASK_ID = str(uuid4())
OWNER = "user-a"
OTHER = "user-b"
SERVER = "https://mcp.test"


def _status(status: TaskStatus, *, completed: int = 2, error: str | None = None):
    resp = MagicMock()
    resp.status = status
    resp.task_type = PublicTaskType.FORECAST
    resp.error = error
    resp.progress = MagicMock(completed=completed, failed=0, running=1, total=4)
    resp.artifact_id = None
    return resp


def _access_token(client_id: str | None):
    if client_id is None:
        return None
    token = MagicMock()
    token.client_id = client_id
    token.expires_at = 2**31
    return token


async def _call(
    *,
    status: TaskStatus = TaskStatus.RUNNING,
    error: str | None = None,
    caller: str | None = OWNER,
    stored_owner: str | None = OWNER,
    stored_poll_token: str | None = "live-token",
):
    client = MagicMock()
    client.token = "jwt-abc"
    ctx = make_test_context(client, mcp_server_url=SERVER)

    with (
        override_settings(transport="streamable-http"),
        patch("futuresearch_mcp.tools._get_client", AsyncMock(return_value=client)),
        patch(
            "futuresearch_mcp.tools.get_task_status_tasks_task_id_status_get.asyncio_detailed",
            MagicMock(),
        ),
        patch(
            "futuresearch_mcp.tools._call_and_check",
            AsyncMock(return_value=_status(status, error=error)),
        ),
        patch(
            "futuresearch_mcp.tools.redis_store.get_task_meta",
            AsyncMock(return_value=None),
        ),
        patch(
            "futuresearch_mcp.tool_helpers.get_access_token",
            MagicMock(return_value=_access_token(caller)),
        ),
        patch(
            "futuresearch_mcp.tool_helpers.redis_store.get_task_owner",
            AsyncMock(return_value=stored_owner),
        ),
        patch(
            "futuresearch_mcp.tool_helpers.redis_store.get_poll_token",
            AsyncMock(return_value=stored_poll_token),
        ),
        patch(
            "futuresearch_mcp.tool_helpers._record_task_ownership",
            AsyncMock(return_value="freshly-minted"),
        ) as minted,
    ):
        return await futuresearch_status(ProgressInput(task_id=TASK_ID), ctx), minted


@pytest.mark.asyncio
async def test_running_task_gets_a_widget():
    result, minted = await _call()

    assert result.structuredContent["poll_token"] == "live-token"
    assert result.structuredContent["progress_url"].endswith(f"/api/progress/{TASK_ID}")
    minted.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_older_than_its_records_still_gets_a_widget():
    """The case the widget used to be unreachable in."""
    result, minted = await _call(stored_owner=None, stored_poll_token=None)

    assert result.structuredContent["poll_token"] == "freshly-minted"
    minted.assert_awaited_once()


@pytest.mark.asyncio
async def test_completed_task_gets_a_widget():
    result, _ = await _call(
        status=TaskStatus.COMPLETED, stored_owner=None, stored_poll_token=None
    )

    assert result.structuredContent["status"] == TaskStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_another_accounts_task_reports_state_without_a_widget():
    result, minted = await _call(caller=OTHER)

    assert result.structuredContent is None
    assert str(TASK_ID) in result.content[0].text
    minted.assert_not_awaited()
