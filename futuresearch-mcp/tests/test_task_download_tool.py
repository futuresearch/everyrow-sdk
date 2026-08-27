from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from futuresearch.errors import FuturesearchClientError

from futuresearch_mcp.models import TaskDownloadInput
from futuresearch_mcp.tools import futuresearch_task_download
from tests.conftest import make_test_context

TASK_ID = str(uuid4())
OWNER = "user-a"
OTHER = "user-b"


def _access_token(client_id: str | None):
    if client_id is None:
        return None
    token = MagicMock()
    token.client_id = client_id
    token.expires_at = 2**31
    return token


async def _call(
    *,
    caller: str | None = OWNER,
    stored_owner: str | None = OWNER,
    stored_poll_token: str | None = "live-token",
    status_error: Exception | None = None,
):
    client = MagicMock()
    client.token = "jwt-abc"
    ctx = make_test_context(client, mcp_server_url="https://mcp.test")
    status = MagicMock()

    with (
        patch(
            "futuresearch_mcp.tools.get_task_status_tasks_task_id_status_get.asyncio_detailed",
            status,
        ),
        patch(
            "futuresearch_mcp.tools._call_and_check",
            AsyncMock(side_effect=status_error),
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
        result = await futuresearch_task_download(
            TaskDownloadInput(task_id=TASK_ID), ctx
        )
    return result, minted


@pytest.mark.asyncio
async def test_live_records_hand_back_the_existing_token():
    """Nothing is rewritten while the records the download needs are alive."""
    result, minted = await _call()

    assert result.isError is not True
    assert result.structuredContent["poll_token"] == "live-token"
    assert result.structuredContent["download_url"].endswith(
        f"/api/results/{TASK_ID}/download"
    )
    minted.assert_not_awaited()


@pytest.mark.asyncio
async def test_expired_records_are_rewritten():
    """A day on, the records are gone and the link has to be re-established."""
    result, minted = await _call(stored_owner=None, stored_poll_token=None)

    assert result.structuredContent["poll_token"] == "freshly-minted"
    minted.assert_awaited_once()


@pytest.mark.asyncio
async def test_another_accounts_task_gets_no_link():
    """Being able to read a task is not enough to take over the link to it."""
    result, minted = await _call(caller=OTHER)

    assert result.isError is True
    minted.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_auth_context_gets_no_link():
    result, minted = await _call(caller=None)

    assert result.isError is True
    minted.assert_not_awaited()


@pytest.mark.asyncio
async def test_unreadable_task_gets_no_link():
    """The engine, not Redis, is what says whether the caller may have a link."""
    result, minted = await _call(
        status_error=FuturesearchClientError("no", status_code=404),
        stored_owner=None,
        stored_poll_token=None,
    )

    assert result.isError is True
    minted.assert_not_awaited()
