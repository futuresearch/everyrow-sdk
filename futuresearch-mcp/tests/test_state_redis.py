"""Tests for redis_store standalone functions and Settings transport properties."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from futuresearch_mcp import redis_store
from futuresearch_mcp.config import Settings


@pytest.fixture(autouse=True)
def _use_fake_redis(fake_redis):
    """Patch get_redis_client to return the test Redis instance."""
    with patch.object(redis_store, "get_redis_client", return_value=fake_redis):
        yield


class TestTaskTokenRoundTrip:
    """store_task_token -> get_task_token"""

    @pytest.mark.asyncio
    async def test_store_and_get(self):
        await redis_store.store_task_token("task-1", "api-key-abc")
        result = await redis_store.get_task_token("task-1")
        assert result == "api-key-abc"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self):
        result = await redis_store.get_task_token("nonexistent")
        assert result is None


class TestTaskCredentialResolution:
    """get_task_credential prefers the owner's live JWT over a frozen copy."""

    @pytest.mark.asyncio
    async def test_resolves_owner_current_token(self):
        await redis_store.store_task_owner("task-o", "user-1")
        await redis_store.store_user_token("user-1", "jwt-v1", 3600)

        assert await redis_store.get_task_credential("task-o") == "jwt-v1"

    @pytest.mark.asyncio
    async def test_reflects_token_refresh(self):
        """A refresh mid-task must be picked up by later polls."""
        await redis_store.store_task_owner("task-o", "user-1")
        await redis_store.store_user_token("user-1", "jwt-v1", 3600)
        await redis_store.store_user_token("user-1", "jwt-v2", 3600)

        assert await redis_store.get_task_credential("task-o") == "jwt-v2"

    @pytest.mark.asyncio
    async def test_owner_without_live_token_returns_none(self):
        await redis_store.store_task_owner("task-o", "user-1")

        assert await redis_store.get_task_credential("task-o") is None

    @pytest.mark.asyncio
    async def test_falls_back_to_per_task_credential(self):
        """API-key submissions, and tasks recorded before owner mapping."""
        await redis_store.store_task_token("task-legacy", "sk-cho-abc")

        assert await redis_store.get_task_credential("task-legacy") == "sk-cho-abc"

    @pytest.mark.asyncio
    async def test_unknown_task_returns_none(self):
        assert await redis_store.get_task_credential("ghost") is None

    @pytest.mark.asyncio
    async def test_expired_ttl_is_not_stored(self):
        await redis_store.store_user_token("user-2", "jwt-dead", 0)

        assert await redis_store.get_user_token("user-2") is None


class TestPollTokenRoundTrip:
    """store_poll_token -> get_poll_token"""

    @pytest.mark.asyncio
    async def test_store_and_get(self):
        await redis_store.store_poll_token("task-p", "poll-secret")
        result = await redis_store.get_poll_token("task-p")
        assert result == "poll-secret"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self):
        result = await redis_store.get_poll_token("ghost")
        assert result is None


class TestSettingsTransport:
    """Settings transport properties."""

    def test_is_stdio_by_default(self):
        s = Settings()  # pyright: ignore[reportCallIssue]
        assert s.is_stdio is True
        assert s.is_http is False

    def test_transport_http(self):
        s = Settings(transport="streamable-http")  # pyright: ignore[reportCallIssue]
        assert s.is_http is True
        assert s.is_stdio is False
