"""Tests for Redis-backed methods on ServerState.

Uses fakeredis to test token and result metadata round-trips
without requiring a real Redis instance.
"""

from __future__ import annotations

import json

import fakeredis.aioredis
import pytest

from everyrow_mcp.state import ServerState


@pytest.fixture
def fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def server_state(fake_redis) -> ServerState:
    """A ServerState wired to fakeredis."""
    s = ServerState()
    s.redis = fake_redis
    return s


class TestTaskTokenRoundTrip:
    """store_task_token → get_task_token → pop_task_token"""

    @pytest.mark.asyncio
    async def test_store_and_get(self, server_state):
        await server_state.store_task_token("task-1", "api-key-abc")
        result = await server_state.get_task_token("task-1")
        assert result == "api-key-abc"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, server_state):
        result = await server_state.get_task_token("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_pop_removes_both_tokens(self, server_state):
        await server_state.store_task_token("task-2", "key")
        await server_state.store_poll_token("task-2", "poll-tok")

        await server_state.pop_task_token("task-2")

        assert await server_state.get_task_token("task-2") is None
        assert await server_state.get_poll_token("task-2") is None


class TestPollTokenRoundTrip:
    """store_poll_token → get_poll_token"""

    @pytest.mark.asyncio
    async def test_store_and_get(self, server_state):
        await server_state.store_poll_token("task-p", "poll-secret")
        result = await server_state.get_poll_token("task-p")
        assert result == "poll-secret"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, server_state):
        result = await server_state.get_poll_token("ghost")
        assert result is None


class TestResultMetaRoundTrip:
    """store_result_meta → get_result_meta"""

    @pytest.mark.asyncio
    async def test_store_and_get(self, server_state):
        meta = json.dumps({"total": 42, "columns": ["a", "b"]})
        await server_state.store_result_meta("task-m", meta)

        raw = await server_state.get_result_meta("task-m")
        assert raw is not None
        parsed = json.loads(raw)
        assert parsed["total"] == 42
        assert parsed["columns"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, server_state):
        result = await server_state.get_result_meta("nope")
        assert result is None


class TestResultPageRoundTrip:
    """store_result_page → get_result_page"""

    @pytest.mark.asyncio
    async def test_store_and_get(self, server_state):
        page = json.dumps([{"id": 1}, {"id": 2}])
        await server_state.store_result_page("task-pg", 0, 10, page)

        result = await server_state.get_result_page("task-pg", 0, 10)
        assert result is not None
        assert json.loads(result) == [{"id": 1}, {"id": 2}]

    @pytest.mark.asyncio
    async def test_different_offsets_are_independent(self, server_state):
        page0 = json.dumps([{"row": 0}])
        page10 = json.dumps([{"row": 10}])
        await server_state.store_result_page("task-multi", 0, 10, page0)
        await server_state.store_result_page("task-multi", 10, 10, page10)

        assert json.loads(await server_state.get_result_page("task-multi", 0, 10)) == [
            {"row": 0}
        ]
        assert json.loads(await server_state.get_result_page("task-multi", 10, 10)) == [
            {"row": 10}
        ]

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, server_state):
        result = await server_state.get_result_page("nothing", 0, 10)
        assert result is None


class TestNoRedis:
    """When redis is None, all methods gracefully return None."""

    @pytest.fixture
    def no_redis_state(self) -> ServerState:
        return ServerState()  # redis=None by default

    @pytest.mark.asyncio
    async def test_task_token_noop(self, no_redis_state):
        await no_redis_state.store_task_token("x", "y")  # should not raise
        assert await no_redis_state.get_task_token("x") is None

    @pytest.mark.asyncio
    async def test_poll_token_noop(self, no_redis_state):
        await no_redis_state.store_poll_token("x", "y")
        assert await no_redis_state.get_poll_token("x") is None

    @pytest.mark.asyncio
    async def test_result_meta_noop(self, no_redis_state):
        await no_redis_state.store_result_meta("x", "{}")
        assert await no_redis_state.get_result_meta("x") is None

    @pytest.mark.asyncio
    async def test_result_page_noop(self, no_redis_state):
        await no_redis_state.store_result_page("x", 0, 10, "[]")
        assert await no_redis_state.get_result_page("x", 0, 10) is None

    @pytest.mark.asyncio
    async def test_pop_noop(self, no_redis_state):
        await no_redis_state.pop_task_token("x")  # should not raise
