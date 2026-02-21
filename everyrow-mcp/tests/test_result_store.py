"""Tests for Redis-backed result retrieval (result_store.py).

Covers pure helpers (_format_columns, _slice_preview, _build_result_response)
and async functions (try_cached_result, try_store_result) with mocked state,
plus the download endpoint (api_download).
"""

from __future__ import annotations

import io
import json
import secrets
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pandas as pd
import pytest
from starlette.applications import Starlette
from starlette.routing import Route

from everyrow_mcp.result_store import (
    _build_result_response,
    _format_columns,
    _slice_preview,
    try_cached_result,
    try_store_result,
)
from everyrow_mcp.routes import api_download
from everyrow_mcp.state import RedisStore, state

# ── Fixtures ───────────────────────────────────────────────────

FAKE_SERVER_URL = "http://testserver"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"name": ["Alice", "Bob", "Carol"], "score": [95, 87, 72]})


@pytest.fixture
def _http_state(fake_redis):
    """Configure global state for HTTP mode and restore after test."""
    orig = {
        "transport": state.transport,
        "store": state.store,
        "mcp_server_url": state.mcp_server_url,
    }

    state.transport = "streamable-http"
    state.store = RedisStore(fake_redis)
    state.mcp_server_url = FAKE_SERVER_URL

    yield

    state.transport = orig["transport"]
    state.store = orig["store"]
    state.mcp_server_url = orig["mcp_server_url"]


# ── Pure helpers ───────────────────────────────────────────────


class TestFormatColumns:
    def test_few_columns(self):
        assert _format_columns(["a", "b", "c"]) == "a, b, c"

    def test_exactly_ten(self):
        cols = [f"c{i}" for i in range(10)]
        result = _format_columns(cols)
        assert result == ", ".join(cols)
        assert "more" not in result

    def test_more_than_ten(self):
        cols = [f"c{i}" for i in range(15)]
        result = _format_columns(cols)
        assert "(+5 more)" in result
        # First 10 should still be present
        for c in cols[:10]:
            assert c in result


class TestSlicePreview:
    def test_basic_slice(self):
        records = [{"id": i} for i in range(10)]
        assert _slice_preview(records, 0, 3) == [{"id": 0}, {"id": 1}, {"id": 2}]

    def test_offset_slice(self):
        records = [{"id": i} for i in range(10)]
        assert _slice_preview(records, 5, 3) == [{"id": 5}, {"id": 6}, {"id": 7}]

    def test_offset_past_end(self):
        records = [{"id": i} for i in range(3)]
        assert _slice_preview(records, 10, 5) == []

    def test_page_extends_past_end(self):
        records = [{"id": i} for i in range(5)]
        result = _slice_preview(records, 3, 10)
        assert result == [{"id": 3}, {"id": 4}]

    def test_empty_records(self):
        assert _slice_preview([], 0, 10) == []


class TestBuildResultResponse:
    def test_all_rows_shown(self):
        preview = [{"name": "Alice"}, {"name": "Bob"}]
        csv_url = f"{FAKE_SERVER_URL}/api/results/task-123/download?token=abc"
        result = _build_result_response(
            task_id="task-123",
            csv_url=csv_url,
            preview=preview,
            total=2,
            columns=["name"],
            offset=0,
            page_size=10,
        )
        assert len(result) == 2
        widget = json.loads(result[0].text)
        assert widget["total"] == 2
        assert widget["csv_url"] == csv_url
        assert "All rows shown" in result[1].text

    def test_has_more_pages(self):
        preview = [{"id": i} for i in range(5)]
        csv_url = f"{FAKE_SERVER_URL}/api/results/task-456/download?token=abc"
        result = _build_result_response(
            task_id="task-456",
            csv_url=csv_url,
            preview=preview,
            total=20,
            columns=["id"],
            offset=0,
            page_size=5,
        )
        summary = result[1].text
        assert "20 rows" in summary
        assert "offset=5" in summary
        assert "everyrow_results" in summary
        # First page includes CSV download link
        assert csv_url in summary

    def test_final_page(self):
        preview = [{"id": 18}, {"id": 19}]
        csv_url = f"{FAKE_SERVER_URL}/api/results/task-789/download?token=abc"
        result = _build_result_response(
            task_id="task-789",
            csv_url=csv_url,
            preview=preview,
            total=20,
            columns=["id"],
            offset=18,
            page_size=5,
        )
        summary = result[1].text
        assert "final page" in summary

    def test_session_url_included_in_widget(self):
        preview = [{"a": 1}]
        csv_url = f"{FAKE_SERVER_URL}/api/results/task-url/download?token=abc"
        result = _build_result_response(
            task_id="task-url",
            csv_url=csv_url,
            preview=preview,
            total=1,
            columns=["a"],
            offset=0,
            page_size=10,
            session_url="https://everyrow.io/sessions/abc",
        )
        widget = json.loads(result[0].text)
        assert widget["session_url"] == "https://everyrow.io/sessions/abc"

    def test_no_session_url_when_empty(self):
        preview = [{"a": 1}]
        csv_url = f"{FAKE_SERVER_URL}/api/results/task-nurl/download?token=abc"
        result = _build_result_response(
            task_id="task-nurl",
            csv_url=csv_url,
            preview=preview,
            total=1,
            columns=["a"],
            offset=0,
            page_size=10,
        )
        widget = json.loads(result[0].text)
        assert "session_url" not in widget


# ── Async functions ────────────────────────────────────────────


class TestTryCachedResult:
    @pytest.mark.asyncio
    async def test_returns_none_when_redis_not_configured(self):
        with patch.object(state, "store", None):
            result = await try_cached_result("task-1", 0, 10)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_cached_meta(self, _http_state):
        result = await try_cached_result("task-2", 0, 10)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_cached_page(self, _http_state):
        meta = json.dumps({"total": 3, "columns": ["name", "score"]})
        page = json.dumps([{"name": "Alice", "score": 95}])
        task_id = "task-3"
        poll_token = "test-token"

        await state.store.store_result_meta(task_id, meta)
        await state.store.store_result_page(task_id, 0, 1, page)
        await state.store.store_poll_token(task_id, poll_token)

        result = await try_cached_result(task_id, 0, 1)

        assert result is not None
        assert len(result) == 2
        widget = json.loads(result[0].text)
        assert widget["total"] == 3

    @pytest.mark.asyncio
    async def test_reads_csv_on_page_miss(self, _http_state):
        meta = json.dumps({"total": 3, "columns": ["name", "score"]})
        csv_text = "name,score\nAlice,95\nBob,87\nCarol,72\n"
        task_id = "task-4"

        await state.store.store_result_meta(task_id, meta)
        await state.store.store_result_csv(task_id, csv_text)

        result = await try_cached_result(task_id, 0, 2)

        assert result is not None
        widget = json.loads(result[0].text)
        assert len(widget["preview"]) == 2

    @pytest.mark.asyncio
    async def test_preserves_session_url_from_meta(self, _http_state):
        meta = json.dumps(
            {
                "total": 1,
                "columns": ["a"],
                "session_url": "https://everyrow.io/sessions/xyz",
            }
        )
        page = json.dumps([{"a": 1}])
        task_id = "task-5"

        await state.store.store_result_meta(task_id, meta)
        await state.store.store_result_page(task_id, 0, 10, page)

        result = await try_cached_result(task_id, 0, 10)

        widget = json.loads(result[0].text)
        assert widget["session_url"] == "https://everyrow.io/sessions/xyz"


class TestTryStoreResult:
    @pytest.mark.asyncio
    async def test_returns_none_when_redis_not_configured(self, sample_df):
        with patch.object(state, "store", None):
            result = await try_store_result("task-1", sample_df, 0, 10)
        assert result is None

    @pytest.mark.asyncio
    async def test_stores_and_returns_response(self, sample_df, _http_state):
        task_id = "task-up"
        await state.store.store_poll_token(task_id, "test-token")

        result = await try_store_result(task_id, sample_df, 0, 2)

        assert result is not None
        assert len(result) == 2
        widget = json.loads(result[0].text)
        assert widget["total"] == 3
        assert len(widget["preview"]) == 2

        # Verify CSV was stored in Redis
        stored_csv = await state.store.get_result_csv(task_id)
        assert stored_csv is not None
        df = pd.read_csv(io.StringIO(stored_csv))
        assert len(df) == 3

        # Verify metadata was cached
        meta_raw = await state.store.get_result_meta(task_id)
        assert meta_raw is not None
        meta = json.loads(meta_raw)
        assert meta["total"] == 3
        assert meta["columns"] == ["name", "score"]

    @pytest.mark.asyncio
    async def test_includes_session_url_in_meta(self, sample_df, _http_state):
        task_id = "task-sess"
        await state.store.store_poll_token(task_id, "test-token")

        await try_store_result(
            task_id,
            sample_df,
            0,
            10,
            session_url="https://everyrow.io/sessions/abc",
        )

        meta_raw = await state.store.get_result_meta(task_id)
        meta = json.loads(meta_raw)
        assert meta["session_url"] == "https://everyrow.io/sessions/abc"

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self, sample_df, _http_state):
        with patch.object(
            state.store,
            "store_result_csv",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Redis down"),
        ):
            result = await try_store_result("task-fail", sample_df, 0, 10)
        assert result is None


# ── Download endpoint ──────────────────────────────────────────


class TestApiDownload:
    @pytest.fixture
    def app(self, _http_state):
        return Starlette(
            routes=[
                Route(
                    "/api/results/{task_id}/download",
                    api_download,
                    methods=["GET", "OPTIONS"],
                ),
            ],
        )

    @pytest.fixture
    async def client(self, app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as c:
            yield c

    @pytest.mark.asyncio
    async def test_valid_download(self, client: httpx.AsyncClient):
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)
        csv_text = "name,score\nAlice,95\nBob,87\n"

        await state.store.store_poll_token(task_id, poll_token)
        await state.store.store_result_csv(task_id, csv_text)

        resp = await client.get(
            f"/api/results/{task_id}/download", params={"token": poll_token}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/csv; charset=utf-8"
        assert "attachment" in resp.headers["content-disposition"]
        assert resp.text == csv_text

    @pytest.mark.asyncio
    async def test_bad_token_returns_403(self, client: httpx.AsyncClient):
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)

        await state.store.store_poll_token(task_id, poll_token)
        await state.store.store_result_csv(task_id, "data")

        resp = await client.get(
            f"/api/results/{task_id}/download", params={"token": "wrong-token"}
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_missing_csv_returns_404(self, client: httpx.AsyncClient):
        task_id = str(uuid4())
        poll_token = secrets.token_urlsafe(16)

        await state.store.store_poll_token(task_id, poll_token)
        # No CSV stored

        resp = await client.get(
            f"/api/results/{task_id}/download", params={"token": poll_token}
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cors_preflight(self, client: httpx.AsyncClient):
        resp = await client.options("/api/results/some-task/download")
        assert resp.status_code == 204
        assert resp.headers["access-control-allow-origin"] == "*"
