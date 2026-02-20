"""Tests for GCS-backed result retrieval (gcs_results.py).

Covers pure helpers (_format_columns, _slice_preview, _build_gcs_response)
and async functions (try_cached_gcs_result, try_upload_gcs_result) with
mocked state and GCS store.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from everyrow_mcp.gcs_results import (
    _build_gcs_response,
    _format_columns,
    _slice_preview,
    try_cached_gcs_result,
    try_upload_gcs_result,
)
from everyrow_mcp.state import state

# ── Fixtures ───────────────────────────────────────────────────

FAKE_CSV_URL = "https://storage.googleapis.com/bucket/results/task/data.csv"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"name": ["Alice", "Bob", "Carol"], "score": [95, 87, 72]})


@pytest.fixture
def mock_gcs_store():
    """A MagicMock standing in for GCSResultStore."""
    store = MagicMock()
    store.upload_result.return_value = FAKE_CSV_URL
    store.generate_signed_url.return_value = FAKE_CSV_URL
    store.download_csv.return_value = [
        {"name": "Alice", "score": 95},
        {"name": "Bob", "score": 87},
        {"name": "Carol", "score": 72},
    ]
    return store


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


class TestBuildGcsResponse:
    def test_all_rows_shown(self):
        preview = [{"name": "Alice"}, {"name": "Bob"}]
        result = _build_gcs_response(
            task_id="task-123",
            csv_url=FAKE_CSV_URL,
            preview=preview,
            total=2,
            columns=["name"],
            offset=0,
            page_size=10,
        )
        assert len(result) == 2
        widget = json.loads(result[0].text)
        assert widget["total"] == 2
        assert widget["csv_url"] == FAKE_CSV_URL
        assert "All rows shown" in result[1].text

    def test_has_more_pages(self):
        preview = [{"id": i} for i in range(5)]
        result = _build_gcs_response(
            task_id="task-456",
            csv_url=FAKE_CSV_URL,
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
        assert FAKE_CSV_URL in summary

    def test_final_page(self):
        preview = [{"id": 18}, {"id": 19}]
        result = _build_gcs_response(
            task_id="task-789",
            csv_url=FAKE_CSV_URL,
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
        result = _build_gcs_response(
            task_id="task-url",
            csv_url=FAKE_CSV_URL,
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
        result = _build_gcs_response(
            task_id="task-nurl",
            csv_url=FAKE_CSV_URL,
            preview=preview,
            total=1,
            columns=["a"],
            offset=0,
            page_size=10,
        )
        widget = json.loads(result[0].text)
        assert "session_url" not in widget


# ── Async functions ────────────────────────────────────────────


class TestTryCachedGcsResult:
    @pytest.mark.asyncio
    async def test_returns_none_when_gcs_not_configured(self):
        with patch.object(state, "gcs_store", None):
            result = await try_cached_gcs_result("task-1", 0, 10)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_cached_meta(self, mock_gcs_store):
        with (
            patch.object(state, "gcs_store", mock_gcs_store),
            patch.object(
                state, "get_result_meta", new_callable=AsyncMock, return_value=None
            ),
        ):
            result = await try_cached_gcs_result("task-2", 0, 10)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_cached_page(self, mock_gcs_store):
        meta = json.dumps({"total": 3, "columns": ["name", "score"]})
        page = json.dumps([{"name": "Alice", "score": 95}])

        with (
            patch.object(state, "gcs_store", mock_gcs_store),
            patch.object(
                state, "get_result_meta", new_callable=AsyncMock, return_value=meta
            ),
            patch.object(
                state, "get_result_page", new_callable=AsyncMock, return_value=page
            ),
        ):
            result = await try_cached_gcs_result("task-3", 0, 1)

        assert result is not None
        assert len(result) == 2
        widget = json.loads(result[0].text)
        assert widget["total"] == 3

    @pytest.mark.asyncio
    async def test_fetches_from_gcs_on_page_miss(self, mock_gcs_store):
        meta = json.dumps({"total": 3, "columns": ["name", "score"]})

        with (
            patch.object(state, "gcs_store", mock_gcs_store),
            patch.object(
                state, "get_result_meta", new_callable=AsyncMock, return_value=meta
            ),
            patch.object(
                state, "get_result_page", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                state, "store_result_page", new_callable=AsyncMock
            ) as mock_store_page,
        ):
            result = await try_cached_gcs_result("task-4", 0, 2)

        assert result is not None
        mock_gcs_store.download_csv.assert_called_once_with("task-4")
        # Should cache the page for next time
        mock_store_page.assert_called_once()

    @pytest.mark.asyncio
    async def test_preserves_session_url_from_meta(self, mock_gcs_store):
        meta = json.dumps(
            {
                "total": 1,
                "columns": ["a"],
                "session_url": "https://everyrow.io/sessions/xyz",
            }
        )
        page = json.dumps([{"a": 1}])

        with (
            patch.object(state, "gcs_store", mock_gcs_store),
            patch.object(
                state, "get_result_meta", new_callable=AsyncMock, return_value=meta
            ),
            patch.object(
                state, "get_result_page", new_callable=AsyncMock, return_value=page
            ),
        ):
            result = await try_cached_gcs_result("task-5", 0, 10)

        widget = json.loads(result[0].text)
        assert widget["session_url"] == "https://everyrow.io/sessions/xyz"


class TestTryUploadGcsResult:
    @pytest.mark.asyncio
    async def test_returns_none_when_gcs_not_configured(self, sample_df):
        with patch.object(state, "gcs_store", None):
            result = await try_upload_gcs_result("task-1", sample_df, 0, 10)
        assert result is None

    @pytest.mark.asyncio
    async def test_uploads_and_returns_response(self, sample_df, mock_gcs_store):
        with (
            patch.object(state, "gcs_store", mock_gcs_store),
            patch.object(
                state, "store_result_meta", new_callable=AsyncMock
            ) as mock_meta,
            patch.object(
                state, "store_result_page", new_callable=AsyncMock
            ) as mock_page,
        ):
            result = await try_upload_gcs_result("task-up", sample_df, 0, 2)

        assert result is not None
        assert len(result) == 2
        widget = json.loads(result[0].text)
        assert widget["total"] == 3
        assert len(widget["preview"]) == 2  # page_size=2

        # Verify metadata was cached
        mock_meta.assert_called_once()
        meta_arg = json.loads(mock_meta.call_args[0][1])
        assert meta_arg["total"] == 3
        assert meta_arg["columns"] == ["name", "score"]

        # Verify page was cached
        mock_page.assert_called_once()

    @pytest.mark.asyncio
    async def test_includes_session_url_in_meta(self, sample_df, mock_gcs_store):
        with (
            patch.object(state, "gcs_store", mock_gcs_store),
            patch.object(
                state, "store_result_meta", new_callable=AsyncMock
            ) as mock_meta,
            patch.object(state, "store_result_page", new_callable=AsyncMock),
        ):
            await try_upload_gcs_result(
                "task-sess",
                sample_df,
                0,
                10,
                session_url="https://everyrow.io/sessions/abc",
            )

        meta_arg = json.loads(mock_meta.call_args[0][1])
        assert meta_arg["session_url"] == "https://everyrow.io/sessions/abc"

    @pytest.mark.asyncio
    async def test_returns_none_on_upload_failure(self, sample_df, mock_gcs_store):
        mock_gcs_store.upload_result.side_effect = RuntimeError("GCS down")
        with (
            patch.object(state, "gcs_store", mock_gcs_store),
        ):
            result = await try_upload_gcs_result("task-fail", sample_df, 0, 10)
        assert result is None
