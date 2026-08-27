from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from futuresearch.errors import FuturesearchClientError
from futuresearch.generated.models import (
    ForecastSpec,
    ForecastSpecForecastType,
    PublicTaskType,
    TaskDetailResponse,
    UnconditionalFraming,
)

from futuresearch_mcp.models import TaskDataInput
from futuresearch_mcp.tools import futuresearch_task_data
from tests.conftest import make_test_context

TASK_ID = str(uuid4())
ROWS = [{"question": "Revenue?", "revenue_p50": 412.0}]


@dataclass
class _Calls:
    progress: AsyncMock
    detail: AsyncMock
    results: AsyncMock


def _progress(status: str = "running") -> dict:
    return {"status": status, "completed": 1, "total": 4, "cursor": "c1"}


def _detail(spec: ForecastSpec | None) -> TaskDetailResponse:
    return TaskDetailResponse(
        task_id=uuid4(),
        session_id=uuid4(),
        task_type=PublicTaskType.FORECAST if spec else PublicTaskType.RANK,
        created_at=None,
        spec=spec,
    )


def _forecast_spec() -> ForecastSpec:
    return ForecastSpec(
        forecast_type=ForecastSpecForecastType.NUMERIC,
        output_field="revenue",
        units="USD bn",
        framing=UnconditionalFraming(),
    )


async def _call(
    *,
    cursor: str | None = None,
    progress: dict | None = None,
    detail: TaskDetailResponse | None = None,
    progress_error: Exception | None = None,
    detail_error: Exception | None = None,
    results_error: Exception | None = None,
):
    calls = _Calls(
        progress=AsyncMock(
            return_value=progress if progress is not None else _progress(),
            side_effect=progress_error,
        ),
        detail=AsyncMock(
            return_value=detail if detail is not None else _detail(_forecast_spec()),
            side_effect=detail_error,
        ),
        results=AsyncMock(
            return_value=(ROWS, len(ROWS), uuid4(), uuid4()),
            side_effect=results_error,
        ),
    )
    params = TaskDataInput(task_id=TASK_ID, cursor=cursor)
    ctx = make_test_context(MagicMock())
    with (
        patch("futuresearch_mcp.tools.build_progress_payload", calls.progress),
        patch("futuresearch_mcp.tools.get_task_detail", calls.detail),
        patch("futuresearch_mcp.tools._fetch_task_result", calls.results),
    ):
        return await futuresearch_task_data(params, ctx), calls


@pytest.mark.asyncio
async def test_returns_progress_in_structured_content():
    result, _ = await _call()

    assert not result.isError
    assert result.structuredContent["status"] == "running"
    assert result.structuredContent["completed"] == 1


@pytest.mark.asyncio
async def test_first_call_includes_task_type_and_spec():
    result, calls = await _call()

    assert calls.detail.await_count == 1
    assert result.structuredContent["task_type"] == "forecast"
    spec = result.structuredContent["spec"]
    assert spec["forecast_type"] == "numeric"
    assert spec["output_field"] == "revenue"
    assert spec["framing"] == {"kind": "unconditional"}


@pytest.mark.asyncio
async def test_following_widget_does_not_refetch_the_spec():
    """What a task is never changes, so a cursor means we already sent it."""
    result, calls = await _call(cursor="c1")

    assert calls.detail.await_count == 0
    assert "spec" not in result.structuredContent
    assert "task_type" not in result.structuredContent


@pytest.mark.asyncio
async def test_non_forecast_task_reports_its_type_without_a_spec():
    result, _ = await _call(detail=_detail(None))

    assert result.structuredContent["task_type"] == "rank"
    assert "spec" not in result.structuredContent


@pytest.mark.asyncio
async def test_results_are_sent_once_the_task_has_stopped():
    result, calls = await _call(progress=_progress("completed"))

    assert calls.results.await_count == 1
    assert result.structuredContent["results"] == ROWS


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pending", "running"])
async def test_results_are_not_sent_before_the_task_stops(status):
    """A poll of a running task carries progress only — there are no rows yet."""
    result, calls = await _call(progress=_progress(status))

    assert calls.results.await_count == 0
    assert "results" not in result.structuredContent


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "failed", "revoked"])
async def test_results_are_sent_for_every_terminal_status(status):
    """A failed run still has whatever rows finished."""
    result, calls = await _call(progress=_progress(status))

    assert calls.results.await_count == 1
    assert "results" in result.structuredContent


@pytest.mark.asyncio
async def test_progress_failure_is_reported_as_an_error():
    result, calls = await _call(
        progress_error=FuturesearchClientError("gone", status_code=404)
    )

    assert result.isError
    # No point asking what the task is when we can't read it.
    assert calls.detail.await_count == 0


@pytest.mark.asyncio
async def test_detail_failure_is_reported_as_an_error():
    result, _ = await _call(
        detail_error=FuturesearchClientError("gone", status_code=404)
    )

    assert result.isError


@pytest.mark.asyncio
async def test_results_failure_is_reported_as_an_error():
    result, _ = await _call(
        progress=_progress("completed"),
        results_error=FuturesearchClientError("gone", status_code=404),
    )

    assert result.isError


@pytest.mark.asyncio
async def test_payload_is_kept_out_of_the_model_facing_content():
    """The rows and summaries ride on structuredContent, not into context."""
    result, _ = await _call(progress=_progress("completed"))

    assert len(result.content) == 1
    assert "revenue" not in result.content[0].text
