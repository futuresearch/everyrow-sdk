"""Unit tests for the pandas DataFrame accessor."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from pydantic import BaseModel, Field

from everyrow.accessor import EveryrowAccessor
from everyrow.generated.models import (
    OperationResponse,
    PublicTaskType,
    TaskResultResponse,
    TaskResultResponseDataType0Item,
    TaskStatus,
    TaskStatusResponse,
)
from everyrow.result import TableResult
from everyrow.session import Session


@pytest.fixture
def mock_session():
    session = MagicMock(spec=Session)
    session.session_id = uuid.uuid4()
    session.client = MagicMock()
    return session


@pytest.fixture(autouse=True)
def mock_env_api_key(monkeypatch):
    monkeypatch.setenv("EVERYROW_API_KEY", "test-key")


def _make_status_response(
    task_id, session_id, artifact_id=None, status=TaskStatus.COMPLETED
):
    return TaskStatusResponse(
        task_id=task_id,
        session_id=session_id,
        status=status,
        task_type=PublicTaskType.AGENT,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        artifact_id=artifact_id,
    )


def _make_table_result(task_id, records, artifact_id=None):
    data_items = [TaskResultResponseDataType0Item.from_dict(r) for r in records]
    return TaskResultResponse(
        task_id=task_id,
        status=TaskStatus.COMPLETED,
        data=data_items,
        artifact_id=artifact_id,
    )


# --- Accessor Registration ---


def test_accessor_is_registered():
    """DataFrame should have .everyrow accessor after importing everyrow."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert hasattr(df, "everyrow")
    assert isinstance(df.everyrow, EveryrowAccessor)


def test_accessor_preserves_dataframe():
    """Accessor should preserve reference to the original DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    accessor = df.everyrow
    pd.testing.assert_frame_equal(accessor._df, df)


def test_last_result_initially_none():
    """last_result should be None before any operation."""
    df = pd.DataFrame({"a": [1]})
    assert df.everyrow.last_result is None


# --- Session Management ---


def test_with_session_sets_session(mock_session):
    """with_session should store the session."""
    df = pd.DataFrame({"a": [1]})
    accessor = df.everyrow.with_session(mock_session)
    assert accessor._session is mock_session


def test_with_session_returns_self(mock_session):
    """with_session should return self for chaining."""
    df = pd.DataFrame({"a": [1]})
    result = df.everyrow.with_session(mock_session)
    assert result is df.everyrow


# --- Screen ---


@pytest.mark.asyncio
async def test_screen_returns_dataframe(mocker, mock_session):
    """screen should return a DataFrame, not TableResult."""
    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()

    mock_submit = mocker.patch(
        "everyrow.ops.screen_operations_screen_post.asyncio",
        new_callable=AsyncMock,
    )
    mock_submit.return_value = OperationResponse(
        task_id=task_id,
        session_id=mock_session.session_id,
        status=TaskStatus.PENDING,
    )

    mock_status = mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_status.return_value = _make_status_response(
        task_id, mock_session.session_id, artifact_id
    )

    mock_result = mocker.patch(
        "everyrow.task.get_task_result_tasks_task_id_result_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_result.return_value = _make_table_result(
        task_id,
        [{"item": "Water", "passes": True}],
        artifact_id,
    )

    df = pd.DataFrame([{"item": "Water"}, {"item": "Poison"}])
    result = await df.everyrow.with_session(mock_session).screen("Filter safe items")

    assert isinstance(result, pd.DataFrame)
    assert "passes" in result.columns


@pytest.mark.asyncio
async def test_screen_stores_last_result(mocker, mock_session):
    """screen should store the full TableResult in last_result."""
    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()

    mock_submit = mocker.patch(
        "everyrow.ops.screen_operations_screen_post.asyncio",
        new_callable=AsyncMock,
    )
    mock_submit.return_value = OperationResponse(
        task_id=task_id,
        session_id=mock_session.session_id,
        status=TaskStatus.PENDING,
    )

    mock_status = mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_status.return_value = _make_status_response(
        task_id, mock_session.session_id, artifact_id
    )

    mock_result = mocker.patch(
        "everyrow.task.get_task_result_tasks_task_id_result_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_result.return_value = _make_table_result(
        task_id,
        [{"item": "Water", "passes": True}],
        artifact_id,
    )

    df = pd.DataFrame([{"item": "Water"}])
    accessor = df.everyrow.with_session(mock_session)
    await accessor.screen("Filter safe items")

    assert accessor.last_result is not None
    assert isinstance(accessor.last_result, TableResult)
    assert accessor.last_result.artifact_id == artifact_id


# --- Rank ---


@pytest.mark.asyncio
async def test_rank_returns_dataframe(mocker, mock_session):
    """rank should return a DataFrame."""
    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()

    mock_submit = mocker.patch(
        "everyrow.ops.rank_operations_rank_post.asyncio",
        new_callable=AsyncMock,
    )
    mock_submit.return_value = OperationResponse(
        task_id=task_id,
        session_id=mock_session.session_id,
        status=TaskStatus.PENDING,
    )

    mock_status = mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_status.return_value = _make_status_response(
        task_id, mock_session.session_id, artifact_id
    )

    mock_result = mocker.patch(
        "everyrow.task.get_task_result_tasks_task_id_result_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_result.return_value = _make_table_result(
        task_id,
        [
            {"company": "A", "score": 0.9},
            {"company": "B", "score": 0.7},
        ],
        artifact_id,
    )

    df = pd.DataFrame([{"company": "A"}, {"company": "B"}])
    result = await df.everyrow.with_session(mock_session).rank(
        "score", task="Rank by quality"
    )

    assert isinstance(result, pd.DataFrame)
    assert "score" in result.columns


# --- Dedupe ---


@pytest.mark.asyncio
async def test_dedupe_returns_dataframe(mocker, mock_session):
    """dedupe should return a DataFrame."""
    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()

    mock_submit = mocker.patch(
        "everyrow.ops.dedupe_operations_dedupe_post.asyncio",
        new_callable=AsyncMock,
    )
    mock_submit.return_value = OperationResponse(
        task_id=task_id,
        session_id=mock_session.session_id,
        status=TaskStatus.PENDING,
    )

    mock_status = mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_status.return_value = _make_status_response(
        task_id, mock_session.session_id, artifact_id
    )

    mock_result = mocker.patch(
        "everyrow.task.get_task_result_tasks_task_id_result_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_result.return_value = _make_table_result(
        task_id,
        [{"name": "Apple Inc", "selected": True}],
        artifact_id,
    )

    df = pd.DataFrame([{"name": "Apple Inc"}, {"name": "Apple"}])
    result = await df.everyrow.with_session(mock_session).dedupe(
        "Same company, ignoring legal suffixes"
    )

    assert isinstance(result, pd.DataFrame)


# --- Merge ---


@pytest.mark.asyncio
async def test_merge_returns_dataframe(mocker, mock_session):
    """merge should return a DataFrame."""
    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()

    mock_submit = mocker.patch(
        "everyrow.ops.merge_operations_merge_post.asyncio",
        new_callable=AsyncMock,
    )
    mock_submit.return_value = OperationResponse(
        task_id=task_id,
        session_id=mock_session.session_id,
        status=TaskStatus.PENDING,
    )

    mock_status = mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_status.return_value = _make_status_response(
        task_id, mock_session.session_id, artifact_id
    )

    mock_result = mocker.patch(
        "everyrow.task.get_task_result_tasks_task_id_result_get.asyncio",
        new_callable=AsyncMock,
    )
    response = TaskResultResponse(
        task_id=task_id,
        status=TaskStatus.COMPLETED,
        data=[
            TaskResultResponseDataType0Item.from_dict(
                {"subsidiary": "YouTube", "parent": "Alphabet"}
            )
        ],
        artifact_id=artifact_id,
    )
    response.additional_properties = {
        "breakdown": {
            "exact": [],
            "fuzzy": [],
            "llm": [[0, 0]],
            "web": [],
            "unmatched_left": [],
            "unmatched_right": [],
        }
    }
    mock_result.return_value = response

    left = pd.DataFrame([{"subsidiary": "YouTube"}])
    right = pd.DataFrame([{"parent": "Alphabet"}])

    result = await left.everyrow.with_session(mock_session).merge(
        right, task="Match subsidiaries to parents"
    )

    assert isinstance(result, pd.DataFrame)


# --- Agent Map ---


@pytest.mark.asyncio
async def test_agent_map_returns_dataframe(mocker, mock_session):
    """agent_map should return a DataFrame."""
    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()

    mock_submit = mocker.patch(
        "everyrow.ops.agent_map_operations_agent_map_post.asyncio",
        new_callable=AsyncMock,
    )
    mock_submit.return_value = OperationResponse(
        task_id=task_id,
        session_id=mock_session.session_id,
        status=TaskStatus.PENDING,
    )

    mock_status = mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_status.return_value = _make_status_response(
        task_id, mock_session.session_id, artifact_id
    )

    mock_result = mocker.patch(
        "everyrow.task.get_task_result_tasks_task_id_result_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_result.return_value = _make_table_result(
        task_id,
        [{"company": "Apple", "answer": "1976"}],
        artifact_id,
    )

    class FoundingYear(BaseModel):
        answer: str = Field(description="The founding year")

    df = pd.DataFrame([{"company": "Apple"}])
    result = await df.everyrow.with_session(mock_session).agent_map(
        "Find the founding year", response_model=FoundingYear
    )

    assert isinstance(result, pd.DataFrame)
    assert "answer" in result.columns


# --- Single Agent ---


@pytest.mark.asyncio
async def test_single_agent_returns_dataframe(mocker, mock_session):
    """single_agent should return a DataFrame."""
    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()

    mock_submit = mocker.patch(
        "everyrow.ops.single_agent_operations_single_agent_post.asyncio",
        new_callable=AsyncMock,
    )
    mock_submit.return_value = OperationResponse(
        task_id=task_id,
        session_id=mock_session.session_id,
        status=TaskStatus.PENDING,
    )

    mock_status = mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_status.return_value = _make_status_response(
        task_id, mock_session.session_id, artifact_id
    )

    mock_result = mocker.patch(
        "everyrow.task.get_task_result_tasks_task_id_result_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_result.return_value = _make_table_result(
        task_id,
        [{"insight": "Revenue is growing"}],
        artifact_id,
    )

    class AnalysisResult(BaseModel):
        insight: str = Field(description="Key insight from data")

    df = pd.DataFrame([{"month": "Jan", "revenue": 100}])
    result = await df.everyrow.with_session(mock_session).single_agent(
        "Analyze trends", response_model=AnalysisResult
    )

    assert isinstance(result, pd.DataFrame)
