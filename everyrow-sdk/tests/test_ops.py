import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from everyrow.generated.models import (
    ArtifactGroupRecord,
    StandaloneArtifactRecord,
    TaskEffort,
    TaskResponse,
    TaskStatus,
    TaskStatusResponse,
)
from everyrow.ops import (
    agent_map,
    create_scalar_artifact,
    create_table_artifact,
    rank_async,
    single_agent,
)
from everyrow.result import ScalarResult, TableResult
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


@pytest.mark.asyncio
async def test_create_scalar_artifact(mocker, mock_session):
    class MyModel(BaseModel):
        name: str
        age: int

    model = MyModel(name="John", age=30)
    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()

    # Mock submit_task
    mock_submit = mocker.patch(
        "everyrow.task.submit_task_tasks_post.asyncio", new_callable=AsyncMock
    )
    mock_submit.return_value = TaskResponse(task_id=task_id)

    # Mock get_task_status
    mock_status = mocker.patch(
        "everyrow.task.get_task_status_endpoint_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_status.return_value = TaskStatusResponse(
        status=TaskStatus.COMPLETED,
        artifact_id=artifact_id,
        task_id=task_id,
        error=None,
    )

    result_artifact_id = await create_scalar_artifact(model, mock_session)

    assert result_artifact_id == artifact_id
    assert mock_submit.called
    assert mock_status.called


@pytest.mark.asyncio
async def test_single_agent(mocker, mock_session):
    class MyInput(BaseModel):
        country: str

    class MyResponse(BaseModel):
        answer: str

    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()

    # Mock submit_task
    mock_submit = mocker.patch(
        "everyrow.task.submit_task_tasks_post.asyncio", new_callable=AsyncMock
    )
    mock_submit.return_value = TaskResponse(task_id=task_id)

    # Mock get_task_status
    mock_status = mocker.patch(
        "everyrow.task.get_task_status_endpoint_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_status.return_value = TaskStatusResponse(
        status=TaskStatus.COMPLETED,
        artifact_id=artifact_id,
        task_id=task_id,
        error=None,
    )

    # Mock get_artifacts
    mock_get_artifacts = mocker.patch(
        "everyrow.task.get_artifacts_artifacts_get.asyncio", new_callable=AsyncMock
    )
    mock_get_artifacts.return_value = [
        StandaloneArtifactRecord(
            uid=artifact_id, type_="standalone", data={"answer": "New Delhi"}
        )
    ]

    result = await single_agent(
        task="What is the capital of the given country?",
        session=mock_session,
        input=MyInput(country="India"),
        response_model=MyResponse,
    )

    assert isinstance(result, ScalarResult)
    assert result.data.answer == "New Delhi"
    assert result.artifact_id == artifact_id


@pytest.mark.asyncio
async def test_single_agent_with_table_output(mocker, mock_session):
    class MyInput(BaseModel):
        country: str

    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()

    # Mock submit_task
    mock_submit = mocker.patch(
        "everyrow.task.submit_task_tasks_post.asyncio", new_callable=AsyncMock
    )
    mock_submit.return_value = TaskResponse(task_id=task_id)

    # Mock get_task_status
    mock_status = mocker.patch(
        "everyrow.task.get_task_status_endpoint_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_status.return_value = TaskStatusResponse(
        status=TaskStatus.COMPLETED,
        artifact_id=artifact_id,
        task_id=task_id,
        error=None,
    )

    # Mock get_artifacts for TableResult
    mock_get_artifacts = mocker.patch(
        "everyrow.task.get_artifacts_artifacts_get.asyncio", new_callable=AsyncMock
    )
    mock_get_artifacts.return_value = [
        ArtifactGroupRecord(
            uid=artifact_id,
            type_="group",
            data=[],
            artifacts=[
                StandaloneArtifactRecord(
                    uid=uuid.uuid4(), type_="standalone", data={"city": "Mumbai"}
                ),
                StandaloneArtifactRecord(
                    uid=uuid.uuid4(), type_="standalone", data={"city": "Delhi"}
                ),
                StandaloneArtifactRecord(
                    uid=uuid.uuid4(), type_="standalone", data={"city": "Bangalore"}
                ),
            ],
        )
    ]

    result = await single_agent(
        task="What are the three largest cities in the given country?",
        session=mock_session,
        input=MyInput(country="India"),
        effort_level=TaskEffort.LOW,
        return_table=True,
    )

    assert isinstance(result, TableResult)
    assert len(result.data) == 3
    assert "city" in result.data.columns
    assert result.artifact_id == artifact_id


@pytest.mark.asyncio
async def test_agent_map(mocker, mock_session):
    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()
    input_artifact_id = uuid.uuid4()

    # Mock create_table_artifact (called because input is DataFrame)
    mock_create_table = mocker.patch(
        "everyrow.ops.create_table_artifact", new_callable=AsyncMock
    )
    mock_create_table.return_value = input_artifact_id

    # Mock submit_task
    mock_submit = mocker.patch(
        "everyrow.task.submit_task_tasks_post.asyncio", new_callable=AsyncMock
    )
    mock_submit.return_value = TaskResponse(task_id=task_id)

    # Mock get_task_status
    mock_status = mocker.patch(
        "everyrow.task.get_task_status_endpoint_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_status.return_value = TaskStatusResponse(
        status=TaskStatus.COMPLETED,
        artifact_id=artifact_id,
        task_id=task_id,
        error=None,
    )

    # Mock get_artifacts
    mock_get_artifacts = mocker.patch(
        "everyrow.task.get_artifacts_artifacts_get.asyncio", new_callable=AsyncMock
    )
    mock_get_artifacts.return_value = [
        ArtifactGroupRecord(
            uid=artifact_id,
            type_="group",
            data=[],
            artifacts=[
                StandaloneArtifactRecord(
                    uid=uuid.uuid4(),
                    type_="standalone",
                    data={"country": "India", "answer": "New Delhi"},
                ),
                StandaloneArtifactRecord(
                    uid=uuid.uuid4(),
                    type_="standalone",
                    data={"country": "USA", "answer": "Washington D.C."},
                ),
            ],
        )
    ]

    input_df = pd.DataFrame([{"country": "India"}, {"country": "USA"}])
    result = await agent_map(
        task="What is the capital of the given country?",
        session=mock_session,
        input=input_df,
    )

    assert isinstance(result, TableResult)
    assert len(result.data) == 2
    assert "answer" in result.data.columns
    assert result.artifact_id == artifact_id


@pytest.mark.asyncio
async def test_agent_map_with_table_output(mocker, mock_session):
    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()
    input_artifact_id = uuid.uuid4()

    # Mock create_table_artifact
    mock_create_table = mocker.patch(
        "everyrow.ops.create_table_artifact", new_callable=AsyncMock
    )
    mock_create_table.return_value = input_artifact_id

    # Mock submit_task
    mock_submit = mocker.patch(
        "everyrow.task.submit_task_tasks_post.asyncio", new_callable=AsyncMock
    )
    mock_submit.return_value = TaskResponse(task_id=task_id)

    # Mock get_task_status
    mock_status = mocker.patch(
        "everyrow.task.get_task_status_endpoint_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_status.return_value = TaskStatusResponse(
        status=TaskStatus.COMPLETED,
        artifact_id=artifact_id,
        task_id=task_id,
        error=None,
    )

    # Mock get_artifacts
    mock_get_artifacts = mocker.patch(
        "everyrow.task.get_artifacts_artifacts_get.asyncio", new_callable=AsyncMock
    )
    mock_get_artifacts.return_value = [
        ArtifactGroupRecord(
            uid=artifact_id,
            type_="group",
            data=[],
            artifacts=[
                StandaloneArtifactRecord(
                    uid=uuid.uuid4(),
                    type_="standalone",
                    data={"country": "India", "city": "Mumbai"},
                ),
                StandaloneArtifactRecord(
                    uid=uuid.uuid4(),
                    type_="standalone",
                    data={"country": "USA", "city": "New York"},
                ),
            ],
        )
    ]

    input_df = pd.DataFrame([{"country": "India"}, {"country": "USA"}])
    result = await agent_map(
        task="What are the three largest cities in the given country?",
        session=mock_session,
        input=input_df,
    )

    assert isinstance(result, TableResult)
    assert len(result.data) == 2
    assert result.artifact_id == artifact_id


@pytest.mark.asyncio
async def test_rank_model_validation(mocker, mock_session) -> None:
    input_df = pd.DataFrame(
        [
            {"country": "China"},
            {"country": "India"},
            {"country": "Indonesia"},
            {"country": "Pakistan"},
            {"country": "USA"},
        ],
    )

    class ResponseModel(BaseModel):
        population_size: int

    input_artifact_id = uuid.uuid4()
    # Mock create_table_artifact (called because input is DataFrame)
    mock_create_table = mocker.patch(
        "everyrow.ops.create_table_artifact", new_callable=AsyncMock
    )
    mock_create_table.return_value = input_artifact_id

    with pytest.raises(
        ValueError,
        match="Field population not in response model ResponseModel",
    ):
        await rank_async(
            task="Find the population of the given country",
            session=mock_session,
            input=input_df,
            field_name="population",
            response_model=ResponseModel,
        )


@pytest.mark.asyncio
async def test_create_table_artifact_converts_nan_to_none(mocker, mock_session):
    """NaN values should be converted to None for JSON compatibility."""

    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()

    mock_submit = mocker.patch(
        "everyrow.task.submit_task_tasks_post.asyncio", new_callable=AsyncMock
    )
    mock_submit.return_value = TaskResponse(task_id=task_id)

    mock_status = mocker.patch(
        "everyrow.task.get_task_status_endpoint_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_status.return_value = TaskStatusResponse(
        status=TaskStatus.COMPLETED,
        artifact_id=artifact_id,
        task_id=task_id,
        error=None,
    )

    df_with_nan = pd.DataFrame([{"name": "Alice", "age": np.nan}])
    await create_table_artifact(df_with_nan, mock_session)

    call_args = mock_submit.call_args
    data_to_create = call_args.kwargs["body"].payload.query.data_to_create
    assert data_to_create == [{"name": "Alice", "age": None}]


@pytest.mark.asyncio
async def test_create_table_artifact_preserves_valid_values(mocker, mock_session):
    """Non-NaN values should be passed through unchanged."""
    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()

    mock_submit = mocker.patch(
        "everyrow.task.submit_task_tasks_post.asyncio", new_callable=AsyncMock
    )
    mock_submit.return_value = TaskResponse(task_id=task_id)

    mock_status = mocker.patch(
        "everyrow.task.get_task_status_endpoint_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_status.return_value = TaskStatusResponse(
        status=TaskStatus.COMPLETED,
        artifact_id=artifact_id,
        task_id=task_id,
        error=None,
    )

    df = pd.DataFrame([{"name": "Alice", "age": 30}])
    await create_table_artifact(df, mock_session)

    call_args = mock_submit.call_args
    data_to_create = call_args.kwargs["body"].payload.query.data_to_create
    assert data_to_create == [{"name": "Alice", "age": 30}]
