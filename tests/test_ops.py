import uuid
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from pydantic import BaseModel

from everyrow_sdk.generated.models import (
    ArtifactGroupRecord,
    StandaloneArtifactRecord,
    TaskEffort,
    TaskResponse,
    TaskStatus,
    TaskStatusResponse,
)
from everyrow_sdk.ops import (
    agent_map,
    create_scalar_artifact,
    single_agent,
)
from everyrow_sdk.result import ScalarResult, TableResult
from everyrow_sdk.session import Session


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
        "everyrow_sdk.task.submit_task_tasks_post.asyncio", new_callable=AsyncMock
    )
    mock_submit.return_value = TaskResponse(task_id=task_id)

    # Mock get_task_status
    mock_status = mocker.patch(
        "everyrow_sdk.task.get_task_status_endpoint_tasks_task_id_status_get.asyncio",
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
        "everyrow_sdk.task.submit_task_tasks_post.asyncio", new_callable=AsyncMock
    )
    mock_submit.return_value = TaskResponse(task_id=task_id)

    # Mock get_task_status
    mock_status = mocker.patch(
        "everyrow_sdk.task.get_task_status_endpoint_tasks_task_id_status_get.asyncio",
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
        "everyrow_sdk.task.get_artifacts_artifacts_get.asyncio", new_callable=AsyncMock
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
        "everyrow_sdk.task.submit_task_tasks_post.asyncio", new_callable=AsyncMock
    )
    mock_submit.return_value = TaskResponse(task_id=task_id)

    # Mock get_task_status
    mock_status = mocker.patch(
        "everyrow_sdk.task.get_task_status_endpoint_tasks_task_id_status_get.asyncio",
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
        "everyrow_sdk.task.get_artifacts_artifacts_get.asyncio", new_callable=AsyncMock
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
        "everyrow_sdk.ops.create_table_artifact", new_callable=AsyncMock
    )
    mock_create_table.return_value = input_artifact_id

    # Mock submit_task
    mock_submit = mocker.patch(
        "everyrow_sdk.task.submit_task_tasks_post.asyncio", new_callable=AsyncMock
    )
    mock_submit.return_value = TaskResponse(task_id=task_id)

    # Mock get_task_status
    mock_status = mocker.patch(
        "everyrow_sdk.task.get_task_status_endpoint_tasks_task_id_status_get.asyncio",
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
        "everyrow_sdk.task.get_artifacts_artifacts_get.asyncio", new_callable=AsyncMock
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
        "everyrow_sdk.ops.create_table_artifact", new_callable=AsyncMock
    )
    mock_create_table.return_value = input_artifact_id

    # Mock submit_task
    mock_submit = mocker.patch(
        "everyrow_sdk.task.submit_task_tasks_post.asyncio", new_callable=AsyncMock
    )
    mock_submit.return_value = TaskResponse(task_id=task_id)

    # Mock get_task_status
    mock_status = mocker.patch(
        "everyrow_sdk.task.get_task_status_endpoint_tasks_task_id_status_get.asyncio",
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
        "everyrow_sdk.task.get_artifacts_artifacts_get.asyncio", new_callable=AsyncMock
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
        return_table_per_row=True,
    )

    assert isinstance(result, TableResult)
    assert len(result.data) == 2
    assert result.artifact_id == artifact_id
