import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from everyrow.constants import EveryrowError
from everyrow.generated.models import (
    ArtifactGroupRecord,
    StandaloneArtifactRecord,
    TaskEffort,
    TaskResponse,
    TaskStatus,
    TaskStatusResponse,
)
from everyrow.ops import (
    DEFAULT_AGENT_RESPONSE_SCHEMA,
    DEFAULT_SCREEN_RESULT_SCHEMA,
    agent_map,
    create_scalar_artifact,
    create_scalar_artifact_from_dict,
    create_table_artifact,
    rank,
    rank_async,
    screen,
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
        response_schema=DEFAULT_AGENT_RESPONSE_SCHEMA,
        return_table=True,
    )

    assert isinstance(result, TableResult)
    assert len(result.data) == 3
    assert "city" in result.data.columns
    assert result.artifact_id == artifact_id


@pytest.mark.asyncio
async def test_agent_map(mocker, mock_session):
    class MyResponse(BaseModel):
        answer: str

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
        response_model=MyResponse,
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
        response_schema=DEFAULT_AGENT_RESPONSE_SCHEMA,
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


# Tests for XOR validation and JSON schema support


@pytest.mark.asyncio
async def test_single_agent_xor_validation_both_provided(mock_session):
    """Error when both response_model and response_schema are provided."""

    class MyResponse(BaseModel):
        answer: str

    with pytest.raises(
        EveryrowError,
        match="Cannot specify both response_model and response_schema",
    ):
        await single_agent(
            task="What is 2+2?",
            session=mock_session,
            response_model=MyResponse,
            response_schema=DEFAULT_AGENT_RESPONSE_SCHEMA,
        )


@pytest.mark.asyncio
async def test_single_agent_xor_validation_neither_provided(mock_session):
    """Error when neither response_model nor response_schema is provided."""
    with pytest.raises(
        EveryrowError,
        match=r"Must specify either response_model .* or response_schema",
    ):
        await single_agent(
            task="What is 2+2?",
            session=mock_session,
        )


@pytest.mark.asyncio
async def test_single_agent_with_json_schema(mocker, mock_session):
    """single_agent with response_schema returns dict data."""
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

    mock_get_artifacts = mocker.patch(
        "everyrow.task.get_artifacts_artifacts_get.asyncio", new_callable=AsyncMock
    )
    mock_get_artifacts.return_value = [
        StandaloneArtifactRecord(
            uid=artifact_id, type_="standalone", data={"answer": "4"}
        )
    ]

    result = await single_agent(
        task="What is 2+2?",
        session=mock_session,
        response_schema=DEFAULT_AGENT_RESPONSE_SCHEMA,
    )

    assert isinstance(result, ScalarResult)
    assert isinstance(result.data, dict)
    assert result.data["answer"] == "4"
    assert result.artifact_id == artifact_id


@pytest.mark.asyncio
async def test_single_agent_with_dict_input(mocker, mock_session):
    """single_agent can accept dict as input."""

    class MyResponse(BaseModel):
        answer: str

    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()
    input_artifact_id = uuid.uuid4()

    # Mock create_scalar_artifact_from_dict
    mock_create_scalar = mocker.patch(
        "everyrow.ops.create_scalar_artifact_from_dict", new_callable=AsyncMock
    )
    mock_create_scalar.return_value = input_artifact_id

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
        input={"country": "India"},
        response_model=MyResponse,
    )

    assert isinstance(result, ScalarResult)
    assert result.data.answer == "New Delhi"
    mock_create_scalar.assert_called_once_with({"country": "India"}, mock_session)


@pytest.mark.asyncio
async def test_create_scalar_artifact_from_dict(mocker, mock_session):
    """create_scalar_artifact_from_dict creates artifact from dict."""
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

    result_artifact_id = await create_scalar_artifact_from_dict(
        {"name": "John", "age": 30}, mock_session
    )

    assert result_artifact_id == artifact_id
    call_args = mock_submit.call_args
    data_to_create = call_args.kwargs["body"].payload.query.data_to_create
    assert data_to_create == {"name": "John", "age": 30}


@pytest.mark.asyncio
async def test_agent_map_xor_validation_both_provided(mock_session):
    """Error when both response_model and response_schema are provided."""

    class MyResponse(BaseModel):
        answer: str

    with pytest.raises(
        EveryrowError,
        match="Cannot specify both response_model and response_schema",
    ):
        await agent_map(
            task="What is the capital?",
            session=mock_session,
            input=pd.DataFrame([{"country": "India"}]),
            response_model=MyResponse,
            response_schema=DEFAULT_AGENT_RESPONSE_SCHEMA,
        )


@pytest.mark.asyncio
async def test_agent_map_xor_validation_neither_provided(mock_session):
    """Error when neither response_model nor response_schema is provided."""
    with pytest.raises(
        EveryrowError,
        match=r"Must specify either response_model .* or response_schema",
    ):
        await agent_map(
            task="What is the capital?",
            session=mock_session,
            input=pd.DataFrame([{"country": "India"}]),
        )


@pytest.mark.asyncio
async def test_screen_xor_validation_both_provided(mock_session):
    """Error when both response_model and response_schema are provided."""

    class MyResult(BaseModel):
        passes: bool

    with pytest.raises(
        EveryrowError,
        match="Cannot specify both response_model and response_schema",
    ):
        await screen(
            task="Is this a valid country?",
            session=mock_session,
            input=pd.DataFrame([{"country": "India"}]),
            response_model=MyResult,
            response_schema=DEFAULT_SCREEN_RESULT_SCHEMA,
        )


@pytest.mark.asyncio
async def test_screen_xor_validation_neither_provided(mock_session):
    """Error when neither response_model nor response_schema is provided."""
    with pytest.raises(
        EveryrowError,
        match=r"Must specify either response_model .* or response_schema",
    ):
        await screen(
            task="Is this a valid country?",
            session=mock_session,
            input=pd.DataFrame([{"country": "India"}]),
        )


@pytest.mark.asyncio
async def test_screen_with_json_schema(mocker, mock_session):
    """screen with response_schema works correctly."""
    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()
    input_artifact_id = uuid.uuid4()

    mock_create_table = mocker.patch(
        "everyrow.ops.create_table_artifact", new_callable=AsyncMock
    )
    mock_create_table.return_value = input_artifact_id

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
                    data={"country": "India", "passes": True},
                ),
            ],
        )
    ]

    input_df = pd.DataFrame([{"country": "India"}])
    result = await screen(
        task="Is this a valid country?",
        session=mock_session,
        input=input_df,
        response_schema=DEFAULT_SCREEN_RESULT_SCHEMA,
    )

    assert isinstance(result, TableResult)
    assert len(result.data) == 1


@pytest.mark.asyncio
async def test_rank_allows_neither_model_nor_schema(mocker, mock_session):
    """rank allows neither response_model nor response_schema (uses field_name/field_type)."""
    task_id = uuid.uuid4()
    artifact_id = uuid.uuid4()
    input_artifact_id = uuid.uuid4()

    mock_create_table = mocker.patch(
        "everyrow.ops.create_table_artifact", new_callable=AsyncMock
    )
    mock_create_table.return_value = input_artifact_id

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
                    data={"country": "China", "population": 1400000000},
                ),
            ],
        )
    ]

    input_df = pd.DataFrame([{"country": "China"}])
    # rank allows neither model nor schema when field_name is provided
    result = await rank(
        task="Find the population",
        session=mock_session,
        input=input_df,
        field_name="population",
        field_type="int",
    )

    assert isinstance(result, TableResult)
