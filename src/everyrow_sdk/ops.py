from typing import Any, Literal, TypeVar, overload
from uuid import UUID

from pandas import DataFrame
from pydantic import BaseModel

from everyrow_sdk.constants import EveryrowError
from everyrow_sdk.generated.models import (
    AgentQueryParams,
    CreateGroupQueryParams,
    CreateGroupRequest,
    CreateQueryParams,
    CreateRequest,
    DedupeMode,
    DedupeQueryParams,
    DedupeRequestParams,
    DeepMergePublicParams,
    DeepMergeRequest,
    DeepRankPublicParams,
    DeepRankRequest,
    DeepScreenPublicParams,
    DeepScreenRequest,
    DeriveExpression,
    DeriveQueryParams,
    DeriveRequest,
    EmbeddingModels,
    MapAgentRequestParams,
    ProcessingMode,
    ReduceAgentRequestParams,
    ResponseSchemaType,
)
from everyrow_sdk.generated.models.submit_task_body import SubmitTaskBody
from everyrow_sdk.generated.types import UNSET
from everyrow_sdk.result import Result, ScalarResult, TableResult
from everyrow_sdk.session import Session
from everyrow_sdk.task import (
    LLM,
    EffortLevel,
    EveryrowTask,
    await_task_completion,
    read_table_result,
    submit_task,
)

T = TypeVar("T", bound=BaseModel)


class DefaultAgentResponse(BaseModel):
    answer: str


@overload
async def single_agent[T: BaseModel](
    task: str,
    session: Session,
    input: BaseModel | UUID | Result | None = None,
    effort_level: EffortLevel = EffortLevel.LOW,
    llm: LLM | None = None,
    response_model: type[T] = DefaultAgentResponse,
    return_table: Literal[False] = False,
) -> ScalarResult[T]: ...


@overload
async def single_agent(
    task: str,
    session: Session,
    input: BaseModel | UUID | Result | None = None,
    effort_level: EffortLevel = EffortLevel.LOW,
    llm: LLM | None = None,
    response_model: type[BaseModel] = DefaultAgentResponse,
    return_table: Literal[True] = True,
) -> TableResult: ...


async def single_agent[T: BaseModel](
    task: str,
    session: Session,
    input: BaseModel | DataFrame | UUID | Result | None = None,
    effort_level: EffortLevel = EffortLevel.LOW,
    llm: LLM | None = None,
    response_model: type[T] = DefaultAgentResponse,
    return_table: bool = False,
) -> ScalarResult[T] | TableResult:
    cohort_task = await single_agent_async(
        task=task,
        session=session,
        input=input,
        effort_level=effort_level,
        llm=llm,
        response_model=response_model,
        return_table=return_table,
    )
    return await cohort_task.await_result(session.client)


async def single_agent_async[T: BaseModel](
    task: str,
    session: Session,
    input: BaseModel | DataFrame | UUID | Result | None = None,
    effort_level: EffortLevel = EffortLevel.LOW,
    llm: LLM | None = None,
    response_model: type[T] = DefaultAgentResponse,
    return_table: bool = False,
) -> EveryrowTask[T]:
    if input is not None:
        input_artifact_ids = [await _process_single_agent_input(input, session)]
    else:
        input_artifact_ids = []

    query = AgentQueryParams(
        task=task,
        llm=llm or UNSET,
        effort_level=effort_level,
        response_schema=response_model.model_json_schema(),
        response_schema_type=ResponseSchemaType.JSON,
        is_expand=return_table,
        include_provenance_and_notes=False,
    )
    request = ReduceAgentRequestParams(
        query=query,
        input_artifacts=input_artifact_ids,
    )
    body = SubmitTaskBody(
        payload=request,
        session_id=session.session_id,
    )

    cohort_task = EveryrowTask(response_model=response_model, is_map=False, is_expand=return_table)
    await cohort_task.submit(body, session.client)
    return cohort_task


async def agent_map(
    task: str,
    session: Session,
    input: DataFrame | UUID | TableResult,
    effort_level: EffortLevel = EffortLevel.LOW,
    llm: LLM | None = None,
    response_model: type[BaseModel] = DefaultAgentResponse,
    return_table_per_row: bool = False,
) -> TableResult:
    cohort_task = await agent_map_async(task, session, input, effort_level, llm, response_model, return_table_per_row)
    result = await cohort_task.await_result(session.client)
    if isinstance(result, TableResult):
        return result
    else:
        raise EveryrowError("Agent map task did not return a table result")


def _convert_pydantic_to_custom_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to the custom response schema format expected by rank.

    The custom format uses _model_name instead of type: object, and uses optional: bool
    instead of required arrays.

    Example:
        class ScreeningResult(BaseModel):
            screening_result: str = Field(..., description="...")

        Converts to:
        {
            "_model_name": "ScreeningResult",
            "screening_result": {
                "type": "str",
                "optional": False,
                "description": "..."
            }
        }
    """
    json_schema = model.model_json_schema()

    # Extract model name from title or use the class name
    model_name = json_schema.get("title", model.__name__)

    # Build the custom schema format
    custom_schema: dict[str, Any] = {"_model_name": model_name}

    # Convert properties
    properties = json_schema.get("properties", {})
    required = set(json_schema.get("required", []))

    # Map JSON schema types to custom format types
    type_mapping = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
    }

    for field_name, field_schema in properties.items():
        # Copy the field schema
        custom_field: dict[str, Any] = {}

        # Map type from JSON schema format to custom format
        field_type = field_schema.get("type")
        if field_type:
            # Convert JSON schema type to custom format type
            custom_field["type"] = type_mapping.get(field_type, field_type)

        # Add description if present
        if "description" in field_schema:
            custom_field["description"] = field_schema["description"]

        # Set optional flag (opposite of required)
        custom_field["optional"] = field_name not in required

        custom_schema[field_name] = custom_field

    return custom_schema


async def agent_map_async(
    task: str,
    session: Session,
    input: DataFrame | UUID | TableResult,
    effort_level: EffortLevel = EffortLevel.LOW,
    llm: LLM | None = None,
    response_model: type[BaseModel] = DefaultAgentResponse,
    return_table_per_row: bool = False,
) -> EveryrowTask[BaseModel]:
    input_artifact_ids = [await _process_agent_map_input(input, session)]
    query = AgentQueryParams(
        task=task,
        effort_level=effort_level,
        llm=llm or UNSET,
        response_schema=_convert_pydantic_to_custom_schema(response_model),
        response_schema_type=ResponseSchemaType.CUSTOM,
        is_expand=return_table_per_row,
        include_provenance_and_notes=False,
    )
    request = MapAgentRequestParams(
        query=query,
        input_artifacts=input_artifact_ids,
        context_artifacts=[],
        join_with_input=True,
    )
    body = SubmitTaskBody(
        payload=request,
        session_id=session.session_id,
    )

    cohort_task = EveryrowTask(response_model=response_model, is_map=True, is_expand=return_table_per_row)
    await cohort_task.submit(body, session.client)
    return cohort_task


async def _process_agent_map_input(
    input: DataFrame | UUID | TableResult,
    session: Session,
) -> UUID:
    if isinstance(input, TableResult):
        return input.artifact_id
    elif isinstance(input, DataFrame):
        return await create_table_artifact(input, session)
    else:
        return input


async def _process_single_agent_input(
    input: BaseModel | DataFrame | UUID | Result,
    session: Session,
) -> UUID:
    if isinstance(input, Result):
        return input.artifact_id
    elif isinstance(input, DataFrame):
        return await create_table_artifact(input, session)
    elif isinstance(input, BaseModel):
        return await create_scalar_artifact(input, session)
    else:
        return input


async def create_scalar_artifact(input: BaseModel, session: Session) -> UUID:
    payload = CreateRequest(query=CreateQueryParams(data_to_create=input.model_dump()))
    body = SubmitTaskBody(
        payload=payload,
        session_id=session.session_id,
    )
    task_id = await submit_task(body, session.client)
    finished_create_artifact_task = await await_task_completion(task_id, session.client)
    return finished_create_artifact_task.artifact_id  # type: ignore (we check artifact_id in await_task_completion)


async def create_table_artifact(input: DataFrame, session: Session) -> UUID:
    payload = CreateGroupRequest(query=CreateGroupQueryParams(data_to_create=input.to_dict(orient="records")))
    body = SubmitTaskBody(
        payload=payload,
        session_id=session.session_id,
    )
    task_id = await submit_task(body, session.client)
    finished_create_artifact_task = await await_task_completion(task_id, session.client)
    return finished_create_artifact_task.artifact_id  # type: ignore (we check artifact_id in await_task_completion)


async def merge(
    task: str,
    session: Session,
    left_table: DataFrame | UUID | TableResult,
    right_table: DataFrame | UUID | TableResult,
    merge_on_left: str | None = None,
    merge_on_right: str | None = None,
    merge_model: LLM | None = None,
    preview: bool = False,
) -> TableResult:
    """Merge two tables using merge operation.

    Args:
        task: The task description for the merge operation
        session: The session to use
        left_table: The left table to merge (DataFrame, UUID, or TableResult)
        right_table: The right table to merge (DataFrame, UUID, or TableResult)
        merge_on_left: Optional column name in left table to merge on
        merge_on_right: Optional column name in right table to merge on
        merge_model: Optional LLM model to use for merge operation
        preview: If True, process only the first few inputs

    Returns:
        TableResult containing the merged table
    """
    cohort_task = await merge_async(
        task=task,
        session=session,
        left_table=left_table,
        right_table=right_table,
        merge_on_left=merge_on_left,
        merge_on_right=merge_on_right,
        merge_model=merge_model,
        preview=preview,
    )
    result = await cohort_task.await_result(session.client)
    if isinstance(result, TableResult):
        return result
    else:
        raise EveryrowError("Merge task did not return a table result")


async def merge_async(
    task: str,
    session: Session,
    left_table: DataFrame | UUID | TableResult,
    right_table: DataFrame | UUID | TableResult,
    merge_on_left: str | None = None,
    merge_on_right: str | None = None,
    merge_model: LLM | None = None,
    preview: bool = False,
) -> EveryrowTask[BaseModel]:
    """Submit a merge task asynchronously."""
    left_artifact_id = await _process_agent_map_input(left_table, session)
    right_artifact_id = await _process_agent_map_input(right_table, session)

    query = DeepMergePublicParams(
        task=task,
        merge_on_left=merge_on_left or UNSET,
        merge_on_right=merge_on_right or UNSET,
        merge_model=merge_model or UNSET,
        preview=preview,
    )
    request = DeepMergeRequest(
        query=query,
        input_artifacts=[left_artifact_id],
        context_artifacts=[right_artifact_id],
    )
    body = SubmitTaskBody(
        payload=request,
        session_id=session.session_id,
    )

    cohort_task = EveryrowTask(response_model=BaseModel, is_map=True, is_expand=False)
    await cohort_task.submit(body, session.client)
    return cohort_task


async def rank(
    task: str,
    session: Session,
    input: DataFrame | UUID | TableResult,
    field_name: str,
    field_type: Literal["float", "int", "str", "bool"] = "float",
    ascending_order: bool = True,
    preview: bool = False,
) -> TableResult:
    """Rank rows in a table using rank operation.

    Args:
        task: The task description for ranking
        session: The session to use
        input: The input table (DataFrame, UUID, or TableResult)
        field_name: The name of the field to extract and sort by
        field_type: The type of the field (default: "float")
        ascending_order: If True, sort in ascending order
        preview: If True, process only the first few inputs

    Returns:
        TableResult containing the ranked table
    """
    cohort_task = await rank_async(
        task=task,
        session=session,
        input=input,
        field_name=field_name,
        field_type=field_type,
        ascending_order=ascending_order,
        preview=preview,
    )
    result = await cohort_task.await_result(session.client)
    if isinstance(result, TableResult):
        return result
    else:
        raise EveryrowError("Rank task did not return a table result")


async def rank_async(
    task: str,
    session: Session,
    input: DataFrame | UUID | TableResult,
    field_name: str,
    field_type: Literal["float", "int", "str", "bool"] = "float",
    ascending_order: bool = True,
    preview: bool = False,
) -> EveryrowTask[BaseModel]:
    """Submit a rank task asynchronously."""
    input_artifact_id = await _process_agent_map_input(input, session)

    # Build response schema with single field
    response_schema = {
        "_model_name": "RankResponse",
        field_name: {
            "type": field_type,
            "optional": False,
        },
    }

    query = DeepRankPublicParams(
        task=task,
        response_schema=response_schema,
        field_to_sort_by=field_name,
        ascending_order=ascending_order,
        preview=preview,
    )
    request = DeepRankRequest(
        query=query,
        input_artifacts=[input_artifact_id],
        context_artifacts=[],
    )
    body = SubmitTaskBody(
        payload=request,
        session_id=session.session_id,
    )

    cohort_task = EveryrowTask(response_model=BaseModel, is_map=True, is_expand=False)
    await cohort_task.submit(body, session.client)
    return cohort_task


async def screen[T: BaseModel](
    task: str,
    session: Session,
    input: DataFrame | UUID | TableResult,
    response_model: type[T] | None = None,
    batch_size: int | None = None,
    preview: bool = False,
) -> TableResult:
    """Screen rows in a table using screen operation.

    Args:
        task: The task description for screening
        session: The session to use
        input: The input table (DataFrame, UUID, or TableResult)
        response_model: Optional Pydantic model for the response schema
        batch_size: Optional batch size for processing (default: 10)
        preview: If True, process only the first few inputs

    Returns:
        TableResult containing the screened table
    """
    cohort_task = await screen_async(
        task=task,
        session=session,
        input=input,
        response_model=response_model,
        batch_size=batch_size,
        preview=preview,
    )
    result = await cohort_task.await_result(session.client)
    if isinstance(result, TableResult):
        return result
    else:
        raise EveryrowError("Screen task did not return a table result")


async def screen_async[T: BaseModel](
    task: str,
    session: Session,
    input: DataFrame | UUID | TableResult,
    response_model: type[T] | None = None,
    batch_size: int | None = None,
    preview: bool = False,
) -> EveryrowTask[T]:
    """Submit a screen task asynchronously."""
    input_artifact_id = await _process_agent_map_input(input, session)

    if response_model is not None:
        response_schema = response_model.model_json_schema()
        response_schema_type = ResponseSchemaType.JSON
    else:
        response_schema = UNSET
        response_schema_type = UNSET

    query = DeepScreenPublicParams(
        task=task,
        batch_size=batch_size or UNSET,
        response_schema=response_schema,
        response_schema_type=response_schema_type,
        preview=preview,
    )
    request = DeepScreenRequest(
        query=query,
        input_artifacts=[input_artifact_id],
    )
    body = SubmitTaskBody(
        payload=request,
        session_id=session.session_id,
    )

    cohort_task: EveryrowTask[T] = EveryrowTask(
        response_model=response_model or DefaultAgentResponse,  # type: ignore[arg-type]
        is_map=True,
        is_expand=False,
    )
    await cohort_task.submit(body, session.client)
    return cohort_task


async def clean(
    session: Session,
    input: DataFrame | UUID | TableResult,
    equivalence_relation: str,
    llm: LLM | None = None,
    chunk_size: int | None = None,
    mode: DedupeMode | None = None,
    embedding_model: EmbeddingModels | None = None,
) -> TableResult:
    """Clean a table by removing duplicates using clean (dedupe) operation.

    Args:
        session: The session to use
        input: The input table (DataFrame, UUID, or TableResult)
        equivalence_relation: Description of what makes items equivalent
        llm: Optional LLM model to use for deduplication
        chunk_size: Optional maximum number of items to process in a single LLM call (default: 40)
        mode: Optional dedupe mode (AGENTIC or DIRECT)
        max_consecutive_empty: Optional stop processing a row after this many consecutive comparisons with no matches
        embedding_model: Optional embedding model to use when reorder_by_embedding is True

    Returns:
        TableResult containing the cleaned table with duplicates removed
    """
    cohort_task = await clean_async(
        session=session,
        input=input,
        equivalence_relation=equivalence_relation,
        llm=llm,
        chunk_size=chunk_size,
        mode=mode,
        embedding_model=embedding_model,
    )
    result = await cohort_task.await_result(session.client)
    if isinstance(result, TableResult):
        return result
    else:
        raise EveryrowError("Clean task did not return a table result")


async def clean_async(
    session: Session,
    input: DataFrame | UUID | TableResult,
    equivalence_relation: str,
    llm: LLM | None = None,
    chunk_size: int | None = None,
    mode: DedupeMode | None = None,
    embedding_model: EmbeddingModels | None = None,
) -> EveryrowTask[BaseModel]:
    """Submit a clean (dedupe) task asynchronously."""
    input_artifact_id = await _process_agent_map_input(input, session)

    query = DedupeQueryParams(
        equivalence_relation=equivalence_relation,
        llm=llm or UNSET,
        chunk_size=chunk_size or UNSET,
        mode=mode or UNSET,
        embedding_model=embedding_model or UNSET,
    )
    request = DedupeRequestParams(
        query=query,
        input_artifacts=[input_artifact_id],
        processing_mode=ProcessingMode.MAP,
    )
    body = SubmitTaskBody(
        payload=request,
        session_id=session.session_id,
    )

    cohort_task = EveryrowTask(response_model=BaseModel, is_map=True, is_expand=False)
    await cohort_task.submit(body, session.client)
    return cohort_task


async def derive(
    session: Session,
    input: DataFrame | UUID | TableResult,
    expressions: dict[str, str],
) -> TableResult:
    """Derive new columns using pandas eval expressions.

    Args:
        session: The session to use
        input: The input table (DataFrame, UUID, or TableResult)
        expressions: A dictionary mapping column names to pandas expressions.
            Example: {"approved": "True", "score": "price * quantity"}

    Returns:
        TableResult containing the table with new derived columns
    """
    input_artifact_id = await _process_agent_map_input(input, session)

    derive_expressions = [
        DeriveExpression(column_name=col_name, expression=expr) for col_name, expr in expressions.items()
    ]

    query = DeriveQueryParams(expressions=derive_expressions)
    request = DeriveRequest(
        query=query,
        input_artifacts=[input_artifact_id],
    )
    body = SubmitTaskBody(
        payload=request,
        session_id=session.session_id,
    )

    task_id = await submit_task(body, session.client)
    finished_task = await await_task_completion(task_id, session.client)

    data = await read_table_result(finished_task.artifact_id, session.client)  # type: ignore
    return TableResult(
        artifact_id=finished_task.artifact_id,  # type: ignore
        data=data,
        error=finished_task.error,
    )
