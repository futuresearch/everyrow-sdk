import json
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload
from uuid import UUID

from pandas import DataFrame

from everyrow.constants import EveryrowError
from everyrow.generated.models import (
    AgentQueryParams,
    CreateGroupQueryParams,
    CreateGroupRequest,
    CreateQueryParams,
    CreateRequest,
    DedupePublicParams,
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
    MapAgentRequestParams,
    ProcessingMode,
    ReduceAgentRequestParams,
    ResponseSchemaType,
)
from everyrow.generated.models.submit_task_body import SubmitTaskBody
from everyrow.generated.types import UNSET
from everyrow.result import Result, ScalarResult, TableResult
from everyrow.session import Session, create_session
from everyrow.task import (
    LLM,
    EffortLevel,
    EveryrowTask,
    await_task_completion,
    read_table_result,
    submit_task,
)

try:
    from pydantic import BaseModel

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    if TYPE_CHECKING:
        from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _ensure_pydantic() -> None:
    """Raise ImportError if Pydantic is not available."""
    if not PYDANTIC_AVAILABLE:
        raise ImportError(
            "Pydantic is required when using response_model. "
            "Install with: pip install everyrow[pydantic]"
        )


# Default JSON Schema constants for users who want the old default behavior
DEFAULT_AGENT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "DefaultAgentResponse",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

DEFAULT_SCREEN_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "DefaultScreenResult",
    "properties": {"passes": {"type": "boolean"}},
    "required": ["passes"],
}


def _validate_response_params(
    response_model: type[BaseModel] | None,
    response_schema: dict[str, Any] | None,
) -> tuple[dict[str, Any], type[BaseModel] | None]:
    """Validate XOR and return (schema_dict, model_or_none).

    Args:
        response_model: A Pydantic model class, or None
        response_schema: A JSON schema dict, or None

    Returns:
        A tuple of (schema_dict, response_model_or_none)

    Raises:
        EveryrowError: If both or neither of response_model/response_schema are provided
    """
    if response_model is not None and response_schema is not None:
        raise EveryrowError(
            "Cannot specify both response_model and response_schema. "
            "Use response_model for Pydantic models or response_schema for JSON schema dicts."
        )
    if response_model is None and response_schema is None:
        raise EveryrowError(
            "Must specify either response_model (Pydantic) or response_schema (JSON schema dict)."
        )

    if response_model is not None:
        _ensure_pydantic()
        return response_model.model_json_schema(), response_model

    assert response_schema is not None  # for type narrowing
    return response_schema, None


def _convert_json_schema_to_custom_schema(
    json_schema: dict[str, Any],
) -> dict[str, Any]:
    """Convert a JSON schema dict to the custom response schema format expected by rank/agent_map.

    The custom format uses _model_name instead of type: object, and uses optional: bool
    instead of required arrays.

    Example input (JSON Schema):
        {
            "type": "object",
            "title": "ScreeningResult",
            "properties": {
                "screening_result": {"type": "string", "description": "..."}
            },
            "required": ["screening_result"]
        }

    Example output (Custom Schema):
        {
            "_model_name": "ScreeningResult",
            "screening_result": {
                "type": "str",
                "optional": False,
                "description": "..."
            }
        }
    """
    # Extract model name from title or use a default
    model_name = json_schema.get("title", "Response")

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
        custom_field: dict[str, Any] = {}

        # Map type from JSON schema format to custom format
        field_type = field_schema.get("type")
        if field_type:
            custom_field["type"] = type_mapping.get(field_type, field_type)

        # Add description if present
        if "description" in field_schema:
            custom_field["description"] = field_schema["description"]

        # Set optional flag (opposite of required)
        custom_field["optional"] = field_name not in required

        custom_schema[field_name] = custom_field

    return custom_schema


def _ensure_custom_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON schema to custom format if needed.

    If the schema already has _model_name, it's already in custom format.
    If it has type: object, convert from JSON schema to custom format.
    """
    if "_model_name" in schema:
        return schema  # Already custom format
    if schema.get("type") == "object":
        return _convert_json_schema_to_custom_schema(schema)
    return schema


class DefaultAgentResponse(BaseModel):
    """Default response model for single_agent (kept for backwards compatibility reference)."""

    answer: str


class DefaultScreenResult(BaseModel):
    """Default response model for screen (kept for backwards compatibility reference)."""

    passes: bool


@overload
async def single_agent[T: BaseModel](
    task: str,
    session: Session | None = None,
    input: dict[str, Any] | BaseModel | UUID | Result | None = None,
    effort_level: EffortLevel = EffortLevel.LOW,
    llm: LLM | None = None,
    response_model: type[T] = ...,
    response_schema: None = None,
    return_table: Literal[False] = False,
) -> ScalarResult[T]: ...


@overload
async def single_agent(
    task: str,
    session: Session | None = None,
    input: dict[str, Any] | BaseModel | UUID | Result | None = None,
    effort_level: EffortLevel = EffortLevel.LOW,
    llm: LLM | None = None,
    response_model: None = None,
    response_schema: dict[str, Any] = ...,
    return_table: Literal[False] = False,
) -> ScalarResult[dict[str, Any]]: ...


@overload
async def single_agent(
    task: str,
    session: Session | None = None,
    input: dict[str, Any] | BaseModel | UUID | Result | None = None,
    effort_level: EffortLevel = EffortLevel.LOW,
    llm: LLM | None = None,
    response_model: type[BaseModel] | None = None,
    response_schema: dict[str, Any] | None = None,
    return_table: Literal[True] = ...,
) -> TableResult: ...


async def single_agent[T: BaseModel](
    task: str,
    session: Session | None = None,
    input: dict[str, Any] | BaseModel | DataFrame | UUID | Result | None = None,
    effort_level: EffortLevel = EffortLevel.LOW,
    llm: LLM | None = None,
    response_model: type[T] | None = None,
    response_schema: dict[str, Any] | None = None,
    return_table: bool = False,
) -> ScalarResult[T] | ScalarResult[dict[str, Any]] | TableResult:
    if session is None:
        async with create_session() as internal_session:
            cohort_task = await single_agent_async(
                task=task,
                session=internal_session,
                input=input,
                effort_level=effort_level,
                llm=llm,
                response_model=response_model,
                response_schema=response_schema,
                return_table=return_table,
            )
            return await cohort_task.await_result()
    cohort_task = await single_agent_async(
        task=task,
        session=session,
        input=input,
        effort_level=effort_level,
        llm=llm,
        response_model=response_model,
        response_schema=response_schema,
        return_table=return_table,
    )
    return await cohort_task.await_result()


async def single_agent_async[T: BaseModel](
    task: str,
    session: Session,
    input: dict[str, Any] | BaseModel | DataFrame | UUID | Result | None = None,
    effort_level: EffortLevel = EffortLevel.LOW,
    llm: LLM | None = None,
    response_model: type[T] | None = None,
    response_schema: dict[str, Any] | None = None,
    return_table: bool = False,
) -> EveryrowTask[T] | EveryrowTask[dict[str, Any]]:
    schema_dict, model_or_none = _validate_response_params(
        response_model, response_schema
    )

    if input is not None:
        input_artifact_ids = [await _process_single_agent_input(input, session)]
    else:
        input_artifact_ids = []

    query = AgentQueryParams(
        task=task,
        llm=llm or UNSET,
        effort_level=effort_level,
        response_schema=schema_dict,
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

    cohort_task: EveryrowTask[T] | EveryrowTask[dict[str, Any]] = EveryrowTask(  # type: ignore[assignment]
        response_model=model_or_none, is_map=False, is_expand=return_table
    )
    await cohort_task.submit(body, session.client)
    return cohort_task


async def agent_map(
    task: str,
    session: Session | None = None,
    input: DataFrame | UUID | TableResult | None = None,
    effort_level: EffortLevel = EffortLevel.LOW,
    llm: LLM | None = None,
    response_model: type[BaseModel] | None = None,
    response_schema: dict[str, Any] | None = None,
) -> TableResult:
    if input is None:
        raise EveryrowError("input is required for agent_map")
    if session is None:
        async with create_session() as internal_session:
            cohort_task = await agent_map_async(
                task,
                internal_session,
                input,
                effort_level,
                llm,
                response_model,
                response_schema,
            )
            result = await cohort_task.await_result()
            if isinstance(result, TableResult):
                return result
            else:
                raise EveryrowError("Agent map task did not return a table result")
    cohort_task = await agent_map_async(
        task, session, input, effort_level, llm, response_model, response_schema
    )
    result = await cohort_task.await_result()
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
    response_model: type[BaseModel] | None = None,
    response_schema: dict[str, Any] | None = None,
) -> EveryrowTask[BaseModel] | EveryrowTask[dict[str, Any]]:
    schema_dict, model_or_none = _validate_response_params(
        response_model, response_schema
    )
    custom_schema = _ensure_custom_schema(schema_dict)

    input_artifact_ids = [await _process_agent_map_input(input, session)]
    query = AgentQueryParams(
        task=task,
        effort_level=effort_level,
        llm=llm or UNSET,
        response_schema=custom_schema,
        response_schema_type=ResponseSchemaType.CUSTOM,
        is_expand=False,
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

    cohort_task: EveryrowTask[BaseModel] | EveryrowTask[dict[str, Any]] = EveryrowTask(
        response_model=model_or_none, is_map=True, is_expand=False
    )
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
    input: dict[str, Any] | BaseModel | DataFrame | UUID | Result,
    session: Session,
) -> UUID:
    if isinstance(input, Result):
        return input.artifact_id
    elif isinstance(input, DataFrame):
        return await create_table_artifact(input, session)
    elif isinstance(input, dict):
        return await create_scalar_artifact_from_dict(input, session)
    elif PYDANTIC_AVAILABLE and isinstance(input, BaseModel):
        return await create_scalar_artifact(input, session)
    else:
        return input  # type: ignore[return-value]


async def create_scalar_artifact(input: BaseModel, session: Session) -> UUID:
    payload = CreateRequest(query=CreateQueryParams(data_to_create=input.model_dump()))
    body = SubmitTaskBody(
        payload=payload,
        session_id=session.session_id,
    )
    task_id = await submit_task(body, session.client)
    finished_create_artifact_task = await await_task_completion(task_id, session.client)
    return finished_create_artifact_task.artifact_id  # type: ignore (we check artifact_id in await_task_completion)


async def create_scalar_artifact_from_dict(
    input: dict[str, Any], session: Session
) -> UUID:
    """Create a scalar artifact from a dict."""
    payload = CreateRequest(query=CreateQueryParams(data_to_create=input))
    body = SubmitTaskBody(
        payload=payload,
        session_id=session.session_id,
    )
    task_id = await submit_task(body, session.client)
    finished_create_artifact_task = await await_task_completion(task_id, session.client)
    return finished_create_artifact_task.artifact_id  # type: ignore (we check artifact_id in await_task_completion)


async def create_table_artifact(input: DataFrame, session: Session) -> UUID:
    # Use to_json to handle NaN/NaT serialization, then parse back to Python objects
    json_str = input.to_json(orient="records")
    assert json_str is not None  # to_json returns str when no path_or_buf provided
    records = json.loads(json_str)
    payload = CreateGroupRequest(query=CreateGroupQueryParams(data_to_create=records))
    body = SubmitTaskBody(
        payload=payload,
        session_id=session.session_id,
    )
    task_id = await submit_task(body, session.client)
    finished_create_artifact_task = await await_task_completion(task_id, session.client)
    return finished_create_artifact_task.artifact_id  # type: ignore (we check artifact_id in await_task_completion)


async def merge(
    task: str,
    session: Session | None = None,
    left_table: DataFrame | UUID | TableResult | None = None,
    right_table: DataFrame | UUID | TableResult | None = None,
    merge_on_left: str | None = None,
    merge_on_right: str | None = None,
) -> TableResult:
    """Merge two tables using merge operation.

    Args:
        task: The task description for the merge operation
        session: Optional session. If not provided, one will be created automatically.
        left_table: The left table to merge (DataFrame, UUID, or TableResult)
        right_table: The right table to merge (DataFrame, UUID, or TableResult)
        merge_on_left: Optional column name in left table to merge on
        merge_on_right: Optional column name in right table to merge on

    Returns:
        TableResult containing the merged table
    """
    if left_table is None or right_table is None:
        raise EveryrowError("left_table and right_table are required for merge")
    if session is None:
        async with create_session() as internal_session:
            cohort_task = await merge_async(
                task=task,
                session=internal_session,
                left_table=left_table,
                right_table=right_table,
                merge_on_left=merge_on_left,
                merge_on_right=merge_on_right,
            )
            result = await cohort_task.await_result()
            if isinstance(result, TableResult):
                return result
            else:
                raise EveryrowError("Merge task did not return a table result")
    cohort_task = await merge_async(
        task=task,
        session=session,
        left_table=left_table,
        right_table=right_table,
        merge_on_left=merge_on_left,
        merge_on_right=merge_on_right,
    )
    result = await cohort_task.await_result()
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
) -> EveryrowTask[BaseModel]:
    """Submit a merge task asynchronously."""
    left_artifact_id = await _process_agent_map_input(left_table, session)
    right_artifact_id = await _process_agent_map_input(right_table, session)

    query = DeepMergePublicParams(
        task=task,
        merge_on_left=merge_on_left or UNSET,
        merge_on_right=merge_on_right or UNSET,
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


async def rank[T: BaseModel](
    task: str,
    session: Session | None = None,
    input: DataFrame | UUID | TableResult | None = None,
    field_name: str | None = None,
    field_type: Literal["float", "int", "str", "bool"] = "float",
    response_model: type[T] | None = None,
    response_schema: dict[str, Any] | None = None,
    ascending_order: bool = True,
) -> TableResult:
    """Rank rows in a table using rank operation.

    Args:
        task: The task description for ranking
        session: Optional session. If not provided, one will be created automatically.
        input: The input table (DataFrame, UUID, or TableResult)
        field_name: The name of the field to extract and sort by
        field_type: The type of the field (default: "float", ignored if response_model or response_schema is provided)
        response_model: Optional Pydantic model for the response schema (mutually exclusive with response_schema)
        response_schema: Optional JSON schema dict for the response (mutually exclusive with response_model)
        ascending_order: If True, sort in ascending order

    Returns:
        TableResult containing the ranked table
    """
    if input is None or field_name is None:
        raise EveryrowError("input and field_name are required for rank")
    if session is None:
        async with create_session() as internal_session:
            cohort_task = await rank_async(
                task=task,
                session=internal_session,
                input=input,
                field_name=field_name,
                field_type=field_type,
                response_model=response_model,
                response_schema=response_schema,
                ascending_order=ascending_order,
            )
            result = await cohort_task.await_result()
            if isinstance(result, TableResult):
                return result
            else:
                raise EveryrowError("Rank task did not return a table result")
    cohort_task = await rank_async(
        task=task,
        session=session,
        input=input,
        field_name=field_name,
        field_type=field_type,
        response_model=response_model,
        response_schema=response_schema,
        ascending_order=ascending_order,
    )
    result = await cohort_task.await_result()
    if isinstance(result, TableResult):
        return result
    else:
        raise EveryrowError("Rank task did not return a table result")


async def rank_async[T: BaseModel](
    task: str,
    session: Session,
    input: DataFrame | UUID | TableResult,
    field_name: str,
    field_type: Literal["float", "int", "str", "bool"] = "float",
    response_model: type[T] | None = None,
    response_schema: dict[str, Any] | None = None,
    ascending_order: bool = True,
) -> EveryrowTask[T] | EveryrowTask[dict[str, Any]]:
    """Submit a rank task asynchronously."""
    input_artifact_id = await _process_agent_map_input(input, session)

    # For rank, we have special handling - if neither response_model nor response_schema
    # is provided, we generate a simple schema based on field_name and field_type
    if response_model is not None and response_schema is not None:
        raise EveryrowError(
            "Cannot specify both response_model and response_schema. "
            "Use response_model for Pydantic models or response_schema for JSON schema dicts."
        )

    model_or_none: type[T] | None = None
    if response_model is not None:
        _ensure_pydantic()
        custom_schema = _convert_pydantic_to_custom_schema(response_model)
        if field_name not in custom_schema:
            raise ValueError(
                f"Field {field_name} not in response model {response_model.__name__}"
            )
        model_or_none = response_model
    elif response_schema is not None:
        custom_schema = _ensure_custom_schema(response_schema)
        if field_name not in custom_schema:
            # Check in properties for JSON schema format
            props = response_schema.get("properties", {})
            if field_name not in props:
                schema_name = response_schema.get("title", "response_schema")
                raise ValueError(f"Field {field_name} not in {schema_name}")
    else:
        # Generate a simple schema based on field_name and field_type
        custom_schema = {
            "_model_name": "RankResponse",
            field_name: {
                "type": field_type,
                "optional": False,
            },
        }

    query = DeepRankPublicParams(
        task=task,
        response_schema=custom_schema,
        field_to_sort_by=field_name,
        ascending_order=ascending_order,
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

    cohort_task: EveryrowTask[T] | EveryrowTask[dict[str, Any]] = EveryrowTask(
        response_model=model_or_none,
        is_map=True,
        is_expand=False,
    )
    await cohort_task.submit(body, session.client)
    return cohort_task


async def screen[T: BaseModel](
    task: str,
    session: Session | None = None,
    input: DataFrame | UUID | TableResult | None = None,
    response_model: type[T] | None = None,
    response_schema: dict[str, Any] | None = None,
) -> TableResult:
    """Screen rows in a table using screen operation.

    Args:
        task: The task description for screening
        session: Optional session. If not provided, one will be created automatically.
        input: The input table (DataFrame, UUID, or TableResult)
        response_model: Optional Pydantic model for the response schema (mutually exclusive with response_schema).
        response_schema: Optional JSON schema dict for the response (mutually exclusive with response_model).
            If neither is provided, must be explicitly specified.

    Returns:
        TableResult containing the screened table
    """
    if input is None:
        raise EveryrowError("input is required for screen")
    if session is None:
        async with create_session() as internal_session:
            cohort_task = await screen_async(
                task=task,
                session=internal_session,
                input=input,
                response_model=response_model,
                response_schema=response_schema,
            )
            result = await cohort_task.await_result()
            if isinstance(result, TableResult):
                return result
            else:
                raise EveryrowError("Screen task did not return a table result")
    cohort_task = await screen_async(
        task=task,
        session=session,
        input=input,
        response_model=response_model,
        response_schema=response_schema,
    )
    result = await cohort_task.await_result()
    if isinstance(result, TableResult):
        return result
    else:
        raise EveryrowError("Screen task did not return a table result")


async def screen_async[T: BaseModel](
    task: str,
    session: Session,
    input: DataFrame | UUID | TableResult,
    response_model: type[T] | None = None,
    response_schema: dict[str, Any] | None = None,
) -> EveryrowTask[T] | EveryrowTask[dict[str, Any]]:
    """Submit a screen task asynchronously."""
    schema_dict, model_or_none = _validate_response_params(
        response_model, response_schema
    )
    input_artifact_id = await _process_agent_map_input(input, session)

    query = DeepScreenPublicParams(
        task=task,
        response_schema=schema_dict,
        response_schema_type=ResponseSchemaType.JSON,
    )
    request = DeepScreenRequest(
        query=query,
        input_artifacts=[input_artifact_id],
    )
    body = SubmitTaskBody(
        payload=request,
        session_id=session.session_id,
    )

    cohort_task: EveryrowTask[T] | EveryrowTask[dict[str, Any]] = EveryrowTask(  # type: ignore[assignment]
        response_model=model_or_none,
        is_map=True,
        is_expand=False,
    )
    await cohort_task.submit(body, session.client)
    return cohort_task


async def dedupe(
    equivalence_relation: str,
    session: Session | None = None,
    input: DataFrame | UUID | TableResult | None = None,
    select_representative: bool = True,
) -> TableResult:
    """Dedupe a table by removing duplicates using dedupe operation.

    Args:
        equivalence_relation: Description of what makes items equivalent
        session: Optional session. If not provided, one will be created automatically.
        input: The input table (DataFrame, UUID, or TableResult)
        select_representative: If True, select a representative for each group of duplicates

    Returns:
        TableResult containing the deduped table with duplicates removed
    """
    if input is None or equivalence_relation is None:
        raise EveryrowError("input and equivalence_relation are required for dedupe")
    if session is None:
        async with create_session() as internal_session:
            cohort_task = await dedupe_async(
                session=internal_session,
                input=input,
                equivalence_relation=equivalence_relation,
                select_representative=select_representative,
            )
            result = await cohort_task.await_result()
            if isinstance(result, TableResult):
                return result
            else:
                raise EveryrowError("Dedupe task did not return a table result")
    cohort_task = await dedupe_async(
        session=session,
        input=input,
        equivalence_relation=equivalence_relation,
        select_representative=select_representative,
    )
    result = await cohort_task.await_result()
    if isinstance(result, TableResult):
        return result
    else:
        raise EveryrowError("Dedupe task did not return a table result")


async def dedupe_async(
    session: Session,
    input: DataFrame | UUID | TableResult,
    equivalence_relation: str,
    select_representative: bool = True,
) -> EveryrowTask[BaseModel]:
    """Submit a dedupe task asynchronously."""
    input_artifact_id = await _process_agent_map_input(input, session)

    query = DedupePublicParams(
        equivalence_relation=equivalence_relation,
        select_representative=select_representative,
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
    session: Session | None = None,
    input: DataFrame | UUID | TableResult | None = None,
    expressions: dict[str, str] | None = None,
) -> TableResult:
    """Derive new columns using pandas eval expressions.

    Args:
        session: Optional session. If not provided, one will be created automatically.
        input: The input table (DataFrame, UUID, or TableResult)
        expressions: A dictionary mapping column names to pandas expressions.
            Example: {"approved": "True", "score": "price * quantity"}

    Returns:
        TableResult containing the table with new derived columns
    """
    if input is None or expressions is None:
        raise EveryrowError("input and expressions are required for derive")
    if session is None:
        async with create_session() as internal_session:
            input_artifact_id = await _process_agent_map_input(input, internal_session)

            derive_expressions = [
                DeriveExpression(column_name=col_name, expression=expr)
                for col_name, expr in expressions.items()
            ]

            query = DeriveQueryParams(expressions=derive_expressions)
            request = DeriveRequest(
                query=query,
                input_artifacts=[input_artifact_id],
            )
            body = SubmitTaskBody(
                payload=request,
                session_id=internal_session.session_id,
            )

            task_id = await submit_task(body, internal_session.client)
            finished_task = await await_task_completion(
                task_id, internal_session.client
            )

            data = await read_table_result(
                finished_task.artifact_id,  # type: ignore[arg-type]
                internal_session.client,
            )
            return TableResult(
                artifact_id=finished_task.artifact_id,  # type: ignore
                data=data,
                error=finished_task.error,
            )
    input_artifact_id = await _process_agent_map_input(input, session)

    derive_expressions = [
        DeriveExpression(column_name=col_name, expression=expr)
        for col_name, expr in expressions.items()
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
