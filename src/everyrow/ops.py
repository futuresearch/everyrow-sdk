import json
from typing import Any, Literal, TypeVar, overload
from uuid import UUID

from pandas import DataFrame
from pydantic import BaseModel

from everyrow.api_utils import handle_response
from everyrow.constants import EveryrowError
from everyrow.generated.api.artifacts import create_artifact_artifacts_post
from everyrow.generated.api.operations import (
    agent_map_operations_agent_map_post,
    dedupe_operations_dedupe_post,
    merge_operations_merge_post,
    rank_operations_rank_post,
    screen_operations_screen_post,
    single_agent_operations_single_agent_post,
)
from everyrow.generated.models import (
    AgentMapOperation,
    AgentMapOperationInputType1Item,
    AgentMapOperationResponseSchemaType0,
    CreateArtifactRequest,
    CreateArtifactRequestDataType0Item,
    CreateArtifactRequestDataType1,
    DedupeOperation,
    DedupeOperationInputType1Item,
    LLMEnumPublic,
    MergeOperation,
    MergeOperationLeftInputType1Item,
    MergeOperationRightInputType1Item,
    PublicEffortLevel,
    RankOperation,
    RankOperationInputType1Item,
    RankOperationResponseSchemaType0,
    ScreenOperation,
    ScreenOperationInputType1Item,
    ScreenOperationResponseSchemaType0,
    SingleAgentOperation,
    SingleAgentOperationInputType1Item,
    SingleAgentOperationInputType2,
    SingleAgentOperationResponseSchemaType0,
)
from everyrow.generated.types import UNSET
from everyrow.result import MergeResult, Result, ScalarResult, TableResult
from everyrow.session import Session, create_session
from everyrow.task import LLM, EffortLevel, EveryrowTask, MergeTask

T = TypeVar("T", bound=BaseModel)
InputData = UUID | list[dict[str, Any]] | dict[str, Any]


class DefaultAgentResponse(BaseModel):
    answer: str


class DefaultScreenResult(BaseModel):
    passes: bool


def _df_to_records(df: DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to a list of records, handling NaN/NaT."""
    json_str = df.to_json(orient="records")
    assert json_str is not None
    return json.loads(json_str)


def _prepare_table_input[T](
    input: DataFrame | UUID | TableResult | None,
    item_class: type[T],
) -> UUID | list[T]:
    """Convert table input to UUID or list of generated model items."""
    if input is None:
        return []
    if isinstance(input, UUID):
        return input
    if isinstance(input, TableResult):
        return input.artifact_id
    if isinstance(input, DataFrame):
        records = _df_to_records(input)
        return [item_class.from_dict(r) for r in records]  # type: ignore[attr-defined]
    raise TypeError(f"Unsupported input type: {type(input)}")


def _prepare_single_input[TItem, TObj](
    input: BaseModel | DataFrame | UUID | Result | None,
    item_class: type[TItem],
    object_class: type[TObj],
) -> UUID | list[TItem] | TObj:
    """Convert single-agent input to the appropriate generated model type."""
    if input is None:
        return object_class.from_dict({})  # type: ignore[attr-defined]
    if isinstance(input, UUID):
        return input
    if isinstance(input, Result):
        return input.artifact_id
    if isinstance(input, DataFrame):
        records = _df_to_records(input)
        return [item_class.from_dict(r) for r in records]  # type: ignore[attr-defined]
    if isinstance(input, BaseModel):
        return object_class.from_dict(input.model_dump())  # type: ignore[attr-defined]
    raise TypeError(f"Unsupported input type: {type(input)}")


# --- Artifact creation ---


async def create_scalar_artifact(input: BaseModel, session: Session) -> UUID:
    """Create a scalar artifact by uploading a single record."""
    body = CreateArtifactRequest(
        data=CreateArtifactRequestDataType1.from_dict(input.model_dump()),
        session_id=session.session_id,
    )
    response = await create_artifact_artifacts_post.asyncio(client=session.client, body=body)
    response = handle_response(response)
    return response.artifact_id


async def create_table_artifact(input: DataFrame, session: Session) -> UUID:
    """Create a table artifact by uploading a list of records."""
    records = _df_to_records(input)
    body = CreateArtifactRequest(
        data=[CreateArtifactRequestDataType0Item.from_dict(r) for r in records],
        session_id=session.session_id,
    )
    response = await create_artifact_artifacts_post.asyncio(client=session.client, body=body)
    response = handle_response(response)
    return response.artifact_id


# --- Single Agent ---


@overload
async def single_agent[T: BaseModel](
    task: str,
    session: Session | None = None,
    input: BaseModel | UUID | Result | None = None,
    effort_level: EffortLevel | None = EffortLevel.LOW,
    llm: LLM | None = None,
    iteration_budget: int | None = None,
    include_research: bool | None = None,
    response_model: type[T] = DefaultAgentResponse,
    return_table: Literal[False] = False,
    webhook_url: str | None = None,
) -> ScalarResult[T]: ...


@overload
async def single_agent(
    task: str,
    session: Session | None = None,
    input: BaseModel | UUID | Result | None = None,
    effort_level: EffortLevel | None = EffortLevel.LOW,
    llm: LLM | None = None,
    iteration_budget: int | None = None,
    include_research: bool | None = None,
    response_model: type[BaseModel] = DefaultAgentResponse,
    return_table: Literal[True] = True,
    webhook_url: str | None = None,
) -> TableResult: ...


async def single_agent[T: BaseModel](
    task: str,
    session: Session | None = None,
    input: BaseModel | DataFrame | UUID | Result | None = None,
    effort_level: EffortLevel | None = EffortLevel.LOW,
    llm: LLM | None = None,
    iteration_budget: int | None = None,
    include_research: bool | None = None,
    response_model: type[T] = DefaultAgentResponse,
    return_table: bool = False,
    webhook_url: str | None = None,
) -> ScalarResult[T] | TableResult:
    """Execute an AI agent task on the provided input.

    Args:
        task: Instructions for the AI agent to execute.
        session: Optional session. If not provided, one will be created automatically.
        input: Input data (BaseModel, DataFrame, UUID, or Result).
        effort_level: Effort level preset (low/medium/high). Mutually exclusive with
            custom params (llm, iteration_budget, include_research). Default: low.
        llm: LLM to use. Required when effort_level is None.
        iteration_budget: Number of agent iterations (0-20). Required when effort_level is None.
        include_research: Include research notes. Required when effort_level is None.
        response_model: Pydantic model for the response schema.
        return_table: If True, return a TableResult instead of ScalarResult.
        webhook_url: Optional URL to receive a POST callback when the task completes or fails.

    Returns:
        ScalarResult or TableResult depending on return_table parameter.
    """
    if session is None:
        async with create_session() as internal_session:
            cohort_task = await single_agent_async(
                task=task,
                session=internal_session,
                input=input,
                effort_level=effort_level,
                llm=llm,
                iteration_budget=iteration_budget,
                include_research=include_research,
                response_model=response_model,
                return_table=return_table,
                webhook_url=webhook_url,
            )
            return await cohort_task.await_result()
    cohort_task = await single_agent_async(
        task=task,
        session=session,
        input=input,
        effort_level=effort_level,
        llm=llm,
        iteration_budget=iteration_budget,
        include_research=include_research,
        response_model=response_model,
        return_table=return_table,
        webhook_url=webhook_url,
    )
    return await cohort_task.await_result()


async def single_agent_async[T: BaseModel](
    task: str,
    session: Session,
    input: BaseModel | DataFrame | UUID | Result | None = None,
    effort_level: EffortLevel | None = EffortLevel.LOW,
    llm: LLM | None = None,
    iteration_budget: int | None = None,
    include_research: bool | None = None,
    response_model: type[T] = DefaultAgentResponse,
    return_table: bool = False,
    webhook_url: str | None = None,
) -> EveryrowTask[T]:
    """Submit a single_agent task asynchronously."""
    input_data = _prepare_single_input(input, SingleAgentOperationInputType1Item, SingleAgentOperationInputType2)

    # Build the operation body with either preset or custom params
    body = SingleAgentOperation(
        input_=input_data,  # type: ignore
        task=task,
        session_id=session.session_id,
        response_schema=SingleAgentOperationResponseSchemaType0.from_dict(response_model.model_json_schema()),
        effort_level=PublicEffortLevel(effort_level.value) if effort_level is not None else UNSET,
        llm=LLMEnumPublic(llm.value) if llm is not None else UNSET,
        iteration_budget=iteration_budget if iteration_budget is not None else UNSET,
        include_research=include_research if include_research is not None else UNSET,
        return_list=return_table,
    )
    if webhook_url is not None:
        body["webhook_url"] = webhook_url

    response = await single_agent_operations_single_agent_post.asyncio(client=session.client, body=body)
    response = handle_response(response)

    cohort_task: EveryrowTask[T] = EveryrowTask(response_model=response_model, is_map=False, is_expand=return_table)
    cohort_task.set_submitted(response.task_id, response.session_id, session.client)
    return cohort_task


# --- Agent Map ---


async def agent_map(
    task: str,
    session: Session | None = None,
    input: DataFrame | UUID | TableResult | None = None,
    effort_level: EffortLevel | None = EffortLevel.LOW,
    llm: LLM | None = None,
    iteration_budget: int | None = None,
    include_research: bool | None = None,
    enforce_row_independence: bool = False,
    response_model: type[BaseModel] = DefaultAgentResponse,
    webhook_url: str | None = None,
) -> TableResult:
    """Execute an AI agent task on each row of the input table.

    Args:
        task: Instructions for the AI agent to execute per row.
        session: Optional session. If not provided, one will be created automatically.
        input: The input table (DataFrame, UUID, or TableResult).
        effort_level: Effort level preset (low/medium/high). Mutually exclusive with
            custom params (llm, iteration_budget, include_research). Default: low.
        llm: LLM to use for each agent. Required when effort_level is None.
        iteration_budget: Number of agent iterations per row (0-20). Required when effort_level is None.
        include_research: Include research notes. Required when effort_level is None.
        response_model: Pydantic model for the response schema.
        webhook_url: Optional URL to receive a POST callback when the task completes or fails.

    Returns:
        TableResult containing the agent results merged with input rows.
    """
    if input is None:
        raise EveryrowError("input is required for agent_map")
    if session is None:
        async with create_session() as internal_session:
            cohort_task = await agent_map_async(
                task=task,
                session=internal_session,
                input=input,
                effort_level=effort_level,
                llm=llm,
                iteration_budget=iteration_budget,
                include_research=include_research,
                enforce_row_independence=enforce_row_independence,
                response_model=response_model,
                webhook_url=webhook_url,
            )
            result = await cohort_task.await_result()
            if isinstance(result, TableResult):
                return result
            raise EveryrowError("Agent map task did not return a table result")
    cohort_task = await agent_map_async(
        task=task,
        session=session,
        input=input,
        effort_level=effort_level,
        llm=llm,
        iteration_budget=iteration_budget,
        include_research=include_research,
        enforce_row_independence=enforce_row_independence,
        response_model=response_model,
        webhook_url=webhook_url,
    )
    result = await cohort_task.await_result()
    if isinstance(result, TableResult):
        return result
    raise EveryrowError("Agent map task did not return a table result")


async def agent_map_async(
    task: str,
    session: Session,
    input: DataFrame | UUID | TableResult,
    effort_level: EffortLevel | None = EffortLevel.LOW,
    llm: LLM | None = None,
    iteration_budget: int | None = None,
    include_research: bool | None = None,
    enforce_row_independence: bool = False,
    response_model: type[BaseModel] = DefaultAgentResponse,
    webhook_url: str | None = None,
) -> EveryrowTask[BaseModel]:
    """Submit an agent_map task asynchronously."""
    input_data = _prepare_table_input(input, AgentMapOperationInputType1Item)

    # Build the operation body with either preset or custom params
    body = AgentMapOperation(
        input_=input_data,  # type: ignore
        task=task,
        session_id=session.session_id,
        response_schema=AgentMapOperationResponseSchemaType0.from_dict(response_model.model_json_schema()),
        effort_level=PublicEffortLevel(effort_level.value) if effort_level is not None else UNSET,
        llm=LLMEnumPublic(llm.value) if llm is not None else UNSET,
        iteration_budget=iteration_budget if iteration_budget is not None else UNSET,
        include_research=include_research if include_research is not None else UNSET,
        join_with_input=True,
        enforce_row_independence=enforce_row_independence,
    )
    if webhook_url is not None:
        body["webhook_url"] = webhook_url

    response = await agent_map_operations_agent_map_post.asyncio(client=session.client, body=body)
    response = handle_response(response)

    cohort_task = EveryrowTask(response_model=response_model, is_map=True, is_expand=False)
    cohort_task.set_submitted(response.task_id, response.session_id, session.client)
    return cohort_task


# --- Screen ---


async def screen[T: BaseModel](
    task: str,
    session: Session | None = None,
    input: DataFrame | UUID | TableResult | None = None,
    response_model: type[T] | None = None,
    webhook_url: str | None = None,
) -> TableResult:
    """Screen rows in a table using AI.

    Args:
        task: The task description for screening
        session: Optional session. If not provided, one will be created automatically.
        input: The input table (DataFrame, UUID, or TableResult)
        response_model: Optional Pydantic model for the response schema.
        webhook_url: Optional URL to receive a POST callback when the task completes or fails.

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
                webhook_url=webhook_url,
            )
            result = await cohort_task.await_result()
            if isinstance(result, TableResult):
                return result
            raise EveryrowError("Screen task did not return a table result")
    cohort_task = await screen_async(task=task, session=session, input=input, response_model=response_model, webhook_url=webhook_url)
    result = await cohort_task.await_result()
    if isinstance(result, TableResult):
        return result
    raise EveryrowError("Screen task did not return a table result")


async def screen_async[T: BaseModel](
    task: str,
    session: Session,
    input: DataFrame | UUID | TableResult,
    response_model: type[T] | None = None,
    webhook_url: str | None = None,
) -> EveryrowTask[T]:
    """Submit a screen task asynchronously."""
    input_data = _prepare_table_input(input, ScreenOperationInputType1Item)
    actual_response_model = response_model or DefaultScreenResult

    body = ScreenOperation(
        input_=input_data,  # type: ignore
        task=task,
        session_id=session.session_id,
        response_schema=ScreenOperationResponseSchemaType0.from_dict(actual_response_model.model_json_schema()),
    )
    if webhook_url is not None:
        body["webhook_url"] = webhook_url

    response = await screen_operations_screen_post.asyncio(client=session.client, body=body)
    response = handle_response(response)

    cohort_task: EveryrowTask[T] = EveryrowTask(
        response_model=actual_response_model,  # type: ignore[arg-type]
        is_map=True,
        is_expand=False,
    )
    cohort_task.set_submitted(response.task_id, response.session_id, session.client)
    return cohort_task


# --- Rank ---


async def rank[T: BaseModel](
    task: str,
    session: Session | None = None,
    input: DataFrame | UUID | TableResult | None = None,
    field_name: str | None = None,
    field_type: Literal["float", "int", "str", "bool"] = "float",
    response_model: type[T] | None = None,
    ascending_order: bool = True,
    webhook_url: str | None = None,
) -> TableResult:
    """Rank rows in a table using AI.

    Args:
        task: The task description for ranking
        session: Optional session. If not provided, one will be created automatically.
        input: The input table (DataFrame, UUID, or TableResult)
        field_name: The name of the field to sort by
        field_type: The type of the field (default: "float", ignored if response_model is provided)
        response_model: Optional Pydantic model for the response schema
        ascending_order: If True, sort in ascending order
        webhook_url: Optional URL to receive a POST callback when the task completes or fails.

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
                ascending_order=ascending_order,
                webhook_url=webhook_url,
            )
            result = await cohort_task.await_result()
            if isinstance(result, TableResult):
                return result
            raise EveryrowError("Rank task did not return a table result")
    cohort_task = await rank_async(
        task=task,
        session=session,
        input=input,
        field_name=field_name,
        field_type=field_type,
        response_model=response_model,
        ascending_order=ascending_order,
        webhook_url=webhook_url,
    )
    result = await cohort_task.await_result()
    if isinstance(result, TableResult):
        return result
    raise EveryrowError("Rank task did not return a table result")


async def rank_async[T: BaseModel](
    task: str,
    session: Session,
    input: DataFrame | UUID | TableResult,
    field_name: str,
    field_type: Literal["float", "int", "str", "bool"] = "float",
    response_model: type[T] | None = None,
    ascending_order: bool = True,
    webhook_url: str | None = None,
) -> EveryrowTask[T]:
    """Submit a rank task asynchronously."""
    input_data = _prepare_table_input(input, RankOperationInputType1Item)

    if response_model is not None:
        response_schema = response_model.model_json_schema()
        # Validate that field_name exists in the model
        properties = response_schema.get("properties", {})
        if field_name not in properties:
            raise ValueError(f"Field {field_name} not in response model {response_model.__name__}")
    else:
        # Build a minimal JSON schema with just the sort field
        json_type_map = {
            "float": "number",
            "int": "integer",
            "str": "string",
            "bool": "boolean",
        }
        response_schema = {
            "type": "object",
            "properties": {field_name: {"type": json_type_map.get(field_type, field_type)}},
            "required": [field_name],
        }

    body = RankOperation(
        input_=input_data,  # type: ignore
        task=task,
        sort_by=field_name,
        session_id=session.session_id,
        response_schema=RankOperationResponseSchemaType0.from_dict(response_schema),
        ascending=ascending_order,
    )
    if webhook_url is not None:
        body["webhook_url"] = webhook_url

    response = await rank_operations_rank_post.asyncio(client=session.client, body=body)
    response = handle_response(response)

    cohort_task: EveryrowTask[T] = EveryrowTask(
        response_model=response_model or BaseModel,  # type: ignore[arg-type]
        is_map=True,
        is_expand=False,
    )
    cohort_task.set_submitted(response.task_id, response.session_id, session.client)
    return cohort_task


# --- Merge ---


async def merge(
    task: str,
    session: Session | None = None,
    left_table: DataFrame | UUID | TableResult | None = None,
    right_table: DataFrame | UUID | TableResult | None = None,
    merge_on_left: str | None = None,
    merge_on_right: str | None = None,
    use_web_search: Literal["auto", "yes", "no"] | None = None,
    webhook_url: str | None = None,
) -> MergeResult:
    """Merge two tables using AI.

    Args:
        task: The task description for the merge operation
        session: Optional session. If not provided, one will be created automatically.
        left_table: The left table to merge (DataFrame, UUID, or TableResult)
        right_table: The right table to merge (DataFrame, UUID, or TableResult)
        merge_on_left: Optional column name in left table to merge on
        merge_on_right: Optional column name in right table to merge on
        use_web_search: Optional. Control web search behavior: "auto" tries LLM merge first then conditionally searches, "no" skips web search entirely, "yes" forces web search on every row. Defaults to "auto" if not provided.
        webhook_url: Optional URL to receive a POST callback when the task completes or fails.

    Returns:
        MergeResult containing the merged table and match breakdown by method (exact, fuzzy, llm, web)

    Example:
        result = await merge(task="...", left_table=df_left, right_table=df_right)
        print(f"Exact matches: {len(result.breakdown.exact)}")
        print(f"LLM matches: {len(result.breakdown.llm)}")
        print(f"Unmatched left rows: {result.breakdown.unmatched_left}")
    """
    if left_table is None or right_table is None:
        raise EveryrowError("left_table and right_table are required for merge")
    if session is None:
        async with create_session() as internal_session:
            merge_task = await merge_async(
                task=task,
                session=internal_session,
                left_table=left_table,
                right_table=right_table,
                merge_on_left=merge_on_left,
                merge_on_right=merge_on_right,
                use_web_search=use_web_search,
                webhook_url=webhook_url,
            )
            return await merge_task.await_result()
    merge_task = await merge_async(
        task=task,
        session=session,
        left_table=left_table,
        right_table=right_table,
        merge_on_left=merge_on_left,
        merge_on_right=merge_on_right,
        use_web_search=use_web_search,
        webhook_url=webhook_url,
    )
    return await merge_task.await_result()


async def merge_async(
    task: str,
    session: Session,
    left_table: DataFrame | UUID | TableResult,
    right_table: DataFrame | UUID | TableResult,
    merge_on_left: str | None = None,
    merge_on_right: str | None = None,
    use_web_search: Literal["auto", "yes", "no"] | None = None,
    webhook_url: str | None = None,
) -> MergeTask:
    """Submit a merge task asynchronously.

    Returns:
        MergeTask that can be awaited for a MergeResult with match breakdown
    """
    left_data = _prepare_table_input(left_table, MergeOperationLeftInputType1Item)
    right_data = _prepare_table_input(right_table, MergeOperationRightInputType1Item)

    body = MergeOperation(
        left_input=left_data,  # type: ignore
        right_input=right_data,  # type: ignore
        task=task,
        left_key=merge_on_left or UNSET,
        right_key=merge_on_right or UNSET,
        use_web_search=use_web_search or UNSET,  # type: ignore
        session_id=session.session_id,
    )
    if webhook_url is not None:
        body["webhook_url"] = webhook_url

    response = await merge_operations_merge_post.asyncio(client=session.client, body=body)
    response = handle_response(response)

    merge_task = MergeTask()
    merge_task.set_submitted(response.task_id, response.session_id, session.client)
    return merge_task


# --- Dedupe ---


async def dedupe(
    equivalence_relation: str,
    session: Session | None = None,
    input: DataFrame | UUID | TableResult | None = None,
    webhook_url: str | None = None,
) -> TableResult:
    """Dedupe a table by removing duplicates using AI.

    Args:
        equivalence_relation: Description of what makes items equivalent
        session: Optional session. If not provided, one will be created automatically.
        input: The input table (DataFrame, UUID, or TableResult)
        webhook_url: Optional URL to receive a POST callback when the task completes or fails.

    Returns:
        TableResult containing the deduped table
    """
    if input is None:
        raise EveryrowError("input is required for dedupe")
    if session is None:
        async with create_session() as internal_session:
            cohort_task = await dedupe_async(
                session=internal_session,
                input=input,
                equivalence_relation=equivalence_relation,
                webhook_url=webhook_url,
            )
            result = await cohort_task.await_result()
            if isinstance(result, TableResult):
                return result
            raise EveryrowError("Dedupe task did not return a table result")
    cohort_task = await dedupe_async(
        session=session,
        input=input,
        equivalence_relation=equivalence_relation,
        webhook_url=webhook_url,
    )
    result = await cohort_task.await_result()
    if isinstance(result, TableResult):
        return result
    raise EveryrowError("Dedupe task did not return a table result")


async def dedupe_async(
    session: Session,
    input: DataFrame | UUID | TableResult,
    equivalence_relation: str,
    webhook_url: str | None = None,
) -> EveryrowTask[BaseModel]:
    """Submit a dedupe task asynchronously."""
    input_data = _prepare_table_input(input, DedupeOperationInputType1Item)

    body = DedupeOperation(
        input_=input_data,  # type: ignore
        equivalence_relation=equivalence_relation,
        session_id=session.session_id,
    )
    if webhook_url is not None:
        body["webhook_url"] = webhook_url

    response = await dedupe_operations_dedupe_post.asyncio(client=session.client, body=body)
    response = handle_response(response)

    cohort_task = EveryrowTask(response_model=BaseModel, is_map=True, is_expand=False)
    cohort_task.set_submitted(response.task_id, response.session_id, session.client)
    return cohort_task
