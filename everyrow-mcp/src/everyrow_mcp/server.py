"""MCP server for everyrow SDK operations."""

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import pandas as pd
from everyrow.api_utils import create_client
from everyrow.generated.api.billing.get_billing_balance_billing_get import (
    asyncio as get_billing,
)
from everyrow.generated.api.tasks import (
    get_task_result_tasks_task_id_result_get,
    get_task_status_tasks_task_id_status_get,
)
from everyrow.generated.client import AuthenticatedClient
from everyrow.generated.models.task_result_response import TaskResultResponse
from everyrow.generated.models.task_status_response import TaskStatusResponse
from everyrow.generated.types import Unset
from everyrow.ops import (
    agent_map_async,
    dedupe_async,
    merge_async,
    rank_async,
    screen_async,
)
from everyrow.session import create_session
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from everyrow_mcp.utils import (
    save_result_to_csv,
    validate_csv_output_path,
    validate_csv_path,
)

# Singleton client, initialized in lifespan
_client: AuthenticatedClient | None = None


@asynccontextmanager
async def lifespan(_server: FastMCP):
    """Initialize singleton client and validate credentials on startup."""
    global _client  # noqa: PLW0603
    try:
        _client = create_client()
        await _client.__aenter__()
        response = await get_billing(client=_client)
        if response is None:
            raise RuntimeError("Failed to authenticate with everyrow API")
    except Exception as e:
        logging.getLogger(__name__).error(f"everyrow-mcp startup failed: {e!r}")
        raise

    yield

    # Cleanup on shutdown
    if _client is not None:
        await _client.__aexit__(None, None, None)
        _client = None


mcp = FastMCP("everyrow_mcp", lifespan=lifespan)

PROGRESS_POLL_DELAY = 12  # seconds to block in everyrow_progress before returning
TASK_STATE_FILE = Path.home() / ".everyrow" / "task.json"


def _get_session_url(session_id: UUID) -> str:
    """Derive session URL from session ID."""
    base_url = os.environ.get("EVERYROW_BASE_URL", "https://everyrow.io")
    return f"{base_url}/sessions/{session_id}"


def _write_task_state(
    task_id: str,
    session_url: str,
    total: int,
    completed: int,
    failed: int,
    running: int,
    status: str,
    started_at: float,
) -> None:
    """Write task tracking state for hooks/status line to read.

    Note: Only one task is tracked at a time. If multiple tasks run concurrently,
    only the most recent one's progress is shown.
    """
    try:
        TASK_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "task_id": task_id,
            "session_url": session_url,
            "total": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "status": status,
            "started_at": started_at,
        }
        with open(TASK_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass  # Non-critical — hooks/status line just won't update


class AgentSubmitInput(BaseModel):
    """Input for submitting an agent_map operation (non-blocking)."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ..., description="Natural language task to perform on each row.", min_length=1
    )
    input_csv: str = Field(..., description="Absolute path to the input CSV file.")
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON schema for the agent's response per row.",
    )

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv(cls, v: str) -> str:
        validate_csv_path(v)
        return v


class RankSubmitInput(BaseModel):
    """Input for submitting a rank operation (non-blocking)."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ..., description="Natural language ranking criteria.", min_length=1
    )
    input_csv: str = Field(..., description="Absolute path to the input CSV file.")
    field_name: str = Field(..., description="Name of the field to sort by.")
    field_type: str = Field(
        default="float", description="Type: 'float', 'int', 'str', or 'bool'"
    )
    ascending_order: bool = Field(
        default=True, description="Sort ascending (True) or descending (False)."
    )
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON schema for the response model.",
    )

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv(cls, v: str) -> str:
        validate_csv_path(v)
        return v

    @field_validator("field_type")
    @classmethod
    def validate_field_type(cls, v: str) -> str:
        valid_types = {"float", "int", "str", "bool"}
        if v not in valid_types:
            raise ValueError(f"field_type must be one of {valid_types}")
        return v


class ScreenSubmitInput(BaseModel):
    """Input for submitting a screen operation (non-blocking)."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ..., description="Natural language screening criteria.", min_length=1
    )
    input_csv: str = Field(..., description="Absolute path to the input CSV file.")
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON schema for the response model.",
    )

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv(cls, v: str) -> str:
        validate_csv_path(v)
        return v


class DedupeSubmitInput(BaseModel):
    """Input for submitting a dedupe operation (non-blocking)."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    equivalence_relation: str = Field(
        ...,
        description="Natural language description of what makes two rows equivalent.",
        min_length=1,
    )
    input_csv: str = Field(..., description="Absolute path to the input CSV file.")

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv(cls, v: str) -> str:
        validate_csv_path(v)
        return v


class MergeSubmitInput(BaseModel):
    """Input for submitting a merge operation (non-blocking)."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ...,
        description="Natural language description of how to match rows.",
        min_length=1,
    )
    left_csv: str = Field(..., description="Absolute path to the left/primary CSV.")
    right_csv: str = Field(..., description="Absolute path to the right/secondary CSV.")
    merge_on_left: str | None = Field(
        default=None, description="Optional column name in left table for merge key."
    )
    merge_on_right: str | None = Field(
        default=None, description="Optional column name in right table for merge key."
    )
    use_web_search: Literal["auto", "yes", "no"] | None = Field(
        default=None, description='Control web search: "auto", "yes", or "no".'
    )

    @field_validator("left_csv", "right_csv")
    @classmethod
    def validate_csv_paths(cls, v: str) -> str:
        validate_csv_path(v)
        return v


class ProgressInput(BaseModel):
    """Input for checking task progress."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task_id: str = Field(..., description="The task ID returned by a _submit tool.")


class ResultsInput(BaseModel):
    """Input for retrieving completed task results."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task_id: str = Field(..., description="The task ID of the completed task.")
    output_path: str = Field(
        ...,
        description="Full absolute path to the output CSV file (must end in .csv).",
    )

    @field_validator("output_path")
    @classmethod
    def validate_output(cls, v: str) -> str:
        validate_csv_output_path(v)
        return v


@mcp.tool(name="everyrow_agent_submit", structured_output=False)
async def everyrow_agent_submit(params: AgentSubmitInput) -> list[TextContent]:
    """Submit a web research agent task on each row of a CSV (returns immediately).

    Use this instead of everyrow_agent for long-running operations. Returns a task_id
    and session_url immediately. Then call everyrow_progress(task_id) to monitor.

    After receiving a result from this tool, share the session_url with the user,
    then immediately call everyrow_progress with the returned task_id.
    """
    if _client is None:
        return [TextContent(type="text", text="Error: MCP server not initialized.")]

    df = pd.read_csv(params.input_csv)

    response_model: type[BaseModel] | None = None
    if params.response_schema:
        response_model = _schema_to_model("AgentResult", params.response_schema)

    async with create_session(client=_client) as session:
        session_url = session.get_url()
        kwargs: dict[str, Any] = {"task": params.task, "session": session, "input": df}
        if response_model:
            kwargs["response_model"] = response_model
        cohort_task = await agent_map_async(**kwargs)
        task_id = str(cohort_task.task_id)

    return [
        TextContent(
            type="text",
            text=(
                f"Submitted: {len(df)} agents starting.\n"
                f"Session: {session_url}\n"
                f"Task ID: {task_id}\n\n"
                f"Share the session_url with the user, then immediately call everyrow_progress(task_id='{task_id}')."
            ),
        )
    ]


@mcp.tool(name="everyrow_rank_submit", structured_output=False)
async def everyrow_rank_submit(params: RankSubmitInput) -> list[TextContent]:
    """Submit a rank/score operation on a CSV (returns immediately).

    Use this instead of everyrow_rank for long-running operations. Returns a task_id
    and session_url immediately. Then call everyrow_progress(task_id) to monitor.

    After receiving a result from this tool, share the session_url with the user,
    then immediately call everyrow_progress with the returned task_id.
    """
    if _client is None:
        return [TextContent(type="text", text="Error: MCP server not initialized.")]

    df = pd.read_csv(params.input_csv)

    response_model: type[BaseModel] | None = None
    if params.response_schema:
        response_model = _schema_to_model("RankResult", params.response_schema)

    async with create_session(client=_client) as session:
        session_url = session.get_url()
        cohort_task = await rank_async(
            task=params.task,
            session=session,
            input=df,
            field_name=params.field_name,
            field_type=params.field_type,  # type: ignore
            response_model=response_model,
            ascending_order=params.ascending_order,
        )
        task_id = str(cohort_task.task_id)

    return [
        TextContent(
            type="text",
            text=(
                f"Submitted: {len(df)} rows for ranking.\n"
                f"Session: {session_url}\n"
                f"Task ID: {task_id}\n\n"
                f"Share the session_url with the user, then immediately call everyrow_progress(task_id='{task_id}')."
            ),
        )
    ]


@mcp.tool(name="everyrow_screen_submit", structured_output=False)
async def everyrow_screen_submit(params: ScreenSubmitInput) -> list[TextContent]:
    """Submit a screen/filter operation on a CSV (returns immediately).

    Use this instead of everyrow_screen for long-running operations. Returns a task_id
    and session_url immediately. Then call everyrow_progress(task_id) to monitor.

    After receiving a result from this tool, share the session_url with the user,
    then immediately call everyrow_progress with the returned task_id.
    """
    if _client is None:
        return [TextContent(type="text", text="Error: MCP server not initialized.")]

    df = pd.read_csv(params.input_csv)

    response_model: type[BaseModel] | None = None
    if params.response_schema:
        response_model = _schema_to_model("ScreenResult", params.response_schema)

    async with create_session(client=_client) as session:
        session_url = session.get_url()
        cohort_task = await screen_async(
            task=params.task,
            session=session,
            input=df,
            response_model=response_model,
        )
        task_id = str(cohort_task.task_id)

    return [
        TextContent(
            type="text",
            text=(
                f"Submitted: {len(df)} rows for screening.\n"
                f"Session: {session_url}\n"
                f"Task ID: {task_id}\n\n"
                f"Share the session_url with the user, then immediately call everyrow_progress(task_id='{task_id}')."
            ),
        )
    ]


@mcp.tool(name="everyrow_dedupe_submit", structured_output=False)
async def everyrow_dedupe_submit(params: DedupeSubmitInput) -> list[TextContent]:
    """Submit a dedupe operation on a CSV (returns immediately).

    Use this instead of everyrow_dedupe for long-running operations. Returns a task_id
    and session_url immediately. Then call everyrow_progress(task_id) to monitor.

    After receiving a result from this tool, share the session_url with the user,
    then immediately call everyrow_progress with the returned task_id.
    """
    if _client is None:
        return [TextContent(type="text", text="Error: MCP server not initialized.")]

    df = pd.read_csv(params.input_csv)

    async with create_session(client=_client) as session:
        session_url = session.get_url()
        cohort_task = await dedupe_async(
            equivalence_relation=params.equivalence_relation,
            session=session,
            input=df,
        )
        task_id = str(cohort_task.task_id)

    return [
        TextContent(
            type="text",
            text=(
                f"Submitted: {len(df)} rows for deduplication.\n"
                f"Session: {session_url}\n"
                f"Task ID: {task_id}\n\n"
                f"Share the session_url with the user, then immediately call everyrow_progress(task_id='{task_id}')."
            ),
        )
    ]


@mcp.tool(name="everyrow_merge_submit", structured_output=False)
async def everyrow_merge_submit(params: MergeSubmitInput) -> list[TextContent]:
    """Submit a merge operation on two CSVs (returns immediately).

    Use this instead of everyrow_merge for long-running operations. Returns a task_id
    and session_url immediately. Then call everyrow_progress(task_id) to monitor.

    After receiving a result from this tool, share the session_url with the user,
    then immediately call everyrow_progress with the returned task_id.
    """
    if _client is None:
        return [TextContent(type="text", text="Error: MCP server not initialized.")]

    left_df = pd.read_csv(params.left_csv)
    right_df = pd.read_csv(params.right_csv)

    async with create_session(client=_client) as session:
        session_url = session.get_url()
        cohort_task = await merge_async(
            task=params.task,
            session=session,
            left_table=left_df,
            right_table=right_df,
            merge_on_left=params.merge_on_left,
            merge_on_right=params.merge_on_right,
            use_web_search=params.use_web_search,
        )
        task_id = str(cohort_task.task_id)

    return [
        TextContent(
            type="text",
            text=(
                f"Submitted: {len(left_df)} left rows for merging.\n"
                f"Session: {session_url}\n"
                f"Task ID: {task_id}\n\n"
                f"Share the session_url with the user, then immediately call everyrow_progress(task_id='{task_id}')."
            ),
        )
    ]


@mcp.tool(name="everyrow_progress", structured_output=False)
async def everyrow_progress(params: ProgressInput) -> list[TextContent]:
    """Check progress of a running everyrow task. Blocks ~12s before returning.

    After receiving a status update, immediately call everyrow_progress again
    unless the task is completed or failed. The tool handles pacing internally.
    Do not add commentary between progress calls — just call again immediately.
    """
    if _client is None:
        return [TextContent(type="text", text="Error: MCP server not initialized.")]

    task_id = params.task_id

    # Block server-side before polling — controls the cadence
    await asyncio.sleep(PROGRESS_POLL_DELAY)

    try:
        status_response = await get_task_status_tasks_task_id_status_get.asyncio(
            task_id=UUID(task_id),
            client=_client,
        )
        if status_response is None:
            raise RuntimeError("No response from status endpoint")
        if not isinstance(status_response, TaskStatusResponse):
            raise RuntimeError(f"Unexpected response type: {type(status_response)}")
    except Exception as e:
        return [
            TextContent(
                type="text",
                text=f"Error polling task: {e!r}\nRetry: call everyrow_progress(task_id='{task_id}').",
            )
        ]

    status_str = status_response.status.value
    progress = status_response.progress
    session_url = _get_session_url(status_response.session_id)

    completed = progress.completed if progress else 0
    failed = progress.failed if progress else 0
    running = progress.running if progress else 0
    total = progress.total if progress else 0

    # Calculate elapsed time from API's created_at timestamp
    if status_response.created_at:
        now = datetime.now(UTC)
        created_at = status_response.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        elapsed_s = round((now - created_at).total_seconds())
        started_at = created_at.timestamp()
    else:
        elapsed_s = 0
        started_at = time.time()

    _write_task_state(
        task_id,
        session_url,
        total,
        completed,
        failed,
        running,
        status_str,
        started_at,
    )

    if status_str in ("completed", "failed", "revoked"):
        error = status_response.error
        if error and not isinstance(error, Unset):
            return [TextContent(type="text", text=f"Task {status_str}: {error}")]
        if status_str == "completed":
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Completed: {completed}/{total} ({failed} failed) in {elapsed_s}s.\n"
                        f"Call everyrow_results(task_id='{task_id}', output_path='/path/to/output.csv') to save the output."
                    ),
                )
            ]
        return [
            TextContent(
                type="text", text=f"Task {status_str}. Report the error to the user."
            )
        ]

    fail_part = f", {failed} failed" if failed else ""
    return [
        TextContent(
            type="text",
            text=(
                f"Running: {completed}/{total} complete, {running} running{fail_part} ({elapsed_s}s elapsed)\n"
                f"Immediately call everyrow_progress(task_id='{task_id}')."
            ),
        )
    ]


@mcp.tool(name="everyrow_results", structured_output=False)
async def everyrow_results(params: ResultsInput) -> list[TextContent]:  # noqa: PLR0911
    """Retrieve results from a completed everyrow task and save to CSV.

    Only call this after everyrow_progress reports status 'completed'.
    The output_path must be a full file path ending in .csv.
    """
    if _client is None:
        return [TextContent(type="text", text="Error: MCP server not initialized.")]

    task_id = params.task_id
    output_file = Path(params.output_path)

    try:
        status_response = await get_task_status_tasks_task_id_status_get.asyncio(
            task_id=UUID(task_id),
            client=_client,
        )
        if status_response is None:
            return [
                TextContent(
                    type="text", text="Error: No response from status endpoint."
                )
            ]
        if not isinstance(status_response, TaskStatusResponse):
            return [
                TextContent(
                    type="text",
                    text=f"Error: Unexpected response type: {type(status_response)}",
                )
            ]
        status_str = status_response.status.value
        if status_str not in ("completed", "failed", "revoked"):
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Task status is {status_str}. Cannot fetch results yet.\n"
                        f"Call everyrow_progress(task_id='{task_id}') to check again."
                    ),
                )
            ]
    except Exception as e:
        return [TextContent(type="text", text=f"Error checking task status: {e!r}")]

    try:
        result_response = await get_task_result_tasks_task_id_result_get.asyncio(
            task_id=UUID(task_id),
            client=_client,
        )
        if result_response is None:
            return [
                TextContent(
                    type="text", text="Error: No response from result endpoint."
                )
            ]
        if not isinstance(result_response, TaskResultResponse):
            return [
                TextContent(
                    type="text",
                    text=f"Error: Unexpected response type: {type(result_response)}",
                )
            ]

        if isinstance(result_response.data, list):
            records = [item.additional_properties for item in result_response.data]
            df = pd.DataFrame(records)
        else:
            return [
                TextContent(type="text", text="Error: Task result has no table data.")
            ]

        save_result_to_csv(df, output_file)
        # Task state file deleted by PostToolUse hook (everyrow-track-results.sh)

        return [
            TextContent(
                type="text",
                text=(
                    f"Saved {len(df)} rows to {output_file}\n\n"
                    "Tip: For multi-step pipelines, custom response models, or preview mode, "
                    "ask Claude to write Python using the everyrow SDK."
                ),
            )
        ]

    except Exception as e:
        return [TextContent(type="text", text=f"Error retrieving results: {e!r}")]


JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_to_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Convert a JSON schema dict to a dynamic Pydantic model.

    This allows the MCP client to pass arbitrary response schemas without
    needing to define Python classes.
    """
    properties = schema.get("properties", schema)
    required = set(schema.get("required", []))

    fields: dict[str, Any] = {}
    for field_name, field_def in properties.items():
        if field_name.startswith("_") or not isinstance(field_def, dict):
            continue

        field_type_str = field_def.get("type", "string")
        python_type = JSON_TYPE_MAP.get(field_type_str, str)
        description = field_def.get("description", "")

        if field_name in required:
            fields[field_name] = (python_type, Field(..., description=description))
        else:
            fields[field_name] = (
                python_type | None,
                Field(default=None, description=description),
            )

    return create_model(name, **fields)


def main():
    """Run the MCP server."""
    # Signal to the SDK that we're inside the MCP server (suppresses plugin hints)
    os.environ["EVERYROW_MCP_SERVER"] = "1"

    # Configure logging to use stderr only (stdout is reserved for JSON-RPC)
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s: %(message)s",
        force=True,
    )

    # Check for API key before starting
    if "EVERYROW_API_KEY" not in os.environ:
        logging.error("EVERYROW_API_KEY environment variable is not set.")
        logging.error("Get an API key at https://everyrow.io/api-key")
        sys.exit(1)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
