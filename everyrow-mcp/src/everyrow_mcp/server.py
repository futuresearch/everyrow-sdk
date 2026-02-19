"""MCP server for everyrow SDK operations."""

import argparse
import asyncio
import json
import logging
import os
import secrets
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import pandas as pd
from everyrow.api_utils import create_client as _create_sdk_client
from everyrow.api_utils import handle_response
from everyrow.generated.api.billing.get_billing_balance_billing_get import (
    asyncio as get_billing,
)
from everyrow.generated.api.tasks import (
    get_task_result_tasks_task_id_result_get,
    get_task_status_tasks_task_id_status_get,
)
from everyrow.generated.client import AuthenticatedClient
from everyrow.generated.models.public_task_type import PublicTaskType
from everyrow.generated.models.task_result_response_data_type_1 import (
    TaskResultResponseDataType1,
)
from everyrow.generated.models.task_status import TaskStatus
from everyrow.generated.types import Unset
from everyrow.ops import (
    agent_map_async,
    dedupe_async,
    merge_async,
    rank_async,
    screen_async,
    single_agent_async,
)
from everyrow.session import create_session, get_session_url
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import ProviderTokenVerifier
from mcp.server.auth.settings import (
    AuthSettings,
    ClientRegistrationOptions,
    RevocationOptions,
)
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import lifespan_wrapper
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import TextContent, ToolAnnotations
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
    model_validator,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from everyrow_mcp.auth import EveryRowAccessToken, EveryRowAuthProvider
from everyrow_mcp.gcs_storage import GCSResultStore
from everyrow_mcp.redis_utils import build_key, create_redis_client
from everyrow_mcp.settings import HttpSettings, StdioSettings
from everyrow_mcp.templates import (
    PROGRESS_HTML,
    RESULTS_HTML,
    SESSION_HTML,
    UI_CSP_META,
)
from everyrow_mcp.utils import (
    load_csv,
    save_result_to_csv,
    validate_csv_output_path,
    validate_csv_path,
)

PROGRESS_POLL_DELAY = 12  # seconds to block in everyrow_progress before returning
TASK_STATE_FILE = Path.home() / ".everyrow" / "task.json"

# Singleton client, initialized in lifespan (stdio mode only)
_client: AuthenticatedClient | None = None
# Transport mode: "stdio" or "streamable-http". Set in main().
_transport: str = "stdio"
# Auth provider (HTTP mode only), set in main().
_auth_provider: EveryRowAuthProvider | None = None
# Public URL of this server (HTTP mode only), set in main().
_mcp_server_url: str = ""
# Maps task_id -> JWT so the progress REST endpoint can call the EveryRow API.
_task_tokens: dict[str, str] = {}
# Maps task_id -> poll_token for authenticating progress/results endpoint callers.
_task_poll_tokens: dict[str, str] = {}
# Cached results for pagination: task_id -> (DataFrame, timestamp, download_token).
_result_cache: dict[str, tuple[pd.DataFrame, float, str]] = {}
RESULT_CACHE_TTL = 600  # 10 minutes
PREVIEW_SIZE = 5
# GCS result store (HTTP mode only), set in _configure_http_mode if GCS_RESULTS_BUCKET is set.
_gcs_store: Any = None
# Parsed settings (HttpSettings or StdioSettings), set in main()/_configure_http_mode.
_settings: HttpSettings | StdioSettings | None = None


def _get_client() -> AuthenticatedClient:
    """Get an EveryRow API client for the current request.

    In stdio mode, returns the singleton client initialized at startup.
    In HTTP mode, creates a per-request client using the authenticated
    user's API key from the OAuth access token.
    """
    if _transport == "stdio":
        if _client is None:
            raise RuntimeError("MCP server not initialized")
        return _client
    # HTTP mode: get JWT from authenticated request
    access_token = get_access_token()
    if not isinstance(access_token, EveryRowAccessToken):
        raise RuntimeError("Not authenticated")
    if _settings is None:
        raise RuntimeError("MCP server not initialized")
    return AuthenticatedClient(
        base_url=_settings.everyrow_api_url,
        token=access_token.supabase_jwt,
        raise_on_unexpected_status=True,
        follow_redirects=True,
    )


@asynccontextmanager
async def _stdio_lifespan(_server: FastMCP):
    """Initialize singleton client and validate credentials on startup (stdio mode)."""
    global _client  # noqa: PLW0603

    _clear_task_state()

    try:
        with _create_sdk_client() as _client:
            response = await get_billing(client=_client)
            if response is None:
                raise RuntimeError("Failed to authenticate with everyrow API")
            yield
    except Exception as e:
        logging.getLogger(__name__).error(f"everyrow-mcp startup failed: {e!r}")
        raise
    finally:
        _client = None
        _clear_task_state()


@asynccontextmanager
async def _http_lifespan(_server: FastMCP):
    """HTTP mode lifespan — verify Redis connectivity on startup."""
    if _auth_provider is not None:
        await _auth_provider._redis.ping()
        logging.getLogger(__name__).info("Redis health check passed")
    yield


mcp = FastMCP("everyrow_mcp", lifespan=_stdio_lifespan)


@mcp.resource(
    "ui://everyrow/progress.html",
    mime_type="text/html;profile=mcp-app",
    meta=UI_CSP_META,
)
def _progress_ui() -> str:
    return PROGRESS_HTML


@mcp.resource(
    "ui://everyrow/results.html",
    mime_type="text/html;profile=mcp-app",
    meta=UI_CSP_META,
)
def _results_ui() -> str:
    return RESULTS_HTML


@mcp.resource(
    "ui://everyrow/session.html",
    mime_type="text/html;profile=mcp-app",
    meta=UI_CSP_META,
)
def _session_ui() -> str:
    return SESSION_HTML


def _submission_text(label: str, session_url: str, task_id: str) -> str:
    """Build human-readable text for submission tool results."""
    if _transport == "stdio":
        return (
            f"{label}\n"
            f"Session: {session_url}\n"
            f"Task ID: {task_id}\n\n"
            f"Share the session_url with the user, then immediately call everyrow_progress(task_id='{task_id}')."
        )
    return (
        f"{label}\n"
        f"Task ID: {task_id}\n\n"
        f"Immediately call everyrow_progress(task_id='{task_id}')."
    )


def _submission_ui_json(
    session_url: str,
    task_id: str,
    total: int,
    token: str,
) -> str:
    """Build JSON for the session MCP App widget, and store the token for polling."""
    _task_tokens[task_id] = token
    poll_token = secrets.token_urlsafe(32)
    _task_poll_tokens[task_id] = poll_token
    data: dict[str, Any] = {
        "session_url": session_url,
        "task_id": task_id,
        "total": total,
        "status": "submitted",
    }
    if _mcp_server_url:
        data["progress_url"] = (
            f"{_mcp_server_url}/api/progress/{task_id}?token={poll_token}"
        )
    return json.dumps(data)


def _clear_task_state() -> None:
    if _transport != "stdio":
        return
    if TASK_STATE_FILE.exists():
        TASK_STATE_FILE.unlink()


def _write_task_state(
    task_id: str,
    task_type: PublicTaskType,
    session_url: str,
    total: int,
    completed: int,
    failed: int,
    running: int,
    status: TaskStatus,
    started_at: datetime,
) -> None:
    """Write task tracking state for hooks/status line to read.

    Note: Only one task is tracked at a time. If multiple tasks run concurrently,
    only the most recent one's progress is shown.
    """
    if _transport != "stdio":
        return
    try:
        TASK_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "task_id": task_id,
            "task_type": task_type.value,
            "session_url": session_url,
            "total": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "status": status.value,
            "started_at": started_at.timestamp(),
        }
        with open(TASK_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        logging.getLogger(__name__).debug(f"Failed to write task state: {e!r}")


class AgentInput(BaseModel):
    """Input for the agent operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ..., description="Natural language task to perform on each row.", min_length=1
    )
    input_csv: str | None = Field(
        default=None, description="Absolute path to the input CSV file."
    )
    input_data: str | None = Field(
        default=None,
        description="Raw CSV content as a string (alternative to input_csv for remote use).",
    )
    input_json: list[dict[str, Any]] | None = Field(
        default=None,
        description="Data as a JSON array of objects. "
        'Example: [{"company": "Acme", "url": "acme.com"}, {"company": "Beta", "url": "beta.io"}]',
    )
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON schema for the agent's response per row.",
    )

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv(cls, v: str | None) -> str | None:
        if v is not None:
            validate_csv_path(v)
        return v

    @model_validator(mode="after")
    def check_input_source(self) -> "AgentInput":
        sources = sum(
            1 for s in (self.input_csv, self.input_data, self.input_json) if s
        )
        if sources == 0:
            raise ValueError("Provide one of input_csv, input_data, or input_json.")
        if sources > 1:
            raise ValueError(
                "Provide only one of input_csv, input_data, or input_json."
            )
        return self


class SingleAgentInput(BaseModel):
    """Input for the single agent operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ...,
        description="Natural language task for the agent to perform.",
        min_length=1,
    )
    input_data: dict[str, Any] | None = Field(
        default=None,
        description="Optional context as key-value pairs (e.g. {'company': 'Acme', 'url': 'acme.com'}).",
    )
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON schema for the agent's response.",
    )


class RankInput(BaseModel):
    """Input for the rank operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ...,
        description="Natural language instructions for scoring a single row.",
        min_length=1,
    )
    input_csv: str | None = Field(
        default=None, description="Absolute path to the input CSV file."
    )
    input_data: str | None = Field(
        default=None,
        description="Raw CSV content as a string (alternative to input_csv for remote use).",
    )
    input_json: list[dict[str, Any]] | None = Field(
        default=None,
        description="Data as a JSON array of objects. "
        'Example: [{"company": "Acme", "url": "acme.com"}, {"company": "Beta", "url": "beta.io"}]',
    )
    field_name: str = Field(..., description="Name of the field to sort by.")
    field_type: Literal["float", "int", "str", "bool"] = Field(
        default="float",
        description="Type of the score field: 'float', 'int', 'str', or 'bool'",
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
    def validate_input_csv(cls, v: str | None) -> str | None:
        if v is not None:
            validate_csv_path(v)
        return v

    @model_validator(mode="after")
    def check_input_source(self) -> "RankInput":
        sources = sum(
            1 for s in (self.input_csv, self.input_data, self.input_json) if s
        )
        if sources == 0:
            raise ValueError("Provide one of input_csv, input_data, or input_json.")
        if sources > 1:
            raise ValueError(
                "Provide only one of input_csv, input_data, or input_json."
            )
        return self


class ScreenInput(BaseModel):
    """Input for the screen operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ..., description="Natural language screening criteria.", min_length=1
    )
    input_csv: str | None = Field(
        default=None, description="Absolute path to the input CSV file."
    )
    input_data: str | None = Field(
        default=None,
        description="Raw CSV content as a string (alternative to input_csv for remote use).",
    )
    input_json: list[dict[str, Any]] | None = Field(
        default=None,
        description="Data as a JSON array of objects. "
        'Example: [{"company": "Acme", "url": "acme.com"}, {"company": "Beta", "url": "beta.io"}]',
    )
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON schema for the response model. "
        "Must include at least one boolean property — screen uses the boolean field to filter rows into pass/fail.",
    )

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv(cls, v: str | None) -> str | None:
        if v is not None:
            validate_csv_path(v)
        return v

    @model_validator(mode="after")
    def check_input_source(self) -> "ScreenInput":
        sources = sum(
            1 for s in (self.input_csv, self.input_data, self.input_json) if s
        )
        if sources == 0:
            raise ValueError("Provide one of input_csv, input_data, or input_json.")
        if sources > 1:
            raise ValueError(
                "Provide only one of input_csv, input_data, or input_json."
            )
        return self


class DedupeInput(BaseModel):
    """Input for the dedupe operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    equivalence_relation: str = Field(
        ...,
        description="Natural language description of what makes two rows equivalent/duplicates. "
        "The LLM will use this to identify which rows represent the same entity.",
        min_length=1,
    )
    input_csv: str = Field(..., description="Absolute path to the input CSV file.")

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv(cls, v: str) -> str:
        validate_csv_path(v)
        return v


class MergeInput(BaseModel):
    """Input for the merge operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ...,
        description="Natural language description of how to match rows.",
        min_length=1,
    )
    left_csv: str = Field(
        ...,
        description="Absolute path to the left CSV. Works like a LEFT JOIN: ALL rows from this table are kept in the output. This should be the table being enriched.",
    )
    right_csv: str = Field(
        ...,
        description="Absolute path to the right CSV. This is the lookup/reference table. Its columns are added to matching left rows; unmatched left rows get nulls.",
    )
    merge_on_left: str | None = Field(
        default=None,
        description="Only set if you expect some exact string matches on the chosen column or want to draw special attention of LLM agents to this particular column. Fine to leave unspecified in all other cases.",
    )
    merge_on_right: str | None = Field(
        default=None,
        description="Only set if you expect some exact string matches on the chosen column or want to draw special attention of LLM agents to this particular column. Fine to leave unspecified in all other cases.",
    )
    use_web_search: Literal["auto", "yes", "no"] | None = Field(
        default=None, description='Control web search: "auto", "yes", or "no".'
    )
    relationship_type: Literal["many_to_one", "one_to_one"] | None = Field(
        default=None,
        description="Leave unset for the default many_to_one, which is correct in most cases. many_to_one: multiple left rows can match one right row (e.g. products → companies). one_to_one: each left row matches at most one right row AND vice versa. Only use one_to_one when both tables represent unique entities of the same kind.",
    )

    @field_validator("left_csv", "right_csv")
    @classmethod
    def validate_csv_paths(cls, v: str) -> str:
        validate_csv_path(v)
        return v


class ProgressInput(BaseModel):
    """Input for checking task progress."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task_id: str = Field(..., description="The task ID returned by the operation tool.")


class ResultsInput(BaseModel):
    """Input for retrieving completed task results."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task_id: str = Field(..., description="The task ID of the completed task.")
    output_path: str | None = Field(
        default=None,
        description="Full absolute path to the output CSV file (must end in .csv). "
        "If omitted, results are returned inline.",
    )
    offset: int = Field(
        default=0,
        description="Row offset for pagination. Default 0 returns the first page.",
        ge=0,
    )
    page_size: int = Field(
        default=PREVIEW_SIZE,
        description="Number of rows per page. Default 5. Max 50.",
        ge=1,
        le=50,
    )

    @field_validator("output_path")
    @classmethod
    def validate_output(cls, v: str | None) -> str | None:
        if v is not None and _transport == "stdio":
            validate_csv_output_path(v)
        return v


@mcp.tool(
    name="everyrow_agent",
    structured_output=False,
    annotations=ToolAnnotations(
        title="Run Web Research Agents",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def everyrow_agent(params: AgentInput) -> list[TextContent]:
    """Run web research agents on each row of a CSV file.

    The dispatched agents will search the web, read pages, and return the
    requested research fields for each row. Agents run in parallel to save
    time and are optimized to find accurate answers at minimum cost.

    Examples:
    - "Find this company's latest funding round and lead investors"
    - "Research the CEO's background and previous companies"
    - "Find pricing information for this product"

    This function submits the task and returns immediately with a task_id and session_url.
    After receiving a result from this tool, share the session_url with the user.
    Then immediately call everyrow_progress(task_id) to monitor.
    Once the task is completed, call everyrow_results to save the output.
    """
    client = _get_client()

    _clear_task_state()
    df = load_csv(
        input_csv=params.input_csv,
        input_data=params.input_data,
        input_json=params.input_json,
    )

    response_model: type[BaseModel] | None = None
    if params.response_schema:
        response_model = _schema_to_model("AgentResult", params.response_schema)

    async with create_session(client=client) as session:
        session_url = session.get_url()
        kwargs: dict[str, Any] = {"task": params.task, "session": session, "input": df}
        if response_model:
            kwargs["response_model"] = response_model
        cohort_task = await agent_map_async(**kwargs)
        task_id = str(cohort_task.task_id)
        _write_task_state(
            task_id,
            task_type=PublicTaskType.AGENT,
            session_url=session_url,
            total=len(df),
            completed=0,
            failed=0,
            running=0,
            status=TaskStatus.RUNNING,
            started_at=datetime.now(UTC),
        )

    return [
        TextContent(
            type="text",
            text=_submission_ui_json(session_url, task_id, len(df), client.token),
        ),
        TextContent(
            type="text",
            text=_submission_text(
                f"Submitted: {len(df)} agents starting.", session_url, task_id
            ),
        ),
    ]


@mcp.tool(
    name="everyrow_single_agent",
    structured_output=False,
    annotations=ToolAnnotations(
        title="Run a Single Research Agent",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def everyrow_single_agent(params: SingleAgentInput) -> list[TextContent]:
    """Run a single web research agent on a task, optionally with context data.

    Unlike everyrow_agent (which processes many rows), this dispatches ONE agent
    to research a single question. The agent can search the web, read pages, and
    return structured results.

    Examples:
    - "Find the current CEO of Apple and their background"
    - "Research the latest funding round for this company" (with input_data: {"company": "Stripe"})
    - "What are the pricing tiers for this product?" (with input_data: {"product": "Snowflake"})

    This function submits the task and returns immediately with a task_id and session_url.
    After receiving a result from this tool, share the session_url with the user.
    Then immediately call everyrow_progress(task_id) to monitor.
    Once the task is completed, call everyrow_results to save the output.
    """
    client = _get_client()

    _clear_task_state()

    response_model: type[BaseModel] | None = None
    if params.response_schema:
        response_model = _schema_to_model("SingleAgentResult", params.response_schema)

    # Convert input_data dict to a BaseModel if provided
    input_model: BaseModel | None = None
    if params.input_data:
        DynamicInput = create_model(
            "DynamicInput", **{k: (type(v), v) for k, v in params.input_data.items()}
        )
        input_model = DynamicInput()

    async with create_session(client=client) as session:
        session_url = session.get_url()
        kwargs: dict[str, Any] = {"task": params.task, "session": session}
        if input_model is not None:
            kwargs["input"] = input_model
        if response_model is not None:
            kwargs["response_model"] = response_model
        cohort_task = await single_agent_async(**kwargs)
        task_id = str(cohort_task.task_id)
        _write_task_state(
            task_id,
            task_type=PublicTaskType.AGENT,
            session_url=session_url,
            total=1,
            completed=0,
            failed=0,
            running=0,
            status=TaskStatus.RUNNING,
            started_at=datetime.now(UTC),
        )

    return [
        TextContent(
            type="text",
            text=_submission_ui_json(session_url, task_id, 1, client.token),
        ),
        TextContent(
            type="text",
            text=_submission_text(
                "Submitted: single agent starting.", session_url, task_id
            ),
        ),
    ]


@mcp.tool(
    name="everyrow_rank",
    structured_output=False,
    annotations=ToolAnnotations(
        title="Score and Rank Rows",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def everyrow_rank(params: RankInput) -> list[TextContent]:
    """Score and sort rows in a CSV file based on any criteria.

    Dispatches web agents to research the criteria to rank the entities in the
    table. Conducts research, and can also apply judgment to the results if the
    criteria are qualitative.

    Examples:
    - "Score this lead from 0 to 10 by likelihood to need data integration solutions"
    - "Score this company out of 100 by AI/ML adoption maturity"
    - "Score this candidate by fit for a senior engineering role, with 100 being the best"

    This function submits the task and returns immediately with a task_id and session_url.
    After receiving a result from this tool, share the session_url with the user.
    Then immediately call everyrow_progress(task_id) to monitor.
    Once the task is completed, call everyrow_results to save the output.

    Args:
        params: RankInput

    Returns:
        Success message containing session_url (for the user to open) and
        task_id (for monitoring progress)
    """
    client = _get_client()

    _clear_task_state()
    df = load_csv(
        input_csv=params.input_csv,
        input_data=params.input_data,
        input_json=params.input_json,
    )

    response_model: type[BaseModel] | None = None
    if params.response_schema:
        response_model = _schema_to_model("RankResult", params.response_schema)

    async with create_session(client=client) as session:
        session_url = session.get_url()
        cohort_task = await rank_async(
            task=params.task,
            session=session,
            input=df,
            field_name=params.field_name,
            field_type=params.field_type,
            response_model=response_model,
            ascending_order=params.ascending_order,
        )
        task_id = str(cohort_task.task_id)
        _write_task_state(
            task_id,
            task_type=PublicTaskType.RANK,
            session_url=session_url,
            total=len(df),
            completed=0,
            failed=0,
            running=0,
            status=TaskStatus.RUNNING,
            started_at=datetime.now(UTC),
        )

    return [
        TextContent(
            type="text",
            text=_submission_ui_json(session_url, task_id, len(df), client.token),
        ),
        TextContent(
            type="text",
            text=_submission_text(
                f"Submitted: {len(df)} rows for ranking.", session_url, task_id
            ),
        ),
    ]


@mcp.tool(
    name="everyrow_screen",
    structured_output=False,
    annotations=ToolAnnotations(
        title="Filter Rows by Criteria",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def everyrow_screen(params: ScreenInput) -> list[TextContent]:
    """Filter rows in a CSV file based on any criteria.

    Dispatches web agents to research the criteria to filter the entities in the
    table. Conducts research, and can also apply judgment to the results if the
    criteria are qualitative.

    Screen produces a boolean pass/fail verdict per row. If you provide a custom
    response_schema, it MUST include at least one boolean property (e.g.
    ``{"passes": {"type": "boolean"}}``). If the screening criteria need more than
    a yes/no answer (e.g. a three-way classification), use everyrow_agent instead.

    Examples:
    - "Is this job posting remote-friendly AND senior-level AND salary disclosed?"
    - "Is this vendor financially stable AND does it have good security practices?"
    - "Is this lead likely to need our product based on company description?"

    This function submits the task and returns immediately with a task_id and session_url.
    After receiving a result from this tool, share the session_url with the user.
    Then immediately call everyrow_progress(task_id) to monitor.
    Once the task is completed, call everyrow_results to save the output.

    Args:
        params: ScreenInput

    Returns:
        Success message containing session_url (for the user to open) and
        task_id (for monitoring progress)
    """
    client = _get_client()

    _clear_task_state()
    df = load_csv(
        input_csv=params.input_csv,
        input_data=params.input_data,
        input_json=params.input_json,
    )

    response_model: type[BaseModel] | None = None
    if params.response_schema:
        response_model = _schema_to_model("ScreenResult", params.response_schema)

    async with create_session(client=client) as session:
        session_url = session.get_url()
        cohort_task = await screen_async(
            task=params.task,
            session=session,
            input=df,
            response_model=response_model,
        )
        task_id = str(cohort_task.task_id)
        _write_task_state(
            task_id,
            task_type=PublicTaskType.SCREEN,
            session_url=session_url,
            total=len(df),
            completed=0,
            failed=0,
            running=0,
            status=TaskStatus.RUNNING,
            started_at=datetime.now(UTC),
        )

    return [
        TextContent(
            type="text",
            text=_submission_ui_json(session_url, task_id, len(df), client.token),
        ),
        TextContent(
            type="text",
            text=_submission_text(
                f"Submitted: {len(df)} rows for screening.", session_url, task_id
            ),
        ),
    ]


@mcp.tool(
    name="everyrow_dedupe",
    structured_output=False,
    annotations=ToolAnnotations(
        title="Deduplicate Rows",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def everyrow_dedupe(params: DedupeInput) -> list[TextContent]:
    """Remove duplicate rows from a CSV file using semantic equivalence.

    Dedupe identifies rows that represent the same entity even when they
    don't match exactly. The duplicate criterion is semantic and LLM-powered:
    agents reason over the data and, when needed, search the web for external
    information to establish equivalence.

    Examples:
    - Dedupe contacts: "Same person even with name abbreviations or career changes"
    - Dedupe companies: "Same company including subsidiaries and name variations"
    - Dedupe research papers: "Same work including preprints and published versions"

    This function submits the task and returns immediately with a task_id and session_url.
    After receiving a result from this tool, share the session_url with the user.
    Then immediately call everyrow_progress(task_id) to monitor.
    Once the task is completed, call everyrow_results to save the output.

    Args:
        params: DedupeInput

    Returns:
        Success message containing session_url (for the user to open) and
        task_id (for monitoring progress)
    """
    client = _get_client()

    _clear_task_state()
    df = pd.read_csv(params.input_csv)

    async with create_session(client=client) as session:
        session_url = session.get_url()
        cohort_task = await dedupe_async(
            equivalence_relation=params.equivalence_relation,
            session=session,
            input=df,
        )
        task_id = str(cohort_task.task_id)
        _write_task_state(
            task_id,
            task_type=PublicTaskType.DEDUPE,
            session_url=session_url,
            total=len(df),
            completed=0,
            failed=0,
            running=0,
            status=TaskStatus.RUNNING,
            started_at=datetime.now(UTC),
        )

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


@mcp.tool(
    name="everyrow_merge",
    structured_output=False,
    annotations=ToolAnnotations(
        title="Merge Two Tables",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def everyrow_merge(params: MergeInput) -> list[TextContent]:
    """Join two CSV files using intelligent entity matching.

    Merge combines two tables even when keys don't match exactly. Uses LLM web
    research and judgment to identify which rows from the first table should
    join those in the second.

    left_csv = the table being enriched (ALL its rows appear in the output).
    right_csv = the lookup/reference table (its columns are appended to matches).

    IMPORTANT defaults — omit parameters when unsure:
    - merge_on_left/merge_on_right: only set if you expect exact string matches on
      the chosen columns or want to draw agent attention to them. Fine to omit.
    - relationship_type: defaults to many_to_one, which is correct in most cases.
      Only set one_to_one when both tables have unique entities of the same kind.

    Examples:
    - Match software products (left, enriched) to parent companies (right, lookup):
      Photoshop -> Adobe. relationship_type: many_to_one (many products per company).
    - Match clinical trial sponsors (left) to pharma companies (right):
      Genentech -> Roche. relationship_type: many_to_one.
    - Join two contact lists with different name formats:
      relationship_type: one_to_one (each person appears once in each list).

    This function submits the task and returns immediately with a task_id and session_url.
    After receiving a result from this tool, share the session_url with the user.
    Then immediately call everyrow_progress(task_id) to monitor.
    Once the task is completed, call everyrow_results to save the output.

    Args:
        params: MergeInput

    Returns:
        Success message containing session_url (for the user to open) and
        task_id (for monitoring progress)
    """
    client = _get_client()

    _clear_task_state()
    left_df = pd.read_csv(params.left_csv)
    right_df = pd.read_csv(params.right_csv)

    async with create_session(client=client) as session:
        session_url = session.get_url()
        cohort_task = await merge_async(
            task=params.task,
            session=session,
            left_table=left_df,
            right_table=right_df,
            merge_on_left=params.merge_on_left,
            merge_on_right=params.merge_on_right,
            use_web_search=params.use_web_search,
            relationship_type=params.relationship_type,
        )
        task_id = str(cohort_task.task_id)
        _write_task_state(
            task_id,
            task_type=PublicTaskType.MERGE,
            session_url=session_url,
            total=len(left_df),
            completed=0,
            failed=0,
            running=0,
            status=TaskStatus.RUNNING,
            started_at=datetime.now(UTC),
        )

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


@mcp.tool(
    name="everyrow_progress",
    structured_output=False,
    annotations=ToolAnnotations(
        title="Check Task Progress",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def everyrow_progress(  # noqa: PLR0912
    params: ProgressInput,
) -> list[TextContent]:
    """Check progress of a running task. Blocks for a time to limit the polling rate.

    After receiving a status update, immediately call everyrow_progress again
    unless the task is completed or failed. The tool handles pacing internally.
    Do not add commentary between progress calls, just call again immediately.
    """
    client = _get_client()

    task_id = params.task_id

    # Block server-side before polling — controls the cadence
    await asyncio.sleep(PROGRESS_POLL_DELAY)

    try:
        status_response = handle_response(
            await get_task_status_tasks_task_id_status_get.asyncio(
                task_id=UUID(task_id),
                client=client,
            )
        )
    except Exception as e:
        return [
            TextContent(type="text", text=json.dumps({"status": "error"})),
            TextContent(
                type="text",
                text=f"Error polling task: {e!r}\nRetry: call everyrow_progress(task_id='{task_id}').",
            ),
        ]

    status = status_response.status
    progress = status_response.progress
    is_terminal = status in (
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.REVOKED,
    )
    is_screen = status_response.task_type == PublicTaskType.SCREEN
    session_url = get_session_url(status_response.session_id)

    completed = progress.completed if progress else 0
    failed = progress.failed if progress else 0
    running = progress.running if progress else 0
    total = progress.total if progress else 0

    # Calculate elapsed time from API timestamps.
    # For terminal states, use updated_at - created_at (actual task duration).
    # For running/pending, use now - created_at (ongoing elapsed time).
    if status_response.created_at:
        created_at = status_response.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        started_at = created_at

        if is_terminal and status_response.updated_at:
            updated_at = status_response.updated_at
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=UTC)
            elapsed_s = round((updated_at - created_at).total_seconds())
        else:
            now = datetime.now(UTC)
            elapsed_s = round((now - created_at).total_seconds())
    else:
        elapsed_s = 0
        started_at = datetime.now(UTC)

    _write_task_state(
        task_id,
        task_type=status_response.task_type,
        session_url=session_url,
        total=total,
        completed=completed,
        failed=failed,
        running=running,
        status=status,
        started_at=started_at,
    )

    # JSON for MCP App progress UI (first TextContent, parsed by progress.html)
    ui_json = TextContent(
        type="text",
        text=json.dumps(
            {
                "status": status.value,
                "completed": completed,
                "total": total,
                "failed": failed,
                "running": running,
                "elapsed_s": elapsed_s,
                "session_url": session_url,
            }
        ),
    )

    if is_terminal:
        error = status_response.error
        if error and not isinstance(error, Unset):
            return [
                ui_json,
                TextContent(type="text", text=f"Task {status.value}: {error}"),
            ]
        if status == TaskStatus.COMPLETED:
            if is_screen:
                completed_msg = f"Screening complete ({elapsed_s}s)."
            else:
                completed_msg = (
                    f"Completed: {completed}/{total} ({failed} failed) in {elapsed_s}s."
                )
            return [
                ui_json,
                TextContent(
                    type="text",
                    text=(
                        f"{completed_msg}\n"
                        f"Call everyrow_results(task_id='{task_id}', output_path='/path/to/output.csv') to save the output."
                    ),
                ),
            ]
        return [
            ui_json,
            TextContent(
                type="text", text=f"Task {status.value}. Report the error to the user."
            ),
        ]

    if is_screen:
        return [
            ui_json,
            TextContent(
                type="text",
                text=(
                    f"Screen running ({elapsed_s}s elapsed).\n"
                    f"Immediately call everyrow_progress(task_id='{task_id}')."
                ),
            ),
        ]

    fail_part = f", {failed} failed" if failed else ""
    return [
        ui_json,
        TextContent(
            type="text",
            text=(
                f"Running: {completed}/{total} complete, {running} running{fail_part} ({elapsed_s}s elapsed)\n"
                f"Immediately call everyrow_progress(task_id='{task_id}')."
            ),
        ),
    ]


@mcp.tool(
    name="everyrow_results",
    structured_output=False,
    annotations=ToolAnnotations(
        title="Save Task Results",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    meta={"ui": {"resourceUri": "ui://everyrow/results.html"}},
)
async def everyrow_results(params: ResultsInput) -> list[TextContent]:  # noqa: PLR0911, PLR0912, PLR0915
    """Retrieve results from a completed everyrow task.

    Only call this after everyrow_progress reports status 'completed'.
    If output_path is provided (stdio mode), saves all results to a CSV file.
    If output_path is omitted, returns a page of results inline.

    Pagination example (100-row result set):
      everyrow_results(task_id='abc')                        → rows 1-5, next offset=5
      everyrow_results(task_id='abc', offset=5)              → rows 6-10, next offset=10
      everyrow_results(task_id='abc', offset=0, page_size=50) → rows 1-50, next offset=50
      everyrow_results(task_id='abc', offset=50, page_size=50) → rows 51-100, final page

    For large result sets a CSV download link is also provided on the first page.
    Share the download link with the user so they can get the full data.
    """
    client = _get_client()
    task_id = params.task_id

    # ── GCS mode: check Redis for cached metadata ──────────────────
    if _gcs_store and _auth_provider is not None:
        cached_meta_raw = await _auth_provider._redis.get(build_key("result", task_id))
        if cached_meta_raw:
            meta = json.loads(cached_meta_raw)
            # Re-generate signed URLs (they expire after 15 min)
            urls = await asyncio.to_thread(_gcs_store.generate_signed_urls, task_id)
            total = meta["total"]
            columns = meta["columns"]
            col_names = ", ".join(columns[:10])
            if len(columns) > 10:
                col_names += f", ... (+{len(columns) - 10} more)"

            # Build page from preview stored in metadata
            page_records = meta.get("preview", [])
            offset = params.offset
            page_size = params.page_size

            widget_json = json.dumps(
                {
                    "results_url": urls.json_url,
                    "preview": page_records,
                    "total": total,
                }
            )

            has_more = offset + page_size < total
            next_offset = offset + page_size if has_more else None

            if has_more:
                page_size_arg = (
                    f", page_size={page_size}" if page_size != PREVIEW_SIZE else ""
                )
                summary = (
                    f"Results: {total} rows, {len(columns)} columns ({col_names}). "
                    f"Showing rows {offset + 1}-{min(offset + page_size, total)} of {total}.\n"
                    f"Call everyrow_results(task_id='{task_id}', offset={next_offset}{page_size_arg}) for the next page."
                )
                if offset == 0:
                    summary += f"\nFull CSV download: {urls.csv_url}\nShare this download link with the user."
            elif offset == 0:
                summary = f"Results: {total} rows, {len(columns)} columns ({col_names}). All rows shown."
            else:
                summary = f"Results: showing rows {offset + 1}-{min(offset + page_size, total)} of {total} (final page)."

            return [
                TextContent(type="text", text=widget_json),
                TextContent(type="text", text=summary),
            ]

    # ── Evict stale in-memory cache entries ─────────────────────────
    now = datetime.now(UTC).timestamp()
    stale = [
        k for k, (_, ts, _tok) in _result_cache.items() if now - ts > RESULT_CACHE_TTL
    ]
    for k in stale:
        _result_cache.pop(k, None)

    # ── Get or fetch the DataFrame ──────────────────────────────────
    if task_id in _result_cache:
        df, _, download_token = _result_cache[task_id]
    else:
        try:
            status_response = handle_response(
                await get_task_status_tasks_task_id_status_get.asyncio(
                    task_id=UUID(task_id),
                    client=client,
                )
            )
            if status_response.status not in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.REVOKED,
            ):
                return [
                    TextContent(
                        type="text",
                        text=(
                            f"Task status is {status_response.status.value}. Cannot fetch results yet.\n"
                            f"Call everyrow_progress(task_id='{task_id}') to check again."
                        ),
                    )
                ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error checking task status: {e!r}")]

        try:
            result_response = handle_response(
                await get_task_result_tasks_task_id_result_get.asyncio(
                    task_id=UUID(task_id),
                    client=client,
                )
            )

            if isinstance(result_response.data, list):
                records = [item.additional_properties for item in result_response.data]
                df = pd.DataFrame(records)
            elif isinstance(result_response.data, TaskResultResponseDataType1):
                df = pd.DataFrame([result_response.data.additional_properties])
            else:
                return [
                    TextContent(
                        type="text", text="Error: Task result has no table data."
                    )
                ]

            # ── GCS mode: upload to GCS + store metadata in Redis ───
            if _gcs_store and _auth_provider is not None:
                try:
                    urls = await asyncio.to_thread(
                        _gcs_store.upload_result, task_id, df
                    )
                    total = len(df)
                    columns = list(df.columns)
                    col_names = ", ".join(columns[:10])
                    if len(columns) > 10:
                        col_names += f", ... (+{len(columns) - 10} more)"

                    # Build preview page
                    offset = min(params.offset, total)
                    page_size = params.page_size
                    page_df = df.iloc[offset : offset + page_size]
                    page_records = page_df.where(page_df.notna(), None).to_dict(
                        orient="records"
                    )

                    # Store metadata in Redis with TTL
                    meta = {
                        "total": total,
                        "columns": columns,
                        "preview": page_records,
                    }
                    await _auth_provider._redis.setex(
                        build_key("result", task_id),
                        RESULT_CACHE_TTL,
                        json.dumps(meta),
                    )

                    widget_json = json.dumps(
                        {
                            "results_url": urls.json_url,
                            "preview": page_records,
                            "total": total,
                        }
                    )

                    has_more = offset + page_size < total
                    next_offset = offset + page_size if has_more else None

                    if has_more:
                        page_size_arg = (
                            f", page_size={page_size}"
                            if page_size != PREVIEW_SIZE
                            else ""
                        )
                        summary = (
                            f"Results: {total} rows, {len(columns)} columns ({col_names}). "
                            f"Showing rows {offset + 1}-{offset + len(page_df)} of {total}.\n"
                            f"Call everyrow_results(task_id='{task_id}', offset={next_offset}{page_size_arg}) for the next page."
                        )
                        if offset == 0:
                            summary += f"\nFull CSV download: {urls.csv_url}\nShare this download link with the user."
                    elif offset == 0:
                        summary = f"Results: {total} rows, {len(columns)} columns ({col_names}). All rows shown."
                    else:
                        summary = f"Results: showing rows {offset + 1}-{offset + len(page_df)} of {total} (final page)."

                    return [
                        TextContent(type="text", text=widget_json),
                        TextContent(type="text", text=summary),
                    ]
                except Exception:
                    logging.getLogger(__name__).exception(
                        "GCS upload failed for task %s, falling back to in-memory cache",
                        task_id,
                    )

            download_token = secrets.token_urlsafe(32)
            _result_cache[task_id] = (df, now, download_token)
        except Exception as e:
            return [TextContent(type="text", text=f"Error retrieving results: {e!r}")]

    # ── stdio mode: save to file (all rows, no pagination) ──────────
    if params.output_path and _transport == "stdio":
        output_file = Path(params.output_path)
        save_result_to_csv(df, output_file)
        _result_cache.pop(task_id, None)
        return [
            TextContent(
                type="text",
                text=(
                    f"Saved {len(df)} rows to {output_file}\n\n"
                    "Tip: For multi-step pipelines, custom response models, or preview mode, "
                    "ask your AI assistant to write Python using the everyrow SDK."
                ),
            )
        ]

    total = len(df)
    col_names = ", ".join(df.columns[:10])
    if len(df.columns) > 10:
        col_names += f", ... (+{len(df.columns) - 10} more)"

    # ── Build page of results ────────────────────────────────────────
    offset = min(params.offset, total)
    page_size = params.page_size
    page_df = df.iloc[offset : offset + page_size]
    page_records = page_df.where(page_df.notna(), None).to_dict(orient="records")
    has_more = offset + page_size < total
    next_offset = offset + page_size if has_more else None

    # TextContent 1: data for the widget.
    # HTTP mode: widget always fetches full data from API.
    # stdio mode: widget reads the page JSON directly.
    if _transport != "stdio" and _mcp_server_url:
        results_url = f"{_mcp_server_url}/api/results/{task_id}?format=json"
        widget_json = json.dumps(
            {
                "results_url": results_url,
                "download_token": download_token,
                "preview": page_records,
                "total": total,
            }
        )
    else:
        widget_json = json.dumps(page_records)

    # TextContent 2: summary for the LLM.
    csv_url = ""
    if _mcp_server_url:
        csv_url = (
            f"{_mcp_server_url}/api/results/{task_id}?token={download_token}&format=csv"
        )

    if has_more:
        page_size_arg = f", page_size={page_size}" if page_size != PREVIEW_SIZE else ""
        summary = (
            f"Results: {total} rows, {len(df.columns)} columns ({col_names}). "
            f"Showing rows {offset + 1}-{offset + len(page_df)} of {total}.\n"
            f"Call everyrow_results(task_id='{task_id}', offset={next_offset}{page_size_arg}) for the next page."
        )
        if csv_url and offset == 0:
            summary += f"\nFull CSV download: {csv_url}\nShare this download link with the user."
    elif offset == 0:
        summary = f"Results: {total} rows, {len(df.columns)} columns ({col_names}). All rows shown."
    else:
        summary = f"Results: showing rows {offset + 1}-{offset + len(page_df)} of {total} (final page)."

    return [
        TextContent(type="text", text=widget_json),
        TextContent(type="text", text=summary),
    ]


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


async def _api_progress(request) -> Any:
    """REST endpoint for the session widget to poll task progress."""
    cors = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET"}

    if request.method == "OPTIONS":
        return Response(status_code=204, headers=cors)

    task_id = request.path_params["task_id"]

    # Validate poll token
    expected_poll = _task_poll_tokens.get(task_id, "")
    request_poll = request.query_params.get("token", "")
    if not expected_poll or not secrets.compare_digest(request_poll, expected_poll):
        return JSONResponse({"error": "Unauthorized"}, status_code=403, headers=cors)

    api_key = _task_tokens.get(task_id)
    if not api_key:
        return JSONResponse({"error": "Unknown task"}, status_code=404, headers=cors)

    try:
        client = AuthenticatedClient(
            base_url=_settings.everyrow_api_url
            if _settings
            else "https://everyrow.io/api/v0",
            token=api_key,
            raise_on_unexpected_status=True,
            follow_redirects=True,
        )
        status_response = handle_response(
            await get_task_status_tasks_task_id_status_get.asyncio(
                task_id=UUID(task_id),
                client=client,
            )
        )

        status = status_response.status
        progress = status_response.progress
        session_url = get_session_url(status_response.session_id)

        completed = progress.completed if progress else 0
        failed = progress.failed if progress else 0
        running = progress.running if progress else 0
        total = progress.total if progress else 0

        elapsed_s = 0
        if status_response.created_at:
            created_at = status_response.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            if (
                status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.REVOKED)
                and status_response.updated_at
            ):
                updated_at = status_response.updated_at
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=UTC)
                elapsed_s = round((updated_at - created_at).total_seconds())
            else:
                elapsed_s = round((datetime.now(UTC) - created_at).total_seconds())

        data = {
            "status": status.value,
            "completed": completed,
            "total": total,
            "failed": failed,
            "running": running,
            "elapsed_s": elapsed_s,
            "session_url": session_url,
        }

        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.REVOKED):
            _task_tokens.pop(task_id, None)
            _task_poll_tokens.pop(task_id, None)

        return JSONResponse(data, headers=cors)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500, headers=cors)


async def _api_results(request) -> Any:
    """REST endpoint for results widget and CSV download.

    Auth: Bearer token in Authorization header, OR ?token= query param (for direct downloads).
    Query params:
        format: "json" (default) or "csv"
    """
    cors = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET",
        "Access-Control-Allow-Headers": "Authorization",
    }
    # Prevent token leakage via Referrer and caching
    security_headers = {
        "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-store, private",
    }

    if request.method == "OPTIONS":
        return Response(status_code=204, headers=cors)

    task_id = request.path_params["task_id"]
    fmt = request.query_params.get("format", "json")

    # Accept token from Authorization header (widget) or query param (direct download)
    token = ""
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    if not token:
        token = request.query_params.get("token", "")

    cached = _result_cache.get(task_id)
    if not cached:
        return JSONResponse(
            {"error": "Results not found or expired"},
            status_code=404,
            headers={**cors, **security_headers},
        )

    df, _ts, expected_token = cached
    if not token or not secrets.compare_digest(token, expected_token):
        return JSONResponse(
            {"error": "Invalid token"},
            status_code=403,
            headers={**cors, **security_headers},
        )

    if fmt == "csv":
        csv_data = df.to_csv(index=False)
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={
                **cors,
                **security_headers,
                "Content-Disposition": f'attachment; filename="results_{task_id[:8]}.csv"',
            },
        )

    # JSON format
    records = df.where(df.notna(), None).to_dict(orient="records")
    return JSONResponse(records, headers={**cors, **security_headers})


def _configure_http_mode(host: str, port: int) -> None:
    """Configure the MCP server for HTTP transport with OAuth."""
    global _transport, _auth_provider, _mcp_server_url, _gcs_store, _settings  # noqa: PLW0603

    _transport = "streamable-http"

    # Validate and parse env vars for HTTP mode
    try:
        settings = HttpSettings()  # pyright: ignore[reportCallIssue]
    except Exception as e:
        logging.error(f"HTTP mode configuration error: {e}")
        sys.exit(1)
    _settings = settings

    # Create Redis client for auth state
    redis_client = create_redis_client(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
        sentinel_endpoints=settings.redis_sentinel_endpoints,
        sentinel_master_name=settings.redis_sentinel_master_name,
    )

    # Create auth provider
    _auth_provider = EveryRowAuthProvider(
        supabase_url=settings.supabase_url,
        supabase_anon_key=settings.supabase_anon_key,
        mcp_server_url=settings.mcp_server_url,
        redis=redis_client,
        encryption_key=settings.redis_encryption_key,
    )

    # Configure auth on the existing FastMCP instance (tools already registered)
    mcp._auth_server_provider = _auth_provider
    mcp._token_verifier = ProviderTokenVerifier(_auth_provider)
    mcp.settings.auth = AuthSettings(
        issuer_url=AnyHttpUrl(settings.mcp_server_url),
        resource_server_url=AnyHttpUrl(settings.mcp_server_url),
        client_registration_options=ClientRegistrationOptions(enabled=True),
        revocation_options=RevocationOptions(enabled=True),
    )
    mcp._mcp_server.lifespan = lifespan_wrapper(mcp, _http_lifespan)
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    )

    # Store server URL for progress polling
    _mcp_server_url = settings.mcp_server_url

    # Initialize GCS result store if enabled via RESULT_STORAGE=gcs
    if settings.result_storage == "gcs":
        _gcs_store = GCSResultStore(settings.gcs_results_bucket)
        logging.getLogger(__name__).info(
            "GCS result store enabled: %s", settings.gcs_results_bucket
        )
    else:
        logging.getLogger(__name__).info("Using in-memory result cache")

    # Re-register session resource with CSP allowing progress endpoint fetch
    @mcp.resource(
        "ui://everyrow/session.html",
        mime_type="text/html;profile=mcp-app",
        meta={
            "ui": {
                "csp": {
                    "resourceDomains": ["https://unpkg.com"],
                    "connectDomains": [settings.mcp_server_url],
                }
            }
        },
    )
    def _session_ui_http() -> str:
        return SESSION_HTML

    # Re-register results resource with CSP allowing results endpoint fetch
    results_connect_domains = [settings.mcp_server_url]
    if _gcs_store:
        results_connect_domains.append("https://storage.googleapis.com")

    @mcp.resource(
        "ui://everyrow/results.html",
        mime_type="text/html;profile=mcp-app",
        meta={
            "ui": {
                "csp": {
                    "resourceDomains": ["https://unpkg.com"],
                    "connectDomains": results_connect_domains,
                }
            }
        },
    )
    def _results_ui_http() -> str:
        return RESULTS_HTML

    # Mount custom routes
    mcp.custom_route("/auth/start/{state}", ["GET"])(_auth_provider.handle_start)
    mcp.custom_route("/auth/callback", ["GET"])(_auth_provider.handle_callback)
    mcp.custom_route("/api/progress/{task_id}", ["GET", "OPTIONS"])(_api_progress)
    mcp.custom_route("/api/results/{task_id}", ["GET", "OPTIONS"])(_api_results)

    async def _health(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    mcp.custom_route("/health", ["GET"])(_health)

    # Wrap the Starlette app with request logging middleware
    _original_streamable_http_app = mcp.streamable_http_app

    def _logging_streamable_http_app():
        app = _original_streamable_http_app()

        class RequestLoggingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                auth_header = request.headers.get("authorization", "(none)")
                if auth_header != "(none)":
                    auth_header = auth_header[:30] + "..."
                logging.warning(
                    "INCOMING %s %s | Host: %s | Auth: %s",
                    request.method,
                    request.url.path,
                    request.headers.get("host", "?"),
                    auth_header,
                )
                response = await call_next(request)
                logging.warning(
                    "RESPONSE %s %s -> %s",
                    request.method,
                    request.url.path,
                    response.status_code,
                )
                return response

        app.add_middleware(RequestLoggingMiddleware)
        return app

    mcp.streamable_http_app = _logging_streamable_http_app


def main():
    """Run the MCP server."""
    global _transport, _settings  # noqa: PLW0603

    parser = argparse.ArgumentParser(description="everyrow MCP server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Use Streamable HTTP transport instead of stdio.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for HTTP transport (default: 0.0.0.0).",
    )
    args = parser.parse_args()

    # Signal to the SDK that we're inside the MCP server (suppresses plugin hints)
    os.environ["EVERYROW_MCP_SERVER"] = "1"

    # Configure logging to use stderr only (stdout is reserved for JSON-RPC)
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s: %(message)s",
        force=True,
    )

    if args.http:
        _configure_http_mode(host=args.host, port=args.port)
        mcp.run(transport="streamable-http")
    else:
        _transport = "stdio"

        # Validate required env vars for stdio mode
        try:
            _settings = StdioSettings()  # pyright: ignore[reportCallIssue]
        except Exception as e:
            logging.error(f"Configuration error: {e}")
            logging.error("Get an API key at https://everyrow.io/api-key")
            sys.exit(1)

        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
