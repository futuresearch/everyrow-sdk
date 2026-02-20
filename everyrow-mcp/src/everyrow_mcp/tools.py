"""MCP tool functions and their helpers."""

import asyncio
import json
import logging
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd
from everyrow.api_utils import handle_response
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
from mcp.types import TextContent, ToolAnnotations
from pydantic import BaseModel, create_model

from everyrow_mcp.app import _clear_task_state, mcp
from everyrow_mcp.gcs_results import try_cached_gcs_result, try_upload_gcs_result
from everyrow_mcp.models import (
    AgentInput,
    DedupeInput,
    MergeInput,
    ProgressInput,
    RankInput,
    ResultsInput,
    ScreenInput,
    SingleAgentInput,
    _schema_to_model,
)
from everyrow_mcp.state import (
    PROGRESS_POLL_DELAY,
    TASK_STATE_FILE,
    state,
)
from everyrow_mcp.utils import load_csv, save_result_to_csv


def _get_client():
    """Get an EveryRow API client for the current request.

    In stdio mode, returns the singleton client initialized at startup.
    In HTTP mode, creates a per-request client using the authenticated
    user's API key from the OAuth access token.
    """
    if state.is_stdio:
        if state.client is None:
            raise RuntimeError("MCP server not initialized")
        return state.client
    # HTTP mode: get JWT from authenticated request
    access_token = get_access_token()
    if access_token is None:
        raise RuntimeError("Not authenticated")
    if state.settings is None:
        raise RuntimeError("MCP server not initialized")
    return AuthenticatedClient(
        base_url=state.settings.everyrow_api_url,
        token=access_token.token,
        raise_on_unexpected_status=True,
        follow_redirects=True,
    )


def _submission_text(label: str, session_url: str, task_id: str) -> str:
    """Build human-readable text for submission tool results."""
    if state.is_stdio:
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


async def _submission_ui_json(
    session_url: str,
    task_id: str,
    total: int,
    token: str,
) -> str:
    """Build JSON for the session MCP App widget, and store the token for polling."""
    await state.store_task_token(task_id, token)
    poll_token = secrets.token_urlsafe(32)
    await state.store_poll_token(task_id, poll_token)
    data: dict[str, Any] = {
        "session_url": session_url,
        "task_id": task_id,
        "total": total,
        "status": "submitted",
    }
    if state.mcp_server_url:
        data["progress_url"] = (
            f"{state.mcp_server_url}/api/progress/{task_id}?token={poll_token}"
        )
    return json.dumps(data)


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
    if state.is_http:
        return
    try:
        TASK_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        task_state = {
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
            json.dump(task_state, f)
    except Exception as e:
        logging.getLogger(__name__).debug(f"Failed to write task state: {e!r}")


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
            text=await _submission_ui_json(session_url, task_id, len(df), client.token),
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
        fields: dict[str, Any] = {k: (type(v), v) for k, v in params.input_data.items()}
        DynamicInput = create_model("DynamicInput", **fields)  # pyright: ignore[reportArgumentType, reportCallIssue]
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
            text=await _submission_ui_json(session_url, task_id, 1, client.token),
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
            text=await _submission_ui_json(session_url, task_id, len(df), client.token),
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
            text=await _submission_ui_json(session_url, task_id, len(df), client.token),
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
    """Remove duplicate rows from a CSV file using semantic equivalence."""
    client = _get_client()
    _clear_task_state()

    df = load_csv(
        input_csv=params.input_csv,
        input_data=params.input_data,
        input_json=params.input_json,
    )

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
            text=await _submission_ui_json(
                session_url=session_url,
                task_id=task_id,
                total=len(df),
                token=client.token,
            ),
        ),
        TextContent(
            type="text",
            text=_submission_text(
                f"Submitted: {len(df)} rows for deduplication.",
                session_url,
                task_id,
            ),
        ),
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
    """Join two CSV files using intelligent entity matching."""
    client = _get_client()
    _clear_task_state()

    left_df = load_csv(
        input_csv=params.left_csv,
        input_data=params.left_input_data,
        input_json=params.left_input_json,
    )

    right_df = load_csv(
        input_csv=params.right_csv,
        input_data=params.right_input_data,
        input_json=params.right_input_json,
    )

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
            text=await _submission_ui_json(
                session_url=session_url,
                task_id=task_id,
                total=len(left_df),
                token=client.token,
            ),
        ),
        TextContent(
            type="text",
            text=_submission_text(
                f"Submitted: {len(left_df)} left rows for merging.",
                session_url,
                task_id,
            ),
        ),
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
async def everyrow_results(params: ResultsInput) -> list[TextContent]:
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

    # ── GCS mode: return from Redis cache if available ─────────────
    gcs_cached = await try_cached_gcs_result(task_id, params.offset, params.page_size)
    if gcs_cached is not None:
        return gcs_cached

    # ── In-memory cache hit ────────────────────────────────────────
    session_url = ""
    await state.evict_stale_results()
    cached = await state.get_cached_result(task_id)
    if cached is not None:
        df, _, download_token = cached
    else:
        # ── Fetch from API ─────────────────────────────────────────
        try:
            df, session_id = await _fetch_task_result(client, task_id)
            session_url = get_session_url(session_id) if session_id else ""
        except TaskNotReady as e:
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Task status is {e.status}. Cannot fetch results yet.\n"
                        f"Call everyrow_progress(task_id='{task_id}') to check again."
                    ),
                )
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error retrieving results: {e!r}")]

        # ── Try GCS upload (returns None if not configured or fails) ──
        gcs_response = await try_upload_gcs_result(
            task_id, df, params.offset, params.page_size, session_url
        )
        if gcs_response is not None:
            return gcs_response

        # ── Fall back to in-memory cache ───────────────────────────
        download_token = secrets.token_urlsafe(32)
        await state.set_cached_result(
            task_id, df, datetime.now(UTC).timestamp(), download_token
        )

    # ── stdio mode: save to file (all rows, no pagination) ──────────
    if params.output_path and state.is_stdio:
        output_file = Path(params.output_path)
        save_result_to_csv(df, output_file)
        await state.pop_cached_result(task_id)
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

    # ── Build paginated inline response ─────────────────────────────
    return _build_inline_response(task_id, df, download_token, params, session_url)


class TaskNotReady(Exception):
    """Raised when a task is not in a terminal state."""

    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__(status)


async def _fetch_task_result(client: Any, task_id: str) -> tuple[pd.DataFrame, str]:
    """Fetch a task's result DataFrame and session ID from the API.

    Checks task status first, then retrieves and parses the result data.

    Returns:
        Tuple of (DataFrame, session_id).

    Raises:
        TaskNotReady: If the task is not in a terminal state.
        ValueError: If the result has no table data.
        Exception: On API errors.
    """
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
        raise TaskNotReady(status_response.status.value)

    session_id = str(status_response.session_id) if status_response.session_id else ""

    result_response = handle_response(
        await get_task_result_tasks_task_id_result_get.asyncio(
            task_id=UUID(task_id),
            client=client,
        )
    )

    if isinstance(result_response.data, list):
        records = [item.additional_properties for item in result_response.data]
        return pd.DataFrame(records), session_id
    if isinstance(result_response.data, TaskResultResponseDataType1):
        return pd.DataFrame([result_response.data.additional_properties]), session_id
    raise ValueError("Task result has no table data.")


_TOKEN_BUDGET = 4000  # target tokens per page of inline results
_CHARS_PER_TOKEN = 4  # conservative estimate for JSON-heavy text


def _recommend_page_size(
    page_json: str, rows_shown: int, total: int, current_page_size: int
) -> int | None:
    """Estimate an optimal page_size based on the current page's token cost.

    Returns a recommended page_size, or None if the current one is fine.
    """
    if rows_shown == 0:
        return None
    avg_chars_per_row = len(page_json) / rows_shown
    avg_tokens_per_row = avg_chars_per_row / _CHARS_PER_TOKEN
    if avg_tokens_per_row <= 0:
        return None
    recommended = max(1, min(int(_TOKEN_BUDGET / avg_tokens_per_row), 100))
    # Only recommend if meaningfully different (>25% change) and there are more rows
    if (
        recommended >= total
        or abs(recommended - current_page_size) / current_page_size < 0.25
    ):
        return None
    return recommended


def _build_inline_response(
    task_id: str,
    df: pd.DataFrame,
    download_token: str,
    params: ResultsInput,
    session_url: str = "",
) -> list[TextContent]:
    """Build paginated inline response for in-memory results."""
    total = len(df)
    col_names = ", ".join(df.columns[:10])
    if len(df.columns) > 10:
        col_names += f", ... (+{len(df.columns) - 10} more)"

    offset = min(params.offset, total)
    page_size = params.page_size
    page_df = df.iloc[offset : offset + page_size]
    page_records = page_df.where(page_df.notna(), None).to_dict(orient="records")
    has_more = offset + page_size < total
    next_offset = offset + page_size if has_more else None

    # Widget JSON: HTTP mode includes results_url, stdio mode is plain records
    if state.is_http and state.mcp_server_url:
        results_url = f"{state.mcp_server_url}/api/results/{task_id}?format=json"
        csv_download_url = f"{state.mcp_server_url}/api/results/{task_id}?token={download_token}&format=csv"
        widget_data: dict[str, Any] = {
            "results_url": results_url,
            "download_token": download_token,
            "csv_url": csv_download_url,
            "preview": page_records,
            "total": total,
        }
        if session_url:
            widget_data["session_url"] = session_url
        widget_json = json.dumps(widget_data)
    else:
        widget_json = json.dumps(page_records)

    # Token-based page_size recommendation
    recommended = _recommend_page_size(widget_json, len(page_df), total, page_size)

    # Summary text for the LLM
    csv_url = ""
    if state.mcp_server_url:
        csv_url = f"{state.mcp_server_url}/api/results/{task_id}?token={download_token}&format=csv"

    default_page_size = ResultsInput.model_fields["page_size"].default
    if has_more:
        # Use recommendation in the call hint; fall back to current page_size
        hint_page_size = recommended if recommended is not None else page_size
        page_size_arg = (
            f", page_size={hint_page_size}"
            if hint_page_size != default_page_size
            else ""
        )
        summary = (
            f"Results: {total} rows, {len(df.columns)} columns ({col_names}). "
            f"Showing rows {offset + 1}-{offset + len(page_df)} of {total}.\n"
        )
        if recommended is not None:
            summary += f"Recommended page_size={recommended} for these results.\n"
        summary += f"Call everyrow_results(task_id='{task_id}', offset={next_offset}{page_size_arg}) for the next page."
        if csv_url and offset == 0:
            summary += f"\nFull CSV download: {csv_url}\nShare this download link with the user."
    elif offset == 0:
        summary = f"Results: {total} rows, {len(df.columns)} columns ({col_names}). All rows shown."
        if csv_url:
            summary += f"\nFull CSV download: {csv_url}\nShare this download link with the user."
    else:
        summary = f"Results: showing rows {offset + 1}-{offset + len(page_df)} of {total} (final page)."

    return [
        TextContent(type="text", text=widget_json),
        TextContent(type="text", text=summary),
    ]
