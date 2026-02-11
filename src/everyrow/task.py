import asyncio
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID

from pandas import DataFrame
from pydantic.main import BaseModel

from everyrow.api_utils import create_client, handle_response
from everyrow.constants import EveryrowError
from everyrow.generated.api.tasks import (
    get_task_result_tasks_task_id_result_get,
    get_task_status_tasks_task_id_status_get,
)
from everyrow.generated.client import AuthenticatedClient
from everyrow.generated.models import (
    LLMEnumPublic,
    TaskResultResponse,
    TaskResultResponseDataType1,
    TaskStatus,
    TaskStatusResponse,
)
from everyrow.generated.types import Unset
from everyrow.result import MergeBreakdown, MergeResult, ScalarResult, TableResult

LLM = LLMEnumPublic

_plugin_hint_shown = False


class EffortLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ProgressInfo:
    """Progress counts from the engine's artifact status tracking."""

    pending: int
    running: int
    completed: int
    failed: int
    total: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProgressInfo":
        return cls(
            pending=d.get("pending", 0),
            running=d.get("running", 0),
            completed=d.get("completed", 0),
            failed=d.get("failed", 0),
            total=d.get("total", 0),
        )


def _get_progress(status: TaskStatusResponse) -> ProgressInfo | None:
    """Extract progress info from a status response's additional_properties."""
    raw = status.additional_properties.get("progress")
    if raw is None or not isinstance(raw, dict):
        return None
    return ProgressInfo.from_dict(raw)


def _ts() -> str:
    """Format current time as [HH:MM:SS]."""
    return time.strftime("[%H:%M:%S]")


def _format_eta(completed: int, total: int, elapsed: float) -> str:
    """Estimate remaining time based on completion rate."""
    if completed <= 0 or elapsed <= 0:
        return ""
    rate = completed / elapsed
    remaining = (total - completed) / rate
    return f"~{remaining:.0f}s remaining"


def _default_progress_output(
    progress: ProgressInfo, total: int, elapsed: float
) -> None:
    """Print a progress line to stderr."""
    pct = (progress.completed / total * 100) if total > 0 else 0
    parts = [
        f"{_ts()}   [{progress.completed}/{total}] {pct:3.0f}%",
        f"| {progress.running} running"
        + (f", {progress.failed} failed" if progress.failed else ""),
    ]
    eta = _format_eta(progress.completed, total, elapsed)
    if eta:
        parts.append(f"| {eta}")
    print(" ".join(parts), file=sys.stderr, flush=True)


def _log_jsonl(path: Path, entry: dict[str, Any]) -> None:
    """Append a JSON line to the progress log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


T = TypeVar("T", bound=BaseModel)


class EveryrowTask[T: BaseModel]:
    def __init__(self, response_model: type[T], is_map: bool, is_expand: bool):
        self.task_id: UUID | None = None
        self.session_id: UUID | None = None
        self._client: AuthenticatedClient | None = None
        self._is_map = is_map
        self._is_expand = is_expand
        self._response_model = response_model

    def set_submitted(
        self,
        task_id: UUID,
        session_id: UUID,
        client: AuthenticatedClient,
    ) -> None:
        self.task_id = task_id
        self.session_id = session_id
        self._client = client

    def _get_session_url(self) -> str | None:
        if self.session_id is None:
            return None
        base_url = os.environ.get("EVERYROW_BASE_URL", "https://everyrow.io")
        return f"{base_url}/sessions/{self.session_id}"

    async def get_status(
        self, client: AuthenticatedClient | None = None
    ) -> TaskStatusResponse:
        if self.task_id is None:
            raise EveryrowError("Task must be submitted before fetching status")
        client = client or self._client
        if client is None:
            raise EveryrowError(
                "No client available. Provide a client or use the task within a session context."
            )
        return await get_task_status(self.task_id, client)

    async def await_result(
        self,
        client: AuthenticatedClient | None = None,
        on_progress: Callable[[ProgressInfo], None] | None = None,
    ) -> TableResult | ScalarResult[T]:
        if self.task_id is None:
            raise EveryrowError("Task must be submitted before awaiting result")
        client = client or self._client
        if client is None:
            raise EveryrowError(
                "No client available. Provide a client or use the task within a session context."
            )
        session_url = self._get_session_url()
        final_status = await await_task_completion(
            self.task_id, client, session_url=session_url, on_progress=on_progress
        )

        result_response = await get_task_result(self.task_id, client)
        artifact_id = result_response.artifact_id

        if isinstance(artifact_id, Unset) or artifact_id is None:
            raise EveryrowError("Task result has no artifact ID")

        error = (
            final_status.error if not isinstance(final_status.error, Unset) else None
        )

        if self._is_map or self._is_expand:
            data = _extract_table_data(result_response)
            return TableResult(
                artifact_id=artifact_id,
                data=data,
                error=error,
            )
        else:
            data = _extract_scalar_data(result_response, self._response_model)
            return ScalarResult(
                artifact_id=artifact_id,
                data=data,
                error=error,
            )


def _maybe_show_plugin_hint() -> None:
    """Show a one-time hint about the Claude Code plugin if not running inside the MCP server."""
    global _plugin_hint_shown  # noqa: PLW0603
    if _plugin_hint_shown or os.environ.get("EVERYROW_MCP_SERVER"):
        return
    _plugin_hint_shown = True
    print(
        "Tip: Use the plugin or MCP server for better management of long-running tasks.\n"
        "     See: https://everyrow.io/docs/installation#tab-claude-code-plugin",
        file=sys.stderr,
        flush=True,
    )


async def await_task_completion(
    task_id: UUID,
    client: AuthenticatedClient,
    session_url: str | None = None,
    on_progress: Callable[[ProgressInfo], None] | None = None,
) -> TaskStatusResponse:
    _maybe_show_plugin_hint()
    max_retries = 3
    retries = 0
    last_snapshot: tuple[int, int, int, int] = (-1, -1, -1, -1)
    start_time = time.time()
    total_announced = False
    jsonl_path = Path(os.path.expanduser("~/.everyrow/progress.jsonl"))

    while True:
        try:
            status_response = await get_task_status(task_id, client)
        except Exception as e:
            if retries >= max_retries:
                raise EveryrowError(
                    f"Failed to get task status after {max_retries} retries"
                ) from e
            retries += 1
            await asyncio.sleep(2)
            continue

        retries = 0
        progress = _get_progress(status_response)

        if progress and progress.total > 0:
            if not total_announced:
                total_announced = True
                msg = f"{_ts()} Starting ({progress.total} agents)..."
                if session_url:
                    msg = f"{_ts()} Session: {session_url}\n" + msg
                print(msg, file=sys.stderr, flush=True)
                _log_jsonl(
                    jsonl_path,
                    {
                        "ts": time.time(),
                        "step": "start",
                        "total": progress.total,
                        "session_url": session_url,
                    },
                )

            snapshot = (
                progress.pending,
                progress.running,
                progress.completed,
                progress.failed,
            )
            if snapshot != last_snapshot:
                last_snapshot = snapshot
                elapsed = time.time() - start_time
                if on_progress:
                    try:
                        on_progress(progress)
                    except Exception as e:
                        print(
                            f"Warning: on_progress callback raised {type(e).__name__}: {e}",
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    _default_progress_output(progress, progress.total, elapsed)
                _log_jsonl(
                    jsonl_path,
                    {
                        "ts": time.time(),
                        "completed": progress.completed,
                        "running": progress.running,
                        "failed": progress.failed,
                        "total": progress.total,
                    },
                )

        if status_response.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.REVOKED,
        ):
            break
        await asyncio.sleep(2)

    elapsed = time.time() - start_time
    if progress and progress.total > 0:
        succeeded = progress.completed
        failed = progress.failed
        print(
            f"{_ts()}   [{progress.total}/{progress.total}] 100% | Done ({elapsed:.1f}s total)",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"{_ts()} Results: {succeeded} succeeded"
            + (f", {failed} failed" if failed else ""),
            file=sys.stderr,
            flush=True,
        )
        _log_jsonl(
            jsonl_path,
            {
                "ts": time.time(),
                "step": "done",
                "elapsed": round(elapsed, 1),
                "succeeded": succeeded,
                "failed": failed,
            },
        )

    if status_response.status == TaskStatus.FAILED:
        error_msg = (
            status_response.error
            if not isinstance(status_response.error, Unset)
            else "Unknown error"
        )
        raise EveryrowError(f"Task failed: {error_msg}")

    if status_response.status == TaskStatus.REVOKED:
        raise EveryrowError("Task was revoked")

    return status_response


async def get_task_status(
    task_id: UUID, client: AuthenticatedClient
) -> TaskStatusResponse:
    response = await get_task_status_tasks_task_id_status_get.asyncio(
        task_id=task_id, client=client
    )
    response = handle_response(response)
    return response


async def get_task_result(
    task_id: UUID, client: AuthenticatedClient
) -> TaskResultResponse:
    response = await get_task_result_tasks_task_id_result_get.asyncio(
        task_id=task_id, client=client
    )
    response = handle_response(response)
    return response


def _extract_table_data(result: TaskResultResponse) -> DataFrame:
    if isinstance(result.data, list):
        records = [item.additional_properties for item in result.data]
        return DataFrame(records)
    raise EveryrowError(
        "Expected table result (list of records), but got scalar or null"
    )


def _extract_scalar_data[T: BaseModel](
    result: TaskResultResponse, response_model: type[T]
) -> T:
    if isinstance(result.data, TaskResultResponseDataType1):
        return response_model(**result.data.additional_properties)
    if isinstance(result.data, list) and len(result.data) == 1:
        return response_model(**result.data[0].additional_properties)
    raise EveryrowError("Expected scalar result, but got table or null")


def _extract_merge_breakdown(result: TaskResultResponse) -> MergeBreakdown:
    """Extract merge breakdown from task result response."""
    # The merge_breakdown is stored in additional_properties, not as a direct attribute
    mb = result.additional_properties.get("merge_breakdown", None)
    if mb is None or isinstance(mb, Unset):
        # Return empty breakdown if not present
        return MergeBreakdown(
            exact=[],
            fuzzy=[],
            llm=[],
            web=[],
            unmatched_left=[],
            unmatched_right=[],
        )

    # mb is a dict from additional_properties, access fields with .get()
    return MergeBreakdown(
        exact=[tuple(p) for p in mb.get("exact", []) or []],
        fuzzy=[tuple(p) for p in mb.get("fuzzy", []) or []],
        llm=[tuple(p) for p in mb.get("llm", []) or []],
        web=[tuple(p) for p in mb.get("web", []) or []],
        unmatched_left=list(mb.get("unmatched_left", []) or []),
        unmatched_right=list(mb.get("unmatched_right", []) or []),
    )


class MergeTask:
    """Task class specifically for merge operations that returns MergeResult."""

    def __init__(self) -> None:
        self.task_id: UUID | None = None
        self.session_id: UUID | None = None
        self._client: AuthenticatedClient | None = None

    def set_submitted(
        self,
        task_id: UUID,
        session_id: UUID,
        client: AuthenticatedClient,
    ) -> None:
        self.task_id = task_id
        self.session_id = session_id
        self._client = client

    def _get_session_url(self) -> str | None:
        if self.session_id is None:
            return None
        base_url = os.environ.get("EVERYROW_BASE_URL", "https://everyrow.io")
        return f"{base_url}/sessions/{self.session_id}"

    async def get_status(
        self, client: AuthenticatedClient | None = None
    ) -> TaskStatusResponse:
        if self.task_id is None:
            raise EveryrowError("Task must be submitted before fetching status")
        client = client or self._client
        if client is None:
            raise EveryrowError(
                "No client available. Provide a client or use the task within a session context."
            )
        return await get_task_status(self.task_id, client)

    async def await_result(
        self, client: AuthenticatedClient | None = None
    ) -> MergeResult:
        if self.task_id is None:
            raise EveryrowError("Task must be submitted before awaiting result")
        client = client or self._client
        if client is None:
            raise EveryrowError(
                "No client available. Provide a client or use the task within a session context."
            )
        session_url = self._get_session_url()
        final_status = await await_task_completion(
            self.task_id, client, session_url=session_url
        )

        result_response = await get_task_result(self.task_id, client)
        artifact_id = result_response.artifact_id

        if isinstance(artifact_id, Unset) or artifact_id is None:
            raise EveryrowError("Task result has no artifact ID")

        error = (
            final_status.error if not isinstance(final_status.error, Unset) else None
        )

        data = _extract_table_data(result_response)
        breakdown = _extract_merge_breakdown(result_response)

        return MergeResult(
            artifact_id=artifact_id,
            data=data,
            error=error,
            breakdown=breakdown,
        )


async def fetch_task_data(
    task_id: UUID | str,
    client: AuthenticatedClient | None = None,
) -> DataFrame:
    """Fetch the result data for a completed task as a pandas DataFrame.

    Args:
        task_id: The UUID of the task to fetch data for (can be a string or UUID).
        client: Optional authenticated client. If not provided, one will be created
            using the EVERYROW_API_KEY environment variable.

    Returns:
        A pandas DataFrame containing the task result data.

    Raises:
        EveryrowError: If the task has not completed, failed, or has no artifact.
    """
    if isinstance(task_id, str):
        task_id = UUID(task_id)

    if client is None:
        client = create_client()

    status_response = await get_task_status(task_id, client)

    if status_response.status != TaskStatus.COMPLETED:
        raise EveryrowError(
            f"Task {task_id} is not completed (status: {status_response.status.value})."
        )

    result_response = await get_task_result(task_id, client)
    return _extract_table_data(result_response)
