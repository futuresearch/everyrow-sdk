import asyncio
import logging
import sys
import time
from collections.abc import Callable
from enum import StrEnum
from typing import TypeVar
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
    TaskProgressInfo,
    TaskResultResponse,
    TaskResultResponseDataType1,
    TaskStatus,
    TaskStatusResponse,
)
from everyrow.generated.types import Unset
from everyrow.result import MergeBreakdown, MergeResult, ScalarResult, TableResult
from everyrow.session import get_session_url

LLM = LLMEnumPublic

# Configure the everyrow logger. Users can customize this logger to redirect
# or silence progress output (e.g., logging.getLogger("everyrow").setLevel(logging.WARNING))
_logger = logging.getLogger("everyrow")
if not _logger.handlers:
    # Only add handler if none exists (avoids duplicate handlers on re-import)
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)


class EffortLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _default_progress_output(task_status: TaskStatusResponse) -> None:
    progress = task_status.progress
    completed = progress and progress.completed
    running = progress and progress.running
    failed = progress and progress.failed
    total = progress and progress.total
    pct = (completed / total * 100) if total and completed else 0
    elapsed_str = ""
    if task_status.created_at:
        elapsed_str = f"({time.time() - task_status.created_at.timestamp():.0f}s)"
    message = (
        f"{elapsed_str:>5s} [{completed}/{total}] {pct:3.0f}%"
        + (f"| {running} running" if running else "")
        + (f"| {failed} failed" if failed else "")
    )
    _logger.info(message)


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
        on_progress: Callable[[TaskProgressInfo], None] | None = None,
    ) -> TableResult | ScalarResult[T]:
        if self.task_id is None:
            raise EveryrowError("Task must be submitted before awaiting result")
        client = client or self._client
        if client is None:
            raise EveryrowError(
                "No client available. Provide a client or use the task within a session context."
            )
        session_url = get_session_url(self.session_id) if self.session_id else None
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


async def await_task_completion(
    task_id: UUID,
    client: AuthenticatedClient,
    session_url: str | None = None,
    on_progress: Callable[[TaskProgressInfo], None] | None = None,
) -> TaskStatusResponse:
    max_retries = 3
    retries = 0
    last_progress: TaskProgressInfo | None = None
    total_announced = False

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
        if status_response.progress and status_response.progress.total > 0:
            if not total_announced:
                total_announced = True
                _logger.info(f"Processing {status_response.progress.total} rows...")
                if session_url:
                    _logger.info(f"Session: {session_url}")

            if status_response.progress != last_progress:
                last_progress = status_response.progress
                if on_progress:
                    try:
                        on_progress(status_response.progress)
                    except Exception as e:
                        _logger.warning(
                            "on_progress callback raised %s: %s", type(e).__name__, e
                        )
                else:
                    _default_progress_output(status_response)

        if status_response.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.REVOKED,
        ):
            break
        await asyncio.sleep(2)

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
    mb = result.merge_breakdown
    if mb is None or isinstance(mb, Unset):
        return MergeBreakdown(
            exact=[],
            fuzzy=[],
            llm=[],
            web=[],
            unmatched_left=[],
            unmatched_right=[],
        )

    return MergeBreakdown(
        exact=[(p[0], p[1]) for p in mb.exact]
        if not isinstance(mb.exact, Unset)
        else [],
        fuzzy=[(p[0], p[1]) for p in mb.fuzzy]
        if not isinstance(mb.fuzzy, Unset)
        else [],
        llm=[(p[0], p[1]) for p in mb.llm] if not isinstance(mb.llm, Unset) else [],
        web=[(p[0], p[1]) for p in mb.web] if not isinstance(mb.web, Unset) else [],
        unmatched_left=list(mb.unmatched_left)
        if not isinstance(mb.unmatched_left, Unset)
        else [],
        unmatched_right=list(mb.unmatched_right)
        if not isinstance(mb.unmatched_right, Unset)
        else [],
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
        session_url = get_session_url(self.session_id) if self.session_id else None
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
