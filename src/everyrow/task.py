import asyncio
from typing import TypeVar, cast
from uuid import UUID

from pandas import DataFrame
from pydantic.main import BaseModel

from everyrow.api_utils import create_client, handle_response
from everyrow.citations import render_citations_group, render_citations_standalone
from everyrow.constants import EveryrowError
from everyrow.generated.api.default import (
    get_artifacts_artifacts_get,
    get_task_status_endpoint_tasks_task_id_status_get,
    submit_task_tasks_post,
)
from everyrow.generated.client import AuthenticatedClient
from everyrow.generated.models import (
    ArtifactGroupRecord,
    LLMEnum,
    StandaloneArtifactRecord,
    TaskEffort,
    TaskStatus,
    TaskStatusResponse,
)
from everyrow.generated.models.submit_task_body import SubmitTaskBody
from everyrow.result import ScalarResult, TableResult

# "export" generated types.
LLM = LLMEnum
EffortLevel = TaskEffort

T = TypeVar("T", bound=BaseModel)


class EveryrowTask[T: BaseModel]:
    def __init__(self, response_model: type[T], is_map: bool, is_expand: bool):
        self.task_id: UUID | None = None
        self.session_id: UUID | None = None
        self._client: AuthenticatedClient | None = None
        self._is_map = is_map
        self._is_expand = is_expand
        self._response_model = response_model

    async def submit(
        self,
        body: SubmitTaskBody,
        client: AuthenticatedClient,
    ) -> UUID:
        task_id = await submit_task(body, client)
        self.task_id = task_id
        self.session_id = body.session_id
        self._client = client
        return task_id

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
    ) -> TableResult | ScalarResult[T]:
        if self.task_id is None:
            raise EveryrowError("Task must be submitted before awaiting result")
        client = client or self._client
        if client is None:
            raise EveryrowError(
                "No client available. Provide a client or use the task within a session context."
            )
        final_status_response = await await_task_completion(self.task_id, client)
        artifact_id = cast(
            UUID, final_status_response.artifact_id
        )  # we check artifact_id in await_task_completion

        if self._is_map or self._is_expand:
            data = await read_table_result(artifact_id, client=client)
            return TableResult(
                artifact_id=artifact_id,
                data=data,
                error=final_status_response.error,
            )
        else:
            data = await read_scalar_result(
                artifact_id, self._response_model, client=client
            )
            return ScalarResult(
                artifact_id=artifact_id,
                data=data,
                error=final_status_response.error,
            )


async def submit_task(body: SubmitTaskBody, client: AuthenticatedClient) -> UUID:
    response = await submit_task_tasks_post.asyncio(client=client, body=body)
    response = handle_response(response)
    return response.task_id


async def await_task_completion(
    task_id: UUID, client: AuthenticatedClient
) -> TaskStatusResponse:
    max_retries = 3
    retries = 0
    while True:
        try:
            status_response = await get_task_status(task_id, client)
        except Exception as e:
            if retries >= max_retries:
                raise EveryrowError(
                    f"Failed to get task status after {max_retries} retries"
                ) from e
            retries += 1
        else:
            retries = 0
            if status_response.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.REVOKED,
            ):
                break
        await asyncio.sleep(1)
    if (
        status_response.status == TaskStatus.FAILED
        or status_response.artifact_id is None
    ):
        raise EveryrowError(
            f"Failed to create input in everyrow: {status_response.error}"
        )

    return status_response


async def get_task_status(
    task_id: UUID, client: AuthenticatedClient
) -> TaskStatusResponse:
    response = await get_task_status_endpoint_tasks_task_id_status_get.asyncio(
        client=client, task_id=task_id
    )
    response = handle_response(response)
    return response


async def read_table_result(
    artifact_id: UUID,
    client: AuthenticatedClient,
) -> DataFrame:
    response = await get_artifacts_artifacts_get.asyncio(
        client=client, artifact_ids=[artifact_id]
    )
    response = handle_response(response)
    if len(response) != 1:
        raise EveryrowError(f"Expected 1 artifact, got {len(response)}")
    artifact = response[0]
    if not isinstance(artifact, ArtifactGroupRecord):
        raise EveryrowError("Expected table result, but got a scalar")

    artifact = render_citations_group(artifact)

    return DataFrame([a.data for a in artifact.artifacts])


async def read_scalar_result[T: BaseModel](
    artifact_id: UUID,
    response_model: type[T],
    client: AuthenticatedClient,
) -> T:
    response = await get_artifacts_artifacts_get.asyncio(
        client=client, artifact_ids=[artifact_id]
    )
    response = handle_response(response)
    if len(response) != 1:
        raise EveryrowError(f"Expected 1 artifact, got {len(response)}")
    artifact = response[0]
    if not isinstance(artifact, StandaloneArtifactRecord):
        raise EveryrowError("Expected scalar result, but got a table")

    artifact = render_citations_standalone(artifact)

    return response_model(**artifact.data)


async def fetch_task_data(
    task_id: UUID | str,
    client: AuthenticatedClient | None = None,
) -> DataFrame:
    """Fetch the result data for a completed task as a pandas DataFrame.

    This is a convenience helper that retrieves the table-level group artifact
    associated with a task and returns it as a DataFrame.

    Args:
        task_id: The UUID of the task to fetch data for (can be a string or UUID).
        client: Optional authenticated client. If not provided, one will be created
            using the EVERYROW_API_KEY environment variable.

    Returns:
        A pandas DataFrame containing the task result data.

    Raises:
        EveryrowError: If the task has not completed, failed, or has no artifact.

    Example:
        >>> from everyrow import fetch_task_data
        >>> df = await fetch_task_data("12345678-1234-1234-1234-123456789abc")
        >>> print(df.head())
    """
    if isinstance(task_id, str):
        task_id = UUID(task_id)

    if client is None:
        client = create_client()

    status_response = await get_task_status(task_id, client)

    if status_response.status not in (TaskStatus.COMPLETED,):
        raise EveryrowError(
            f"Task {task_id} is not completed (status: {status_response.status.value}). Error: {status_response.error}"
        )

    if status_response.artifact_id is None:
        raise EveryrowError(f"Task {task_id} has no associated artifact.")

    return await read_table_result(status_response.artifact_id, client)
