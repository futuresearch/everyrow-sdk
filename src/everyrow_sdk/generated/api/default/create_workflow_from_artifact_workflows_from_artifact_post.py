from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.create_workflow_from_artifact_request import CreateWorkflowFromArtifactRequest
from ...models.create_workflow_from_artifact_response import CreateWorkflowFromArtifactResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: CreateWorkflowFromArtifactRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/workflows/from-artifact",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> CreateWorkflowFromArtifactResponse | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = CreateWorkflowFromArtifactResponse.from_dict(response.json())

        return response_200

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[CreateWorkflowFromArtifactResponse | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: CreateWorkflowFromArtifactRequest,
) -> Response[CreateWorkflowFromArtifactResponse | HTTPValidationError]:
    """Create Workflow From Artifact

     Create a workflow from an artifact by traversing its dependency graph.

    Args:
        body (CreateWorkflowFromArtifactRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CreateWorkflowFromArtifactResponse | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    body: CreateWorkflowFromArtifactRequest,
) -> CreateWorkflowFromArtifactResponse | HTTPValidationError | None:
    """Create Workflow From Artifact

     Create a workflow from an artifact by traversing its dependency graph.

    Args:
        body (CreateWorkflowFromArtifactRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CreateWorkflowFromArtifactResponse | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: CreateWorkflowFromArtifactRequest,
) -> Response[CreateWorkflowFromArtifactResponse | HTTPValidationError]:
    """Create Workflow From Artifact

     Create a workflow from an artifact by traversing its dependency graph.

    Args:
        body (CreateWorkflowFromArtifactRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CreateWorkflowFromArtifactResponse | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: CreateWorkflowFromArtifactRequest,
) -> CreateWorkflowFromArtifactResponse | HTTPValidationError | None:
    """Create Workflow From Artifact

     Create a workflow from an artifact by traversing its dependency graph.

    Args:
        body (CreateWorkflowFromArtifactRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CreateWorkflowFromArtifactResponse | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
