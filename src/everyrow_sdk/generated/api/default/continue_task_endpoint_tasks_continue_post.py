from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.continue_task_request import ContinueTaskRequest
from ...models.http_validation_error import HTTPValidationError
from ...models.insufficient_balance_error import InsufficientBalanceError
from ...models.task_response import TaskResponse
from ...types import Response


def _get_kwargs(
    *,
    body: ContinueTaskRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/tasks/continue",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | InsufficientBalanceError | TaskResponse | None:
    if response.status_code == 200:
        response_200 = TaskResponse.from_dict(response.json())

        return response_200

    if response.status_code == 402:
        response_402 = InsufficientBalanceError.from_dict(response.json())

        return response_402

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[HTTPValidationError | InsufficientBalanceError | TaskResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: ContinueTaskRequest,
) -> Response[HTTPValidationError | InsufficientBalanceError | TaskResponse]:
    r"""Continue Task Endpoint

     Continue a task by (re)-executing children in place.

    Reason for continue: failed/revoked (\"failed\"), continue after preview (\"from_preview\")

    For MAP tasks, this resets child artifacts and re-submits
    only those children, keeping the existing task and successful results.

    For non-MAP tasks, this returns an error as they cannot be continued.

    Args:
        body (ContinueTaskRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | InsufficientBalanceError | TaskResponse]
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
    body: ContinueTaskRequest,
) -> HTTPValidationError | InsufficientBalanceError | TaskResponse | None:
    r"""Continue Task Endpoint

     Continue a task by (re)-executing children in place.

    Reason for continue: failed/revoked (\"failed\"), continue after preview (\"from_preview\")

    For MAP tasks, this resets child artifacts and re-submits
    only those children, keeping the existing task and successful results.

    For non-MAP tasks, this returns an error as they cannot be continued.

    Args:
        body (ContinueTaskRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | InsufficientBalanceError | TaskResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: ContinueTaskRequest,
) -> Response[HTTPValidationError | InsufficientBalanceError | TaskResponse]:
    r"""Continue Task Endpoint

     Continue a task by (re)-executing children in place.

    Reason for continue: failed/revoked (\"failed\"), continue after preview (\"from_preview\")

    For MAP tasks, this resets child artifacts and re-submits
    only those children, keeping the existing task and successful results.

    For non-MAP tasks, this returns an error as they cannot be continued.

    Args:
        body (ContinueTaskRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | InsufficientBalanceError | TaskResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: ContinueTaskRequest,
) -> HTTPValidationError | InsufficientBalanceError | TaskResponse | None:
    r"""Continue Task Endpoint

     Continue a task by (re)-executing children in place.

    Reason for continue: failed/revoked (\"failed\"), continue after preview (\"from_preview\")

    For MAP tasks, this resets child artifacts and re-submits
    only those children, keeping the existing task and successful results.

    For non-MAP tasks, this returns an error as they cannot be continued.

    Args:
        body (ContinueTaskRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | InsufficientBalanceError | TaskResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
