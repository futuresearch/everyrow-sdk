from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.rollback_to_message_request import RollbackToMessageRequest
from ...models.rollback_to_message_response import RollbackToMessageResponse
from ...types import Response


def _get_kwargs(
    *,
    body: RollbackToMessageRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/tasks/chat/rollback",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | RollbackToMessageResponse | None:
    if response.status_code == 200:
        response_200 = RollbackToMessageResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | RollbackToMessageResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: RollbackToMessageRequest,
) -> Response[HTTPValidationError | RollbackToMessageResponse]:
    """Rollback To Message Endpoint

     Delete a given message and all chat messages, tasks, and artifacts created at or after that message.

    ⚠️ DANGEROUS: This endpoint permanently deletes data and requires
    ENGINE_ENABLE_DANGEROUS_DEBUG_FEATURES=true.

    Args:
        body (RollbackToMessageRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | RollbackToMessageResponse]
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
    body: RollbackToMessageRequest,
) -> HTTPValidationError | RollbackToMessageResponse | None:
    """Rollback To Message Endpoint

     Delete a given message and all chat messages, tasks, and artifacts created at or after that message.

    ⚠️ DANGEROUS: This endpoint permanently deletes data and requires
    ENGINE_ENABLE_DANGEROUS_DEBUG_FEATURES=true.

    Args:
        body (RollbackToMessageRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | RollbackToMessageResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: RollbackToMessageRequest,
) -> Response[HTTPValidationError | RollbackToMessageResponse]:
    """Rollback To Message Endpoint

     Delete a given message and all chat messages, tasks, and artifacts created at or after that message.

    ⚠️ DANGEROUS: This endpoint permanently deletes data and requires
    ENGINE_ENABLE_DANGEROUS_DEBUG_FEATURES=true.

    Args:
        body (RollbackToMessageRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | RollbackToMessageResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: RollbackToMessageRequest,
) -> HTTPValidationError | RollbackToMessageResponse | None:
    """Rollback To Message Endpoint

     Delete a given message and all chat messages, tasks, and artifacts created at or after that message.

    ⚠️ DANGEROUS: This endpoint permanently deletes data and requires
    ENGINE_ENABLE_DANGEROUS_DEBUG_FEATURES=true.

    Args:
        body (RollbackToMessageRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | RollbackToMessageResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
