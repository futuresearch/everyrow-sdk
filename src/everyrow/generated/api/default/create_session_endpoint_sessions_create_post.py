from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.create_session_request import CreateSessionRequest
from ...models.create_session_response import CreateSessionResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: CreateSessionRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/sessions/create",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> CreateSessionResponse | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = CreateSessionResponse.from_dict(response.json())

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
) -> Response[CreateSessionResponse | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: CreateSessionRequest,
) -> Response[CreateSessionResponse | HTTPValidationError]:
    """Create Session Endpoint

     Create a new session for the current user.

    Returns:
        CreateSessionResponse containing the new session's ID.

    Raises:
        HTTPException: 500 if session creation fails.

    Args:
        body (CreateSessionRequest): Request to create a new session.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CreateSessionResponse | HTTPValidationError]
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
    body: CreateSessionRequest,
) -> CreateSessionResponse | HTTPValidationError | None:
    """Create Session Endpoint

     Create a new session for the current user.

    Returns:
        CreateSessionResponse containing the new session's ID.

    Raises:
        HTTPException: 500 if session creation fails.

    Args:
        body (CreateSessionRequest): Request to create a new session.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CreateSessionResponse | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: CreateSessionRequest,
) -> Response[CreateSessionResponse | HTTPValidationError]:
    """Create Session Endpoint

     Create a new session for the current user.

    Returns:
        CreateSessionResponse containing the new session's ID.

    Raises:
        HTTPException: 500 if session creation fails.

    Args:
        body (CreateSessionRequest): Request to create a new session.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CreateSessionResponse | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: CreateSessionRequest,
) -> CreateSessionResponse | HTTPValidationError | None:
    """Create Session Endpoint

     Create a new session for the current user.

    Returns:
        CreateSessionResponse containing the new session's ID.

    Raises:
        HTTPException: 500 if session creation fails.

    Args:
        body (CreateSessionRequest): Request to create a new session.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CreateSessionResponse | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
