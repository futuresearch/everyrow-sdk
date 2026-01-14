from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.create_api_key_request import CreateAPIKeyRequest
from ...models.create_api_key_response import CreateAPIKeyResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: CreateAPIKeyRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/api-keys/create",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> CreateAPIKeyResponse | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = CreateAPIKeyResponse.from_dict(response.json())

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
) -> Response[CreateAPIKeyResponse | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: CreateAPIKeyRequest,
) -> Response[CreateAPIKeyResponse | HTTPValidationError]:
    """Create Api Key Endpoint

     Create a new API key for the current user.

    The full key is returned only once and cannot be retrieved again.
    Store it securely - only the hash is kept in the database.

    Args:
        body (CreateAPIKeyRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CreateAPIKeyResponse | HTTPValidationError]
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
    body: CreateAPIKeyRequest,
) -> CreateAPIKeyResponse | HTTPValidationError | None:
    """Create Api Key Endpoint

     Create a new API key for the current user.

    The full key is returned only once and cannot be retrieved again.
    Store it securely - only the hash is kept in the database.

    Args:
        body (CreateAPIKeyRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CreateAPIKeyResponse | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: CreateAPIKeyRequest,
) -> Response[CreateAPIKeyResponse | HTTPValidationError]:
    """Create Api Key Endpoint

     Create a new API key for the current user.

    The full key is returned only once and cannot be retrieved again.
    Store it securely - only the hash is kept in the database.

    Args:
        body (CreateAPIKeyRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CreateAPIKeyResponse | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: CreateAPIKeyRequest,
) -> CreateAPIKeyResponse | HTTPValidationError | None:
    """Create Api Key Endpoint

     Create a new API key for the current user.

    The full key is returned only once and cannot be retrieved again.
    Store it securely - only the hash is kept in the database.

    Args:
        body (CreateAPIKeyRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CreateAPIKeyResponse | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
