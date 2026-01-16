from http import HTTPStatus
from typing import Any
from urllib.parse import quote
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.revoke_api_key_response import RevokeAPIKeyResponse
from ...types import Response


def _get_kwargs(
    key_id: UUID,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/api-keys/{key_id}/revoke".format(
            key_id=quote(str(key_id), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | RevokeAPIKeyResponse | None:
    if response.status_code == 200:
        response_200 = RevokeAPIKeyResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | RevokeAPIKeyResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    key_id: UUID,
    *,
    client: AuthenticatedClient,
) -> Response[HTTPValidationError | RevokeAPIKeyResponse]:
    """Revoke Api Key Endpoint

     Revoke an API key (soft delete).

    The key will no longer be usable for authentication, but the record
    remains in the database for audit purposes.

    Args:
        key_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | RevokeAPIKeyResponse]
    """

    kwargs = _get_kwargs(
        key_id=key_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    key_id: UUID,
    *,
    client: AuthenticatedClient,
) -> HTTPValidationError | RevokeAPIKeyResponse | None:
    """Revoke Api Key Endpoint

     Revoke an API key (soft delete).

    The key will no longer be usable for authentication, but the record
    remains in the database for audit purposes.

    Args:
        key_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | RevokeAPIKeyResponse
    """

    return sync_detailed(
        key_id=key_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    key_id: UUID,
    *,
    client: AuthenticatedClient,
) -> Response[HTTPValidationError | RevokeAPIKeyResponse]:
    """Revoke Api Key Endpoint

     Revoke an API key (soft delete).

    The key will no longer be usable for authentication, but the record
    remains in the database for audit purposes.

    Args:
        key_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | RevokeAPIKeyResponse]
    """

    kwargs = _get_kwargs(
        key_id=key_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    key_id: UUID,
    *,
    client: AuthenticatedClient,
) -> HTTPValidationError | RevokeAPIKeyResponse | None:
    """Revoke Api Key Endpoint

     Revoke an API key (soft delete).

    The key will no longer be usable for authentication, but the record
    remains in the database for audit purposes.

    Args:
        key_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | RevokeAPIKeyResponse
    """

    return (
        await asyncio_detailed(
            key_id=key_id,
            client=client,
        )
    ).parsed
