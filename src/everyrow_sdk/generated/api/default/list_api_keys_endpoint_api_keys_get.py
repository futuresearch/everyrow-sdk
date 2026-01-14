from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.api_key_info import APIKeyInfo
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    include_revoked: bool | Unset = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["include_revoked"] = include_revoked

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/api-keys",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | list[APIKeyInfo] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = APIKeyInfo.from_dict(response_200_item_data)

            response_200.append(response_200_item)

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
) -> Response[HTTPValidationError | list[APIKeyInfo]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    include_revoked: bool | Unset = False,
) -> Response[HTTPValidationError | list[APIKeyInfo]]:
    """List Api Keys Endpoint

     List all API keys for the current user.

    By default, only active (non-revoked) keys are returned.

    Args:
        include_revoked (bool | Unset):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | list[APIKeyInfo]]
    """

    kwargs = _get_kwargs(
        include_revoked=include_revoked,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    include_revoked: bool | Unset = False,
) -> HTTPValidationError | list[APIKeyInfo] | None:
    """List Api Keys Endpoint

     List all API keys for the current user.

    By default, only active (non-revoked) keys are returned.

    Args:
        include_revoked (bool | Unset):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | list[APIKeyInfo]
    """

    return sync_detailed(
        client=client,
        include_revoked=include_revoked,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    include_revoked: bool | Unset = False,
) -> Response[HTTPValidationError | list[APIKeyInfo]]:
    """List Api Keys Endpoint

     List all API keys for the current user.

    By default, only active (non-revoked) keys are returned.

    Args:
        include_revoked (bool | Unset):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | list[APIKeyInfo]]
    """

    kwargs = _get_kwargs(
        include_revoked=include_revoked,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    include_revoked: bool | Unset = False,
) -> HTTPValidationError | list[APIKeyInfo] | None:
    """List Api Keys Endpoint

     List all API keys for the current user.

    By default, only active (non-revoked) keys are returned.

    Args:
        include_revoked (bool | Unset):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | list[APIKeyInfo]
    """

    return (
        await asyncio_detailed(
            client=client,
            include_revoked=include_revoked,
        )
    ).parsed
