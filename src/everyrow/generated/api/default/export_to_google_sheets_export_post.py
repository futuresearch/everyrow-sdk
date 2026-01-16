from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.export_request import ExportRequest
from ...models.export_to_google_sheets_export_post_response_export_to_google_sheets_export_post import (
    ExportToGoogleSheetsExportPostResponseExportToGoogleSheetsExportPost,
)
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: ExportRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/export",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ExportToGoogleSheetsExportPostResponseExportToGoogleSheetsExportPost | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = ExportToGoogleSheetsExportPostResponseExportToGoogleSheetsExportPost.from_dict(response.json())

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
) -> Response[ExportToGoogleSheetsExportPostResponseExportToGoogleSheetsExportPost | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: ExportRequest,
) -> Response[ExportToGoogleSheetsExportPostResponseExportToGoogleSheetsExportPost | HTTPValidationError]:
    """Export To Google Sheets

    Args:
        body (ExportRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ExportToGoogleSheetsExportPostResponseExportToGoogleSheetsExportPost | HTTPValidationError]
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
    body: ExportRequest,
) -> ExportToGoogleSheetsExportPostResponseExportToGoogleSheetsExportPost | HTTPValidationError | None:
    """Export To Google Sheets

    Args:
        body (ExportRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ExportToGoogleSheetsExportPostResponseExportToGoogleSheetsExportPost | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: ExportRequest,
) -> Response[ExportToGoogleSheetsExportPostResponseExportToGoogleSheetsExportPost | HTTPValidationError]:
    """Export To Google Sheets

    Args:
        body (ExportRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ExportToGoogleSheetsExportPostResponseExportToGoogleSheetsExportPost | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: ExportRequest,
) -> ExportToGoogleSheetsExportPostResponseExportToGoogleSheetsExportPost | HTTPValidationError | None:
    """Export To Google Sheets

    Args:
        body (ExportRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ExportToGoogleSheetsExportPostResponseExportToGoogleSheetsExportPost | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
