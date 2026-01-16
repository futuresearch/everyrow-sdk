from http import HTTPStatus
from typing import Any
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.artifact_group_record import ArtifactGroupRecord
from ...models.http_validation_error import HTTPValidationError
from ...models.standalone_artifact_record import StandaloneArtifactRecord
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    artifact_ids: list[UUID],
    include_failed: bool | Unset = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_artifact_ids = []
    for artifact_ids_item_data in artifact_ids:
        artifact_ids_item = str(artifact_ids_item_data)
        json_artifact_ids.append(artifact_ids_item)

    params["artifact_ids"] = json_artifact_ids

    params["include_failed"] = include_failed

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/artifacts",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | list[ArtifactGroupRecord | StandaloneArtifactRecord] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:

            def _parse_response_200_item(data: object) -> ArtifactGroupRecord | StandaloneArtifactRecord:
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    response_200_item_type_0 = StandaloneArtifactRecord.from_dict(data)

                    return response_200_item_type_0
                except (TypeError, ValueError, AttributeError, KeyError):
                    pass
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_item_type_1 = ArtifactGroupRecord.from_dict(data)

                return response_200_item_type_1

            response_200_item = _parse_response_200_item(response_200_item_data)

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
) -> Response[HTTPValidationError | list[ArtifactGroupRecord | StandaloneArtifactRecord]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    artifact_ids: list[UUID],
    include_failed: bool | Unset = False,
) -> Response[HTTPValidationError | list[ArtifactGroupRecord | StandaloneArtifactRecord]]:
    """Get Artifacts

     Get artifacts by artifact ID.

    Caution: This is not protected by RLS, so it will return the artifacts
    as long as the IDs are valid. The reason for it is that we would need to handle
    case for both authenticated and non-authenticated users similarly to how we handle
    stuff on the frontend, and that doesn't seem to be worth it at the moment.

    Alternative would be ditching this whole endpoint and replicating fetching the
    same artifacts in the frontend from the DB directly. It would be probably much
    faster and simpler anyway, with only slight duplicity of the code between python
    and typescript (as we need to be able to fetch artifacts here for the various
    workflow purposes).

    Args:
        artifact_ids (list[UUID]):
        include_failed (bool | Unset):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | list[ArtifactGroupRecord | StandaloneArtifactRecord]]
    """

    kwargs = _get_kwargs(
        artifact_ids=artifact_ids,
        include_failed=include_failed,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    artifact_ids: list[UUID],
    include_failed: bool | Unset = False,
) -> HTTPValidationError | list[ArtifactGroupRecord | StandaloneArtifactRecord] | None:
    """Get Artifacts

     Get artifacts by artifact ID.

    Caution: This is not protected by RLS, so it will return the artifacts
    as long as the IDs are valid. The reason for it is that we would need to handle
    case for both authenticated and non-authenticated users similarly to how we handle
    stuff on the frontend, and that doesn't seem to be worth it at the moment.

    Alternative would be ditching this whole endpoint and replicating fetching the
    same artifacts in the frontend from the DB directly. It would be probably much
    faster and simpler anyway, with only slight duplicity of the code between python
    and typescript (as we need to be able to fetch artifacts here for the various
    workflow purposes).

    Args:
        artifact_ids (list[UUID]):
        include_failed (bool | Unset):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | list[ArtifactGroupRecord | StandaloneArtifactRecord]
    """

    return sync_detailed(
        client=client,
        artifact_ids=artifact_ids,
        include_failed=include_failed,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    artifact_ids: list[UUID],
    include_failed: bool | Unset = False,
) -> Response[HTTPValidationError | list[ArtifactGroupRecord | StandaloneArtifactRecord]]:
    """Get Artifacts

     Get artifacts by artifact ID.

    Caution: This is not protected by RLS, so it will return the artifacts
    as long as the IDs are valid. The reason for it is that we would need to handle
    case for both authenticated and non-authenticated users similarly to how we handle
    stuff on the frontend, and that doesn't seem to be worth it at the moment.

    Alternative would be ditching this whole endpoint and replicating fetching the
    same artifacts in the frontend from the DB directly. It would be probably much
    faster and simpler anyway, with only slight duplicity of the code between python
    and typescript (as we need to be able to fetch artifacts here for the various
    workflow purposes).

    Args:
        artifact_ids (list[UUID]):
        include_failed (bool | Unset):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | list[ArtifactGroupRecord | StandaloneArtifactRecord]]
    """

    kwargs = _get_kwargs(
        artifact_ids=artifact_ids,
        include_failed=include_failed,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    artifact_ids: list[UUID],
    include_failed: bool | Unset = False,
) -> HTTPValidationError | list[ArtifactGroupRecord | StandaloneArtifactRecord] | None:
    """Get Artifacts

     Get artifacts by artifact ID.

    Caution: This is not protected by RLS, so it will return the artifacts
    as long as the IDs are valid. The reason for it is that we would need to handle
    case for both authenticated and non-authenticated users similarly to how we handle
    stuff on the frontend, and that doesn't seem to be worth it at the moment.

    Alternative would be ditching this whole endpoint and replicating fetching the
    same artifacts in the frontend from the DB directly. It would be probably much
    faster and simpler anyway, with only slight duplicity of the code between python
    and typescript (as we need to be able to fetch artifacts here for the various
    workflow purposes).

    Args:
        artifact_ids (list[UUID]):
        include_failed (bool | Unset):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | list[ArtifactGroupRecord | StandaloneArtifactRecord]
    """

    return (
        await asyncio_detailed(
            client=client,
            artifact_ids=artifact_ids,
            include_failed=include_failed,
        )
    ).parsed
