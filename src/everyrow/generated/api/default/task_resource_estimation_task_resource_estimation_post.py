from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.concatenate_request import ConcatenateRequest
from ...models.create_group_request import CreateGroupRequest
from ...models.create_request import CreateRequest
from ...models.dedupe_request_params import DedupeRequestParams
from ...models.deep_merge_request import DeepMergeRequest
from ...models.deep_rank_request import DeepRankRequest
from ...models.deep_screen_request import DeepScreenRequest
from ...models.derive_request import DeriveRequest
from ...models.drop_columns_request import DropColumnsRequest
from ...models.filter_request import FilterRequest
from ...models.flatten_request import FlattenRequest
from ...models.group_by_request import GroupByRequest
from ...models.http_validation_error import HTTPValidationError
from ...models.join_request import JoinRequest
from ...models.map_agent_request_params import MapAgentRequestParams
from ...models.map_multi_agent_request_params import MapMultiAgentRequestParams
from ...models.reduce_agent_request_params import ReduceAgentRequestParams
from ...models.reduce_multi_agent_request_params import ReduceMultiAgentRequestParams
from ...models.resource_estimation_response import ResourceEstimationResponse
from ...models.upload_csv_payload import UploadCsvPayload
from ...types import Response


def _get_kwargs(
    *,
    body: ConcatenateRequest
    | CreateGroupRequest
    | CreateRequest
    | DedupeRequestParams
    | DeepMergeRequest
    | DeepRankRequest
    | DeepScreenRequest
    | DeriveRequest
    | DropColumnsRequest
    | FilterRequest
    | FlattenRequest
    | GroupByRequest
    | JoinRequest
    | MapAgentRequestParams
    | MapMultiAgentRequestParams
    | ReduceAgentRequestParams
    | ReduceMultiAgentRequestParams
    | UploadCsvPayload,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/task-resource-estimation",
    }

    if isinstance(body, MapAgentRequestParams):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, ReduceAgentRequestParams):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, FilterRequest):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, DeriveRequest):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, JoinRequest):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, ConcatenateRequest):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, DropColumnsRequest):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, DedupeRequestParams):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, UploadCsvPayload):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, MapMultiAgentRequestParams):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, ReduceMultiAgentRequestParams):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, CreateRequest):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, CreateGroupRequest):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, FlattenRequest):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, GroupByRequest):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, DeepRankRequest):
        _kwargs["json"] = body.to_dict()
    elif isinstance(body, DeepMergeRequest):
        _kwargs["json"] = body.to_dict()
    else:
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ResourceEstimationResponse | None:
    if response.status_code == 200:
        response_200 = ResourceEstimationResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | ResourceEstimationResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: ConcatenateRequest
    | CreateGroupRequest
    | CreateRequest
    | DedupeRequestParams
    | DeepMergeRequest
    | DeepRankRequest
    | DeepScreenRequest
    | DeriveRequest
    | DropColumnsRequest
    | FilterRequest
    | FlattenRequest
    | GroupByRequest
    | JoinRequest
    | MapAgentRequestParams
    | MapMultiAgentRequestParams
    | ReduceAgentRequestParams
    | ReduceMultiAgentRequestParams
    | UploadCsvPayload,
) -> Response[HTTPValidationError | ResourceEstimationResponse]:
    """Task Resource Estimation

    Args:
        body (ConcatenateRequest | CreateGroupRequest | CreateRequest | DedupeRequestParams |
            DeepMergeRequest | DeepRankRequest | DeepScreenRequest | DeriveRequest |
            DropColumnsRequest | FilterRequest | FlattenRequest | GroupByRequest | JoinRequest |
            MapAgentRequestParams | MapMultiAgentRequestParams | ReduceAgentRequestParams |
            ReduceMultiAgentRequestParams | UploadCsvPayload):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ResourceEstimationResponse]
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
    body: ConcatenateRequest
    | CreateGroupRequest
    | CreateRequest
    | DedupeRequestParams
    | DeepMergeRequest
    | DeepRankRequest
    | DeepScreenRequest
    | DeriveRequest
    | DropColumnsRequest
    | FilterRequest
    | FlattenRequest
    | GroupByRequest
    | JoinRequest
    | MapAgentRequestParams
    | MapMultiAgentRequestParams
    | ReduceAgentRequestParams
    | ReduceMultiAgentRequestParams
    | UploadCsvPayload,
) -> HTTPValidationError | ResourceEstimationResponse | None:
    """Task Resource Estimation

    Args:
        body (ConcatenateRequest | CreateGroupRequest | CreateRequest | DedupeRequestParams |
            DeepMergeRequest | DeepRankRequest | DeepScreenRequest | DeriveRequest |
            DropColumnsRequest | FilterRequest | FlattenRequest | GroupByRequest | JoinRequest |
            MapAgentRequestParams | MapMultiAgentRequestParams | ReduceAgentRequestParams |
            ReduceMultiAgentRequestParams | UploadCsvPayload):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ResourceEstimationResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: ConcatenateRequest
    | CreateGroupRequest
    | CreateRequest
    | DedupeRequestParams
    | DeepMergeRequest
    | DeepRankRequest
    | DeepScreenRequest
    | DeriveRequest
    | DropColumnsRequest
    | FilterRequest
    | FlattenRequest
    | GroupByRequest
    | JoinRequest
    | MapAgentRequestParams
    | MapMultiAgentRequestParams
    | ReduceAgentRequestParams
    | ReduceMultiAgentRequestParams
    | UploadCsvPayload,
) -> Response[HTTPValidationError | ResourceEstimationResponse]:
    """Task Resource Estimation

    Args:
        body (ConcatenateRequest | CreateGroupRequest | CreateRequest | DedupeRequestParams |
            DeepMergeRequest | DeepRankRequest | DeepScreenRequest | DeriveRequest |
            DropColumnsRequest | FilterRequest | FlattenRequest | GroupByRequest | JoinRequest |
            MapAgentRequestParams | MapMultiAgentRequestParams | ReduceAgentRequestParams |
            ReduceMultiAgentRequestParams | UploadCsvPayload):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ResourceEstimationResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: ConcatenateRequest
    | CreateGroupRequest
    | CreateRequest
    | DedupeRequestParams
    | DeepMergeRequest
    | DeepRankRequest
    | DeepScreenRequest
    | DeriveRequest
    | DropColumnsRequest
    | FilterRequest
    | FlattenRequest
    | GroupByRequest
    | JoinRequest
    | MapAgentRequestParams
    | MapMultiAgentRequestParams
    | ReduceAgentRequestParams
    | ReduceMultiAgentRequestParams
    | UploadCsvPayload,
) -> HTTPValidationError | ResourceEstimationResponse | None:
    """Task Resource Estimation

    Args:
        body (ConcatenateRequest | CreateGroupRequest | CreateRequest | DedupeRequestParams |
            DeepMergeRequest | DeepRankRequest | DeepScreenRequest | DeriveRequest |
            DropColumnsRequest | FilterRequest | FlattenRequest | GroupByRequest | JoinRequest |
            MapAgentRequestParams | MapMultiAgentRequestParams | ReduceAgentRequestParams |
            ReduceMultiAgentRequestParams | UploadCsvPayload):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ResourceEstimationResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
