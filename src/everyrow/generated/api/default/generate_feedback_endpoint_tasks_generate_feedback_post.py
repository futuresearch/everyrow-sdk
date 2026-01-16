from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.controller_improvement_round import ControllerImprovementRound
from ...models.generate_feedback_request import GenerateFeedbackRequest
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: GenerateFeedbackRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/tasks/generate_feedback",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ControllerImprovementRound | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = ControllerImprovementRound.from_dict(response.json())

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
) -> Response[ControllerImprovementRound | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: GenerateFeedbackRequest,
) -> Response[ControllerImprovementRound | HTTPValidationError]:
    """Generate Feedback Endpoint

     Generate controller feedback for a completed MAP task.

    This endpoint invokes a Celery task that calls generate_feedback_for_map_task
    with the provided task_id and default values for other parameters.

    Args:
        body (GenerateFeedbackRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ControllerImprovementRound | HTTPValidationError]
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
    body: GenerateFeedbackRequest,
) -> ControllerImprovementRound | HTTPValidationError | None:
    """Generate Feedback Endpoint

     Generate controller feedback for a completed MAP task.

    This endpoint invokes a Celery task that calls generate_feedback_for_map_task
    with the provided task_id and default values for other parameters.

    Args:
        body (GenerateFeedbackRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ControllerImprovementRound | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: GenerateFeedbackRequest,
) -> Response[ControllerImprovementRound | HTTPValidationError]:
    """Generate Feedback Endpoint

     Generate controller feedback for a completed MAP task.

    This endpoint invokes a Celery task that calls generate_feedback_for_map_task
    with the provided task_id and default values for other parameters.

    Args:
        body (GenerateFeedbackRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ControllerImprovementRound | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: GenerateFeedbackRequest,
) -> ControllerImprovementRound | HTTPValidationError | None:
    """Generate Feedback Endpoint

     Generate controller feedback for a completed MAP task.

    This endpoint invokes a Celery task that calls generate_feedback_for_map_task
    with the provided task_id and default values for other parameters.

    Args:
        body (GenerateFeedbackRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ControllerImprovementRound | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
