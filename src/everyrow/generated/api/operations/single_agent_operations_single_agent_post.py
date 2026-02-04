from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.insufficient_balance_error import InsufficientBalanceError
from ...models.operation_response import OperationResponse
from ...models.single_agent_operation import SingleAgentOperation
from ...types import Response


def _get_kwargs(
    *,
    body: SingleAgentOperation,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/operations/single-agent",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | InsufficientBalanceError | OperationResponse | None:
    if response.status_code == 200:
        response_200 = OperationResponse.from_dict(response.json())

        return response_200

    if response.status_code == 402:
        response_402 = InsufficientBalanceError.from_dict(response.json())

        return response_402

    if response.status_code == 422:
        response_422 = ErrorResponse.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ErrorResponse | InsufficientBalanceError | OperationResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: SingleAgentOperation,
) -> Response[ErrorResponse | InsufficientBalanceError | OperationResponse]:
    """Single AI research agent

     Run a single AI agent to perform research and generate a response.

    **Configuration options** (mutually exclusive):

    1. **Use a preset** - set `effort_level` to one of:
       - `low`: Fast, minimal research (Gemini Flash, 0 iterations, no provenance)
       - `medium`: Balanced (Gemini Flash High, 5 iterations, with provenance)
       - `high`: Thorough research (Claude Opus, 10 iterations, with provenance)

    2. **Fully customize** - set `effort_level=null` and provide ALL of:
       - `llm`: The LLM model to use
       - `iteration_budget`: Number of agent iterations (0-20)
       - `include_research`: Whether to include research notes

    You cannot mix these approaches - either use a preset OR specify all custom parameters.

    Args:
        body (SingleAgentOperation):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | InsufficientBalanceError | OperationResponse]
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
    body: SingleAgentOperation,
) -> ErrorResponse | InsufficientBalanceError | OperationResponse | None:
    """Single AI research agent

     Run a single AI agent to perform research and generate a response.

    **Configuration options** (mutually exclusive):

    1. **Use a preset** - set `effort_level` to one of:
       - `low`: Fast, minimal research (Gemini Flash, 0 iterations, no provenance)
       - `medium`: Balanced (Gemini Flash High, 5 iterations, with provenance)
       - `high`: Thorough research (Claude Opus, 10 iterations, with provenance)

    2. **Fully customize** - set `effort_level=null` and provide ALL of:
       - `llm`: The LLM model to use
       - `iteration_budget`: Number of agent iterations (0-20)
       - `include_research`: Whether to include research notes

    You cannot mix these approaches - either use a preset OR specify all custom parameters.

    Args:
        body (SingleAgentOperation):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | InsufficientBalanceError | OperationResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: SingleAgentOperation,
) -> Response[ErrorResponse | InsufficientBalanceError | OperationResponse]:
    """Single AI research agent

     Run a single AI agent to perform research and generate a response.

    **Configuration options** (mutually exclusive):

    1. **Use a preset** - set `effort_level` to one of:
       - `low`: Fast, minimal research (Gemini Flash, 0 iterations, no provenance)
       - `medium`: Balanced (Gemini Flash High, 5 iterations, with provenance)
       - `high`: Thorough research (Claude Opus, 10 iterations, with provenance)

    2. **Fully customize** - set `effort_level=null` and provide ALL of:
       - `llm`: The LLM model to use
       - `iteration_budget`: Number of agent iterations (0-20)
       - `include_research`: Whether to include research notes

    You cannot mix these approaches - either use a preset OR specify all custom parameters.

    Args:
        body (SingleAgentOperation):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | InsufficientBalanceError | OperationResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)
    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: SingleAgentOperation,
) -> ErrorResponse | InsufficientBalanceError | OperationResponse | None:
    """Single AI research agent

     Run a single AI agent to perform research and generate a response.

    **Configuration options** (mutually exclusive):

    1. **Use a preset** - set `effort_level` to one of:
       - `low`: Fast, minimal research (Gemini Flash, 0 iterations, no provenance)
       - `medium`: Balanced (Gemini Flash High, 5 iterations, with provenance)
       - `high`: Thorough research (Claude Opus, 10 iterations, with provenance)

    2. **Fully customize** - set `effort_level=null` and provide ALL of:
       - `llm`: The LLM model to use
       - `iteration_budget`: Number of agent iterations (0-20)
       - `include_research`: Whether to include research notes

    You cannot mix these approaches - either use a preset OR specify all custom parameters.

    Args:
        body (SingleAgentOperation):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | InsufficientBalanceError | OperationResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
