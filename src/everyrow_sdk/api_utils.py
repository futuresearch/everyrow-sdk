import os
from typing import TypeVar

from everyrow_sdk.constants import DEFAULT_EVERYROW_API_URL, EveryrowError
from everyrow_sdk.generated.client import AuthenticatedClient
from everyrow_sdk.generated.models.http_validation_error import HTTPValidationError
from everyrow_sdk.generated.models.insufficient_balance_error import (
    InsufficientBalanceError,
)


def create_client() -> AuthenticatedClient:
    """Create an AuthenticatedClient from environment variables.

    Reads EVERYROW_API_KEY and EVERYROW_API_URL from environment variables.

    Returns:
        AuthenticatedClient: A configured client instance

    Raises:
        ValueError: If EVERYROW_API_KEY is not set in environment
    """
    if "EVERYROW_API_KEY" not in os.environ:
        raise ValueError("EVERYROW_API_KEY is not set; cannot initialize client")
    return AuthenticatedClient(
        base_url=os.environ.get("EVERYROW_API_URL", DEFAULT_EVERYROW_API_URL),
        token=os.environ["EVERYROW_API_KEY"],
        raise_on_unexpected_status=True,
    )


T = TypeVar("T")


def handle_response[T](
    response: T | HTTPValidationError | InsufficientBalanceError | None,
) -> T:
    if isinstance(response, HTTPValidationError):
        raise EveryrowError(response.detail)
    if isinstance(response, InsufficientBalanceError):
        raise EveryrowError(response.message)
    if response is None:
        raise EveryrowError("Unknown error")

    return response
