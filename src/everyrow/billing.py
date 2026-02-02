"""Billing utilities for everyrow SDK."""

from everyrow.api_utils import create_client
from everyrow.generated.api.billing import get_billing_balance_billing_get
from everyrow.generated.models import BillingResponse

# Re-export the generated type for public API
__all__ = ["BillingResponse", "get_billing_balance", "get_billing_balance_async"]


def get_billing_balance() -> BillingResponse:
    """Get the current billing balance for the authenticated user.

    Returns:
        BillingResponse: The user's current balance containing current_balance_dollars

    Raises:
        errors.UnexpectedStatus: If the request fails
        ValueError: If EVERYROW_API_KEY is not set
    """
    client = create_client()
    response = get_billing_balance_billing_get.sync(client=client)
    if response is None:
        raise RuntimeError("Failed to get billing balance")
    return response


async def get_billing_balance_async() -> BillingResponse:
    """Get the current billing balance for the authenticated user (async).

    Returns:
        BillingResponse: The user's current balance containing current_balance_dollars

    Raises:
        errors.UnexpectedStatus: If the request fails
        ValueError: If EVERYROW_API_KEY is not set
    """
    client = create_client()
    response = await get_billing_balance_billing_get.asyncio(client=client)
    if response is None:
        raise RuntimeError("Failed to get billing balance")
    return response
