"""Billing utilities for checking account balance."""

import os

import httpx

from everyrow.constants import DEFAULT_EVERYROW_API_URL


def get_billing_balance() -> float:
    """Get the current billing balance for the authenticated user.

    Returns:
        The current balance in dollars.

    Raises:
        ValueError: If EVERYROW_API_KEY is not set.
        httpx.HTTPStatusError: If the API request fails.
    """
    api_key = os.environ.get("EVERYROW_API_KEY")
    if not api_key:
        raise ValueError("EVERYROW_API_KEY environment variable is not set")

    base_url = os.environ.get("EVERYROW_API_URL", DEFAULT_EVERYROW_API_URL)
    response = httpx.get(
        f"{base_url}/billing",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    return response.json()["current_balance_dollars"]
