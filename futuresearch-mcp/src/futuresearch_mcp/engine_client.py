from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import cache
from importlib.metadata import version

import httpx
from futuresearch.generated.client import AuthenticatedClient
from mcp.server.auth.middleware.auth_context import get_access_token

from futuresearch_mcp import redis_store
from futuresearch_mcp.config import settings
from futuresearch_mcp.request_context import get_cohort_account_id

logger = logging.getLogger(__name__)

ACCOUNT_ID_HEADER = "x-cohort-account-id"

# Act as whichever account the credential itself resolves to.
NO_ACCOUNT_SELECTED = ""


@cache
def _sdk_version() -> str:
    return f"futuresearch-python/{version('futuresearch')}"


def _engine_headers(
    account_id: str, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    headers = {"X-SDK-Version": _sdk_version()}
    if extra:
        headers.update(extra)
    if account_id:
        headers[ACCOUNT_ID_HEADER] = account_id
    return headers


def build_engine_client(
    *,
    token: str,
    account_id: str,
    extra_headers: Mapping[str, str] | None = None,
) -> AuthenticatedClient:
    """Build an Engine API client acting as ``account_id``."""
    return AuthenticatedClient(
        base_url=settings.futuresearch_api_url,
        token=token,
        headers=_engine_headers(account_id, extra_headers),
        raise_on_unexpected_status=True,
        follow_redirects=True,
    )


def engine_httpx_client(
    *,
    token: str,
    account_id: str,
    timeout: httpx.Timeout | float | None = None,
) -> httpx.AsyncClient:
    """Build a raw httpx client for the Engine API."""
    return httpx.AsyncClient(
        base_url=settings.futuresearch_api_url,
        headers=_engine_headers(account_id, {"Authorization": f"Bearer {token}"}),
        timeout=timeout if timeout is not None else httpx.Timeout(10.0),
    )


async def resolve_account_id() -> str:
    """The account the current request should act as.

    An inbound header wins over the login-time selection.
    """
    if inbound := get_cohort_account_id():
        return inbound
    return await _stored_account_id()


async def _stored_account_id() -> str:
    """Resolve the login-time account selection for the current connection.

    Keyed on the access token, so the selection is scoped to one connection.
    """
    if not settings.is_http:
        return NO_ACCOUNT_SELECTED
    try:
        access_token = get_access_token()
    except Exception:
        logger.warning("Failed to get access token", exc_info=True)
        return NO_ACCOUNT_SELECTED
    if access_token is None:
        return NO_ACCOUNT_SELECTED
    try:
        return await redis_store.get_account_selection(access_token.token) or ""
    except Exception:
        logger.warning("Account selection lookup failed", exc_info=True)
        return NO_ACCOUNT_SELECTED
