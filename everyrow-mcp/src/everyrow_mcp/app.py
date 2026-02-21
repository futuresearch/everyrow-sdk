"""FastMCP application instance, lifespans, and resource handlers."""

import logging
from contextlib import asynccontextmanager

from everyrow.api_utils import create_client as _create_sdk_client
from everyrow.generated.api.billing.get_billing_balance_billing_get import (
    asyncio as get_billing,
)
from mcp.server.fastmcp import FastMCP

from everyrow_mcp.state import TASK_STATE_FILE, state
from everyrow_mcp.templates import PROGRESS_HTML, UI_CSP_META
from everyrow_mcp.tool_helpers import (
    make_http_auth_client_factory,
    make_singleton_client_factory,
)


def _clear_task_state() -> None:
    if state.is_http:
        return
    if TASK_STATE_FILE.exists():
        TASK_STATE_FILE.unlink()


@asynccontextmanager
async def _stdio_lifespan(_server: FastMCP):
    """Initialize singleton client and validate credentials on startup (stdio mode)."""
    _clear_task_state()

    try:
        with _create_sdk_client() as client:
            response = await get_billing(client=client)
            if response is None:
                raise RuntimeError("Failed to authenticate with everyrow API")
            yield make_singleton_client_factory(client)
    except Exception as e:
        logging.getLogger(__name__).error(f"everyrow-mcp startup failed: {e!r}")
        raise
    finally:
        _clear_task_state()


@asynccontextmanager
async def _http_lifespan(_server: FastMCP):
    """HTTP mode lifespan — verify Redis on startup.

    NOTE: This runs per MCP *session*, not per server. Do NOT close
    shared resources (auth_provider, Redis) here — they must survive
    across sessions. Process exit handles cleanup.
    """
    log = logging.getLogger(__name__)
    await state.store.ping()
    log.info("Redis health check passed")
    yield make_http_auth_client_factory()


@asynccontextmanager
async def _no_auth_http_lifespan(_server: FastMCP):
    """HTTP no-auth mode: singleton client from API key, verify Redis."""
    await state.store.ping()
    with _create_sdk_client() as client:
        response = await get_billing(client=client)
        if response is None:
            raise RuntimeError("Failed to authenticate with everyrow API")
        yield make_singleton_client_factory(client)


mcp = FastMCP("everyrow_mcp", lifespan=_stdio_lifespan)


@mcp.resource(
    "ui://everyrow/progress.html",
    mime_type="text/html;profile=mcp-app",
    meta=UI_CSP_META,
)
def _progress_ui() -> str:
    return PROGRESS_HTML


# NOTE: results.html and session.html are registered in http_config.py
# (with HTTP-aware CSP) when running in HTTP mode. Only progress.html
# is registered here because it's used in both transport modes.
