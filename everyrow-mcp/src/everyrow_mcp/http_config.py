"""HTTP mode configuration for the everyrow MCP server."""

from __future__ import annotations

import logging
import sys
from typing import Any
from urllib.parse import urlparse

from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import lifespan_wrapper
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import AnyHttpUrl
from redis.asyncio import Redis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from everyrow_mcp.auth import EveryRowAuthProvider, SupabaseTokenVerifier
from everyrow_mcp.config import DevHttpSettings, HttpSettings
from everyrow_mcp.middleware import RateLimitMiddleware
from everyrow_mcp.redis_utils import create_redis_client
from everyrow_mcp.routes import api_download, api_progress
from everyrow_mcp.state import RedisStore, state
from everyrow_mcp.templates import RESULTS_HTML, SESSION_HTML

logger = logging.getLogger(__name__)


def configure_http_mode(
    mcp: FastMCP,
    http_lifespan: Any,
    host: str,
    port: int,
    *,
    no_auth: bool = False,
) -> None:
    """Configure the MCP server for HTTP transport."""
    state.transport = "streamable-http"

    if no_auth:
        logger.warning("Running in --no-auth mode (development only)")
        settings, redis_client = _load_settings_and_redis(DevHttpSettings)
        auth_provider = None
        mcp_server_url = f"http://localhost:{port}"
    else:
        settings, redis_client = _load_settings_and_redis(HttpSettings, sentinel=True)
        auth_provider = EveryRowAuthProvider(
            supabase_url=settings.supabase_url,
            supabase_anon_key=settings.supabase_anon_key,
            mcp_server_url=settings.mcp_server_url,
            redis=redis_client,
        )
        state.auth_provider = auth_provider
        verifier = SupabaseTokenVerifier(settings.supabase_url, redis=redis_client)
        _configure_mcp_auth(mcp, settings, auth_provider, verifier)
        mcp_server_url = settings.mcp_server_url

    _configure_shared(
        mcp,
        http_lifespan,
        host,
        port,
        mcp_server_url,
        redis_client=redis_client,
        no_auth=no_auth,
        auth_provider=auth_provider,
    )


# ── Internal helpers ─────────────────────────────────────────────────


def _load_settings_and_redis(
    settings_cls: type,
    *,
    sentinel: bool = False,
) -> tuple[Any, Redis]:
    """Load settings from env and create a Redis client."""
    try:
        settings = settings_cls()  # pyright: ignore[reportCallIssue]
    except Exception as e:
        logging.error(f"HTTP mode configuration error: {e}")
        sys.exit(1)

    redis_client = create_redis_client(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
        sentinel_endpoints=getattr(settings, "redis_sentinel_endpoints", None)
        if sentinel
        else None,
        sentinel_master_name=getattr(settings, "redis_sentinel_master_name", None)
        if sentinel
        else None,
    )

    state.settings = settings
    state.store = RedisStore(redis_client)
    return settings, redis_client


def _configure_mcp_auth(
    mcp: FastMCP,
    settings: HttpSettings,
    auth_provider: EveryRowAuthProvider,
    verifier: SupabaseTokenVerifier,
) -> None:
    """Wire OAuth provider and JWT verifier into FastMCP."""
    mcp._auth_server_provider = auth_provider  # type: ignore[arg-type]
    mcp._token_verifier = verifier
    mcp.settings.auth = AuthSettings(
        issuer_url=AnyHttpUrl(settings.mcp_server_url),
        resource_server_url=AnyHttpUrl(settings.mcp_server_url),
        client_registration_options=ClientRegistrationOptions(enabled=True),
    )
    hostname = urlparse(settings.mcp_server_url).hostname or "localhost"
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[hostname],
    )
    logger.warning(
        "Auth configured: issuer=%s resource_server=%s allowed_hosts=%s",
        mcp.settings.auth.issuer_url,
        mcp.settings.auth.resource_server_url,
        mcp.settings.transport_security.allowed_hosts,
    )


def _ui_csp(connect_domains: list[str]) -> dict:
    """Build a CSP policy for MCP App widgets."""
    return {
        "resourceDomains": ["https://unpkg.com"],
        "connectDomains": connect_domains,
    }


def _configure_shared(
    mcp: FastMCP,
    http_lifespan: Any,
    host: str,
    port: int,
    mcp_server_url: str,
    *,
    redis_client: Redis,
    no_auth: bool = False,
    auth_provider: EveryRowAuthProvider | None = None,
) -> None:
    """Configuration shared between auth and no-auth HTTP modes."""
    mcp._mcp_server.lifespan = lifespan_wrapper(mcp, http_lifespan)
    mcp.settings.host = host
    mcp.settings.port = port
    state.mcp_server_url = mcp_server_url

    widget_csp = _ui_csp([mcp_server_url])

    # Patch tool meta to include CSP
    for tool_name in ("everyrow_progress", "everyrow_results"):
        tool = mcp._tool_manager._tools.get(tool_name)
        if tool and tool.meta and "ui" in tool.meta:
            tool.meta["ui"]["csp"] = widget_csp

    @mcp.resource(
        "ui://everyrow/session.html",
        mime_type="text/html;profile=mcp-app",
        meta={"ui": {"csp": widget_csp}},
    )
    def _session_ui_http() -> str:
        return SESSION_HTML

    @mcp.resource(
        "ui://everyrow/results.html",
        mime_type="text/html;profile=mcp-app",
        meta={"ui": {"csp": widget_csp}},
    )
    def _results_ui_http() -> str:
        return RESULTS_HTML

    # REST routes
    mcp.custom_route("/api/progress/{task_id}", ["GET", "OPTIONS"])(api_progress)
    mcp.custom_route("/api/results/{task_id}/download", ["GET", "OPTIONS"])(
        api_download
    )

    async def _health(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    mcp.custom_route("/health", ["GET"])(_health)

    if auth_provider is not None:
        mcp.custom_route("/auth/start/{state}", ["GET"])(auth_provider.handle_start)
        mcp.custom_route("/auth/callback", ["GET"])(auth_provider.handle_callback)

    # Middleware
    _add_middleware(mcp, redis_client, rate_limit=not no_auth)


def _add_middleware(
    mcp: FastMCP,
    redis_client: Redis,
    *,
    rate_limit: bool = True,
) -> None:
    """Wrap the ASGI app with request logging and optional rate limiting."""
    _original = mcp.streamable_http_app

    def _wrapped():
        app = _original()

        class RequestLoggingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                has_auth = "authorization" in request.headers
                logging.debug(
                    "INCOMING %s %s | Host: %s | Auth: %s",
                    request.method,
                    request.url.path,
                    request.headers.get("host", "?"),
                    "present" if has_auth else "none",
                )
                response = await call_next(request)
                logging.debug(
                    "RESPONSE %s %s -> %s",
                    request.method,
                    request.url.path,
                    response.status_code,
                )
                return response

        app.add_middleware(RequestLoggingMiddleware)

        if rate_limit:
            app.add_middleware(RateLimitMiddleware, redis=redis_client)

        return app

    mcp.streamable_http_app = _wrapped
