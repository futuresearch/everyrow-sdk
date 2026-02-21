"""HTTP mode configuration for the everyrow MCP server."""

from __future__ import annotations

import logging
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

from everyrow_mcp.app import _http_lifespan, _no_auth_http_lifespan
from everyrow_mcp.auth import EveryRowAuthProvider, SupabaseTokenVerifier
from everyrow_mcp.config import _get_dev_http_settings, _get_http_settings
from everyrow_mcp.middleware import RateLimitMiddleware
from everyrow_mcp.redis_utils import create_redis_client
from everyrow_mcp.routes import api_download, api_progress
from everyrow_mcp.state import RedisStore, Transport, state
from everyrow_mcp.templates import RESULTS_HTML, SESSION_HTML

logger = logging.getLogger(__name__)


def configure_http_mode(
    mcp: FastMCP,
    host: str,
    port: int,
    *,
    no_auth: bool = False,
) -> None:
    """Configure the MCP server for HTTP transport."""
    state.transport = Transport.HTTP
    state.dev_mode = no_auth

    if no_auth:
        settings = _get_dev_http_settings()
        mcp_server_url = f"http://localhost:{port}"
        lifespan = _no_auth_http_lifespan
    else:
        settings = _get_http_settings()
        mcp_server_url = settings.mcp_server_url
        lifespan = _http_lifespan

    redis_client = create_redis_client(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
        sentinel_endpoints=getattr(settings, "redis_sentinel_endpoints", None),
        sentinel_master_name=getattr(settings, "redis_sentinel_master_name", None),
    )
    state.store = RedisStore(redis_client)
    state.everyrow_api_url = settings.everyrow_api_url
    state.preview_size = settings.preview_size
    state.mcp_server_url = mcp_server_url

    if not no_auth:
        verifier = SupabaseTokenVerifier(settings.supabase_url, redis=redis_client)
        auth_provider = EveryRowAuthProvider(
            redis=redis_client,
            token_verifier=verifier,
        )
        state.auth_provider = auth_provider
        _configure_mcp_auth(mcp, auth_provider, verifier)
    else:
        auth_provider = None

    mcp._mcp_server.lifespan = lifespan_wrapper(mcp, lifespan)
    mcp.settings.host = host
    mcp.settings.port = port

    widget_csp = _ui_csp([mcp_server_url])
    _patch_tool_csp(mcp, widget_csp)

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


def _configure_mcp_auth(
    mcp: FastMCP,
    auth_provider: EveryRowAuthProvider,
    verifier: SupabaseTokenVerifier,
) -> None:
    """Wire OAuth provider and JWT verifier into FastMCP."""
    settings = _get_http_settings()
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


def _patch_tool_csp(mcp: FastMCP, csp: dict) -> None:
    """Patch CSP policy onto tool metadata for MCP App widgets."""
    for tool_name in ("everyrow_progress", "everyrow_results"):
        tool = mcp._tool_manager._tools.get(tool_name)
        if tool and tool.meta and "ui" in tool.meta:
            tool.meta["ui"]["csp"] = csp


def _ui_csp(connect_domains: list[str]) -> dict:
    """Build a CSP policy for MCP App widgets."""
    return {
        "resourceDomains": ["https://unpkg.com"],
        "connectDomains": connect_domains,
    }


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

        if rate_limit:
            app.add_middleware(RateLimitMiddleware, redis=redis_client)

        app.add_middleware(RequestLoggingMiddleware)

        return app

    mcp.streamable_http_app = _wrapped
