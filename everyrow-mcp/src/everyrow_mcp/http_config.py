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
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from everyrow_mcp.auth import EveryRowAuthProvider, SupabaseTokenVerifier
from everyrow_mcp.config import DevHttpSettings, http_settings
from everyrow_mcp.middleware import RateLimitMiddleware
from everyrow_mcp.redis_utils import create_redis_client
from everyrow_mcp.routes import api_download, api_progress
from everyrow_mcp.state import RedisStore, state
from everyrow_mcp.templates import RESULTS_HTML, SESSION_HTML


def configure_http_mode(
    mcp: FastMCP,
    http_lifespan: Any,
    host: str,
    port: int,
    *,
    no_auth: bool = False,
) -> None:
    """Configure the MCP server for HTTP transport.

    When *no_auth* is True the server skips OAuth setup and uses a singleton
    client from EVERYROW_API_KEY (like stdio mode).  Intended for local
    development only.
    """
    log = logging.getLogger(__name__)
    state.transport = "streamable-http"

    if no_auth:
        _configure_no_auth(mcp, http_lifespan, host, port, log)
    else:
        _configure_auth(mcp, http_lifespan, host, port, log)


# ── Internal helpers ─────────────────────────────────────────────────


def _configure_auth(
    mcp: FastMCP,
    http_lifespan: Any,
    host: str,
    port: int,
    _log: logging.Logger,
) -> None:
    """Full OAuth HTTP mode -- our server is the authorization server."""
    settings = http_settings
    state.settings = settings

    redis_client = create_redis_client(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
        sentinel_endpoints=settings.redis_sentinel_endpoints,
        sentinel_master_name=settings.redis_sentinel_master_name,
    )
    state.store = RedisStore(redis_client)

    # Token verifier validates Supabase JWTs via JWKS
    verifier = SupabaseTokenVerifier(settings.supabase_url, redis=redis_client)

    # Auth provider (handles registration + OAuth flow via Supabase)
    auth_provider = EveryRowAuthProvider(redis=redis_client, token_verifier=verifier)
    state.auth_provider = auth_provider

    # Wire auth into FastMCP
    mcp._auth_server_provider = auth_provider  # type: ignore[arg-type]
    mcp._token_verifier = verifier
    mcp.settings.auth = AuthSettings(
        issuer_url=AnyHttpUrl(settings.mcp_server_url),
        resource_server_url=AnyHttpUrl(settings.mcp_server_url),
        client_registration_options=ClientRegistrationOptions(enabled=True),
    )
    parsed = urlparse(settings.mcp_server_url)
    allowed_host = parsed.hostname or "localhost"
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[allowed_host],
    )

    mcp_server_url = settings.mcp_server_url

    logging.warning(
        "Auth configured: issuer=%s resource_server=%s allowed_hosts=%s",
        mcp.settings.auth.issuer_url,
        mcp.settings.auth.resource_server_url,
        mcp.settings.transport_security.allowed_hosts,
    )

    _configure_shared(
        mcp,
        http_lifespan,
        host,
        port,
        mcp_server_url,
        redis_client=redis_client,
        auth_provider=auth_provider,
    )


def _configure_no_auth(
    mcp: FastMCP,
    http_lifespan: Any,
    host: str,
    port: int,
    log: logging.Logger,
) -> None:
    """No-auth HTTP mode for local development."""
    log.warning("Running in --no-auth mode (development only)")

    try:
        settings = DevHttpSettings()  # pyright: ignore[reportCallIssue]
    except Exception as e:
        logging.error(f"--no-auth configuration error: {e}")
        sys.exit(1)
    state.settings = settings

    redis_client = create_redis_client(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
    )
    state.store = RedisStore(redis_client)

    mcp_server_url = f"http://localhost:{port}"

    _configure_shared(mcp, http_lifespan, host, port, mcp_server_url, no_auth=True)


def _configure_shared(
    mcp: FastMCP,
    http_lifespan: Any,
    host: str,
    port: int,
    mcp_server_url: str,
    *,
    redis_client: Any | None = None,
    no_auth: bool = False,
    auth_provider: EveryRowAuthProvider | None = None,
) -> None:
    """Configuration shared between auth and no-auth HTTP modes."""
    mcp._mcp_server.lifespan = lifespan_wrapper(mcp, http_lifespan)
    mcp.settings.host = host
    mcp.settings.port = port

    state.mcp_server_url = mcp_server_url

    # CSP: allow widgets to fetch from unpkg (SDK) and our server (API calls)
    connect_domains: list[str] = [mcp_server_url]
    widget_csp = {
        "resourceDomains": ["https://unpkg.com"],
        "connectDomains": connect_domains,
    }

    # Patch tool meta to include CSP (host may read CSP from tool, not resource)
    for tool_name in ("everyrow_progress", "everyrow_results"):
        tool = mcp._tool_manager._tools.get(tool_name)
        if tool and tool.meta and "ui" in tool.meta:
            tool.meta["ui"]["csp"] = widget_csp

    @mcp.resource(
        "ui://everyrow/session.html",
        mime_type="text/html;profile=mcp-app",
        meta={
            "ui": {
                "csp": {
                    "resourceDomains": ["https://unpkg.com"],
                    "connectDomains": connect_domains,
                }
            }
        },
    )
    def _session_ui_http() -> str:
        return SESSION_HTML

    @mcp.resource(
        "ui://everyrow/results.html",
        mime_type="text/html;profile=mcp-app",
        meta={
            "ui": {
                "csp": {
                    "resourceDomains": ["https://unpkg.com"],
                    "connectDomains": connect_domains,
                }
            }
        },
    )
    def _results_ui_http() -> str:
        return RESULTS_HTML

    # Progress, download + health routes
    mcp.custom_route("/api/progress/{task_id}", ["GET", "OPTIONS"])(api_progress)
    mcp.custom_route("/api/results/{task_id}/download", ["GET", "OPTIONS"])(
        api_download
    )

    async def _health(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    mcp.custom_route("/health", ["GET"])(_health)

    # Auth routes (only in auth mode)
    if auth_provider is not None:
        mcp.custom_route("/auth/start/{state}", ["GET"])(auth_provider.handle_start)
        mcp.custom_route("/auth/callback", ["GET"])(auth_provider.handle_callback)

    # Request logging + rate-limit middleware
    _original_streamable_http_app = mcp.streamable_http_app

    def _middleware_streamable_http_app():
        app = _original_streamable_http_app()

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

        if not no_auth and redis_client is not None:
            app.add_middleware(RateLimitMiddleware, redis=redis_client)

        return app

    mcp.streamable_http_app = _middleware_streamable_http_app
