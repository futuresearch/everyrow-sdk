"""HTTP mode configuration for the everyrow MCP server."""

from __future__ import annotations

import logging
import sys
from typing import Any

from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import lifespan_wrapper
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import AnyHttpUrl
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from everyrow_mcp.auth import SupabaseTokenVerifier
from everyrow_mcp.config import DevHttpSettings, HttpSettings
from everyrow_mcp.gcs_storage import GCSResultStore
from everyrow_mcp.redis_utils import create_redis_client
from everyrow_mcp.routes import api_progress
from everyrow_mcp.state import state
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
    log: logging.Logger,
) -> None:
    """Full OAuth HTTP mode."""
    try:
        settings = HttpSettings()  # pyright: ignore[reportCallIssue]
    except Exception as e:
        logging.error(f"HTTP mode configuration error: {e}")
        sys.exit(1)
    state.settings = settings

    redis_client = create_redis_client(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
        sentinel_endpoints=settings.redis_sentinel_endpoints,
        sentinel_master_name=settings.redis_sentinel_master_name,
    )
    state.redis = redis_client

    # Token verifier (resource-server mode — Supabase is the authorization server)
    verifier = SupabaseTokenVerifier(settings.supabase_url)
    mcp._token_verifier = verifier

    supabase_issuer = settings.supabase_url.rstrip("/") + "/auth/v1"
    mcp.settings.auth = AuthSettings(
        issuer_url=AnyHttpUrl(supabase_issuer),
        resource_server_url=AnyHttpUrl(settings.mcp_server_url),
    )
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
    )

    mcp_server_url = settings.mcp_server_url

    # GCS result store (required in auth mode)
    state.gcs_store = GCSResultStore(
        settings.gcs_results_bucket,
        signed_url_expiry_minutes=settings.signed_url_expiry_minutes,
    )
    log.info("GCS result store enabled: %s", settings.gcs_results_bucket)

    _configure_shared(mcp, http_lifespan, host, port, mcp_server_url)


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
    state.redis = redis_client

    mcp_server_url = f"http://localhost:{port}"

    # GCS is optional in no-auth mode
    if settings.gcs_results_bucket:
        state.gcs_store = GCSResultStore(
            settings.gcs_results_bucket,
            signed_url_expiry_minutes=settings.signed_url_expiry_minutes,
        )
        log.info("GCS result store enabled: %s", settings.gcs_results_bucket)

    _configure_shared(mcp, http_lifespan, host, port, mcp_server_url)


def _configure_shared(
    mcp: FastMCP,
    http_lifespan: Any,
    host: str,
    port: int,
    mcp_server_url: str,
) -> None:
    """Configuration shared between auth and no-auth HTTP modes."""
    mcp._mcp_server.lifespan = lifespan_wrapper(mcp, http_lifespan)
    mcp.settings.host = host
    mcp.settings.port = port

    state.mcp_server_url = mcp_server_url

    # Re-register session resource with CSP allowing progress endpoint fetch
    connect_domains: list[str] = [mcp_server_url]

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

    # Re-register results resource with CSP allowing GCS fetch
    results_connect = list(connect_domains)
    if state.gcs_store is not None:
        results_connect.append("https://storage.googleapis.com")

    @mcp.resource(
        "ui://everyrow/results.html",
        mime_type="text/html;profile=mcp-app",
        meta={
            "ui": {
                "csp": {
                    "resourceDomains": ["https://unpkg.com"],
                    "connectDomains": results_connect,
                }
            }
        },
    )
    def _results_ui_http() -> str:
        return RESULTS_HTML

    # Progress + health routes
    mcp.custom_route("/api/progress/{task_id}", ["GET", "OPTIONS"])(api_progress)

    async def _health(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    mcp.custom_route("/health", ["GET"])(_health)

    # Request logging middleware
    _original_streamable_http_app = mcp.streamable_http_app

    def _logging_streamable_http_app():
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
        return app

    mcp.streamable_http_app = _logging_streamable_http_app
