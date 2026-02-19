"""HTTP mode configuration for the everyrow MCP server."""

from __future__ import annotations

import logging
import sys

from mcp.server.auth.provider import ProviderTokenVerifier
from mcp.server.auth.settings import (
    AuthSettings,
    ClientRegistrationOptions,
    RevocationOptions,
)
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import lifespan_wrapper
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import AnyHttpUrl
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from everyrow_mcp.auth import EveryRowAuthProvider
from everyrow_mcp.gcs_storage import GCSResultStore
from everyrow_mcp.redis_utils import create_redis_client
from everyrow_mcp.routes import api_progress, api_results
from everyrow_mcp.settings import HttpSettings
from everyrow_mcp.state import state
from everyrow_mcp.templates import RESULTS_HTML, SESSION_HTML


def configure_http_mode(
    mcp: FastMCP,
    http_lifespan: object,
    host: str,
    port: int,
) -> None:
    """Configure the MCP server for HTTP transport with OAuth."""
    state.transport = "streamable-http"

    # Validate and parse env vars for HTTP mode
    try:
        settings = HttpSettings()  # pyright: ignore[reportCallIssue]
    except Exception as e:
        logging.error(f"HTTP mode configuration error: {e}")
        sys.exit(1)
    state.settings = settings

    # Create Redis client for auth state
    redis_client = create_redis_client(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
        sentinel_endpoints=settings.redis_sentinel_endpoints,
        sentinel_master_name=settings.redis_sentinel_master_name,
    )

    # Create auth provider
    state.auth_provider = EveryRowAuthProvider(
        supabase_url=settings.supabase_url,
        supabase_anon_key=settings.supabase_anon_key,
        mcp_server_url=settings.mcp_server_url,
        redis=redis_client,
        encryption_key=settings.redis_encryption_key,
    )

    # Configure auth on the existing FastMCP instance (tools already registered)
    mcp._auth_server_provider = state.auth_provider
    mcp._token_verifier = ProviderTokenVerifier(state.auth_provider)
    mcp.settings.auth = AuthSettings(
        issuer_url=AnyHttpUrl(settings.mcp_server_url),
        resource_server_url=AnyHttpUrl(settings.mcp_server_url),
        client_registration_options=ClientRegistrationOptions(enabled=True),
        revocation_options=RevocationOptions(enabled=True),
    )
    mcp._mcp_server.lifespan = lifespan_wrapper(mcp, http_lifespan)
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    )

    # Store server URL for progress polling
    state.mcp_server_url = settings.mcp_server_url

    # Initialize GCS result store if enabled via RESULT_STORAGE=gcs
    if settings.result_storage == "gcs":
        state.gcs_store = GCSResultStore(settings.gcs_results_bucket)
        logging.getLogger(__name__).info(
            "GCS result store enabled: %s", settings.gcs_results_bucket
        )
    else:
        logging.getLogger(__name__).info("Using in-memory result cache")

    # Re-register session resource with CSP allowing progress endpoint fetch
    @mcp.resource(
        "ui://everyrow/session.html",
        mime_type="text/html;profile=mcp-app",
        meta={
            "ui": {
                "csp": {
                    "resourceDomains": ["https://unpkg.com"],
                    "connectDomains": [settings.mcp_server_url],
                }
            }
        },
    )
    def _session_ui_http() -> str:
        return SESSION_HTML

    # Re-register results resource with CSP allowing results endpoint fetch
    results_connect_domains = [settings.mcp_server_url]
    if state.gcs_store:
        results_connect_domains.append("https://storage.googleapis.com")

    @mcp.resource(
        "ui://everyrow/results.html",
        mime_type="text/html;profile=mcp-app",
        meta={
            "ui": {
                "csp": {
                    "resourceDomains": ["https://unpkg.com"],
                    "connectDomains": results_connect_domains,
                }
            }
        },
    )
    def _results_ui_http() -> str:
        return RESULTS_HTML

    # Mount custom routes
    mcp.custom_route("/auth/start/{state}", ["GET"])(state.auth_provider.handle_start)
    mcp.custom_route("/auth/callback", ["GET"])(state.auth_provider.handle_callback)
    mcp.custom_route("/api/progress/{task_id}", ["GET", "OPTIONS"])(api_progress)
    mcp.custom_route("/api/results/{task_id}", ["GET", "OPTIONS"])(api_results)

    async def _health(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    mcp.custom_route("/health", ["GET"])(_health)

    # Wrap the Starlette app with request logging middleware
    _original_streamable_http_app = mcp.streamable_http_app

    def _logging_streamable_http_app():
        app = _original_streamable_http_app()

        class RequestLoggingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                auth_header = request.headers.get("authorization", "(none)")
                if auth_header != "(none)":
                    auth_header = auth_header[:30] + "..."
                logging.warning(
                    "INCOMING %s %s | Host: %s | Auth: %s",
                    request.method,
                    request.url.path,
                    request.headers.get("host", "?"),
                    auth_header,
                )
                response = await call_next(request)
                logging.warning(
                    "RESPONSE %s %s -> %s",
                    request.method,
                    request.url.path,
                    response.status_code,
                )
                return response

        app.add_middleware(RequestLoggingMiddleware)
        return app

    mcp.streamable_http_app = _logging_streamable_http_app
