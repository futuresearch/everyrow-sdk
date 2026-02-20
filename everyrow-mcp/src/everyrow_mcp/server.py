"""MCP server for everyrow SDK operations."""

import argparse
import logging
import os
import sys

import everyrow_mcp.tools  # noqa: F401  — registers @mcp.tool() decorators
from everyrow_mcp.app import (
    _http_lifespan,
    _no_auth_http_lifespan,
    mcp,
)
from everyrow_mcp.config import StdioSettings
from everyrow_mcp.http_config import configure_http_mode

# Re-export models, helpers, and tools so existing imports from
# ``everyrow_mcp.server`` keep working (tests, conftest, etc.).
from everyrow_mcp.models import (  # noqa: F401
    AgentInput,
    DedupeInput,
    MergeInput,
    ProgressInput,
    RankInput,
    ResultsInput,
    ScreenInput,
    SingleAgentInput,
    _schema_to_model,
)
from everyrow_mcp.state import state
from everyrow_mcp.tool_helpers import _write_task_state  # noqa: F401
from everyrow_mcp.tools import (  # noqa: F401
    everyrow_agent,
    everyrow_dedupe,
    everyrow_merge,
    everyrow_progress,
    everyrow_rank,
    everyrow_results,
    everyrow_screen,
    everyrow_single_agent,
)


def main():
    """Run the MCP server."""
    parser = argparse.ArgumentParser(description="everyrow MCP server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Use Streamable HTTP transport instead of stdio.",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable OAuth (dev only). Requires EVERYROW_API_KEY.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for HTTP transport (default: 0.0.0.0).",
    )
    args = parser.parse_args()

    if args.no_auth and not args.http:
        parser.error("--no-auth requires --http")

    if args.no_auth and not os.environ.get("ALLOW_NO_AUTH"):
        print(
            "ERROR: --no-auth requires the ALLOW_NO_AUTH=1 environment variable.\n"
            "This prevents accidental unauthenticated deployments in production.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Signal to the SDK that we're inside the MCP server (suppresses plugin hints)
    os.environ["EVERYROW_MCP_SERVER"] = "1"

    # Configure logging to use stderr only (stdout is reserved for JSON-RPC)
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s: %(message)s",
        force=True,
    )

    if args.http:
        lifespan = _no_auth_http_lifespan if args.no_auth else _http_lifespan
        configure_http_mode(
            mcp, lifespan, host=args.host, port=args.port, no_auth=args.no_auth
        )
        mcp.run(transport="streamable-http")
    else:
        state.transport = "stdio"

        # Validate required env vars for stdio mode
        try:
            state.settings = StdioSettings()  # pyright: ignore[reportCallIssue]
        except Exception as e:
            logging.error(f"Configuration error: {e}")
            logging.error("Get an API key at https://everyrow.io/api-key")
            sys.exit(1)

        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
