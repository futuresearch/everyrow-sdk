"""MCP server for everyrow SDK operations."""

import argparse
import logging
import os
import sys

import everyrow_mcp.tools  # noqa: F401  — registers @mcp.tool() decorators
from everyrow_mcp.app import _http_lifespan, mcp
from everyrow_mcp.config import StdioSettings
from everyrow_mcp.http_config import configure_http_mode
from everyrow_mcp.state import state


def main():
    """Run the MCP server."""
    parser = argparse.ArgumentParser(description="everyrow MCP server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Use Streamable HTTP transport instead of stdio.",
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
        configure_http_mode(mcp, _http_lifespan, host=args.host, port=args.port)
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
