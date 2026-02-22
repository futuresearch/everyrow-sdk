"""MCP server for everyrow SDK operations."""

import argparse
import logging
import os
import sys
from textwrap import dedent

from pydantic import BaseModel

import everyrow_mcp.tools  # noqa: F401  — registers @mcp.tool() decorators
from everyrow_mcp.app import mcp
from everyrow_mcp.config import get_dev_http_settings, get_http_settings
from everyrow_mcp.http_config import configure_http_mode
from everyrow_mcp.redis_utils import create_redis_client
from everyrow_mcp.state import RedisStore, Transport, state
from everyrow_mcp.tool_descriptions import set_tool_descriptions


class InputArgs(BaseModel):
    http: bool = False
    no_auth: bool = False
    port: int = 8000
    host: str = "0.0.0.0"


def parse_args() -> InputArgs:
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
    input_args = InputArgs.model_validate(vars(parser.parse_args()))

    if input_args.no_auth and not input_args.http:
        parser.error("--no-auth requires --http")

    if input_args.no_auth and not os.environ.get("ALLOW_NO_AUTH"):
        print(
            dedent("""ERROR: --no-auth requires the ALLOW_NO_AUTH=1 environment variable.\n
            This prevents accidental unauthenticated deployments in production."""),
            file=sys.stderr,
        )
        sys.exit(1)

    return input_args


def main():
    """Run the MCP server."""
    input_args = parse_args()
    # Signal to the SDK that we're inside the MCP server (suppresses plugin hints)
    os.environ["EVERYROW_MCP_SERVER"] = "1"
    state.transport = Transport.HTTP if input_args.http else Transport.STDIO
    state.no_auth = input_args.no_auth

    set_tool_descriptions(state.transport)
    if input_args.http:
        if input_args.no_auth:
            settings = get_dev_http_settings()
            state.mcp_server_url = f"http://localhost:{input_args.port}"
        else:
            settings = get_http_settings()
            state.mcp_server_url = settings.mcp_server_url

        redis_client = create_redis_client(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            sentinel_endpoints=settings.redis_sentinel_endpoints,
            sentinel_master_name=settings.redis_sentinel_master_name,
        )
        state.store = RedisStore(redis_client)

        configure_http_mode(
            mcp,
            redis_client=redis_client,
            host=input_args.host,
            port=input_args.port,
        )
    else:
        # Configure logging to use stderr only (stdout is reserved for JSON-RPC)
        logging.basicConfig(
            level=logging.WARNING,
            stream=sys.stderr,
            format="%(levelname)s: %(message)s",
            force=True,
        )

        # Validate EVERYROW_API_KEY is set (used by SDK client in lifespan)
        if not os.environ.get("EVERYROW_API_KEY"):
            logging.error("Configuration error: EVERYROW_API_KEY is required")
            logging.error("Get an API key at https://everyrow.io/api-key")
            sys.exit(1)

    mcp.run(transport=state.transport.value)


if __name__ == "__main__":
    main()
