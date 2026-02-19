"""Test MCP client that connects to the OAuth-protected server.

Simulates how Claude.ai connects: OAuth discovery → DCR → PKCE → token → MCP.

Usage:
    # Start server first:
    MCP_SERVER_URL=http://localhost:8000 SUPABASE_URL=... SUPABASE_ANON_KEY=... \
        uv run everyrow-mcp --http --port 8000

    # Run this client:
    uv run python scripts/test_mcp_client.py
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import re
import secrets
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlparse

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_SERVER = "http://localhost:8000"
CALLBACK_PORT = 9999
CALLBACK_URI = f"http://127.0.0.1:{CALLBACK_PORT}/callback"


class CallbackHandler(BaseHTTPRequestHandler):
    """Tiny HTTP server that captures the OAuth callback."""

    auth_code: str | None = None

    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        CallbackHandler.auth_code = params.get("code", [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h2>Authenticated! You can close this tab.</h2>")

    def log_message(self, *args):
        pass


def do_oauth_flow() -> str:
    """Run the full OAuth 2.1 flow and return a Bearer access token."""
    http = httpx.Client()

    # 1. Discover
    print("1. Discovering OAuth endpoints...")
    meta = http.get(f"{MCP_SERVER}/.well-known/oauth-authorization-server").json()
    register_url = meta["registration_endpoint"]
    authorize_url = meta["authorization_endpoint"]
    token_url = meta["token_endpoint"]
    print(f"   authorize: {authorize_url}")
    print(f"   token:     {token_url}")

    # 2. DCR
    print("2. Registering OAuth client...")
    reg = http.post(
        register_url,
        json={
            "client_name": "mcp-test-client",
            "redirect_uris": [CALLBACK_URI],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",  # public client — PKCE is the sole proof
        },
    ).json()
    client_id = reg["client_id"]
    print(f"   client_id: {client_id}")

    # 3. PKCE
    verifier = secrets.token_urlsafe(32)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )

    # 4. Callback server
    server = HTTPServer(("127.0.0.1", CALLBACK_PORT), CallbackHandler)
    thread = Thread(target=server.handle_request, daemon=True)
    thread.start()

    # 5. Browser login
    state = secrets.token_urlsafe(16)
    auth_url = (
        f"{authorize_url}?client_id={client_id}"
        f"&redirect_uri={CALLBACK_URI}"
        f"&response_type=code"
        f"&code_challenge={challenge}"
        f"&code_challenge_method=S256"
        f"&state={state}"
    )
    print("3. Opening browser for Google login...")
    webbrowser.open(auth_url)

    # 6. Wait
    print("   Waiting for login callback...")
    thread.join(timeout=120)
    server.server_close()

    if not CallbackHandler.auth_code:
        raise RuntimeError("No auth code received — did you complete the login?")
    print(f"   Got auth code: {CallbackHandler.auth_code[:20]}...")

    # 7. Token exchange
    print("4. Exchanging code for tokens...")
    token_resp = http.post(
        token_url,
        data={
            "grant_type": "authorization_code",
            "code": CallbackHandler.auth_code,
            "redirect_uri": CALLBACK_URI,
            "client_id": client_id,
            "code_verifier": verifier,
        },
    ).json()

    if "access_token" not in token_resp:
        raise RuntimeError(f"Token exchange failed: {token_resp}")

    access_token = token_resp["access_token"]
    print(f"   access_token: {access_token[:20]}...")
    print(f"   refresh_token: {token_resp['refresh_token'][:20]}...")

    http.close()
    return access_token


async def test_mcp(access_token: str, call_tool: bool = False):
    """Connect to the MCP server and list tools, like Claude would."""
    print("\n5. Connecting to MCP server with Bearer token...")

    headers = {"Authorization": f"Bearer {access_token}"}

    async with streamablehttp_client(url=f"{MCP_SERVER}/mcp", headers=headers) as (
        read_stream,
        write_stream,
        _get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            result = await session.initialize()
            print(f"   Server: {result.serverInfo.name} v{result.serverInfo.version}")
            print(f"   Protocol: {result.protocolVersion}")

            tools = await session.list_tools()
            print(f"\n6. Available tools ({len(tools.tools)}):")
            for tool in tools.tools:
                ann = tool.annotations
                title = ann.title if ann else tool.name
                print(f"   - {tool.name} ({title})")

            if call_tool:
                await _call_and_poll(session)

    print("\n" + "=" * 60)
    print("  Full OAuth + MCP flow succeeded!")
    print("  This is exactly how Claude.ai connects to remote MCP servers.")
    print("=" * 60)


def _extract_task_id(tool_result) -> str | None:
    """Extract task_id from MCP tool response text."""
    for part in tool_result.content:
        try:
            data = json.loads(part.text)
            task_id = data.get("task_id")
            if task_id:
                return task_id
        except (ValueError, AttributeError, TypeError):
            pass
        m = re.search(r"Task ID:\s*([0-9a-f-]{36})", part.text)
        if m:
            return m.group(1)
        m = re.search(r"task_id=['\"]?([0-9a-f-]{36})", part.text)
        if m:
            return m.group(1)
    return None


async def _call_and_poll(session: ClientSession) -> None:
    """Call everyrow_agent, poll progress, and fetch results."""
    print("\n7. Calling everyrow_agent with inline data...")
    tool_result = await session.call_tool(
        "everyrow_agent",
        {
            "params": {
                "task": "What year was this company founded?",
                "input_data": "company\nApple\nGoogle\nAmazon",
                "response_schema": {
                    "founded_year": {
                        "type": "integer",
                        "description": "Year the company was founded",
                    }
                },
            }
        },
    )
    print(f"   Tool response ({len(tool_result.content)} parts):")
    for part in tool_result.content:
        print(f"   {part.text}")

    task_id = _extract_task_id(tool_result)
    if not task_id:
        return

    print(f"\n8. Polling everyrow_progress({task_id})...")
    for i in range(60):
        progress = await session.call_tool(
            "everyrow_progress",
            {"params": {"task_id": task_id}},
        )
        status_text = progress.content[0].text
        print(f"   [{i + 1}] {status_text}")
        if "completed" in status_text.lower() or "failed" in status_text.lower():
            break
        await asyncio.sleep(5)

    print(f"\n9. Fetching results for {task_id}...")
    results = await session.call_tool(
        "everyrow_results",
        {"params": {"task_id": task_id}},
    )
    for part in results.content:
        print(f"   {part.text[:500]}")


def main():
    parser = argparse.ArgumentParser(description="MCP OAuth E2E Test")
    parser.add_argument(
        "--call-tool",
        action="store_true",
        help="Also call everyrow_screen to test a full tool invocation",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MCP OAuth E2E Test (simulates Claude.ai)")
    print("=" * 60)
    print()

    access_token = do_oauth_flow()
    asyncio.run(test_mcp(access_token, call_tool=args.call_tool))


if __name__ == "__main__":
    main()
