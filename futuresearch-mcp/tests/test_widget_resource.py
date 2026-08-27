"""Tests for the session widget resource as it is actually served."""

import pytest
from mcp.server.fastmcp import FastMCP

from futuresearch_mcp import server
from futuresearch_mcp.http_config import _register_widgets

WIDGET_URI = "ui://futuresearch/session.html"


async def _serve_widget() -> str:
    mcp = FastMCP("test")
    _register_widgets(mcp, "https://mcp.example.test")
    contents = await mcp.read_resource(WIDGET_URI)
    return "".join(str(c.content) for c in contents)


@pytest.mark.asyncio
async def test_widget_wraps_arguments_the_way_the_tool_declares_them():
    """Each tool takes one pydantic model, so its arguments nest under params."""
    tools = await server.mcp.list_tools()
    schema = next(t for t in tools if t.name == "futuresearch_task_data").inputSchema
    assert schema["required"] == ["params"]

    html = await _serve_widget()

    assert 'name:"futuresearch_task_data"' in html
    assert "arguments:{params:{" in html


@pytest.mark.asyncio
async def test_widget_asks_the_host_before_calling_a_server_tool():
    """A host that won't proxy tool calls must be left on the REST path."""
    html = await _serve_widget()

    assert "getHostCapabilities" in html
    assert "caps.serverTools" in html


@pytest.mark.asyncio
async def test_no_placeholder_survives_into_the_page():
    """An unsubstituted token would be a bare identifier, so a syntax error."""
    html = await _serve_widget()

    assert "SCRIPT_SRC" not in html
