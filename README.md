# everyrow

[![PyPI version](https://img.shields.io/pypi/v/everyrow.svg)](https://pypi.org/project/everyrow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This monorepo contains two packages for [everyrow.io](https://everyrow.io): agent ops at spreadsheet scale.

## Packages

| Package | Description | PyPI |
|---------|-------------|------|
| [**everyrow-sdk**](./everyrow-sdk/) | Python SDK for screen, rank, dedupe, merge, and agent operations on dataframes | [![PyPI](https://img.shields.io/pypi/v/everyrow.svg)](https://pypi.org/project/everyrow/) |
| [**everyrow-mcp**](./everyrow-mcp/) | MCP server exposing everyrow operations as tools for LLM applications | [![PyPI](https://img.shields.io/pypi/v/everyrow-mcp.svg)](https://pypi.org/project/everyrow-mcp/) |

## Quick Start

### Python SDK

```bash
pip install everyrow
```

See [everyrow-sdk/README.md](./everyrow-sdk/README.md) for full documentation.

### MCP Server

For Claude Desktop, download the `.mcpb` bundle from [GitHub Releases](https://github.com/futuresearch/everyrow-sdk/releases).

Or install with pip/uvx:

```bash
pip install everyrow-mcp
```

See [everyrow-mcp/README.md](./everyrow-mcp/README.md) for configuration details.

## IDE/Agent Plugins

This repo also provides integrations for coding agents:

- **Claude Code**: `claude plugin marketplace add futuresearch/everyrow-sdk`
- **Gemini CLI**: `gemini extensions install https://github.com/futuresearch/everyrow-sdk`
- **Cursor**: Add remote rule `https://github.com/futuresearch/everyrow-sdk.git`
- **Codex CLI**: See [everyrow-sdk README](./everyrow-sdk/README.md#coding-agent-plugins)

## Development

Each package has its own dependencies and development environment:

```bash
# SDK
cd everyrow-sdk
uv sync
uv run pytest

# MCP Server
cd everyrow-mcp
uv sync
uv run pytest
```

## License

MIT - See [LICENSE.txt](./LICENSE.txt)

---

Built by [FutureSearch](https://futuresearch.ai).
