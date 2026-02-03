---
title: "Install EveryRow: Python SDK, Claude Code Plugin, and MCP Server"
description: Get started with everyrow. Install the Python SDK via pip, add the coding agent plugin for Claude Code, Gemini CLI, Codex, or Cursor, or configure the MCP server.
---

# Installation

## SDK

```bash
pip install everyrow
```

Requires Python 3.12+. Get an API key at [everyrow.io/api-key](https://everyrow.io/api-key) ($20 free credit).

```bash
export EVERYROW_API_KEY=your_key_here
```

## Coding Agent Plugins

### Claude Code

[Official Docs](https://code.claude.com/docs/en/discover-plugins#add-from-github)

```sh
claude plugin marketplace add futuresearch/everyrow-sdk
claude plugin install everyrow@futuresearch
```

### Gemini CLI

[Official Docs](https://geminicli.com/docs/extensions/#installing-an-extension). Requires version >= 0.25.0.

```sh
gemini extensions install https://github.com/futuresearch/everyrow-sdk
gemini extensions enable everyrow [--scope <user or workspace>]
```

Then within the CLI:

```sh
/settings > Preview Features > Enable
/settings > Agent Skills > Enable
/skills enable everyrow-sdk
/skills reload
/model > Manual > gemini-3-pro-preview
```

### Codex CLI

[Official Docs](https://developers.openai.com/codex/skills#install-new-skills)

```sh
python ~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo futuresearch/everyrow-sdk --path skills/everyrow-sdk
```

Restart Codex to pick up the new skill.

### Cursor

[Official Docs](https://cursor.com/docs/context/skills#installing-skills-from-github)

1. Open Cursor Settings → Rules
2. In the Project Rules section, click Add Rule
3. Select Remote Rule (Github)
4. Enter: `https://github.com/futuresearch/everyrow-sdk.git`

## MCP Server

### Claude Desktop

Download the latest `.mcpb` bundle from [GitHub Releases](https://github.com/futuresearch/everyrow-sdk/releases) and double-click to install. You'll be prompted for your API key during setup.

> **Note:** Works in Claude Desktop's **Chat** mode only (not Cowork mode due to a [known limitation](https://github.com/anthropics/claude-code/issues/20377)).

### Manual Config

Set your API key:

```bash
export EVERYROW_API_KEY=your_key_here
```

Add to your MCP config (with [uv](https://docs.astral.sh/uv/) installed):

```json
{
  "mcpServers": {
    "everyrow": {
      "command": "uvx",
      "args": ["everyrow-mcp"],
      "env": {
        "EVERYROW_API_KEY": "${EVERYROW_API_KEY}"
      }
    }
  }
}
```

Or install with pip and use `"command": "everyrow-mcp"` instead of uvx.
