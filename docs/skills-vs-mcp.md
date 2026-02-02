---
title: Skills vs MCP
description: Compare the two ways to use everyrow with Claude Code
---

# Skills vs MCP

everyrow integrates with Claude Code through **Skills** (recommended) or **MCP server**. Here's how they compare:

### Skills (Recommended)

Skills give Claude Code **guided workflows** with best practices built in.

**How it works:**
1. Claude reads your skill instructions
2. Claude writes Python code following everyrow patterns
3. The code runs in your local environment
4. Results save to your filesystem

**Best for:**
- Complex multi-step workflows (dedupe → merge → research)
- Custom data transformations
- Integration with your existing Python scripts
- Full control over execution

![Skills workflow](/docs/images/skills-flow.svg)

### MCP Server

MCP provides **direct tool calls** without code generation.

**How it works:**
1. Claude calls everyrow MCP tools directly
2. Results return in the conversation
3. No intermediate Python code

**Best for:**
- Quick one-off operations
- Simple lookups and enrichments
- Environments where code execution is restricted

![MCP workflow](/docs/images/mcp-flow.svg)

### Comparison Table

| Feature | Skills | MCP Server |
|---------|--------|------------|
| Setup complexity | Simple (2 commands) | Moderate (config file) |
| Code visibility | Full Python code shown | Tool calls only |
| Customization | High (edit generated code) | Limited |
| Multi-step workflows | Excellent | Basic |
| Works in Claude Desktop | No | Yes (Chat mode only) |
| Debugging | Easy (see all code) | Harder (opaque calls) |

## Next Steps

- [Installation Guide](/docs/installation) - Full setup instructions
- [API Reference](/docs/reference/AGENT) - Detailed function documentation
- [Case Studies](/docs/notebooks/dedupe-crm-company-records) - Real-world examples
