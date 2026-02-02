---
title: Claude Code Integration
description: Use everyrow with Claude Code for AI-powered data processing in your terminal
---

# Claude Code Integration

Claude Code is Anthropic's official CLI for Claude. everyrow integrates with Claude Code through **Skills** (recommended) or **MCP server**, letting you process data files directly in your terminal conversations.

## Quick Start

```bash
# Install the everyrow skill
claude plugin marketplace add futuresearch/everyrow-sdk
claude plugin install everyrow@futuresearch
```

Then ask Claude Code to process your data:

```
You: I have leads.csv with company names. Can you research each company's
funding status and employee count?

Claude: I'll use everyrow to research each company...
```

## Skills vs MCP: Which to Use?

everyrow offers two integration methods for Claude Code. Here's how they compare:

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

```
┌─────────────┐    reads     ┌─────────────────┐
│ Claude Code │ ──────────► │  Skill Prompts  │
└─────────────┘              └─────────────────┘
       │
       │ writes
       ▼
┌─────────────┐    calls     ┌─────────────────┐
│ Python Code │ ──────────► │  everyrow API   │
└─────────────┘              └─────────────────┘
       │
       │ saves
       ▼
┌─────────────┐
│  Your CSV   │
└─────────────┘
```

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

```
┌─────────────┐   tool call   ┌─────────────────┐
│ Claude Code │ ───────────► │  MCP Server     │
└─────────────┘              └─────────────────┘
                                     │
                              calls  │
                                     ▼
                             ┌─────────────────┐
                             │  everyrow API   │
                             └─────────────────┘
```

### Comparison Table

| Feature | Skills | MCP Server |
|---------|--------|------------|
| Setup complexity | Simple (2 commands) | Moderate (config file) |
| Code visibility | Full Python code shown | Tool calls only |
| Customization | High (edit generated code) | Limited |
| Multi-step workflows | Excellent | Basic |
| Works in Claude Desktop | No | Yes (Chat mode only) |
| Debugging | Easy (see all code) | Harder (opaque calls) |

## Installation

### Method 1: Skills (Recommended)

1. **Add from marketplace:**
   ```bash
   claude plugin marketplace add futuresearch/everyrow-sdk
   ```

2. **Install the skill:**
   ```bash
   claude plugin install everyrow@futuresearch
   ```

3. **Set your API key:**
   ```bash
   export EVERYROW_API_KEY=your_key_here
   ```

Get an API key at [everyrow.io/api-key](https://everyrow.io/api-key) ($20 free credit included).

### Method 2: MCP Server

1. **Set your API key:**
   ```bash
   export EVERYROW_API_KEY=your_key_here
   ```

2. **Download the bundle** from [GitHub Releases](https://github.com/futuresearch/everyrow-sdk/releases) and double-click to install.

   Or add manually to your MCP config:
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

## Example Workflows

### Research Company Data

```
You: I have prospects.csv with company names. Research each company's
industry, employee count, and latest funding round.

Claude: I'll use everyrow to research each company. Here's the code I'll run:

    import everyrow
    df = everyrow.research(
        "prospects.csv",
        {"industry": "What industry is this company in?",
         "employees": "Approximately how many employees?",
         "funding": "What was their latest funding round and amount?"}
    )
    df.to_csv("prospects_enriched.csv")

[Runs code and shows results]
```

### Deduplicate Records

```
You: contacts.csv has duplicate entries. Clean them up.

Claude: I'll deduplicate using everyrow:

    import everyrow
    df = everyrow.dedupe("contacts.csv")
    df.to_csv("contacts_deduped.csv")

Found 47 duplicate groups, merged down to 312 unique records.
```

### Screen by Criteria

```
You: Filter investment_opportunities.csv to companies with >$10M ARR
in the fintech space.

Claude: I'll screen using everyrow:

    import everyrow
    df = everyrow.screen(
        "investment_opportunities.csv",
        "Company has >$10M ARR and operates in fintech"
    )
    df.to_csv("qualified_opportunities.csv")

Screened 500 companies, 23 passed the criteria.
```

## Troubleshooting

### "Skill not found" error

Ensure the skill is installed:
```bash
claude plugin list
# Should show: everyrow@futuresearch
```

If not listed, reinstall:
```bash
claude plugin install everyrow@futuresearch
```

### API key not working

1. Check your key is exported:
   ```bash
   echo $EVERYROW_API_KEY
   ```

2. Verify at [everyrow.io/api-key](https://everyrow.io/api-key) that your key is active and has credits.

### MCP server not connecting

1. Test the server manually:
   ```bash
   uvx everyrow-mcp
   ```

2. Check your MCP config path:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

## Next Steps

- [Installation Guide](/installation) - Full setup for all platforms
- [API Reference](/reference/AGENT) - Detailed function documentation
- [Case Studies](/notebooks/dedupe-crm-company-records) - Real-world examples
