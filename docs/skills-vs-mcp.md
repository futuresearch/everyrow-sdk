# SDK Skills vs MCP

everyrow integrates with your agent through **SDK Skills** or **MCP server**. Here's how they compare:

### SDK skills

Skills give your agent **guided workflows** with best practices built in.

**How it works:**
1. Your agent reads your skill instructions
2. Your agent writes Python code using the everyrow SDK following everyrow patterns
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
1. Your agent calls everyrow MCP tools directly
2. Results saved to your filesystem and returned in the conversation
3. No intermediate Python code

**Best for:**
- Quick one-off operations
- Simple lookups and enrichments
- Environments where code execution is restricted, or more guarantees are needed

![MCP workflow](/docs/images/mcp-flow.svg)

### Comparison Table

| Feature | SDK skills | MCP Server |
|---------|--------|------------|
| Setup complexity | Requires python environment | Requires MCP client (e.g. Claude Desktop) |
| Code visibility | Full Python code shown | Tool calls only |
| Customization | High (edit generated code) | Limited |
| Multi-step workflows | Excellent | Basic |
| Works in Claude Desktop | Cowork mode only | Yes (For now, Chat mode only) |
| Debugging | Requires some python knowledge | Not needed |

## Next Steps

- [Installation Guide](/docs/installation) - Full setup instructions
- [API Reference](/docs/reference/RESEARCH) - Detailed function documentation
- [Case Studies](/docs/notebooks/dedupe-crm-company-records) - Real-world examples
