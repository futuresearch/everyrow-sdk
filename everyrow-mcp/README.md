# everyrow MCP Server

MCP (Model Context Protocol) server for [everyrow](https://everyrow.io): agent ops at spreadsheet scale.

This server exposes everyrow's 5 core operations as MCP tools, allowing LLM applications (like Claude Code) to screen, rank, dedupe, merge, and run agents on CSV files.

## Installation

```bash
# Using uv (recommended)
cd everyrow-mcp
uv pip install -e .

# Or using pip
pip install -e .
```

## Configuration

### Environment Variables

The server requires an everyrow API key:

```bash
export EVERYROW_API_KEY=your_key_here
```

Get an API key at [everyrow.io/api-key](https://everyrow.io/api-key) ($20 free credit).

### Claude Code Configuration

Add to your Claude Code settings (`~/.claude/settings.json` or project `.claude/settings.json`):

```json
{
  "mcpServers": {
    "everyrow": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/everyrow-sdk/everyrow-mcp",
        "everyrow-mcp"
      ],
      "env": {
        "EVERYROW_API_KEY": "your_key_here"
      }
    }
  }
}
```

Or if installed globally:

```json
{
  "mcpServers": {
    "everyrow": {
      "command": "everyrow-mcp",
      "env": {
        "EVERYROW_API_KEY": "your_key_here"
      }
    }
  }
}
```

## Available Tools

### everyrow_screen

Filter CSV rows based on criteria that require judgment.

```
Parameters:
- task: Natural language description of screening criteria
- input_csv: Absolute path to input CSV
- output_path: Directory or full .csv path for output
- response_schema: (optional) JSON schema for response fields
```

Example: Filter job postings for "remote-friendly AND senior-level AND salary disclosed"

### everyrow_rank

Score and sort CSV rows based on qualitative criteria.

```
Parameters:
- task: Natural language description of ranking criteria
- input_csv: Absolute path to input CSV
- output_path: Directory or full .csv path for output
- field_name: Name of the score field to add
- field_type: Type of field (float, int, str, bool)
- ascending_order: Sort direction (default: true)
- response_schema: (optional) JSON schema for response fields
```

Example: Rank leads by "likelihood to need data integration solutions"

### everyrow_dedupe

Remove duplicate rows using semantic equivalence.

```
Parameters:
- equivalence_relation: Natural language description of what makes rows duplicates
- input_csv: Absolute path to input CSV
- output_path: Directory or full .csv path for output
- select_representative: Keep one row per duplicate group (default: true)
```

Example: Dedupe contacts where "same person even with name abbreviations or career changes"

### everyrow_merge

Join two CSV files using intelligent entity matching.

```
Parameters:
- task: Natural language description of how to match rows
- left_csv: Absolute path to primary CSV
- right_csv: Absolute path to secondary CSV
- output_path: Directory or full .csv path for output
- merge_on_left: (optional) Column name in left table
- merge_on_right: (optional) Column name in right table
```

Example: Match software products to parent companies (Photoshop -> Adobe)

### everyrow_agent

Run web research agents on each row of a CSV.

```
Parameters:
- task: Natural language description of research task
- input_csv: Absolute path to input CSV
- output_path: Directory or full .csv path for output
- response_schema: (optional) JSON schema for response fields
```

Example: "Find this company's latest funding round and lead investors"

## Output Path Handling

The `output_path` parameter accepts two formats:

1. **Directory**: Output file is named `{operation}_{input_name}.csv`
   - Input: `/data/companies.csv`, Output path: `/output/`
   - Result: `/output/screened_companies.csv`

2. **Full file path**: Use the exact path specified
   - Output path: `/output/my_results.csv`
   - Result: `/output/my_results.csv`

The server validates output paths before making API requests to avoid wasted costs.

## Response Schemas

For `screen`, `rank`, and `agent` operations, you can provide a custom response schema as JSON:

```json
{
  "properties": {
    "score": {
      "type": "number",
      "description": "Relevance score from 0 to 100"
    },
    "reasoning": {
      "type": "string",
      "description": "Explanation for the score"
    }
  },
  "required": ["score"]
}
```

## Development

```bash
cd everyrow-mcp
uv sync
uv run pytest
```

## License

MIT - See [LICENSE.txt](../LICENSE.txt)
