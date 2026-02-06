---
title: "Getting Started"
description: Install everyrow and run your first operation.
---

# Getting Started

## Prerequisites

- Python 3.12+
- API key from [everyrow.io/api-key](https://everyrow.io/api-key)

## Installation

```bash
pip install everyrow
export EVERYROW_API_KEY=your_key_here
```

See [Installation](/installation) for other options (MCP servers, coding agent plugins).

## Example

Screen rows using natural language criteria:

```python
import asyncio
import pandas as pd
from everyrow.ops import screen
from pydantic import BaseModel, Field

jobs = pd.DataFrame([
    {"company": "Airtable",   "post": "Async-first team, 8+ yrs exp, $185-220K base"},
    {"company": "Vercel",     "post": "Lead our NYC team. Competitive comp, DOE"},
    {"company": "Notion",     "post": "In-office SF. Staff eng, $200K + equity"},
    {"company": "Linear",     "post": "Bootcamp grads welcome! $85K, remote-friendly"},
    {"company": "Descript",   "post": "Work from anywhere. Principal architect, $250K"},
    {"company": "Retool",     "post": "Flexible location. Building infra. Comp TBD"},
])

class JobScreenResult(BaseModel):
    qualifies: bool = Field(description="True if meets ALL criteria")

async def main():
    result = await screen(
        task="""
            Qualifies if ALL THREE are met:
            1. Remote-friendly (allows remote, hybrid, or distributed)
            2. Senior-level (5+ yrs exp OR title includes Senior/Staff/Principal)
            3. Salary disclosed (specific numbers like "$150K", not "competitive" or "DOE")
        """,
        input=jobs,
        response_model=JobScreenResult,
    )
    print(result.data.head())  # Airtable, Descript pass. Others fail one or more.

asyncio.run(main())
```

This handles cases where string matching fails—e.g., "DOE" means salary is *not* disclosed, "bootcamp grads welcome" implies *not* senior-level.

## Operations

| Operation | Description |
|-----------|-------------|
| [Screen](/reference/SCREEN) | Filter rows by criteria requiring judgment |
| [Rank](/reference/RANK) | Score rows by qualitative factors |
| [Dedupe](/reference/DEDUPE) | Deduplicate when fuzzy matching fails |
| [Merge](/reference/MERGE) | Join tables when keys don't match exactly |
| [Research](/reference/RESEARCH) | Run web agents to research each row |

## See Also

- [Guides](/filter-dataframe-with-llm) — step-by-step tutorials
- [Case Studies](/notebooks/basic-usage) — worked examples
- [Skills vs MCP](/skills-vs-mcp) — integration options
