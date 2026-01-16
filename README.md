![hero](https://github.com/user-attachments/assets/7ed92cea-4c81-4ccd-9f7f-738763bfdee4)

# <picture><img src="images/future-search-logo-128.webp" alt="FutureSearch" height="24" align="top"></picture> everyrow SDK

The everyrow python SDK is an interface to [everyrow.io](https://everyrow.io) for intelligent data processing utilities powered by AI agents. Rank, Dedupe, Merge, and Screen your dataframes using natural language instructions, or apply web agents to research every row.

## Table of Contents

- [Getting Started](#getting-started)
- [Installation](#installation)
- [Claude Code Plugin](#claude-code-plugin)
- [Usage](#usage)
  - [Rank](#rank)
  - [Dedupe](#dedupe)
  - [Merge](#merge)
  - [Screen](#screen)
  - [Agent Tasks](#agent-tasks)
  - [Async Operations](#async-operations)
- [Case Studies](#case-studies)
- [Development](#development)
- [License](#license)

## Getting Started

Get an API key at [everyrow.io](https://everyrow.io).

```bash
# Set in your environment or .env file
export EVERYROW_API_KEY=your_api_key_here
```

## Installation

```bash
pip install everyrow
```

For development:

```bash
uv pip install -e .
uv sync
```

**Requirements:** Python >= 3.12

## Claude Code Plugin

This repository includes a plugin for [Claude Code](https://code.claude.com/) that teaches Claude how to write code using the everyrow SDK.

```sh
# from Claude Code
/plugin marketplace add futuresearch/everyrow-sdk
/plugin install everyrow@futuresearch

# from terminal
claude plugin marketplace add futuresearch/everyrow-sdk
claude plugin install everyrow@futuresearch
```

## Usage

The SDK provides flexible session management. For quick one-off operations, sessions are created automatically:

```python
from everyrow.ops import single_agent

# Simplest usage - session created automatically
result = await single_agent(
    task="What is the capital of France?",
    input={"country": "France"},
)
```

For multiple operations, use an explicit session to group them together:

```python
from everyrow import create_session

async with create_session(name="My Session") as session:
    print(f"View session at: {session.get_url()}")
    # All operations in this block share the same session
```

For maximum control, create the client explicitly:

```python
from everyrow import create_client, create_session

async with create_client() as client:
    async with create_session(client=client, name="My Session") as session:
        # ... use session for operations
```

Sessions are logical groupings of tasks visible on the [everyrow.io](https://everyrow.io) interface.

### Rank

Score and rank rows based on complex criteria.

```python
from everyrow.ops import rank

result = await rank(
    task="Score by contribution to AI research",
    input=dataframe,
    field_name="contribution_score",
)
print(result.data)
```

### Dedupe

Intelligently deduplicate data using AI-powered equivalence detection.

```python
from everyrow.ops import dedupe

result = await dedupe(
    input=dataframe,
    equivalence_relation="Two entries are duplicates if they represent the same research work",
)
print(result.data)
```

### Merge

Match and merge two tables using AI.

```python
from everyrow.ops import merge

result = await merge(
    task="Match trial sponsors with parent companies",
    left_table=trial_data,
    right_table=pharma_companies,
    merge_on_left="sponsor",
    merge_on_right="company",
)
print(result.data)
```

### Screen

Evaluate and filter rows based on criteria that require research.

```python
from everyrow.ops import screen
from pydantic import BaseModel

class VendorAssessment(BaseModel):
    approved: bool
    risk_level: str
    recommendation: str

result = await screen(
    task="Evaluate vendor security and financial stability",
    input=vendors,
    response_model=VendorAssessment,
)
print(result.data)
```

### Agent Tasks

For single-input tasks, use `single_agent`. For batch processing, use `agent_map`.

```python
from everyrow.ops import single_agent, agent_map
from pandas import DataFrame

# Single input
result = await single_agent(
    task="What is the capital of the given country?",
    input={"country": "India"},
)

# Batch processing
result = await agent_map(
    task="What is the capital of the given country?",
    input=DataFrame([{"country": "India"}, {"country": "USA"}]),
)
```

### Async Operations

All utilities have async variants for background processing. These require an explicit session since the task needs to persist beyond the function call.

```python
from everyrow import create_session
from everyrow.ops import rank_async

async with create_session(name="Async Ranking") as session:
    task = await rank_async(
        session=session,
        task="Score this organization",
        input=dataframe,
        field_name="score",
    )

    # Continue with other work...
    result = await task.await_result()
```

## Case Studies

The `case_studies/` directory contains example workflows. To run them:

```bash
uv sync --group case-studies
```

## Development

### Setup

```bash
uv sync
lefthook install
```

### Running Tests

```bash
uv run pytest
```

### Linting & Formatting

```bash
uv run ruff check .
uv run ruff check --fix .
uv run ruff format .
```

### Type Checking

```bash
uv run basedpyright
```

### Generating OpenAPI Client

```bash
./generate_openapi.sh
```

Note: The `everyrow/generated/` directory is excluded from linting as it contains auto-generated code.

## License

This project is licensed under the MIT License - see LICENSE.txt file for details.
