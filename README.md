# everyrow SDK

The everyrow SDK provides intelligent data processing utilities powered by AI agents. Transform, dedupe, merge, rank, and screen your dataframes using natural language instructions. Whether you're deduplicating research papers, merging complex datasets, ranking organizations, or screening vendors, the SDK handles the heavy lifting by combining AI research capabilities with structured data operations.

## Installation

```bash
uv pip install -e .
```

Or install dependencies:

```bash
uv sync
```

## Requirements

- Python >= 3.12

## Configuration

Get an API key from https://everyrow.io and set it to get started:

```bash
# Set in your environment or .env file
EVERYROW_API_KEY=your_api_key_here
```

## Usage

### Quick Start

```python
from everyrow_sdk import create_session
from everyrow_sdk.ops import dedupe
from pandas import DataFrame

async with create_session() as session:
    data = DataFrame([...])
    result = await dedupe(
        session=session,
        input=data,
        equivalence_relation="Two items are duplicates if...",
    )
    print(result.data)
```

### Core Utilities

#### Rank: `rank`

Extract and rank rows based on AI-generated scores:

```python
from everyrow_sdk.ops import rank

result = await rank(
    session=session,
    task="Score this organization by their contribution to AI research",
    input=dataframe,
    field_name="contribution_score",
    ascending_order=False,
)
```

#### Dedupe: `dedupe`

Intelligently deduplicate your data using AI-powered equivalence detection:

```python
from everyrow_sdk.ops import dedupe

result = await dedupe(
    session=session,
    input=dataframe,
    equivalence_relation="Two entries are duplicates if they represent the same research work",
)
```

#### Merge: `merge`

Merge two tables using AI to match related rows:

```python
from everyrow_sdk.ops import merge

result = await merge(
    session=session,
    task="Match clinical trial sponsors with parent companies",
    left_table=trial_data,
    right_table=company_data,
    merge_on_left="sponsor",
    merge_on_right="company",
)
```

#### Screen: `screen`

Evaluate and filter rows based on criteria that require research:

```python
from everyrow_sdk.ops import screen
from pydantic import BaseModel

class Assessment(BaseModel):
    risk_level: str
    recommendation: str

result = await screen(
    session=session,
    task="Evaluate vendor security and financial stability",
    input=vendors,
    response_model=Assessment,
)
```

### Viewing Sessions

Every session has a web interface URL:

```python
async with create_session(name="My Session") as session:
    print(f"View session at: {session.get_url()}")
    # ... use session for operations
```

### Agent Tasks

For single-input tasks, use `single_agent`:

```python
from everyrow_sdk.ops import single_agent
from pydantic import BaseModel

class Input(BaseModel):
    country: str

result = await single_agent(
    session=session,
    task="What is the capital of the given country?",
    input=Input(country="India"),
)
```

For batch processing, use `agent_map`:

```python
from everyrow_sdk.ops import agent_map

result = await agent_map(
    session=session,
    task="What is the capital of the given country?",
    input=DataFrame([{"country": "India"}, {"country": "USA"}]),
)
```

### Async Operations

All utilities have async variants for background processing:

```python
from everyrow_sdk.ops import rank_async

task = await rank_async(
    session=session,
    task="Score this organization",
    input=dataframe,
    field_name="score",
)

# Continue with other work...

result = await task.await_result(session.client)
```

## Case Studies

The `case_studies/` directory contains example workflows demonstrating real-world usage of the SDK. To run case studies, install the optional dependencies:

```bash
uv sync --group case-studies
```

Then you can run the case study scripts or open the Jupyter notebooks in your preferred environment.

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

Note: The `everyrow_sdk/generated/` directory is excluded from linting as it contains auto-generated code.

## License

This project is licensed under the MIT License - see LICENSE.txt file for details.
