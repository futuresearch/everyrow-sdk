---
name: everyrow-sdk
description: Helps write Python code using the everyrow SDK for AI-powered data processing - transforming, deduping, merging, ranking, and screening dataframes with natural language instructions
---

# everyrow SDK

The everyrow SDK provides intelligent data processing utilities powered by AI agents. Use this skill when writing Python code that needs to:
- Deduplicate data using semantic understanding
- Merge tables using AI-powered matching
- Rank/score rows based on AI evaluation
- Screen/filter rows based on research-intensive criteria
- Run AI agents over dataframe rows

## Installation

```bash
pip install everyrow
# or
uv pip install everyrow
```

## Configuration

Requires an API key from https://everyrow.io:

```bash
export EVERYROW_API_KEY=your_api_key_here
```

## Core Pattern

All operations are async and require a session:

```python
import asyncio
from everyrow import create_client, create_session

async def main():
    async with create_client() as client:
        async with create_session(client=client, name="My Session") as session:
            print(f"View session at: {session.get_url()}")
            # ... use session for operations

asyncio.run(main())
```

Simplified version (client created automatically):

```python
from everyrow import create_session

async with create_session(name="My Session") as session:
    # ... use session
```

## Operations

### dedupe - Deduplicate data

Remove duplicates using AI-powered semantic matching:

```python
from everyrow.ops import dedupe

result = await dedupe(
    session=session,
    input=dataframe,
    equivalence_relation="Two entries are duplicates if they represent the same research work, even with different titles or authors listed",
)
# result.data contains the deduplicated DataFrame
```

### merge - Merge tables with AI matching

Join tables where the match criteria requires understanding:

```python
from everyrow.ops import merge

result = await merge(
    session=session,
    task="Match clinical trial sponsors with their parent pharmaceutical companies",
    left_table=trial_data,
    right_table=company_data,
    merge_on_left="sponsor",
    merge_on_right="company",
)
```

### rank - Score and rank rows

Add AI-generated scores to rank rows:

```python
from everyrow.ops import rank

result = await rank(
    session=session,
    task="Score this organization by their contribution to open source AI research (0-100)",
    input=dataframe,
    field_name="contribution_score",
    ascending_order=False,
)
```

### screen - Evaluate and filter rows

Evaluate rows based on criteria that requires research:

```python
from everyrow.ops import screen
from pydantic import BaseModel

class VendorAssessment(BaseModel):
    approved: bool
    risk_level: str  # "low", "medium", "high"
    security_concerns: str
    recommendation: str

result = await screen(
    session=session,
    task="""Evaluate vendor security and financial stability:
    1. Have they had data breaches in the past 3 years?
    2. Are there signs of financial distress?
    Only approve vendors with low/medium risk.""",
    input=vendors_df,
    response_model=VendorAssessment,
    batch_size=5,
)
```

### single_agent - Single input task

Run an AI agent on a single input:

```python
from everyrow.ops import single_agent
from pydantic import BaseModel

class Input(BaseModel):
    company: str

result = await single_agent(
    session=session,
    task="Research this company and summarize their main products",
    input=Input(company="Anthropic"),
    return_table=False,  # Set True for table output
)
```

### agent_map - Batch processing

Run an AI agent across multiple rows:

```python
from everyrow.ops import agent_map
from pandas import DataFrame

result = await agent_map(
    session=session,
    task="What is the capital of the given country?",
    input=DataFrame([{"country": "India"}, {"country": "USA"}]),
    return_table_per_row=False,
)
```

## Async Variants

All operations have `_async` variants for background processing:

```python
from everyrow.ops import rank_async

task = await rank_async(
    session=session,
    task="Score organizations",
    input=dataframe,
    field_name="score",
)

# Do other work...

result = await task.await_result(session.client)
```

## Result Structure

All operations return a result object with:
- `result.data` - The output DataFrame
- `result.artifact_id` - ID for the stored artifact

## Effort Levels

Control processing intensity with `effort_level`:

```python
from everyrow.generated.models import TaskEffort

result = await single_agent(
    session=session,
    task="...",
    input=input_data,
    effort_level=TaskEffort.LOW,  # or MEDIUM, HIGH
)
```
