---
name: everyrow-sdk
description: Helps write Python code using the everyrow SDK for AI-powered data processing - transforming, deduping, merging, ranking, and screening dataframes with natural language instructions
---

# everyrow SDK

The everyrow SDK provides intelligent data processing utilities powered by AI agents. Use this skill when writing Python code that needs to:
- Rank/score rows based on qualitative criteria
- Deduplicate data using semantic understanding
- Merge tables using AI-powered matching
- Screen/filter rows based on research-intensive criteria
- Run AI agents over dataframe rows

## Installation

```bash
pip install everyrow
```

## Configuration

Before writing any everyrow code, check if `EVERYROW_API_KEY` is set. If not, prompt the user:

> everyrow requires an API key. Do you have one?
> - If yes, paste it here
> - If no, get one at https://everyrow.io/api-key and paste it back

Once the user provides the key, set it:

```bash
export EVERYROW_API_KEY=<their_key>
```

## Results

All operations return a result object. The data is available as a pandas DataFrame in `result.data`:

```python
result = await rank(...)
print(result.data.head())  # pandas DataFrame
```

## Operations

For quick one-off operations, sessions are created automatically.

### rank - Score and rank rows

Score rows based on criteria you can't put in a database field:

```python
from everyrow.ops import rank

result = await rank(
    task="Score by likelihood to need data integration solutions",
    input=leads_dataframe,
    field_name="integration_need_score",
)
print(result.data.head())
```

### dedupe - Deduplicate data

Remove duplicates using AI-powered semantic matching. The AI understands that "AbbVie Inc", "Abbvie", and "AbbVie Pharmaceutical" are the same company:

```python
from everyrow.ops import dedupe

result = await dedupe(
    input=crm_data,
    equivalence_relation="Two entries are duplicates if they represent the same legal entity",
)
print(result.data.head())
```

Results include `equivalence_class_id` (groups duplicates), `equivalence_class_name` (human-readable cluster name), and `selected` (the canonical record in each cluster).

### merge - Merge tables with AI matching

Join two tables when the keys don't match exactly. The AI knows "Photoshop" belongs to "Adobe" and "Genentech" is a Roche subsidiary:

```python
from everyrow.ops import merge

result = await merge(
    task="Match each software product to its parent company",
    left_table=software_products,
    right_table=approved_suppliers,
    merge_on_left="software_name",
    merge_on_right="company_name",
)
print(result.data.head())
```

### screen - Evaluate and filter rows

Filter rows based on criteria that require research:

```python
from everyrow.ops import screen
from pydantic import BaseModel, Field

class ScreenResult(BaseModel):
    passes: bool = Field(description="True if company meets the criteria")

result = await screen(
    task="""
        Find companies with >75% recurring revenue that would benefit from
        Taiwan tensions - CHIPS Act beneficiaries, defense contractors,
        cybersecurity firms. Exclude companies dependent on Taiwan manufacturing.
    """,
    input=sp500_companies,
    response_model=ScreenResult,
)
print(result.data.head())
```

### single_agent - Single input task

Run an AI agent on a single input:

```python
from everyrow.ops import single_agent

result = await single_agent(
    task="What is the capital of the given country?",
    input={"country": "India"},
)
print(result.data.head())
```

### agent_map - Batch processing

Run an AI agent across multiple rows:

```python
from everyrow.ops import agent_map
from pandas import DataFrame

result = await agent_map(
    task="What is the capital of the given country?",
    input=DataFrame([{"country": "India"}, {"country": "USA"}]),
)
print(result.data.head())
```

## Explicit Sessions

For multiple operations or when you need visibility into progress, use an explicit session:

```python
from everyrow import create_session

async with create_session(name="My Session") as session:
    print(f"View session at: {session.get_url()}")
    # All operations here share the same session
```

Sessions are visible on the everyrow.io dashboard.

## Async Operations

All operations have `_async` variants for background processing. These need an explicit session since the task persists beyond the function call:

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
    print(f"Task ID: {task.task_id}")  # Print this! Useful if your script crashes.

    # Continue with other work...
    result = await task.await_result()
```

**Tip:** Print the task ID after submitting. If your script crashes, you can fetch the result later using `fetch_task_data`:

```python
from everyrow import fetch_task_data

# Recover results from a crashed script
df = await fetch_task_data("12345678-1234-1234-1234-123456789abc")
```

## Best Practices

Everyrow operations have associated costs. To avoid re-running them unnecessarily:

- **Separate data processing from analysis**: Save everyrow results to a file (CSV, Parquet, etc.), then do analysis in a separate script. This way, if analysis code has bugs, you don't re-trigger the everyrow step.
- **Use intermediate checkpoints**: For multi-step pipelines, consider saving results after each everyrow operation.
    - You are able to chain multiple operations together without needing to download and re-upload intermediate results via the SDK. However for most control, implement each step as a dedicated job, possibly orchestrated by tools such as Apache Airflow or Prefect.
- **Test with `preview=True`**: Operations like `rank`, `screen`, and `merge` support `preview=True` to process only a few rows first.
