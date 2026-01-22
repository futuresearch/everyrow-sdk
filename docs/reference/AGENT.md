# Agent

Run AI agents to research information that isn't in your data.

## The problem

You want to enrich your data with information from the internet:

- Find a company's latest funding round
- Look up the CEO for an organization
- Research competitive positioning for a product
- Get pricing information from a website

This isn't a database lookup. It requires an agent that can search, read, extract, and reason.

| Function | Use when... |
| -------- | ----------- |
| `single_agent` | You have zero or one inputs to research |
| `agent_map` | You have a table and need to research each row in parallel |

## `single_agent`

Run an agent on a single input.

```python
from pydantic import BaseModel
from everyrow.ops import single_agent

class CompanyInput(BaseModel):
    company: str

result = await single_agent(
    task="Find the company's most recent annual revenue and employee count",
    input=CompanyInput(company="Stripe"),
)
print(result.data)
```

### No input required

You can run an agent without any input data.

```python
result = await single_agent(
    task="""
        What company has reported the greatest cost reduction
        due to internal AI usage over the past 12 months?
    """,
)
```

## `agent_map`

Run an agent on every row of a table.

```python
from pandas import DataFrame
from everyrow.ops import agent_map

companies = DataFrame([
    {"company": "Stripe"},
    {"company": "Databricks"},
    {"company": "Canva"},
])

result = await agent_map(
    task="Find the company's most recent annual revenue",
    input=companies,
)
print(result.data)
```

Each row gets its own agent that researches independently.

## Parameters

`single_agent` and `agent_map` have the nearly the same parameters.

| Name | Type | Description |
| ---- | ---- | ----------- |
| `task` | str | The agent task describing what to research |
| `session` | Session | Optional, auto-created if omitted |
| `input` | BaseModel \| DataFrame \| UUID | Optional input context |
| `effort_level` | EffortLevel | LOW, MEDIUM, or HIGH (default: LOW) |
| `llm` | LLM | Optional agent LLM override |
| `response_model` | BaseModel | Optional schema for structured output |
| `return_table(_per_row)` | bool | If True, each agent returns a table instead of a scalar result |

### Effort levels

The effort level lets you control how thorough the research is.

- `LOW`: Quick lookups, basic web searches, fast and cheap (default)
- `MEDIUM`: More thorough research, multiple sources consulted
- `HIGH`: Deep research, cross-referencing sources, higher accuracy

### Response model

Both `single_agent` and `agent_map` support structured output via custom Pydantic models.

```python
from pandas import DataFrame
from pydantic import BaseModel, Field
from everyrow.ops import agent_map

companies = DataFrame([
    {"company": "Stripe"},
    {"company": "Databricks"},
    {"company": "Canva"},
])

class CompanyFinancials(BaseModel):
    annual_revenue_usd: int = Field(description="Most recent annual revenue in USD")
    employee_count: int = Field(description="Current number of employees")
    last_funding_round: str = Field(description="Most recent funding round, e.g. 'Series C'")

result = await agent_map(
    task="Research each company's financials and latest funding",
    input=companies,
    response_model=CompanyFinancials,
)
```

Now the output has `annual_revenue_usd`, `employee_count`, and `last_funding_round` columns.

### Returning a table

If you want each agent to return multiple rows, set `return_table_per_row=True`.

When using `single_agent`, this lets you generate a dataset from scratch. And when using `agent_map`, it lets you expand each input row into multiple output rows.

```python
from pandas import DataFrame
from pydantic import BaseModel, Field
from everyrow.ops import agent_map, single_agent

class CompanyInfo(BaseModel):
    company: str = Field(description="Company name")
    market_cap: int = Field(description="Market cap in USD")

companies = await single_agent(
    task="Find the three largest US healthcare companies by market cap",
    response_model=CompanyInfo,
    return_table=True,  # Return a table of companies
)

class ExecutiveInfo(BaseModel):
    name: str = Field(description="Executive's full name")
    title: str = Field(description="Their job title")

result = await agent_map(
    task="Find all C-suite executives at this company",
    input=companies,
    response_model=ExecutiveInfo,
    return_table_per_row=True,  # For each company, return a list of executives
)
```
