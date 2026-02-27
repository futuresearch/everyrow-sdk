---
title: classify
description: API reference for the EveryRow classify tool, which assigns each row of a dataset into one of the provided categories using web research.
---

# Classify

`classify` takes a DataFrame and a list of allowed categories, then assigns each row to exactly one category using web research that scales to the difficulty of the classification. Supports binary (yes/no) and multi-category classification with optional reasoning output.

## Examples

### GICS sector classification

```python
from pandas import DataFrame
from everyrow.ops import classify

companies = DataFrame([
    {"company": "Apple"},
    {"company": "JPMorgan Chase"},
    {"company": "ExxonMobil"},
    {"company": "Pfizer"},
    {"company": "Procter & Gamble"},
    {"company": "Tesla"},
    {"company": "AT&T"},
    {"company": "Caterpillar"},
    {"company": "Duke Energy"},
    {"company": "Simon Property Group"},
])

result = await classify(
    task="Classify this company by its GICS industry sector",
    categories=[
        "Energy", "Materials", "Industrials", "Consumer Discretionary",
        "Consumer Staples", "Health Care", "Financials",
        "Information Technology", "Communication Services",
        "Utilities", "Real Estate",
    ],
    input=companies,
)
print(result.data[["company", "classification"]])
```

Output:

| company              | classification         |
|----------------------|------------------------|
| Apple                | Information Technology |
| JPMorgan Chase       | Financials             |
| ExxonMobil           | Energy                 |
| Pfizer               | Health Care            |
| Procter & Gamble     | Consumer Staples       |
| Tesla                | Consumer Discretionary |
| AT&T                 | Communication Services |
| Caterpillar          | Industrials            |
| Duke Energy          | Utilities              |
| Simon Property Group | Real Estate            |

### Binary classification

For yes/no questions, use two categories:

```python
result = await classify(
    task="Is this company founder-led?",
    categories=["yes", "no"],
    input=companies,
)
```

### Custom output column and reasoning

```python
result = await classify(
    task="Classify each company by its primary industry sector",
    categories=["Technology", "Finance", "Healthcare", "Energy"],
    input=companies,
    classification_field="sector",
    include_reasoning=True,
)
print(result.data[["company", "sector", "reasoning"]])
```

## Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `task` | str | required | Natural-language instructions describing how to classify each row |
| `categories` | list[str] | required | Allowed category values (minimum 2). Each row is assigned exactly one. |
| `input` | DataFrame | required | Rows to classify |
| `classification_field` | str | `"classification"` | Name of the output column for the assigned category |
| `include_reasoning` | bool | `False` | If True, adds a `reasoning` column with the agent's justification |
| `session` | Session | Optional, auto-created if omitted | |

## Output

One column is added to each input row (name controlled by `classification_field`):

| Column | Type | Description |
|--------|------|-------------|
| `classification` | str | One of the provided `categories` values |
| `reasoning` | str | Agent's justification (only if `include_reasoning=True`) |

## Via MCP

MCP tool: `everyrow_classify`

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | string | Classification instructions |
| `categories` | list[string] | Allowed categories (minimum 2) |
| `classification_field` | string | Output column name (default: `"classification"`) |
| `include_reasoning` | boolean | Include reasoning column (default: false) |
