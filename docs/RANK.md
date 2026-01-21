# Rank

AI-powered ranking using natural language criteria

## The problem

If your dataset already contains the data you want to rank by, then sorting is easy.

But what do you do if your data is in an unstructured format? Or what if it requires researching every row to find what you need?

For instance, let's say you're trying to prioritize sales leads. You may have employee count, industry code, and funding stage. But "likelihood to need data integration tools" isn't in any database.

## How it works

`rank` uses AI research agents to find the metric you specify for each row of your dataset. Then it sorts the rows by that metric.

Our research agents can search the internet, read webpages and documents, extract relevant information, and reason with nuance about what they find.

## Examples

You describe the metric you want to rank by in plain English.

```python
from everyrow.ops import rank

result = await rank(
    task="Score by likelihood to need data integration solutions",
    input=leads_dataframe,
    field_name="integration_need_score",
    ascending_order=False,  # highest first
)
print(result.data.head())
```

The task can be as specific as you want. You can describe the metric in detail, list which sources to use, and explain how to resolve ambiguities.

```python
result = await rank(
    task="""
        Score 0-100 by likelihood to adopt research tools in the next 12 months.

        High scores: teams actively publishing, hiring researchers, or with
        recent funding for R&D. Low scores: pure trading shops, firms with
        no public research output.

        Consult the company's website, job postings, and LinkedIn profile for information.
    """,
    input=investment_firms,
    field_name="research_adoption_score",
    ascending_order=False,  # highest first
)
print(result.data.head())
```

### Structured output

If you want more than just a number, pass a Pydantic model.

Note that you don't need specify fields for reasoning, explanation or sources. That information is included automatically.

```python
from pydantic import BaseModel, Field

class AcquisitionScore(BaseModel):
    fit_score: float = Field(description="0-100, strategic alignment with our business")
    annual_revenue_usd: int = Field(description="Their estimated annual revenue in USD")

result = await rank(
    task="Score acquisition targets by product-market fit and revenue quality",
    input=potential_acquisitions,
    field_name="fit_score",
    response_model=AcquisitionScore,
    ascending_order=False,  # highest first
)
print(result.data.head())
```

Now every row has both `fit_score` and `annual_revenue_usd` fields, each of which includes its own explanation.

When specifying a response model, make sure that it contains `field_name`. Otherwise, you'll get an error. Also, the `field_type` parameter is ignored when you pass a response model.

## Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| `task` | str | The task for the agent describing how to find your metric |
| `session` | Session | Optional, auto-created if omitted |
| `input` | DataFrame | Your data |
| `field_name` | str | Column name for the metric |
| `field_type` | str | The type of the field (default: "float") |
| `response_model` | BaseModel | Optional response model for multiple output fields |
| `ascending_order` | bool | True = lowest first (default) |
| `preview` | bool | True = process only a few rows |

## Case studies

- [Ranking 1000 Businesses by Data Fragmentation Risk](https://futuresearch.ai/lead-scoring-data-fragmentation/): Ranking 1,000 B2B leads by data fragmentation risk
- [Rank Leads Like an Analyst, Not a Marketer](https://futuresearch.ai/lead-scoring-without-crm/): Using `rank` to score leads instead of a CRM
