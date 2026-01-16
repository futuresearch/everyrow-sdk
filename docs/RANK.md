# Rank

Score rows based on criteria you can't put in a database field.

## The problem

Traditional lead scoring uses firmographic data—employee count, industry code, funding stage. But "likelihood to need data integration tools" isn't in any database.

Ultramain Systems and Ukraine International Airlines both show up as "Aviation" in your CRM. One sells software to airlines (simple ops, unified systems). The other *is* an airline (complex ops, dozens of data sources, legacy integrations everywhere). Their actual needs are opposite, but they look identical on paper.

## How it works

You describe what you want to score in plain English. For each row, agents research the company and assign a score with reasoning.

```python
from everyrow.ops import rank

result = await rank(
    task="Score by likelihood to need data integration solutions",
    input=leads_dataframe,
    field_name="integration_need_score",
)
```

The task can be as specific as you want:

```python
result = await rank(
    task="""
        Score 0-100 by likelihood to adopt research tools in the next 12 months.

        High scores: teams actively publishing, hiring researchers, or with
        recent funding for R&D. Low scores: pure trading shops, firms with
        no public research output.
    """,
    input=investment_firms,
    field_name="research_adoption_score",
    ascending_order=False,  # highest first
)
```

## Structured output

If you want more than just a number, pass a Pydantic model:

```python
from pydantic import BaseModel, Field

class LeadScore(BaseModel):
    score: float = Field(description="0-100, higher = more likely to need integration")
    reasoning: str = Field(description="Why this score")
    key_signal: str = Field(description="The single most important factor")

result = await rank(
    task="Score by data integration needs",
    input=leads,
    field_name="score",
    response_model=LeadScore,
)
```

Now each row has `score`, `reasoning`, and `key_signal` columns.

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `task` | str | What to score and how |
| `input` | DataFrame | Your data |
| `field_name` | str | Column name for the score |
| `response_model` | BaseModel | Optional structured output |
| `ascending_order` | bool | True = lowest first (default) |
| `session` | Session | Optional, auto-created if omitted |

## Performance

| Rows | Time | Cost |
|------|------|------|
| 100 | ~2 min | ~$1.50 |
| 1,000 | ~7 min | ~$13 |
| 5,000 | ~25 min | ~$60 |

## Case studies

- [Lead Scoring with Data Fragmentation](https://futuresearch.ai/lead-scoring-data-fragmentation/) — 1,000 B2B leads ranked by data fragmentation risk
- [Lead Scoring Without CRM](https://futuresearch.ai/lead-scoring-without-crm/) — 85 investment firms scored for $28 (Clay wanted $145)
