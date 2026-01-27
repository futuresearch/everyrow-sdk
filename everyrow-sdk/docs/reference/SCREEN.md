# Screen

Filter rows based on criteria that require research.

## The problem

You want to find S&P 500 companies that:
- Have >75% recurring revenue
- Would benefit from Taiwan tensions (CHIPS Act, defense, cybersecurity)
- Aren't dependent on Taiwan for manufacturing

None of this is in a database. Bloomberg can filter on P/E ratio. It can't filter on "benefits from geopolitical tensions."

Even simpler stuff fails with traditional methods. Filtering job postings for "remote-friendly" with regex gets 68% precision—hundreds of false positives to sort through manually.

## How it works

You describe what should pass. Agents research each row (pulling 10-Ks, earnings calls, news) and decide.

```python
from everyrow.ops import screen
from pydantic import BaseModel, Field

class ScreenResult(BaseModel):
    passes: bool = Field(description="True if company meets criteria")

result = await screen(
    task="""
        Find companies with >75% recurring revenue that would benefit from
        Taiwan tensions. Include CHIPS Act beneficiaries, defense contractors,
        cybersecurity firms. Exclude companies dependent on Taiwan manufacturing.
    """,
    input=sp500,
    response_model=ScreenResult,
)
print(result.data.head())
```

Only passing rows come back.

## Richer output

Want to know *why* something passed? Add fields:

```python
class VendorRisk(BaseModel):
    approved: bool = Field(description="True if vendor is acceptable")
    risk_level: str = Field(description="low / medium / high")
    security_issues: str = Field(description="Any breaches or incidents")
    recommendation: str = Field(description="Summary")

result = await screen(
    task="""
        Assess each vendor for enterprise use. Research:
        1. Security incidents in past 3 years
        2. Financial stability (layoffs, funding issues)

        Approve only low/medium risk with no unresolved critical incidents.
    """,
    input=vendors,
    response_model=VendorRisk,
)
print(result.data.head())
```

Now you get `risk_level`, `security_issues`, and `recommendation` for every row that passed.

## The pass/fail field

Your response model needs a boolean field. It can be named anything—`passes`, `approved`, `include`, whatever. The system figures out which field is the filter.

```python
class Simple(BaseModel):
    passes: bool

class Detailed(BaseModel):
    approved: bool  # this is the filter
    confidence: float
    notes: str
```

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `task` | str | What should pass |
| `input` | DataFrame | Rows to screen |
| `response_model` | BaseModel | Optional. Must have a boolean field. Defaults to `passes: bool` |
| `session` | Session | Optional, auto-created if omitted |

## Performance

| Rows | Time | Cost | Precision |
|------|------|------|-----------|
| 100 | ~3 min | ~$0.70 | >90% |
| 500 | ~12 min | ~$3.30 | >90% |
| 1,000 | ~20 min | ~$6 | >90% |

Compare: regex on "remote-friendly" job postings gets 68% precision.

## Case studies

- [Thematic Stock Screen](https://futuresearch.ai/thematic-stock-screening/) — 63 of 502 S&P 500 companies passed, $3.29
- [Job Posting Screen](https://futuresearch.ai/job-posting-screening/) — >90% precision vs 68% for regex
- [Screening Workflow](https://futuresearch.ai/screening-workflow/) — iterate on criteria without rerunning everything
