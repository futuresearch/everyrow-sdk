# everyrow-polars

Polars integration for [everyrow](https://github.com/futuresearch/everyrow-sdk): LLM-powered DataFrame operations that feel native to Polars.

## Installation

```bash
pip install everyrow-polars
```

You'll need an everyrow API key. Get one at [everyrow.io](https://everyrow.io) and set it as an environment variable:

```bash
export EVERYROW_API_KEY=your_api_key
```

## Usage

Import `everyrow_polars` to register the `everyrow` namespace on all Polars DataFrames:

```python
import polars as pl
import everyrow_polars  # registers df.everyrow namespace

df = pl.read_csv("leads.csv")
```

### Screen: Filter by LLM judgment

```python
screened = df.everyrow.screen(
    "Remote-friendly senior role with disclosed salary"
)
# Returns DataFrame with `passes` column
remote_senior = screened.filter(pl.col("passes"))
```

### Rank: Score and sort rows

```python
ranked = df.everyrow.rank(
    "Likelihood to need data integration solutions",
    field_name="integration_score",
    descending=True,
)
top_leads = ranked.head(20)
```

### Dedupe: Remove semantic duplicates

```python
deduped = df.everyrow.dedupe(
    "Same person despite name variations or career changes"
)
# Returns DataFrame with `equivalence_class_id` and `selected` columns
unique = deduped.filter(pl.col("selected"))
```

### Merge: Intelligent entity matching

```python
products = pl.read_csv("software_products.csv")
suppliers = pl.read_csv("approved_suppliers.csv")

matched = products.everyrow.merge(
    suppliers,
    "Match each software product to its parent company",
    left_on="product_name",
    right_on="company_name",
)
```

### Research: Web research per row

```python
enriched = df.everyrow.research(
    "Find this company's latest funding round and lead investors",
    effort_level="medium",
)
```

## Structured Output

All operations support structured output via Pydantic models:

```python
from pydantic import BaseModel, Field

class VendorAssessment(BaseModel):
    risk_level: str = Field(description="low/medium/high")
    reasoning: str

assessed = vendors.everyrow.screen(
    "Evaluate vendor security and financial stability",
    response_model=VendorAssessment,
)
# Result has `risk_level` and `reasoning` columns
```

## Async Support

All methods have async variants for use in async contexts (e.g., Jupyter notebooks with async cells, FastAPI endpoints):

```python
# In an async context
result = await df.everyrow.screen_async("...")
result = await df.everyrow.rank_async("...")
result = await df.everyrow.dedupe_async("...")
result = await df.everyrow.merge_async(other_df, "...")
result = await df.everyrow.research_async("...")
```

## How It Works

Under the hood, everyrow-polars:

1. Converts your Polars DataFrame to pandas (via Arrow, which is efficient)
2. Calls the everyrow SDK operation
3. Converts the result back to a Polars DataFrame

The conversion overhead is minimal for most use cases. Everyrow operations are inherently I/O-bound (they call an API), so the pandas conversion is not the bottleneck.

## Limitations

- **Eager DataFrames only**: LazyFrames are not supported. Call `.collect()` first if you have a LazyFrame.
- **Sync methods block**: The sync methods (`screen`, `rank`, etc.) use `asyncio.run()` internally. In async contexts, use the `*_async` variants to avoid blocking.

## License

MIT
