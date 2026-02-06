# dagster-everyrow

Dagster integration for [everyrow](https://everyrow.io): AI-powered data operations at scale.

This package provides a Dagster resource and asset factories for everyrow's core operations: screen, rank, dedupe, merge, and research. Every operation logs a clickable session URL to Dagster metadata, so you can inspect per-row reasoning directly from the Dagster UI.

## Installation

```bash
pip install dagster-everyrow
```

You'll need an everyrow API key. Get one at [everyrow.io/api-key](https://everyrow.io/api-key) ($20 free credit).

## Quick Start

### 1. Configure the Resource

```python
import dagster as dg
from dagster_everyrow import EveryrowResource

defs = dg.Definitions(
    resources={
        "everyrow": EveryrowResource(
            api_key=dg.EnvVar("EVERYROW_API_KEY"),
        ),
    },
    assets=[...],
)
```

### 2. Use in Assets

```python
import dagster as dg
import pandas as pd
from dagster_everyrow import EveryrowResource

@dg.asset
def raw_leads() -> pd.DataFrame:
    return pd.read_csv("s3://bucket/leads.csv")

@dg.asset
def scored_leads(
    context: dg.AssetExecutionContext,
    everyrow: EveryrowResource,
    raw_leads: pd.DataFrame,
) -> pd.DataFrame:
    result = everyrow.rank(
        task="Score by likelihood to need data integration solutions",
        input=raw_leads,
        field_name="integration_score",
    )

    # Log session URL for observability
    context.add_output_metadata({
        "everyrow_session": dg.MetadataValue.url(result.session_url),
        "row_count": len(result.data),
    })

    return result.data
```

## Available Operations

### Screen

Filter rows based on criteria that require judgment.

```python
result = everyrow.screen(
    task="Remote-friendly AND senior-level AND salary disclosed",
    input=leads_df,
)
# result.data has a 'passes' column
passing = result.data[result.data["passes"]]
```

### Rank

Score and sort rows based on qualitative criteria.

```python
result = everyrow.rank(
    task="Likelihood to convert in next quarter",
    input=leads_df,
    field_name="conversion_score",
    ascending_order=False,
)
# result.data is sorted by conversion_score
```

### Dedupe

Find and mark duplicate rows using semantic equivalence.

```python
result = everyrow.dedupe(
    equivalence_relation="Same person despite name variations or career changes",
    input=contacts_df,
)
# result.data has 'equivalence_class_id' and 'selected' columns
deduped = result.data[result.data["selected"]]
```

### Merge

Join two tables using intelligent entity matching.

```python
result = everyrow.merge(
    task="Match software products to parent companies",
    left_table=products_df,
    right_table=companies_df,
    merge_on_left="product_name",
    merge_on_right="company_name",
)
# result.data contains the merged rows
```

### Research

Run web research agents on each row.

```python
result = everyrow.research(
    task="Find this company's latest funding round and lead investors",
    input=companies_df,
)
# result.data has an 'answer' column with research results
```

## Asset Factories

For config-driven pipelines, use the asset factories:

```python
from dagster_everyrow import everyrow_screen_asset, everyrow_rank_asset

# Creates an asset that screens the input
screened_leads = everyrow_screen_asset(
    name="screened_leads",
    ins={"raw_leads": dg.AssetIn()},
    task="Remote-friendly senior role with disclosed salary",
    input_fn=lambda raw_leads: raw_leads,
    description="Leads filtered for remote senior roles",
)

# Creates an asset that ranks the input
scored_leads = everyrow_rank_asset(
    name="scored_leads",
    ins={"screened_leads": dg.AssetIn()},
    task="Likelihood to convert in next quarter",
    field_name="conversion_score",
    input_fn=lambda screened_leads: screened_leads,
)
```

Available factories:
- `everyrow_screen_asset`
- `everyrow_rank_asset`
- `everyrow_dedupe_asset`
- `everyrow_merge_asset`
- `everyrow_research_asset`

## Observability

Every everyrow operation logs rich metadata to Dagster:

- **Session URL**: Clickable link to inspect per-row reasoning in the everyrow web UI
- **Row counts**: Input/output row counts, duplicates found, etc.

This means your data team lead can look at the Dagster UI, click into a materialization, and immediately see _why_ the LLM scored things the way it did.

## Development

```bash
cd dagster-everyrow
uv sync
uv run pytest
```

## License

MIT - See [LICENSE.txt](../LICENSE.txt)
