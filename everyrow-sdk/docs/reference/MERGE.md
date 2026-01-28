# Merge

Join two tables when the keys don't match exactly—or at all.

## The problem

You've got a list of software products and a list of approved vendors. You need to match them up. But:

- "Photoshop" → "Adobe" (zero string similarity)
- "Genentech" → "Roche" (subsidiary)
- "MSD" → "Merck" (regional name)
- "VS Code" → "Microsoft" (product vs company)

Fuzzy matching won't help—these aren't typos, they're semantic relationships. And the relationships aren't in any lookup table you can buy.

## How it works

You describe how the tables should match. Agents figure out the mapping.

```python
from everyrow.ops import merge

result = await merge(
    task="Match each software product to its parent company",
    left_table=software_products,
    right_table=approved_vendors,
    merge_on_left="product_name",
    merge_on_right="company_name",
)
print(result.data.head())
```

For ambiguous cases, add context:

```python
result = await merge(
    task="""
        Match clinical trial sponsors to parent pharma companies.

        Watch for:
        - Subsidiaries (Genentech → Roche, Janssen → J&J)
        - Regional names (MSD is Merck outside the US)
        - Abbreviations (BMS → Bristol-Myers Squibb)
    """,
    left_table=trials,
    right_table=pharma_companies,
    merge_on_left="sponsor",
    merge_on_right="company",
)
print(result.data.head())
```

## What you get back

A DataFrame with all left table columns plus matched right table columns. Rows that don't match get nulls for the right columns (like a left join).

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `task` | str | How to match the tables |
| `left_table` | DataFrame | Primary table (all rows kept) |
| `right_table` | DataFrame | Table to match from |
| `merge_on_left` | Optional[str] | Column in left table. Model will try to guess if not specified. |
| `merge_on_right` | Optional[str] | Column in right table. Model will try to guess if not specified. |
| `session` | Session | Optional, auto-created if omitted |

## Performance

| Size | Time | Cost |
|------|------|------|
| 100 × 50 | ~3 min | ~$2 |
| 2,000 × 50 | ~8 min | ~$9 |
| 1,000 × 1,000 | ~12 min | ~$15 |

## Case studies

- [Software Supplier Matching](https://futuresearch.ai/software-supplier-matching/) — 2,000 products to 50 vendors, 91% accuracy, zero false positives
- [HubSpot Contact Merge](https://futuresearch.ai/merge-hubspot-contacts/) — 99.9% recall despite GitHub handles, typos, and partial emails
- [CRM Merge Workflow](https://futuresearch.ai/crm-merge-workflow/) — joining fund-level and contact-level data
