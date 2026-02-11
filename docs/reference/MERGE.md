---
title: merge
description: API reference for EveryRow merge tool, which left-joins two Python Pandas DataFrames using LLM-powered agents to resolve key mappings.
---

# Merge

`merge` left-joins two DataFrames using LLM-powered agents to resolve the key mapping instead of requiring exact or fuzzy key matches. Agents resolve semantic relationships by reasoning over the data and, when needed, searching the web for external information to establish matches: subsidiaries, regional names, abbreviations, and product-to-parent-company mappings.

## Examples

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

## Related docs

### Guides
- [Fuzzy Join Without Matching Keys](/fuzzy-join-without-keys)

### Notebooks
- [LLM Merging at Scale](/notebooks/llm-powered-merging-at-scale)
- [Match Software Vendors to Requirements](/notebooks/match-software-vendors-to-requirements)
- [Merge Contacts with Company Data](/notebooks/merge-contacts-with-company-data)
- [Merge Overlapping Contact Lists](/notebooks/merge-overlapping-contact-lists)

### Blog posts
- [Software Supplier Matching](https://futuresearch.ai/software-supplier-matching/)
- [HubSpot Contact Merge](https://futuresearch.ai/merge-hubspot-contacts/)
- [CRM Merge Workflow](https://futuresearch.ai/crm-merge-workflow/)
