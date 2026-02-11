---
title: API Reference
description: Complete API reference for everyrow — screen, rank, dedupe, merge, and research operations powered by LLM web research agents.
---

# API Reference

Five operations for processing data with LLM-powered web research agents. Each takes a DataFrame and a natural-language instruction.

## screen

```python
result = await screen(task=..., input=df, response_model=Model)
```

`screen` takes a DataFrame and a natural-language filter predicate, evaluates each row using web research agents, and returns only the rows that pass. The filter condition does not need to be computable from existing columns. Agents can research external information to make the determination.

[Full reference →](/reference/SCREEN)
Guides: [Filter a DataFrame with LLMs](/filter-dataframe-with-llm)
Notebooks: [LLM Screening at Scale](/notebooks/llm-powered-screening-at-scale), [Screen Stocks by Investment Thesis](/notebooks/screen-stocks-by-investment-thesis)

## rank

```python
result = await rank(task=..., input=df, field_name="score")
```

`rank` takes a DataFrame and a natural-language scoring criterion, dispatches web research agents to compute a score for each row, and returns the DataFrame sorted by that score. The sort key does not need to exist in your data. Agents derive it at runtime by searching the web, reading pages, and reasoning over what they find.

[Full reference →](/reference/RANK)
Guides: [Sort a Dataset Using Web Data](/rank-by-external-metric)
Notebooks: [Score Leads from Fragmented Data](/notebooks/score-leads-from-fragmented-data), [Score Leads Without CRM History](/notebooks/score-leads-without-crm-history)

## dedupe

```python
result = await dedupe(input=df, equivalence_relation="...")
```

`dedupe` groups duplicate rows in a DataFrame based on a natural-language equivalence relation, assigns cluster IDs, and selects a canonical row per cluster. The duplicate criterion is semantic and LLM-powered: agents reason over the data and, when needed, search the web for external information to establish equivalence. This handles abbreviations, name variations, job changes, and entity relationships that no string similarity threshold can capture.

[Full reference →](/reference/DEDUPE)
Guides: [Remove Duplicates from ML Training Data](/deduplicate-training-data-ml), [Resolve Duplicate Entities](/resolve-entities-python)
Notebooks: [Dedupe CRM Company Records](/notebooks/dedupe-crm-company-records)

## merge

```python
result = await merge(task=..., left_table=df1, right_table=df2)
```

`merge` left-joins two DataFrames using LLM-powered agents to resolve the key mapping instead of requiring exact or fuzzy key matches. Agents resolve semantic relationships by reasoning over the data and, when needed, searching the web for external information to establish matches: subsidiaries, regional names, abbreviations, and product-to-parent-company mappings.

[Full reference →](/reference/MERGE)
Guides: [Fuzzy Join Without Matching Keys](/fuzzy-join-without-keys)
Notebooks: [LLM Merging at Scale](/notebooks/llm-powered-merging-at-scale), [Match Software Vendors to Requirements](/notebooks/match-software-vendors-to-requirements)

## agent_map / single_agent

```python
result = await agent_map(task=..., input=df)
```

`single_agent` runs one web research agent on a single input (or no input). `agent_map` runs an agent on every row of a DataFrame in parallel. Both dispatch agents that search the web, read pages, and return structured results. The transform is live web research: agents fetch and synthesize external information to populate new columns.

[Full reference →](/reference/RESEARCH)
Guides: [Add a Column with Web Lookup](/add-column-web-lookup), [Classify and Label Data with an LLM](/classify-dataframe-rows-llm)
Notebooks: [LLM Web Research Agents at Scale](/notebooks/llm-web-research-agents-at-scale), [Agent Map Regulatory Status](/notebooks/agent-map-regulatory-status)
