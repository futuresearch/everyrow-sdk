# SEO Case Studies Plan

This document outlines 9 canonical technical problems that the everyrow SDK solves, along with target search queries for each. These will inform the creation of SEO-optimized documentation in `/docs/` designed to capture developer/data scientist search intent.

## Background

### Goal

Create documentation that surfaces in Google results for developers and data scientists searching for solutions to problems that everyrow solves. These should be **technical framings**, not business use cases.

### Target Audience

- Data scientists working with pandas
- ML engineers preparing training data
- Data engineers building ETL pipelines
- Analysts doing data cleaning/matching
- Developers working with messy datasets

### Competitive Landscape

| Tool                                    | What It Does                                    | Limitations                                                |
| --------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------- |
| **fuzzy_pandas, d6tjoin, fuzzymatcher** | String similarity joins                         | No semantic understanding; "Photoshop" can't match "Adobe" |
| **recordlinkage**                       | Probabilistic record linkage                    | Requires manual blocking rules and threshold tuning        |
| **Splink**                              | Scalable entity resolution                      | Requires training, complex setup, no LLM understanding     |
| **dedupe.io**                           | ML-based deduplication                          | Requires labeled training data                             |
| **LOTUS**                               | Semantic operators (sem_filter, sem_join, etc.) | LLM knowledge only—no web research, no dedupe operator     |
| **PandasAI**                            | Natural language to pandas code                 | Generates pandas code, doesn't do semantic operations      |

### everyrow Differentiators

1. **Web research agents** - actually looks things up, not just LLM reasoning
2. **Semantic deduplication** as first-class operation
3. **World knowledge** for joins - knows subsidiaries, parent companies, regional names
4. **Structured output** via Pydantic models
5. **No training data required** - works out of the box

---

## The 9 Canonical Technical Problems

### Framework: What's Missing from Pandas

| Pandas Operation             | Limitation                | Semantic Version (everyrow)                  |
| ---------------------------- | ------------------------- | -------------------------------------------- |
| `merge()`                    | Exact key match only      | `merge` - join by meaning/relationship       |
| `drop_duplicates()`          | Exact match only          | `dedupe` - find entities that are the same   |
| `query()` / boolean indexing | SQL-like conditions only  | `screen` - filter by understanding           |
| `apply()`                    | Function runs locally     | `agent_map` - apply function that researches |
| `sort_values()`              | Values must exist in data | `rank` - sort by computed/researched metric  |

---

## Problem 1: Fuzzy Join Without Common Keys

Status: Done, forked from existing python notebook

### The Technical Problem

You have two DataFrames that need to be joined, but there's no shared ID—just entity names that may not match exactly due to typos, abbreviations, or semantic relationships (product → company, subsidiary → parent).

### Why Existing Tools Fail

- `pd.merge()` requires exact key match
- Fuzzy join libraries (fuzzywuzzy, rapidfuzz) only do string similarity
- "Photoshop" → "Adobe" has zero string similarity but is a valid match
- No threshold works universally: 0.9 misses valid matches, 0.7 creates false positives

### Target Search Queries

| Query                                      | Estimated Intent                   |
| ------------------------------------------ | ---------------------------------- |
| `pandas merge without matching column`     | Primary - direct pandas pain point |
| `join dataframes by similar values python` | Fuzzy join intent                  |
| `fuzzy join two tables pandas`             | Explicit fuzzy join                |
| `merge tables without common key python`   | Variation                          |

### everyrow Operation

`merge`

### Proposed Doc Filename

`docs/fuzzy-join-without-keys.md`

---

## Problem 2: Entity Resolution in Python

### The Technical Problem

Match records across datasets that represent the same real-world entity, despite variations in naming, formatting, or data quality. This is the general case of matching entities across any data sources.

### Why Existing Tools Fail

- Splink and dedupe.io require training data and complex configuration
- recordlinkage requires manual blocking rules
- None use LLM understanding for semantic equivalence

### Target Search Queries

| Query                                      | Estimated Intent               |
| ------------------------------------------ | ------------------------------ |
| `entity resolution python`                 | Primary - classic term         |
| `record linkage pandas`                    | Academic/data engineering term |
| `match same entity across datasets python` | Descriptive                    |
| `python library for entity matching`       | Tool search                    |

### everyrow Operation

`merge` / `dedupe`

### Proposed Doc Filename

`docs/entity-resolution-python.md`

---

## Problem 3: Find Duplicate Rows by Meaning

### The Technical Problem

`pd.drop_duplicates()` only finds exact duplicates. You need to find rows that represent the same entity despite different representations: "AbbVie Inc" vs "Abbvie Pharmaceutical", "IBM" vs "Big Blue", name typos, abbreviations.

### Why Existing Tools Fail

- pandas only does exact deduplication
- Fuzzy matching requires threshold tuning that never works universally
- No tool understands that "Genentech" and "Roche" might be duplicates (subsidiary relationship)

### Target Search Queries

| Query                              | Estimated Intent   |
| ---------------------------------- | ------------------ |
| `find similar duplicates pandas`   | Primary pain point |
| `deduplicate by meaning python`    | Semantic intent    |
| `semantic deduplication dataframe` | Technical term     |
| `fuzzy duplicate detection pandas` | Fuzzy approach     |

### everyrow Operation

`dedupe`

### Proposed Doc Filename

`docs/semantic-deduplication-dataframe.md`

---

## Problem 4: Deduplicate ML Training Data

### The Technical Problem

Near-duplicates in training data cause data leakage, overfitting, and memorization. You need to identify semantically similar examples that aren't exact matches—paraphrases, reformatted text, or records representing the same underlying entity.

### Why Existing Tools Fail

- pandas `drop_duplicates()` only catches exact matches
- datasketch (MinHash/LSH) works for near-exact text but not semantic similarity
- dedupe.io requires training data
- No tool handles "same meaning, different words" without manual setup

### Target Search Queries

| Query                                             | Estimated Intent           |
| ------------------------------------------------- | -------------------------- |
| `deduplicate training data python`                | Primary - ML practitioners |
| `remove near duplicates dataset machine learning` | Descriptive                |
| `clean training data duplicates`                  | Data cleaning framing      |
| `find duplicate examples in dataset ML`           | Detection framing          |

### everyrow Operation

`dedupe`

### Proposed Doc Filename

`docs/deduplicate-training-data-ml.md`

### Notes

This is a **high-value target**: large ML/HuggingFace audience, massive pain point, actively searching.

---

## Problem 5: Filter DataFrame by Natural Language Condition

### The Technical Problem

`df.query()` and boolean indexing only support SQL-like conditions. You need to filter rows based on criteria that require understanding: "is this a remote-friendly job posting" (not just contains "remote"), "is this comment sarcastic", "does this company operate in healthcare".

### Why Existing Tools Fail

- pandas can only filter on boolean expressions
- Regex/keyword matching has poor precision (68% in job posting case study)
- `"remote" in text` matches "No remote work available"

### Target Search Queries

| Query                                         | Estimated Intent           |
| --------------------------------------------- | -------------------------- |
| `filter dataframe by natural language python` | Primary - novel capability |
| `pandas query with semantic condition`        | Semantic framing           |
| `filter rows by meaning not regex`            | Pain point framing         |
| `natural language filter pandas`              | Direct search              |

### everyrow Operation

`screen`

### Proposed Doc Filename

`docs/natural-language-filter-dataframe.md`

### Notes

**Low competition, genuinely new capability.** LOTUS has `sem_filter` but everyrow adds web research for conditions requiring lookup.

---

## Problem 6: Apply LLM Classification to Every Row

### The Technical Problem

You want to label or classify each row semantically—categorize into types, apply tags, detect attributes. There's no pandas-native way to do batched LLM classification with structured output.

### Why Existing Tools Fail

- Raw API calls require manual batching, retries, error handling
- No structured output enforcement
- PandasAI generates code, doesn't do semantic operations on data

### Target Search Queries

| Query                                    | Estimated Intent  |
| ---------------------------------------- | ----------------- |
| `classify each row dataframe LLM python` | Primary           |
| `batch label data with GPT`              | Practical framing |
| `apply LLM to dataframe rows`            | Operation framing |
| `auto label dataset python LLM`          | Labeling intent   |

### everyrow Operation

`screen` (with categorical response model)

### Proposed Doc Filename

`docs/llm-classification-dataframe.md`

---

## Problem 7: Add Column by Looking Up Data for Each Row

### The Technical Problem

`df.apply()` runs a local function. But what if computing each value requires looking something up—GitHub stats, citation counts, company metadata, package versions? You need a research agent per row.

### Why Existing Tools Fail

- `df.apply()` with requests is fragile, slow, doesn't handle complex research
- No retry logic, rate limiting, or caching
- Can't handle research that requires multiple lookups or reasoning
- LOTUS `sem_map` uses LLM knowledge only—can't look things up

### Target Search Queries

| Query                                      | Estimated Intent |
| ------------------------------------------ | ---------------- |
| `add column from API call each row pandas` | Primary pattern  |
| `web scrape per row dataframe python`      | Scraping framing |
| `batch lookup for dataframe rows`          | Batch operation  |
| `pandas apply with web request`            | Direct pattern   |

### Developer Examples (Not Business)

- Add GitHub stars to a list of repos
- Add citation count to a list of papers
- Add package size to a list of npm packages
- Add latest release date to software list
- Add license type to open source projects

### everyrow Operation

`agent_map`

### Proposed Doc Filename

`docs/add-column-web-lookup.md`

---

## Problem 8: Sort/Rank by Metric That Doesn't Exist in Data

### The Technical Problem

`df.sort_values()` requires the column to exist. What if you want to sort by "maintenance activity", "research impact", or "code quality"—metrics that must be computed via external lookup or research?

### Why Existing Tools Fail

- pandas requires values to exist before sorting
- No tool combines "research per row" + "sort by result"

### Target Search Queries

| Query                                     | Estimated Intent      |
| ----------------------------------------- | --------------------- |
| `rank dataframe by computed value python` | Primary               |
| `sort by external metric pandas`          | External data framing |
| `score rows via lookup python`            | Scoring framing       |
| `compute ranking from web data python`    | Research framing      |

### Developer Examples (Not Leads)

- Rank Python packages by maintenance activity (look up commits, issues)
- Rank papers by citation count (look up citations)
- Rank repos by community health (look up contributors, stars, PRs)
- Rank models by benchmark performance (look up leaderboard)
- Rank datasets by quality metrics (look up stats, usage)

### everyrow Operation

`rank`

### Proposed Doc Filename

`docs/rank-by-external-metric.md`

---

## Problem 9: Validate/Refresh Data Against External Truth

### The Technical Problem

Your data may be stale, inaccurate, or unverified. You need to check each row against external sources: Is this company still active? Is this URL still live? Has this person changed jobs? Is this the current CEO?

### Why Existing Tools Fail

- No pandas operation for "verify this value is still true"
- Manual validation doesn't scale
- APIs have fixed schemas that may not match your validation needs

### Target Search Queries

| Query                                               | Estimated Intent     |
| --------------------------------------------------- | -------------------- |
| `validate dataframe against external source python` | Primary              |
| `verify data accuracy python`                       | Verification framing |
| `refresh stale data dataframe`                      | Refresh framing      |
| `check if data is current python`                   | Currency check       |

### Developer Examples

- Verify company names are real/still active
- Check if URLs are still live
- Validate email domains exist
- Confirm people are still at listed companies
- Check if packages are still maintained

### everyrow Operation

`screen` (filter to valid rows) or `agent_map` (add validation columns)

### Proposed Doc Filename

`docs/validate-data-external-source.md`

---

## Summary Table

| #   | Problem                         | Primary Search Query                         | everyrow Op      | Doc Filename                           |
| --- | ------------------------------- | -------------------------------------------- | ---------------- | -------------------------------------- |
| 1   | Fuzzy join without keys         | `pandas merge without matching column`       | merge            | `fuzzy-join-without-keys.md`           |
| 2   | Entity resolution               | `entity resolution python`                   | merge/dedupe     | `entity-resolution-python.md`          |
| 3   | Find duplicates by meaning      | `find similar duplicates pandas`             | dedupe           | `semantic-deduplication-dataframe.md`  |
| 4   | Dedupe ML training data         | `deduplicate training data python`           | dedupe           | `deduplicate-training-data-ml.md`      |
| 5   | Natural language filter         | `filter dataframe natural language python`   | screen           | `natural-language-filter-dataframe.md` |
| 6   | LLM classification at scale     | `classify each row dataframe LLM python`     | screen           | `llm-classification-dataframe.md`      |
| 7   | Add column via web lookup       | `add column web lookup each row pandas`      | agent_map        | `add-column-web-lookup.md`             |
| 8   | Sort by researched metric       | `rank dataframe by external metric`          | rank             | `rank-by-external-metric.md`           |
| 9   | Validate against external truth | `validate dataframe against external source` | screen/agent_map | `validate-data-external-source.md`     |

---

## Next Steps

1. **Validate search volume** - Use Ahrefs/SEMrush/Google Keyword Planner to check volume for primary queries
2. **Check existing SERP quality** - Are current top results actually solving the problem?
3. **Draft case studies** - Create docs with:
   - Clear problem statement matching search intent
   - Why existing tools fail
   - Code example with everyrow
   - Comparison to alternatives
4. **Internal linking** - Cross-reference between related docs
5. **README updates** - Link to new docs from main README

---

## Search Volume Validation Tools

| Tool                   | Access                       | Best For                            |
| ---------------------- | ---------------------------- | ----------------------------------- |
| Google Keyword Planner | Free with Google Ads account | Volume ranges                       |
| Ahrefs                 | Paid                         | Volume + difficulty + SERP analysis |
| SEMrush                | Paid                         | Similar to Ahrefs                   |
| Google Trends          | Free                         | Relative interest over time         |
| AlsoAsked.com          | Free tier                    | "People Also Ask" queries           |
| AnswerThePublic        | Free tier                    | Question variations                 |

---

## Appendix: Competitive Positioning

### vs. LOTUS (Stanford)

- LOTUS has `sem_filter`, `sem_join`, `sem_map`, `sem_topk`, `sem_agg`
- LOTUS uses **LLM knowledge only**—no web research
- LOTUS has **no deduplication operator**
- everyrow adds: web research agents, dedupe, world knowledge for joins

### vs. Splink/dedupe.io

- Require training data and manual configuration
- Don't use LLM understanding
- everyrow works out of the box with natural language

### vs. PandasAI

- PandasAI generates pandas code from natural language
- everyrow does semantic operations _on the data_
- Different use cases: PandasAI for "write me a groupby", everyrow for "filter by meaning"
