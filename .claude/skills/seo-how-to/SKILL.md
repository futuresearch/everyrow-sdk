---
name: seo-how-to
description: Create SEO-optimized technical how-to docs for everyrow SDK. Finds datasets, runs SDK demos, measures metrics, and writes docs targeting developer search intent. Use when asked to "write a how-to", "create SEO doc", or "write technical case study".
---

# SEO How-To Doc Generator

Generate technical how-to documentation for one of the 9 canonical problems from `docs/SEO_CASE_STUDIES_PLAN.md`. These docs target developer/data scientist search intent with working code examples and real metrics.

**Output:** `docs/{filename}.md` - a technical doc with code snippets, metrics, and SEO-optimized content

---

## Before Running

1. Read the plan: `docs/SEO_CASE_STUDIES_PLAN.md` - understand all 9 problems and their search queries
2. Verify API key is set: `echo $EVERYROW_API_KEY` or check `.env`
3. Get today's date for the doc metadata

---

## Input

User must specify one of:

- **Problem number** (1-9) from the plan
- **Problem name** (e.g., "fuzzy join without keys", "dedupe training data")

If not specified, show the list:

```
Which problem should I create a how-to doc for?

1. Fuzzy join without keys (merge)
2. Entity resolution (merge/dedupe)
3. Find duplicates by meaning (dedupe)
4. Dedupe ML training data (dedupe)
5. Filter by natural language (screen)
6. LLM classification at scale (screen)
7. Add column via web lookup (agent_map)
8. Sort by researched metric (rank)
9. Validate against external truth (screen/agent_map)
```

---

## Phase 1: Understand the Problem

Read the problem details from `docs/SEO_CASE_STUDIES_PLAN.md`:

- **Target search queries** - what developers are searching for
- **Why existing tools fail** - the gap we're filling
- **everyrow operation** - which SDK function to use
- **Proposed filename** - the doc name

Summarize back to the user:

```
Problem {N}: {title}
Operation: {operation}
Primary search: "{query}"
Proposed doc: docs/{filename}.md
```

The end links should be akin to github.com/futuresearch/everyrow-sdk/docs/entity-resolution-pandas.md (or when possible, with a verb instead of a noun).

---

## Phase 2: Find or Create a Dataset

### Check Available Datasets

Look for datasets in these locations:

1. **aletheia_evals journeys** (delphos/cohort/engine/src/aletheia_evals/data/journeys/)

   - See `docs/SEO_CASE_STUDIES_PLAN.md` Appendix or pharos/.claude/skills/case-study/DATASETS.md for reference
   - These are structured with input.csv files ready to use

2. **everyrow-sdk case_studies/** - existing notebooks may have datasets

3. **futuresearch.ai/solutions examples/** - written up case studies might have what you need, though they won't have data, they'll have everything else

4. **Public datasets** - search for open datasets matching the problem

There is an agent for this you can invoke if necessary, delphos/pharos/.claude/agents/dataset-finder.md, but that tool is intended for a different use case. You could also use the python libraries referenced by that agent directly, if you want yfinance, wikipedia, etc. data.

In some cases, kaggle or huggingface might be the best by far. You might need to explore how to search and find datasets from there.

If you cannot produce a viable dataset, terminate early, inform the user that you failed, and ask for assistance, explaining what you tried and what you need. Do not fabricate data, or proceed with a poor quality or too-small dataset.

If you have multiple good options, explain and ask the user for their choice.

### Prepare the Dataset

1. Load the dataset
2. Select relevant columns (minimize to what's needed) or a subset of rows (the example should roughly be 200-500 rows for a rank/screen/map, 500-1500 rows for merge, 1500-5000 rows for dedupe, for a combination of reproducibility and affordability, the range between "large enough that doing it by hand is infeasible, but small enough that it costs <$20 to run.)
3. Save to `docs/data/{problem-name}/` if needed

Report to user:

```
Dataset: {source description}
Rows: {count}
Columns: {list}
Location: {path or "inline in script"}
```

---

## Phase 3: Run SDK on Small Subset (Proof of Concept)

Run the SDK on a small subset to:

1. Verify the operation works
2. Get example output
3. Measure approximate latency

Small here means:

Dedupe: Possibly whole dataset! Up to 1k rows
Merge: Possibly whole dataset, up to ~500 rows for each table
Screen: Up to 100 rows
Rank: Up to 20 rows
Agent Map: Up to 20 rows

### Write the Proof-of-Concept Script

Write to `/tmp/seo-poc-{problem}.py`:

Check the SDK docs. But try to make the solution complete, not just the part where you call the SDK. Need to fetch the data from wikipedia? Include that! Need to preprocess it in pandas? Do that. Need to run analysis on the output? Put that there too.

Definitely don't include a lot of boilerplate to make it seem more complete. It should be as short as possible, while being a complete reproducible how-to.

### Execute the POC

```bash
cd /path/to/everyrow-sdk
source .env 2>/dev/null || true
export EVERYROW_API_KEY
uv run python /tmp/seo-poc-{problem}.py
```

Note start and end time for latency measurement.

### Capture Results

Tell the user:

- **Session URL** - for reference and screenshots
- **Latency** - approximate runtime
- **Output sample** - first 5-10 rows
- **Key observations** - what worked, interesting edge cases

Then ask them to review. If they like it, they may ask you to scale it up, even at the cost of $$$ for the larger dataset, though should be under $20.

---

## Phase 4: Queue Full Run

Make sure the human is ok with the estimated cost (going from the cost of the trial run). Then implement the changes for the full run, and execute it.

Write the full script to `/tmp/seo-full-{problem}.py`

Make sure you capture latency numbers for this, which will likely go in the final piece.

### Required Metrics

| Metric             | Source                     | Notes           |
| ------------------ | -------------------------- | --------------- |
| **Cost**           | Session page (everyrow.io) | Total $ spent   |
| **Latency**        | User measurement           | Total runtime   |
| **Rows processed** | Script output              | Input row count |

### Optional Metrics (if applicable)

| Metric       | Source                       | Notes                             |
| ------------ | ---------------------------- | --------------------------------- |
| **Accuracy** | Manual review / ground truth | % correct matches/classifications |

### Compute Derived Metrics

```python
cost_per_row = total_cost / rows_processed
rows_per_second = rows_processed / latency_seconds
```

---

## Phase 5: Write the Final Code

Ensure the code is complete, from fetching/loading the dataset, pre-processing it, using the SDK, and analyzing the outputs. But keep it as short as possible that does this, though I'd include:

```bash
pip install everyrow
export EVERYROW_API_KEY=your_key_here  # Get at everyrow.io/api-key
```

---

## Phase 6: Write the Doc

Write to `docs/{filename}.md`. Conventions here are:

1. Minimize markdown headers. Should be mostly prose and code.
2. Avoid bullet point lists. Avoid bolding the first part of any sentence.
3. Keep it short and technically professional, but take the time to explain clearly what problem is being solved and what to expect in the solution. Lead with a summary of the metrics.

### SEO Considerations

1. **Title** should match the primary search query closely
2. **First paragraph** should include the query terms naturally
3. **Tables** for metrics and comparisons (scannable)

---

## Phase 7: Ask for Review

Once you're content everything is done as specified, Tell the user:

```

Doc written to: docs/{filename}.md

Summary:

- Problem: {title}
- Operation: {operation}
- Dataset: {description}
- Metrics: {cost}, {latency}, {accuracy if known}

Please review and:

1. Verify code snippets work
2. Check metrics are accurate
3. Add any screenshots from the session
4. Adjust prose as needed

Want me to make any changes?

```

---

## File Locations

everyrow-sdk/
├── docs/
│ ├── SEO_CASE_STUDIES_PLAN.md # The plan (read first)
│ ├── {filename}.md # Your output
│ └── data/ # Datasets if needed
│ └── {problem-name}/
│ └── input.csv
├── examples/ # Reference examples
└── .claude/skills/seo-how-to/ # This skill

## Example Invocation

```
User: "Create an SEO how-to doc for problem 4 (dedupe training data)"

Claude:
1. Reads SEO_CASE_STUDIES_PLAN.md for problem 4 details
2. Searches for ML training data datasets (aletheia_evals, HuggingFace, etc.)
3. Creates/selects appropriate dataset with near-duplicates
4. Writes and runs POC script on 30 rows
5. Reports results, asks user to run full 200-row version
6. User runs full version, reports: cost=$0.40, time=90s, 200→180 unique
7. Claude computes metrics and writes docs/deduplicate-training-data-ml.md
8. User reviews and makes final edits
```
