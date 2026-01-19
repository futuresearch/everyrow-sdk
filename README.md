![hero](https://github.com/user-attachments/assets/254fa2ed-c1f3-4ee8-b93d-d169edf32f27)

# <picture><img src="images/future-search-logo-128.webp" alt="FutureSearch" height="24" align="bottom"></picture> everyrow SDK

Python SDK for [everyrow.io](https://everyrow.io). Rank, dedupe, merge, and screen your dataframes using natural language—or run web agents to research every row.

## Table of Contents

New to everyrow? Head to [Getting Started](#getting-started)

Looking to use our agent-backed utilities? Check out:
- [Rank](#rank)
- [Dedupe](#dedupe)
- [Merge](#merge)
- [Screen](#screen)
- [Agent Tasks](#agent-tasks)

## Getting Started

Get an API key at [everyrow.io/api-key](https://everyrow.io/api-key), then set it in your environment:

```bash
export EVERYROW_API_KEY=your_api_key_here
```

### Installation

```bash
pip install everyrow
```

For development:

```bash
uv pip install -e .
uv sync
```

Requires Python >= 3.12

### Claude Code Plugin

There's a plugin for [Claude Code](https://code.claude.com/) that teaches Claude how to use the SDK:

```sh
# from Claude Code
/plugin marketplace add futuresearch/everyrow-sdk
/plugin install everyrow@futuresearch

# from terminal
claude plugin marketplace add futuresearch/everyrow-sdk
claude plugin install everyrow@futuresearch
```

## Rank

Score rows based on criteria you can't put in a database field. The AI researches each row and assigns scores based on qualitative factors.

```python
from everyrow.ops import rank

result = await rank(
    task="Score by likelihood to need data integration solutions",
    input=leads_dataframe,
    field_name="integration_need_score",
)
```

Say you want to rank leads by "likelihood to need data integration tools"—Ultramain Systems (sells software to airlines) looks similar to Ukraine International Airlines (is an airline) by industry code, but their actual needs are completely different. Traditional scoring can't tell them apart.

**Case studies:** [Lead Scoring with Data Fragmentation](https://futuresearch.ai/lead-scoring-data-fragmentation/) (1,000 leads, 7 min, $13) · [Lead Scoring Without CRM](https://futuresearch.ai/lead-scoring-without-crm/) ($28 vs $145 with Clay)

[Full documentation →](docs/RANK.md)

### Dedupe

Deduplicate when fuzzy matching falls short. The AI understands that "AbbVie Inc", "Abbvie", and "AbbVie Pharmaceutical" are the same company, or that "Big Blue" means IBM.

```python
from everyrow.ops import dedupe

result = await dedupe(
    input=crm_data,
    equivalence_relation="Two entries are duplicates if they represent the same legal entity",
)
```

The `equivalence_relation` tells the AI what counts as a duplicate—natural language, not regex. Results include `equivalence_class_id` (groups duplicates), `equivalence_class_name` (human-readable cluster name), and `selected` (the canonical record in each cluster).

**Case studies:** [CRM Deduplication](https://futuresearch.ai/crm-deduplication/) (500→124 rows, 2 min, $1.67) · [Researcher Deduplication](https://futuresearch.ai/researcher-dedupe-case-study/) (98% accuracy with career changes)

[Full documentation →](docs/DEDUPE.md)

### Merge

Join two tables when the keys don't match exactly—or at all. The AI knows "Photoshop" belongs to "Adobe" and "Genentech" is a Roche subsidiary, even with zero string similarity.

```python
from everyrow.ops import merge

result = await merge(
    task="Match each software product to its parent company",
    left_table=software_products,
    right_table=approved_suppliers,
    merge_on_left="software_name",
    merge_on_right="company_name",
)
```

Handles subsidiaries, abbreviations (MSD → Merck), regional names, typos, and pseudonyms. Fuzzy matching thresholds always fail somewhere—0.9 misses "Colfi" ↔ "Dr. Ioana Colfescu", 0.7 false-positives on "John Smith" ↔ "Jane Smith".

**Case studies:** [Software Supplier Matching](https://futuresearch.ai/software-supplier-matching/) (2,000 products, 91% accuracy, $9) · [HubSpot Contact Merge](https://futuresearch.ai/merge-hubspot-contacts/) (99.9% recall) · [CRM Merge Workflow](https://futuresearch.ai/crm-merge-workflow/)

[Full documentation →](docs/MERGE.md)

### Screen

Filter rows based on criteria that require research—things you can't express in SQL. The AI actually researches each row (10-Ks, earnings reports, news) before deciding pass/fail.

```python
from everyrow.ops import screen
from pydantic import BaseModel, Field

class ScreenResult(BaseModel):
    passes: bool = Field(description="True if company meets the criteria")

result = await screen(
    task="""
        Find companies with >75% recurring revenue that would benefit from
        Taiwan tensions - CHIPS Act beneficiaries, defense contractors,
        cybersecurity firms. Exclude companies dependent on Taiwan manufacturing.
    """,
    input=sp500_companies,
    response_model=ScreenResult,
)
```

Works for investment theses, geopolitical exposure, vendor risk assessment, job posting filtering, lead qualification—anything requiring judgment. Screening 500 S&P 500 companies takes ~12 min and $3 with >90% precision. Regex gets 68%.

**Case studies:** [Thematic Stock Screen](https://futuresearch.ai/thematic-stock-screening/) (63/502 passed, $3.29) · [Job Posting Screen](https://futuresearch.ai/job-posting-screening/) (>90% vs 68% regex) · [Lead Screening Workflow](https://futuresearch.ai/screening-workflow/)

[Full documentation →](docs/SCREEN.md)

### Agent Tasks

For single-input tasks, use `single_agent`. For batch processing, use `agent_map`.

```python
from everyrow.ops import single_agent, agent_map
from pandas import DataFrame

# Single input
result = await single_agent(
    task="What is the capital of the given country?",
    input={"country": "India"},
)

# Batch processing
result = await agent_map(
    task="What is the capital of the given country?",
    input=DataFrame([{"country": "India"}, {"country": "USA"}]),
)
```

Our agents are tuned on [Deep Research Bench](https://arxiv.org/abs/2506.06287), a benchmark we built for evaluating web research on questions that require extensive searching and cross-referencing.

## Advanced

### Sessions

For quick one-off operations, sessions are created automatically:

```python
from everyrow.ops import single_agent

result = await single_agent(
    task="What is the capital of France?",
    input={"country": "France"},
)
```

For multiple operations, use an explicit session:

```python
from everyrow import create_session

async with create_session(name="My Session") as session:
    print(f"View session at: {session.get_url()}")
    # All operations here share the same session
```

If you want more explicit control over the client (for example, to reuse it across sessions or configure custom settings), you can create it directly:

```python
from everyrow import create_client, create_session

async with create_client() as client:
    async with create_session(client=client, name="My Session") as session:
        # ...
```

Sessions are visible on the [everyrow.io](https://everyrow.io) dashboard.

### Async Operations

All utilities have async variants for background processing. These need an explicit session since the task persists beyond the function call:

```python
from everyrow import create_session
from everyrow.ops import rank_async

async with create_session(name="Async Ranking") as session:
    task = await rank_async(
        session=session,
        task="Score this organization",
        input=dataframe,
        field_name="score",
    )

    # Continue with other work...
    result = await task.await_result()
```

## Case Studies

More at [futuresearch.ai/solutions](https://futuresearch.ai/solutions/).

**Notebooks:**
- [CRM Deduplication](case_studies/dedupe/case_01_crm_data.ipynb)
- [Thematic Stock Screen](case_studies/screen/thematic_stock_screen.ipynb)
- [Oil Price Margin Screen](case_studies/screen/oil_price_margin_screen.ipynb)

**On futuresearch.ai:**
- [Lead Scoring with Data Fragmentation](https://futuresearch.ai/lead-scoring-data-fragmentation/)
- [Software Supplier Matching](https://futuresearch.ai/software-supplier-matching/)
- [Researcher Deduplication](https://futuresearch.ai/researcher-dedupe-case-study/)

To run notebooks:

```bash
uv sync --group case-studies
```

## Development

```bash
uv sync
lefthook install
```

```bash
uv run pytest              # tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run basedpyright        # type check
./generate_openapi.sh      # regenerate client
```

The `everyrow/generated/` directory is excluded from linting (auto-generated code).

## License

This project is licensed under the MIT License - see LICENSE.txt file for details.
