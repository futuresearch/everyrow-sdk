# How to Merge DataFrames Without a Matching Column in Python

When you need to join two pandas DataFrames but there's no shared ID column, `pd.merge()` won't help. Some techniques exist to do fuzzy matching on single columns, but this will miss the harder cases requiring semantic knowledge, and doesn't take advantage of data in other columns that give clues to which rows match.

This guide shows how to merge tables using LLM-powered understanding, up and to including using agentic websearch to get additional information, to get the highest quality match available. We show how to do it pretty cheaply, since naive LLM-based solutions can be extremely expensive in token costs.

In this example, we join two tables of 400-500 rows of company data, where one set has company names, and the other has stock tickers, or where names are spelled differently across sources.

| Metric         | Value       |
| -------------- | ----------- |
| Rows processed | 438         |
| Accuracy       | 100%        |
| Cost           | $1.00       |
| Time           | ~30 seconds |

```bash
pip install everyrow
export EVERYROW_API_KEY=your_key_here  # Get one at everyrow.io
```

We'll use two datasets of S&P 500 companies from different sources. Download [company_info.csv](data/company_info.csv) and [valuations.csv](data/valuations.csv), or run the [full notebook](case_studies/match-software-vendors-to-requirements/notebook.ipynb).

```python
import asyncio
import pandas as pd
from everyrow.ops import merge

# Two tables from different sources - no shared column
companies = pd.read_csv("company_info.csv")    # has: company, price, mkt_cap, shares
valuations = pd.read_csv("valuations.csv")      # has: ticker, fair_value

async def main():
    result = await merge(
        task="Match companies to their stock tickers",
        left_table=companies,
        right_table=valuations,
    )

    # The result is a DataFrame with all columns joined
    print(result.data.head())

    #                company   price      mkt_cap     shares  ticker  fair_value
    # 0                   3M  101.74  61.70678828  606514530     MMM       39.18
    # 1          A. O. Smith   32.38  4.904416495  151464376     AOS        6.59
    # 2  Abbott Laboratories   34.87  51.22933139 1469152033     ABT      119.19

asyncio.run(main())
```

The SDK figures out that "3M" corresponds to ticker "MMM", "Alphabet Inc." to "GOOGL", and so on. No merge columns are specified because there's nothing to match on directly.

The merge operation uses a cascade of matching strategies, stopping at the simplest one that works for each row:

| Strategy    | When Used                               | Cost        |
| ----------- | --------------------------------------- | ----------- |
| Exact match | Identical strings                       | Free        |
| Fuzzy match | Typos, case differences                 | Free        |
| LLM match   | Semantic equivalence (company → ticker) | ~$0.002/row |
| Web search  | Stale or obscure data                   | ~$0.01/row  |

For the company-to-ticker merge above, 99.8% of rows matched via LLM reasoning alone. The remaining 0.2% required a quick web lookup.

The same approach works when your data has typos or corruption. In testing with 10% character-level noise in company names (e.g., "Alphaeet Iqc." instead of "Alphabet Inc."), the cascade achieved 100% accuracy at $0.44 for 438 rows. The fuzzy matcher catches obvious typos, and the LLM handles cases where corruption makes string similarity unreliable.

This approach works well when your tables represent the same entities but use different identifiers: company names vs tickers, product names vs SKUs, subsidiary names vs parent companies. For tables that do share a common column, the SDK will use exact matching first and only escalate to more expensive methods when needed.

---

See the full analysis with multiple experiments in the [merge tutorial notebook](case_studies/match-software-vendors-to-requirements/notebook.ipynb).
