# How to Rank a DataFrame by a Metric That Requires The Web

`pandas.sort_values()` requires the column to already exist. EveryRow can rank or sort data on criteria you don't have in your dataset, if it can find it on the web. It's designed to do this as cost efficiently as possible.

This guide shows how to rank 300 PyPI packages by two different metrics that require external lookup: days since last release (from PyPI) and number of contributors (from GitHub).

| Metric                 | Rows | Cost  | Time        | Session                                                                   |
| ---------------------- | ---- | ----- | ----------- | ------------------------------------------------------------------------- |
| Days since release     | 300  | $3.90 | 4.3 minutes | [view](https://everyrow.io/sessions/24190033-4656-4366-86e9-79295c6f4510) |
| Number of contributors | 300  | $4.13 | 6.0 minutes | [view](https://everyrow.io/sessions/8b63da61-8597-45ae-ab8b-3b4d28dd1a33) |

```bash
pip install everyrow
export EVERYROW_API_KEY=your_key_here  # Get one at everyrow.io/api-key
```

The dataset is the top 300 PyPI packages by monthly downloads, fetched from the [top-pypi-packages](https://hugovk.github.io/top-pypi-packages/) API. The only columns are `package` and `monthly_downloads`—no release dates.

```python
import asyncio
import requests
import pandas as pd
from everyrow.ops import rank

# Fetch top PyPI packages
response = requests.get(
    "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json"
)
packages = response.json()["rows"][50:350]  # Skip AWS libs at top
df = pd.DataFrame(packages).rename(
    columns={"project": "package", "download_count": "monthly_downloads"}
)

async def main():
    result = await rank(
        task="""
            Find the number of days since this package's last release on PyPI.
            Look up the package on pypi.org to find the release date.
            Return the number of days as an integer.
        """,
        input=df,
        field_name="days_since_release",
        field_type="int",
        ascending_order=True,  # Most recent first
    )
    print(result.data[["package", "days_since_release"]])

asyncio.run(main())
```

```
                   package  days_since_release
0                pyparsing                   0
1                 httplib2                   1
2              yandexcloud                   2
3             multiprocess                   2
4                  pyarrow                   3
...
295        ptyprocess                1850
296              toml                1907
297               ply                2897
298      webencodings                3213
```

The SDK dispatched LLM-powered web research agents to review each row. They are flexible agents, so while in this case we have instructions to guide them where to look, they can be given open ended tasks, though they might use more tokens doing that, leading to higher costs. In this case, it found that `pyparsing` was released today (Jan 20 2026), and `webencodings` hasn't been updated in 8.8 years.

The same approach works for any metric you can describe. Here's the same dataset ranked by number of GitHub contributors:

```python
result = await rank(
    task="""
        Find the number of contributors to this package's GitHub repository.
        Look up the package's source repo from PyPI, then find the contributor
        count on GitHub. Return the number as an integer.
    """,
    input=df,
    field_name="num_contributors",
    field_type="int",
    ascending_order=False,  # Most contributors first
)
```

```
                    package  num_contributors
0                     torch              4191
1                 langchain              3858
2            langchain-core              3858
3              transformers              3608
4              scikit-learn              3157
...
295        jsonpath-ng                 2
296         et-xmlfile                 1
297     beautifulsoup4                 1
298        ruamel-yaml                 1
299            pkginfo                 1
```

`torch` has 4,191 contributors; `pkginfo` has 1. The task prompt tells the agent what to look up and where—citation counts, benchmark scores, API response times, or anything else you can describe.
