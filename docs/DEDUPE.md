# everyrow_sdk.dedupe Documentation

The `dedupe` operation deduplicates datasets by combining fuzzy string matching, embedding similarity, and the intelligence of LLMs to cluster similar items and select the best representative for each cluster.

## Sample Usage

### Code
```python
from everyrow import create_client, create_session
from everyrow.ops import dedupe
import pandas as pd

input_df = pd.read_csv("/case_studies/dedupe/case_02_researchers.csv")

async with create_client() as client:
    async with create_session(client, name="Researcher Dedupe") as session:
        result = await dedupe(
            session=session,
            input=input_df,
            equivalence_relation=(
                "Two rows are duplicates if they represent the same person "
                "despite different email/organization (career changes). "
                "Consider name variations like typos, nicknames (Robert/Bob), "
                "and format differences (John Smith/J. Smith)."
            ),
        )
        result.data.to_csv("deduplicated.csv", index=False)
```

The `equivalence_relation` parameter tells the AI what counts as a duplicate. Unlike regex or fuzzy matching, this is natural language that captures the semantic intent.

### Example Input

| row_id | name | organization | email | github |
|--------|------|--------------|-------|--------|
| 2 | A. Butoi | Rycolab | alexandra.butoi@personal.edu | butoialexandra |
| 8 | Alexandra Butoi | Ryoclab | — | butoialexandra |
| 43 | Namoi Saphra | — | nsaphra@alumni | nsaphra |
| 47 | Naomi Saphra | Harvard University | nsaphra@fas.harvard.edu | nsaphra |
| 18 | T. Gupta | AUTON Lab (Former) | — | tejus-gupta |
| 26 | Tejus Gupta | AUTON Lab | tejusg@cs.cmu.edu | tejus-gupta |

*Rows 2+8, 43+47, and 18+26 are duplicates (same person, different records).*

### Example Output

After running `dedupe`, duplicate rows are merged into canonical representatives:

| row_id | name | organization | email | github |
|--------|------|--------------|-------|--------|
| 2 | Alexandra Butoi | Rycolab | alexandra.butoi@personal.edu | butoialexandra |
| 43 | Naomi Saphra | Harvard University | nsaphra@fas.harvard.edu | nsaphra |
| 18 | Tejus Gupta | AUTON Lab | tejusg@cs.cmu.edu | tejus-gupta |

*The most complete/canonical record is selected as the representative for each equivalence class.*

## Case-studies
- [Deduplicating CRM data](case_studies/dedupe/case_01_crm_data.ipynb)

## Costs

The cost of the `dedupe` increases both with the number of rows, the amount of content per row, and the number of duplicates.

The following gives a broad overview of the cost of the `dedupe` operation:

![](docs/images/dedupe_cost.png)

## Details

The `dedupe` operation deduplicates data through a five-stage pipeline:

1. **Semantic Item Comparison**: Each row is compared against others using an LLM that understands context—recognizing that "A. Butoi" and "Alexandra Butoi" are likely the same person, or that "BAIR Lab (Former)" indicates a career transition rather than a different organization.

2. **Association Matrix Construction**: Pairwise comparison results are assembled into a matrix of match/no-match decisions. To scale efficiently, items are first clustered by embedding similarity, so only semantically similar items are compared.

3. **Equivalence Class Creation**: Connected components in the association graph form equivalence classes. If A matches B and B matches C, then A, B, and C form a single cluster representing one entity.

4. **Validation**: Each multi-member cluster is re-evaluated to catch false positives—cases where the initial comparison was too aggressive.

5. **Candidate Selection**: For each equivalence class, the most complete/canonical record is selected as the representative (e.g., preferring "Alexandra Butoi" over "A. Butoi").


## API Reference

### `dedupe(session, input, equivalence_relation)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `session` | `Session` | Session created via `create_session()` |
| `input` | `pd.DataFrame` | Input dataframe with potential duplicates |
| `equivalence_relation` | `str` | Natural language description of what makes two rows duplicates |
