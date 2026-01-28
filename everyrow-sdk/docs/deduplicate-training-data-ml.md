# How to Deduplicate Training Data in Python

Near-duplicates in ML training data cause data leakage, overfitting, and memorization. This guide shows how to find and remove semantically similar examples that aren't exact matches—paraphrases, reformatted text, or records conveying the same information with different words.

| Metric             | Value                                                                     |
| ------------------ | ------------------------------------------------------------------------- |
| Input rows         | 3,000                                                                     |
| Unique after dedupe| 1,928                                                                     |
| Duplicates removed | 1,072 (35.7%)                                                             |
| Time               | 5.3 minutes                                                               |
| Cost               | $4.21                                                                     |
| Session            | [view](https://everyrow.io/sessions/ccaa306d-ef68-499b-a684-c0b08f9bfef3) |

Standard deduplication with `pandas.drop_duplicates()` only catches exact matches. MinHash/LSH (datasketch) works for near-exact text but not semantic similarity. Libraries like dedupe.io require labeled training data. None handle "same meaning, different words" without manual setup.

```bash
pip install everyrow datasets
export EVERYROW_API_KEY=your_key_here  # Get one at everyrow.io/api-key
```

```python
import asyncio
import pandas as pd
from datasets import load_dataset
from everyrow.ops import dedupe

# Load a dataset with potential semantic duplicates
# Using PAWS - paraphrase pairs from Wikipedia
dataset = load_dataset(
    "google-research-datasets/paws",
    "labeled_final",
    split="train"
)

# Extract sentences into a dataframe
sentences = []
seen = set()
for row in dataset:
    for s in [row["sentence1"], row["sentence2"]]:
        if s not in seen:
            seen.add(s)
            sentences.append(s)
        if len(sentences) >= 3000:
            break
    if len(sentences) >= 3000:
        break

df = pd.DataFrame({"text": sentences})
print(f"Training examples: {len(df)}")

async def dedupe_training_data():
    result = await dedupe(
        input=df,
        equivalence_relation="""
            Two sentences are duplicates if they convey the same meaning,
            even if phrased differently. This includes:
            - Paraphrases (same meaning, different words or word order)
            - Minor grammatical variations
            - Sentences about the same fact that would be redundant

            NOT duplicates if they describe different facts, even if
            they share many words.
        """,
    )

    # Get deduplicated dataset
    clean_df = result.data[result.data["selected"] == True]
    print(f"After deduplication: {len(clean_df)}")

    return clean_df

clean_data = asyncio.run(dedupe_training_data())
```

The output includes three columns added to your data: `equivalence_class_id` groups duplicates together, `equivalence_class_name` gives each cluster a readable label, and `selected` marks the canonical example to keep. Filter to `selected == True` to get your deduplicated dataset.

Here are examples of duplicates the system found:

```
Cluster: "Glenn Howard's Ontario Championship win"
  ✓ Glenn Howard won the Ontario Championship for the 17th time as either third or skip.
    For the 17th time the Glenn Howard won the Ontario Championship as third or skip.

Cluster: "Chananian village location"
  ✓ Chananian is a village in Azad Kashmir, the Leepa Valley, Hattian Bala District, Pakistan.
    Chananian is a village in Leepa Valley, Hattian Bala District of Azad Kashmir, Pakistan.
    Chananian is a village in the Leepa Valley, Hattian Bala district of Azad Kashmir, Pakistan.

Cluster: "Person's birth and death details"
  ✓ David Spurlock was born on 18 November 1959 in Dallas, Texas, and moved to Memphis...
    J. David Spurlock was born on November 18, 1959 in Dallas, Texas. He moved to Memphis...
```

These are semantic duplicates that exact-match deduplication would miss entirely. The sentences have different word order, date formats ("November 18" vs "18 November"), name variations ("David Spurlock" vs "J. David Spurlock"), and grammatical structure—but they describe the same facts and would be redundant in a training set.

The 35.7% reduction rate is typical for datasets that weren't explicitly deduplicated during creation. For datasets scraped from the web or aggregated from multiple sources, reduction rates can be higher. The cost scales linearly—expect roughly $1.40 per 1,000 rows for text data of this complexity.
