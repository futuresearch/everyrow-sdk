# Dedupe

Deduplicate when fuzzy matching can't cut it.

## The problem

You've got a messy CRM export. "AbbVie Inc", "Abbvie", and "AbbVie Pharmaceutical" are obviously the same company. So are "Big Blue" and "IBM Corporation". A person who was at "BAIR Lab" last year and "Google DeepMind" this year is still the same person.

Fuzzy matching forces you to pick a threshold. Set it high (0.9) and you miss "Big Blue" ↔ "IBM" (completely different strings). Set it low (0.7) and you false-positive on "John Smith" ↔ "Jane Smith". There's no threshold that works.

## How it works

You describe what "duplicate" means for your data. The system figures out which rows match.

```python
from everyrow.ops import dedupe

result = await dedupe(
    input=crm_data,
    equivalence_relation="Two entries are duplicates if they represent the same legal entity",
)
print(result.data.head())
```

The `equivalence_relation` is natural language. Be as specific as you need:

```python
result = await dedupe(
    input=researchers,
    equivalence_relation="""
        Two rows are duplicates if they're the same person, even if:
        - They changed jobs (different org/email)
        - Name is abbreviated (A. Smith vs Alex Smith)
        - There are typos (Naomi vs Namoi)
        - They use a nickname (Bob vs Robert)
    """,
)
print(result.data.head())
```

## What you get back

Three columns added to your data:

- `equivalence_class_id` — rows with the same ID are duplicates of each other
- `equivalence_class_name` — human-readable label for the cluster ("Alexandra Butoi", "Naomi Saphra", etc.)
- `selected` — True for the canonical record in each cluster (usually the most complete one)

To get just the deduplicated rows:

```python
deduped = result.data[result.data["selected"] == True]
```

## Example

Input:

| name | org | email |
|------|-----|-------|
| A. Butoi | Rycolab | a.butoi@edu |
| Alexandra Butoi | Ryoclab | — |
| Namoi Saphra | — | nsaphra@alumni |
| Naomi Saphra | Harvard | nsaphra@harvard.edu |

Output (selected rows only):

| name | org | email |
|------|-----|-------|
| Alexandra Butoi | Rycolab | a.butoi@edu |
| Naomi Saphra | Harvard | nsaphra@harvard.edu |

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `input` | DataFrame | Data with potential duplicates |
| `equivalence_relation` | str | What makes two rows duplicates |
| `session` | Session | Optional, auto-created if omitted |

## Performance

| Rows | Time | Cost |
|------|------|------|
| 200 | ~90 sec | ~$0.40 |
| 500 | ~2 min | ~$1.67 |
| 2,000 | ~8 min | ~$7 |

## Case studies

- [CRM Deduplication](https://futuresearch.ai/crm-deduplication/) — 500 rows down to 124 (75% were duplicates)
- [Researcher Deduplication](https://futuresearch.ai/researcher-dedupe-case-study/) — 98% accuracy handling career changes and typos
