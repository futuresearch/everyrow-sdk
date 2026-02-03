from typing import TypeVar
from uuid import UUID

import attrs
from pandas import DataFrame
from pydantic import BaseModel

T = TypeVar("T", bound=str | BaseModel)


@attrs.define
class ScalarResult[T: str | BaseModel]:
    artifact_id: UUID
    data: T
    error: str | None


@attrs.define
class TableResult:
    artifact_id: UUID
    data: DataFrame
    error: str | None


@attrs.define
class MergeBreakdown:
    """Breakdown of match methods for a merge operation.

    Each list contains (left_row_index, right_row_index) pairs using 0-based indices.
    """

    exact: list[tuple[int, int]]
    """Pairs matched via exact string match on merge columns."""

    fuzzy: list[tuple[int, int]]
    """Pairs matched via fuzzy string match (Levenshtein >= 0.9)."""

    llm: list[tuple[int, int]]
    """Pairs matched via direct LLM matching (no web search)."""

    web: list[tuple[int, int]]
    """Pairs matched via LLM with web research context."""

    unmatched_left: list[int]
    """Left row indices that had no match in the right table."""

    unmatched_right: list[int]
    """Right row indices that had no match in the left table."""


@attrs.define
class MergeResult:
    """Result of a merge operation including match breakdown.

    Example:
        result = await merge(task="...", left_table=df_left, right_table=df_right)
        print(f"Exact matches: {len(result.breakdown.exact)}")
        print(f"LLM matches: {len(result.breakdown.llm)}")
        print(f"Unmatched left rows: {result.breakdown.unmatched_left}")
    """

    artifact_id: UUID
    """The artifact ID of the merged table."""

    data: DataFrame
    """The merged DataFrame."""

    error: str | None
    """Error message if the task failed."""

    breakdown: MergeBreakdown
    """Match breakdown grouped by method (exact, fuzzy, llm, web)."""


Result = ScalarResult | TableResult | MergeResult
