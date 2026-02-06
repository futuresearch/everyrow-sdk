"""Pandas DataFrame accessor for everyrow operations.

Provides a convenient `df.everyrow` accessor for AI-powered DataFrame operations.

The accessor is automatically registered when importing anything from everyrow.

Example:
    >>> import pandas as pd
    >>> from everyrow import create_session  # Registers the accessor
    >>>
    >>> df = pd.DataFrame({"company": ["Apple Inc", "Google LLC", "Meta"]})
    >>> result = await df.everyrow.screen("Filter to companies with 'Inc' or 'LLC' in name")
    >>> print(result)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
from pydantic import BaseModel

from everyrow.ops import (
    agent_map,
    dedupe,
    merge,
    rank,
    screen,
    single_agent,
)
from everyrow.result import Result, TableResult
from everyrow.session import Session
from everyrow.task import EffortLevel

if TYPE_CHECKING:
    pass


@pd.api.extensions.register_dataframe_accessor("everyrow")
class EveryrowAccessor:
    """Pandas DataFrame accessor for everyrow operations.

    Access via `df.everyrow.<method>()` after importing everyrow.
    All methods are async and should be awaited.

    Attributes:
        last_result: The full result object from the last operation,
            containing artifact_id, error info, and (for merge) breakdown.

    Example:
        >>> df = pd.DataFrame({"job": ["ML Engineer", "Data Scientist", "PM"]})
        >>> filtered = await df.everyrow.screen("Filter to ML/AI roles")
        >>> print(df.everyrow.last_result.artifact_id)  # Access metadata
    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._df = pandas_obj
        self._session: Session | None = None
        self._last_result: Result | None = None

    @property
    def last_result(self) -> Result | None:
        """The full result object from the last operation.

        Returns TableResult, MergeResult, or ScalarResult depending on the operation.
        Useful for accessing artifact_id, error messages, or merge breakdown.
        """
        return self._last_result

    def with_session(self, session: Session) -> EveryrowAccessor:
        """Set the session for subsequent operations.

        Args:
            session: An everyrow Session from create_session()

        Returns:
            self for method chaining

        Example:
            >>> async with create_session() as session:
            ...     result = await df.everyrow.with_session(session).screen("...")
        """
        self._session = session
        return self

    # --- Screen ---

    async def screen(
        self,
        task: str,
        response_model: type[BaseModel] | None = None,
    ) -> pd.DataFrame:
        """Screen rows using AI based on natural language criteria.

        Args:
            task: Natural language description of screening criteria.
            response_model: Optional Pydantic model for structured output.
                Defaults to a model with a single `passes: bool` field.

        Returns:
            DataFrame with rows that pass the screening criteria,
            plus additional columns from the response model.

        Example:
            >>> jobs = pd.DataFrame({
            ...     "title": ["ML Engineer", "PM", "Data Scientist"],
            ...     "salary": [150000, 120000, 140000]
            ... })
            >>> await jobs.everyrow.screen("Filter to ML/AI roles with salary > 130k")
        """
        result = await screen(
            task=task,
            session=self._session,
            input=self._df,
            response_model=response_model,
        )
        self._last_result = result
        return result.data

    # --- Rank ---

    async def rank(
        self,
        field_name: str,
        task: str,
        field_type: Literal["float", "int", "str", "bool"] = "float",
        ascending: bool = True,
        response_model: type[BaseModel] | None = None,
    ) -> pd.DataFrame:
        """Rank rows using AI based on natural language criteria.

        Args:
            field_name: Name of the field to sort by (will be added to output).
            task: Natural language description of ranking criteria.
            field_type: Type of the ranking field. Defaults to "float".
            ascending: Sort order. Defaults to True (lowest first).
            response_model: Optional Pydantic model for structured output.
                Must include field_name if provided.

        Returns:
            DataFrame sorted by the AI-generated ranking field.

        Example:
            >>> candidates = pd.DataFrame({
            ...     "name": ["Alice", "Bob", "Carol"],
            ...     "resume": ["10 years ML...", "2 years PM...", "5 years DS..."]
            ... })
            >>> await candidates.everyrow.rank(
            ...     "fit_score",
            ...     task="Rank candidates by fit for senior ML engineer role",
            ...     ascending=False  # Best candidates first
            ... )
        """
        result = await rank(
            task=task,
            session=self._session,
            input=self._df,
            field_name=field_name,
            field_type=field_type,
            response_model=response_model,
            ascending_order=ascending,
        )
        self._last_result = result
        return result.data

    # --- Dedupe ---

    async def dedupe(self, equivalence_relation: str) -> pd.DataFrame:
        """Remove duplicate rows using AI-powered equivalence matching.

        Args:
            equivalence_relation: Natural language description of what makes
                two rows equivalent/duplicates.

        Returns:
            DataFrame with duplicates removed.

        Example:
            >>> companies = pd.DataFrame({
            ...     "name": ["Apple Inc", "Apple", "Google LLC", "Alphabet/Google"]
            ... })
            >>> await companies.everyrow.dedupe("Same company, ignoring legal suffixes")
        """
        result = await dedupe(
            equivalence_relation=equivalence_relation,
            session=self._session,
            input=self._df,
        )
        self._last_result = result
        return result.data

    # --- Merge ---

    async def merge(
        self,
        right: pd.DataFrame,
        task: str,
        left_on: str | None = None,
        right_on: str | None = None,
        use_web_search: Literal["auto", "yes", "no"] | None = None,
    ) -> pd.DataFrame:
        """Merge with another DataFrame using AI-powered matching.

        Args:
            right: The right DataFrame to merge with.
            task: Natural language description of the merge criteria.
            left_on: Optional column name in left (this) DataFrame to merge on.
            right_on: Optional column name in right DataFrame to merge on.
            use_web_search: Control web search behavior:
                - "auto": Try LLM first, then web search if needed (default)
                - "no": Skip web search entirely
                - "yes": Force web search on every row

        Returns:
            Merged DataFrame.

        Note:
            Access `df.everyrow.last_result.breakdown` after merge to see
            match statistics (exact, fuzzy, llm, web matches).

        Example:
            >>> subsidiaries = pd.DataFrame({"name": ["YouTube", "Instagram"]})
            >>> parents = pd.DataFrame({"company": ["Alphabet", "Meta"]})
            >>> merged = await subsidiaries.everyrow.merge(
            ...     parents,
            ...     task="Match subsidiaries to parent companies",
            ...     left_on="name",
            ...     right_on="company"
            ... )
            >>> print(subsidiaries.everyrow.last_result.breakdown)
        """
        result = await merge(
            task=task,
            session=self._session,
            left_table=self._df,
            right_table=right,
            merge_on_left=left_on,
            merge_on_right=right_on,
            use_web_search=use_web_search,
        )
        self._last_result = result
        return result.data

    # --- Agent Map ---

    async def agent_map(
        self,
        task: str,
        effort_level: EffortLevel | None = EffortLevel.LOW,
        response_model: type[BaseModel] | None = None,
    ) -> pd.DataFrame:
        """Run an AI agent on each row of the DataFrame.

        Args:
            task: Natural language instructions for the agent to execute per row.
            effort_level: Effort level preset (LOW, MEDIUM, HIGH).
                Controls LLM choice, iteration budget, and research.
                Defaults to LOW.
            response_model: Optional Pydantic model for structured output.

        Returns:
            DataFrame with agent results merged with input rows.

        Example:
            >>> companies = pd.DataFrame({"name": ["Apple", "Google"]})
            >>> enriched = await companies.everyrow.agent_map(
            ...     "Research this company and return their founding year and HQ city",
            ...     effort_level=EffortLevel.MEDIUM
            ... )
        """
        kwargs: dict = {
            "task": task,
            "session": self._session,
            "input": self._df,
            "effort_level": effort_level,
        }
        if response_model is not None:
            kwargs["response_model"] = response_model

        result = await agent_map(**kwargs)
        self._last_result = result
        return result.data

    # --- Single Agent ---

    async def single_agent(
        self,
        task: str,
        effort_level: EffortLevel | None = EffortLevel.LOW,
        response_model: type[BaseModel] | None = None,
    ) -> pd.DataFrame:
        """Run a single AI agent on the entire DataFrame.

        Unlike agent_map which runs per-row, single_agent processes the
        entire DataFrame as context for a single agent task.

        Args:
            task: Natural language instructions for the agent.
            effort_level: Effort level preset (LOW, MEDIUM, HIGH).
                Defaults to LOW.
            response_model: Optional Pydantic model for structured output.

        Returns:
            DataFrame with agent results.

        Example:
            >>> sales = pd.DataFrame({
            ...     "month": ["Jan", "Feb", "Mar"],
            ...     "revenue": [100, 150, 120]
            ... })
            >>> analysis = await sales.everyrow.single_agent(
            ...     "Analyze this sales data and identify trends",
            ...     effort_level=EffortLevel.MEDIUM
            ... )
        """
        kwargs: dict = {
            "task": task,
            "session": self._session,
            "input": self._df,
            "effort_level": effort_level,
            "return_table": True,
        }
        if response_model is not None:
            kwargs["response_model"] = response_model

        result = await single_agent(**kwargs)
        self._last_result = result
        if isinstance(result, TableResult):
            return result.data
        # ScalarResult - shouldn't happen with return_table=True but handle it
        return pd.DataFrame([{"result": result.data}])
