"""Polars DataFrame namespace extension for everyrow operations."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

import polars as pl
from everyrow.ops import agent_map, dedupe, merge, rank, screen
from everyrow.task import EffortLevel

if TYPE_CHECKING:
    from pydantic import BaseModel


@pl.api.register_dataframe_namespace("everyrow")
class EveryrowNamespace:
    """Everyrow operations as native Polars DataFrame methods.

    This namespace is automatically registered when you import everyrow_polars.
    All operations convert the Polars DataFrame to pandas internally (via Arrow),
    run the everyrow operation, and return a new Polars DataFrame.

    Example:
        import polars as pl
        import everyrow_polars

        df = pl.read_csv("leads.csv")
        screened = df.everyrow.screen("Remote-friendly senior role")
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def _to_pandas(self):
        """Convert the Polars DataFrame to pandas for everyrow operations."""
        return self._df.to_pandas()

    def _from_pandas(self, pdf) -> pl.DataFrame:
        """Convert a pandas DataFrame back to Polars."""
        return pl.from_pandas(pdf)

    def _run_sync(self, coro):
        """Run an async coroutine synchronously.

        Uses asyncio.run() which creates a new event loop. This is the cleanest
        approach for sync contexts. For async contexts (e.g., Jupyter with
        async cells), use the *_async() methods instead.
        """
        return asyncio.run(coro)

    # -------------------------------------------------------------------------
    # Screen
    # -------------------------------------------------------------------------

    def screen(
        self,
        task: str,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> pl.DataFrame:
        """Filter rows based on criteria that require LLM judgment.

        Evaluates each row against natural language criteria. Returns the
        original DataFrame with additional columns indicating whether each
        row passes the screen.

        Args:
            task: Natural language description of the screening criteria.
                Rows that meet the criteria will have `passes=True`.
            response_model: Optional Pydantic model for structured output.
                If not provided, returns a `passes` boolean column.

        Returns:
            A new Polars DataFrame with the original columns plus screening
            results (at minimum, a `passes` boolean column).

        Example:
            >>> df = pl.read_csv("job_postings.csv")
            >>> screened = df.everyrow.screen(
            ...     "Remote-friendly AND senior-level AND salary disclosed"
            ... )
            >>> remote_senior = screened.filter(pl.col("passes"))
        """
        return self._run_sync(
            self.screen_async(task=task, response_model=response_model)
        )

    async def screen_async(
        self,
        task: str,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> pl.DataFrame:
        """Async version of screen(). Use in async contexts like Jupyter."""
        pdf = self._to_pandas()
        result = await screen(task=task, input=pdf, response_model=response_model)
        return self._from_pandas(result.data)

    # -------------------------------------------------------------------------
    # Rank
    # -------------------------------------------------------------------------

    def rank(
        self,
        task: str,
        *,
        field_name: str = "score",
        field_type: Literal["float", "int", "str", "bool"] = "float",
        descending: bool = False,
        response_model: type[BaseModel] | None = None,
    ) -> pl.DataFrame:
        """Score and sort rows based on qualitative criteria.

        Evaluates each row and assigns a score based on the task description,
        then sorts the table by that score.

        Args:
            task: Natural language description of the ranking criteria.
                Describes what makes a row score higher or lower.
            field_name: Name of the score column to add. Defaults to "score".
            field_type: Type of the score field. Defaults to "float".
            descending: If True, sort highest scores first. Defaults to False.
            response_model: Optional Pydantic model for structured output.
                Must include `field_name` as a property if provided.

        Returns:
            A new Polars DataFrame sorted by the score, with the score column added.

        Example:
            >>> df = pl.read_csv("leads.csv")
            >>> ranked = df.everyrow.rank(
            ...     "Likelihood to need data integration solutions",
            ...     field_name="integration_score",
            ...     descending=True,
            ... )
            >>> top_leads = ranked.head(20)
        """
        return self._run_sync(
            self.rank_async(
                task=task,
                field_name=field_name,
                field_type=field_type,
                descending=descending,
                response_model=response_model,
            )
        )

    async def rank_async(
        self,
        task: str,
        *,
        field_name: str = "score",
        field_type: Literal["float", "int", "str", "bool"] = "float",
        descending: bool = False,
        response_model: type[BaseModel] | None = None,
    ) -> pl.DataFrame:
        """Async version of rank(). Use in async contexts like Jupyter."""
        pdf = self._to_pandas()
        result = await rank(
            task=task,
            input=pdf,
            field_name=field_name,
            field_type=field_type,
            response_model=response_model,
            ascending_order=not descending,
        )
        return self._from_pandas(result.data)

    # -------------------------------------------------------------------------
    # Dedupe
    # -------------------------------------------------------------------------

    def dedupe(
        self,
        equivalence_relation: str,
    ) -> pl.DataFrame:
        """Remove duplicate rows using semantic equivalence.

        Identifies rows that represent the same entity even when they don't
        match exactly. Returns the DataFrame with additional columns marking
        equivalence classes and selected representatives.

        Args:
            equivalence_relation: Natural language description of what makes
                two rows equivalent/duplicates.

        Returns:
            A new Polars DataFrame with additional columns:
            - `equivalence_class_id`: ID grouping equivalent rows
            - `selected`: Boolean indicating the representative row per group

        Example:
            >>> df = pl.read_csv("contacts.csv")
            >>> deduped = df.everyrow.dedupe(
            ...     "Same person despite name variations or career changes"
            ... )
            >>> unique_contacts = deduped.filter(pl.col("selected"))
        """
        return self._run_sync(
            self.dedupe_async(equivalence_relation=equivalence_relation)
        )

    async def dedupe_async(
        self,
        equivalence_relation: str,
    ) -> pl.DataFrame:
        """Async version of dedupe(). Use in async contexts like Jupyter."""
        pdf = self._to_pandas()
        result = await dedupe(equivalence_relation=equivalence_relation, input=pdf)
        return self._from_pandas(result.data)

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    def merge(
        self,
        right: pl.DataFrame,
        task: str,
        *,
        left_on: str | None = None,
        right_on: str | None = None,
        use_web_search: Literal["auto", "yes", "no"] | None = None,
    ) -> pl.DataFrame:
        """Join with another DataFrame using intelligent entity matching.

        Combines two tables even when keys don't match exactly. The LLM
        performs reasoning (and optionally web research) to identify which
        rows should be joined.

        Args:
            right: The right DataFrame to merge with.
            task: Natural language description of how to match rows.
            left_on: Optional column name in this DataFrame to merge on.
            right_on: Optional column name in the right DataFrame to merge on.
            use_web_search: Control web search behavior:
                - "auto": Try LLM first, then search if needed (default)
                - "no": Skip web search entirely
                - "yes": Force web search on every row

        Returns:
            A new Polars DataFrame with matched rows joined.

        Example:
            >>> products = pl.read_csv("software_products.csv")
            >>> suppliers = pl.read_csv("approved_suppliers.csv")
            >>> matched = products.everyrow.merge(
            ...     suppliers,
            ...     "Match each software product to its parent company",
            ...     left_on="product_name",
            ...     right_on="company_name",
            ... )
        """
        return self._run_sync(
            self.merge_async(
                right=right,
                task=task,
                left_on=left_on,
                right_on=right_on,
                use_web_search=use_web_search,
            )
        )

    async def merge_async(
        self,
        right: pl.DataFrame,
        task: str,
        *,
        left_on: str | None = None,
        right_on: str | None = None,
        use_web_search: Literal["auto", "yes", "no"] | None = None,
    ) -> pl.DataFrame:
        """Async version of merge(). Use in async contexts like Jupyter."""
        left_pdf = self._to_pandas()
        right_pdf = right.to_pandas()
        result = await merge(
            task=task,
            left_table=left_pdf,
            right_table=right_pdf,
            merge_on_left=left_on,
            merge_on_right=right_on,
            use_web_search=use_web_search,
        )
        return self._from_pandas(result.data)

    # -------------------------------------------------------------------------
    # Research (agent_map)
    # -------------------------------------------------------------------------

    def research(
        self,
        task: str,
        *,
        effort_level: Literal["low", "medium", "high"] = "low",
        response_model: type[BaseModel] | None = None,
    ) -> pl.DataFrame:
        """Run web research agents on each row.

        Performs web research and extraction tasks on each row independently.
        Useful for enriching data with information from the web.

        Args:
            task: Natural language description of the research task to perform
                on each row.
            effort_level: How much effort to spend per row:
                - "low": Quick lookup, minimal iterations
                - "medium": Moderate research depth
                - "high": Thorough research with multiple iterations
            response_model: Optional Pydantic model for structured output.
                If not provided, returns an `answer` string column.

        Returns:
            A new Polars DataFrame with the original columns plus research
            results (the response_model fields, or an `answer` column).

        Example:
            >>> df = pl.read_csv("companies.csv")
            >>> enriched = df.everyrow.research(
            ...     "Find this company's latest funding round and lead investors",
            ...     effort_level="medium",
            ... )
        """
        return self._run_sync(
            self.research_async(
                task=task, effort_level=effort_level, response_model=response_model
            )
        )

    async def research_async(
        self,
        task: str,
        *,
        effort_level: Literal["low", "medium", "high"] = "low",
        response_model: type[BaseModel] | None = None,
    ) -> pl.DataFrame:
        """Async version of research(). Use in async contexts like Jupyter."""
        pdf = self._to_pandas()
        effort = EffortLevel(effort_level)
        if response_model is not None:
            result = await agent_map(
                task=task,
                input=pdf,
                effort_level=effort,
                response_model=response_model,
            )
        else:
            result = await agent_map(
                task=task,
                input=pdf,
                effort_level=effort,
            )
        return self._from_pandas(result.data)
