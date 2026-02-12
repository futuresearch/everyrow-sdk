"""Dagster resource for everyrow operations."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import dagster as dg
import pandas as pd
from everyrow.ops import agent_map, dedupe, merge, rank, screen
from everyrow.result import TableResult
from everyrow.session import Session, create_session
from pydantic import BaseModel

if TYPE_CHECKING:
    pass

T = TypeVar("T")


class EveryrowResult:
    """Result wrapper that includes the session URL for Dagster metadata."""

    def __init__(self, data: pd.DataFrame, session_url: str):
        self.data = data
        self.session_url = session_url


class EveryrowResource(dg.ConfigurableResource):
    """Dagster resource for everyrow AI-powered data operations.

    Provides access to everyrow's core operations: screen, rank, dedupe, merge,
    and research (agent_map). Each operation returns results along with a session
    URL that can be logged as Dagster metadata for observability.

    Example:
        ```python
        import dagster as dg
        from dagster_everyrow import EveryrowResource

        defs = dg.Definitions(
            resources={
                "everyrow": EveryrowResource(
                    api_key=dg.EnvVar("EVERYROW_API_KEY"),
                ),
            },
            assets=[...],
        )
        ```
    """

    api_key: str

    def _run_async(self, coro: Awaitable[T]) -> T:
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
            # nest_asyncio is optional - only needed when running in an existing event loop
            try:
                import nest_asyncio  # type: ignore[import-not-found]  # noqa: PLC0415

                nest_asyncio.apply()
            except ImportError:
                pass
            return loop.run_until_complete(coro)  # type: ignore[arg-type]
        except RuntimeError:
            return asyncio.run(coro)  # type: ignore[arg-type]

    async def _run_with_session(
        self,
        operation_name: str,
        coro_factory: Callable[[Session], Awaitable[T]],
    ) -> tuple[T, str]:
        """Run an operation with an explicit session and return result + URL."""
        # Temporarily set the API key for the everyrow SDK
        old_key = os.environ.get("EVERYROW_API_KEY")
        os.environ["EVERYROW_API_KEY"] = self.api_key
        try:
            async with create_session(name=f"dagster-{operation_name}") as session:
                result = await coro_factory(session)
                return result, session.get_url()
        finally:
            if old_key is not None:
                os.environ["EVERYROW_API_KEY"] = old_key
            elif "EVERYROW_API_KEY" in os.environ:
                del os.environ["EVERYROW_API_KEY"]

    def screen(
        self,
        task: str,
        input: pd.DataFrame,
        response_model: type[BaseModel] | None = None,
    ) -> EveryrowResult:
        """Screen rows based on criteria that require judgment.

        Evaluates each row against natural language criteria and adds a `passes`
        column indicating whether the row meets the criteria.

        Args:
            task: Natural language description of the screening criteria.
            input: DataFrame to screen.
            response_model: Optional Pydantic model for structured output.

        Returns:
            EveryrowResult with screened data and session URL.

        Example:
            ```python
            result = everyrow.screen(
                task="Qualifies if remote-friendly AND senior-level",
                input=leads_df,
            )
            context.add_output_metadata({
                "everyrow_session": dg.MetadataValue.url(result.session_url),
            })
            return result.data[result.data["passes"]]
            ```
        """

        async def _run(session: Session) -> TableResult:
            return await screen(
                task=task,
                input=input,
                session=session,
                response_model=response_model,
            )

        result, url = self._run_async(self._run_with_session("screen", _run))
        return EveryrowResult(data=result.data, session_url=url)

    def rank(
        self,
        task: str,
        input: pd.DataFrame,
        field_name: str,
        field_type: Literal["float", "int", "str", "bool"] = "float",
        ascending_order: bool = True,
        response_model: type[BaseModel] | None = None,
    ) -> EveryrowResult:
        """Score and sort rows based on qualitative criteria.

        Evaluates each row and assigns a score, then sorts the table by that score.

        Args:
            task: Natural language description of the ranking criteria.
            input: DataFrame to rank.
            field_name: Name of the score field to add.
            field_type: Type of the score field.
            ascending_order: Sort direction.
            response_model: Optional Pydantic model for structured output.

        Returns:
            EveryrowResult with ranked data and session URL.

        Example:
            ```python
            result = everyrow.rank(
                task="Score by likelihood to need data integration",
                input=companies_df,
                field_name="integration_score",
                ascending_order=False,
            )
            context.add_output_metadata({
                "everyrow_session": dg.MetadataValue.url(result.session_url),
                "top_score": float(result.data[field_name].max()),
            })
            return result.data
            ```
        """

        async def _run(session: Session) -> TableResult:
            return await rank(
                task=task,
                input=input,
                field_name=field_name,
                field_type=field_type,
                ascending_order=ascending_order,
                session=session,
                response_model=response_model,
            )

        result, url = self._run_async(self._run_with_session("rank", _run))
        return EveryrowResult(data=result.data, session_url=url)

    def dedupe(
        self,
        equivalence_relation: str,
        input: pd.DataFrame,
    ) -> EveryrowResult:
        """Find and mark duplicate rows using semantic equivalence.

        Identifies rows that represent the same entity even when they don't match
        exactly. Returns the original table with added `equivalence_class_id` and
        `selected` columns.

        Args:
            equivalence_relation: Natural language description of what makes
                rows equivalent/duplicates.
            input: DataFrame to dedupe.

        Returns:
            EveryrowResult with dedupe annotations and session URL. Use
            `result.data[result.data["selected"]]` to get one row per group.

        Example:
            ```python
            result = everyrow.dedupe(
                equivalence_relation="Same person despite name variations",
                input=contacts_df,
            )
            deduped = result.data[result.data["selected"]]
            context.add_output_metadata({
                "everyrow_session": dg.MetadataValue.url(result.session_url),
                "duplicates_found": len(result.data) - len(deduped),
            })
            return deduped
            ```
        """

        async def _run(session: Session) -> TableResult:
            return await dedupe(
                equivalence_relation=equivalence_relation,
                input=input,
                session=session,
            )

        result, url = self._run_async(self._run_with_session("dedupe", _run))
        return EveryrowResult(data=result.data, session_url=url)

    def merge(
        self,
        task: str,
        left_table: pd.DataFrame,
        right_table: pd.DataFrame,
        merge_on_left: str | None = None,
        merge_on_right: str | None = None,
        use_web_search: Literal["auto", "yes", "no"] | None = None,
    ) -> EveryrowResult:
        """Join two tables using intelligent entity matching.

        Matches rows between tables using a combination of exact matching, fuzzy
        matching, LLM reasoning, and optional web research.

        Args:
            task: Natural language description of how to match rows.
            left_table: Primary DataFrame.
            right_table: Secondary DataFrame to merge in.
            merge_on_left: Optional column name in left table as merge key.
            merge_on_right: Optional column name in right table as merge key.
            use_web_search: Control web search behavior ("auto", "yes", "no").

        Returns:
            EveryrowResult with merged data and session URL.

        Example:
            ```python
            result = everyrow.merge(
                task="Match software products to parent companies",
                left_table=products_df,
                right_table=companies_df,
                merge_on_left="product_name",
                merge_on_right="company_name",
            )
            context.add_output_metadata({
                "everyrow_session": dg.MetadataValue.url(result.session_url),
            })
            return result.data
            ```
        """

        async def _run(session: Session) -> TableResult:
            return await merge(
                task=task,
                left_table=left_table,
                right_table=right_table,
                merge_on_left=merge_on_left,
                merge_on_right=merge_on_right,
                use_web_search=use_web_search,
                session=session,
            )

        result, url = self._run_async(self._run_with_session("merge", _run))
        return EveryrowResult(data=result.data, session_url=url)

    def research(
        self,
        task: str,
        input: pd.DataFrame,
        response_model: type[BaseModel] | None = None,
    ) -> EveryrowResult:
        """Run web research agents on each row of a table.

        Performs web research and extraction tasks on each row independently.
        Useful for enriching data with information from the web.

        Args:
            task: Natural language description of the research task.
            input: DataFrame with rows to research.
            response_model: Optional Pydantic model for structured output.

        Returns:
            EveryrowResult with enriched data and session URL.

        Example:
            ```python
            result = everyrow.research(
                task="Find this company's latest funding round",
                input=companies_df,
            )
            context.add_output_metadata({
                "everyrow_session": dg.MetadataValue.url(result.session_url),
                "rows_researched": len(result.data),
            })
            return result.data
            ```
        """

        async def _run(session: Session) -> TableResult:
            kwargs: dict[str, Any] = {
                "task": task,
                "input": input,
                "session": session,
            }
            if response_model is not None:
                kwargs["response_model"] = response_model
            return await agent_map(**kwargs)

        result, url = self._run_async(self._run_with_session("research", _run))
        return EveryrowResult(data=result.data, session_url=url)
