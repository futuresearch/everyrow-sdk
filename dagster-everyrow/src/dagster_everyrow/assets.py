"""Asset factories for everyrow operations."""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import dagster as dg
import pandas as pd
from pydantic import BaseModel

from dagster_everyrow.resource import EveryrowResource


def everyrow_screen_asset(
    name: str,
    ins: dict[str, dg.AssetIn],
    task: str,
    input_fn: Callable[..., pd.DataFrame],
    description: str | None = None,
    group_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    filter_passing: bool = True,
    key_prefix: str | Sequence[str] | None = None,
    deps: Sequence[dg.CoercibleToAssetKey] | None = None,
) -> dg.AssetsDefinition:
    """Factory to create an asset that screens data using everyrow.

    Args:
        name: Name of the output asset.
        ins: Dict mapping input names to AssetIn specs.
        task: Natural language screening criteria.
        input_fn: Function that receives input DataFrames and returns the
            DataFrame to screen. Receives kwargs matching the `ins` keys.
        description: Asset description.
        group_name: Asset group name.
        response_model: Optional Pydantic model for structured output.
        filter_passing: If True (default), return only rows where passes=True.
            If False, return all rows with the passes column.
        key_prefix: Optional prefix for the asset key.
        deps: Additional asset dependencies.

    Returns:
        A Dagster asset definition.

    Example:
        ```python
        screened_leads = everyrow_screen_asset(
            name="screened_leads",
            ins={"raw_leads": dg.AssetIn()},
            task="Remote-friendly AND senior-level AND salary disclosed",
            input_fn=lambda raw_leads: raw_leads,
            description="Leads filtered for remote senior roles",
        )
        ```
    """

    @dg.asset(
        name=name,
        ins=ins,
        description=description or f"Screen: {task[:80]}...",
        group_name=group_name,
        key_prefix=key_prefix,
        deps=deps,
    )
    def _asset(
        context: dg.AssetExecutionContext,
        everyrow: EveryrowResource,
        **kwargs: Any,
    ) -> pd.DataFrame:
        input_df = input_fn(**kwargs)
        result = everyrow.screen(
            task=task,
            input=input_df,
            response_model=response_model,
        )

        output_df: pd.DataFrame = result.data
        if filter_passing:
            output_df = pd.DataFrame(output_df[output_df["passes"]]).drop(
                columns=["passes"]
            )

        context.add_output_metadata(
            {
                "everyrow_session": dg.MetadataValue.url(result.session_url),
                "input_rows": len(input_df),
                "output_rows": len(output_df),
                "rows_filtered": len(input_df) - len(output_df),
            }
        )
        return output_df

    return _asset


def everyrow_rank_asset(
    name: str,
    ins: dict[str, dg.AssetIn],
    task: str,
    field_name: str,
    input_fn: Callable[..., pd.DataFrame],
    field_type: Literal["float", "int", "str", "bool"] = "float",
    ascending_order: bool = False,
    description: str | None = None,
    group_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    key_prefix: str | Sequence[str] | None = None,
    deps: Sequence[dg.CoercibleToAssetKey] | None = None,
) -> dg.AssetsDefinition:
    """Factory to create an asset that ranks data using everyrow.

    Args:
        name: Name of the output asset.
        ins: Dict mapping input names to AssetIn specs.
        task: Natural language ranking criteria.
        field_name: Name of the score field to add.
        input_fn: Function that receives input DataFrames and returns the
            DataFrame to rank.
        field_type: Type of the score field.
        ascending_order: Sort direction (default False = highest first).
        description: Asset description.
        group_name: Asset group name.
        response_model: Optional Pydantic model for structured output.
        key_prefix: Optional prefix for the asset key.
        deps: Additional asset dependencies.

    Returns:
        A Dagster asset definition.

    Example:
        ```python
        scored_leads = everyrow_rank_asset(
            name="scored_leads",
            ins={"raw_leads": dg.AssetIn()},
            task="Likelihood to need data integration solutions",
            field_name="integration_score",
            input_fn=lambda raw_leads: raw_leads,
        )
        ```
    """

    @dg.asset(
        name=name,
        ins=ins,
        description=description or f"Rank by: {task[:80]}...",
        group_name=group_name,
        key_prefix=key_prefix,
        deps=deps,
    )
    def _asset(
        context: dg.AssetExecutionContext,
        everyrow: EveryrowResource,
        **kwargs: Any,
    ) -> pd.DataFrame:
        input_df = input_fn(**kwargs)
        result = everyrow.rank(
            task=task,
            input=input_df,
            field_name=field_name,
            field_type=field_type,
            ascending_order=ascending_order,
            response_model=response_model,
        )

        metadata: dict[str, Any] = {
            "everyrow_session": dg.MetadataValue.url(result.session_url),
            "rows": len(result.data),
            "sorted_by": field_name,
            "ascending": ascending_order,
        }
        if field_type in ("float", "int") and len(result.data) > 0:
            col = result.data[field_name]
            metadata[f"max_{field_name}"] = float(col.max())  # type: ignore[arg-type]
            metadata[f"min_{field_name}"] = float(col.min())  # type: ignore[arg-type]
        context.add_output_metadata(metadata)
        return result.data

    return _asset


def everyrow_dedupe_asset(
    name: str,
    ins: dict[str, dg.AssetIn],
    equivalence_relation: str,
    input_fn: Callable[..., pd.DataFrame],
    description: str | None = None,
    group_name: str | None = None,
    select_representative: bool = True,
    key_prefix: str | Sequence[str] | None = None,
    deps: Sequence[dg.CoercibleToAssetKey] | None = None,
) -> dg.AssetsDefinition:
    """Factory to create an asset that dedupes data using everyrow.

    Args:
        name: Name of the output asset.
        ins: Dict mapping input names to AssetIn specs.
        equivalence_relation: Natural language description of what makes rows
            equivalent/duplicates.
        input_fn: Function that receives input DataFrames and returns the
            DataFrame to dedupe.
        description: Asset description.
        group_name: Asset group name.
        select_representative: If True (default), return one row per duplicate
            group. If False, return all rows with equivalence class info.
        key_prefix: Optional prefix for the asset key.
        deps: Additional asset dependencies.

    Returns:
        A Dagster asset definition.

    Example:
        ```python
        deduped_contacts = everyrow_dedupe_asset(
            name="deduped_contacts",
            ins={"raw_contacts": dg.AssetIn()},
            equivalence_relation="Same person despite name variations",
            input_fn=lambda raw_contacts: raw_contacts,
        )
        ```
    """

    @dg.asset(
        name=name,
        ins=ins,
        description=description or f"Dedupe: {equivalence_relation[:80]}...",
        group_name=group_name,
        key_prefix=key_prefix,
        deps=deps,
    )
    def _asset(
        context: dg.AssetExecutionContext,
        everyrow: EveryrowResource,
        **kwargs: Any,
    ) -> pd.DataFrame:
        input_df = input_fn(**kwargs)
        result = everyrow.dedupe(
            equivalence_relation=equivalence_relation,
            input=input_df,
        )

        output_df: pd.DataFrame = result.data
        if select_representative:
            output_df = pd.DataFrame(output_df[output_df["selected"]]).drop(
                columns=["equivalence_class_id", "selected"]
            )

        context.add_output_metadata(
            {
                "everyrow_session": dg.MetadataValue.url(result.session_url),
                "input_rows": len(input_df),
                "output_rows": len(output_df),
                "duplicates_removed": len(input_df) - len(output_df),
                "clusters": int(result.data["equivalence_class_id"].nunique()),  # type: ignore[arg-type]
            }
        )
        return output_df

    return _asset


def everyrow_merge_asset(
    name: str,
    ins: dict[str, dg.AssetIn],
    task: str,
    left_input_fn: Callable[..., pd.DataFrame],
    right_input_fn: Callable[..., pd.DataFrame],
    merge_on_left: str | None = None,
    merge_on_right: str | None = None,
    use_web_search: Literal["auto", "yes", "no"] | None = None,
    description: str | None = None,
    group_name: str | None = None,
    key_prefix: str | Sequence[str] | None = None,
    deps: Sequence[dg.CoercibleToAssetKey] | None = None,
) -> dg.AssetsDefinition:
    """Factory to create an asset that merges two tables using everyrow.

    Args:
        name: Name of the output asset.
        ins: Dict mapping input names to AssetIn specs. Must include inputs
            for both left and right tables.
        task: Natural language description of how to match rows.
        left_input_fn: Function that receives inputs and returns left DataFrame.
        right_input_fn: Function that receives inputs and returns right DataFrame.
        merge_on_left: Optional column name in left table as merge key.
        merge_on_right: Optional column name in right table as merge key.
        use_web_search: Control web search behavior.
        description: Asset description.
        group_name: Asset group name.
        key_prefix: Optional prefix for the asset key.
        deps: Additional asset dependencies.

    Returns:
        A Dagster asset definition.

    Example:
        ```python
        matched_products = everyrow_merge_asset(
            name="matched_products",
            ins={
                "products": dg.AssetIn(),
                "companies": dg.AssetIn(),
            },
            task="Match software products to parent companies",
            left_input_fn=lambda products, **_: products,
            right_input_fn=lambda companies, **_: companies,
            merge_on_left="product_name",
            merge_on_right="company_name",
        )
        ```
    """

    @dg.asset(
        name=name,
        ins=ins,
        description=description or f"Merge: {task[:80]}...",
        group_name=group_name,
        key_prefix=key_prefix,
        deps=deps,
    )
    def _asset(
        context: dg.AssetExecutionContext,
        everyrow: EveryrowResource,
        **kwargs: Any,
    ) -> pd.DataFrame:
        left_df = left_input_fn(**kwargs)
        right_df = right_input_fn(**kwargs)

        result = everyrow.merge(
            task=task,
            left_table=left_df,
            right_table=right_df,
            merge_on_left=merge_on_left,
            merge_on_right=merge_on_right,
            use_web_search=use_web_search,
        )

        context.add_output_metadata(
            {
                "everyrow_session": dg.MetadataValue.url(result.session_url),
                "left_rows": len(left_df),
                "right_rows": len(right_df),
                "output_rows": len(result.data),
            }
        )
        return result.data

    return _asset


def everyrow_research_asset(
    name: str,
    ins: dict[str, dg.AssetIn],
    task: str,
    input_fn: Callable[..., pd.DataFrame],
    description: str | None = None,
    group_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    key_prefix: str | Sequence[str] | None = None,
    deps: Sequence[dg.CoercibleToAssetKey] | None = None,
) -> dg.AssetsDefinition:
    """Factory to create an asset that enriches data via web research.

    Args:
        name: Name of the output asset.
        ins: Dict mapping input names to AssetIn specs.
        task: Natural language description of the research task.
        input_fn: Function that receives input DataFrames and returns the
            DataFrame to research.
        description: Asset description.
        group_name: Asset group name.
        response_model: Optional Pydantic model for structured output.
        key_prefix: Optional prefix for the asset key.
        deps: Additional asset dependencies.

    Returns:
        A Dagster asset definition.

    Example:
        ```python
        enriched_companies = everyrow_research_asset(
            name="enriched_companies",
            ins={"companies": dg.AssetIn()},
            task="Find latest funding round and lead investors",
            input_fn=lambda companies: companies,
        )
        ```
    """

    @dg.asset(
        name=name,
        ins=ins,
        description=description or f"Research: {task[:80]}...",
        group_name=group_name,
        key_prefix=key_prefix,
        deps=deps,
    )
    def _asset(
        context: dg.AssetExecutionContext,
        everyrow: EveryrowResource,
        **kwargs: Any,
    ) -> pd.DataFrame:
        input_df = input_fn(**kwargs)
        result = everyrow.research(
            task=task,
            input=input_df,
            response_model=response_model,
        )

        context.add_output_metadata(
            {
                "everyrow_session": dg.MetadataValue.url(result.session_url),
                "rows_researched": len(result.data),
            }
        )
        return result.data

    return _asset
