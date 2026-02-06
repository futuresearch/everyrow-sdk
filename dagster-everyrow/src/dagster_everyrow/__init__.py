"""Dagster integration for everyrow: AI-powered data operations at scale.

This package provides a Dagster resource and asset factories for everyrow's
core operations: screen, rank, dedupe, merge, and research.

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

from dagster_everyrow.assets import (
    everyrow_dedupe_asset,
    everyrow_merge_asset,
    everyrow_rank_asset,
    everyrow_research_asset,
    everyrow_screen_asset,
)
from dagster_everyrow.resource import (
    EveryrowResource,
    EveryrowResult,
)

__all__ = [
    "EveryrowResource",
    "EveryrowResult",
    "everyrow_dedupe_asset",
    "everyrow_merge_asset",
    "everyrow_rank_asset",
    "everyrow_research_asset",
    "everyrow_screen_asset",
]
