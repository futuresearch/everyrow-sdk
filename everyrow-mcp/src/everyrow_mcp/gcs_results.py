"""GCS-backed result retrieval for the everyrow MCP server.

Handles checking Redis for cached GCS metadata, uploading new results to GCS,
and building the MCP TextContent responses for both paths.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import pandas as pd
from mcp.types import TextContent

from everyrow_mcp.models import PREVIEW_SIZE
from everyrow_mcp.redis_utils import build_key
from everyrow_mcp.state import RESULT_CACHE_TTL, state

if TYPE_CHECKING:
    from everyrow_mcp.gcs_storage import ResultURLs

logger = logging.getLogger(__name__)


def _format_columns(columns: list[str]) -> str:
    """Format column names for display, truncating after 10."""
    col_names = ", ".join(columns[:10])
    if len(columns) > 10:
        col_names += f", ... (+{len(columns) - 10} more)"
    return col_names


def _build_gcs_response(
    task_id: str,
    urls: ResultURLs,
    preview: list[dict],
    total: int,
    columns: list[str],
    offset: int,
    page_size: int,
) -> list[TextContent]:
    """Build MCP TextContent response for GCS-backed results."""
    col_names = _format_columns(columns)

    widget_json = json.dumps(
        {
            "results_url": urls.json_url,
            "preview": preview,
            "total": total,
        }
    )

    has_more = offset + page_size < total
    next_offset = offset + page_size if has_more else None

    if has_more:
        page_size_arg = f", page_size={page_size}" if page_size != PREVIEW_SIZE else ""
        summary = (
            f"Results: {total} rows, {len(columns)} columns ({col_names}). "
            f"Showing rows {offset + 1}-{min(offset + page_size, total)} of {total}.\n"
            f"Call everyrow_results(task_id='{task_id}', offset={next_offset}{page_size_arg}) for the next page."
        )
        if offset == 0:
            summary += (
                f"\nFull CSV download: {urls.csv_url}\n"
                "Share this download link with the user."
            )
    elif offset == 0:
        summary = (
            f"Results: {total} rows, {len(columns)} columns ({col_names}). "
            "All rows shown."
        )
    else:
        summary = (
            f"Results: showing rows {offset + 1}-{min(offset + page_size, total)} "
            f"of {total} (final page)."
        )

    return [
        TextContent(type="text", text=widget_json),
        TextContent(type="text", text=summary),
    ]


async def try_cached_gcs_result(
    task_id: str,
    offset: int,
    page_size: int,
) -> list[TextContent] | None:
    """Check Redis for cached GCS metadata and return a response if found.

    Returns None if GCS is not configured or no cached metadata exists.
    """
    if state.gcs_store is None or state.auth_provider is None:
        return None

    cached_meta_raw = await state.auth_provider._redis.get(build_key("result", task_id))
    if not cached_meta_raw:
        return None

    meta = json.loads(cached_meta_raw)
    urls = await asyncio.to_thread(state.gcs_store.generate_signed_urls, task_id)

    return _build_gcs_response(
        task_id=task_id,
        urls=urls,
        preview=meta.get("preview", []),
        total=meta["total"],
        columns=meta["columns"],
        offset=offset,
        page_size=page_size,
    )


async def try_upload_gcs_result(
    task_id: str,
    df: pd.DataFrame,
    offset: int,
    page_size: int,
) -> list[TextContent] | None:
    """Upload a DataFrame to GCS, cache metadata in Redis, and return a response.

    Returns None if GCS is not configured or the upload fails (caller should
    fall back to in-memory cache).
    """
    if state.gcs_store is None or state.auth_provider is None:
        return None

    try:
        urls = await asyncio.to_thread(state.gcs_store.upload_result, task_id, df)
        total = len(df)
        columns = list(df.columns)

        # Build preview page
        clamped_offset = min(offset, total)
        page_df = df.iloc[clamped_offset : clamped_offset + page_size]
        preview = page_df.where(page_df.notna(), None).to_dict(orient="records")

        # Store metadata in Redis with TTL
        meta = {"total": total, "columns": columns, "preview": preview}
        await state.auth_provider._redis.setex(
            build_key("result", task_id),
            RESULT_CACHE_TTL,
            json.dumps(meta),
        )

        return _build_gcs_response(
            task_id=task_id,
            urls=urls,
            preview=preview,
            total=total,
            columns=columns,
            offset=clamped_offset,
            page_size=page_size,
        )
    except Exception:
        logger.exception(
            "GCS upload failed for task %s, falling back to in-memory cache",
            task_id,
        )
        return None
