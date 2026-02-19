"""GCS-backed result retrieval for the everyrow MCP server.

Handles checking Redis for cached GCS metadata, uploading new results to GCS,
and building the MCP TextContent responses for both paths.

Caching strategy:
  - Base metadata (total, columns) cached at  result:{task_id}
  - Per-page previews cached at               result:{task_id}:page:{offset}:{page_size}
  - On a page cache miss, the JSON is fetched from GCS and the page is sliced.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import pandas as pd
from mcp.types import TextContent

from everyrow_mcp.state import state

if TYPE_CHECKING:
    from everyrow_mcp.gcs_storage import ResultURLs

logger = logging.getLogger(__name__)


def _format_columns(columns: list[str]) -> str:
    """Format column names for display, truncating after 10."""
    col_names = ", ".join(columns[:10])
    if len(columns) > 10:
        col_names += f", ... (+{len(columns) - 10} more)"
    return col_names


def _slice_preview(records: list[dict], offset: int, page_size: int) -> list[dict]:
    """Slice a page from a list of record dicts."""
    clamped = min(offset, len(records))
    return records[clamped : clamped + page_size]


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
        page_size_arg = (
            f", page_size={page_size}"
            if page_size != (state.settings.preview_size if state.settings else 5)
            else ""
        )
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
    """Return a GCS-backed result page, using per-page Redis cache.

    Returns None if GCS is not configured or no cached metadata exists
    (i.e. the task was never uploaded to GCS).
    """
    if state.gcs_store is None:
        return None

    # Check base metadata — if absent, this task isn't in GCS
    cached_meta_raw = await state.get_result_meta(task_id)
    if not cached_meta_raw:
        return None

    meta = json.loads(cached_meta_raw)
    total: int = meta["total"]
    columns: list[str] = meta["columns"]

    # Check per-page cache
    cached_page = await state.get_result_page(task_id, offset, page_size)
    if cached_page is not None:
        preview = json.loads(cached_page)
    else:
        # Page cache miss — fetch from GCS and slice
        try:
            all_records = await asyncio.to_thread(
                state.gcs_store.download_json, task_id
            )
            preview = _slice_preview(all_records, offset, page_size)
            await state.store_result_page(
                task_id, offset, page_size, json.dumps(preview)
            )
        except Exception:
            logger.warning("Failed to fetch page from GCS for task %s", task_id)
            preview = []

    urls = await asyncio.to_thread(state.gcs_store.generate_signed_urls, task_id)

    return _build_gcs_response(
        task_id=task_id,
        urls=urls,
        preview=preview,
        total=total,
        columns=columns,
        offset=min(offset, total),
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
    if state.gcs_store is None:
        return None

    try:
        urls = await asyncio.to_thread(state.gcs_store.upload_result, task_id, df)
        total = len(df)
        columns = list(df.columns)

        # Store base metadata (without preview)
        meta = {"total": total, "columns": columns}
        await state.store_result_meta(task_id, json.dumps(meta))

        # Build and cache page preview
        clamped_offset = min(offset, total)
        page_df = df.iloc[clamped_offset : clamped_offset + page_size]
        preview = page_df.where(page_df.notna(), None).to_dict(orient="records")
        await state.store_result_page(task_id, offset, page_size, json.dumps(preview))

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
