"""Redis-backed result retrieval for the everyrow MCP server.

Handles checking Redis for cached metadata, storing CSV results,
and building the MCP TextContent responses.

Caching strategy:
  - Base metadata (total, columns) cached at  result:{task_id}
  - Per-page previews cached at               result:{task_id}:page:{offset}:{page_size}
  - Full CSV stored at                        result:{task_id}:csv  (1h TTL)
  - On a page cache miss, the CSV is read from Redis and the page is sliced.
"""

from __future__ import annotations

import io
import json
import logging

import pandas as pd
from mcp.types import TextContent

from everyrow_mcp.state import state

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


def _build_csv_url(task_id: str) -> str:
    """Build the internal download URL for a task's CSV."""
    poll_token = ""  # Will be filled async; see callers
    return f"{state.mcp_server_url}/api/results/{task_id}/download?token={poll_token}"


def _build_result_response(
    task_id: str,
    csv_url: str,
    preview: list[dict],
    total: int,
    columns: list[str],
    offset: int,
    page_size: int,
    session_url: str = "",
) -> list[TextContent]:
    """Build MCP TextContent response for Redis-backed results."""
    col_names = _format_columns(columns)

    widget_data: dict = {
        "csv_url": csv_url,
        "preview": preview,
        "total": total,
    }
    if session_url:
        widget_data["session_url"] = session_url
    widget_json = json.dumps(widget_data)

    has_more = offset + page_size < total
    next_offset = offset + page_size if has_more else None

    if has_more:
        page_size_arg = (
            f", page_size={page_size}" if page_size != state.preview_size else ""
        )
        summary = (
            f"Results: {total} rows, {len(columns)} columns ({col_names}). "
            f"Showing rows {offset + 1}-{min(offset + page_size, total)} of {total}.\n"
            f"Call everyrow_results(task_id='{task_id}', offset={next_offset}{page_size_arg}) for the next page."
        )
        if offset == 0:
            summary += (
                f"\nFull CSV download: {csv_url}\n"
                "IMPORTANT: Display this download link to the user as a clickable URL in your response."
            )
    elif offset == 0:
        summary = (
            f"Results: {total} rows, {len(columns)} columns ({col_names}). "
            f"All rows shown.\n"
            f"Full CSV download: {csv_url}\n"
            "IMPORTANT: Display this download link to the user as a clickable URL in your response."
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


async def _get_csv_url(task_id: str) -> str:
    """Build the CSV download URL with the current poll token."""
    poll_token = await state.store.get_poll_token(task_id) or ""
    return f"{state.mcp_server_url}/api/results/{task_id}/download?token={poll_token}"


async def try_cached_result(
    task_id: str,
    offset: int,
    page_size: int,
) -> list[TextContent] | None:
    """Return a Redis-backed result page, using per-page cache.

    Returns None if Redis is not available or no cached metadata exists.
    """
    if state.store is None:
        return None

    # Check base metadata — if absent, this task isn't cached
    cached_meta_raw = await state.store.get_result_meta(task_id)
    if not cached_meta_raw:
        return None

    meta = json.loads(cached_meta_raw)
    total: int = meta["total"]
    columns: list[str] = meta["columns"]
    session_url: str = meta.get("session_url", "")

    # Check per-page cache
    cached_page = await state.store.get_result_page(task_id, offset, page_size)
    if cached_page is not None:
        preview = json.loads(cached_page)
    else:
        # Page cache miss — read full CSV from Redis and slice
        try:
            csv_text = await state.store.get_result_csv(task_id)
            if csv_text is None:
                preview = []
            else:
                df = pd.read_csv(io.StringIO(csv_text))
                all_records = df.where(df.notna(), None).to_dict(orient="records")
                preview = _slice_preview(all_records, offset, page_size)
                await state.store.store_result_page(
                    task_id, offset, page_size, json.dumps(preview)
                )
        except Exception:
            logger.warning("Failed to read CSV from Redis for task %s", task_id)
            preview = []

    csv_url = await _get_csv_url(task_id)

    return _build_result_response(
        task_id=task_id,
        csv_url=csv_url,
        preview=preview,
        total=total,
        columns=columns,
        offset=min(offset, total),
        page_size=page_size,
        session_url=session_url,
    )


async def try_store_result(
    task_id: str,
    df: pd.DataFrame,
    offset: int,
    page_size: int,
    session_url: str = "",
) -> list[TextContent] | None:
    """Store a DataFrame in Redis and return a response.

    Returns None if Redis is not available (caller should fall back to
    inline results).
    """
    if state.store is None:
        return None

    try:
        # Store full CSV in Redis
        await state.store.store_result_csv(task_id, df.to_csv(index=False))

        total = len(df)
        columns = list(df.columns)

        # Store base metadata
        meta: dict = {"total": total, "columns": columns}
        if session_url:
            meta["session_url"] = session_url
        await state.store.store_result_meta(task_id, json.dumps(meta))

        # Build and cache page preview
        clamped_offset = min(offset, total)
        page_df = df.iloc[clamped_offset : clamped_offset + page_size]
        preview = page_df.where(page_df.notna(), None).to_dict(orient="records")
        await state.store.store_result_page(
            task_id, offset, page_size, json.dumps(preview)
        )

        csv_url = await _get_csv_url(task_id)

        return _build_result_response(
            task_id=task_id,
            csv_url=csv_url,
            preview=preview,
            total=total,
            columns=columns,
            offset=clamped_offset,
            page_size=page_size,
            session_url=session_url,
        )
    except Exception:
        logger.exception(
            "Failed to store results in Redis for task %s, falling back to inline",
            task_id,
        )
        return None
