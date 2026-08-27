from __future__ import annotations

import logging
from uuid import UUID

from futuresearch.errors import _call_and_check
from futuresearch.generated.api.tasks import get_task_status_tasks_task_id_status_get
from futuresearch.generated.client import AuthenticatedClient

from futuresearch_mcp.tool_helpers import _UI_EXCLUDE, TaskState, dedupe_summaries

logger = logging.getLogger(__name__)


async def _fetch_summaries(
    client: AuthenticatedClient, task_id: str, cursor: str | None
) -> tuple[list[dict] | None, str | None]:
    try:
        params: dict[str, str] = {}
        if cursor:
            params["cursor"] = cursor
        httpx_client = client.get_async_httpx_client()
        resp = await httpx_client.request(
            method="get",
            url=f"/tasks/{task_id}/summaries",
            params=params,
        )
        if resp.status_code == 200:
            data = resp.json()
            raw = data.get("summaries") or None
            if raw:
                raw = dedupe_summaries(raw)
            return raw, data.get("cursor") or cursor
    except Exception as err:
        logger.warning(f"Failed to fetch summaries for task {task_id}: {err!r}")
    return None, cursor


async def _fetch_timeline(
    client: AuthenticatedClient, task_id: str
) -> list[dict] | None:
    """Fetch the stored aggregate timeline.

    Returns a list of timeline entries (each with aggregate + micro_summaries),
    or None if the endpoint is unavailable or returns no data.
    """
    try:
        httpx_client = client.get_async_httpx_client()
        resp = await httpx_client.request(
            method="get",
            url=f"/tasks/{task_id}/summaries/timeline",
        )
        if resp.status_code == 200:
            data = resp.json()
            timeline = data.get("timeline")
            if timeline:
                # Dedupe micro-summaries within each entry
                for entry in timeline:
                    micros = entry.get("micro_summaries")
                    if micros:
                        entry["micro_summaries"] = dedupe_summaries(micros)
                return timeline
    except Exception as err:
        logger.warning(f"Failed to fetch timeline for task {task_id}: {err!r}")
    return None


async def _fetch_aggregate(
    client: AuthenticatedClient, task_id: str, cursor: str | None
) -> tuple[str | None, list[dict] | None, str | None]:
    """Fetch aggregate + micro-summaries.

    Returns (aggregate_text, micro_summaries, updated_cursor).
    Falls back to plain summaries when the aggregate endpoint is unavailable.
    """
    try:
        params: dict[str, str] = {}
        if cursor:
            params["cursor"] = cursor
        httpx_client = client.get_async_httpx_client()
        resp = await httpx_client.request(
            method="get",
            url=f"/tasks/{task_id}/summaries/aggregate",
            params=params,
        )
        if resp.status_code == 200:
            data = resp.json()
            micros = data.get("micro_summaries") or None
            if micros:
                micros = dedupe_summaries(micros)
            return (
                data.get("aggregate") or None,
                micros,
                data.get("cursor") or cursor,
            )
    except Exception as err:
        logger.warning(f"Failed to fetch aggregate for task {task_id}: {err!r}")

    # Fallback: plain summaries without aggregate
    summaries, new_cursor = await _fetch_summaries(client, task_id, cursor)
    return None, summaries, new_cursor


async def _backfill_timeline(
    client: AuthenticatedClient, task_id: str, payload: dict
) -> None:
    """Populate payload with stored timeline + tail aggregate for re-mount.

    Fetches the stored aggregate timeline (no LLM call), then fills any gap
    with one aggregate call for micro-summaries that arrived after the last
    stored aggregate.
    """
    timeline = await _fetch_timeline(client, task_id)
    if timeline:
        # Find the latest micro-summary timestamp to fill the gap
        last_cursor = max(
            (
                ms.get("updated_at", "")
                for entry in timeline
                for ms in entry.get("micro_summaries", [])
            ),
            default=None,
        )
        if last_cursor:
            aggregate, summaries, new_cursor = await _fetch_aggregate(
                client, task_id, last_cursor
            )
            if aggregate:
                payload["aggregate_summary"] = aggregate
                payload["summaries"] = summaries
                payload["cursor"] = new_cursor
    else:
        # No stored timeline — fall back to generating one aggregate
        aggregate, summaries, new_cursor = await _fetch_aggregate(client, task_id, None)
        if aggregate:
            payload["aggregate_summary"] = aggregate
        if summaries:
            payload["summaries"] = summaries
        if new_cursor:
            payload["cursor"] = new_cursor


async def _apply_live_aggregate(
    client: AuthenticatedClient, task_id: str, payload: dict, cursor: str | None
) -> None:
    """Merge the current aggregate + micro-summaries into a progress payload."""
    aggregate, summaries, new_cursor = await _fetch_aggregate(client, task_id, cursor)
    if aggregate:
        payload["aggregate_summary"] = aggregate
    if summaries:
        payload["summaries"] = summaries
    if new_cursor:
        payload["cursor"] = new_cursor


async def build_progress_payload(
    client: AuthenticatedClient, task_id: str, cursor: str | None
) -> dict:
    """Build the widget's view of a task's progress.

    Without a *cursor* the caller is asking about a task it isn't yet
    following, so what has happened so far comes along. With one it is
    following already, and only what has happened since needs fetching.
    """
    status_response = await _call_and_check(
        get_task_status_tasks_task_id_status_get.asyncio_detailed(
            task_id=UUID(task_id),
            client=client,
        )
    )
    ts = TaskState(status_response)
    payload = ts.model_dump(mode="json", exclude=_UI_EXCLUDE)

    if not ts.is_terminal:
        if not cursor:
            await _backfill_timeline(client, task_id, payload)
        await _apply_live_aggregate(
            client, task_id, payload, cursor or payload.get("cursor")
        )

    return payload
