"""REST endpoints for the futuresearch MCP server (progress polling)."""

from __future__ import annotations

import csv
import json
import logging
import secrets
from uuid import UUID

import pandas as pd
from futuresearch.errors import FuturesearchError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from futuresearch_mcp import redis_store
from futuresearch_mcp.engine_client import NO_ACCOUNT_SELECTED, build_engine_client
from futuresearch_mcp.progress import build_progress_payload
from futuresearch_mcp.result_store import _sanitize_records
from futuresearch_mcp.tool_helpers import _fetch_task_result

logger = logging.getLogger(__name__)

# Machine-readable marker telling the widget its polling can never succeed
# again, so it stops instead of retrying a credential that is gone.
SESSION_EXPIRED = "session_expired"


def _session_expired(cors: dict[str, str]) -> JSONResponse:
    """401 for a task whose owner has no live credential."""
    return JSONResponse(
        {"error": "Session expired", "code": SESSION_EXPIRED},
        status_code=401,
        headers=cors,
    )


def _cors_headers() -> dict[str, str]:
    """CORS headers for widget endpoints.

    MCP App widgets run in sandboxed iframes whose origin will never match
    the server's own URL.  Because auth is via Bearer tokens (not cookies),
    a wildcard origin is safe — no ambient credentials are leaked.
    """
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET",
        "Access-Control-Allow-Headers": "Authorization",
    }


def _validate_uuid(task_id: str) -> JSONResponse | None:
    """Return a 400 response if task_id is not a valid UUID, else None."""
    try:
        UUID(task_id)
    except ValueError:
        return JSONResponse(
            {"error": "Invalid task ID"},
            status_code=400,
            headers=_cors_headers(),
        )
    return None


def _extract_bearer_or_query_token(request: Request, task_id: str) -> str:
    """Extract a poll token from Authorization header or ?token= query param."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:]
    provided = request.query_params.get("token", "")
    if provided:
        logger.info(
            "Poll token provided via query param for task %s — prefer Authorization header",
            task_id,
        )
    return provided


async def _validate_poll_token(task_id: str, request: Request) -> JSONResponse | None:
    """Return an error response if the poll token is missing/invalid, else None.

    Checks Authorization: Bearer header first, falls back to ?token= query
    param (for clickable CSV download links).  Non-destructive — the token
    remains in Redis for repeated progress polling.
    """
    expected = await redis_store.get_poll_token(task_id)
    provided = _extract_bearer_or_query_token(request, task_id)
    if not expected or not provided or not secrets.compare_digest(provided, expected):
        logger.warning("Invalid poll token for task %s", task_id)
        return JSONResponse(
            {"error": "Unauthorized"}, status_code=403, headers=_cors_headers()
        )
    return None


def _progress_failure(
    exc: FuturesearchError, task_id: str, cors: dict[str, str]
) -> JSONResponse:
    """Map an SDK error to a response, logging enough to diagnose it.

    A 4xx from upstream means the credential we hold is no longer accepted,
    or the task is gone; reporting either as a 500 tells the widget to retry
    something that can never succeed.
    """
    # A permanent client-side condition must not log at error level: the
    # widget polls every ~10s, so one page per poll until the tab is closed.
    log = logger.warning if exc.status_code == 404 else logger.error
    log(
        "Progress poll failed for task %s: %s status=%s code=%s: %s",
        task_id,
        type(exc).__name__,
        exc.status_code,
        exc.error_code,
        exc.message,
    )
    if exc.status_code in (401, 403):
        return _session_expired(cors)
    if exc.status_code == 404:
        # Terminal, same as the unknown-task branch in api_progress.
        return JSONResponse({"error": "Unknown task"}, status_code=404, headers=cors)
    return JSONResponse(
        {"error": "Internal server error"}, status_code=500, headers=cors
    )


async def api_progress(request: Request) -> Response:  # noqa: PLR0911
    """REST endpoint for the session widget to poll task progress."""
    cors = _cors_headers()
    if request.method == "OPTIONS":
        return Response(
            status_code=204,
            headers={**cors, "Access-Control-Max-Age": "3600"},
        )

    task_id = request.path_params["task_id"]

    if err := _validate_uuid(task_id):
        return err

    if err := await _validate_poll_token(task_id, request):
        return err

    api_key = await redis_store.get_task_credential(task_id)

    if not api_key:
        # We know whose task it is but hold no live credential: their session
        # lapsed. Distinct from a task we have no record of at all.
        if await redis_store.get_task_owner(task_id):
            return _session_expired(cors)
        return JSONResponse({"error": "Unknown task"}, status_code=404, headers=cors)

    try:
        client = build_engine_client(
            token=api_key,
            account_id=await redis_store.get_task_account(task_id)
            or NO_ACCOUNT_SELECTED,
        )
        payload = await build_progress_payload(
            client, task_id, request.query_params.get("cursor")
        )
        return JSONResponse(payload, headers=cors)
    except FuturesearchError as exc:
        return _progress_failure(exc, task_id, cors)
    except Exception:
        logger.exception("Progress poll failed for task %s", task_id)
        return JSONResponse(
            {"error": "Internal server error"}, status_code=500, headers=cors
        )


async def _validate_poll_token_bearer_only(
    task_id: str, request: Request
) -> JSONResponse | None:
    """Validate poll token from Authorization header only (no query params).

    Used for API endpoints where query-param auth is inappropriate
    (e.g. token minting — the poll token must not leak into URLs).
    """
    expected = await redis_store.get_poll_token(task_id)
    auth_header = request.headers.get("authorization", "")
    provided = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
    if not expected or not provided or not secrets.compare_digest(provided, expected):
        logger.warning("Invalid poll token (bearer-only) for task %s", task_id)
        return JSONResponse(
            {"error": "Unauthorized"}, status_code=403, headers=_cors_headers()
        )
    return None


async def api_download(request: Request) -> Response:  # noqa: PLR0911
    """REST endpoint to download task results as CSV or JSON.

    Authenticates via the poll token (Authorization: Bearer header or
    ?token= query param). No separate download token needed.
    """
    cors = _cors_headers()
    if request.method == "OPTIONS":
        return Response(
            status_code=204,
            headers={**cors, "Access-Control-Max-Age": "3600"},
        )

    task_id = request.path_params["task_id"]

    if err := _validate_uuid(task_id):
        return err

    if err := await _validate_poll_token(task_id, request):
        return err

    fmt = request.query_params.get("format", "csv")
    if fmt not in ("csv", "json"):
        return JSONResponse(
            {"error": "Unsupported format"}, status_code=400, headers=cors
        )

    # Fetch results via the public API (parquet-first path handles citation
    # resolution and internal column stripping automatically).
    api_key = await redis_store.get_task_credential(task_id)
    if not api_key:
        if await redis_store.get_task_owner(task_id):
            return _session_expired(cors)
        return JSONResponse(
            {"error": "Results not found or expired"}, status_code=404, headers=cors
        )
    try:
        client = build_engine_client(
            token=api_key,
            account_id=await redis_store.get_task_account(task_id)
            or NO_ACCOUNT_SELECTED,
        )
        rows, _total, _session_id, _artifact_id = await _fetch_task_result(
            client, task_id
        )
        records: list[dict] = _sanitize_records(rows)
    except Exception:
        logger.warning("Failed to fetch results for task %s", task_id, exc_info=True)
        return JSONResponse(
            {"error": "Results not found or expired"}, status_code=404, headers=cors
        )
    safe_prefix = "".join(c for c in task_id[:8] if c.isalnum() or c == "-")

    if fmt == "json":
        return Response(
            content=json.dumps(records),
            media_type="application/json",
            headers={
                **cors,
                "X-Content-Type-Options": "nosniff",
                "Content-Disposition": f'attachment; filename="results_{safe_prefix}.json"',
            },
        )

    # CSV generated on-the-fly from the already-resolved records.
    csv_text = pd.DataFrame(records).to_csv(index=False, quoting=csv.QUOTE_ALL)
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={
            **cors,
            "Content-Disposition": f'attachment; filename="results_{safe_prefix}.csv"',
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )
