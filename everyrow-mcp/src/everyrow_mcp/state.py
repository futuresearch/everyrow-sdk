"""Centralized server state for the everyrow MCP server.

Replaces scattered module-level globals with a single dataclass.
Provides Redis-backed token storage methods for multi-pod deployments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from everyrow.generated.client import AuthenticatedClient

from everyrow_mcp.config import HttpSettings, StdioSettings
from everyrow_mcp.gcs_storage import GCSResultStore
from everyrow_mcp.redis_utils import build_key

logger = logging.getLogger(__name__)

PROGRESS_POLL_DELAY = 12
TASK_STATE_FILE = Path.home() / ".everyrow" / "task.json"
RESULT_CACHE_TTL = 600
TOKEN_TTL = 86400  # 24 hours — must outlive the longest possible task


@dataclass
class ServerState:
    """Mutable state shared across the MCP server.

    This is a dataclass (not a BaseModel) because it holds runtime-only mutable
    state — asyncio.Lock, Redis client, AuthenticatedClient — that doesn't fit
    Pydantic's validation/serialization model.
    """

    client: AuthenticatedClient | None = None
    transport: str = "stdio"
    mcp_server_url: str = ""
    gcs_store: GCSResultStore | None = None
    settings: StdioSettings | HttpSettings | None = None
    redis: Any | None = None
    auth_provider: Any | None = None

    # ── Transport helpers ─────────────────────────────────────────

    @property
    def is_stdio(self) -> bool:
        return self.transport == "stdio"

    @property
    def is_http(self) -> bool:
        return self.transport != "stdio"

    # ── Redis access (sealed — no direct .redis outside this class) ───

    async def redis_ping(self) -> None:
        """Ping Redis to verify connectivity. Raises if Redis is unavailable."""
        if self.redis is not None:
            await self.redis.ping()

    async def get_result_meta(self, task_id: str) -> str | None:
        """Get cached GCS result metadata from Redis."""
        redis = self.redis
        if redis is None:
            return None
        try:
            return await redis.get(build_key("result", task_id))
        except Exception:
            logger.warning("Failed to get result metadata from Redis for %s", task_id)
            return None

    async def store_result_meta(self, task_id: str, meta_json: str) -> None:
        """Store GCS result metadata in Redis with TTL."""
        redis = self.redis
        if redis is None:
            return
        try:
            await redis.setex(
                build_key("result", task_id),
                RESULT_CACHE_TTL,
                meta_json,
            )
        except Exception:
            logger.warning("Failed to store result metadata in Redis for %s", task_id)

    async def get_result_page(
        self, task_id: str, offset: int, page_size: int
    ) -> str | None:
        """Get a cached page preview from Redis."""
        redis = self.redis
        if redis is None:
            return None
        try:
            return await redis.get(
                build_key("result", task_id, "page", str(offset), str(page_size))
            )
        except Exception:
            return None

    async def store_result_page(
        self, task_id: str, offset: int, page_size: int, preview_json: str
    ) -> None:
        """Cache a page preview in Redis with TTL."""
        redis = self.redis
        if redis is None:
            return
        try:
            await redis.setex(
                build_key("result", task_id, "page", str(offset), str(page_size)),
                RESULT_CACHE_TTL,
                preview_json,
            )
        except Exception:
            pass

    # ── Redis-backed token storage (multi-pod safe) ──────────────

    async def store_task_token(self, task_id: str, token: str) -> None:
        """Store an API token for a task in Redis."""
        redis = self.redis
        if redis is None:
            return
        try:
            await redis.setex(build_key("task_token", task_id), TOKEN_TTL, token)
        except Exception:
            logger.warning("Failed to store task token in Redis for %s", task_id)

    async def get_task_token(self, task_id: str) -> str | None:
        """Get an API token for a task from Redis."""
        redis = self.redis
        if redis is None:
            return None
        try:
            return await redis.get(build_key("task_token", task_id))
        except Exception:
            logger.warning("Failed to get task token from Redis for %s", task_id)
            return None

    async def store_poll_token(self, task_id: str, poll_token: str) -> None:
        """Store a poll token for a task in Redis."""
        redis = self.redis
        if redis is None:
            return
        try:
            await redis.setex(build_key("poll_token", task_id), TOKEN_TTL, poll_token)
        except Exception:
            logger.warning("Failed to store poll token in Redis for %s", task_id)

    async def get_poll_token(self, task_id: str) -> str | None:
        """Get a poll token for a task from Redis."""
        redis = self.redis
        if redis is None:
            return None
        try:
            return await redis.get(build_key("poll_token", task_id))
        except Exception:
            logger.warning("Failed to get poll token from Redis for %s", task_id)
            return None

    async def pop_task_token(self, task_id: str) -> None:
        """Remove tokens for a task from Redis."""
        redis = self.redis
        if redis is None:
            return
        try:
            await redis.delete(
                build_key("task_token", task_id),
                build_key("poll_token", task_id),
            )
        except Exception:
            logger.warning("Failed to delete tokens from Redis for %s", task_id)


state = ServerState()
