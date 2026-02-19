"""Centralized server state for the everyrow MCP server.

Replaces scattered module-level globals with a single dataclass.
Provides Redis-backed token storage methods for multi-pod deployments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from everyrow.generated.client import AuthenticatedClient

from everyrow_mcp.auth import EveryRowAuthProvider
from everyrow_mcp.gcs_storage import GCSResultStore
from everyrow_mcp.redis_utils import build_key
from everyrow_mcp.settings import HttpSettings, StdioSettings

logger = logging.getLogger(__name__)

PROGRESS_POLL_DELAY = 12
TASK_STATE_FILE = Path.home() / ".everyrow" / "task.json"
RESULT_CACHE_TTL = 600


@dataclass
class ServerState:
    """Mutable state shared across the MCP server."""

    client: AuthenticatedClient | None = None
    transport: str = "stdio"
    auth_provider: EveryRowAuthProvider | None = None
    mcp_server_url: str = ""
    task_tokens: dict[str, str] = field(default_factory=dict)
    task_poll_tokens: dict[str, str] = field(default_factory=dict)
    result_cache: dict[str, tuple[pd.DataFrame, float, str]] = field(
        default_factory=dict
    )
    gcs_store: GCSResultStore | None = None
    settings: HttpSettings | StdioSettings | None = None

    # ── Redis-backed token storage (multi-pod safe) ──────────────

    @property
    def _redis(self) -> Any | None:
        """Return the Redis client if available (HTTP mode)."""
        if self.auth_provider is not None:
            return self.auth_provider._redis
        return None

    async def store_task_token(self, task_id: str, token: str) -> None:
        """Store an API token for a task (local dict + Redis if available)."""
        self.task_tokens[task_id] = token
        redis = self._redis
        if redis is not None:
            try:
                await redis.setex(
                    build_key("task_token", task_id),
                    RESULT_CACHE_TTL,
                    token,
                )
            except Exception:
                logger.debug("Failed to store task token in Redis for %s", task_id)

    async def get_task_token(self, task_id: str) -> str | None:
        """Get an API token for a task (local dict, fall back to Redis)."""
        token = self.task_tokens.get(task_id)
        if token is not None:
            return token
        redis = self._redis
        if redis is not None:
            try:
                token = await redis.get(build_key("task_token", task_id))
                if token is not None:
                    self.task_tokens[task_id] = token
                return token
            except Exception:
                logger.debug("Failed to get task token from Redis for %s", task_id)
        return None

    async def store_poll_token(self, task_id: str, poll_token: str) -> None:
        """Store a poll token for a task (local dict + Redis if available)."""
        self.task_poll_tokens[task_id] = poll_token
        redis = self._redis
        if redis is not None:
            try:
                await redis.setex(
                    build_key("poll_token", task_id),
                    RESULT_CACHE_TTL,
                    poll_token,
                )
            except Exception:
                logger.debug("Failed to store poll token in Redis for %s", task_id)

    async def get_poll_token(self, task_id: str) -> str | None:
        """Get a poll token for a task (local dict, fall back to Redis)."""
        token = self.task_poll_tokens.get(task_id)
        if token is not None:
            return token
        redis = self._redis
        if redis is not None:
            try:
                token = await redis.get(build_key("poll_token", task_id))
                if token is not None:
                    self.task_poll_tokens[task_id] = token
                return token
            except Exception:
                logger.debug("Failed to get poll token from Redis for %s", task_id)
        return None

    async def pop_task_token(self, task_id: str) -> None:
        """Remove tokens for a task from both local dict and Redis."""
        self.task_tokens.pop(task_id, None)
        self.task_poll_tokens.pop(task_id, None)
        redis = self._redis
        if redis is not None:
            try:
                await redis.delete(
                    build_key("task_token", task_id),
                    build_key("poll_token", task_id),
                )
            except Exception:
                logger.debug("Failed to delete tokens from Redis for %s", task_id)


state = ServerState()
