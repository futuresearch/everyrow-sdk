"""HTTP middleware for the EveryRow MCP server."""

from __future__ import annotations

import logging
import time

from redis.asyncio import Redis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from everyrow_mcp.redis_store import build_key

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis-based fixed-window rate limiter per client IP.

    Returns 429 with ``Retry-After`` header when the limit is exceeded.
    Fails open if Redis is unavailable so a Redis outage does not block
    legitimate traffic.
    """

    def __init__(
        self,
        app,
        *,
        redis: Redis,
        max_requests: int = 100,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)
        self._redis = redis
        self._max_requests = max_requests
        self._window_seconds = window_seconds

    async def dispatch(self, request: Request, call_next) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        window_id = str(int(time.time()) // self._window_seconds)
        key = build_key("rate", client_ip, window_id)

        try:
            count = await self._redis.incr(key)
            if count == 1:
                await self._redis.expire(key, self._window_seconds)

            if count > self._max_requests:
                ttl = await self._redis.ttl(key)
                retry_after = max(ttl, 1)
                return JSONResponse(
                    {"detail": "Rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )
        except Exception:
            logger.warning("Rate-limit check failed (Redis unavailable)", exc_info=True)

        return await call_next(request)
