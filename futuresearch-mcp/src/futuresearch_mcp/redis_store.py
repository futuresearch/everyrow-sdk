from __future__ import annotations

import base64
import hashlib
import logging
import re
from enum import StrEnum
from functools import lru_cache

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from redis.asyncio import Redis, Sentinel
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff

from futuresearch_mcp.config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────

HEALTH_CHECK_INTERVAL = 30

PROGRESS_POLL_DELAY = 12
TOKEN_TTL = 86400  # 24 hours — must outlive the longest possible task


class Transport(StrEnum):
    STDIO = "stdio"
    HTTP = "streamable-http"


# ── Redis infrastructure ──────────────────────────────────────


_KEY_UNSAFE = re.compile(r"[^a-zA-Z0-9._\-]")


def build_key(*parts: str) -> str:
    """Build a namespaced Redis key, sanitising user-controlled characters."""
    sanitized = [_KEY_UNSAFE.sub("_", p) for p in parts]
    return "mcp:" + ":".join(sanitized)


# ── Token encryption at rest ─────────────────────────────────


@lru_cache(maxsize=1)
def _get_fernet() -> Fernet | None:
    """Get a Fernet cipher for encrypting sensitive values in Redis.

    Returns None when encryption is not configured (e.g. stdio mode
    where UPLOAD_SECRET is typically unset).
    """
    if not settings.upload_secret:
        return None
    key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"futuresearch-mcp-fernet",
    ).derive(settings.upload_secret.encode())
    return Fernet(base64.urlsafe_b64encode(key))


def encrypt_value(value: str) -> str:
    """Encrypt a string value for Redis storage. No-op without UPLOAD_SECRET."""
    f = _get_fernet()
    if f is None:
        if settings.is_http:
            raise RuntimeError(
                "UPLOAD_SECRET must be set in HTTP mode — cannot store sensitive values in plaintext."
            )
        return value
    return f.encrypt(value.encode()).decode()


def decrypt_value(value: str) -> str:
    """Decrypt a string value from Redis. No-op without UPLOAD_SECRET."""
    f = _get_fernet()
    if f is None:
        if settings.is_http:
            raise RuntimeError(
                "UPLOAD_SECRET must be set in HTTP mode — cannot read encrypted values without the key."
            )
        return value
    return f.decrypt(value.encode()).decode()


def create_redis_client(
    *,
    host: str = "localhost",
    port: int = 6379,
    db: int = settings.redis_db,
    password: str | None = None,
    ssl: bool = False,
    sentinel_endpoints: str | None = None,
    sentinel_master_name: str | None = None,
) -> Redis:
    """Create an async Redis client with retry and health-check support.

    If *sentinel_endpoints* is provided (comma-separated "host:port" pairs),
    connects via Sentinel; otherwise connects directly.
    """
    retry = Retry(ExponentialBackoff(), retries=3)

    if sentinel_endpoints and sentinel_master_name:
        sentinels = []
        for ep in sentinel_endpoints.split(","):
            h, p = ep.strip().rsplit(":", 1)
            sentinels.append((h, int(p)))

        sentinel = Sentinel(
            sentinels,
            sentinel_kwargs={"password": password, "ssl": ssl}
            if password
            else {"ssl": ssl},
            retry=retry,
        )
        client: Redis = sentinel.master_for(
            sentinel_master_name,
            db=db,
            password=password,
            ssl=ssl,
            decode_responses=True,
            health_check_interval=HEALTH_CHECK_INTERVAL,
            retry=retry,
        )
        logger.info(
            "Redis: Sentinel mode, master=%s, db=%d, ssl=%s",
            sentinel_master_name,
            db,
            ssl,
        )
        return client

    client = Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        ssl=ssl,
        decode_responses=True,
        health_check_interval=HEALTH_CHECK_INTERVAL,
        retry=retry,
    )
    logger.info("Redis: direct mode, host=%s:%d, db=%d, ssl=%s", host, port, db, ssl)
    return client


_redis_client: Redis | None = None


def get_redis_client() -> Redis:
    global _redis_client  # noqa: PLW0603
    if _redis_client is None:
        _redis_client = create_redis_client(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            ssl=settings.redis_ssl,
            sentinel_endpoints=settings.redis_sentinel_endpoints,
            sentinel_master_name=settings.redis_sentinel_master_name,
        )
    return _redis_client


def set_redis_client(client: Redis | None) -> None:
    """Override the Redis client (for testing)."""
    global _redis_client  # noqa: PLW0603
    _redis_client = client


async def store_task_token(task_id: str, token: str) -> None:
    await get_redis_client().setex(
        build_key("task_token", task_id), TOKEN_TTL, encrypt_value(token)
    )


async def get_task_token(task_id: str) -> str | None:
    encrypted = await get_redis_client().get(build_key("task_token", task_id))
    if encrypted is None:
        return None
    return decrypt_value(encrypted)


async def pop_task_token(task_id: str) -> None:
    await get_redis_client().delete(build_key("task_token", task_id))


# ── Task credentials ──────────────────────────────────────────
#
# A task outlives the credential that submitted it. API keys never expire, so
# they are stored verbatim per task (above). Supabase JWTs do expire, and their
# lifetime is counted from login — not from submission — so a JWT frozen at
# submission time can die long before the task's own TOKEN_TTL elapses. For
# those we store only the owner's user id per task and keep the JWT itself in a
# per-user slot that the OAuth layer overwrites on every refresh, so a poll
# always presents a live credential rather than a snapshot.


def user_token_key(user_id: str) -> str:
    """Redis key for a user's current Supabase JWT."""
    return build_key("user_token", user_id)


async def store_user_token(user_id: str, token: str, ttl: int) -> None:
    """Record a user's current JWT, expiring with the JWT itself."""
    if ttl <= 0:
        return
    await get_redis_client().setex(user_token_key(user_id), ttl, encrypt_value(token))


async def get_user_token(user_id: str) -> str | None:
    encrypted = await get_redis_client().get(user_token_key(user_id))
    if encrypted is None:
        return None
    return decrypt_value(encrypted)


async def store_task_owner(task_id: str, user_id: str) -> None:
    """Record which user submitted a task, so polls can resolve a live JWT."""
    await get_redis_client().setex(build_key("task_owner", task_id), TOKEN_TTL, user_id)


async def get_task_owner(task_id: str) -> str | None:
    return await get_redis_client().get(build_key("task_owner", task_id))


async def store_task_account(task_id: str, account_id: str) -> None:
    """Record which account submitted a task, so later polls present the same one."""
    if not account_id:
        return
    await get_redis_client().setex(
        build_key("task_account", task_id), TOKEN_TTL, account_id
    )


async def get_task_account(task_id: str) -> str | None:
    return await get_redis_client().get(build_key("task_account", task_id))


async def get_task_credential(task_id: str) -> str | None:
    """Resolve a usable API credential for a task, or None if unavailable.

    Prefers the owner's current JWT; falls back to a per-task credential for
    API-key submissions and for tasks recorded before owner mapping existed
    (the latter drain within TOKEN_TTL).
    """
    owner = await get_task_owner(task_id)
    if owner:
        if token := await get_user_token(owner):
            return token
        # Owner known but no live JWT: their session lapsed with nothing
        # refreshing it. Returning None yields an honest "expired" response
        # rather than a 401 from upstream.
        logger.info("No live credential for owner of task %s", task_id)
        return None
    return await get_task_token(task_id)


# ── Poll tokens ───────────────────────────────────────────────


async def store_poll_token(task_id: str, poll_token: str) -> None:
    """Store an encrypted poll token."""
    await get_redis_client().setex(
        name=build_key("poll_token", task_id),
        time=TOKEN_TTL,
        value=encrypt_value(poll_token),
    )


async def get_poll_token(task_id: str) -> str | None:
    encrypted = await get_redis_client().get(build_key("poll_token", task_id))
    if encrypted is None:
        return None
    return decrypt_value(encrypted)


# ── Task metadata (forecast type, output_field, units, …) ─────


async def store_task_meta(task_id: str, meta_json: str) -> None:
    """Store widget metadata for a task as a JSON string.

    Used so that ``futuresearch_status`` can render the right widget
    variant (e.g. forecast cards) without re-deriving from the original
    submission params. Not sensitive — stored plain.
    """
    await get_redis_client().setex(
        name=build_key("task_meta", task_id),
        time=TOKEN_TTL,
        value=meta_json,
    )


async def get_task_meta(task_id: str) -> str | None:
    return await get_redis_client().get(build_key("task_meta", task_id))


# ── MCP account selection (login-page choice) ─────────────────


def account_selection_key(access_token: str) -> str:
    """Redis key for the account selected during a connection's login flow."""
    fingerprint = hashlib.sha256(access_token.encode()).hexdigest()
    return build_key("acct", fingerprint)


async def get_account_selection(access_token: str) -> str | None:
    """Return the account_id selected at login for the presented access token."""
    return await get_redis_client().get(account_selection_key(access_token))
