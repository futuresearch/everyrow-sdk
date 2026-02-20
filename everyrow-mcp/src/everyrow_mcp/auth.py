"""OAuth auth for EveryRow MCP server.

Resource-server mode only: SupabaseTokenVerifier validates Supabase-issued
JWTs via JWKS. The authorization server role (client registration, PKCE flow,
token exchange, refresh) is handled entirely by Supabase's OAuth 2.1 Server.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

import jwt as pyjwt
from jwt import PyJWKClient
from mcp.server.auth.provider import AccessToken, TokenVerifier
from redis.asyncio import Redis

from everyrow_mcp.redis_utils import build_key

logger = logging.getLogger(__name__)


class SupabaseTokenVerifier(TokenVerifier):
    """Verify Supabase-issued JWTs using the project's JWKS endpoint."""

    def __init__(
        self,
        supabase_url: str,
        *,
        redis: Redis | None = None,
        revocation_ttl: int = 3600,
    ) -> None:
        self._issuer = supabase_url.rstrip("/") + "/auth/v1"
        self._jwks_client = PyJWKClient(
            f"{self._issuer}/.well-known/jwks.json",
            cache_keys=True,
            lifespan=300,
            max_cached_keys=16,
        )
        self._redis = redis
        self._revocation_ttl = revocation_ttl
        self._jwks_lock = asyncio.Lock()

    @staticmethod
    def _token_fingerprint(token: str) -> str:
        """SHA-256 fingerprint of the raw token (Supabase JWTs lack ``jti``)."""
        return hashlib.sha256(token.encode()).hexdigest()

    async def revoke_token(self, token: str) -> bool:
        """Add *token* to the Redis deny-list. Returns False if Redis is unavailable."""
        if self._redis is None:
            return False
        try:
            key = build_key("revoked", self._token_fingerprint(token))
            await self._redis.setex(key, self._revocation_ttl, "1")
            return True
        except Exception:
            logger.warning("Failed to revoke token (Redis unavailable)", exc_info=True)
            return False

    async def _is_revoked(self, token: str) -> bool:
        """Check whether *token* is in the deny-list. Fails open on Redis errors."""
        if self._redis is None:
            return False
        try:
            key = build_key("revoked", self._token_fingerprint(token))
            return await self._redis.exists(key) > 0
        except Exception:
            logger.warning("Revocation check failed (Redis unavailable)", exc_info=True)
            return False

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            async with self._jwks_lock:
                signing_key = await asyncio.to_thread(
                    self._jwks_client.get_signing_key_from_jwt, token
                )
            payload: dict[str, Any] = pyjwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self._issuer,
                audience="authenticated",
                options={"require": ["exp", "sub", "iss", "aud"]},
            )

            if await self._is_revoked(token):
                logger.debug("Token is revoked")
                return None

            sub = payload.get("sub")
            if not sub:
                logger.debug("JWT missing required 'sub' claim")
                return None
            return AccessToken(
                token=token,
                client_id=sub,
                scopes=payload.get("scope", "").split() if payload.get("scope") else [],
                expires_at=payload.get("exp"),
            )
        except pyjwt.PyJWTError:
            logger.debug("JWT verification failed", exc_info=True)
            return None
