"""OAuth auth for EveryRow MCP server.

Resource-server mode only: SupabaseTokenVerifier validates Supabase-issued
JWTs via JWKS. The authorization server role (client registration, PKCE flow,
token exchange, refresh) is handled entirely by Supabase's OAuth 2.1 Server.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import jwt as pyjwt
from jwt import PyJWKClient
from mcp.server.auth.provider import AccessToken, TokenVerifier

logger = logging.getLogger(__name__)


class SupabaseTokenVerifier(TokenVerifier):
    """Verify Supabase-issued JWTs using the project's JWKS endpoint."""

    def __init__(self, supabase_url: str) -> None:
        self._issuer = supabase_url.rstrip("/") + "/auth/v1"
        self._jwks_client = PyJWKClient(
            f"{self._issuer}/.well-known/jwks.json", cache_keys=True
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            signing_key = await asyncio.to_thread(
                self._jwks_client.get_signing_key_from_jwt, token
            )
            payload: dict[str, Any] = pyjwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self._issuer,
                audience="authenticated",
            )
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
