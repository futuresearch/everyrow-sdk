"""OAuth auth for EveryRow MCP server.

Hybrid model:
- SupabaseTokenVerifier validates Supabase JWTs via JWKS (resource-server mode)
- EveryRowAuthProvider handles client registration + OAuth flow, delegating
  user authentication to Supabase and returning the Supabase JWT directly
  as the MCP access token.

No Fernet encryption — tokens stored in Redis as plain JSON.
Refresh tokens supported — opaque tokens in Redis mapped to Supabase refresh tokens.
Token lookup done via JWKS, not Redis.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import secrets
import time
from typing import Any, TypeVar
from urllib.parse import urlencode

import httpx
import jwt as pyjwt
from jwt import PyJWKClient
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    TokenVerifier,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import BaseModel
from redis.asyncio import Redis
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from everyrow_mcp.redis_utils import build_key

_M = TypeVar("_M", bound=BaseModel)

logger = logging.getLogger(__name__)

# TTLs
ACCESS_TOKEN_TTL = 3300  # 55 min (expire before Supabase JWT's 1h TTL)
AUTH_CODE_TTL = 300  # 5 minutes
PENDING_AUTH_TTL = 600  # 10 minutes
CLIENT_REGISTRATION_TTL = 2_592_000  # 30 days
REFRESH_TOKEN_TTL = 604_800  # 7 days


# ── Token verifier (resource-server mode) ─────────────────────────────


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
            header = pyjwt.get_unverified_header(token)
            algorithm = header.get("alg", "RS256")
            payload: dict[str, Any] = pyjwt.decode(
                token,
                signing_key.key,
                algorithms=[algorithm],
                issuer=self._issuer,
                audience="authenticated",
            )
            return AccessToken(
                token=token,
                client_id=payload.get("sub", "unknown"),
                scopes=payload.get("scope", "").split() if payload.get("scope") else [],
                expires_at=payload.get("exp"),
            )
        except pyjwt.PyJWTError:
            logger.debug("JWT verification failed", exc_info=True)
            return None


# ── Auth code with embedded Supabase JWT ──────────────────────────────


class EveryRowAuthorizationCode(AuthorizationCode):
    """Extends AuthorizationCode with the user's Supabase JWT."""

    supabase_jwt: str
    supabase_refresh_token: str = ""


class EveryRowRefreshToken(RefreshToken):
    """Extends RefreshToken with the Supabase refresh token."""

    supabase_refresh_token: str


class PendingAuth(BaseModel):
    """Saved between /authorize and /auth/callback."""

    client_id: str
    params: AuthorizationParams
    supabase_code_verifier: str = ""
    supabase_redirect_url: str = ""


# ── Simplified OAuth provider ─────────────────────────────────────────


class EveryRowAuthProvider(
    OAuthAuthorizationServerProvider[
        EveryRowAuthorizationCode, EveryRowRefreshToken, AccessToken
    ]
):
    """OAuth provider: handles registration + auth flow, delegates login to Supabase.

    Issues the Supabase JWT directly as the MCP access token.
    Token verification is handled by SupabaseTokenVerifier (JWKS), not by
    load_access_token.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_anon_key: str,
        mcp_server_url: str,
        redis: Redis,
    ) -> None:
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_anon_key = supabase_anon_key
        self.mcp_server_url = mcp_server_url.rstrip("/")
        self._redis = redis
        self._http = httpx.AsyncClient()

    # ── Redis helpers (plain JSON, no encryption) ─────────────────

    async def _redis_set(
        self, key: str, obj: BaseModel, ttl: int | None = None
    ) -> None:
        data = obj.model_dump_json()
        if ttl is not None:
            await self._redis.setex(key, ttl, data)
        else:
            await self._redis.set(key, data)

    async def _redis_get(self, key: str, model_class: type[_M]) -> _M | None:
        data = await self._redis.get(key)
        if data is None:
            return None
        return model_class.model_validate_json(data)  # type: ignore[return-value]

    # ── Client registration ───────────────────────────────────────

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return await self._redis_get(
            build_key("client", client_id), OAuthClientInformationFull
        )

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        cid = client_info.client_id
        if cid is None:
            raise ValueError("client_id is required")
        await self._redis_set(
            build_key("client", cid), client_info, ttl=CLIENT_REGISTRATION_TTL
        )

    # ── Authorization ─────────────────────────────────────────────

    async def authorize(
        self,
        client: OAuthClientInformationFull,
        params: AuthorizationParams,
    ) -> str:
        state = secrets.token_urlsafe(32)

        # Generate PKCE pair for the Supabase leg
        supabase_verifier = secrets.token_urlsafe(32)
        challenge_bytes = hashlib.sha256(supabase_verifier.encode()).digest()
        supabase_challenge = (
            base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode()
        )

        supabase_params = {
            "provider": "google",
            "redirect_to": f"{self.mcp_server_url}/auth/callback",
            "flow_type": "pkce",
            "code_challenge": supabase_challenge,
            "code_challenge_method": "s256",
        }
        supabase_redirect_url = (
            f"{self.supabase_url}/auth/v1/authorize?{urlencode(supabase_params)}"
        )

        pending = PendingAuth(
            client_id=client.client_id or "",
            params=params,
            supabase_code_verifier=supabase_verifier,
            supabase_redirect_url=supabase_redirect_url,
        )
        await self._redis_set(
            build_key("pending", state), pending, ttl=PENDING_AUTH_TTL
        )
        return f"{self.mcp_server_url}/auth/start/{state}"

    # ── Start (sets cookie, redirects to Supabase) ────────────────

    async def handle_start(self, request: Request) -> Response:
        state = request.path_params.get("state")
        if not state:
            return Response("Missing state", status_code=400)

        pending = await self._redis_get(build_key("pending", state), PendingAuth)
        if pending is None:
            return Response("Invalid state", status_code=400)

        response = RedirectResponse(url=pending.supabase_redirect_url, status_code=302)
        response.set_cookie(
            key="mcp_auth_state",
            value=state,
            max_age=PENDING_AUTH_TTL,
            httponly=True,
            samesite="lax",
            secure=self.mcp_server_url.startswith("https"),
        )
        return response

    # ── Callback (Supabase redirects here after login) ────────────

    async def handle_callback(self, request: Request) -> Response:
        code = request.query_params.get("code")
        state = request.cookies.get("mcp_auth_state")
        if not code or not state:
            return Response("Missing code or state cookie", status_code=400)

        pending = await self._redis_get(build_key("pending", state), PendingAuth)
        if pending is None:
            return Response("No pending authorization found", status_code=400)

        try:
            (
                _user_id,
                _email,
                supabase_jwt,
                supabase_refresh,
            ) = await self._exchange_supabase_code(
                code, code_verifier=pending.supabase_code_verifier
            )
        except Exception:
            logger.exception("Failed to exchange Supabase code")
            return Response("Failed to authenticate with Supabase", status_code=500)

        # Issue authorization code carrying the Supabase JWT
        auth_code_str = secrets.token_urlsafe(32)
        auth_code_obj = EveryRowAuthorizationCode(
            code=auth_code_str,
            client_id=pending.client_id,
            redirect_uri=pending.params.redirect_uri,
            redirect_uri_provided_explicitly=pending.params.redirect_uri_provided_explicitly,
            code_challenge=pending.params.code_challenge,
            scopes=pending.params.scopes or [],
            expires_at=time.time() + AUTH_CODE_TTL,
            resource=pending.params.resource,
            supabase_jwt=supabase_jwt,
            supabase_refresh_token=supabase_refresh,
        )
        await self._redis_set(
            build_key("authcode", auth_code_str), auth_code_obj, ttl=AUTH_CODE_TTL
        )
        await self._redis.delete(build_key("pending", state))

        redirect_params: dict[str, str] = {"code": auth_code_str}
        if pending.params.state:
            redirect_params["state"] = pending.params.state
        return RedirectResponse(
            url=f"{pending.params.redirect_uri}?{urlencode(redirect_params)}",
            status_code=302,
        )

    # ── Authorization code exchange ───────────────────────────────

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> EveryRowAuthorizationCode | None:
        code_obj = await self._redis_get(
            build_key("authcode", authorization_code), EveryRowAuthorizationCode
        )
        if code_obj is None or code_obj.client_id != client.client_id:
            return None
        return code_obj

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: EveryRowAuthorizationCode,
    ) -> OAuthToken:
        await self._redis.delete(build_key("authcode", authorization_code.code))

        # Derive expires_in from the JWT's actual exp claim
        jwt_claims = pyjwt.decode(
            authorization_code.supabase_jwt, options={"verify_signature": False}
        )
        expires_in = max(0, jwt_claims.get("exp", 0) - int(time.time()))

        # Store refresh token in Redis if available
        refresh_token_str: str | None = None
        if authorization_code.supabase_refresh_token:
            refresh_token_str = secrets.token_urlsafe(32)
            rt = EveryRowRefreshToken(
                token=refresh_token_str,
                client_id=client.client_id or "",
                scopes=authorization_code.scopes,
                supabase_refresh_token=authorization_code.supabase_refresh_token,
            )
            await self._redis_set(
                build_key("refresh", refresh_token_str), rt, ttl=REFRESH_TOKEN_TTL
            )

        return OAuthToken(
            access_token=authorization_code.supabase_jwt,
            token_type="Bearer",
            expires_in=expires_in,
            refresh_token=refresh_token_str,
        )

    # ── Token verification (handled by SupabaseTokenVerifier) ─────

    async def load_access_token(self, token: str) -> AccessToken | None:
        # Not used — SupabaseTokenVerifier handles verification via JWKS
        return None

    # ── Refresh / revocation ─────────────────────────────────────

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> EveryRowRefreshToken | None:
        rt = await self._redis_get(
            build_key("refresh", refresh_token), EveryRowRefreshToken
        )
        if rt is None or rt.client_id != client.client_id:
            return None
        return rt

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: EveryRowRefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        # Delete old token (rotation)
        await self._redis.delete(build_key("refresh", refresh_token.token))

        # Call Supabase to refresh
        new_jwt, new_supabase_refresh = await self._refresh_supabase_token(
            refresh_token.supabase_refresh_token
        )

        # Decode new JWT for exp
        jwt_claims = pyjwt.decode(new_jwt, options={"verify_signature": False})
        expires_in = max(0, jwt_claims.get("exp", 0) - int(time.time()))

        # Issue new refresh token
        new_rt_str = secrets.token_urlsafe(32)
        new_rt = EveryRowRefreshToken(
            token=new_rt_str,
            client_id=client.client_id or "",
            scopes=scopes or refresh_token.scopes,
            supabase_refresh_token=new_supabase_refresh,
        )
        await self._redis_set(
            build_key("refresh", new_rt_str), new_rt, ttl=REFRESH_TOKEN_TTL
        )

        return OAuthToken(
            access_token=new_jwt,
            token_type="Bearer",
            expires_in=expires_in,
            refresh_token=new_rt_str,
        )

    async def revoke_token(self, token: AccessToken | EveryRowRefreshToken) -> None:
        if isinstance(token, EveryRowRefreshToken):
            await self._redis.delete(build_key("refresh", token.token))

    # ── Supabase integration ──────────────────────────────────────

    async def _exchange_supabase_code(
        self, code: str, code_verifier: str = ""
    ) -> tuple[str, str, str, str]:
        """Exchange a Supabase OAuth code for user identity and tokens.

        Returns (user_id, email, access_token, refresh_token).
        """
        resp = await self._http.post(
            f"{self.supabase_url}/auth/v1/token?grant_type=pkce",
            json={"auth_code": code, "code_verifier": code_verifier},
            headers={
                "apikey": self.supabase_anon_key,
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        user = data["user"]
        return (
            user["id"],
            user.get("email", ""),
            data["access_token"],
            data["refresh_token"],
        )

    async def _refresh_supabase_token(
        self, supabase_refresh_token: str
    ) -> tuple[str, str]:
        """Refresh a Supabase session. Returns (new_access_token, new_refresh_token)."""
        resp = await self._http.post(
            f"{self.supabase_url}/auth/v1/token?grant_type=refresh_token",
            json={"refresh_token": supabase_refresh_token},
            headers={
                "apikey": self.supabase_anon_key,
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["access_token"], data["refresh_token"]
