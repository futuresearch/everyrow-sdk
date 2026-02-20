"""OAuth auth for EveryRow MCP server.

Why custom auth code instead of Supabase's built-in OAuth 2.1 Server?
----------------------------------------------------------------------
Claude AI's MCP client requires the full OAuth 2.1 Authorization Server flow:
``/.well-known/oauth-authorization-server``, dynamic client registration
(``/register``), ``/authorize``, and ``/token`` endpoints.

We first tried pointing to Supabase as the AS (commit ``aac80ee``), but
Supabase's OAuth 2.1 Server feature is **not enabled** on our project -- its
``/.well-known/oauth-authorization-server/auth/v1`` endpoint returns **404**.
Enabling it would require: turning it on in the Supabase dashboard, migrating
JWT signing to asymmetric keys, building a custom consent UI, and enabling
dynamic client registration. Too much dashboard/infra work for now.

So our MCP server acts as the full AS itself (``EveryRowAuthProvider``),
delegating user login to Supabase Google OAuth behind the scenes. This code
was already written, security-hardened, and tested (pre-``aac80ee``).

Future improvement: migrate to native Supabase OAuth 2.1 Server once it's
configured on the project.

Hybrid model:
- SupabaseTokenVerifier validates Supabase JWTs via JWKS (resource-server mode)
- EveryRowAuthProvider handles client registration + OAuth flow, delegating
  user authentication to Supabase and returning the Supabase JWT directly
  as the MCP access token.

Redis storage model:
  Auth codes, refresh tokens, and client registrations are stored as plain JSON
  in Redis (no at-rest encryption). This is acceptable because:
  1. All tokens are short-lived (auth codes 5 min, refresh tokens 7 days) and
     single-use (consumed atomically via GETDEL).
  2. Redis must be deployed on an internal network with AUTH enabled and
     TLS in transit -- see deployment docs.
  3. The access tokens themselves (Supabase JWTs) are never stored in Redis;
     they are verified via JWKS on every request.
  If your threat model includes Redis host compromise, add Fernet/AES envelope
  encryption in _redis_set/_redis_get and rotate the key via settings.

Refresh tokens supported -- opaque tokens in Redis mapped to Supabase refresh tokens.
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

# Rate limits
REGISTRATION_RATE_LIMIT = 10  # max registrations per window
REGISTRATION_RATE_WINDOW = 60  # seconds


# ── Token verifier (resource-server mode) ─────────────────────────────


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


# ── OAuth provider ────────────────────────────────────────────────────


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
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    async def close(self) -> None:
        """Close the HTTP client. Call from your lifespan handler."""
        await self._http.aclose()

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _decode_trusted_supabase_jwt(token: str) -> dict[str, Any]:
        """Decode a JWT that was *just* received from Supabase's token endpoint.

        This skips signature verification because the token came from a
        server-to-server exchange over HTTPS -- it was never exposed to the
        client.  NEVER use this for tokens received from end users.
        """
        return pyjwt.decode(token, options={"verify_signature": False})

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
        # Global rate limit on registrations (atomic pipeline avoids orphan keys
        # if process crashes between INCR and EXPIRE).
        rl_key = build_key("ratelimit", "register")
        pipe = self._redis.pipeline()
        pipe.incr(rl_key)
        pipe.expire(rl_key, REGISTRATION_RATE_WINDOW)
        count, _ = await pipe.execute()
        if count > REGISTRATION_RATE_LIMIT:
            raise ValueError("Registration rate limit exceeded")

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
        # Defense-in-depth: verify redirect_uri against registered URIs
        if params.redirect_uri and client.redirect_uris:
            if str(params.redirect_uri) not in [str(u) for u in client.redirect_uris]:
                raise ValueError("redirect_uri does not match any registered URI")

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
            path="/auth/callback",
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

        # Defense-in-depth: re-verify redirect_uri against registered client
        client_info = await self.get_client(pending.client_id)
        if client_info is None:
            return Response("Unknown client", status_code=400)
        if pending.params.redirect_uri and client_info.redirect_uris:
            if str(pending.params.redirect_uri) not in [
                str(u) for u in client_info.redirect_uris
            ]:
                logger.warning(
                    "redirect_uri mismatch for client %s in callback",
                    pending.client_id,
                )
                return Response("Invalid redirect_uri", status_code=400)

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
        response = RedirectResponse(
            url=f"{pending.params.redirect_uri}?{urlencode(redirect_params)}",
            status_code=302,
        )
        response.delete_cookie(
            "mcp_auth_state",
            path="/auth/callback",
            httponly=True,
            samesite="lax",
            secure=self.mcp_server_url.startswith("https"),
        )
        return response

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

        jwt_claims = self._decode_trusted_supabase_jwt(authorization_code.supabase_jwt)
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
        # Not used -- SupabaseTokenVerifier handles verification via JWKS
        return None

    # ── Refresh / revocation ─────────────────────────────────────

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> EveryRowRefreshToken | None:
        # Atomic GETDEL: prevents race condition where two concurrent refresh
        # requests both succeed with the same token, defeating rotation.
        key = build_key("refresh", refresh_token)
        data = await self._redis.getdel(key)
        if data is None:
            return None
        rt = EveryRowRefreshToken.model_validate_json(data)
        if rt.client_id != client.client_id:
            return None
        return rt

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: EveryRowRefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        # Old refresh token already consumed atomically in load_refresh_token (GETDEL)
        new_jwt, new_supabase_refresh = await self._refresh_supabase_token(
            refresh_token.supabase_refresh_token
        )

        jwt_claims = self._decode_trusted_supabase_jwt(new_jwt)
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
