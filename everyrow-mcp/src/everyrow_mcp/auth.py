"""OAuth 2.1 authorization provider for EveryRow MCP server.

Implements the MCP OAuthAuthorizationServerProvider protocol, delegating
user authentication to Supabase (Google SSO) and passing the Supabase JWT
straight through to the EveryRow API.

Session state is stored in Redis so it survives pod restarts.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import time
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import BaseModel
from redis.asyncio import Redis
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from everyrow_mcp.redis_utils import build_key

logger = logging.getLogger(__name__)

# Token lifetimes
ACCESS_TOKEN_TTL = 3300  # 55 min (expire before Supabase JWT's 1h TTL)
REFRESH_TOKEN_TTL = 86400 * 30  # 30 days
AUTH_CODE_TTL = 300  # 5 minutes
PENDING_AUTH_TTL = 600  # 10 minutes


class EveryRowAccessToken(AccessToken):
    """Extends AccessToken with the user's Supabase JWT."""

    supabase_jwt: str


class EveryRowRefreshToken(RefreshToken):
    """Extends RefreshToken with the user's Supabase JWT and refresh token."""

    supabase_jwt: str
    supabase_refresh_token: str


class EveryRowAuthorizationCode(AuthorizationCode):
    """Extends AuthorizationCode with the user's Supabase JWT and refresh token."""

    supabase_jwt: str
    supabase_refresh_token: str


class PendingAuth(BaseModel):
    """Saved between /authorize and /auth/callback."""

    client_id: str
    params: AuthorizationParams
    supabase_code_verifier: str = ""
    supabase_redirect_url: str = ""
    created_at: float = 0.0


def _jwt_hash(jwt: str) -> str:
    """Return a short SHA-256 hash of a JWT for use in secondary index keys."""
    return hashlib.sha256(jwt.encode()).hexdigest()[:16]


class EveryRowAuthProvider(
    OAuthAuthorizationServerProvider[
        EveryRowAuthorizationCode, EveryRowRefreshToken, EveryRowAccessToken
    ]
):
    """OAuth provider that authenticates via Supabase and passes the JWT through."""

    def __init__(
        self,
        supabase_url: str,
        supabase_anon_key: str,
        mcp_server_url: str,
        redis: Redis,
        encryption_key: str | None = None,
    ) -> None:
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_anon_key = supabase_anon_key
        self.mcp_server_url = mcp_server_url.rstrip("/")
        self._redis = redis
        self._fernet = Fernet(encryption_key.encode()) if encryption_key else None

    # ── Redis helpers ────────────────────────────────────────────────

    async def _redis_set(
        self, key: str, obj: BaseModel, ttl: int | None = None
    ) -> None:
        """Serialize a Pydantic model and store it in Redis (encrypted if key set)."""
        data: str = obj.model_dump_json()
        if self._fernet:
            data = self._fernet.encrypt(data.encode()).decode()
        if ttl is not None:
            await self._redis.setex(key, ttl, data)
        else:
            await self._redis.set(key, data)

    async def _redis_get(
        self, key: str, model_class: type[BaseModel]
    ) -> BaseModel | None:
        """Fetch and deserialize a Pydantic model from Redis (decrypting if needed)."""
        data = await self._redis.get(key)
        if data is None:
            return None
        if self._fernet:
            data = self._fernet.decrypt(data.encode()).decode()
        return model_class.model_validate_json(data)

    async def _redis_pop(
        self, key: str, model_class: type[BaseModel]
    ) -> BaseModel | None:
        """Atomically GET + DEL a key, returning the deserialized model."""
        data = await self._redis.getdel(key)
        if data is None:
            return None
        if self._fernet:
            data = self._fernet.decrypt(data.encode()).decode()
        return model_class.model_validate_json(data)

    # ── OAuth client management ─────────────────────────────────────

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return await self._redis_get(
            build_key("client", client_id), OAuthClientInformationFull
        )  # type: ignore[return-value]

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        await self._redis_set(build_key("client", client_info.client_id), client_info)

    # ── Authorization ───────────────────────────────────────────────

    async def authorize(
        self,
        client: OAuthClientInformationFull,
        params: AuthorizationParams,
    ) -> str:
        # Generate a state token to link callback back to this auth request
        state = secrets.token_urlsafe(32)

        # Generate PKCE pair for the Supabase leg of the OAuth flow
        supabase_verifier = secrets.token_urlsafe(32)
        challenge_bytes = hashlib.sha256(supabase_verifier.encode()).digest()
        supabase_challenge = (
            base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode()
        )

        # Build the Supabase redirect URL
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
            client_id=client.client_id,
            params=params,
            supabase_code_verifier=supabase_verifier,
            supabase_redirect_url=supabase_redirect_url,
            created_at=time.time(),
        )
        await self._redis_set(
            build_key("pending", state), pending, ttl=PENDING_AUTH_TTL
        )

        # Redirect to our /auth/start/{state} intermediate route, which sets
        # a cookie with the state and then redirects to Supabase.
        return f"{self.mcp_server_url}/auth/start/{state}"

    # ── Start (custom route — sets cookie, redirects to Supabase) ──

    async def handle_start(self, request: Request) -> Response:
        """Set a cookie with the OAuth state and redirect to Supabase."""
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

    # ── Callback (custom route, not part of protocol) ───────────────

    async def handle_callback(self, request: Request) -> Response:
        """Handle the Supabase OAuth callback."""
        code = request.query_params.get("code")
        state = request.cookies.get("mcp_auth_state")
        if not code or not state:
            return Response("Missing code or state cookie", status_code=400)

        pending = await self._redis_pop(build_key("pending", state), PendingAuth)

        if pending is None:
            return Response("No pending authorization found", status_code=400)

        if time.time() - pending.created_at > PENDING_AUTH_TTL:
            return Response("Authorization request expired", status_code=400)

        # Exchange Supabase code for user identity and JWT (with PKCE verifier)
        try:
            (
                user_id,
                _email,
                supabase_jwt,
                supabase_refresh,
            ) = await self._exchange_supabase_code(
                code, code_verifier=pending.supabase_code_verifier
            )
        except Exception:
            logger.exception("Failed to exchange Supabase code")
            return Response("Failed to authenticate with Supabase", status_code=500)

        logger.info("Authenticated user %s via Supabase JWT", user_id)

        # Issue authorization code
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

        # Redirect back to OAuth client
        redirect_params: dict[str, str] = {"code": auth_code_str}
        if pending.params.state:
            redirect_params["state"] = pending.params.state

        redirect_url = f"{pending.params.redirect_uri}?{urlencode(redirect_params)}"
        return RedirectResponse(url=redirect_url, status_code=302)

    # ── Authorization code exchange ─────────────────────────────────

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> EveryRowAuthorizationCode | None:
        code_obj = await self._redis_get(
            build_key("authcode", authorization_code), EveryRowAuthorizationCode
        )
        if code_obj is None:
            return None
        if code_obj.client_id != client.client_id:
            return None
        return code_obj  # type: ignore[return-value]

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: EveryRowAuthorizationCode,
    ) -> OAuthToken:
        # Remove used code (one-time use)
        await self._redis.delete(build_key("authcode", authorization_code.code))

        # Issue access token
        access_token_str = secrets.token_urlsafe(32)
        now = int(time.time())
        access_token_obj = EveryRowAccessToken(
            token=access_token_str,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=now + ACCESS_TOKEN_TTL,
            resource=authorization_code.resource,
            supabase_jwt=authorization_code.supabase_jwt,
        )
        await self._redis_set(
            build_key("access", access_token_str),
            access_token_obj,
            ttl=ACCESS_TOKEN_TTL,
        )

        # Issue refresh token
        refresh_token_str = secrets.token_urlsafe(32)
        refresh_token_obj = EveryRowRefreshToken(
            token=refresh_token_str,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=now + REFRESH_TOKEN_TTL,
            supabase_jwt=authorization_code.supabase_jwt,
            supabase_refresh_token=authorization_code.supabase_refresh_token,
        )
        await self._redis_set(
            build_key("refresh", refresh_token_str),
            refresh_token_obj,
            ttl=REFRESH_TOKEN_TTL,
        )

        # Secondary index: track access tokens by (client_id, jwt_hash)
        idx_key = build_key(
            "idx",
            "access_by_cj",
            client.client_id,
            _jwt_hash(authorization_code.supabase_jwt),
        )
        await self._redis.sadd(idx_key, access_token_str)
        await self._redis.expire(idx_key, ACCESS_TOKEN_TTL)

        return OAuthToken(
            access_token=access_token_str,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL,
            refresh_token=refresh_token_str,
            scope=" ".join(authorization_code.scopes)
            if authorization_code.scopes
            else None,
        )

    # ── Token refresh ───────────────────────────────────────────────

    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str,
    ) -> EveryRowRefreshToken | None:
        token_obj = await self._redis_get(
            build_key("refresh", refresh_token), EveryRowRefreshToken
        )
        if token_obj is None:
            return None
        if token_obj.client_id != client.client_id:
            return None
        return token_obj  # type: ignore[return-value]

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: EveryRowRefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        # Refresh the Supabase JWT (propagates on failure — client must re-auth)
        new_jwt, new_supabase_refresh = await self._refresh_supabase_token(
            refresh_token.supabase_refresh_token
        )

        # Revoke old refresh token
        await self._redis.delete(build_key("refresh", refresh_token.token))

        # Revoke old access tokens for this client with same JWT via secondary index
        idx_key = build_key(
            "idx",
            "access_by_cj",
            client.client_id,
            _jwt_hash(refresh_token.supabase_jwt),
        )
        old_access_tokens = await self._redis.smembers(idx_key)
        if old_access_tokens:
            access_keys = [build_key("access", t) for t in old_access_tokens]
            await self._redis.delete(*access_keys)
        await self._redis.delete(idx_key)

        # Issue new access token with refreshed JWT
        now = int(time.time())
        new_scopes = scopes if scopes else refresh_token.scopes
        access_token_str = secrets.token_urlsafe(32)
        access_token_obj = EveryRowAccessToken(
            token=access_token_str,
            client_id=client.client_id,
            scopes=new_scopes,
            expires_at=now + ACCESS_TOKEN_TTL,
            supabase_jwt=new_jwt,
        )
        await self._redis_set(
            build_key("access", access_token_str),
            access_token_obj,
            ttl=ACCESS_TOKEN_TTL,
        )

        # Issue new refresh token with refreshed Supabase tokens
        new_refresh_token_str = secrets.token_urlsafe(32)
        new_refresh_token_obj = EveryRowRefreshToken(
            token=new_refresh_token_str,
            client_id=client.client_id,
            scopes=new_scopes,
            expires_at=now + REFRESH_TOKEN_TTL,
            supabase_jwt=new_jwt,
            supabase_refresh_token=new_supabase_refresh,
        )
        await self._redis_set(
            build_key("refresh", new_refresh_token_str),
            new_refresh_token_obj,
            ttl=REFRESH_TOKEN_TTL,
        )

        # Update secondary index for new access token
        new_idx_key = build_key(
            "idx", "access_by_cj", client.client_id, _jwt_hash(new_jwt)
        )
        await self._redis.sadd(new_idx_key, access_token_str)
        await self._redis.expire(new_idx_key, ACCESS_TOKEN_TTL)

        return OAuthToken(
            access_token=access_token_str,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL,
            refresh_token=new_refresh_token_str,
            scope=" ".join(new_scopes) if new_scopes else None,
        )

    # ── Access token verification ───────────────────────────────────

    async def load_access_token(self, token: str) -> EveryRowAccessToken | None:
        return await self._redis_get(build_key("access", token), EveryRowAccessToken)  # type: ignore[return-value]

    # ── Token revocation ────────────────────────────────────────────

    async def revoke_token(
        self,
        token: EveryRowAccessToken | EveryRowRefreshToken,
    ) -> None:
        if isinstance(token, EveryRowAccessToken):
            await self._redis.delete(build_key("access", token.token))
        elif isinstance(token, EveryRowRefreshToken):
            await self._redis.delete(build_key("refresh", token.token))

    # ── Supabase integration ────────────────────────────────────────

    async def _exchange_supabase_code(
        self, code: str, code_verifier: str = ""
    ) -> tuple[str, str, str, str]:
        """Exchange a Supabase OAuth code for user identity and tokens.

        Returns:
            Tuple of (user_id, email, access_token, refresh_token).
        """
        body: dict[str, str] = {
            "auth_code": code,
            "code_verifier": code_verifier,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.supabase_url}/auth/v1/token?grant_type=pkce",
                json=body,
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

    async def _refresh_supabase_token(self, refresh_token: str) -> tuple[str, str]:
        """Refresh a Supabase JWT using a refresh token.

        Returns:
            Tuple of (new_access_token, new_refresh_token).
        """
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.supabase_url}/auth/v1/token?grant_type=refresh_token",
                json={"refresh_token": refresh_token},
                headers={
                    "apikey": self.supabase_anon_key,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["access_token"], data["refresh_token"]
