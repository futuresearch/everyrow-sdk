"""OAuth 2.1 authorization provider for the EveryRow MCP server."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import secrets
import time
from typing import TYPE_CHECKING, Any
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
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from everyrow_mcp.redis_utils import build_key

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class SupabaseTokenVerifier(TokenVerifier):
    """Verify Supabase-issued JWTs using the project's JWKS endpoint."""

    def __init__(
        self,
        supabase_url: str,
        *,
        audience: str = "authenticated",
        redis: Redis,
        revocation_ttl: int = 3600,
    ) -> None:
        self._issuer = supabase_url.rstrip("/") + "/auth/v1"
        self._audience = audience
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
        return hashlib.sha256(token.encode()).hexdigest()

    async def deny_token(self, token: str) -> bool:
        try:
            key = build_key("revoked", self._token_fingerprint(token))
            await self._redis.setex(key, self._revocation_ttl, "1")
            return True
        except Exception:
            logger.warning("Failed to deny token (Redis unavailable)", exc_info=True)
            return False

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            async with self._jwks_lock:
                signing_key = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._jwks_client.get_signing_key_from_jwt, token
                    ),
                    timeout=10.0,
                )
            payload: dict[str, Any] = pyjwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                issuer=self._issuer,
                audience=self._audience,
                options={"require": ["exp", "sub", "iss", "aud"]},
            )

            _key = build_key("revoked", self._token_fingerprint(token))
            if await self._redis.exists(_key) > 0:
                logger.debug("Token is denied")
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
        except TimeoutError:
            logger.warning("JWKS fetch timed out (10s)")
            return None
        except pyjwt.PyJWTError:
            logger.debug("JWT verification failed", exc_info=True)
            return None


class EveryRowAuthorizationCode(AuthorizationCode):
    """Extends AuthorizationCode with the user's Supabase access token."""

    supabase_access_token: str
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
#
# Auth flow:
#
#   Claude MCP client            EveryRowAuthProvider          Supabase
#   ──────────────────           ────────────────────          ────────
#   1. POST /register  ──────►  store client_id in Redis
#   2. GET  /authorize ──────►  generate PKCE pair
#                                save PendingAuth ─────────►  redirect to
#                                                             Google OAuth
#   3.                 ◄─────────────────────────────────────  callback with
#                                                             auth code
#   4. GET /auth/callback ───►  exchange code for tokens ──►  POST /token
#                                issue auth code (Redis)       (PKCE)
#                                redirect with ?code=…
#   5. POST /token     ──────►  load+consume code (GETDEL)
#                                return Supabase JWT as
#                                MCP access_token
#   6. (refresh)       ──────►  rotate refresh token (GETDEL)
#                                refresh via Supabase ──────►  POST /token
#                                return new JWT                (refresh)


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
        *,
        token_verifier: SupabaseTokenVerifier | None = None,
        rate_limit: int = 10,
        rate_window: int = 60,
        access_token_ttl: int = 3300,
        auth_code_ttl: int = 300,
        pending_auth_ttl: int = 600,
        client_registration_ttl: int = 2_592_000,
        refresh_token_ttl: int = 604_800,
    ) -> None:
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_anon_key = supabase_anon_key
        self.mcp_server_url = mcp_server_url.rstrip("/")
        self._redis = redis
        self._token_verifier = token_verifier
        self._rate_limit = rate_limit
        self._rate_window = rate_window
        self._access_token_ttl = access_token_ttl
        self._auth_code_ttl = auth_code_ttl
        self._pending_auth_ttl = pending_auth_ttl
        self._client_registration_ttl = client_registration_ttl
        self._refresh_token_ttl = refresh_token_ttl
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    async def close(self) -> None:
        await self._http.aclose()

    # ── Helpers ─────────────────────────────────────────────────

    # SECURITY: This method skips signature verification. It MUST only be
    # called on tokens received directly from Supabase's token endpoint via
    # server-to-server HTTPS exchange. NEVER pass user-supplied tokens here.
    @staticmethod
    def _UNSAFE_decode_server_jwt(token: str) -> dict[str, Any]:
        """Decode a Supabase JWT received from a trusted server-to-server exchange.

        Skips signature verification — the token came from Supabase's token
        endpoint over HTTPS and was never exposed to the client.
        NEVER use this for tokens received from end users.
        """
        return pyjwt.decode(token, options={"verify_signature": False})

    async def _check_rate_limit(self, action: str, client_ip: str) -> None:
        rl_key = build_key("ratelimit", action, client_ip or "global")
        pipe = self._redis.pipeline()
        pipe.incr(rl_key)
        pipe.expire(rl_key, self._rate_window)
        count, _ = await pipe.execute()
        if count > self._rate_limit:
            raise ValueError(f"{action.title()} rate limit exceeded")

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        _key = build_key("client", client_id)
        client_data = await self._redis.get(_key)
        if client_data is None:
            return None
        return OAuthClientInformationFull.model_validate_json(client_data)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        cid = client_info.client_id
        if cid is None:
            raise ValueError("client_id is required")
        await self._redis.setex(
            name=build_key("client", cid),
            time=self._client_registration_ttl,
            value=client_info.model_dump_json(),
        )

    async def authorize(
        self,
        client: OAuthClientInformationFull,
        params: AuthorizationParams,
    ) -> str:
        if client.redirect_uris:
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
        await self._redis.setex(
            name=build_key("pending", state),
            time=self._pending_auth_ttl,
            value=pending.model_dump_json(),
        )
        return f"{self.mcp_server_url}/auth/start/{state}"

    async def handle_start(self, request: Request) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        try:
            await self._check_rate_limit("start", client_ip)
        except ValueError:
            return Response("Rate limit exceeded", status_code=429)

        state = request.path_params.get("state")
        if not state:
            return Response("Missing state", status_code=400)

        _key = build_key("pending", state)
        pending_data = await self._redis.get(_key)
        if pending_data is None:
            return Response("Invalid state", status_code=400)
        pending = PendingAuth.model_validate_json(pending_data)

        response = RedirectResponse(url=pending.supabase_redirect_url, status_code=302)
        response.set_cookie(
            key="mcp_auth_state",
            value=state,
            max_age=self._pending_auth_ttl,
            httponly=True,
            samesite="lax",
            secure=self.mcp_server_url.startswith("https"),
            path="/auth/callback",
        )
        return response

    async def handle_callback(self, request: Request) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        try:
            await self._check_rate_limit("callback", client_ip)
        except ValueError:
            return Response("Rate limit exceeded", status_code=429)

        code = request.query_params.get("code")
        state = request.cookies.get("mcp_auth_state")
        if not code or not state:
            return Response("Missing code or state cookie", status_code=400)

        _key = build_key("pending", state)
        pending_data = await self._redis.getdel(_key)
        if pending_data is None:
            return Response("No pending authorization found", status_code=400)
        pending = PendingAuth.model_validate_json(pending_data)

        client_info = await self.get_client(pending.client_id)
        if client_info is None or (
            pending.params.redirect_uri
            and client_info.redirect_uris
            and str(pending.params.redirect_uri)
            not in [str(u) for u in client_info.redirect_uris]
        ):
            if client_info is not None:
                logger.warning(
                    "redirect_uri mismatch for client %s in callback",
                    pending.client_id,
                )
            return Response("Invalid client or redirect_uri", status_code=400)

        try:
            (
                _user_id,
                _email,
                supabase_access_token,
                supabase_refresh,
            ) = await self._exchange_supabase_code(
                code, code_verifier=pending.supabase_code_verifier
            )
        except Exception:
            logger.exception("Failed to exchange Supabase code")
            return Response("Failed to authenticate with Supabase", status_code=500)

        auth_code_str = secrets.token_urlsafe(32)
        auth_code_obj = EveryRowAuthorizationCode(
            code=auth_code_str,
            client_id=pending.client_id,
            redirect_uri=pending.params.redirect_uri,
            redirect_uri_provided_explicitly=pending.params.redirect_uri_provided_explicitly,
            code_challenge=pending.params.code_challenge,
            scopes=pending.params.scopes or [],
            expires_at=time.time() + self._auth_code_ttl,
            resource=pending.params.resource,
            supabase_access_token=supabase_access_token,
            supabase_refresh_token=supabase_refresh,
        )
        await self._redis.setex(
            name=build_key("authcode", auth_code_str),
            time=self._auth_code_ttl,
            value=auth_code_obj.model_dump_json(),
        )

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

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> EveryRowAuthorizationCode | None:
        if len(authorization_code) > 256:
            return None

        _key = build_key("authcode", authorization_code)
        code_data = await self._redis.getdel(_key)
        if code_data is None:
            return None
        code_obj = EveryRowAuthorizationCode.model_validate_json(code_data)
        if code_obj.client_id != client.client_id:
            return None
        return code_obj

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: EveryRowAuthorizationCode,
    ) -> OAuthToken:
        jwt_claims = self._UNSAFE_decode_server_jwt(
            authorization_code.supabase_access_token
        )
        expires_in = max(0, jwt_claims.get("exp", 0) - int(time.time()))

        refresh_token_str: str | None = None
        if authorization_code.supabase_refresh_token:
            refresh_token_str = secrets.token_urlsafe(32)
            rt = EveryRowRefreshToken(
                token=refresh_token_str,
                client_id=client.client_id or "",
                scopes=authorization_code.scopes,
                supabase_refresh_token=authorization_code.supabase_refresh_token,
            )
            await self._redis.setex(
                name=build_key("refresh", refresh_token_str),
                time=self._refresh_token_ttl,
                value=rt.model_dump_json(),
            )

        return OAuthToken(
            access_token=authorization_code.supabase_access_token,
            token_type="Bearer",
            expires_in=expires_in,
            refresh_token=refresh_token_str,
        )

    async def load_access_token(self, token: str) -> AccessToken | None:  # noqa: ARG002
        return None

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> EveryRowRefreshToken | None:
        if len(refresh_token) > 256:
            return None

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

        jwt_claims = self._UNSAFE_decode_server_jwt(new_jwt)
        expires_in = max(0, jwt_claims.get("exp", 0) - int(time.time()))

        # OAuth 2.1 §6.3: requested scopes must not exceed original grant.
        if scopes:
            narrowed = list(set(scopes) & set(refresh_token.scopes))
            if not narrowed:
                raise ValueError(
                    "Requested scopes have no overlap with the original grant"
                )
            final_scopes = narrowed
        else:
            final_scopes = refresh_token.scopes

        # Issue new refresh token
        new_rt_str = secrets.token_urlsafe(32)
        new_rt = EveryRowRefreshToken(
            token=new_rt_str,
            client_id=client.client_id or "",
            scopes=final_scopes,
            supabase_refresh_token=new_supabase_refresh,
        )
        await self._redis.setex(
            name=build_key("refresh", new_rt_str),
            time=self._refresh_token_ttl,
            value=new_rt.model_dump_json(),
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
        elif isinstance(token, AccessToken) and self._token_verifier is not None:
            await self._token_verifier.deny_token(token.token)

    async def _exchange_supabase_code(
        self, code: str, code_verifier: str = ""
    ) -> tuple[str, str, str, str]:
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
