"""Tests for Supabase JWT verification and OAuth provider."""

import time
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import fakeredis.aioredis
import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)
from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl
from starlette.requests import Request

from everyrow_mcp.auth import (
    EveryRowAuthorizationCode,
    EveryRowAuthProvider,
    PendingAuth,
    SupabaseTokenVerifier,
)
from everyrow_mcp.redis_utils import build_key

SUPABASE_URL = "https://test.supabase.co"
ISSUER = SUPABASE_URL + "/auth/v1"


# ── Verifier fixtures ────────────────────────────────────────────────


@pytest.fixture
def rsa_keypair():
    """Generate an RSA key pair for signing test JWTs."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture
def verifier(rsa_keypair):
    """Create a SupabaseTokenVerifier with a mocked JWKS client."""
    _private_key, public_key = rsa_keypair
    verifier = SupabaseTokenVerifier(SUPABASE_URL)

    mock_signing_key = MagicMock()
    mock_signing_key.key = public_key.public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )
    mock_signing_key._algorithm = "RS256"
    verifier._jwks_client = MagicMock()
    verifier._jwks_client.get_signing_key_from_jwt = MagicMock(
        return_value=mock_signing_key
    )

    return verifier


def _make_jwt(private_key, claims: dict | None = None) -> str:
    """Create a signed JWT with default claims, optionally overriding."""
    payload = {
        "sub": "user-123",
        "aud": "authenticated",
        "iss": ISSUER,
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
        "scope": "read write",
    }
    if claims:
        payload.update(claims)
    return jwt.encode(payload, private_key, algorithm="RS256")


# ── Provider fixtures ────────────────────────────────────────────────


@pytest.fixture
def fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def provider(fake_redis) -> EveryRowAuthProvider:
    return EveryRowAuthProvider(
        supabase_url=SUPABASE_URL,
        supabase_anon_key="test-anon-key",
        mcp_server_url="https://mcp.example.com",
        redis=fake_redis,
    )


@pytest.fixture
def client_info() -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id="test-client-id",
        client_secret="test-client-secret",
        redirect_uris=[AnyUrl("https://claude.ai/callback")],
        client_name="Test Client",
    )


@pytest.fixture
def auth_params() -> AuthorizationParams:
    return AuthorizationParams(
        state="client-state-abc",
        scopes=["read"],
        code_challenge="test-code-challenge",
        redirect_uri=AnyUrl("https://claude.ai/callback"),
        redirect_uri_provided_explicitly=True,
    )


# ── Token verifier tests ────────────────────────────────────────────


class TestSupabaseTokenVerifier:
    @pytest.mark.asyncio
    async def test_valid_jwt(self, verifier, rsa_keypair):
        private_key, _ = rsa_keypair
        token = _make_jwt(private_key)

        result = await verifier.verify_token(token)

        assert result is not None
        assert result.token == token
        assert result.client_id == "user-123"
        assert result.scopes == ["read", "write"]
        assert result.expires_at is not None

    @pytest.mark.asyncio
    async def test_valid_jwt_no_scope(self, verifier, rsa_keypair):
        private_key, _ = rsa_keypair
        token = _make_jwt(private_key, {"scope": ""})

        result = await verifier.verify_token(token)

        assert result is not None
        assert result.scopes == []

    @pytest.mark.asyncio
    async def test_expired_jwt(self, verifier, rsa_keypair):
        private_key, _ = rsa_keypair
        token = _make_jwt(private_key, {"exp": int(time.time()) - 100})

        assert await verifier.verify_token(token) is None

    @pytest.mark.asyncio
    async def test_wrong_issuer(self, verifier, rsa_keypair):
        private_key, _ = rsa_keypair
        token = _make_jwt(private_key, {"iss": "https://evil.example.com/auth/v1"})

        assert await verifier.verify_token(token) is None

    @pytest.mark.asyncio
    async def test_wrong_audience(self, verifier, rsa_keypair):
        private_key, _ = rsa_keypair
        token = _make_jwt(private_key, {"aud": "wrong-audience"})

        assert await verifier.verify_token(token) is None

    @pytest.mark.asyncio
    async def test_invalid_signature(self, verifier):
        other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        token = _make_jwt(other_key)

        assert await verifier.verify_token(token) is None

    @pytest.mark.asyncio
    async def test_malformed_token(self, verifier):
        assert await verifier.verify_token("not-a-jwt") is None

    @pytest.mark.asyncio
    async def test_jwks_endpoint_url(self):
        with patch("everyrow_mcp.auth.PyJWKClient") as mock_jwk_cls:
            SupabaseTokenVerifier("https://my-project.supabase.co")
            mock_jwk_cls.assert_called_once_with(
                "https://my-project.supabase.co/auth/v1/.well-known/jwks.json",
                cache_keys=True,
            )

    @pytest.mark.asyncio
    async def test_trailing_slash_normalized(self):
        with patch("everyrow_mcp.auth.PyJWKClient") as mock_jwk_cls:
            v = SupabaseTokenVerifier("https://my-project.supabase.co/")
            mock_jwk_cls.assert_called_once_with(
                "https://my-project.supabase.co/auth/v1/.well-known/jwks.json",
                cache_keys=True,
            )
            assert v._issuer == "https://my-project.supabase.co/auth/v1"


# ── Provider tests ───────────────────────────────────────────────────


class TestClientRegistration:
    @pytest.mark.asyncio
    async def test_register_and_get_client(self, provider, client_info):
        await provider.register_client(client_info)
        result = await provider.get_client("test-client-id")
        assert result is not None
        assert result.client_id == "test-client-id"

    @pytest.mark.asyncio
    async def test_get_nonexistent_client(self, provider):
        assert await provider.get_client("nonexistent") is None


class TestAuthorize:
    @pytest.mark.asyncio
    async def test_authorize_returns_start_redirect(
        self, provider, client_info, auth_params
    ):
        await provider.register_client(client_info)
        redirect_url = await provider.authorize(client_info, auth_params)

        parsed = urlparse(redirect_url)
        assert parsed.path.startswith("/auth/start/")

        state = parsed.path.split("/auth/start/")[1]
        pending = await provider._redis_get(build_key("pending", state), PendingAuth)
        assert pending is not None

        supabase_url = urlparse(pending.supabase_redirect_url)
        assert supabase_url.hostname == "test.supabase.co"
        assert supabase_url.path == "/auth/v1/authorize"

        params = parse_qs(supabase_url.query)
        assert params["provider"] == ["google"]
        assert params["flow_type"] == ["pkce"]


class TestCallback:
    @pytest.mark.asyncio
    async def test_callback_happy_path(self, provider, client_info, auth_params):
        await provider.register_client(client_info)
        redirect_url = await provider.authorize(client_info, auth_params)
        state = urlparse(redirect_url).path.split("/auth/start/")[1]

        token_resp = MagicMock(spec=httpx.Response)
        token_resp.status_code = 200
        token_resp.raise_for_status = MagicMock(return_value=None)
        token_resp.json = MagicMock(
            return_value={
                "user": {"id": "user-123", "email": "user@example.com"},
                "access_token": "supabase-jwt-token",
                "refresh_token": "supabase-refresh-token",
            }
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=token_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("everyrow_mcp.auth.httpx.AsyncClient", return_value=mock_client):
            scope = {
                "type": "http",
                "method": "GET",
                "path": "/auth/callback",
                "query_string": b"code=supabase-code",
                "headers": [(b"cookie", f"mcp_auth_state={state}".encode())],
            }
            response = await provider.handle_callback(Request(scope))

        assert response.status_code == 302
        location = response.headers["location"]
        params = parse_qs(urlparse(location).query)
        assert "code" in params
        assert params["state"] == ["client-state-abc"]

        # Auth code should carry the Supabase JWT
        auth_code = params["code"][0]
        stored = await provider._redis_get(
            build_key("authcode", auth_code), EveryRowAuthorizationCode
        )
        assert stored is not None
        assert stored.supabase_jwt == "supabase-jwt-token"

    @pytest.mark.asyncio
    async def test_callback_missing_code(self, provider):
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/auth/callback",
            "query_string": b"",
            "headers": [(b"cookie", b"mcp_auth_state=some-state")],
        }
        response = await provider.handle_callback(Request(scope))
        assert response.status_code == 400


class TestTokenExchange:
    @pytest.mark.asyncio
    async def test_exchange_returns_supabase_jwt_as_access_token(
        self, provider, client_info
    ):
        await provider.register_client(client_info)

        auth_code = EveryRowAuthorizationCode(
            code="test-code",
            client_id="test-client-id",
            redirect_uri=AnyUrl("https://claude.ai/callback"),
            redirect_uri_provided_explicitly=True,
            code_challenge="test-challenge",
            scopes=["read"],
            expires_at=time.time() + 300,
            supabase_jwt="the-supabase-jwt",
        )
        await provider._redis_set(
            build_key("authcode", "test-code"), auth_code, ttl=300
        )

        token = await provider.exchange_authorization_code(client_info, auth_code)

        # The MCP access token IS the Supabase JWT
        assert token.access_token == "the-supabase-jwt"
        assert token.token_type == "Bearer"


class TestStubs:
    @pytest.mark.asyncio
    async def test_load_access_token_returns_none(self, provider):
        """load_access_token is unused (verifier handles it) — always returns None."""
        assert await provider.load_access_token("any-token") is None

    @pytest.mark.asyncio
    async def test_load_refresh_token_returns_none(self, provider, client_info):
        assert await provider.load_refresh_token(client_info, "any-token") is None
