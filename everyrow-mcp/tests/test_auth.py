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
    CLIENT_REGISTRATION_TTL,
    REFRESH_TOKEN_TTL,
    EveryRowAuthorizationCode,
    EveryRowAuthProvider,
    EveryRowRefreshToken,
    PendingAuth,
    SupabaseTokenVerifier,
)
from everyrow_mcp.redis_utils import build_key

SUPABASE_URL = "https://test.supabase.co"
ISSUER = SUPABASE_URL + "/auth/v1"

# Shared secret for creating decodable (but unsigned) test JWTs
_TEST_JWT_SECRET = "test-secret"


def _make_unsigned_jwt(exp: int | None = None, sub: str = "user-123") -> str:
    """Create a JWT that can be decoded with verify_signature=False."""
    payload = {
        "sub": sub,
        "aud": "authenticated",
        "iss": ISSUER,
        "exp": exp or int(time.time()) + 3600,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, _TEST_JWT_SECRET, algorithm="HS256")


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

    @pytest.mark.asyncio
    async def test_algorithm_from_header_not_private_attr(self, rsa_keypair):
        """verify_token reads alg from the JWT header, not signing_key._algorithm."""
        private_key, public_key = rsa_keypair
        token = _make_jwt(private_key)

        verifier = SupabaseTokenVerifier(SUPABASE_URL)

        # Signing key with NO _algorithm attr
        mock_signing_key = MagicMock(spec=[])  # empty spec = no attributes
        mock_signing_key.key = public_key.public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        )
        verifier._jwks_client = MagicMock()
        verifier._jwks_client.get_signing_key_from_jwt = MagicMock(
            return_value=mock_signing_key
        )

        result = await verifier.verify_token(token)
        assert result is not None
        assert result.client_id == "user-123"

    @pytest.mark.asyncio
    async def test_jwks_call_runs_in_thread(self, rsa_keypair):
        """get_signing_key_from_jwt is called via asyncio.to_thread."""
        private_key, public_key = rsa_keypair
        token = _make_jwt(private_key)

        verifier = SupabaseTokenVerifier(SUPABASE_URL)
        mock_signing_key = MagicMock()
        mock_signing_key.key = public_key.public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        )
        verifier._jwks_client = MagicMock()
        verifier._jwks_client.get_signing_key_from_jwt = MagicMock(
            return_value=mock_signing_key
        )

        with patch(
            "everyrow_mcp.auth.asyncio.to_thread", new_callable=AsyncMock
        ) as mock_to_thread:
            mock_to_thread.return_value = mock_signing_key
            result = await verifier.verify_token(token)

            mock_to_thread.assert_called_once_with(
                verifier._jwks_client.get_signing_key_from_jwt, token
            )
            assert result is not None


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

    @pytest.mark.asyncio
    async def test_register_client_has_ttl(self, provider, client_info, fake_redis):
        """Client registration should have a TTL to prevent unbounded growth."""
        await provider.register_client(client_info)
        ttl = await fake_redis.ttl(build_key("client", "test-client-id"))
        assert ttl > 0
        assert ttl <= CLIENT_REGISTRATION_TTL


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


def _make_supabase_token_response(
    jwt_token: str | None = None,
    refresh_token: str = "supabase-refresh-token",
) -> MagicMock:
    """Build a mock httpx.Response for Supabase token exchange."""
    if jwt_token is None:
        jwt_token = _make_unsigned_jwt()
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.raise_for_status = MagicMock(return_value=None)
    resp.json = MagicMock(
        return_value={
            "user": {"id": "user-123", "email": "user@example.com"},
            "access_token": jwt_token,
            "refresh_token": refresh_token,
        }
    )
    return resp


class TestCallback:
    @pytest.mark.asyncio
    async def test_callback_happy_path(self, provider, client_info, auth_params):
        await provider.register_client(client_info)
        redirect_url = await provider.authorize(client_info, auth_params)
        state = urlparse(redirect_url).path.split("/auth/start/")[1]

        fake_jwt = _make_unsigned_jwt()
        token_resp = _make_supabase_token_response(jwt_token=fake_jwt)

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=token_resp)
        provider._http = mock_http

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

        # Auth code should carry the Supabase JWT and refresh token
        auth_code = params["code"][0]
        stored = await provider._redis_get(
            build_key("authcode", auth_code), EveryRowAuthorizationCode
        )
        assert stored is not None
        assert stored.supabase_jwt == fake_jwt
        assert stored.supabase_refresh_token == "supabase-refresh-token"

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
        fake_jwt = _make_unsigned_jwt(exp=int(time.time()) + 3600)

        auth_code = EveryRowAuthorizationCode(
            code="test-code",
            client_id="test-client-id",
            redirect_uri=AnyUrl("https://claude.ai/callback"),
            redirect_uri_provided_explicitly=True,
            code_challenge="test-challenge",
            scopes=["read"],
            expires_at=time.time() + 300,
            supabase_jwt=fake_jwt,
            supabase_refresh_token="supabase-rt",
        )
        await provider._redis_set(
            build_key("authcode", "test-code"), auth_code, ttl=300
        )

        token = await provider.exchange_authorization_code(client_info, auth_code)

        # The MCP access token IS the Supabase JWT
        assert token.access_token == fake_jwt
        assert token.token_type == "Bearer"
        # expires_in derived from JWT exp, not hardcoded
        assert token.expires_in is not None
        assert 3500 < token.expires_in <= 3600
        # Refresh token issued
        assert token.refresh_token is not None

    @pytest.mark.asyncio
    async def test_exchange_no_refresh_when_missing(self, provider, client_info):
        """When supabase_refresh_token is empty, no refresh token is issued."""
        auth_code = EveryRowAuthorizationCode(
            code="test-code-2",
            client_id="test-client-id",
            redirect_uri=AnyUrl("https://claude.ai/callback"),
            redirect_uri_provided_explicitly=True,
            code_challenge="test-challenge",
            scopes=["read"],
            expires_at=time.time() + 300,
            supabase_jwt=_make_unsigned_jwt(exp=int(time.time()) + 3600),
            supabase_refresh_token="",
        )
        await provider._redis_set(
            build_key("authcode", "test-code-2"), auth_code, ttl=300
        )

        token = await provider.exchange_authorization_code(client_info, auth_code)
        assert token.refresh_token is None

    @pytest.mark.asyncio
    async def test_exchange_expired_jwt_gives_zero_expires_in(
        self, provider, client_info
    ):
        """If the JWT is already expired, expires_in should be 0."""
        auth_code = EveryRowAuthorizationCode(
            code="test-code-3",
            client_id="test-client-id",
            redirect_uri=AnyUrl("https://claude.ai/callback"),
            redirect_uri_provided_explicitly=True,
            code_challenge="test-challenge",
            scopes=["read"],
            expires_at=time.time() + 300,
            supabase_jwt=_make_unsigned_jwt(exp=int(time.time()) - 100),
            supabase_refresh_token="",
        )

        token = await provider.exchange_authorization_code(client_info, auth_code)
        assert token.expires_in == 0


class TestRefreshToken:
    @pytest.mark.asyncio
    async def test_load_refresh_token(self, provider, client_info):
        rt = EveryRowRefreshToken(
            token="rt-abc",
            client_id="test-client-id",
            scopes=["read"],
            supabase_refresh_token="supabase-rt",
        )
        await provider._redis_set(
            build_key("refresh", "rt-abc"), rt, ttl=REFRESH_TOKEN_TTL
        )

        loaded = await provider.load_refresh_token(client_info, "rt-abc")
        assert loaded is not None
        assert loaded.token == "rt-abc"
        assert loaded.supabase_refresh_token == "supabase-rt"

    @pytest.mark.asyncio
    async def test_load_refresh_token_wrong_client(self, provider, client_info):
        rt = EveryRowRefreshToken(
            token="rt-abc",
            client_id="other-client",
            scopes=["read"],
            supabase_refresh_token="supabase-rt",
        )
        await provider._redis_set(
            build_key("refresh", "rt-abc"), rt, ttl=REFRESH_TOKEN_TTL
        )

        assert await provider.load_refresh_token(client_info, "rt-abc") is None

    @pytest.mark.asyncio
    async def test_load_refresh_token_not_found(self, provider, client_info):
        assert await provider.load_refresh_token(client_info, "nonexistent") is None

    @pytest.mark.asyncio
    async def test_exchange_refresh_token(self, provider, client_info, fake_redis):
        new_jwt = _make_unsigned_jwt()
        refresh_resp = MagicMock(spec=httpx.Response)
        refresh_resp.status_code = 200
        refresh_resp.raise_for_status = MagicMock(return_value=None)
        refresh_resp.json = MagicMock(
            return_value={
                "access_token": new_jwt,
                "refresh_token": "new-supabase-rt",
            }
        )

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=refresh_resp)
        provider._http = mock_http

        old_rt = EveryRowRefreshToken(
            token="old-rt",
            client_id="test-client-id",
            scopes=["read"],
            supabase_refresh_token="old-supabase-rt",
        )
        await provider._redis_set(
            build_key("refresh", "old-rt"), old_rt, ttl=REFRESH_TOKEN_TTL
        )

        # load_refresh_token now atomically consumes the token (GETDEL)
        loaded_rt = await provider.load_refresh_token(client_info, "old-rt")
        assert loaded_rt is not None

        token = await provider.exchange_refresh_token(
            client_info, loaded_rt, scopes=["read"]
        )

        # New JWT returned
        assert token.access_token == new_jwt
        assert token.token_type == "Bearer"
        assert token.expires_in is not None
        assert token.expires_in > 0
        # New refresh token issued (rotation)
        assert token.refresh_token is not None
        assert token.refresh_token != "old-rt"

        # Old token deleted
        assert await fake_redis.get(build_key("refresh", "old-rt")) is None

        # New token stored in Redis
        new_rt_data = await fake_redis.get(build_key("refresh", token.refresh_token))
        assert new_rt_data is not None

        # Verify Supabase was called correctly
        mock_http.post.assert_called_once()
        call_url = mock_http.post.call_args[0][0]
        assert "grant_type=refresh_token" in call_url

    @pytest.mark.asyncio
    async def test_revoke_refresh_token(self, provider, fake_redis):
        rt = EveryRowRefreshToken(
            token="revoke-me",
            client_id="test-client-id",
            scopes=[],
            supabase_refresh_token="supabase-rt",
        )
        await provider._redis_set(
            build_key("refresh", "revoke-me"), rt, ttl=REFRESH_TOKEN_TTL
        )

        await provider.revoke_token(rt)

        assert await fake_redis.get(build_key("refresh", "revoke-me")) is None


class TestStubs:
    @pytest.mark.asyncio
    async def test_load_access_token_returns_none(self, provider):
        """load_access_token is unused (verifier handles it) — always returns None."""
        assert await provider.load_access_token("any-token") is None


class TestHttpxReuse:
    def test_provider_creates_http_client(self, provider):
        """Provider __init__ creates a reusable httpx.AsyncClient."""
        assert isinstance(provider._http, httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_exchange_uses_shared_client(
        self, provider, client_info, auth_params
    ):
        """_exchange_supabase_code uses self._http, not a new client."""
        await provider.register_client(client_info)
        redirect_url = await provider.authorize(client_info, auth_params)
        state = urlparse(redirect_url).path.split("/auth/start/")[1]

        fake_jwt = _make_unsigned_jwt()
        token_resp = _make_supabase_token_response(jwt_token=fake_jwt)

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=token_resp)
        provider._http = mock_http

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/auth/callback",
            "query_string": b"code=supabase-code",
            "headers": [(b"cookie", f"mcp_auth_state={state}".encode())],
        }
        await provider.handle_callback(Request(scope))

        # Verify the shared client was used
        mock_http.post.assert_called_once()
