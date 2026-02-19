"""Tests for the OAuth authorization provider."""

import time
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import fakeredis.aioredis
import httpx
import pytest
from cryptography.fernet import Fernet
from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl
from starlette.requests import Request

from everyrow_mcp.auth import (
    PENDING_AUTH_TTL,
    EveryRowAccessToken,
    EveryRowAuthorizationCode,
    EveryRowAuthProvider,
    EveryRowRefreshToken,
    PendingAuth,
    _jwt_hash,
)
from everyrow_mcp.redis_utils import build_key

TEST_ENCRYPTION_KEY = Fernet.generate_key().decode()


@pytest.fixture
def fake_redis():
    """Create a fakeredis async client."""
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def provider(fake_redis) -> EveryRowAuthProvider:
    """Create a test auth provider backed by fakeredis with encryption."""
    return EveryRowAuthProvider(
        supabase_url="https://test.supabase.co",
        supabase_anon_key="test-anon-key",
        mcp_server_url="https://mcp.example.com",
        redis=fake_redis,
        encryption_key=TEST_ENCRYPTION_KEY,
    )


@pytest.fixture
def client_info() -> OAuthClientInformationFull:
    """Create a test OAuth client."""
    return OAuthClientInformationFull(
        client_id="test-client-id",
        client_secret="test-client-secret",
        redirect_uris=[AnyUrl("https://claude.ai/callback")],
        client_name="Test Client",
    )


@pytest.fixture
def auth_params() -> AuthorizationParams:
    """Create test authorization parameters."""
    return AuthorizationParams(
        state="client-state-abc",
        scopes=["read"],
        code_challenge="test-code-challenge",
        redirect_uri=AnyUrl("https://claude.ai/callback"),
        redirect_uri_provided_explicitly=True,
    )


class TestClientRegistration:
    """Tests for OAuth client registration."""

    @pytest.mark.asyncio
    async def test_register_and_get_client(
        self, provider: EveryRowAuthProvider, client_info: OAuthClientInformationFull
    ):
        """Test registering a client and retrieving it."""
        await provider.register_client(client_info)
        result = await provider.get_client("test-client-id")
        assert result is not None
        assert result.client_id == "test-client-id"
        assert result.client_name == "Test Client"

    @pytest.mark.asyncio
    async def test_get_nonexistent_client(self, provider: EveryRowAuthProvider):
        """Test that getting a nonexistent client returns None."""
        result = await provider.get_client("nonexistent")
        assert result is None


class TestAuthorize:
    """Tests for the authorize flow."""

    @pytest.mark.asyncio
    async def test_authorize_returns_start_redirect(
        self,
        provider: EveryRowAuthProvider,
        client_info: OAuthClientInformationFull,
        auth_params: AuthorizationParams,
    ):
        """Test that authorize returns a redirect to our /auth/start/{state} route."""
        await provider.register_client(client_info)
        redirect_url = await provider.authorize(client_info, auth_params)

        # authorize() returns our intermediate /auth/start/{state} URL
        parsed = urlparse(redirect_url)
        assert parsed.path.startswith("/auth/start/")
        state = parsed.path.split("/auth/start/")[1]
        assert len(state) > 0

        # The pending auth should exist in Redis (read via provider to decrypt)
        pending = await provider._redis_get(build_key("pending", state), PendingAuth)
        assert pending is not None

        # The Supabase URL is stored in the pending auth for handle_start to use
        supabase_url = urlparse(pending.supabase_redirect_url)
        assert supabase_url.hostname == "test.supabase.co"
        assert supabase_url.path == "/auth/v1/authorize"

        params = parse_qs(supabase_url.query)
        assert params["provider"] == ["google"]
        assert params["flow_type"] == ["pkce"]
        assert "code_challenge" in params
        assert params["code_challenge_method"] == ["s256"]
        # redirect_to points to our callback (clean, no state in URL)
        assert params["redirect_to"] == ["https://mcp.example.com/auth/callback"]

    @pytest.mark.asyncio
    async def test_authorize_stores_pending_auth(
        self,
        provider: EveryRowAuthProvider,
        client_info: OAuthClientInformationFull,
        auth_params: AuthorizationParams,
    ):
        """Test that authorize stores a pending auth entry."""
        await provider.register_client(client_info)
        redirect_url = await provider.authorize(client_info, auth_params)

        # Extract state from the redirect URL
        state = urlparse(redirect_url).path.split("/auth/start/")[1]
        pending = await provider._redis_get(build_key("pending", state), PendingAuth)
        assert pending is not None
        assert pending.client_id == "test-client-id"
        assert pending.params.state == "client-state-abc"
        assert pending.supabase_code_verifier != ""


class TestCallback:
    """Tests for the OAuth callback handler."""

    def _make_mock_response(self, json_data=None):
        """Create a mock httpx.Response (synchronous methods)."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.raise_for_status = MagicMock(return_value=None)
        if json_data is not None:
            resp.json = MagicMock(return_value=json_data)
        return resp

    @pytest.mark.asyncio
    async def test_callback_happy_path(
        self,
        provider: EveryRowAuthProvider,
        client_info: OAuthClientInformationFull,
        auth_params: AuthorizationParams,
    ):
        """Test the full callback flow: Supabase code -> JWT -> auth code -> redirect."""
        await provider.register_client(client_info)
        redirect_url = await provider.authorize(client_info, auth_params)

        # Get the state from the redirect URL
        state = urlparse(redirect_url).path.split("/auth/start/")[1]

        # Mock httpx.AsyncClient
        token_resp = self._make_mock_response(
            json_data={
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
                "headers": [
                    (b"cookie", f"mcp_auth_state={state}".encode()),
                ],
            }
            request = Request(scope)
            response = await provider.handle_callback(request)

        assert response.status_code == 302
        location = response.headers["location"]
        parsed = urlparse(location)
        params = parse_qs(parsed.query)

        assert "code" in params
        assert params["state"] == ["client-state-abc"]

        # Verify auth code was stored in Redis with Supabase JWT (read via provider)
        auth_code = params["code"][0]
        stored_code = await provider._redis_get(
            build_key("authcode", auth_code), EveryRowAuthorizationCode
        )
        assert stored_code is not None
        assert stored_code.client_id == "test-client-id"
        assert stored_code.supabase_jwt == "supabase-jwt-token"
        assert stored_code.supabase_refresh_token == "supabase-refresh-token"

        # Only one POST call (token exchange)
        mock_client.post.assert_called_once()
        mock_client.patch.assert_not_called()

    @pytest.mark.asyncio
    async def test_callback_missing_code(self, provider: EveryRowAuthProvider):
        """Test callback with missing code parameter."""
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/auth/callback",
            "query_string": b"",
            "headers": [
                (b"cookie", b"mcp_auth_state=some-state"),
            ],
        }
        request = Request(scope)
        response = await provider.handle_callback(request)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_callback_missing_state_cookie(self, provider: EveryRowAuthProvider):
        """Test callback with no state cookie."""
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/auth/callback",
            "query_string": b"code=some-code",
            "headers": [],
        }
        request = Request(scope)
        response = await provider.handle_callback(request)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_callback_invalid_state(self, provider: EveryRowAuthProvider):
        """Test callback with invalid state cookie."""
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/auth/callback",
            "query_string": b"code=some-code",
            "headers": [
                (b"cookie", b"mcp_auth_state=invalid-state"),
            ],
        }
        request = Request(scope)
        response = await provider.handle_callback(request)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_callback_expired_state(
        self,
        provider: EveryRowAuthProvider,
        client_info: OAuthClientInformationFull,
        auth_params: AuthorizationParams,
    ):
        """Test callback with expired pending auth."""
        await provider.register_client(client_info)
        redirect_url = await provider.authorize(client_info, auth_params)

        # Get state and manipulate created_at to make it expired
        state = urlparse(redirect_url).path.split("/auth/start/")[1]
        pending = await provider._redis_get(build_key("pending", state), PendingAuth)
        assert pending is not None
        pending.created_at = time.time() - PENDING_AUTH_TTL - 1
        await provider._redis_set(build_key("pending", state), pending)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/auth/callback",
            "query_string": b"code=supabase-code",
            "headers": [
                (b"cookie", f"mcp_auth_state={state}".encode()),
            ],
        }
        request = Request(scope)
        response = await provider.handle_callback(request)
        assert response.status_code == 400


class TestAuthorizationCodeExchange:
    """Tests for authorization code loading and exchange."""

    @pytest.mark.asyncio
    async def test_load_authorization_code(
        self,
        provider: EveryRowAuthProvider,
        client_info: OAuthClientInformationFull,
    ):
        """Test loading a valid authorization code."""
        await provider.register_client(client_info)

        # Insert an auth code via provider (encrypted)
        auth_code_obj = EveryRowAuthorizationCode(
            code="test-code",
            client_id="test-client-id",
            redirect_uri=AnyUrl("https://claude.ai/callback"),
            redirect_uri_provided_explicitly=True,
            code_challenge="test-challenge",
            scopes=["read"],
            expires_at=time.time() + 300,
            supabase_jwt="test-jwt",
            supabase_refresh_token="test-refresh",
        )
        await provider._redis_set(
            build_key("authcode", "test-code"), auth_code_obj, ttl=300
        )

        result = await provider.load_authorization_code(client_info, "test-code")
        assert result is not None
        assert result.code == "test-code"
        assert result.supabase_jwt == "test-jwt"

    @pytest.mark.asyncio
    async def test_load_expired_authorization_code(
        self,
        provider: EveryRowAuthProvider,
        client_info: OAuthClientInformationFull,
    ):
        """Test that expired auth codes return None (Redis TTL handles expiry)."""
        await provider.register_client(client_info)

        # Don't store anything — simulates TTL expiry
        result = await provider.load_authorization_code(client_info, "expired-code")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_wrong_client_authorization_code(
        self,
        provider: EveryRowAuthProvider,
        client_info: OAuthClientInformationFull,
    ):
        """Test that auth codes for wrong client return None."""
        auth_code_obj = EveryRowAuthorizationCode(
            code="other-code",
            client_id="other-client-id",
            redirect_uri=AnyUrl("https://claude.ai/callback"),
            redirect_uri_provided_explicitly=True,
            code_challenge="test-challenge",
            scopes=["read"],
            expires_at=time.time() + 300,
            supabase_jwt="test-jwt",
            supabase_refresh_token="test-refresh",
        )
        await provider._redis_set(
            build_key("authcode", "other-code"), auth_code_obj, ttl=300
        )

        result = await provider.load_authorization_code(client_info, "other-code")
        assert result is None

    @pytest.mark.asyncio
    async def test_exchange_authorization_code(
        self,
        provider: EveryRowAuthProvider,
        client_info: OAuthClientInformationFull,
        fake_redis,
    ):
        """Test exchanging an auth code for tokens."""
        await provider.register_client(client_info)

        auth_code = EveryRowAuthorizationCode(
            code="exchange-code",
            client_id="test-client-id",
            redirect_uri=AnyUrl("https://claude.ai/callback"),
            redirect_uri_provided_explicitly=True,
            code_challenge="test-challenge",
            scopes=["read"],
            expires_at=time.time() + 300,
            supabase_jwt="exchange-jwt",
            supabase_refresh_token="exchange-refresh",
        )
        await provider._redis_set(
            build_key("authcode", "exchange-code"), auth_code, ttl=300
        )

        token = await provider.exchange_authorization_code(client_info, auth_code)

        assert token.access_token is not None
        assert token.refresh_token is not None
        assert token.token_type == "Bearer"

        # Auth code should be consumed
        assert await fake_redis.get(build_key("authcode", "exchange-code")) is None

        # Access token should have Supabase JWT embedded (read via provider to decrypt)
        stored_access = await provider._redis_get(
            build_key("access", token.access_token), EveryRowAccessToken
        )
        assert stored_access is not None
        assert stored_access.supabase_jwt == "exchange-jwt"

        # Refresh token should have Supabase JWT and refresh token embedded
        stored_refresh = await provider._redis_get(
            build_key("refresh", token.refresh_token), EveryRowRefreshToken
        )
        assert stored_refresh is not None
        assert stored_refresh.supabase_jwt == "exchange-jwt"
        assert stored_refresh.supabase_refresh_token == "exchange-refresh"


class TestAccessToken:
    """Tests for access token verification."""

    @pytest.mark.asyncio
    async def test_load_valid_access_token(self, provider: EveryRowAuthProvider):
        """Test loading a valid access token."""
        token_obj = EveryRowAccessToken(
            token="valid-token",
            client_id="test-client",
            scopes=["read"],
            expires_at=int(time.time()) + 3600,
            supabase_jwt="valid-jwt",
        )
        await provider._redis_set(
            build_key("access", "valid-token"), token_obj, ttl=3600
        )

        result = await provider.load_access_token("valid-token")
        assert result is not None
        assert result.supabase_jwt == "valid-jwt"
        assert result.client_id == "test-client"

    @pytest.mark.asyncio
    async def test_load_expired_access_token(self, provider: EveryRowAuthProvider):
        """Test that expired access tokens return None (Redis TTL handles expiry)."""
        # Not stored — simulates TTL expiry
        result = await provider.load_access_token("expired-token")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_nonexistent_access_token(self, provider: EveryRowAuthProvider):
        """Test that nonexistent tokens return None."""
        result = await provider.load_access_token("nonexistent")
        assert result is None


class TestRefreshToken:
    """Tests for token refresh."""

    @pytest.mark.asyncio
    async def test_load_valid_refresh_token(
        self,
        provider: EveryRowAuthProvider,
        client_info: OAuthClientInformationFull,
    ):
        """Test loading a valid refresh token."""
        await provider.register_client(client_info)

        token_obj = EveryRowRefreshToken(
            token="valid-refresh",
            client_id="test-client-id",
            scopes=["read"],
            expires_at=int(time.time()) + 86400,
            supabase_jwt="refresh-jwt",
            supabase_refresh_token="refresh-token",
        )
        await provider._redis_set(
            build_key("refresh", "valid-refresh"), token_obj, ttl=86400
        )

        result = await provider.load_refresh_token(client_info, "valid-refresh")
        assert result is not None
        assert result.supabase_jwt == "refresh-jwt"

    @pytest.mark.asyncio
    async def test_load_expired_refresh_token(
        self,
        provider: EveryRowAuthProvider,
        client_info: OAuthClientInformationFull,
    ):
        """Test that expired refresh tokens return None (Redis TTL handles expiry)."""
        await provider.register_client(client_info)

        # Not stored — simulates TTL expiry
        result = await provider.load_refresh_token(client_info, "expired-refresh")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_wrong_client_refresh_token(
        self,
        provider: EveryRowAuthProvider,
        client_info: OAuthClientInformationFull,
    ):
        """Test that refresh tokens for wrong client return None."""
        await provider.register_client(client_info)

        token_obj = EveryRowRefreshToken(
            token="other-refresh",
            client_id="other-client-id",
            scopes=["read"],
            expires_at=int(time.time()) + 86400,
            supabase_jwt="other-jwt",
            supabase_refresh_token="other-refresh-token",
        )
        await provider._redis_set(
            build_key("refresh", "other-refresh"), token_obj, ttl=86400
        )

        result = await provider.load_refresh_token(client_info, "other-refresh")
        assert result is None

    @pytest.mark.asyncio
    async def test_exchange_refresh_token(
        self,
        provider: EveryRowAuthProvider,
        client_info: OAuthClientInformationFull,
        fake_redis,
    ):
        """Test exchanging a refresh token rotates tokens and refreshes JWT."""
        await provider.register_client(client_info)

        # Set up existing tokens (via provider so they're encrypted)
        old_refresh = EveryRowRefreshToken(
            token="old-refresh",
            client_id="test-client-id",
            scopes=["read"],
            expires_at=int(time.time()) + 86400,
            supabase_jwt="old-jwt",
            supabase_refresh_token="old-supabase-refresh",
        )
        await provider._redis_set(
            build_key("refresh", "old-refresh"), old_refresh, ttl=86400
        )

        old_access = EveryRowAccessToken(
            token="old-access",
            client_id="test-client-id",
            scopes=["read"],
            expires_at=int(time.time()) + 3600,
            supabase_jwt="old-jwt",
        )
        await provider._redis_set(
            build_key("access", "old-access"), old_access, ttl=3600
        )

        # Set up secondary index for old access token
        idx_key = build_key(
            "idx", "access_by_cj", "test-client-id", _jwt_hash("old-jwt")
        )
        await fake_redis.sadd(idx_key, "old-access")

        # Mock the Supabase refresh call
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock(return_value=None)
        mock_resp.json = MagicMock(
            return_value={
                "access_token": "new-supabase-jwt",
                "refresh_token": "new-supabase-refresh",
            }
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("everyrow_mcp.auth.httpx.AsyncClient", return_value=mock_client):
            new_token = await provider.exchange_refresh_token(
                client_info, old_refresh, scopes=["read"]
            )

        # Old tokens should be revoked
        assert await fake_redis.get(build_key("refresh", "old-refresh")) is None
        assert await fake_redis.get(build_key("access", "old-access")) is None

        # New tokens should exist with refreshed JWT (read via provider to decrypt)
        new_access = await provider._redis_get(
            build_key("access", new_token.access_token), EveryRowAccessToken
        )
        assert new_access is not None
        assert new_access.supabase_jwt == "new-supabase-jwt"

        new_refresh = await provider._redis_get(
            build_key("refresh", new_token.refresh_token), EveryRowRefreshToken
        )
        assert new_refresh is not None
        assert new_refresh.supabase_jwt == "new-supabase-jwt"
        assert new_refresh.supabase_refresh_token == "new-supabase-refresh"

    @pytest.mark.asyncio
    async def test_exchange_refresh_token_supabase_failure_propagates(
        self,
        provider: EveryRowAuthProvider,
        client_info: OAuthClientInformationFull,
    ):
        """Test that Supabase refresh failure propagates as an exception."""
        await provider.register_client(client_info)

        old_refresh = EveryRowRefreshToken(
            token="old-refresh",
            client_id="test-client-id",
            scopes=["read"],
            expires_at=int(time.time()) + 86400,
            supabase_jwt="old-jwt",
            supabase_refresh_token="revoked-supabase-refresh",
        )
        await provider._redis_set(
            build_key("refresh", "old-refresh"), old_refresh, ttl=86400
        )

        # Mock Supabase returning an error (e.g. revoked refresh token)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 400
        mock_resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Bad Request", request=MagicMock(), response=mock_resp
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("everyrow_mcp.auth.httpx.AsyncClient", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await provider.exchange_refresh_token(
                client_info, old_refresh, scopes=["read"]
            )


class TestEncryption:
    """Tests for at-rest encryption in Redis."""

    @pytest.mark.asyncio
    async def test_data_encrypted_at_rest(
        self, provider: EveryRowAuthProvider, fake_redis
    ):
        """Test that sensitive data stored in Redis is encrypted, not plain JSON."""
        token = EveryRowAccessToken(
            token="enc-test",
            client_id="test-client",
            scopes=["read"],
            expires_at=int(time.time()) + 3600,
            supabase_jwt="secret-jwt-value",
        )
        await provider._redis_set(build_key("access", "enc-test"), token, ttl=3600)

        # Read raw bytes from Redis — should NOT contain the plaintext JWT
        raw = await fake_redis.get(build_key("access", "enc-test"))
        assert raw is not None
        assert "secret-jwt-value" not in raw

        # But reading through the provider should decrypt successfully
        loaded = await provider._redis_get(
            build_key("access", "enc-test"), EveryRowAccessToken
        )
        assert loaded is not None
        assert loaded.supabase_jwt == "secret-jwt-value"


class TestTokenRevocation:
    """Tests for token revocation."""

    @pytest.mark.asyncio
    async def test_revoke_access_token(
        self, provider: EveryRowAuthProvider, fake_redis
    ):
        """Test revoking an access token."""
        token = EveryRowAccessToken(
            token="revoke-me",
            client_id="test-client",
            scopes=["read"],
            supabase_jwt="revoke-jwt",
        )
        await provider._redis_set(build_key("access", "revoke-me"), token)

        await provider.revoke_token(token)
        assert await fake_redis.get(build_key("access", "revoke-me")) is None

    @pytest.mark.asyncio
    async def test_revoke_refresh_token(
        self, provider: EveryRowAuthProvider, fake_redis
    ):
        """Test revoking a refresh token."""
        token = EveryRowRefreshToken(
            token="revoke-refresh",
            client_id="test-client",
            scopes=["read"],
            supabase_jwt="revoke-jwt",
            supabase_refresh_token="revoke-refresh-token",
        )
        await provider._redis_set(build_key("refresh", "revoke-refresh"), token)

        await provider.revoke_token(token)
        assert await fake_redis.get(build_key("refresh", "revoke-refresh")) is None

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_token(self, provider: EveryRowAuthProvider):
        """Test that revoking a nonexistent token is a no-op."""
        token = EveryRowAccessToken(
            token="nonexistent",
            client_id="test-client",
            scopes=["read"],
            supabase_jwt="none-jwt",
        )
        # Should not raise
        await provider.revoke_token(token)
