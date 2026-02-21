"""Tests for Supabase JWT verification and EveryRowAuthProvider."""

import asyncio
import secrets
import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)
from mcp.shared.auth import OAuthClientInformationFull

from everyrow_mcp.auth import (
    AUTH_CODE_TTL,
    EveryRowAuthorizationCode,
    EveryRowAuthProvider,
    EveryRowRefreshToken,
    SupabaseTokenVerifier,
)

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
def mock_redis():
    """In-memory dict-backed async Redis mock."""
    store: dict[str, str] = {}

    redis = AsyncMock()

    async def _setex(key, _ttl, value):
        store[key] = value

    async def _exists(key):
        return 1 if key in store else 0

    async def _delete(key):
        store.pop(key, None)

    redis.setex = AsyncMock(side_effect=_setex)
    redis.exists = AsyncMock(side_effect=_exists)
    redis.delete = AsyncMock(side_effect=_delete)
    redis._store = store  # exposed for assertions
    return redis


@pytest.fixture
def verifier(rsa_keypair, mock_redis):
    """Create a SupabaseTokenVerifier with a mocked JWKS client and Redis."""
    _private_key, public_key = rsa_keypair
    verifier = SupabaseTokenVerifier(SUPABASE_URL, redis=mock_redis)

    mock_signing_key = MagicMock()
    mock_signing_key.key = public_key.public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )
    verifier._jwks_client = MagicMock()
    verifier._jwks_client.get_signing_key_from_jwt = MagicMock(
        return_value=mock_signing_key
    )

    return verifier


def _make_jwt(
    private_key,
    claims: dict | None = None,
    *,
    remove_claims: list[str] | None = None,
) -> str:
    """Create a signed JWT with default claims, optionally overriding/removing."""
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
    if remove_claims:
        for key in remove_claims:
            payload.pop(key, None)
    return jwt.encode(payload, private_key, algorithm="RS256")


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
                lifespan=300,
                max_cached_keys=16,
            )

    @pytest.mark.asyncio
    async def test_trailing_slash_normalized(self):
        with patch("everyrow_mcp.auth.PyJWKClient") as mock_jwk_cls:
            v = SupabaseTokenVerifier("https://my-project.supabase.co/")
            mock_jwk_cls.assert_called_once_with(
                "https://my-project.supabase.co/auth/v1/.well-known/jwks.json",
                cache_keys=True,
                lifespan=300,
                max_cached_keys=16,
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


# ── Token deny-list tests ──────────────────────────────────────────


class TestTokenDenyList:
    @pytest.mark.asyncio
    async def test_deny_adds_to_denylist(self, verifier, mock_redis):
        """deny_token stores a fingerprint key in Redis."""
        token = "some-token"
        result = await verifier.deny_token(token)

        assert result is True
        assert len(mock_redis._store) == 1
        key = next(iter(mock_redis._store))
        assert key.startswith("mcp:revoked:")

    @pytest.mark.asyncio
    async def test_denied_token_rejected(self, verifier, rsa_keypair):
        """A denied token is rejected by verify_token."""
        private_key, _ = rsa_keypair
        token = _make_jwt(private_key)

        await verifier.deny_token(token)
        result = await verifier.verify_token(token)

        assert result is None

    @pytest.mark.asyncio
    async def test_non_denied_token_passes(self, verifier, rsa_keypair):
        """A token that has not been denied passes verification."""
        private_key, _ = rsa_keypair
        token = _make_jwt(private_key)

        result = await verifier.verify_token(token)

        assert result is not None
        assert result.client_id == "user-123"

    @pytest.mark.asyncio
    async def test_deny_without_redis_returns_false(self, rsa_keypair):
        """deny_token returns False when no Redis is configured."""
        _, _public_key = rsa_keypair
        verifier = SupabaseTokenVerifier(SUPABASE_URL)  # no redis
        result = await verifier.deny_token("some-token")
        assert result is False

    @pytest.mark.asyncio
    async def test_denylist_check_fails_open(self, verifier, rsa_keypair, mock_redis):
        """If Redis raises during _is_denied, the token is NOT rejected."""
        private_key, _ = rsa_keypair
        token = _make_jwt(private_key)

        mock_redis.exists = AsyncMock(side_effect=ConnectionError("Redis down"))

        result = await verifier.verify_token(token)
        assert result is not None
        assert result.client_id == "user-123"


# ── Required claims tests ───────────────────────────────────────────


class TestRequiredClaims:
    @pytest.mark.asyncio
    async def test_missing_exp_rejected(self, verifier, rsa_keypair):
        private_key, _ = rsa_keypair
        token = _make_jwt(private_key, remove_claims=["exp"])
        assert await verifier.verify_token(token) is None

    @pytest.mark.asyncio
    async def test_missing_sub_rejected(self, verifier, rsa_keypair):
        private_key, _ = rsa_keypair
        token = _make_jwt(private_key, remove_claims=["sub"])
        assert await verifier.verify_token(token) is None

    @pytest.mark.asyncio
    async def test_missing_aud_rejected(self, verifier, rsa_keypair):
        private_key, _ = rsa_keypair
        token = _make_jwt(private_key, remove_claims=["aud"])
        assert await verifier.verify_token(token) is None


# ── JWKS lock concurrency test ──────────────────────────────────────


class TestJwksLock:
    @pytest.mark.asyncio
    async def test_concurrent_verify_serialized_by_lock(self, verifier, rsa_keypair):
        """Multiple concurrent verify_token calls should serialize JWKS fetches."""
        private_key, _ = rsa_keypair
        token = _make_jwt(private_key)

        call_order: list[str] = []
        original_get_key = verifier._jwks_client.get_signing_key_from_jwt

        def tracked_get_key(t):
            call_order.append("start")
            result = original_get_key(t)
            call_order.append("end")
            return result

        verifier._jwks_client.get_signing_key_from_jwt = tracked_get_key

        results = await asyncio.gather(
            verifier.verify_token(token),
            verifier.verify_token(token),
            verifier.verify_token(token),
        )

        # All should succeed
        assert all(r is not None for r in results)

        # Calls should be serialized: start/end pairs should not interleave
        for i in range(0, len(call_order), 2):
            assert call_order[i] == "start"
            assert call_order[i + 1] == "end"


# ── Auth provider tests ─────────────────────────────────────────────


@pytest.fixture
def provider_redis():
    """In-memory dict-backed async Redis mock with get/set/getdel/delete."""
    store: dict[str, str] = {}

    redis = AsyncMock()

    async def _set(key, value):
        store[key] = value

    async def _setex(key, _ttl, value):
        store[key] = value

    async def _get(key):
        return store.get(key)

    async def _getdel(key):
        return store.pop(key, None)

    async def _delete(key):
        store.pop(key, None)

    async def _incr(key):
        store[key] = str(int(store.get(key, "0")) + 1)
        return int(store[key])

    async def _expire(key, _ttl):
        pass

    redis.set = AsyncMock(side_effect=_set)
    redis.setex = AsyncMock(side_effect=_setex)
    redis.get = AsyncMock(side_effect=_get)
    redis.getdel = AsyncMock(side_effect=_getdel)
    redis.delete = AsyncMock(side_effect=_delete)
    redis.incr = AsyncMock(side_effect=_incr)
    redis.expire = AsyncMock(side_effect=_expire)
    redis._store = store

    # Pipeline mock for register_client rate limiting
    pipe_mock = MagicMock()
    pipe_mock.incr = MagicMock()
    pipe_mock.expire = MagicMock()
    pipe_mock.execute = AsyncMock(return_value=[1, True])
    redis.pipeline = MagicMock(return_value=pipe_mock)

    return redis


@pytest.fixture
def provider(provider_redis):
    """Create an EveryRowAuthProvider with mocked Redis."""
    return EveryRowAuthProvider(
        supabase_url=SUPABASE_URL,
        supabase_anon_key="test-anon-key",
        mcp_server_url="https://mcp.example.com",
        redis=provider_redis,
    )


@pytest.fixture
def test_client():
    """A minimal OAuthClientInformationFull for tests."""
    return OAuthClientInformationFull(
        client_id="test-client-id",
        redirect_uris=["https://example.com/callback"],
    )


class TestAuthProvider:
    @pytest.mark.asyncio
    async def test_auth_code_consumed_atomically(
        self, provider, provider_redis, test_client
    ):
        """Loading an auth code via _redis_getdel deletes it; second load returns None."""
        # Store an auth code directly in Redis
        auth_code_str = secrets.token_urlsafe(32)
        auth_code_obj = EveryRowAuthorizationCode(
            code=auth_code_str,
            client_id="test-client-id",
            redirect_uri="https://example.com/callback",
            redirect_uri_provided_explicitly=True,
            code_challenge="test-challenge",
            scopes=["read"],
            expires_at=time.time() + AUTH_CODE_TTL,
            supabase_access_token="fake-supabase-jwt",
            supabase_refresh_token="fake-refresh",
        )
        await provider._redis_set(
            f"mcp:authcode:{auth_code_str}", auth_code_obj, ttl=AUTH_CODE_TTL
        )

        # First load should succeed and consume the code
        result1 = await provider.load_authorization_code(test_client, auth_code_str)
        assert result1 is not None
        assert result1.code == auth_code_str

        # Second load should return None (code was atomically deleted)
        result2 = await provider.load_authorization_code(test_client, auth_code_str)
        assert result2 is None

    @pytest.mark.asyncio
    async def test_refresh_scope_narrowing(self, provider, test_client):
        """Refresh with broader scopes only gets the intersection."""
        refresh_token = EveryRowRefreshToken(
            token="rt-123",
            client_id="test-client-id",
            scopes=["read", "write"],
            supabase_refresh_token="supa-rt",
        )

        # Mock the Supabase refresh call
        fake_jwt = jwt.encode(
            {"sub": "user-1", "exp": int(time.time()) + 3600},
            "secret",
            algorithm="HS256",
        )
        with patch.object(
            provider,
            "_refresh_supabase_token",
            new_callable=AsyncMock,
            return_value=(fake_jwt, "new-supa-rt"),
        ):
            result = await provider.exchange_refresh_token(
                test_client, refresh_token, scopes=["read", "write", "admin"]
            )

        assert result.access_token == fake_jwt
        # Should only get the intersection: ["read", "write"] & ["read", "write", "admin"]
        # Load the new refresh token from Redis to check scopes
        new_rt_str = result.refresh_token
        assert new_rt_str is not None
        new_rt = await provider._redis_get(
            f"mcp:refresh:{new_rt_str}", EveryRowRefreshToken
        )
        assert new_rt is not None
        assert set(new_rt.scopes) == {"read", "write"}

    @pytest.mark.asyncio
    async def test_refresh_scope_preserved_when_empty(self, provider, test_client):
        """Empty scopes list preserves original scopes from the refresh token."""
        refresh_token = EveryRowRefreshToken(
            token="rt-456",
            client_id="test-client-id",
            scopes=["read", "write"],
            supabase_refresh_token="supa-rt",
        )

        fake_jwt = jwt.encode(
            {"sub": "user-1", "exp": int(time.time()) + 3600},
            "secret",
            algorithm="HS256",
        )
        with patch.object(
            provider,
            "_refresh_supabase_token",
            new_callable=AsyncMock,
            return_value=(fake_jwt, "new-supa-rt"),
        ):
            result = await provider.exchange_refresh_token(
                test_client, refresh_token, scopes=[]
            )

        new_rt_str = result.refresh_token
        assert new_rt_str is not None
        new_rt = await provider._redis_get(
            f"mcp:refresh:{new_rt_str}", EveryRowRefreshToken
        )
        assert new_rt is not None
        assert set(new_rt.scopes) == {"read", "write"}
