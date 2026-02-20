"""Tests for Supabase JWT verification."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)

from everyrow_mcp.auth import SupabaseTokenVerifier

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
