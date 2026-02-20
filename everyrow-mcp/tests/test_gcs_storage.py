"""Tests for the GCS storage module."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from everyrow_mcp.gcs_storage import GCSResultStore


@pytest.fixture
def mock_gcs():
    """Mock google.cloud.storage Client and Bucket."""
    with patch("everyrow_mcp.gcs_storage.storage.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_client_cls.return_value = mock_client

        # Cache blobs by path so the same mock is returned for the same path
        blobs: dict[str, MagicMock] = {}

        def make_blob(path):
            if path not in blobs:
                blob = MagicMock()
                blob.name = path
                blob.generate_signed_url.return_value = (
                    f"https://storage.googleapis.com/signed/{path}"
                )
                blobs[path] = blob
            return blobs[path]

        mock_bucket.blob.side_effect = make_blob

        yield {
            "client_cls": mock_client_cls,
            "client": mock_client,
            "bucket": mock_bucket,
            "blobs": blobs,
        }


@pytest.fixture
def store(mock_gcs) -> GCSResultStore:  # noqa: ARG001
    """Create a GCSResultStore backed by mocked GCS."""
    return GCSResultStore("test-bucket")


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({"name": ["Alice", "Bob"], "score": [95, 87]})


class TestGCSResultStore:
    """Tests for GCSResultStore."""

    def test_init_creates_client_and_bucket(self, mock_gcs):
        """Test that init creates a GCS client and gets the bucket."""
        GCSResultStore("my-bucket")
        mock_gcs["client_cls"].assert_called_once()
        mock_gcs["client"].bucket.assert_called_once_with("my-bucket")

    def test_upload_result_uploads_csv(
        self, store: GCSResultStore, mock_gcs, sample_df: pd.DataFrame
    ):
        """Test that upload_result uploads a CSV blob."""
        store.upload_result("task-abc-123", sample_df)

        calls = mock_gcs["bucket"].blob.call_args_list
        blob_paths = [c.args[0] for c in calls]
        assert "results/task-abc-123/data.csv" in blob_paths

    def test_upload_result_csv_content(
        self, store: GCSResultStore, mock_gcs, sample_df: pd.DataFrame
    ):
        """Test that CSV blob gets correct content type."""
        store.upload_result("task-abc-123", sample_df)

        csv_blob = mock_gcs["bucket"].blob("results/task-abc-123/data.csv")
        csv_blob.upload_from_string.assert_called_once()
        call_kwargs = csv_blob.upload_from_string.call_args
        assert call_kwargs.kwargs["content_type"] == "text/csv"

    def test_upload_result_returns_signed_url(
        self, store: GCSResultStore, sample_df: pd.DataFrame
    ):
        """Test that upload_result returns a signed URL string."""
        result = store.upload_result("task-abc-123", sample_df)

        assert isinstance(result, str)
        assert "storage.googleapis.com" in result

    def test_upload_result_signed_url_params(
        self, store: GCSResultStore, mock_gcs, sample_df: pd.DataFrame
    ):
        """Test that signed URLs are generated with correct parameters."""
        store.upload_result("task-abc-123", sample_df)

        csv_blob = mock_gcs["bucket"].blob("results/task-abc-123/data.csv")
        csv_call = csv_blob.generate_signed_url.call_args
        assert csv_call.kwargs["expiration"] == timedelta(minutes=15)
        assert csv_call.kwargs["version"] == "v4"
        assert "attachment" in csv_call.kwargs["response_disposition"]
        assert "task-abc" in csv_call.kwargs["response_disposition"]

    def test_generate_signed_url(self, store: GCSResultStore, mock_gcs):
        """Test that generate_signed_url creates a URL for an existing blob."""
        result = store.generate_signed_url("task-xyz-456")

        assert isinstance(result, str)
        calls = mock_gcs["bucket"].blob.call_args_list
        blob_paths = [c.args[0] for c in calls]
        assert "results/task-xyz-456/data.csv" in blob_paths

    def test_generate_signed_url_params(self, store: GCSResultStore, mock_gcs):
        """Test that regenerated signed URLs have correct parameters."""
        store.generate_signed_url("task-xyz-456")

        csv_blob = mock_gcs["bucket"].blob("results/task-xyz-456/data.csv")
        csv_call = csv_blob.generate_signed_url.call_args
        assert csv_call.kwargs["version"] == "v4"
        assert "task-xyz" in csv_call.kwargs["response_disposition"]
