"""GCS storage for task result data."""

from __future__ import annotations

import logging
from datetime import timedelta
from io import StringIO

import pandas as pd
from google.cloud import storage

logger = logging.getLogger(__name__)


def _blob_path(task_id: str) -> str:
    """Build the GCS blob path for a task result CSV."""
    return f"results/{task_id}/data.csv"


class GCSResultStore:
    """Upload DataFrames to GCS and generate signed download URLs."""

    def __init__(
        self,
        bucket_name: str,
        signed_url_expiry_minutes: int = 15,
    ) -> None:
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)
        self._expiry = timedelta(minutes=signed_url_expiry_minutes)

    def upload_result(self, task_id: str, df: pd.DataFrame) -> str:
        """Upload DataFrame as CSV, return a signed download URL."""
        blob = self._bucket.blob(_blob_path(task_id))
        blob.upload_from_string(
            df.to_csv(index=False),
            content_type="text/csv",
        )
        return blob.generate_signed_url(
            expiration=self._expiry,
            version="v4",
            response_disposition=f'attachment; filename="results_{task_id[:8]}.csv"',
        )

    def download_csv(self, task_id: str) -> list[dict]:
        """Download CSV result data from GCS and return as records."""
        blob = self._bucket.blob(_blob_path(task_id))
        df = pd.read_csv(StringIO(blob.download_as_text()))
        return df.where(df.notna(), None).to_dict(orient="records")

    def generate_signed_url(self, task_id: str) -> str:
        """Re-generate a signed download URL for an existing upload."""
        blob = self._bucket.blob(_blob_path(task_id))
        return blob.generate_signed_url(
            expiration=self._expiry,
            version="v4",
            response_disposition=f'attachment; filename="results_{task_id[:8]}.csv"',
        )
