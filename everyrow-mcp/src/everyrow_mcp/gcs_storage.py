"""GCS storage for task result data."""

from __future__ import annotations

import json
import logging
from datetime import timedelta

import pandas as pd
from google.cloud import storage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_DEFAULT_SIGNED_URL_EXPIRY_MINUTES = 15


def _blob_path(task_id: str, ext: str) -> str:
    """Build the GCS blob path for a task result file."""
    return f"results/{task_id}/data.{ext}"


class ResultURLs(BaseModel):
    """Signed URLs for JSON and CSV result downloads."""

    json_url: str
    csv_url: str


class GCSResultStore:
    """Upload DataFrames to GCS and generate signed URLs."""

    def __init__(
        self,
        bucket_name: str,
        signed_url_expiry_minutes: int = _DEFAULT_SIGNED_URL_EXPIRY_MINUTES,
    ) -> None:
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)
        self._expiry = timedelta(minutes=signed_url_expiry_minutes)

    def upload_result(self, task_id: str, df: pd.DataFrame) -> ResultURLs:
        """Upload DataFrame as JSON + CSV, return signed URLs."""
        json_blob = self._bucket.blob(_blob_path(task_id, "json"))
        json_blob.upload_from_string(
            df.to_json(orient="records"),
            content_type="application/json",
        )

        csv_blob = self._bucket.blob(_blob_path(task_id, "csv"))
        csv_blob.upload_from_string(
            df.to_csv(index=False),
            content_type="text/csv",
        )

        return ResultURLs(
            json_url=json_blob.generate_signed_url(
                expiration=self._expiry, version="v4"
            ),
            csv_url=csv_blob.generate_signed_url(
                expiration=self._expiry,
                version="v4",
                response_disposition=f'attachment; filename="results_{task_id[:8]}.csv"',
            ),
        )

    def download_json(self, task_id: str) -> list[dict]:
        """Download the JSON result data from GCS."""
        blob = self._bucket.blob(_blob_path(task_id, "json"))
        return json.loads(blob.download_as_text())

    def generate_signed_urls(self, task_id: str) -> ResultURLs:
        """Re-generate signed URLs for an existing upload."""
        json_blob = self._bucket.blob(_blob_path(task_id, "json"))
        csv_blob = self._bucket.blob(_blob_path(task_id, "csv"))
        return ResultURLs(
            json_url=json_blob.generate_signed_url(
                expiration=self._expiry, version="v4"
            ),
            csv_url=csv_blob.generate_signed_url(
                expiration=self._expiry,
                version="v4",
                response_disposition=f'attachment; filename="results_{task_id[:8]}.csv"',
            ),
        )
