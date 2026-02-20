"""GCS storage for task result data."""

from __future__ import annotations

import json
import logging
from datetime import timedelta

import pandas as pd
from google.cloud import storage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SIGNED_URL_EXPIRY = timedelta(minutes=15)


class ResultURLs(BaseModel):
    """Signed URLs for JSON and CSV result downloads."""

    json_url: str
    csv_url: str


class GCSResultStore:
    """Upload DataFrames to GCS and generate signed URLs."""

    def __init__(self, bucket_name: str) -> None:
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)

    def upload_result(self, task_id: str, df: pd.DataFrame) -> ResultURLs:
        """Upload DataFrame as JSON + CSV, return signed URLs."""
        json_blob = self._bucket.blob(f"results/{task_id}/data.json")
        json_blob.upload_from_string(
            df.to_json(orient="records"),
            content_type="application/json",
        )

        csv_blob = self._bucket.blob(f"results/{task_id}/data.csv")
        csv_blob.upload_from_string(
            df.to_csv(index=False),
            content_type="text/csv",
        )

        return ResultURLs(
            json_url=json_blob.generate_signed_url(
                expiration=SIGNED_URL_EXPIRY, version="v4"
            ),
            csv_url=csv_blob.generate_signed_url(
                expiration=SIGNED_URL_EXPIRY,
                version="v4",
                response_disposition=f'attachment; filename="results_{task_id[:8]}.csv"',
            ),
        )

    def download_json(self, task_id: str) -> list[dict]:
        """Download the JSON result data from GCS."""
        blob = self._bucket.blob(f"results/{task_id}/data.json")
        return json.loads(blob.download_as_text())

    def generate_signed_urls(self, task_id: str) -> ResultURLs:
        """Re-generate signed URLs for an existing upload."""
        json_blob = self._bucket.blob(f"results/{task_id}/data.json")
        csv_blob = self._bucket.blob(f"results/{task_id}/data.csv")
        return ResultURLs(
            json_url=json_blob.generate_signed_url(
                expiration=SIGNED_URL_EXPIRY, version="v4"
            ),
            csv_url=csv_blob.generate_signed_url(
                expiration=SIGNED_URL_EXPIRY,
                version="v4",
                response_disposition=f'attachment; filename="results_{task_id[:8]}.csv"',
            ),
        )
