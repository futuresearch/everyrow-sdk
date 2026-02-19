from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _BaseSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    preview_size: int = Field(
        default=5, description="Number of rows in the initial results preview"
    )


class HttpSettings(_BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    mcp_server_url: str
    supabase_url: str
    supabase_anon_key: str
    redis_encryption_key: str

    redis_host: str
    redis_port: int
    redis_db: int
    redis_password: str | None = Field(default=None)
    redis_sentinel_endpoints: str | None = Field(
        default=None, description="Comma-separated host:port pairs"
    )
    redis_sentinel_master_name: str | None = Field(default=None)

    result_storage: Literal["memory", "gcs"]
    gcs_results_bucket: str | None = Field(default=None)

    everyrow_api_url: str

    @model_validator(mode="after")
    def _validate_gcs(self):
        if self.result_storage == "gcs" and not self.gcs_results_bucket:
            raise ValueError("RESULT_STORAGE=gcs requires GCS_RESULTS_BUCKET")
        return self


class StdioSettings(_BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    everyrow_api_key: str
    everyrow_api_url: str = Field(default="https://everyrow.io/api/v0")
