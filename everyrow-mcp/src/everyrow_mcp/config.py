from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _BaseSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    everyrow_api_url: str = Field(default="https://everyrow.io/api/v0")
    preview_size: int = Field(
        default=5, description="Number of rows in the initial results preview"
    )
    token_budget: int = Field(
        default=20000,
        description="Target token budget per page of inline results",
    )


class HttpSettings(_BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    mcp_server_url: str
    supabase_url: str
    supabase_anon_key: str

    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=13)
    redis_password: str | None = Field(default=None)
    redis_sentinel_endpoints: str | None = Field(
        default=None, description="Comma-separated host:port pairs"
    )
    redis_sentinel_master_name: str | None = Field(default=None)

    result_storage: Literal["memory", "gcs"]
    gcs_results_bucket: str | None = Field(default=None)

    signed_url_expiry_minutes: int = Field(
        default=15, description="GCS signed URL expiry in minutes"
    )

    @model_validator(mode="after")
    def _validate_redis(self):
        has_sentinel = self.redis_sentinel_endpoints and self.redis_sentinel_master_name
        has_direct = self.redis_host != "localhost" or self.redis_port != 6379
        if not has_sentinel and not has_direct:
            raise ValueError(
                "Redis: set REDIS_SENTINEL_ENDPOINTS + REDIS_SENTINEL_MASTER_NAME "
                "or REDIS_HOST + REDIS_PORT"
            )
        return self

    @model_validator(mode="after")
    def _validate_gcs(self):
        if self.result_storage == "gcs" and not self.gcs_results_bucket:
            raise ValueError("RESULT_STORAGE=gcs requires GCS_RESULTS_BUCKET")
        return self


class StdioSettings(_BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    everyrow_api_key: str
