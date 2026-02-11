#!/usr/bin/env python3
"""End-to-end test: run agent_map against local engine and verify progress output.

Usage:
    EVERYROW_API_URL=http://localhost:8000/api/v0 \
    EVERYROW_API_KEY=<jwt> \
    python -u tests/test_progress_e2e.py

Requires: local engine on :8000 + Celery worker running.
"""

import asyncio

import pandas as pd
from pydantic import BaseModel, Field

from everyrow.ops import agent_map
from everyrow.task import EffortLevel

# 15 rows — enough for multiple intermediate progress updates at 2s poll interval
companies = pd.DataFrame(
    [
        {"company": "Anthropic"},
        {"company": "Stripe"},
        {"company": "Notion"},
        {"company": "Linear"},
        {"company": "Vercel"},
        {"company": "Figma"},
        {"company": "Datadog"},
        {"company": "Cloudflare"},
        {"company": "Snowflake"},
        {"company": "Databricks"},
        {"company": "GitLab"},
        {"company": "HashiCorp"},
        {"company": "Confluent"},
        {"company": "Elastic"},
        {"company": "MongoDB"},
    ]
)


class CompanyHQ(BaseModel):
    city: str = Field(description="City where the company is headquartered")
    country: str = Field(description="Country where the company is headquartered")


async def main():
    result = await agent_map(
        task="What city and country is the headquarters of this company?",
        input=companies,
        response_model=CompanyHQ,
        effort_level=EffortLevel.LOW,
    )
    print(result.data.to_string())


asyncio.run(main())
