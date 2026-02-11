#!/usr/bin/env python3
"""Verbose progress test: log every poll response to see what the engine returns.

This dumps every status poll result to understand why intermediate
completed counts don't change with agent_map.
"""

import asyncio
import json
import sys
import time

import pandas as pd
from pydantic import BaseModel, Field

from everyrow import create_session
from everyrow.generated.models import TaskStatus
from everyrow.ops import agent_map_async
from everyrow.task import (
    EffortLevel,
    _get_progress,
    get_task_status,
)

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
    ]
)


class CompanyHQ(BaseModel):
    city: str = Field(description="City where the company is headquartered")
    country: str = Field(description="Country where the company is headquartered")


async def main():
    async with create_session(name="Progress Verbose Test") as session:
        task = await agent_map_async(
            session=session,
            task="What city and country is the headquarters of this company?",
            input=companies,
            response_model=CompanyHQ,
            effort_level=EffortLevel.LOW,
        )
        print(f"Task submitted: {task.task_id}", file=sys.stderr, flush=True)

        client = task._client
        start = time.time()
        poll_num = 0
        while True:
            status = await get_task_status(task.task_id, client)
            progress = _get_progress(status)
            elapsed = time.time() - start
            poll_num += 1
            entry = {
                "poll": poll_num,
                "elapsed": round(elapsed, 1),
                "status": status.status.value,
                "progress": {
                    "pending": progress.pending,
                    "running": progress.running,
                    "completed": progress.completed,
                    "failed": progress.failed,
                    "total": progress.total,
                }
                if progress
                else None,
            }
            print(json.dumps(entry), file=sys.stderr, flush=True)

            if status.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.REVOKED,
            ):
                break
            await asyncio.sleep(2)

        result = await task.await_result()
        print(result.data.to_string())


asyncio.run(main())
