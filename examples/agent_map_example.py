import asyncio
from textwrap import dedent

from pandas import DataFrame
from pydantic import BaseModel, Field

from everyrow.ops import agent_map
from everyrow.spinner import spinner


class CompanyFinancials(BaseModel):
    annual_revenue_usd: int = Field(description="Most recent annual revenue in USD")
    employee_count: int = Field(description="Current number of employees")
    founded_year: int = Field(description="Year the company was founded")


async def main():
    # Research financial information for tech companies
    companies = DataFrame(
        [
            {"company": "Stripe", "industry": "Payments"},
            {"company": "Databricks", "industry": "Data & AI"},
            {"company": "Canva", "industry": "Design"},
            {"company": "Figma", "industry": "Design"},
            {"company": "Notion", "industry": "Productivity"},
        ]
    )

    # Example 1: Basic usage with default response
    print("Example 1: Basic agent_map")
    async with spinner("Researching company revenues..."):
        basic_result = await agent_map(
            task="Find the company's most recent annual revenue in USD",
            input=companies,
        )
    print("Basic Results:")
    print(basic_result.data.to_string())
    print(f"\nArtifact ID: {basic_result.artifact_id}")

    # Example 2: Structured output with a response model
    print("\n" + "=" * 80)
    print("Example 2: Structured output with response model")
    task = dedent("""
        Research the company's financials. Find:
        1. Their most recent annual revenue (in USD)
        2. Current employee count
        3. Year founded

        If the company is a subsidiary, report figures for the subsidiary
        specifically, not the parent company.
    """)

    async with spinner("Researching company financials..."):
        structured_result = await agent_map(
            task=task,
            input=companies,
            response_model=CompanyFinancials,
        )
    print("Structured Results:")
    print(structured_result.data.to_string())
    print(f"\nArtifact ID: {structured_result.artifact_id}")


if __name__ == "__main__":
    asyncio.run(main())
