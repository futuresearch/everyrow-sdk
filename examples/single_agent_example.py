import asyncio

from pydantic import BaseModel, Field

from everyrow.ops import single_agent
from everyrow.spinner import spinner


class Competitor(BaseModel):
    company: str = Field(description="Company name")
    pricing_tier: str = Field(
        description="Pricing model, e.g. 'Freemium, $10-50/user/mo'"
    )
    target_market: str = Field(description="Primary customer segment")
    key_features: str = Field(description="Top 3 features or differentiators")


async def main():
    # Step 1: Generate a dataset of competitors
    print("Step 1: Research competitors")
    async with spinner("Researching competitors..."):
        competitors = await single_agent(
            task="Find the top 10 competitors in the B2B expense management software market",
            response_model=Competitor,
            return_table=True,
        )
    print(competitors.data.to_string())

    # Step 2: Distill insights from the dataset
    print("\n" + "=" * 80)
    print("Step 2: Identify market gaps")
    async with spinner("Analyzing market gaps..."):
        insights = await single_agent(
            task="""
                What gaps exist in the B2B expense management software market
                that these competitors aren't addressing?
            """,
            input=competitors,
        )
    print(insights.data.answer)


if __name__ == "__main__":
    asyncio.run(main())
