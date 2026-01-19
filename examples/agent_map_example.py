import asyncio

from pandas import DataFrame

from everyrow.ops import agent_map


async def main():
    result = await agent_map(
        task="Find the company's most recent annual revenue and number of employees. If the company is a subsidiary, report figures for the subsidiary specifically, not the parent company.",
        input=DataFrame([
            {"company": "Stripe"},
            {"company": "Databricks"},
            {"company": "Canva"},
        ]),
    )
    print(f"Results:\n{result.data.to_string()}")


if __name__ == "__main__":
    asyncio.run(main())
