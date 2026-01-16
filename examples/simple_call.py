import asyncio

from pandas import DataFrame
from pydantic import BaseModel

from everyrow.ops import agent_map, single_agent


class MyInput(BaseModel):
    country: str


async def main():
    # Single input - no session needed
    print("Single agent call:")
    result = await single_agent(
        task="What is the capital of the given country?",
        input=MyInput(country="France"),
    )
    print(f"Result: {result.data}")

    # Batch processing - no session needed
    print("\nBatch processing with agent_map:")
    batch_result = await agent_map(
        task="What is the capital of the given country?",
        input=DataFrame([{"country": "India"}, {"country": "USA"}, {"country": "Japan"}]),
    )
    print(f"Results:\n{batch_result.data.to_string()}")


if __name__ == "__main__":
    asyncio.run(main())
