import asyncio

from pydantic import BaseModel

from everyrow.ops import single_agent


class MyInput(BaseModel):
    company: str


async def main():
    result = await single_agent(
        task="Find the company's most recent annual revenue and number of employees. If the company is a subsidiary, report figures for the subsidiary specifically, not the parent company.",
        input=MyInput(company="Stripe"),
    )
    print(f"Result: {result.data}")


if __name__ == "__main__":
    asyncio.run(main())
