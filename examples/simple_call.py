import asyncio
from datetime import datetime

from pandas import DataFrame
from pydantic import BaseModel

from everyrow import create_client, create_session
from everyrow.generated.models import TaskEffort
from everyrow.ops import agent_map, create_scalar_artifact, single_agent
from everyrow.session import Session


async def upload_simple_scalar(session: Session):
    class MyModel(BaseModel):
        name: str
        age: int

    model = MyModel(name="John", age=30)
    artifact_id = await create_scalar_artifact(model, session)
    print(artifact_id)


async def call_single_agent(session: Session):
    class MyInput(BaseModel):
        country: str

    result = await single_agent(
        session=session,
        task="What is the capital of the given country?",
        return_table=False,
        input=MyInput(country="India"),
    )
    print(result)


async def call_single_agent_with_table_output(session: Session):
    class MyInput(BaseModel):
        country: str

    result = await single_agent(
        session=session,
        task="What are the three largest cities in the given country?",
        return_table=True,
        effort_level=TaskEffort.LOW,
        input=MyInput(country="India"),
    )
    print(result)


async def call_agent_map(session: Session):
    result = await agent_map(
        session=session,
        task="What is the capital of the given country?",
        input=DataFrame([{"country": "India"}, {"country": "USA"}]),
    )
    print(result)


async def call_agent_map_with_table_output(session: Session):
    result = await agent_map(
        session=session,
        task="What are the three largest cities in the given country?",
        return_table_per_row=True,
        input=DataFrame([{"country": "India"}, {"country": "USA"}]),
    )
    print(result)


async def main():
    async with create_client() as client:
        session_name = f"everyrow SDK Examples {datetime.now().isoformat()}"
        async with create_session(client=client, name=session_name) as session:
            print(f"Session URL: {session.get_url()}")
            print("Uploading simple scalar...")
            await upload_simple_scalar(session)
            print("Calling single agent...")
            await call_single_agent(session)
            print("Calling single agent with table output...")
            await call_single_agent_with_table_output(session)
            print("Calling agent map...")
            await call_agent_map(session)
            print("Calling agent map with table output...")
            await call_agent_map_with_table_output(session)


if __name__ == "__main__":
    asyncio.run(main())
