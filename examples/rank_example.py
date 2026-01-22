import asyncio
from textwrap import dedent

from pandas import DataFrame
from pydantic import BaseModel, Field

from everyrow.ops import rank


class ContributionRanking(BaseModel):
    contribution_score: int = Field(
        description="Total citation count"
    )
    most_significant_contribution: str = Field(
        description="Single most important paper authored by a firm leader"
    )


async def main():
    # Rank AI research organizations by the total citation count of the leadership team
    # This requires researching each org's leaders and all their publications
    ai_research_orgs = DataFrame(
        [
            {"organization": "OpenAI", "type": "Private lab", "founded": 2015},
            {"organization": "Google DeepMind", "type": "Corporate lab", "founded": 2010},
            {"organization": "Anthropic", "type": "Private lab", "founded": 2021},
            {"organization": "Meta FAIR", "type": "Corporate lab", "founded": 2013},
            {"organization": "Microsoft Research", "type": "Corporate lab", "founded": 1991},
            {"organization": "Stanford HAI", "type": "Academic", "founded": 2019},
            {"organization": "MIT CSAIL", "type": "Academic", "founded": 2003},
            {"organization": "Berkeley AI Research", "type": "Academic", "founded": 2010},
            {"organization": "Mistral AI", "type": "Private lab", "founded": 2023},
            {"organization": "xAI", "type": "Private lab", "founded": 2023},
            {"organization": "Cohere", "type": "Private lab", "founded": 2019},
            {"organization": "Allen Institute for AI", "type": "Non-profit", "founded": 2014},
        ]
    )

    task = dedent("""
        Research the total citation count of all leaders of the given AI research organization.

        A leader is defined as a C-Suite or founder of the company.
        Citation count should count all major publications. Top ten by each person is sufficient.
    """)

    # Example 1: Basic ranking with a single score field
    print("Example 1: Basic ranking")
    result = await rank(
        task=task,
        input=ai_research_orgs,
        field_name="contribution_score",
        ascending_order=False,
    )
    print("AI Research Organization Rankings:")
    print(result.data.to_string())
    print(f"\nArtifact ID: {result.artifact_id}")

    # Example 2: Ranking with a custom response model for additional context
    print("\n" + "=" * 80)
    print("Example 2: Ranking with detailed response model")
    detailed_result = await rank(
        task=task + "\n\nAlso include their single most significant contribution.",
        input=ai_research_orgs,
        field_name="contribution_score",
        response_model=ContributionRanking,
        ascending_order=False,
    )
    print("Detailed Rankings with Context:")
    print(detailed_result.data.to_string())
    print(f"\nArtifact ID: {detailed_result.artifact_id}")


if __name__ == "__main__":
    asyncio.run(main())
