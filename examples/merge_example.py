import asyncio
from textwrap import dedent

from pandas import DataFrame

from everyrow.ops import merge
from everyrow.spinner import spinner


async def main():
    # Merge clinical trial data with pharmaceutical companies
    # The challenge: trial data uses sponsor names (often abbreviated or subsidiary names)
    # while company data uses parent company names - requires research to match correctly

    clinical_trials = DataFrame(
        [
            {
                "trial_id": "NCT05432109",
                "sponsor": "Genentech",
                "indication": "Non-small cell lung cancer",
                "phase": "Phase 3",
            },
            {
                "trial_id": "NCT05891234",
                "sponsor": "Janssen Pharmaceuticals",
                "indication": "Multiple myeloma",
                "phase": "Phase 2",
            },
            {
                "trial_id": "NCT05567890",
                "sponsor": "MSD",
                "indication": "Melanoma",
                "phase": "Phase 3",
            },
            {
                "trial_id": "NCT05234567",
                "sponsor": "AbbVie Inc",
                "indication": "Rheumatoid arthritis",
                "phase": "Phase 3",
            },
            {
                "trial_id": "NCT05678901",
                "sponsor": "BMS",
                "indication": "Acute myeloid leukemia",
                "phase": "Phase 2",
            },
        ]
    )

    pharma_companies = DataFrame(
        [
            {
                "company": "Roche Holding AG",
                "hq_country": "Switzerland",
                "2024_revenue_billions": 58.7,
            },
            {
                "company": "Johnson & Johnson",
                "hq_country": "United States",
                "2024_revenue_billions": 85.2,
            },
            {
                "company": "Merck & Co.",
                "hq_country": "United States",
                "2024_revenue_billions": 60.1,
            },
            {
                "company": "AbbVie",
                "hq_country": "United States",
                "2024_revenue_billions": 56.3,
            },
            {
                "company": "Bristol-Myers Squibb",
                "hq_country": "United States",
                "2024_revenue_billions": 45.0,
            },
        ]
    )

    async with spinner("Merging clinical trials with parent company data..."):
        result = await merge(
            task=dedent("""
                Merge clinical trial data with parent pharmaceutical company information.

                The sponsor names in the trials table are often subsidiaries or abbreviations:
                - Research which parent company owns each trial sponsor
                - Match trials to their parent company's financial data

                For example, Genentech is a subsidiary of Roche, Janssen is part of J&J,
                MSD is Merck's name outside the US, BMS is Bristol-Myers Squibb.
            """),
            left_table=clinical_trials,
            right_table=pharma_companies,
            merge_on_left="sponsor",
            merge_on_right="company",
        )
    print("Clinical Trials with Parent Company Data:")
    print(result.data.to_string())
    print(f"\nArtifact ID: {result.artifact_id}")


if __name__ == "__main__":
    asyncio.run(main())
