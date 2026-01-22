import asyncio
from textwrap import dedent

from pandas import DataFrame
from pydantic import BaseModel

from everyrow.ops import screen


class VendorRiskAssessment(BaseModel):
    approved: bool
    risk_level: str  # "low", "medium", "high"
    security_concerns: str
    financial_stability_notes: str
    recommendation: str


async def main():
    # Screen potential enterprise software vendors for partnership
    # This requires actual research - not just pattern matching on the input data
    vendors = DataFrame(
        [
            {"company": "Okta", "category": "Identity Management", "website": "okta.com"},
            {"company": "LastPass", "category": "Password Management", "website": "lastpass.com"},
            {"company": "Snowflake", "category": "Data Warehouse", "website": "snowflake.com"},
            {"company": "Cloudflare", "category": "CDN & Security", "website": "cloudflare.com"},
            {"company": "MongoDB", "category": "Database", "website": "mongodb.com"},
        ]
    )

    print("Running vendor risk assessment screening...")
    result = await screen(
        task=dedent("""Perform vendor risk assessment for each company. Research and evaluate:

            1. Security track record: Have they had any significant data breaches or security
            incidents in the past 3 years? How did they respond?

            2. Financial stability: Are there signs of financial distress (major layoffs,
            funding difficulties, declining revenue)?

            3. Overall recommendation: Based on your research, should we proceed with
            this vendor for enterprise use?

            Only approve vendors with low or medium risk and no unresolved critical security incidents."""),
        input=vendors,
        response_model=VendorRiskAssessment,
        batch_size=5,
    )
    print("Vendor Risk Assessment Results:")
    print(result.data.to_string())
    print(f"\nArtifact ID: {result.artifact_id}")


if __name__ == "__main__":
    asyncio.run(main())
