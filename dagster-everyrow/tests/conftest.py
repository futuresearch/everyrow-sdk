"""Shared pytest fixtures for dagster-everyrow tests."""

import pandas as pd
import pytest


@pytest.fixture
def leads_df() -> pd.DataFrame:
    """Create a leads DataFrame for testing."""
    return pd.DataFrame(
        [
            {
                "company": "Airtable",
                "title": "Senior Engineer",
                "salary": "$185000",
                "location": "Remote",
            },
            {
                "company": "Vercel",
                "title": "Lead Engineer",
                "salary": "Competitive",
                "location": "NYC",
            },
            {
                "company": "Notion",
                "title": "Staff Engineer",
                "salary": "$200000",
                "location": "San Francisco",
            },
        ]
    )


@pytest.fixture
def companies_df() -> pd.DataFrame:
    """Create a companies DataFrame for testing."""
    return pd.DataFrame(
        [
            {"name": "TechStart", "industry": "Software", "size": 50},
            {"name": "AILabs", "industry": "AI/ML", "size": 30},
            {"name": "DataFlow", "industry": "Data", "size": 100},
        ]
    )


@pytest.fixture
def contacts_df() -> pd.DataFrame:
    """Create a contacts DataFrame with duplicates for testing."""
    return pd.DataFrame(
        [
            {
                "name": "John Smith",
                "email": "john.smith@acme.com",
                "company": "Acme Corp",
            },
            {
                "name": "J. Smith",
                "email": "jsmith@acme.com",
                "company": "Acme Corporation",
            },
            {"name": "Mike Johnson", "email": "mike@data.com", "company": "DataFlow"},
        ]
    )


@pytest.fixture
def products_df() -> pd.DataFrame:
    """Create a products DataFrame for testing."""
    return pd.DataFrame(
        [
            {"product_name": "Photoshop", "category": "Design"},
            {"product_name": "VSCode", "category": "Development"},
        ]
    )


@pytest.fixture
def suppliers_df() -> pd.DataFrame:
    """Create a suppliers DataFrame for testing."""
    return pd.DataFrame(
        [
            {"company_name": "Adobe Inc", "approved": True},
            {"company_name": "Microsoft Corporation", "approved": True},
        ]
    )
