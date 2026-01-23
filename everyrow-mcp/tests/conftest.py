"""Shared pytest fixtures for everyrow MCP server tests."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def jobs_csv(tmp_path: Path) -> str:
    """Create a jobs CSV for testing."""
    df = pd.DataFrame(
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
            {
                "company": "Linear",
                "title": "Junior Developer",
                "salary": "$85000",
                "location": "Remote",
            },
            {
                "company": "Descript",
                "title": "Principal Architect",
                "salary": "$250000",
                "location": "Remote",
            },
        ]
    )
    path = tmp_path / "jobs.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def companies_csv(tmp_path: Path) -> str:
    """Create a companies CSV for testing."""
    df = pd.DataFrame(
        [
            {"name": "TechStart", "industry": "Software", "size": 50},
            {"name": "AILabs", "industry": "AI/ML", "size": 30},
            {"name": "DataFlow", "industry": "Data", "size": 100},
            {"name": "CloudNine", "industry": "Cloud", "size": 75},
            {"name": "OldBank", "industry": "Finance", "size": 5000},
        ]
    )
    path = tmp_path / "companies.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def contacts_csv(tmp_path: Path) -> str:
    """Create a contacts CSV with duplicates for testing."""
    df = pd.DataFrame(
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
            {
                "name": "Alexandra Butoi",
                "email": "a.butoi@tech.io",
                "company": "TechStart",
            },
            {
                "name": "A. Butoi",
                "email": "alexandra.b@tech.io",
                "company": "TechStart Inc",
            },
            {"name": "Mike Johnson", "email": "mike@data.com", "company": "DataFlow"},
        ]
    )
    path = tmp_path / "contacts.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def products_csv(tmp_path: Path) -> str:
    """Create a products CSV for testing."""
    df = pd.DataFrame(
        [
            {
                "product_name": "Photoshop",
                "category": "Design",
                "vendor": "Adobe Systems",
            },
            {
                "product_name": "VSCode",
                "category": "Development",
                "vendor": "Microsoft",
            },
            {
                "product_name": "Slack",
                "category": "Communication",
                "vendor": "Salesforce",
            },
        ]
    )
    path = tmp_path / "products.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def suppliers_csv(tmp_path: Path) -> str:
    """Create a suppliers CSV for testing."""
    df = pd.DataFrame(
        [
            {"company_name": "Adobe Inc", "approved": True},
            {"company_name": "Microsoft Corporation", "approved": True},
            {"company_name": "Salesforce Inc", "approved": True},
        ]
    )
    path = tmp_path / "suppliers.csv"
    df.to_csv(path, index=False)
    return str(path)
