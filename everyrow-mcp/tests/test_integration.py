"""Integration tests for the MCP server.

These tests make real API calls to everyrow and require EVERYROW_API_KEY to be set.
Run with: pytest tests/test_integration.py -v -s

Note: These tests cost money ($1-2 per operation), so they are skipped by default.
Run with --run-integration to enable them.
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from everyrow_mcp.server import (
    AgentInput,
    DedupeInput,
    MergeInput,
    RankInput,
    ScreenInput,
    everyrow_agent,
    everyrow_dedupe,
    everyrow_merge,
    everyrow_rank,
    everyrow_screen,
)

# Skip all tests in this module unless --run-integration is passed
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"),
    reason="Integration tests are skipped by default. Set RUN_INTEGRATION_TESTS=1 to run.",
)


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to test fixtures."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Return a temporary output directory."""
    return tmp_path


@pytest.fixture
def small_jobs_csv(tmp_path: Path) -> Path:
    """Create a small jobs CSV for testing (5 rows to minimize cost)."""
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
    path = tmp_path / "small_jobs.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def small_companies_csv(tmp_path: Path) -> Path:
    """Create a small companies CSV for testing."""
    df = pd.DataFrame(
        [
            {"name": "TechStart", "industry": "Software", "size": 50},
            {"name": "AILabs", "industry": "AI/ML", "size": 30},
            {"name": "DataFlow", "industry": "Data", "size": 100},
            {"name": "CloudNine", "industry": "Cloud", "size": 75},
            {"name": "OldBank", "industry": "Finance", "size": 5000},
        ]
    )
    path = tmp_path / "small_companies.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def small_contacts_csv(tmp_path: Path) -> Path:
    """Create a small contacts CSV with duplicates for dedupe testing."""
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
    path = tmp_path / "small_contacts.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def products_csv(tmp_path: Path) -> Path:
    """Create a products CSV for merge testing."""
    df = pd.DataFrame(
        [
            {"product": "Photoshop", "category": "Design"},
            {"product": "VSCode", "category": "Development"},
            {"product": "Slack", "category": "Communication"},
        ]
    )
    path = tmp_path / "products.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def suppliers_csv(tmp_path: Path) -> Path:
    """Create a suppliers CSV for merge testing."""
    df = pd.DataFrame(
        [
            {"company": "Adobe Inc", "approved": True},
            {"company": "Microsoft Corporation", "approved": True},
            {"company": "Salesforce Inc", "approved": True},
        ]
    )
    path = tmp_path / "suppliers.csv"
    df.to_csv(path, index=False)
    return path


class TestScreenIntegration:
    """Integration tests for the screen tool."""

    @pytest.mark.asyncio
    async def test_screen_jobs(self, small_jobs_csv: Path, output_dir: Path):
        """Test screening jobs for remote senior roles."""
        params = ScreenInput(
            task="""
                Filter for positions that meet ALL criteria:
                1. Remote-friendly (location says Remote)
                2. Senior-level (title includes Senior, Staff, Principal, or Lead)
                3. Salary disclosed (specific dollar amount, not "Competitive")
            """,
            input_csv=str(small_jobs_csv),
            output_path=str(output_dir),
        )

        result = await everyrow_screen(params)
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        assert result_data["input_rows"] == 5

        # Check output file was created
        output_file = Path(result_data["output_file"])
        assert output_file.exists()

        # Read and verify output
        output_df = pd.read_csv(output_file)
        print(f"\nScreen result: {len(output_df)} rows")
        print(output_df)

        # We expect Airtable and Descript to pass (remote, senior, salary disclosed)
        # Vercel fails (salary not disclosed), Notion fails (not remote), Linear fails (not senior)
        assert len(output_df) <= 3  # At most 3 should pass


class TestRankIntegration:
    """Integration tests for the rank tool."""

    @pytest.mark.asyncio
    async def test_rank_companies(self, small_companies_csv: Path, output_dir: Path):
        """Test ranking companies by AI/ML maturity."""
        params = RankInput(
            task="Score by AI/ML adoption maturity and innovation focus. Higher score = more AI focused.",
            input_csv=str(small_companies_csv),
            output_path=str(output_dir),
            field_name="ai_score",
            field_type="float",
            ascending_order=False,  # Highest first
        )

        result = await everyrow_rank(params)
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        assert result_data["rows"] == 5

        # Check output file
        output_file = Path(result_data["output_file"])
        assert output_file.exists()

        output_df = pd.read_csv(output_file)
        print("\nRank result:")
        print(output_df)

        # AILabs should likely be near the top
        assert "ai_score" in output_df.columns


class TestDedupeIntegration:
    """Integration tests for the dedupe tool."""

    @pytest.mark.asyncio
    async def test_dedupe_contacts(self, small_contacts_csv: Path, output_dir: Path):
        """Test deduplicating contacts."""
        params = DedupeInput(
            equivalence_relation="""
                Two rows are duplicates if they represent the same person.
                Consider name abbreviations (J. Smith = John Smith),
                and company name variations (Acme Corp = Acme Corporation).
            """,
            input_csv=str(small_contacts_csv),
            output_path=str(output_dir),
            select_representative=True,
        )

        result = await everyrow_dedupe(params)
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        assert result_data["input_rows"] == 5

        # Check output file
        output_file = Path(result_data["output_file"])
        assert output_file.exists()

        output_df = pd.read_csv(output_file)
        print(f"\nDedupe result: {len(output_df)} rows")
        print(output_df)

        # Dedupe returns all rows with a 'selected' column marking representatives
        # We expect the equivalence_class_name to group duplicates
        if "selected" in output_df.columns:
            selected_df = output_df[output_df["selected"]]
            print(f"Selected representatives: {len(selected_df)}")
            # We expect 3 unique people (John/J. Smith, Alexandra/A. Butoi, Mike Johnson)
            assert len(selected_df) == 3
        else:
            # If no selected column, just verify output exists
            assert len(output_df) > 0


class TestMergeIntegration:
    """Integration tests for the merge tool."""

    @pytest.mark.asyncio
    async def test_merge_products_suppliers(
        self, products_csv: Path, suppliers_csv: Path, output_dir: Path
    ):
        """Test merging products with suppliers."""
        params = MergeInput(
            task="""
                Match each product to its parent company in the suppliers list.
                Photoshop is made by Adobe, VSCode by Microsoft, Slack by Salesforce.
            """,
            left_csv=str(products_csv),
            right_csv=str(suppliers_csv),
            output_path=str(output_dir),
        )

        result = await everyrow_merge(params)
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        assert result_data["left_rows"] == 3
        assert result_data["right_rows"] == 3

        # Check output file
        output_file = Path(result_data["output_file"])
        assert output_file.exists()

        output_df = pd.read_csv(output_file)
        print("\nMerge result:")
        print(output_df)

        # Should have merged data from both tables
        assert len(output_df) >= 1


class TestAgentIntegration:
    """Integration tests for the agent tool."""

    @pytest.mark.asyncio
    async def test_agent_company_research(self, output_dir: Path):
        """Test agent researching companies."""
        # Use only 2 companies to minimize cost
        df = pd.DataFrame(
            [
                {"name": "Anthropic"},
                {"name": "OpenAI"},
            ]
        )
        input_csv = output_dir / "companies_to_research.csv"
        df.to_csv(input_csv, index=False)

        params = AgentInput(
            task="Find the company's headquarters city and approximate employee count.",
            input_csv=str(input_csv),
            output_path=str(output_dir),
            response_schema={
                "properties": {
                    "headquarters": {
                        "type": "string",
                        "description": "City where HQ is located",
                    },
                    "employees": {
                        "type": "string",
                        "description": "Approximate employee count",
                    },
                },
                "required": ["headquarters"],
            },
        )

        result = await everyrow_agent(params)
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        assert result_data["rows_processed"] == 2

        # Check output file
        output_file = Path(result_data["output_file"])
        assert output_file.exists()

        output_df = pd.read_csv(output_file)
        print("\nAgent result:")
        print(output_df)

        # Should have research results
        assert "headquarters" in output_df.columns or "answer" in output_df.columns
