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

# CSV fixtures are defined in conftest.py


class TestScreenIntegration:
    """Integration tests for the screen tool."""

    @pytest.mark.asyncio
    async def test_screen_jobs(self, jobs_csv: Path, tmp_path: Path):
        """Test screening jobs for remote senior roles."""
        params = ScreenInput(
            task="""
                Filter for positions that meet ALL criteria:
                1. Remote-friendly (location says Remote)
                2. Senior-level (title includes Senior, Staff, Principal, or Lead)
                3. Salary disclosed (specific dollar amount, not "Competitive")
            """,
            input_csv=str(jobs_csv),
            output_path=str(tmp_path),
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
    async def test_rank_companies(self, companies_csv: Path, tmp_path: Path):
        """Test ranking companies by AI/ML maturity."""
        params = RankInput(
            task="Score by AI/ML adoption maturity and innovation focus. Higher score = more AI focused.",
            input_csv=str(companies_csv),
            output_path=str(tmp_path),
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
    async def test_dedupe_contacts(self, contacts_csv: Path, tmp_path: Path):
        """Test deduplicating contacts."""
        params = DedupeInput(
            equivalence_relation="""
                Two rows are duplicates if they represent the same person.
                Consider name abbreviations (J. Smith = John Smith),
                and company name variations (Acme Corp = Acme Corporation).
            """,
            input_csv=str(contacts_csv),
            output_path=str(tmp_path),
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
        self, products_csv: Path, suppliers_csv: Path, tmp_path: Path
    ):
        """Test merging products with suppliers."""
        params = MergeInput(
            task="""
                Match each product to its parent company in the suppliers list.
                Photoshop is made by Adobe, VSCode by Microsoft, Slack by Salesforce.
            """,
            left_csv=str(products_csv),
            right_csv=str(suppliers_csv),
            output_path=str(tmp_path),
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
    async def test_agent_company_research(self, tmp_path: Path):
        """Test agent researching companies."""
        # Use only 2 companies to minimize cost
        df = pd.DataFrame(
            [
                {"name": "Anthropic"},
                {"name": "OpenAI"},
            ]
        )
        input_csv = tmp_path / "companies_to_research.csv"
        df.to_csv(input_csv, index=False)

        params = AgentInput(
            task="Find the company's headquarters city and approximate employee count.",
            input_csv=str(input_csv),
            output_path=str(tmp_path),
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
