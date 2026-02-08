"""Tests for the MCP server tools.

These tests mock the everyrow SDK operations to test the MCP tool logic
without making actual API calls.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest

from everyrow_mcp.server import (
    AgentInput,
    AgentSubmitInput,
    DedupeInput,
    MergeInput,
    ProgressInput,
    RankInput,
    ResultsInput,
    ScreenInput,
    _active_tasks,
    _schema_to_model,
    everyrow_agent,
    everyrow_agent_submit,
    everyrow_dedupe,
    everyrow_merge,
    everyrow_progress,
    everyrow_rank,
    everyrow_results,
    everyrow_screen,
)

# CSV fixtures are defined in conftest.py


class TestSchemaToModel:
    """Tests for _schema_to_model helper."""

    def test_simple_schema(self):
        """Test converting a simple schema."""
        schema = {
            "properties": {
                "score": {"type": "number", "description": "A score"},
                "name": {"type": "string", "description": "A name"},
            },
            "required": ["score"],
        }

        model = _schema_to_model("TestModel", schema)

        # Check model was created with correct fields
        assert "score" in model.model_fields
        assert "name" in model.model_fields

    def test_schema_without_required(self):
        """Test schema where all fields are optional."""
        schema = {
            "properties": {
                "value": {"type": "integer"},
            }
        }

        model = _schema_to_model("OptionalModel", schema)
        assert "value" in model.model_fields

    def test_all_types(self):
        """Test all supported JSON schema types."""
        schema = {
            "properties": {
                "str_field": {"type": "string"},
                "int_field": {"type": "integer"},
                "float_field": {"type": "number"},
                "bool_field": {"type": "boolean"},
            }
        }

        model = _schema_to_model("AllTypes", schema)
        assert len(model.model_fields) == 4


class TestScreenTool:
    """Tests for everyrow_screen tool."""

    @pytest.mark.asyncio
    async def test_screen_basic(self, jobs_csv: str, tmp_path: Path):
        """Test basic screen operation."""
        # Create mock result
        mock_result = MagicMock()
        mock_result.data = pd.DataFrame(
            {
                "company": ["Airtable", "Descript"],
                "title": ["Senior Engineer", "Principal Architect"],
            }
        )

        with patch("everyrow_mcp.server.screen", new_callable=AsyncMock) as mock_screen:
            mock_screen.return_value = mock_result

            params = ScreenInput(
                task="Filter for remote senior roles with disclosed salary",
                input_csv=jobs_csv,
                output_path=str(tmp_path),
            )

            result = await everyrow_screen(params)
            result_data = json.loads(result)

            assert result_data["status"] == "success"
            assert result_data["input_rows"] == 5
            assert result_data["output_rows"] == 2
            assert "screened_jobs.csv" in result_data["output_file"]

    @pytest.mark.asyncio
    async def test_screen_with_custom_schema(self, jobs_csv: str, tmp_path: Path):
        """Test screen with custom response schema."""
        mock_result = MagicMock()
        mock_result.data = pd.DataFrame({"company": ["Airtable"]})

        with patch("everyrow_mcp.server.screen", new_callable=AsyncMock) as mock_screen:
            mock_screen.return_value = mock_result

            params = ScreenInput(
                task="Filter for remote roles",
                input_csv=jobs_csv,
                output_path=str(tmp_path / "custom_output.csv"),
                response_schema={
                    "properties": {
                        "passes": {"type": "boolean"},
                        "reason": {"type": "string"},
                    }
                },
            )

            result = await everyrow_screen(params)
            result_data = json.loads(result)

            assert result_data["status"] == "success"
            assert result_data["output_file"].endswith("custom_output.csv")


class TestRankTool:
    """Tests for everyrow_rank tool."""

    @pytest.mark.asyncio
    async def test_rank_basic(self, companies_csv: str, tmp_path: Path):
        """Test basic rank operation."""
        mock_result = MagicMock()
        mock_result.data = pd.DataFrame(
            {
                "name": ["AILabs", "TechStart", "DataFlow Inc"],
                "ai_score": [95.0, 85.0, 60.0],
            }
        )

        with patch("everyrow_mcp.server.rank", new_callable=AsyncMock) as mock_rank:
            mock_rank.return_value = mock_result

            params = RankInput(
                task="Score by AI/ML maturity",
                input_csv=companies_csv,
                output_path=str(tmp_path),
                field_name="ai_score",
                ascending_order=False,
            )

            result = await everyrow_rank(params)
            result_data = json.loads(result)

            assert result_data["status"] == "success"
            assert result_data["sorted_by"] == "ai_score"
            assert result_data["ascending"] is False
            assert "ranked_companies.csv" in result_data["output_file"]


class TestDedupeTool:
    """Tests for everyrow_dedupe tool."""

    @pytest.mark.asyncio
    async def test_dedupe_basic(self, contacts_csv: str, tmp_path: Path):
        """Test basic dedupe operation."""
        mock_result = MagicMock()
        mock_result.data = pd.DataFrame(
            {
                "name": ["John Smith", "Alexandra Butoi", "Mike Johnson", "Sarah Lee"],
                "email": [
                    "john.smith@acme.com",
                    "a.butoi@techstart.io",
                    "mike.j@dataflow.com",
                    "sarah@cloudnine.io",
                ],
            }
        )

        with patch("everyrow_mcp.server.dedupe", new_callable=AsyncMock) as mock_dedupe:
            mock_dedupe.return_value = mock_result

            params = DedupeInput(
                equivalence_relation="Same person even with name abbreviations",
                input_csv=contacts_csv,
                output_path=str(tmp_path),
            )

            result = await everyrow_dedupe(params)
            result_data = json.loads(result)

            assert result_data["status"] == "success"
            assert result_data["input_rows"] == 5
            assert result_data["output_rows"] == 4
            assert result_data["duplicates_removed"] == 1
            assert "deduped_contacts.csv" in result_data["output_file"]


class TestMergeTool:
    """Tests for everyrow_merge tool."""

    @pytest.mark.asyncio
    async def test_merge_basic(
        self, products_csv: str, suppliers_csv: str, tmp_path: Path
    ):
        """Test basic merge operation."""
        mock_result = MagicMock()
        mock_result.data = pd.DataFrame(
            {
                "product_name": ["Photoshop", "VSCode", "Slack"],
                "vendor": ["Adobe Systems", "Microsoft", "Salesforce"],
                "approved": [True, True, True],
            }
        )

        with patch("everyrow_mcp.server.merge", new_callable=AsyncMock) as mock_merge:
            mock_merge.return_value = mock_result

            params = MergeInput(
                task="Match products to approved suppliers",
                left_csv=products_csv,
                right_csv=suppliers_csv,
                output_path=str(tmp_path),
                merge_on_left="vendor",
                merge_on_right="company_name",
            )

            result = await everyrow_merge(params)
            result_data = json.loads(result)

            assert result_data["status"] == "success"
            assert result_data["left_rows"] == 3
            assert result_data["right_rows"] == 3
            assert "merged_products.csv" in result_data["output_file"]


class TestAgentTool:
    """Tests for everyrow_agent tool."""

    @pytest.mark.asyncio
    async def test_agent_basic(self, companies_csv: str, tmp_path: Path):
        """Test basic agent operation."""
        mock_result = MagicMock()
        mock_result.data = pd.DataFrame(
            {
                "name": ["TechStart", "AILabs"],
                "answer": ["Series A, $10M from Sequoia", "Seed, $2M from a16z"],
            }
        )

        with patch(
            "everyrow_mcp.server.agent_map", new_callable=AsyncMock
        ) as mock_agent:
            mock_agent.return_value = mock_result

            params = AgentInput(
                task="Find this company's latest funding round",
                input_csv=companies_csv,
                output_path=str(tmp_path),
            )

            result = await everyrow_agent(params)
            result_data = json.loads(result)

            assert result_data["status"] == "success"
            assert result_data["rows_processed"] == 2
            assert "agent_companies.csv" in result_data["output_file"]

    @pytest.mark.asyncio
    async def test_agent_with_schema(self, companies_csv: str, tmp_path: Path):
        """Test agent with custom response schema."""
        mock_result = MagicMock()
        mock_result.data = pd.DataFrame(
            {
                "name": ["TechStart"],
                "funding_round": ["Series A"],
                "amount": [10000000],
            }
        )

        with patch(
            "everyrow_mcp.server.agent_map", new_callable=AsyncMock
        ) as mock_agent:
            mock_agent.return_value = mock_result

            params = AgentInput(
                task="Find funding info",
                input_csv=companies_csv,
                output_path=str(tmp_path / "funding.csv"),
                response_schema={
                    "properties": {
                        "funding_round": {"type": "string"},
                        "amount": {"type": "integer"},
                    }
                },
            )

            result = await everyrow_agent(params)
            result_data = json.loads(result)

            assert result_data["status"] == "success"
            assert result_data["output_file"].endswith("funding.csv")


class TestInputValidation:
    """Tests for input validation."""

    def test_screen_input_validates_csv_path(self, tmp_path: Path):
        """Test ScreenInput validates CSV path."""
        with pytest.raises(ValueError, match="does not exist"):
            ScreenInput(
                task="test",
                input_csv=str(tmp_path / "nonexistent.csv"),
                output_path=str(tmp_path),
            )

    def test_rank_input_validates_field_type(self, tmp_path: Path):
        """Test RankInput validates field_type."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        with pytest.raises(ValueError, match="must be one of"):
            RankInput(
                task="test",
                input_csv=str(csv_file),
                output_path=str(tmp_path),
                field_name="score",
                field_type="invalid",
            )

    def test_merge_input_validates_both_csvs(self, tmp_path: Path):
        """Test MergeInput validates both CSV paths."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        with pytest.raises(ValueError, match="does not exist"):
            MergeInput(
                task="test",
                left_csv=str(csv_file),
                right_csv=str(tmp_path / "nonexistent.csv"),
                output_path=str(tmp_path),
            )


# =============================================================================
# Submit / Progress / Results tool tests
# =============================================================================


def _make_mock_task(task_id=None):
    """Create a mock EveryrowTask with a task_id."""
    task = MagicMock()
    task.task_id = task_id or uuid4()
    return task


def _make_mock_session(session_id=None):
    """Create a mock Session."""
    session = MagicMock()
    session.session_id = session_id or uuid4()
    session.get_url.return_value = f"https://everyrow.io/sessions/{session.session_id}"
    return session


def _make_mock_client():
    """Create a mock AuthenticatedClient."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


class TestAgentSubmit:
    """Tests for everyrow_agent_submit."""

    @pytest.mark.asyncio
    async def test_submit_returns_task_id(self, companies_csv: str):
        """Test that submit returns immediately with task_id and session_url."""
        mock_task = _make_mock_task()
        mock_session = _make_mock_session()
        mock_client = _make_mock_client()
        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("everyrow_mcp.server.agent_map_async", new_callable=AsyncMock) as mock_op,
            patch("everyrow_mcp.server.create_client", return_value=mock_client),
            patch("everyrow_mcp.server.create_session", return_value=mock_session_ctx),
        ):
            mock_op.return_value = mock_task

            params = AgentSubmitInput(
                task="Find HQ for each company",
                input_csv=companies_csv,
            )
            result = await everyrow_agent_submit(params)
            data = json.loads(result)

            assert data["task_id"] == str(mock_task.task_id)
            assert "session_url" in data
            assert data["total"] == 5  # companies.csv has 5 rows
            assert "instructions" in data
            assert "everyrow_progress" in data["instructions"]

            # Verify task was stored
            assert str(mock_task.task_id) in _active_tasks
            # Clean up
            del _active_tasks[str(mock_task.task_id)]


class TestProgress:
    """Tests for everyrow_progress."""

    @pytest.mark.asyncio
    async def test_progress_unknown_task(self):
        """Test progress with unknown task_id."""
        params = ProgressInput(task_id="nonexistent-id")

        with patch("everyrow_mcp.server.asyncio.sleep", new_callable=AsyncMock):
            result = await everyrow_progress(params)

        data = json.loads(result)
        assert data["status"] == "error"
        assert "Unknown task_id" in data["error"]

    @pytest.mark.asyncio
    async def test_progress_running_task(self):
        """Test progress returns status counts for a running task."""
        task_id = str(uuid4())
        mock_client = _make_mock_client()

        # Register a task
        import time
        _active_tasks[task_id] = {
            "client": mock_client,
            "started_at": time.monotonic(),
            "session": MagicMock(),
            "session_ctx": MagicMock(),
            "total": 10,
            "session_url": "https://everyrow.io/sessions/test",
            "input_csv": "/tmp/test.csv",
            "prefix": "agent",
        }

        # Mock the status response
        mock_status = MagicMock()
        mock_status.status = MagicMock(value="running")
        mock_status.error = None
        mock_status.additional_properties = {
            "progress": {"pending": 2, "running": 3, "completed": 4, "failed": 1, "total": 10}
        }

        try:
            with (
                patch("everyrow_mcp.server.get_task_status_tasks_task_id_status_get.asyncio", new_callable=AsyncMock, return_value=mock_status),
                patch("everyrow_mcp.server.asyncio.sleep", new_callable=AsyncMock),
            ):
                params = ProgressInput(task_id=task_id)
                result = await everyrow_progress(params)

            data = json.loads(result)
            assert data["status"] == "running"
            assert data["completed"] == 4
            assert data["failed"] == 1
            assert data["running"] == 3
            assert data["total"] == 10
            assert "Immediately call everyrow_progress" in data["instructions"]
        finally:
            _active_tasks.pop(task_id, None)

    @pytest.mark.asyncio
    async def test_progress_completed_task(self):
        """Test progress returns completion instructions when done."""
        task_id = str(uuid4())
        mock_client = _make_mock_client()

        import time
        _active_tasks[task_id] = {
            "client": mock_client,
            "started_at": time.monotonic(),
            "session": MagicMock(),
            "session_ctx": MagicMock(),
            "total": 5,
            "session_url": "https://everyrow.io/sessions/test",
            "input_csv": "/tmp/test.csv",
            "prefix": "agent",
        }

        mock_status = MagicMock()
        mock_status.status = MagicMock(value="completed")
        mock_status.error = None
        mock_status.additional_properties = {
            "progress": {"pending": 0, "running": 0, "completed": 5, "failed": 0, "total": 5}
        }

        try:
            with (
                patch("everyrow_mcp.server.get_task_status_tasks_task_id_status_get.asyncio", new_callable=AsyncMock, return_value=mock_status),
                patch("everyrow_mcp.server.asyncio.sleep", new_callable=AsyncMock),
            ):
                params = ProgressInput(task_id=task_id)
                result = await everyrow_progress(params)

            data = json.loads(result)
            assert data["status"] == "completed"
            assert data["completed"] == 5
            assert "everyrow_results" in data["instructions"]
        finally:
            _active_tasks.pop(task_id, None)


class TestResults:
    """Tests for everyrow_results."""

    @pytest.mark.asyncio
    async def test_results_unknown_task(self, tmp_path: Path):
        """Test results with unknown task_id."""
        params = ResultsInput(task_id="nonexistent-id", output_path=str(tmp_path))
        result = await everyrow_results(params)
        data = json.loads(result)
        assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_results_saves_csv(self, companies_csv: str, tmp_path: Path):
        """Test results retrieves data and saves to CSV."""
        task_id = str(uuid4())
        mock_client = _make_mock_client()
        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        import time
        _active_tasks[task_id] = {
            "client": mock_client,
            "started_at": time.monotonic(),
            "session": MagicMock(),
            "session_ctx": mock_session_ctx,
            "total": 3,
            "session_url": "https://everyrow.io/sessions/test",
            "input_csv": companies_csv,
            "prefix": "agent",
        }

        # Mock result response with additional_properties data
        mock_item1 = MagicMock()
        mock_item1.additional_properties = {"name": "TechStart", "answer": "Series A"}
        mock_item2 = MagicMock()
        mock_item2.additional_properties = {"name": "AILabs", "answer": "Seed"}

        mock_result = MagicMock()
        mock_result.data = [mock_item1, mock_item2]

        with patch("everyrow_mcp.server.get_task_result_tasks_task_id_result_get.asyncio", new_callable=AsyncMock, return_value=mock_result):
            params = ResultsInput(task_id=task_id, output_path=str(tmp_path))
            result = await everyrow_results(params)

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["rows"] == 2
        assert "agent_companies.csv" in data["output_file"]

        # Verify CSV was written
        output_df = pd.read_csv(data["output_file"])
        assert len(output_df) == 2
        assert list(output_df.columns) == ["name", "answer"]

        # Verify task was cleaned up
        assert task_id not in _active_tasks
