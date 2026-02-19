"""Tests for the MCP server tools.

These tests mock the everyrow SDK operations to test the MCP tool logic
without making actual API calls.
"""

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pandas as pd
import pytest
from everyrow.generated.models.public_task_type import PublicTaskType
from everyrow.generated.models.task_progress_info import TaskProgressInfo
from everyrow.generated.models.task_result_response import TaskResultResponse
from everyrow.generated.models.task_result_response_data_type_0_item import (
    TaskResultResponseDataType0Item,
)
from everyrow.generated.models.task_result_response_data_type_1 import (
    TaskResultResponseDataType1,
)
from everyrow.generated.models.task_status import TaskStatus
from everyrow.generated.models.task_status_response import TaskStatusResponse
from pydantic import ValidationError

from everyrow_mcp.models import (
    AgentInput,
    DedupeInput,
    MergeInput,
    ProgressInput,
    RankInput,
    ResultsInput,
    ScreenInput,
    SingleAgentInput,
    _schema_to_model,
)
from everyrow_mcp.state import state
from everyrow_mcp.tools import (
    everyrow_agent,
    everyrow_progress,
    everyrow_results,
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


class TestInputValidation:
    """Tests for input validation."""

    def test_screen_input_validates_csv_path(self, tmp_path: Path):
        """Test ScreenInput validates CSV path."""
        with pytest.raises(ValueError, match="does not exist"):
            ScreenInput(
                task="test",
                input_csv=str(tmp_path / "nonexistent.csv"),
            )

    def test_rank_input_validates_field_type(self, tmp_path: Path):
        """Test RankInput validates field_type."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        with pytest.raises(ValidationError, match="Input should be"):
            RankInput(
                task="test",
                input_csv=str(csv_file),
                field_name="score",
                field_type="invalid",  # pyright: ignore[reportArgumentType]
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
            )


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


def _make_async_context_manager(return_value):
    """Create a mock async context manager that yields return_value."""

    @asynccontextmanager
    async def mock_ctx():
        yield return_value

    return mock_ctx()


def _make_task_status_response(
    *,
    task_id: UUID | None = None,
    session_id: UUID | None = None,
    status: str = "running",
    completed: int = 0,
    failed: int = 0,
    running: int = 0,
    pending: int = 0,
    total: int = 10,
) -> TaskStatusResponse:
    """Create a real TaskStatusResponse for testing."""
    return TaskStatusResponse(
        task_id=task_id or uuid4(),
        session_id=session_id or uuid4(),
        status=TaskStatus(status),
        task_type=PublicTaskType.AGENT,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        progress=TaskProgressInfo(
            pending=pending,
            running=running,
            completed=completed,
            failed=failed,
            total=total,
        ),
    )


def _make_task_result_response(
    data: list[dict],
    *,
    task_id: UUID | None = None,
) -> TaskResultResponse:
    """Create a real TaskResultResponse for testing."""
    items = [TaskResultResponseDataType0Item.from_dict(d) for d in data]
    return TaskResultResponse(
        task_id=task_id or uuid4(),
        status=TaskStatus.COMPLETED,
        data=items,
    )


class TestAgent:
    """Tests for everyrow_agent."""

    @pytest.mark.asyncio
    async def test_submit_returns_task_id(self, companies_csv: str):
        """Test that submit returns immediately with task_id and session_url."""
        mock_task = _make_mock_task()
        mock_session = _make_mock_session()
        mock_client = _make_mock_client()

        with (
            patch(
                "everyrow_mcp.tools.agent_map_async", new_callable=AsyncMock
            ) as mock_op,
            patch.object(state, "client", mock_client),
            patch(
                "everyrow_mcp.tools.create_session",
                return_value=_make_async_context_manager(mock_session),
            ),
        ):
            mock_op.return_value = mock_task

            params = AgentInput(
                task="Find HQ for each company",
                input_csv=companies_csv,
            )
            result = await everyrow_agent(params)

            # First TextContent is JSON for MCP App session UI
            ui_data = json.loads(result[0].text)
            assert ui_data["task_id"] == str(mock_task.task_id)
            assert "session_url" in ui_data
            assert ui_data["status"] == "submitted"

            # Second TextContent is human-readable instructions
            text = result[1].text
            assert str(mock_task.task_id) in text
            assert "Session:" in text
            assert "everyrow_progress" in text


class TestProgress:
    """Tests for everyrow_progress."""

    @pytest.mark.asyncio
    async def test_progress_api_error(self):
        """Test progress with API error returns helpful message."""
        mock_client = _make_mock_client()
        task_id = str(uuid4())

        with (
            patch.object(state, "client", mock_client),
            patch(
                "everyrow_mcp.tools.get_task_status_tasks_task_id_status_get.asyncio",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API error"),
            ),
            patch("everyrow_mcp.tools.asyncio.sleep", new_callable=AsyncMock),
        ):
            params = ProgressInput(task_id=task_id)
            result = await everyrow_progress(params)

        assert json.loads(result[0].text)["status"] == "error"
        assert "Error polling task" in result[1].text
        assert "Retry:" in result[1].text

    @pytest.mark.asyncio
    async def test_progress_running_task(self):
        """Test progress returns status counts for a running task."""
        task_id = str(uuid4())
        mock_client = _make_mock_client()
        status_response = _make_task_status_response(
            status="running",
            completed=4,
            failed=1,
            running=3,
            pending=2,
            total=10,
        )

        with (
            patch.object(state, "client", mock_client),
            patch(
                "everyrow_mcp.tools.get_task_status_tasks_task_id_status_get.asyncio",
                new_callable=AsyncMock,
                return_value=status_response,
            ),
            patch("everyrow_mcp.tools.asyncio.sleep", new_callable=AsyncMock),
            patch("everyrow_mcp.tools._write_task_state"),
        ):
            params = ProgressInput(task_id=task_id)
            result = await everyrow_progress(params)

        # First TextContent is JSON for MCP App progress UI
        ui_data = json.loads(result[0].text)
        assert ui_data["completed"] == 4
        assert ui_data["total"] == 10
        assert ui_data["failed"] == 1
        assert ui_data["running"] == 3
        assert ui_data["status"] == "running"

        # Second TextContent is human-readable
        text = result[1].text
        assert "4/10 complete" in text
        assert "1 failed" in text
        assert "3 running" in text
        assert "everyrow_progress" in text

    @pytest.mark.asyncio
    async def test_progress_completed_task(self):
        """Test progress returns completion instructions when done."""
        task_id = str(uuid4())
        mock_client = _make_mock_client()
        status_response = _make_task_status_response(
            status="completed",
            completed=5,
            failed=0,
            running=0,
            pending=0,
            total=5,
        )

        with (
            patch.object(state, "client", mock_client),
            patch(
                "everyrow_mcp.tools.get_task_status_tasks_task_id_status_get.asyncio",
                new_callable=AsyncMock,
                return_value=status_response,
            ),
            patch("everyrow_mcp.tools.asyncio.sleep", new_callable=AsyncMock),
            patch("everyrow_mcp.tools._write_task_state"),
        ):
            params = ProgressInput(task_id=task_id)
            result = await everyrow_progress(params)

        # First TextContent is JSON for MCP App progress UI
        ui_data = json.loads(result[0].text)
        assert ui_data["status"] == "completed"
        assert ui_data["completed"] == 5
        assert ui_data["total"] == 5

        # Second TextContent is human-readable
        text = result[1].text
        assert "Completed: 5/5" in text
        assert "everyrow_results" in text


class TestResults:
    """Tests for everyrow_results."""

    @pytest.mark.asyncio
    async def test_results_api_error(self, tmp_path: Path):
        """Test results with API error returns helpful message."""
        mock_client = _make_mock_client()
        task_id = str(uuid4())
        output_file = tmp_path / "output.csv"

        with (
            patch.object(state, "client", mock_client),
            patch(
                "everyrow_mcp.tools.get_task_status_tasks_task_id_status_get.asyncio",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API error"),
            ),
        ):
            params = ResultsInput(task_id=task_id, output_path=str(output_file))
            result = await everyrow_results(params)

        assert "Error retrieving results" in result[0].text

    @pytest.mark.asyncio
    async def test_results_saves_csv(self, tmp_path: Path):
        """Test results retrieves data and saves to CSV."""
        task_id = str(uuid4())
        mock_client = _make_mock_client()
        output_file = tmp_path / "output.csv"

        status_response = _make_task_status_response(status="completed")
        result_response = _make_task_result_response(
            [
                {"name": "TechStart", "answer": "Series A"},
                {"name": "AILabs", "answer": "Seed"},
            ]
        )

        with (
            patch.object(state, "client", mock_client),
            patch(
                "everyrow_mcp.tools.get_task_status_tasks_task_id_status_get.asyncio",
                new_callable=AsyncMock,
                return_value=status_response,
            ),
            patch(
                "everyrow_mcp.tools.get_task_result_tasks_task_id_result_get.asyncio",
                new_callable=AsyncMock,
                return_value=result_response,
            ),
        ):
            params = ResultsInput(task_id=task_id, output_path=str(output_file))
            result = await everyrow_results(params)
        text = result[0].text

        assert "Saved 2 rows to" in text
        assert "output.csv" in text

        # Verify CSV was written
        output_df = pd.read_csv(output_file)
        assert len(output_df) == 2
        assert list(output_df.columns) == ["name", "answer"]

    @pytest.mark.asyncio
    async def test_results_inline_without_output_path(self):
        """Test results returns a page of JSON records + compact summary."""
        task_id = str(uuid4())
        mock_client = _make_mock_client()

        status_response = _make_task_status_response(status="completed")
        result_response = _make_task_result_response(
            [
                {"name": "TechStart", "answer": "Series A"},
                {"name": "AILabs", "answer": "Seed"},
            ]
        )

        state.result_cache.pop(task_id, None)
        with (
            patch.object(state, "client", mock_client),
            patch(
                "everyrow_mcp.tools.get_task_status_tasks_task_id_status_get.asyncio",
                new_callable=AsyncMock,
                return_value=status_response,
            ),
            patch(
                "everyrow_mcp.tools.get_task_result_tasks_task_id_result_get.asyncio",
                new_callable=AsyncMock,
                return_value=result_response,
            ),
        ):
            params = ResultsInput(task_id=task_id)
            result = await everyrow_results(params)

        # First TextContent is JSON records for MCP App results table
        records = json.loads(result[0].text)
        assert len(records) == 2
        assert records[0]["name"] == "TechStart"
        assert records[1]["name"] == "AILabs"
        assert records[0]["answer"] == "Series A"

        # Second TextContent is compact summary for LLM
        assert "2 rows" in result[1].text
        assert "2 columns" in result[1].text
        assert "All rows shown" in result[1].text

        # Clean up cache
        state.result_cache.pop(task_id, None)

    @pytest.mark.asyncio
    async def test_results_pagination_with_offset(self):
        """Test paginated results return correct page with next offset hint."""
        task_id = str(uuid4())
        mock_client = _make_mock_client()

        num_rows = state.settings.preview_size + 3
        data = [{"id": i, "val": f"row_{i}"} for i in range(num_rows)]
        status_response = _make_task_status_response(status="completed")
        result_response = _make_task_result_response(data)

        state.result_cache.pop(task_id, None)
        with (
            patch.object(state, "client", mock_client),
            patch(
                "everyrow_mcp.tools.get_task_status_tasks_task_id_status_get.asyncio",
                new_callable=AsyncMock,
                return_value=status_response,
            ),
            patch(
                "everyrow_mcp.tools.get_task_result_tasks_task_id_result_get.asyncio",
                new_callable=AsyncMock,
                return_value=result_response,
            ),
        ):
            # First page (offset=0)
            result = await everyrow_results(ResultsInput(task_id=task_id))

        records = json.loads(result[0].text)
        assert len(records) == state.settings.preview_size
        assert records[0]["id"] == 0

        summary = result[1].text
        assert f"{num_rows} rows" in summary
        assert f"offset={state.settings.preview_size}" in summary

        # Second page (offset=state.settings.preview_size) — hits cache
        with patch.object(state, "client", mock_client):
            result2 = await everyrow_results(
                ResultsInput(task_id=task_id, offset=state.settings.preview_size)
            )

        records2 = json.loads(result2[0].text)
        assert len(records2) == 3
        assert records2[0]["id"] == state.settings.preview_size

        assert "final page" in result2[1].text

        state.result_cache.pop(task_id, None)

    @pytest.mark.asyncio
    async def test_results_custom_page_size(self):
        """Test that page_size parameter controls how many rows are returned."""
        task_id = str(uuid4())
        mock_client = _make_mock_client()

        data = [{"id": i, "val": f"row_{i}"} for i in range(20)]
        status_response = _make_task_status_response(status="completed")
        result_response = _make_task_result_response(data)

        state.result_cache.pop(task_id, None)
        with (
            patch.object(state, "client", mock_client),
            patch(
                "everyrow_mcp.tools.get_task_status_tasks_task_id_status_get.asyncio",
                new_callable=AsyncMock,
                return_value=status_response,
            ),
            patch(
                "everyrow_mcp.tools.get_task_result_tasks_task_id_result_get.asyncio",
                new_callable=AsyncMock,
                return_value=result_response,
            ),
        ):
            # Request page_size=10
            result = await everyrow_results(ResultsInput(task_id=task_id, page_size=10))

        records = json.loads(result[0].text)
        assert len(records) == 10
        assert records[0]["id"] == 0
        assert records[9]["id"] == 9

        summary = result[1].text
        assert "20 rows" in summary
        assert "offset=10" in summary
        assert "page_size=10" in summary  # non-default page_size appears in hint

        # Second page
        with patch.object(state, "client", mock_client):
            result2 = await everyrow_results(
                ResultsInput(task_id=task_id, offset=10, page_size=10)
            )

        records2 = json.loads(result2[0].text)
        assert len(records2) == 10
        assert records2[0]["id"] == 10
        assert "final page" in result2[1].text

        state.result_cache.pop(task_id, None)

    @pytest.mark.asyncio
    async def test_results_cache_reuse(self):
        """Test that second call uses cache and doesn't re-fetch from API."""
        task_id = str(uuid4())
        mock_client = _make_mock_client()

        status_response = _make_task_status_response(status="completed")
        result_response = _make_task_result_response(
            [{"name": "A", "val": "1"}, {"name": "B", "val": "2"}]
        )

        state.result_cache.pop(task_id, None)
        mock_status = AsyncMock(return_value=status_response)
        mock_result = AsyncMock(return_value=result_response)

        with (
            patch.object(state, "client", mock_client),
            patch(
                "everyrow_mcp.tools.get_task_status_tasks_task_id_status_get.asyncio",
                mock_status,
            ),
            patch(
                "everyrow_mcp.tools.get_task_result_tasks_task_id_result_get.asyncio",
                mock_result,
            ),
        ):
            # First call fetches from API
            await everyrow_results(ResultsInput(task_id=task_id))
            assert mock_status.call_count == 1
            assert mock_result.call_count == 1

            # Second call uses cache — no additional API calls
            await everyrow_results(ResultsInput(task_id=task_id))
            assert mock_status.call_count == 1
            assert mock_result.call_count == 1

        state.result_cache.pop(task_id, None)

    @pytest.mark.asyncio
    async def test_results_scalar_single_agent(self):
        """Test results handles scalar (single_agent) TaskResultResponseDataType1."""
        task_id = str(uuid4())
        mock_client = _make_mock_client()

        status_response = _make_task_status_response(status="completed")
        scalar_data = TaskResultResponseDataType1.from_dict(
            {"ceo": "Tim Cook", "company": "Apple"}
        )
        result_response = TaskResultResponse(
            task_id=uuid4(),
            status=TaskStatus.COMPLETED,
            data=scalar_data,
        )

        state.result_cache.pop(task_id, None)
        with (
            patch.object(state, "client", mock_client),
            patch(
                "everyrow_mcp.tools.get_task_status_tasks_task_id_status_get.asyncio",
                new_callable=AsyncMock,
                return_value=status_response,
            ),
            patch(
                "everyrow_mcp.tools.get_task_result_tasks_task_id_result_get.asyncio",
                new_callable=AsyncMock,
                return_value=result_response,
            ),
        ):
            params = ResultsInput(task_id=task_id)
            result = await everyrow_results(params)

        records = json.loads(result[0].text)
        assert len(records) == 1
        assert records[0]["ceo"] == "Tim Cook"
        assert records[0]["company"] == "Apple"

        assert "1 rows" in result[1].text

        state.result_cache.pop(task_id, None)

    @pytest.mark.asyncio
    async def test_results_http_mode_returns_download_url(self):
        """In HTTP mode, large results return a signed download URL."""
        task_id = str(uuid4())
        mock_client = _make_mock_client()

        num_rows = state.settings.preview_size + 5
        data = [{"id": i, "val": f"row_{i}"} for i in range(num_rows)]
        status_response = _make_task_status_response(status="completed")
        result_response = _make_task_result_response(data)

        state.result_cache.pop(task_id, None)
        with (
            patch("everyrow_mcp.tools._get_client", return_value=mock_client),
            patch.object(state, "transport", "streamable-http"),
            patch.object(state, "mcp_server_url", "https://mcp.example.com"),
            patch(
                "everyrow_mcp.tools.get_task_status_tasks_task_id_status_get.asyncio",
                new_callable=AsyncMock,
                return_value=status_response,
            ),
            patch(
                "everyrow_mcp.tools.get_task_result_tasks_task_id_result_get.asyncio",
                new_callable=AsyncMock,
                return_value=result_response,
            ),
        ):
            params = ResultsInput(task_id=task_id)
            result = await everyrow_results(params)

        # TextContent 1: widget metadata with results_url and preview
        widget_data = json.loads(result[0].text)
        assert "results_url" in widget_data
        assert "https://mcp.example.com/api/results/" in widget_data["results_url"]
        assert widget_data["total"] == num_rows
        assert len(widget_data["preview"]) == state.settings.preview_size

        # TextContent 2: summary with download URL
        summary = result[1].text
        assert f"{num_rows} rows" in summary
        assert "download" in summary.lower()

        state.result_cache.pop(task_id, None)

    @pytest.mark.asyncio
    async def test_results_http_mode_small_data_no_download_url(self):
        """In HTTP mode, small results (all fit in preview) don't include download URL."""
        task_id = str(uuid4())
        mock_client = _make_mock_client()

        data = [{"name": "A"}, {"name": "B"}]
        status_response = _make_task_status_response(status="completed")
        result_response = _make_task_result_response(data)

        state.result_cache.pop(task_id, None)
        with (
            patch("everyrow_mcp.tools._get_client", return_value=mock_client),
            patch.object(state, "transport", "streamable-http"),
            patch.object(state, "mcp_server_url", "https://mcp.example.com"),
            patch(
                "everyrow_mcp.tools.get_task_status_tasks_task_id_status_get.asyncio",
                new_callable=AsyncMock,
                return_value=status_response,
            ),
            patch(
                "everyrow_mcp.tools.get_task_result_tasks_task_id_result_get.asyncio",
                new_callable=AsyncMock,
                return_value=result_response,
            ),
        ):
            params = ResultsInput(task_id=task_id)
            result = await everyrow_results(params)

        # Widget metadata still has results_url
        widget_data = json.loads(result[0].text)
        assert "results_url" in widget_data
        assert widget_data["total"] == 2

        # Summary says all rows shown, no download URL
        summary = result[1].text
        assert "All rows shown" in summary
        assert "download" not in summary.lower()

        state.result_cache.pop(task_id, None)


class TestAgentInlineInput:
    """Tests for everyrow_agent with inline CSV data."""

    @pytest.mark.asyncio
    async def test_submit_with_inline_data(self):
        """Test agent submission with input_data instead of input_csv."""
        mock_task = _make_mock_task()
        mock_session = _make_mock_session()
        mock_client = _make_mock_client()

        with (
            patch(
                "everyrow_mcp.tools.agent_map_async", new_callable=AsyncMock
            ) as mock_op,
            patch.object(state, "client", mock_client),
            patch(
                "everyrow_mcp.tools.create_session",
                return_value=_make_async_context_manager(mock_session),
            ),
        ):
            mock_op.return_value = mock_task

            params = AgentInput(
                task="Find HQ for each company",
                input_data="name,industry\nTechStart,Software\nAILabs,AI\n",
            )
            result = await everyrow_agent(params)

            # First TextContent is JSON for MCP App session UI
            ui_data = json.loads(result[0].text)
            assert ui_data["task_id"] == str(mock_task.task_id)
            assert ui_data["total"] == 2

            # Second TextContent is human-readable
            text = result[1].text
            assert str(mock_task.task_id) in text
            assert "2 agents starting" in text

            # Verify the DataFrame passed to the SDK had 2 rows
            call_kwargs = mock_op.call_args[1]
            assert len(call_kwargs["input"]) == 2


class TestAgentInputValidation:
    """Tests for AgentInput model validation with inline data."""

    def test_requires_one_input_source(self):
        """Test that no input source raises."""
        with pytest.raises(ValidationError, match="Provide one of"):
            AgentInput(task="test")

    def test_rejects_both_input_sources(self, companies_csv: str):
        """Test that providing both raises."""
        with pytest.raises(ValidationError, match="only one"):
            AgentInput(
                task="test",
                input_csv=companies_csv,
                input_data="name,industry\nA,B\n",
            )

    def test_accepts_input_csv(self, companies_csv: str):
        """Test that input_csv alone is valid."""
        params = AgentInput(task="test", input_csv=companies_csv)
        assert params.input_csv == companies_csv
        assert params.input_data is None

    def test_accepts_input_data(self):
        """Test that input_data alone is valid."""
        params = AgentInput(task="test", input_data="a,b\n1,2\n")
        assert params.input_data is not None
        assert params.input_csv is None

    def test_accepts_input_json(self):
        """Test that input_json alone is valid."""
        data = [
            {"company": "Acme", "url": "acme.com"},
            {"company": "Beta", "url": "beta.io"},
        ]
        params = AgentInput(task="test", input_json=data)
        assert params.input_json == data
        assert params.input_csv is None
        assert params.input_data is None

    def test_rejects_input_json_with_csv(self, companies_csv: str):
        """Test that input_json + input_csv raises."""
        with pytest.raises(ValidationError, match="only one"):
            AgentInput(
                task="test",
                input_csv=companies_csv,
                input_json=[{"a": 1}],
            )


class TestResultsInputValidation:
    """Tests for ResultsInput with optional output_path."""

    def test_output_path_optional(self):
        """Test that output_path can be omitted."""
        params = ResultsInput(task_id="some-id")
        assert params.output_path is None

    def test_output_path_still_validated(self, tmp_path: Path):
        """Test that output_path is validated when provided."""
        params = ResultsInput(task_id="some-id", output_path=str(tmp_path / "out.csv"))
        assert params.output_path is not None

    def test_output_path_rejects_non_csv(self, tmp_path: Path):
        """Test that non-CSV output_path is rejected."""
        with pytest.raises(ValidationError, match=r"must end in \.csv"):
            ResultsInput(task_id="some-id", output_path=str(tmp_path / "out.txt"))


class TestInputModelsUnchanged:
    """Verify that input models require an input source."""

    def test_rank_requires_input_source(self):
        """RankInput requires either input_csv or input_data."""
        with pytest.raises(ValidationError):
            RankInput(task="test", field_name="score")

    def test_rank_accepts_input_data(self):
        """RankInput accepts input_data as alternative to input_csv."""
        params = RankInput(task="test", field_name="score", input_data="col\nval")
        assert params.input_data == "col\nval"
        assert params.input_csv is None

    def test_rank_rejects_both_inputs(self):
        """RankInput rejects both input_csv and input_data."""
        with pytest.raises(ValidationError):
            RankInput(
                task="test",
                field_name="score",
                input_csv="/tmp/test.csv",
                input_data="col\nval",
            )

    def test_screen_requires_input_source(self):
        """ScreenInput requires either input_csv or input_data."""
        with pytest.raises(ValidationError):
            ScreenInput(task="test")

    def test_screen_accepts_input_data(self):
        """ScreenInput accepts input_data as alternative to input_csv."""
        params = ScreenInput(task="test", input_data="col\nval")
        assert params.input_data == "col\nval"
        assert params.input_csv is None

    def test_screen_rejects_both_inputs(self):
        """ScreenInput rejects both input_csv and input_data."""
        with pytest.raises(ValidationError):
            ScreenInput(task="test", input_csv="/tmp/test.csv", input_data="col\nval")

    def test_dedupe_requires_input_csv(self):
        """DedupeInput still requires input_csv as a string."""
        with pytest.raises(ValidationError):
            DedupeInput(equivalence_relation="same entity")

    def test_merge_requires_csv_paths(self):
        """MergeInput still requires left_csv and right_csv."""
        with pytest.raises(ValidationError):
            MergeInput(task="test")

    def test_single_agent_requires_task(self):
        """SingleAgentInput requires a task."""
        with pytest.raises(ValidationError):
            SingleAgentInput()

    def test_single_agent_accepts_no_input(self):
        """SingleAgentInput works with just a task (no input_data)."""
        params = SingleAgentInput(task="Find the CEO of Apple")
        assert params.input_data is None

    def test_single_agent_accepts_input_data(self):
        """SingleAgentInput accepts context as key-value pairs."""
        params = SingleAgentInput(
            task="Find funding info",
            input_data={"company": "Stripe", "url": "stripe.com"},
        )
        assert params.input_data == {"company": "Stripe", "url": "stripe.com"}
