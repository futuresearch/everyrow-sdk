"""Tests for the EveryrowResource."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pandas as pd
import pytest

from dagster_everyrow import EveryrowResource


class MockSession:
    """Mock session for testing."""

    def __init__(self):
        self.session_id = uuid4()

    def get_url(self) -> str:
        return f"https://everyrow.io/sessions/{self.session_id}"


class MockTableResult:
    """Mock TableResult for testing."""

    def __init__(self, data: pd.DataFrame):
        self.artifact_id = uuid4()
        self.data = data
        self.error = None


@pytest.fixture
def resource() -> EveryrowResource:
    """Create a test resource."""
    return EveryrowResource(api_key="test-api-key")


def test_screen_returns_result_with_session_url(
    resource: EveryrowResource, leads_df: pd.DataFrame
):
    """Test screen returns result with session URL."""
    result_df = leads_df.copy()
    result_df["passes"] = [True, False, True]
    result_df["reasoning"] = ["good", "bad", "good"]

    mock_result = MockTableResult(result_df)
    mock_session = MockSession()

    with (
        patch("dagster_everyrow.resource.create_session") as mock_create_session,
        patch(
            "dagster_everyrow.resource.screen", new_callable=AsyncMock
        ) as mock_screen,
    ):
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_session
        mock_cm.__aexit__.return_value = None
        mock_create_session.return_value = mock_cm

        mock_screen.return_value = mock_result

        result = resource.screen(
            task="Remote-friendly senior role",
            input=leads_df,
        )

        assert result.session_url == mock_session.get_url()
        assert len(result.data) == 3
        assert "passes" in result.data.columns

        mock_screen.assert_called_once()
        call_kwargs = mock_screen.call_args.kwargs
        assert call_kwargs["task"] == "Remote-friendly senior role"
        assert call_kwargs["session"] == mock_session


def test_rank_returns_result_with_session_url(
    resource: EveryrowResource, companies_df: pd.DataFrame
):
    """Test rank returns result with session URL."""
    result_df = companies_df.copy()
    result_df["integration_score"] = [85.0, 95.0, 60.0]

    mock_result = MockTableResult(result_df)
    mock_session = MockSession()

    with (
        patch("dagster_everyrow.resource.create_session") as mock_create_session,
        patch("dagster_everyrow.resource.rank", new_callable=AsyncMock) as mock_rank,
    ):
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_session
        mock_cm.__aexit__.return_value = None
        mock_create_session.return_value = mock_cm

        mock_rank.return_value = mock_result

        result = resource.rank(
            task="Score by likelihood to need data integration",
            input=companies_df,
            field_name="integration_score",
            ascending_order=False,
        )

        assert result.session_url == mock_session.get_url()
        assert "integration_score" in result.data.columns

        call_kwargs = mock_rank.call_args.kwargs
        assert call_kwargs["field_name"] == "integration_score"
        assert call_kwargs["ascending_order"] is False


def test_dedupe_returns_result_with_session_url(
    resource: EveryrowResource, contacts_df: pd.DataFrame
):
    """Test dedupe returns result with session URL."""
    result_df = contacts_df.copy()
    result_df["equivalence_class_id"] = [0, 0, 1]
    result_df["selected"] = [True, False, True]

    mock_result = MockTableResult(result_df)
    mock_session = MockSession()

    with (
        patch("dagster_everyrow.resource.create_session") as mock_create_session,
        patch(
            "dagster_everyrow.resource.dedupe", new_callable=AsyncMock
        ) as mock_dedupe,
    ):
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_session
        mock_cm.__aexit__.return_value = None
        mock_create_session.return_value = mock_cm

        mock_dedupe.return_value = mock_result

        result = resource.dedupe(
            equivalence_relation="Same person despite name variations",
            input=contacts_df,
        )

        assert result.session_url == mock_session.get_url()
        assert "equivalence_class_id" in result.data.columns
        assert "selected" in result.data.columns


def test_merge_returns_result_with_session_url(
    resource: EveryrowResource,
    products_df: pd.DataFrame,
    suppliers_df: pd.DataFrame,
):
    """Test merge returns result with session URL."""
    result_df = pd.DataFrame(
        [
            {"product_name": "Photoshop", "company_name": "Adobe Inc"},
            {"product_name": "VSCode", "company_name": "Microsoft Corporation"},
        ]
    )

    mock_result = MockTableResult(result_df)
    mock_session = MockSession()

    with (
        patch("dagster_everyrow.resource.create_session") as mock_create_session,
        patch("dagster_everyrow.resource.merge", new_callable=AsyncMock) as mock_merge,
    ):
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_session
        mock_cm.__aexit__.return_value = None
        mock_create_session.return_value = mock_cm

        mock_merge.return_value = mock_result

        result = resource.merge(
            task="Match software products to parent companies",
            left_table=products_df,
            right_table=suppliers_df,
            merge_on_left="product_name",
            merge_on_right="company_name",
        )

        assert result.session_url == mock_session.get_url()
        assert len(result.data) == 2


def test_research_returns_result_with_session_url(
    resource: EveryrowResource, companies_df: pd.DataFrame
):
    """Test research returns result with session URL."""
    result_df = companies_df.copy()
    result_df["answer"] = [
        "Series A, $10M from Sequoia",
        "Seed, $2M from a16z",
        "Series B, $50M from Accel",
    ]

    mock_result = MockTableResult(result_df)
    mock_session = MockSession()

    with (
        patch("dagster_everyrow.resource.create_session") as mock_create_session,
        patch(
            "dagster_everyrow.resource.agent_map", new_callable=AsyncMock
        ) as mock_agent,
    ):
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_session
        mock_cm.__aexit__.return_value = None
        mock_create_session.return_value = mock_cm

        mock_agent.return_value = mock_result

        result = resource.research(
            task="Find this company's latest funding round",
            input=companies_df,
        )

        assert result.session_url == mock_session.get_url()
        assert "answer" in result.data.columns
