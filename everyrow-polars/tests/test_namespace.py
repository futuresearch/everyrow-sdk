"""Tests for the everyrow Polars namespace."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pandas as pd
import polars as pl
import pytest
from pydantic import BaseModel

import everyrow_polars  # noqa: F401 - registers namespace


class MockTableResult:
    """Mock TableResult from everyrow SDK."""

    def __init__(self, data: pd.DataFrame):
        self.artifact_id = uuid4()
        self.data = data
        self.error = None


class MockMergeResult:
    """Mock MergeResult from everyrow SDK."""

    def __init__(self, data: pd.DataFrame):
        self.artifact_id = uuid4()
        self.data = data
        self.error = None
        self.breakdown = None


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a sample Polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "company": ["Acme Inc", "Beta Corp", "Gamma LLC"],
            "role": ["Engineer", "Manager", "Director"],
        }
    )


class TestNamespaceRegistration:
    """Test that the namespace is properly registered."""

    def test_namespace_exists(self, sample_df: pl.DataFrame):
        """The everyrow namespace should be accessible on DataFrames."""
        assert hasattr(sample_df, "everyrow")

    def test_namespace_has_methods(self, sample_df: pl.DataFrame):
        """The namespace should have all expected methods."""
        ns = sample_df.everyrow
        assert hasattr(ns, "screen")
        assert hasattr(ns, "screen_async")
        assert hasattr(ns, "rank")
        assert hasattr(ns, "rank_async")
        assert hasattr(ns, "dedupe")
        assert hasattr(ns, "dedupe_async")
        assert hasattr(ns, "merge")
        assert hasattr(ns, "merge_async")
        assert hasattr(ns, "research")
        assert hasattr(ns, "research_async")


class TestScreen:
    """Tests for the screen operation."""

    @pytest.mark.asyncio
    async def test_screen_async(self, sample_df: pl.DataFrame):
        """screen_async should call the SDK and return a Polars DataFrame."""
        # Mock result with passes column added
        result_pdf = sample_df.to_pandas()
        result_pdf["passes"] = [True, False, True]

        with patch("everyrow_polars.namespace.screen", new_callable=AsyncMock) as mock:
            mock.return_value = MockTableResult(result_pdf)

            result = await sample_df.everyrow.screen_async("test task")

            assert isinstance(result, pl.DataFrame)
            assert "passes" in result.columns
            assert result.shape[0] == 3
            mock.assert_called_once()

    def test_screen_sync(self, sample_df: pl.DataFrame):
        """screen should work synchronously."""
        result_pdf = sample_df.to_pandas()
        result_pdf["passes"] = [True, False, True]

        with patch("everyrow_polars.namespace.screen", new_callable=AsyncMock) as mock:
            mock.return_value = MockTableResult(result_pdf)

            result = sample_df.everyrow.screen("test task")

            assert isinstance(result, pl.DataFrame)
            assert "passes" in result.columns

    @pytest.mark.asyncio
    async def test_screen_with_response_model(self, sample_df: pl.DataFrame):
        """screen should accept a response_model."""

        class CustomResult(BaseModel):
            passes: bool
            confidence: float

        result_pdf = sample_df.to_pandas()
        result_pdf["passes"] = [True, False, True]
        result_pdf["confidence"] = [0.9, 0.3, 0.85]

        with patch("everyrow_polars.namespace.screen", new_callable=AsyncMock) as mock:
            mock.return_value = MockTableResult(result_pdf)

            result = await sample_df.everyrow.screen_async(
                "test task", response_model=CustomResult
            )

            assert "passes" in result.columns
            assert "confidence" in result.columns
            # Verify response_model was passed
            call_kwargs = mock.call_args.kwargs
            assert call_kwargs["response_model"] == CustomResult


class TestRank:
    """Tests for the rank operation."""

    @pytest.mark.asyncio
    async def test_rank_async(self, sample_df: pl.DataFrame):
        """rank_async should add a score column and sort."""
        result_pdf = sample_df.to_pandas()
        result_pdf["score"] = [0.8, 0.3, 0.95]

        with patch("everyrow_polars.namespace.rank", new_callable=AsyncMock) as mock:
            mock.return_value = MockTableResult(result_pdf)

            result = await sample_df.everyrow.rank_async("test task")

            assert isinstance(result, pl.DataFrame)
            assert "score" in result.columns
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_rank_custom_field_name(self, sample_df: pl.DataFrame):
        """rank should use custom field_name."""
        result_pdf = sample_df.to_pandas()
        result_pdf["priority"] = [1, 2, 3]

        with patch("everyrow_polars.namespace.rank", new_callable=AsyncMock) as mock:
            mock.return_value = MockTableResult(result_pdf)

            await sample_df.everyrow.rank_async(
                "test task", field_name="priority", field_type="int"
            )

            call_kwargs = mock.call_args.kwargs
            assert call_kwargs["field_name"] == "priority"
            assert call_kwargs["field_type"] == "int"

    @pytest.mark.asyncio
    async def test_rank_descending(self, sample_df: pl.DataFrame):
        """rank with descending=True should pass ascending_order=False to SDK."""
        result_pdf = sample_df.to_pandas()
        result_pdf["score"] = [0.95, 0.8, 0.3]

        with patch("everyrow_polars.namespace.rank", new_callable=AsyncMock) as mock:
            mock.return_value = MockTableResult(result_pdf)

            await sample_df.everyrow.rank_async("test task", descending=True)

            call_kwargs = mock.call_args.kwargs
            assert call_kwargs["ascending_order"] is False


class TestDedupe:
    """Tests for the dedupe operation."""

    @pytest.mark.asyncio
    async def test_dedupe_async(self, sample_df: pl.DataFrame):
        """dedupe_async should add equivalence_class_id and selected columns."""
        result_pdf = sample_df.to_pandas()
        result_pdf["equivalence_class_id"] = [1, 1, 2]
        result_pdf["selected"] = [True, False, True]

        with patch("everyrow_polars.namespace.dedupe", new_callable=AsyncMock) as mock:
            mock.return_value = MockTableResult(result_pdf)

            result = await sample_df.everyrow.dedupe_async("same person")

            assert isinstance(result, pl.DataFrame)
            assert "equivalence_class_id" in result.columns
            assert "selected" in result.columns
            mock.assert_called_once()


class TestMerge:
    """Tests for the merge operation."""

    @pytest.mark.asyncio
    async def test_merge_async(self, sample_df: pl.DataFrame):
        """merge_async should join two DataFrames."""
        right_df = pl.DataFrame(
            {
                "company_name": ["Acme", "Beta"],
                "industry": ["Tech", "Finance"],
            }
        )

        # Merged result
        result_pdf = pd.DataFrame(
            {
                "name": ["Alice", "Bob"],
                "company": ["Acme Inc", "Beta Corp"],
                "role": ["Engineer", "Manager"],
                "company_name": ["Acme", "Beta"],
                "industry": ["Tech", "Finance"],
            }
        )

        with patch("everyrow_polars.namespace.merge", new_callable=AsyncMock) as mock:
            mock.return_value = MockMergeResult(result_pdf)

            result = await sample_df.everyrow.merge_async(
                right_df,
                "Match company to company_name",
                left_on="company",
                right_on="company_name",
            )

            assert isinstance(result, pl.DataFrame)
            assert "industry" in result.columns
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_merge_with_web_search(self, sample_df: pl.DataFrame):
        """merge should pass use_web_search parameter."""
        right_df = pl.DataFrame({"x": [1]})
        result_pdf = pd.DataFrame({"x": [1]})

        with patch("everyrow_polars.namespace.merge", new_callable=AsyncMock) as mock:
            mock.return_value = MockMergeResult(result_pdf)

            await sample_df.everyrow.merge_async(right_df, "test", use_web_search="yes")

            call_kwargs = mock.call_args.kwargs
            assert call_kwargs["use_web_search"] == "yes"


class TestResearch:
    """Tests for the research operation."""

    @pytest.mark.asyncio
    async def test_research_async(self, sample_df: pl.DataFrame):
        """research_async should enrich data with web research."""
        result_pdf = sample_df.to_pandas()
        result_pdf["answer"] = ["Founded 2020", "Founded 2015", "Founded 2018"]

        with patch(
            "everyrow_polars.namespace.agent_map", new_callable=AsyncMock
        ) as mock:
            mock.return_value = MockTableResult(result_pdf)

            result = await sample_df.everyrow.research_async("Find founding year")

            assert isinstance(result, pl.DataFrame)
            assert "answer" in result.columns
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_research_effort_level(self, sample_df: pl.DataFrame):
        """research should pass effort_level to agent_map."""
        result_pdf = sample_df.to_pandas()
        result_pdf["answer"] = ["a", "b", "c"]

        with patch(
            "everyrow_polars.namespace.agent_map", new_callable=AsyncMock
        ) as mock:
            mock.return_value = MockTableResult(result_pdf)

            await sample_df.everyrow.research_async("test", effort_level="high")

            call_kwargs = mock.call_args.kwargs
            # EffortLevel.HIGH has value "high"
            assert call_kwargs["effort_level"].value == "high"
