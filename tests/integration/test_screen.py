"""Integration tests for screen operation."""

import pandas as pd
import pytest
from pydantic import BaseModel, Field

from everyrow.ops import screen
from everyrow.result import TableResult

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_screen_returns_table_with_passes_field(session):
    """Test that screen returns a TableResult with passes boolean."""
    input_df = pd.DataFrame(
        [
            {"item": "Water", "category": "Beverage"},
            {"item": "Apple juice", "category": "Beverage"},
        ]
    )

    result = await screen(
        task="Screen items for safety as food/drink. Pass only items safe for human consumption.",
        input=input_df,
        session=session,
    )

    assert isinstance(result, TableResult)
    assert result.artifact_id is not None
    assert "passes" in result.data.columns
    # Both safe items should pass and be in the result
    assert len(result.data) == 2
    assert result.data["passes"].all()  # pyright: ignore[reportGeneralTypeIssues]


async def test_screen_filters_out_failing_items(session):
    """Test that screen filters out items that don't pass."""
    input_df = pd.DataFrame(
        [
            {"item": "Water", "category": "Beverage"},
            {"item": "Arsenic", "category": "Chemical"},
            {"item": "Apple juice", "category": "Beverage"},
        ]
    )

    result = await screen(
        task="Screen items for safety as food/drink. Pass only items safe for human consumption. Arsenic is toxic and must fail.",
        input=input_df,
        session=session,
    )

    assert isinstance(result, TableResult)
    # Arsenic should be filtered out (screen returns only passing rows)
    items_in_result = result.data["item"].tolist()
    assert "Arsenic" not in items_in_result

    # Safe items should be present
    assert "Water" in items_in_result
    assert "Apple juice" in items_in_result

    # All returned rows should have passes=True
    assert result.data["passes"].all()  # pyright: ignore[reportGeneralTypeIssues]


async def test_screen_with_custom_response_model(session):
    """Test screen with a custom response model adds fields to research."""

    class SafetyAssessment(BaseModel):
        passes: bool = Field(description="Whether item is safe")
        safety_rating: str = Field(
            description="Safety rating: Safe, Caution, or Dangerous"
        )
        reason: str = Field(description="Brief reason for the rating")

    input_df = pd.DataFrame(
        [
            {"item": "Milk", "category": "Dairy"},
        ]
    )

    result = await screen(
        task="Assess safety for human consumption. Milk is safe.",
        input=input_df,
        response_model=SafetyAssessment,
        session=session,
    )

    assert isinstance(result, TableResult)
    assert "passes" in result.data.columns
    # Milk should pass and be in the result
    assert len(result.data) == 1
    assert result.data["passes"].iloc[0] == True  # noqa: E712
