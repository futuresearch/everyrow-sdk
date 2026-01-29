"""Integration tests for dedupe operation."""

import pandas as pd
import pytest

from everyrow.ops import dedupe
from everyrow.result import TableResult

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_dedupe_returns_table_with_equivalence_fields(papers_df):
    """Test that dedupe returns a TableResult with equivalence class fields."""
    result = await dedupe(
        equivalence_relation="""
            Two entries are duplicates if they represent the same research paper.
            ArXiv preprints and published conference versions of the same paper
            are considered duplicates.
        """,
        input=papers_df,
    )

    assert isinstance(result, TableResult)
    assert result.artifact_id is not None
    assert "equivalence_class_id" in result.data.columns
    assert "selected" in result.data.columns


async def test_dedupe_identifies_duplicates(papers_df):
    """Test that dedupe correctly identifies duplicate papers."""
    result = await dedupe(
        equivalence_relation="""
            Two entries are duplicates if they represent the same research paper.
            ArXiv preprints and published conference versions are duplicates.
            "Attention Is All You Need" appears twice - once as NeurIPS and once as arXiv.
        """,
        input=papers_df,
    )

    assert isinstance(result, TableResult)

    # The two "Attention Is All You Need" papers should have same equivalence_class_id
    attention_papers = result.data[
        result.data["title"].str.contains("Attention", case=False)
    ]
    assert len(attention_papers) == 2
    assert attention_papers["equivalence_class_id"].nunique() == 1  # pyright: ignore[reportAttributeAccessIssue]

    # BERT should have a different equivalence class
    bert_papers = result.data[result.data["title"].str.contains("BERT", case=False)]
    attention_class = attention_papers["equivalence_class_id"].iloc[0]  # pyright: ignore[reportAttributeAccessIssue]
    bert_class = bert_papers["equivalence_class_id"].iloc[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert attention_class != bert_class


async def test_dedupe_selects_one_per_class():
    """Test that dedupe marks exactly one entry as selected per equivalence class."""
    input_df = pd.DataFrame(
        [
            {"title": "Paper A - Preprint", "version": "v1"},
            {"title": "Paper A - Published", "version": "v2"},
            {"title": "Paper B", "version": "v1"},
        ]
    )

    result = await dedupe(
        equivalence_relation="""
            Two entries are duplicates if they are versions of the same paper.
            "Paper A - Preprint" and "Paper A - Published" are the same paper.
        """,
        input=input_df,
    )

    assert isinstance(result, TableResult)

    # For each equivalence class, exactly one should be selected
    for class_id in result.data["equivalence_class_id"].unique():
        class_rows = result.data[result.data["equivalence_class_id"] == class_id]
        selected_count = class_rows["selected"].sum()
        assert selected_count == 1, (
            f"Class {class_id} has {selected_count} selected entries, expected 1"
        )


async def test_dedupe_unique_items_all_selected():
    """Test that unique (non-duplicate) items each get their own class and are selected."""
    input_df = pd.DataFrame(
        [
            {"item": "Apple"},
            {"item": "Banana"},
            {"item": "Cherry"},
        ]
    )

    result = await dedupe(
        equivalence_relation="Items are duplicates only if they are the exact same fruit name.",
        input=input_df,
    )

    assert isinstance(result, TableResult)
    # Each should have its own equivalence class
    assert result.data["equivalence_class_id"].nunique() == 3
    # All should be selected since they're unique
    assert result.data["selected"].all()  # pyright: ignore[reportGeneralTypeIssues]
