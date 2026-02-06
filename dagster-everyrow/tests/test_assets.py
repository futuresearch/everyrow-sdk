"""Tests for the asset factories."""

from unittest.mock import MagicMock

import dagster as dg
import pandas as pd

from dagster_everyrow import (
    EveryrowResource,
    everyrow_dedupe_asset,
    everyrow_merge_asset,
    everyrow_rank_asset,
    everyrow_research_asset,
    everyrow_screen_asset,
)


class MockEveryrowResult:
    """Mock result for testing."""

    def __init__(
        self, data: pd.DataFrame, session_url: str = "https://everyrow.io/sessions/test"
    ):
        self.data = data
        self.session_url = session_url


# Screen asset tests


def test_screen_asset_creates_with_correct_name():
    """Test that screen asset is created with correct name."""
    asset = everyrow_screen_asset(
        name="screened_leads",
        ins={"raw_leads": dg.AssetIn()},
        task="Remote-friendly senior role",
        input_fn=lambda raw_leads: raw_leads,
    )

    assert asset.key == dg.AssetKey("screened_leads")


def test_screen_asset_filters_passing_rows_by_default():
    """Test that screen asset filters to passing rows by default."""
    input_df = pd.DataFrame([{"company": "A"}, {"company": "B"}, {"company": "C"}])
    result_df = input_df.copy()
    result_df["passes"] = [True, False, True]

    mock_result = MockEveryrowResult(result_df)

    asset = everyrow_screen_asset(
        name="screened_leads",
        ins={"raw_leads": dg.AssetIn()},
        task="Test task",
        input_fn=lambda raw_leads: raw_leads,
    )

    asset_fn = asset.op.compute_fn.decorated_fn
    mock_context = MagicMock(spec=dg.AssetExecutionContext)
    mock_resource = MagicMock(spec=EveryrowResource)
    mock_resource.screen.return_value = mock_result

    output = asset_fn(
        context=mock_context,
        everyrow=mock_resource,
        raw_leads=input_df,
    )

    assert len(output) == 2
    assert list(output["company"]) == ["A", "C"]
    assert "passes" not in output.columns


def test_screen_asset_keeps_all_rows_when_filter_passing_false():
    """Test that screen asset keeps all rows when filter_passing=False."""
    input_df = pd.DataFrame([{"company": "A"}, {"company": "B"}])
    result_df = input_df.copy()
    result_df["passes"] = [True, False]

    mock_result = MockEveryrowResult(result_df)

    asset = everyrow_screen_asset(
        name="screened_leads",
        ins={"raw_leads": dg.AssetIn()},
        task="Test task",
        input_fn=lambda raw_leads: raw_leads,
        filter_passing=False,
    )

    asset_fn = asset.op.compute_fn.decorated_fn
    mock_context = MagicMock(spec=dg.AssetExecutionContext)
    mock_resource = MagicMock(spec=EveryrowResource)
    mock_resource.screen.return_value = mock_result

    output = asset_fn(
        context=mock_context,
        everyrow=mock_resource,
        raw_leads=input_df,
    )

    assert len(output) == 2
    assert "passes" in output.columns


# Rank asset tests


def test_rank_asset_creates_with_correct_name():
    """Test that rank asset is created with correct name."""
    asset = everyrow_rank_asset(
        name="scored_leads",
        ins={"raw_leads": dg.AssetIn()},
        task="Likelihood to convert",
        field_name="conversion_score",
        input_fn=lambda raw_leads: raw_leads,
    )

    assert asset.key == dg.AssetKey("scored_leads")


def test_rank_asset_returns_ranked_data():
    """Test that rank asset returns ranked data with score field."""
    input_df = pd.DataFrame([{"company": "A"}, {"company": "B"}])
    result_df = input_df.copy()
    result_df["conversion_score"] = [95.0, 60.0]

    mock_result = MockEveryrowResult(result_df)

    asset = everyrow_rank_asset(
        name="scored_leads",
        ins={"raw_leads": dg.AssetIn()},
        task="Likelihood to convert",
        field_name="conversion_score",
        input_fn=lambda raw_leads: raw_leads,
    )

    asset_fn = asset.op.compute_fn.decorated_fn
    mock_context = MagicMock(spec=dg.AssetExecutionContext)
    mock_resource = MagicMock(spec=EveryrowResource)
    mock_resource.rank.return_value = mock_result

    output = asset_fn(
        context=mock_context,
        everyrow=mock_resource,
        raw_leads=input_df,
    )

    assert "conversion_score" in output.columns
    assert list(output["conversion_score"]) == [95.0, 60.0]


# Dedupe asset tests


def test_dedupe_asset_creates_with_correct_name():
    """Test that dedupe asset is created with correct name."""
    asset = everyrow_dedupe_asset(
        name="deduped_contacts",
        ins={"raw_contacts": dg.AssetIn()},
        equivalence_relation="Same person",
        input_fn=lambda raw_contacts: raw_contacts,
    )

    assert asset.key == dg.AssetKey("deduped_contacts")


def test_dedupe_asset_selects_representative_by_default():
    """Test that dedupe asset selects one row per group by default."""
    input_df = pd.DataFrame(
        [
            {"name": "John Smith"},
            {"name": "J. Smith"},
            {"name": "Mike Johnson"},
        ]
    )
    result_df = input_df.copy()
    result_df["equivalence_class_id"] = [0, 0, 1]
    result_df["selected"] = [True, False, True]

    mock_result = MockEveryrowResult(result_df)

    asset = everyrow_dedupe_asset(
        name="deduped_contacts",
        ins={"raw_contacts": dg.AssetIn()},
        equivalence_relation="Same person",
        input_fn=lambda raw_contacts: raw_contacts,
    )

    asset_fn = asset.op.compute_fn.decorated_fn
    mock_context = MagicMock(spec=dg.AssetExecutionContext)
    mock_resource = MagicMock(spec=EveryrowResource)
    mock_resource.dedupe.return_value = mock_result

    output = asset_fn(
        context=mock_context,
        everyrow=mock_resource,
        raw_contacts=input_df,
    )

    assert len(output) == 2
    assert "equivalence_class_id" not in output.columns
    assert "selected" not in output.columns


# Merge asset tests


def test_merge_asset_creates_with_correct_name():
    """Test that merge asset is created with correct name."""
    asset = everyrow_merge_asset(
        name="matched_products",
        ins={"products": dg.AssetIn(), "companies": dg.AssetIn()},
        task="Match products to companies",
        left_input_fn=lambda products, **_: products,
        right_input_fn=lambda companies, **_: companies,
    )

    assert asset.key == dg.AssetKey("matched_products")


def test_merge_asset_returns_merged_data():
    """Test that merge asset returns merged data."""
    products_df = pd.DataFrame([{"product_name": "Photoshop"}])
    companies_df = pd.DataFrame([{"company_name": "Adobe"}])
    result_df = pd.DataFrame([{"product_name": "Photoshop", "company_name": "Adobe"}])

    mock_result = MockEveryrowResult(result_df)

    asset = everyrow_merge_asset(
        name="matched_products",
        ins={"products": dg.AssetIn(), "companies": dg.AssetIn()},
        task="Match products to companies",
        left_input_fn=lambda products, **_: products,
        right_input_fn=lambda companies, **_: companies,
    )

    asset_fn = asset.op.compute_fn.decorated_fn
    mock_context = MagicMock(spec=dg.AssetExecutionContext)
    mock_resource = MagicMock(spec=EveryrowResource)
    mock_resource.merge.return_value = mock_result

    output = asset_fn(
        context=mock_context,
        everyrow=mock_resource,
        products=products_df,
        companies=companies_df,
    )

    assert len(output) == 1
    assert output.iloc[0]["product_name"] == "Photoshop"
    assert output.iloc[0]["company_name"] == "Adobe"


# Research asset tests


def test_research_asset_creates_with_correct_name():
    """Test that research asset is created with correct name."""
    asset = everyrow_research_asset(
        name="enriched_companies",
        ins={"companies": dg.AssetIn()},
        task="Find funding info",
        input_fn=lambda companies: companies,
    )

    assert asset.key == dg.AssetKey("enriched_companies")


def test_research_asset_returns_enriched_data():
    """Test that research asset returns enriched data."""
    input_df = pd.DataFrame([{"company": "TechStart"}])
    result_df = input_df.copy()
    result_df["answer"] = ["Series A, $10M"]

    mock_result = MockEveryrowResult(result_df)

    asset = everyrow_research_asset(
        name="enriched_companies",
        ins={"companies": dg.AssetIn()},
        task="Find funding info",
        input_fn=lambda companies: companies,
    )

    asset_fn = asset.op.compute_fn.decorated_fn
    mock_context = MagicMock(spec=dg.AssetExecutionContext)
    mock_resource = MagicMock(spec=EveryrowResource)
    mock_resource.research.return_value = mock_result

    output = asset_fn(
        context=mock_context,
        everyrow=mock_resource,
        companies=input_df,
    )

    assert "answer" in output.columns
    assert output.iloc[0]["answer"] == "Series A, $10M"
