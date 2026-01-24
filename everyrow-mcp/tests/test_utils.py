"""Tests for utility functions."""

from pathlib import Path

import pandas as pd
import pytest

from everyrow_mcp.utils import (
    resolve_output_path,
    save_result_to_csv,
    validate_csv_path,
    validate_output_path,
)


class TestValidateCsvPath:
    """Tests for validate_csv_path."""

    def test_valid_csv_file(self, tmp_path: Path):
        """Test validation passes for existing CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        # Should not raise
        validate_csv_path(str(csv_file))

    def test_relative_path_fails(self):
        """Test validation fails for relative path."""
        with pytest.raises(ValueError, match="must be absolute"):
            validate_csv_path("relative/path.csv")

    def test_nonexistent_file_fails(self, tmp_path: Path):
        """Test validation fails for nonexistent file."""
        with pytest.raises(ValueError, match="does not exist"):
            validate_csv_path(str(tmp_path / "nonexistent.csv"))

    def test_directory_fails(self, tmp_path: Path):
        """Test validation fails for directory."""
        with pytest.raises(ValueError, match="not a file"):
            validate_csv_path(str(tmp_path))

    def test_non_csv_file_fails(self, tmp_path: Path):
        """Test validation fails for non-CSV file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")

        with pytest.raises(ValueError, match="must be a CSV"):
            validate_csv_path(str(txt_file))


class TestValidateOutputPath:
    """Tests for validate_output_path."""

    def test_valid_directory(self, tmp_path: Path):
        """Test validation passes for existing directory."""
        validate_output_path(str(tmp_path))

    def test_valid_csv_path(self, tmp_path: Path):
        """Test validation passes for CSV path with existing parent."""
        csv_path = tmp_path / "output.csv"
        validate_output_path(str(csv_path))

    def test_relative_path_fails(self):
        """Test validation fails for relative path."""
        with pytest.raises(ValueError, match="must be absolute"):
            validate_output_path("relative/output.csv")

    def test_nonexistent_directory_fails(self, tmp_path: Path):
        """Test validation fails for nonexistent directory."""
        with pytest.raises(ValueError, match="does not exist"):
            validate_output_path(str(tmp_path / "nonexistent"))

    def test_nonexistent_parent_fails(self, tmp_path: Path):
        """Test validation fails for CSV path with nonexistent parent."""
        with pytest.raises(ValueError, match="does not exist"):
            validate_output_path(str(tmp_path / "nonexistent" / "output.csv"))


class TestResolveOutputPath:
    """Tests for resolve_output_path."""

    def test_full_csv_path(self, tmp_path: Path):
        """Test resolution with full CSV path."""
        output = str(tmp_path / "my_output.csv")
        result = resolve_output_path(output, "/input/data.csv", "screened")
        assert result == Path(output)

    def test_directory_generates_filename(self, tmp_path: Path):
        """Test resolution with directory generates filename."""
        result = resolve_output_path(str(tmp_path), "/input/companies.csv", "screened")
        assert result == tmp_path / "screened_companies.csv"

    def test_different_prefixes(self, tmp_path: Path):
        """Test different prefixes generate correct filenames."""
        for prefix in ["screened", "ranked", "deduped", "merged", "agent"]:
            result = resolve_output_path(str(tmp_path), "/data/test.csv", prefix)
            assert result == tmp_path / f"{prefix}_test.csv"


class TestSaveResultToCsv:
    """Tests for save_result_to_csv."""

    def test_save_dataframe(self, tmp_path: Path):
        """Test saving a DataFrame to CSV."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        output_path = tmp_path / "output.csv"

        save_result_to_csv(df, output_path)

        # Verify file was created and has correct content
        assert output_path.exists()
        loaded = pd.read_csv(output_path)
        assert list(loaded.columns) == ["a", "b"]
        assert len(loaded) == 3
