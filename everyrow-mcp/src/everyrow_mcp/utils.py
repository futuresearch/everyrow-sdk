"""Utility functions for the everyrow MCP server."""

from pathlib import Path

import pandas as pd


def validate_csv_path(path: str) -> None:
    """Validate that a CSV file exists and is readable.

    Args:
        path: Path to the CSV file

    Raises:
        ValueError: If path is not absolute, doesn't exist, or isn't a CSV file
    """
    p = Path(path)

    if not p.is_absolute():
        raise ValueError(f"Path must be absolute: {path}")

    if not p.exists():
        raise ValueError(f"File does not exist: {path}")

    if not p.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if p.suffix.lower() != ".csv":
        raise ValueError(f"File must be a CSV file: {path}")


def validate_output_path(path: str) -> None:
    """Validate that an output path is valid before processing.

    The path can be either:
    - A directory (must exist)
    - A file path ending in .csv (parent directory must exist)

    Args:
        path: Output path to validate

    Raises:
        ValueError: If path is not absolute or parent directory doesn't exist
    """
    p = Path(path)

    if not p.is_absolute():
        raise ValueError(f"Output path must be absolute: {path}")

    # If it looks like a CSV file path
    if path.lower().endswith(".csv"):
        parent = p.parent
        if not parent.exists():
            raise ValueError(f"Parent directory does not exist: {parent}")
        if not parent.is_dir():
            raise ValueError(f"Parent path is not a directory: {parent}")
    else:
        # It's a directory
        if not p.exists():
            raise ValueError(f"Output directory does not exist: {path}")
        if not p.is_dir():
            raise ValueError(f"Output path is not a directory: {path}")


def resolve_output_path(output_path: str, input_path: str, prefix: str) -> Path:
    """Resolve the output path, generating a filename if needed.

    Args:
        output_path: The output path (directory or full file path)
        input_path: The input file path (used to generate output filename)
        prefix: Prefix to add to the generated filename (e.g., 'screened', 'ranked')

    Returns:
        Full path to the output file
    """
    out = Path(output_path)

    if output_path.lower().endswith(".csv"):
        return out

    # Generate filename from input
    input_name = Path(input_path).stem
    output_name = f"{prefix}_{input_name}.csv"
    return out / output_name


def save_result_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV.

    Args:
        df: DataFrame to save
        path: Path to save to
    """
    df.to_csv(path, index=False)
